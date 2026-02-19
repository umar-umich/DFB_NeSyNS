"""
dataset/nesy_defake_dataset.py
==============================
NeSyDeFake Dataset — FULL-SEGMENT VERSION (Step 1 Revision)

Design principles (aligned with colleague suggestions):
  1. Full 32-frame segment per video — no sub-clip splitting.
     This gives the temporal branch the full temporal window to capture
     long-range motion and consistency artifacts.
  2. All three branches (temporal, spatial, frequency) receive the SAME
     32 frames. Different normalizations are applied per branch.
     This ensures both branches represent the same data from different views,
     which is the stated design goal.
  3. Learnable per-branch projection heads (Linear layers) are responsible
     for mapping each branch's raw feature dim to a common projection_dim.
     That logic lives in the detector/fusion module — NOT here.
  4. Ablation code is fully preserved. active_branches in the config
     controls which streams the detector uses; the dataset always produces
     all three streams so ablations don't require dataset changes.
  5. Data augmentation is applied consistently across all 32 frames in a
     clip using a shared seed, so all branches see the same augmentation.

Data flow per sample:
  Raw 32 frames (HxWxC uint8)
      │
      ├─ augment (shared seed, train only)
      │
      ├─→ temporal_clip  : (T=32, C, H, W)  — VJEPA2 / VideoMAE normalization
      ├─→ spatial_frames : (T=32, C, H, W)  — CLIP / DINOv2 normalization
      ├─→ freq_frames    : (T=32, C, H, W)  — frequency normalization (0.5/0.5)
      └─→ raw_frames     : (T=32, C, H, W)  — [0, 1] no normalization (for SemanticGrounding)

Collated batch shapes:
  temporal_clip  : (B, T, C, H, W)
  spatial_frames : (B, T, C, H, W)
  freq_frames    : (B, T, C, H, W)
  raw_frames     : (B, T, C, H, W)
  label          : (B,)

Changes from previous version:
  - Removed _build_clip_dataset() — no more sub-clip splitting.
  - Removed target_clip_size / clip_list / clip_labels split logic.
  - All branches now receive (T, C, H, W) tensors instead of (C, H, W).
  - Base class is called with video_mode=True, clip_size=frame_num (32).
    The base class collect_img_and_label_for_one_dataset() logic then
    delivers one list of 32 frame paths per video entry, which we use
    directly — no further splitting.
  - WeightedRandomSampler support added for class balancing.
"""

import random
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset


class NeSyDeFakeDataset(DeepfakeAbstractBaseDataset):
    """
    Dataset for NeSyDeFake: produces synchronized multi-branch tensors
    for all 32 frames of a video segment.

    All three streams (temporal, spatial, frequency) receive the full
    32-frame segment with branch-specific normalizations applied.
    """

    def __init__(self, config: dict, mode: str = "train"):
        # ── How many frames per segment ───────────────────────────────────
        self.segment_size = config["frame_num"][mode]   # typically 32

        # ── Tell base class: video mode ON, load full segment as one clip ─
        # clip_size == segment_size means base class builds ONE entry per
        # video with exactly segment_size frame paths — no internal splitting.
        config["video_mode"] = True
        config["clip_size"]  = self.segment_size

        self.resolution = config["resolution"]
        self.mode       = mode

        # ── Parent handles JSON parsing, LMDB setup, image_list/label_list ─
        super().__init__(config, mode)

        # ── Sanity report ─────────────────────────────────────────────────
        real_count = sum(1 for l in self.label_list if l == 0)
        fake_count = sum(1 for l in self.label_list if l == 1)
        print(
            f"\n{'='*60}"
            f"\nNeSyDeFakeDataset [{mode}]"
            f"\n  Total segments : {len(self.image_list)}"
            f"\n  Segment length : {self.segment_size} frames"
            f"\n  Real / Fake    : {real_count} / {fake_count}"
            f"\n  Resolution     : {self.resolution}×{self.resolution}"
            f"\n{'='*60}\n"
        )

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _load_frames(self, frame_paths: List[str]) -> List[np.ndarray]:
        """
        Load and resize all frames in a segment.
        Falls back to duplicating the last good frame on load errors.
        Pads with black frames only if the segment is completely empty.
        """
        frames: List[np.ndarray] = []

        for path in frame_paths:
            try:
                img = self.load_rgb(path)          # returns PIL Image (already resized)
                frames.append(np.array(img))
            except Exception as exc:
                print(f"[NeSyDeFake] Failed to load {path}: {exc}")
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(
                        np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
                    )

        # Defensive pad — should rarely trigger if base class works correctly
        while len(frames) < self.segment_size:
            frames.append(
                frames[-1].copy() if frames
                else np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
            )

        return frames[:self.segment_size]

    def _augment_frames(
        self,
        frames: List[np.ndarray],
        seed: Optional[int],
    ) -> List[np.ndarray]:
        """
        Apply the same spatial augmentation to every frame in the segment
        using a shared seed so all branches stay pixel-aligned.
        """
        if seed is None or not self.config.get("use_data_augmentation", False):
            return frames

        augmented = []
        for frame in frames:
            aug_frame, _, _ = self.data_aug(frame, None, None, seed)
            augmented.append(aug_frame)
        return augmented

    def _frames_to_tensor(
        self,
        frames: List[np.ndarray],
        normalize_fn,
    ) -> torch.Tensor:
        """
        Convert a list of HxWxC uint8 numpy arrays to a (T, C, H, W) tensor
        and apply the given per-frame normalization function.
        """
        tensors = []
        for frame in frames:
            t = self.to_tensor(frame)       # → (C, H, W), float32 in [0, 1]
            t = normalize_fn(t)
            tensors.append(t)
        return torch.stack(tensors, dim=0)  # → (T, C, H, W)

    @staticmethod
    def _extract_video_name(frame_path: str, fallback_idx: int) -> str:
        try:
            sep = "\\" if "\\" in frame_path else "/"
            parts = frame_path.split(sep)
            if "frames" in parts:
                return parts[parts.index("frames") - 1]
            return parts[-2] if len(parts) > 1 else f"video_{fallback_idx}"
        except Exception:
            return f"video_{fallback_idx}"

    # ------------------------------------------------------------------ #
    #  Core dataset interface                                              #
    # ------------------------------------------------------------------ #

    def __getitem__(self, index: int) -> dict:
        """
        Return one full-segment sample with all four synchronized streams.

        Returns
        -------
        dict
            temporal_clip  : (T, C, H, W)  — temporal branch input
            spatial_frames : (T, C, H, W)  — spatial branch input
            freq_frames    : (T, C, H, W)  — frequency branch input
            raw_frames     : (T, C, H, W)  — [0,1] for SemanticGrounding
            label          : int
            video_name     : str
        """
        frame_paths: List[str] = self.image_list[index]
        label: int             = self.label_list[index]

        # Ensure list (base class should always give us a list in video_mode)
        if not isinstance(frame_paths, list):
            frame_paths = [frame_paths]

        # 1. Load raw uint8 frames
        frames = self._load_frames(frame_paths)

        # 2. Consistent augmentation across all frames (train only)
        aug_seed = (
            random.randint(0, 2**32 - 1)
            if self.mode == "train" and self.config.get("use_data_augmentation", False)
            else None
        )
        frames = self._augment_frames(frames, aug_seed)

        # 3. Build per-branch tensors from the SAME augmented frames
        #    Each branch receives (T, C, H, W) — same content, different norm.
        temporal_clip  = self._frames_to_tensor(frames, self.normalize_temporal)
        spatial_frames = self._frames_to_tensor(frames, self.normalize_spatial)
        freq_frames    = self._frames_to_tensor(frames, self.normalize_frequency)

        # raw_frames: [0, 1] range, no normalization (used by SemanticGrounding)
        raw_frames = self._frames_to_tensor(frames, lambda x: x)

        video_name = self._extract_video_name(frame_paths[0], index)

        return {
            "temporal_clip":  temporal_clip,   # (T, C, H, W)
            "spatial_frames": spatial_frames,  # (T, C, H, W)
            "freq_frames":    freq_frames,      # (T, C, H, W)
            "raw_frames":     raw_frames,       # (T, C, H, W)
            "label":          label,
            "video_name":     video_name,
        }

    def __len__(self) -> int:
        assert len(self.image_list) == len(self.label_list)
        return len(self.image_list)

    # ------------------------------------------------------------------ #
    #  Collation                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def collate_fn(batch: list) -> dict:
        """
        Stack a list of per-sample dicts into batched tensors.

        Output shapes:
            temporal_clip  : (B, T, C, H, W)
            spatial_frames : (B, T, C, H, W)
            freq_frames    : (B, T, C, H, W)
            raw_frames     : (B, T, C, H, W)
            label          : (B,)
        """
        temporal_clips  = torch.stack([s["temporal_clip"]  for s in batch])
        spatial_frames  = torch.stack([s["spatial_frames"] for s in batch])
        freq_frames     = torch.stack([s["freq_frames"]    for s in batch])
        raw_frames      = torch.stack([s["raw_frames"]     for s in batch])
        labels          = torch.tensor([s["label"]         for s in batch],
                                       dtype=torch.long)

        return {
            "temporal_clip":  temporal_clips,   # (B, T, C, H, W)
            "spatial_frames": spatial_frames,   # (B, T, C, H, W)
            "freq_frames":    freq_frames,       # (B, T, C, H, W)
            "raw_frames":     raw_frames,        # (B, T, C, H, W)
            "label":          labels,            # (B,)
            # Keep these keys as None for compatibility with base trainer
            "landmark": None,
            "mask":     None,
        }

    # ------------------------------------------------------------------ #
    #  Class-balance sampler                                               #
    # ------------------------------------------------------------------ #

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """
        Build a WeightedRandomSampler that over-samples the minority class
        so each training epoch sees a balanced class distribution.

        Returns
        -------
        WeightedRandomSampler
        """
        labels = np.array(self.label_list)
        class_counts = np.bincount(labels)
        # Weight per class = 1 / count so rare classes get higher weight
        class_weights = 1.0 / class_counts.astype(np.float64)
        # Assign per-sample weight
        sample_weights = class_weights[labels]
        return WeightedRandomSampler(
            weights     = torch.from_numpy(sample_weights).float(),
            num_samples = len(sample_weights),
            replacement = True,
        )

    # ------------------------------------------------------------------ #
    #  DataLoader factory                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def prepare_data_loader(config: dict, mode: str = "train") -> DataLoader:
        """
        Factory method: create a DataLoader with correct sampler and settings.

        Sampler priority (train mode):
          1. DistributedSampler   — if config['ddp'] is True
          2. WeightedRandomSampler — if config['balance_classes'] is True
          3. shuffle=True          — default

        Parameters
        ----------
        config : dict
            Full training configuration.
        mode : str
            'train' or 'test'.

        Returns
        -------
        DataLoader
        """
        dataset    = NeSyDeFakeDataset(config, mode=mode)
        batch_size = (config["train_batchSize"] if mode == "train"
                      else config["test_batchSize"])
        sampler    = None
        shuffle    = (mode == "train")

        if mode == "train":
            if config.get("ddp", False):
                sampler = DistributedSampler(dataset, shuffle=True)
                shuffle = False
            elif config.get("balance_classes", False):
                sampler = dataset.get_weighted_sampler()
                shuffle = False  # sampler handles ordering

        return DataLoader(
            dataset     = dataset,
            batch_size  = batch_size,
            shuffle     = shuffle,
            sampler     = sampler,
            num_workers = int(config["workers"]),
            collate_fn  = NeSyDeFakeDataset.collate_fn,
            pin_memory  = True,
            drop_last   = (mode == "train"),
        )
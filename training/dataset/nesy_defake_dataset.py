"""
dataset/nesy_defake_dataset.py
==============================
NeSyDeFake Dataset — FRAME-LEVEL VERSION (Step 5 Revision)

Major change from previous version:
  - Temporal branch REMOVED. video_mode = False.
  - Each sample is a SINGLE FRAME, not a 32-frame clip.
  - Dataset returns (C, H, W) tensors per branch, not (T, C, H, W).
  - Semantic attributes loaded from cached .npz files (73-d per frame).
  - Memory footprint per sample drops ~32x (1 frame vs 32).

Design principles:
  1. Single frame per sample — no temporal dimension.
     The base class with video_mode=False delivers individual frame paths.
  2. Two active branches (spatial, frequency) receive the SAME frame
     with branch-specific normalizations.
  3. Semantic attributes (73-d) are loaded from preprocessed .npz files
     for the causal discovery module. Zero-vector fallback if not yet extracted.
  4. Ablation-friendly: active_branches in config controls which streams
     the detector uses; the dataset always produces both streams.

Data flow per sample:
  Single frame (HxWxC uint8)
      │
      ├─ augment (train only)
      │
      ├─→ spatial_frames : (C, H, W)  — CLIP normalization
      ├─→ freq_frames    : (C, H, W)  — raw [0,1] for FAD-CLIP
      ├─→ raw_frames     : (C, H, W)  — [0, 1] no normalization
      └─→ semantic_attrs : (73,)      — cached per-frame attributes

Collated batch shapes:
  spatial_frames : (B, C, H, W)
  freq_frames    : (B, C, H, W)
  raw_frames     : (B, C, H, W)
  semantic_attrs : (B, 73)
  label          : (B,)
"""

import os
import random
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset

# Number of per-frame semantic features from preprocessing
SEMANTIC_DIM = 73


class NeSyDeFakeDataset(DeepfakeAbstractBaseDataset):
    """
    Dataset for NeSyDeFake: produces synchronized multi-branch tensors
    for individual frames (no temporal/video mode).

    Both spatial and frequency branches receive the same frame with
    branch-specific normalizations applied.
    """

    def __init__(self, config: dict, mode: str = "train"):
        self.resolution = config["resolution"]
        self.mode = mode

        # ── Force frame-level mode ────────────────────────────────────────
        config["video_mode"] = False
        # frame_num still controls how many frames per video are sampled
        # by the base class, but each frame becomes its own sample.

        # ── Semantic features path ────────────────────────────────────────
        # Expected structure: {preprocessed_root}/{dataset}/{sub_dataset}/
        #                     semantic_features/{video_name}.npz
        self.semantic_features_root = config.get(
            'semantic_features_root',
            os.path.join(
                config.get('preprocess', {}).get('output_root_path',
                           config.get('dataset_root_path', '')),
            )
        )
        self.use_semantic = config.get('load_semantic_features', True)

        # ── Parent handles JSON parsing, image_list/label_list ────────────
        super().__init__(config, mode)

        # ── Sanity report ─────────────────────────────────────────────────
        real_count = sum(1 for l in self.label_list if l == 0)
        fake_count = sum(1 for l in self.label_list if l == 1)

        aug_active = (
            mode == 'train' and self.config.get('use_data_augmentation', False)
        )
        balance_active = (
            mode == 'train' and self.config.get('balance_classes', False)
        )

        print(
            f"\n{'='*60}"
            f"\nNeSyDeFakeDataset [{mode}] — FRAME-LEVEL"
            f"\n  Total frames      : {len(self.image_list)}"
            f"\n  Real / Fake       : {real_count} / {fake_count}"
            f"\n  Resolution        : {self.resolution}x{self.resolution}"
            f"\n  Augmentation      : {'ON' if aug_active else 'OFF'}"
            f"\n  Balanced sampling : {'ON' if balance_active else 'OFF'}"
            f"\n  Semantic features : {'ON' if self.use_semantic else 'OFF'}"
            f"\n  Active branches   : spatial, frequency (no temporal)"
            f"\n{'='*60}\n"
        )

    # ------------------------------------------------------------------ #
    #  Semantic feature loading                                            #
    # ------------------------------------------------------------------ #

    def _load_semantic_for_frame(self, frame_path: str, frame_idx: int) -> np.ndarray:
        """
        Load per-frame semantic attributes from the cached .npz file.

        The .npz is stored at:
          {preprocessed_root}/.../semantic_features/{video_name}.npz
        and contains 'per_frame' array of shape (T, 73).

        We index into it with the frame number extracted from the filename.

        Returns (73,) float32 array. Zero-vector on any failure.
        """
        if not self.use_semantic:
            return np.zeros(SEMANTIC_DIM, dtype=np.float32)

        try:
            # Parse video name and frame number from path
            # Path format: .../frames/{video_name}/{frame_num}.png
            sep = "/" if "/" in frame_path else "\\"
            parts = frame_path.split(sep)

            # Find 'frames' directory to locate video name
            if 'frames' in parts:
                frames_idx = parts.index('frames')
                video_name = parts[frames_idx + 1]
                # Build semantic features path by replacing 'frames/{video}' with
                # 'semantic_features/{video}.npz'
                npz_parts = parts[:frames_idx] + ['semantic_features', f'{video_name}.npz']
                npz_path = sep.join(npz_parts)
            else:
                return np.zeros(SEMANTIC_DIM, dtype=np.float32)

            # Extract frame number from filename (e.g., '007.png' -> 7)
            frame_filename = parts[-1]
            frame_num = int(os.path.splitext(frame_filename)[0])

            if not os.path.exists(npz_path):
                return np.zeros(SEMANTIC_DIM, dtype=np.float32)

            data = np.load(npz_path, allow_pickle=True)
            per_frame = data['per_frame']  # (T, 73)

            if frame_num < per_frame.shape[0]:
                return per_frame[frame_num].astype(np.float32)
            else:
                # Frame index out of range — return zeros
                return np.zeros(SEMANTIC_DIM, dtype=np.float32)

        except Exception:
            return np.zeros(SEMANTIC_DIM, dtype=np.float32)

    # ------------------------------------------------------------------ #
    #  Core dataset interface                                              #
    # ------------------------------------------------------------------ #

    def __getitem__(self, index: int) -> dict:
        """
        Return one frame sample with spatial, frequency, and raw tensors,
        plus cached semantic attributes.

        Returns
        -------
        dict
            spatial_frames : (C, H, W)  — spatial branch input (CLIP norm)
            freq_frames    : (C, H, W)  — frequency branch input (raw [0,1])
            raw_frames     : (C, H, W)  — [0,1] for visualization / grounding
            semantic_attrs : (73,)      — cached per-frame semantic features
            label          : int
            name           : str        — frame path for video-level metric aggregation
        """
        frame_path = self.image_list[index]
        label: int = self.label_list[index]

        # Handle case where base class returns a list (shouldn't in frame mode)
        if isinstance(frame_path, list):
            frame_path = frame_path[0]

        # 1. Load raw uint8 frame
        try:
            image = self.load_rgb(frame_path)
        except Exception as e:
            print(f"[NeSyDeFake] Failed to load {frame_path}: {e}")
            # Fallback to first sample
            return self.__getitem__(0)
        image = np.array(image)

        # 2. Data augmentation (train only)
        if self.mode == 'train' and self.config.get('use_data_augmentation', False):
            image, _, _ = self.data_aug(image, None, None)

        # 3. Convert to tensor [0, 1]
        img_tensor = self.to_tensor(image)  # (C, H, W) float32

        # 4. Branch-specific normalizations
        spatial_frames = self.normalize_spatial(img_tensor)
        freq_frames = self.normalize_frequency(img_tensor)
        raw_frames = img_tensor  # already [0, 1]

        # 5. Load cached semantic attributes
        semantic_attrs = self._load_semantic_for_frame(frame_path, index)

        return {
            "spatial_frames": spatial_frames,    # (C, H, W)
            "freq_frames":    freq_frames,       # (C, H, W)
            "raw_frames":     raw_frames,        # (C, H, W)
            "semantic_attrs": torch.from_numpy(semantic_attrs),  # (73,)
            "label":          label,
            "name":           frame_path,        # for video-level aggregation
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
            spatial_frames : (B, C, H, W)
            freq_frames    : (B, C, H, W)
            raw_frames     : (B, C, H, W)
            semantic_attrs : (B, 73)
            label          : (B,)
        """
        spatial_frames = torch.stack([s["spatial_frames"] for s in batch])
        freq_frames    = torch.stack([s["freq_frames"]    for s in batch])
        raw_frames     = torch.stack([s["raw_frames"]     for s in batch])
        semantic_attrs = torch.stack([s["semantic_attrs"]  for s in batch])
        labels         = torch.tensor([s["label"] for s in batch],
                                       dtype=torch.long)
        names          = [s["name"] for s in batch]

        return {
            "spatial_frames": spatial_frames,   # (B, C, H, W)
            "freq_frames":    freq_frames,      # (B, C, H, W)
            "raw_frames":     raw_frames,       # (B, C, H, W)
            "semantic_attrs": semantic_attrs,   # (B, 73)
            "label":          labels,           # (B,)
            "name":           names,            # list of str
            # Compatibility keys
            "landmark":       None,
            "mask":           None,
        }

    # ------------------------------------------------------------------ #
    #  Class-balance sampler                                               #
    # ------------------------------------------------------------------ #

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """
        Build a WeightedRandomSampler that over-samples the minority class.
        """
        labels = np.array(self.label_list)
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts.astype(np.float64)
        sample_weights = class_weights[labels]
        return WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).float(),
            num_samples=len(sample_weights),
            replacement=True,
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
                shuffle = False

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=int(config["workers"]),
            collate_fn=NeSyDeFakeDataset.collate_fn,
            pin_memory=True,
            drop_last=(mode == "train"),
        )
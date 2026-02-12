"""
NeSyDeFake Dataset Class

Design philosophy:
- Delegates all data collection (JSON parsing, frame path resolution, clip slicing,
  shuffling) entirely to the abstract base class, exactly as every other detector does.
- __getitem__ receives one pre-built clip (list of frame paths) and is responsible
  only for loading pixels, augmenting, and returning the four feature streams.
- Spatial stream   : single middle frame, CLIP/DINOv2 normalisation.
                     Using one canonical frame avoids blurred/mixed-pose embeddings
                     and keeps spatial & frequency streams temporally aligned.
- Frequency stream : single middle frame, SRM normalisation.
                     Artifacts are localised and transient — averaging dilutes signal.
- Raw stream       : average of first + middle + last frames, [0, 1] no norm.
                     Facial attributes are stable; averaging reduces per-frame noise.
"""

import random
# from copy import deepcopy
from typing import List

import numpy as np
import torch

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset


class NeSyDeFakeDataset(DeepfakeAbstractBaseDataset):
    """
    Dataset for NeSyDeFake: temporal + spatial + frequency + raw feature streams.

    Configuration keys consumed here (all others are handled by the base class):

        clip_size            (int)  – frames per clip; MUST be set (base class requires it
                                      when video_mode is True).  Default: 16.
        foundation_models    (dict) – normalization config for each stream (base class
                                      reads this in _setup_normalization_transforms).
        use_data_augmentation (bool) – whether to apply albumentations pipeline.

    The base class is responsible for:
        • Parsing dataset JSON files
        • Selecting / subsampling frames per video (frame_num)
        • Slicing those frames into non-overlapping clips of length clip_size
        • Building self.image_list  – list[list[str]]  (one inner list = one clip)
        • Building self.label_list  – list[int]
        • Shuffling
        • Providing load_rgb, to_tensor, normalize_temporal/spatial/frequency,
          data_aug, init_data_aug_method
    """

    # ------------------------------------------------------------------ #
    #  Keyframe indices used for the raw (semantic grounding) stream       #
    # ------------------------------------------------------------------ #
    # Spatial and frequency streams both use the single middle frame.
    # The raw stream averages first + middle + last to reduce per-frame
    # noise (blinks, motion blur) for stable facial attribute signals.

    def __init__(self, config=None, mode="train"):
        # video_mode MUST be True so the base class builds clip-level image_list
        config["video_mode"] = True

        # clip_size drives the base class clip-slicing logic
        if "clip_size" not in config:
            config["clip_size"] = 8

        # Call parent — this populates self.image_list / self.label_list
        # Each entry in self.image_list is already a list[str] of length clip_size.
        super().__init__(config, mode)

        # Convenience references set by the base class
        self.clip_size: int = config["clip_size"]
        self.resolution: int = config["resolution"]

        print(
            f"\nNeSyDeFakeDataset ready  [{mode}]"
            f"\n  clips  : {len(self.image_list)}"
            f"\n  clip_size : {self.clip_size}"
        )

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _load_frames(self, frame_paths: List[str]) -> List[np.ndarray]:
        """
        Load and return frames as HxWxC uint8 numpy arrays.
        Pads with the last valid frame (or black) if any path fails.
        """
        frames: List[np.ndarray] = []
        for path in frame_paths:
            try:
                img = self.load_rgb(path)          # PIL Image, already resized
                frames.append(np.array(img))
            except Exception as exc:
                print(f"[NeSyDeFake] Failed to load {path}: {exc}")
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(
                        np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
                    )

        # Should not happen if base class is correct, but pad defensively
        while len(frames) < self.clip_size:
            frames.append(frames[-1].copy() if frames else
                          np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8))

        return frames[: self.clip_size]

    # ------------------------------------------------------------------ #
    #  __getitem__                                                          #
    # ------------------------------------------------------------------ #

    def __getitem__(self, index: int) -> dict:
        frame_paths = self.image_list[index]
        label = self.label_list[index]

        # 1. Load Frames
        frames = self._load_frames(frame_paths)
        
        # 2. Determine Middle Index (Canonical Anchor)
        middle_idx = (len(frames) - 1) // 2

        # 3. Augment
        # We augment the full clip to ensure geometric consistency across time
        aug_seed = random.randint(0, 2 ** 32 - 1) if self.mode == "train" else None
        augmented_frames = []

        for frame in frames:
            if self.mode == "train" and self.config["use_data_augmentation"]:
                # Albumentations handles the copy internally usually, or modifies in place 
                # but since we are in a loop creating new 'aug_frame' variables, it's safe.
                aug_frame, _, _ = self.data_aug(frame, None, None, aug_seed)
            else:
                # OPTIMIZATION: Removed deepcopy. 
                # 'frame' is treated as read-only until to_tensor converts it.
                aug_frame = frame 
            augmented_frames.append(aug_frame)

        # 4. Stream 1: Temporal (VideoMAE)
        temporal_tensors = []
        for frame in augmented_frames:
            t = self.to_tensor(frame) 
            t = self.normalize_temporal(t)
            temporal_tensors.append(t)
        
        # Shape: (T, C, H, W). Check if your model needs (C, T, H, W)!
        temporal_clip = torch.stack(temporal_tensors, dim=0) 

        # 5. Extract Middle Frame (The Anchor)
        anchor_frame_tensor = self.to_tensor(augmented_frames[middle_idx])

        # Stream 2: Spatial (CLIP/DINO) - Uses Anchor
        spatial_frame = self.normalize_spatial(anchor_frame_tensor.clone())

        # Stream 3: Frequency (SRM) - Uses Anchor
        # We clone because normalize_frequency might modify the tensor
        frequency_frame = self.normalize_frequency(anchor_frame_tensor.clone())

        # Stream 4: Raw (Semantic) - Uses Anchor
        # LOGIC FIX: Do not average pixels (prevents motion blur).
        # We use the raw [0,1] tensor of the middle frame.
        raw_frame = anchor_frame_tensor # Already [0,1], no normalization

        # Video name (Optional, kept your logic)
        video_name = self._extract_video_name(frame_paths[0], index)

        return {
            "temporal_clip":    temporal_clip,
            "spatial_frame":    spatial_frame,
            "frequency_frame":  frequency_frame,
            "raw_frame":        raw_frame,
            "label":            label,
            "video_name":       video_name,
        }

    # ------------------------------------------------------------------ #
    #  Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _extract_video_name(self, frame_path: str, fallback_idx: int) -> str:
        """Infer video name from the frame path (parent dir of 'frames' folder)."""
        try:
            sep = "\\" if "\\" in frame_path else "/"
            parts = frame_path.split(sep)
            if "frames" in parts:
                return parts[parts.index("frames") - 1]
            return parts[-2] if len(parts) > 1 else f"video_{fallback_idx}"
        except Exception:
            return f"video_{fallback_idx}"

    # ------------------------------------------------------------------ #
    #  collate_fn                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def collate_fn(batch: list) -> dict:
        """Collate a list of sample dicts into batched tensors."""
        temporal_clips   = torch.stack([s["temporal_clip"]   for s in batch])  # (B, T, C, H, W)
        spatial_frames   = torch.stack([s["spatial_frame"]   for s in batch])  # (B, C, H, W)
        frequency_frames = torch.stack([s["frequency_frame"] for s in batch])  # (B, C, H, W)
        raw_frames       = torch.stack([s["raw_frame"]       for s in batch])  # (B, C, H, W)
        labels           = torch.tensor([s["label"] for s in batch], dtype=torch.long)

        return {
            "temporal_clip":   temporal_clips,
            "spatial_frame":   spatial_frames,
            "frequency_frame": frequency_frames,
            "raw_frame":       raw_frames,
            "label":           labels,
            # Keep these keys so downstream code that inspects the dict stays happy
            "landmark":        None,
            "mask":            None,
        }

    # ------------------------------------------------------------------ #
    #  __len__  (delegates to base class; overridden only for clarity)     #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return super().__len__()

    # ------------------------------------------------------------------ #
    #  DataLoader factory                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def prepare_data_loader(config: dict, mode: str = "train"):
        """
        Convenience factory.  Use this instead of constructing DataLoader manually.
        """
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler

        dataset = NeSyDeFakeDataset(config, mode=mode)

        batch_size = config["train_batchSize"] if mode == "train" else config["test_batchSize"]
        shuffle    = mode == "train"
        sampler    = None

        if config.get("ddp", False) and mode == "train":
            sampler = DistributedSampler(dataset)
            shuffle = False

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=int(config["workers"]),
            collate_fn=NeSyDeFakeDataset.collate_fn,
            sampler=sampler,
            pin_memory=True,
            drop_last=(mode == "train"),   # avoids incomplete final batch during training
        )
"""
NeSyDeFake Dataset Class - CORRECTED VERSION

Key insight from reviewing abstract_dataset.py:
- When video_mode=False: base class loads individual frames (image_list = single frame paths)
- When video_mode=True:  base class loads video clips (image_list = list of frame paths)

Strategy:
- Set video_mode=True and clip_size=32 (all frames per video)
- Base class will create ONE entry per video with all 32 frames
- We then split each 32-frame entry into 4 clips of 8 frames each
- This gives us 4× training samples while respecting base class behavior
"""

import random
from typing import List

import numpy as np
import torch

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset


class NeSyDeFakeDataset(DeepfakeAbstractBaseDataset):
    """
    Dataset for NeSyDeFake: temporal + spatial + frequency + raw feature streams.
    
    Works by:
    1. Base class loads full videos (32 frames each) via video_mode=True
    2. We split each video into 4 non-overlapping clips
    3. Each clip provides 4 synchronized streams
    """

    def __init__(self, config=None, mode="train"):
        # Critical: Store the actual clip size we want (8 frames)
        self.target_clip_size = config.get("clip_size", 8)
        
        # Tell base class to load FULL videos (all 32 frames)
        # This ensures we get all frames per video
        config["video_mode"] = True
        config["clip_size"] = 32  # Load all frames per video
        
        self.resolution = config["resolution"]
        self.mode = mode
        
        # Call parent - this populates self.image_list with full videos
        # Each entry in image_list will be a list of 32 frame paths
        super().__init__(config, mode)
        
        # NOW split each video into multiple clips
        self._build_clip_dataset()
        
        print(
            f"\n{'='*60}"
            f"\nNeSyDeFakeDataset initialized [{mode}]"
            f"\n  Original videos: {self.num_videos}"
            f"\n  Total clips: {len(self.clip_list)}"
            f"\n  Clips per video: {len(self.clip_list) / max(self.num_videos, 1):.1f}"
            f"\n  Clip size: {self.target_clip_size} frames"
            f"\n  Resolution: {self.resolution}x{self.resolution}"
            f"\n{'='*60}\n"
        )

    def _build_clip_dataset(self):
        """
        Split each video (32 frames) into multiple non-overlapping clips (8 frames each).
        
        For a video with 32 frames:
          - Creates 4 clips: [0-7], [8-15], [16-23], [24-31]
        
        Handles edge cases:
          - Videos with fewer frames than target_clip_size → pad
          - Videos with non-divisible frame counts → use as many full clips as possible
        """
        self.num_videos = len(self.image_list)
        
        clip_list = []
        clip_labels = []
        
        for video_idx, (frame_paths, label) in enumerate(
            zip(self.image_list, self.label_list)
        ):
            # Ensure frame_paths is a list (should be from base class with video_mode=True)
            if not isinstance(frame_paths, list):
                frame_paths = [frame_paths]
            
            num_frames = len(frame_paths)
            
            # Calculate how many complete clips we can extract
            num_clips = num_frames // self.target_clip_size
            
            if num_clips == 0:
                # Video has fewer frames than target_clip_size
                # Pad with last frame to reach target_clip_size
                if num_frames > 0:
                    padding_needed = self.target_clip_size - num_frames
                    padded_frames = frame_paths + [frame_paths[-1]] * padding_needed
                    clip_list.append(padded_frames[:self.target_clip_size])
                    clip_labels.append(label)
                else:
                    # Skip empty videos
                    print(f"Warning: Video {video_idx} has 0 frames, skipping")
            else:
                # Extract all complete non-overlapping clips
                for clip_idx in range(num_clips):
                    start_idx = clip_idx * self.target_clip_size
                    end_idx = start_idx + self.target_clip_size
                    clip_frames = frame_paths[start_idx:end_idx]
                    
                    # Sanity check
                    if len(clip_frames) != self.target_clip_size:
                        print(
                            f"Warning: Clip {clip_idx} from video {video_idx} "
                            f"has {len(clip_frames)} frames, expected {self.target_clip_size}"
                        )
                        continue
                    
                    clip_list.append(clip_frames)
                    clip_labels.append(label)
        
        # Store clip-level data
        self.clip_list = clip_list
        self.clip_labels = clip_labels
        
        # Update parent class attributes for compatibility
        self.image_list = clip_list
        self.label_list = clip_labels
        
        # Update data_dict
        self.data_dict = {
            'image': self.clip_list,
            'label': self.clip_labels,
        }

    def _load_frames(self, frame_paths: List[str]) -> List[np.ndarray]:
        """
        Load frames as HxWxC uint8 numpy arrays.
        Uses base class load_rgb method which handles LMDB and regular files.
        """
        frames: List[np.ndarray] = []
        
        for path in frame_paths:
            try:
                img = self.load_rgb(path)  # PIL Image, already resized by base class
                frames.append(np.array(img))
            except Exception as exc:
                print(f"[NeSyDeFake] Failed to load {path}: {exc}")
                # Fallback: duplicate last frame or create black frame
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(
                        np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
                    )
        
        # Defensive padding (should not be needed if base class works correctly)
        while len(frames) < self.target_clip_size:
            if frames:
                frames.append(frames[-1].copy())
            else:
                frames.append(
                    np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
                )
        
        return frames[:self.target_clip_size]

    def __getitem__(self, index: int) -> dict:
        """
        Return one clip with all 4 synchronized feature streams.
        
        Returns:
            dict with keys:
                - temporal_clip: (T, C, H, W) - full 8-frame sequence
                - spatial_frame: (C, H, W) - middle frame for CLIP/DINOv2
                - frequency_frame: (C, H, W) - middle frame for SRM
                - raw_frame: (C, H, W) - middle frame [0,1] normalized
                - label: int - 0 for real, 1 for fake
                - video_name: str - for debugging
        """
        frame_paths = self.clip_list[index]
        label = self.clip_labels[index]
        
        # 1. Load all frames for this clip
        frames = self._load_frames(frame_paths)
        
        # 2. Determine middle frame index (anchor for spatial/frequency/raw)
        middle_idx = (len(frames) - 1) // 2
        
        # 3. Apply augmentation (if training)
        # Use consistent augmentation across all frames in the clip
        aug_seed = random.randint(0, 2 ** 32 - 1) if self.mode == "train" else None
        augmented_frames = []
        
        for frame in frames:
            if self.mode == "train" and self.config["use_data_augmentation"]:
                aug_frame, _, _ = self.data_aug(frame, None, None, aug_seed)
            else:
                aug_frame = frame
            augmented_frames.append(aug_frame)
        
        # 4. Build Temporal Stream (all frames)
        temporal_tensors = []
        for frame in augmented_frames:
            t = self.to_tensor(frame)  # Converts to [0,1] tensor
            t = self.normalize_temporal(t)  # VideoMAE normalization
            temporal_tensors.append(t)
        
        temporal_clip = torch.stack(temporal_tensors, dim=0)  # (T, C, H, W)
        
        # 5. Extract middle frame (the anchor)
        anchor_frame_tensor = self.to_tensor(augmented_frames[middle_idx])  # [0,1] range
        
        # 6. Build other streams from anchor frame
        spatial_frame = self.normalize_spatial(anchor_frame_tensor.clone())
        frequency_frame = self.normalize_frequency(anchor_frame_tensor.clone())
        raw_frame = anchor_frame_tensor  # Keep [0,1] range, no additional normalization
        
        # 7. Video name (for debugging)
        video_name = self._extract_video_name(frame_paths[0], index)
        
        return {
            "temporal_clip": temporal_clip,      # (T, C, H, W) = (8, 3, 224, 224)
            "spatial_frame": spatial_frame,      # (C, H, W) = (3, 224, 224)
            "frequency_frame": frequency_frame,  # (C, H, W) = (3, 224, 224)
            "raw_frame": raw_frame,              # (C, H, W) = (3, 224, 224)
            "label": label,
            "video_name": video_name,
        }

    def _extract_video_name(self, frame_path: str, fallback_idx: int) -> str:
        """Extract video name from frame path."""
        try:
            sep = "\\" if "\\" in frame_path else "/"
            parts = frame_path.split(sep)
            if "frames" in parts:
                return parts[parts.index("frames") - 1]
            return parts[-2] if len(parts) > 1 else f"video_{fallback_idx}"
        except Exception:
            return f"video_{fallback_idx}"

    @staticmethod
    def collate_fn(batch: list) -> dict:
        """
        Collate a list of samples into a batched dictionary.
        
        Args:
            batch: List of dicts from __getitem__
            
        Returns:
            dict with batched tensors
        """
        temporal_clips = torch.stack([s["temporal_clip"] for s in batch])
        spatial_frames = torch.stack([s["spatial_frame"] for s in batch])
        frequency_frames = torch.stack([s["frequency_frame"] for s in batch])
        raw_frames = torch.stack([s["raw_frame"] for s in batch])
        labels = torch.tensor([s["label"] for s in batch], dtype=torch.long)
        
        return {
            "temporal_clip": temporal_clips,      # (B, T, C, H, W)
            "spatial_frame": spatial_frames,      # (B, C, H, W)
            "frequency_frame": frequency_frames,  # (B, C, H, W)
            "raw_frame": raw_frames,              # (B, C, H, W)
            "label": labels,                      # (B,)
            # Compatibility keys for base class collate_fn expectations
            "landmark": None,
            "mask": None,
        }

    def __len__(self) -> int:
        """Return total number of clips."""
        return len(self.clip_list)

    @staticmethod
    def prepare_data_loader(config: dict, mode: str = "train"):
        """
        Factory method to create a DataLoader with correct settings.
        
        Args:
            config: Configuration dictionary
            mode: 'train' or 'test'
            
        Returns:
            torch.utils.data.DataLoader
        """
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler
        
        dataset = NeSyDeFakeDataset(config, mode=mode)
        
        batch_size = config["train_batchSize"] if mode == "train" else config["test_batchSize"]
        shuffle = mode == "train"
        sampler = None
        
        if config.get("ddp", False):
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False  # Sampler handles shuffling in DDP
        
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=int(config["workers"]),
            collate_fn=NeSyDeFakeDataset.collate_fn,
            sampler=sampler,
            pin_memory=True,
            drop_last=(mode == "train"),
        )
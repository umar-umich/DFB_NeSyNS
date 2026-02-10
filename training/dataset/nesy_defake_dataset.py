"""
NeSyDeFake Dataset Class
Handles efficient clip sampling for temporal, spatial, and frequency features
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import random
from typing import Dict, List, Tuple

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset


class NeSyDeFakeDataset(DeepfakeAbstractBaseDataset):
    """
    Custom dataset for NeSyDeFake framework with clip-based sampling.
    
    Key Design:
    - Parent class collects video frames (ideally 64, but can be less)
    - Each __getitem__ call returns ONE clip of 16 frames
    - The dataset length is multiplied by num_clips_per_video to ensure good epoch coverage
    - Temporal stream: Uses all 16 frames in the clip
    - Spatial/Frequency streams: Uses 1 middle frame from the clip
    """
    
    def __init__(self, config, mode='train'):
        # Clip configuration
        self.clip_size = config.get('clip_size', 16)  # Frames per clip
        self.num_clips_per_video = config.get('num_clips_per_video', 32)  # Clips to sample per video
        
        # Sampling strategy
        self.sampling_strategy = config.get('sampling', {}).get('sampling_strategy', 'sliding_window')
        self.overlap_ratio = config.get('sampling', {}).get('overlap_ratio', 0.5)
        
        # Frame selection for spatial/frequency
        self.spatial_frame_idx = config.get('sampling', {}).get('spatial_frame_selection', 'middle')
        
        # Call parent init - this will collect video frame paths
        super().__init__(config, mode)
        
        # Expand dataset by creating multiple clip indices per video
        self._expand_dataset_with_clips()
        
        print(f"NeSyDeFake Dataset initialized:")
        print(f"  - Mode: {mode}")
        print(f"  - Base videos: {len(self.image_list)}")
        print(f"  - Total samples (with clips): {len(self.clip_data)}")
        print(f"  - Clip size: {self.clip_size}")
        print(f"  - Target clips per video: {self.num_clips_per_video}")
        print(f"  - Sampling strategy: {self.sampling_strategy}")
    
    def _expand_dataset_with_clips(self):
        """
        Pre-compute clip indices for each video to create expanded dataset.
        This allows each training sample to be one clip.
        """
        self.clip_data = []
        
        for video_idx in range(len(self.image_list)):
            frame_paths = self.image_list[video_idx]
            label = self.label_list[video_idx]
            
            # Ensure frame_paths is a list
            if not isinstance(frame_paths, list):
                frame_paths = [frame_paths]
            
            total_frames = len(frame_paths)
            
            # Generate clip indices for this video
            clip_indices = self._get_clip_indices_for_video(total_frames)
            
            # Create a sample for each clip
            for start_idx, end_idx in clip_indices:
                self.clip_data.append({
                    'video_idx': video_idx,
                    'frame_paths': frame_paths,
                    'clip_start': start_idx,
                    'clip_end': end_idx,
                    'label': label,
                    'total_frames': total_frames
                })
    
    def _get_clip_indices_for_video(self, total_frames: int) -> List[Tuple[int, int]]:
        """
        Generate clip start/end indices for a single video based on sampling strategy.
        
        Args:
            total_frames: Total number of available frames in the video
            
        Returns:
            List of (start_idx, end_idx) tuples for each clip
        """
        if self.sampling_strategy == 'sliding_window':
            return self._sliding_window_sampling(total_frames)
        elif self.sampling_strategy == 'uniform':
            return self._uniform_sampling(total_frames)
        elif self.sampling_strategy == 'random':
            return self._random_sampling(total_frames)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
    
    def _sliding_window_sampling(self, total_frames: int) -> List[Tuple[int, int]]:
        """
        Sliding window with overlap for maximum coverage.
        Handles edge cases where video has fewer frames than expected.
        """
        clips = []
        
        # If video is too short, just use what we have
        if total_frames < self.clip_size:
            # Repeat frames to reach clip_size or just use available frames
            clips.append((0, total_frames))
            # Duplicate this clip to reach num_clips_per_video
            while len(clips) < self.num_clips_per_video:
                clips.append((0, total_frames))
            return clips[:self.num_clips_per_video]
        
        # Normal case: sliding window
        stride = max(1, int(self.clip_size * (1 - self.overlap_ratio)))
        
        start_idx = 0
        while start_idx + self.clip_size <= total_frames:
            end_idx = start_idx + self.clip_size
            clips.append((start_idx, end_idx))
            start_idx += stride
            
            # If we have enough clips, stop
            if len(clips) >= self.num_clips_per_video:
                break
        
        # If we don't have enough clips, add more from different positions
        while len(clips) < self.num_clips_per_video:
            if len(clips) == 0:
                # Edge case: shouldn't happen, but add safety
                clips.append((0, min(self.clip_size, total_frames)))
            else:
                # Add clips with random offsets
                max_start = max(0, total_frames - self.clip_size)
                random_start = random.randint(0, max_start)
                random_end = random_start + self.clip_size
                
                # Check if this clip already exists (avoid exact duplicates)
                if (random_start, random_end) not in clips:
                    clips.append((random_start, random_end))
                else:
                    # If exact duplicate, just add it anyway (we need enough samples)
                    clips.append((random_start, random_end))
        
        return clips[:self.num_clips_per_video]
    
    def _uniform_sampling(self, total_frames: int) -> List[Tuple[int, int]]:
        """
        Uniformly sample clips across the video.
        """
        clips = []
        
        if total_frames < self.clip_size:
            # Video too short
            for _ in range(self.num_clips_per_video):
                clips.append((0, total_frames))
            return clips
        
        # Calculate spacing between clip centers
        if self.num_clips_per_video == 1:
            centers = [total_frames // 2]
        else:
            # Ensure centers are within valid range
            min_center = self.clip_size // 2
            max_center = total_frames - self.clip_size // 2
            
            if max_center <= min_center:
                # Not enough room for multiple clips
                centers = [total_frames // 2] * self.num_clips_per_video
            else:
                centers = np.linspace(min_center, max_center, self.num_clips_per_video, dtype=int)
        
        for center in centers:
            start_idx = max(0, center - self.clip_size // 2)
            end_idx = min(total_frames, start_idx + self.clip_size)
            
            # Adjust if we're at the boundary
            if end_idx - start_idx < self.clip_size:
                start_idx = max(0, end_idx - self.clip_size)
            
            clips.append((start_idx, end_idx))
        
        return clips
    
    def _random_sampling(self, total_frames: int) -> List[Tuple[int, int]]:
        """
        Randomly sample clips (with replacement if needed).
        """
        clips = []
        
        if total_frames < self.clip_size:
            # Video too short
            for _ in range(self.num_clips_per_video):
                clips.append((0, total_frames))
            return clips
        
        max_start = total_frames - self.clip_size
        
        for _ in range(self.num_clips_per_video):
            start_idx = random.randint(0, max_start)
            end_idx = start_idx + self.clip_size
            clips.append((start_idx, end_idx))
        
        return clips
    
    def __getitem__(self, index):
        """
        Get a single clip sample with properly normalized features.
        
        Returns:
            dict with:
                - 'temporal_clip': (clip_size, C, H, W) - VideoMAE normalized
                - 'spatial_frame': (C, H, W) - CLIP/DINOv2 normalized  
                - 'frequency_frame': (C, H, W) - SRM normalized
                - 'raw_frame': (C, H, W) - Raw [0,1] for semantic grounding
                - 'label': Binary label
        """
        # Get clip data
        clip_info = self.clip_data[index]
        frame_paths = clip_info['frame_paths']
        clip_start = clip_info['clip_start']
        clip_end = clip_info['clip_end']
        label = clip_info['label']
        
        # Extract clip frames
        clip_frame_paths = frame_paths[clip_start:clip_end]
        
        # Load frames (as numpy arrays)
        frames = []
        for frame_path in clip_frame_paths:
            try:
                image = self.load_rgb(frame_path)
                frames.append(np.array(image))
            except Exception as e:
                print(f"Error loading frame {frame_path}: {e}")
                if frames:
                    frames.append(frames[-1])
                else:
                    frames.append(np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8))
        
        # Pad if necessary
        while len(frames) < self.clip_size:
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8))
        
        frames = frames[:self.clip_size]
        
        # Select middle frame
        mid_idx = len(frames) // 2
        spatial_frame = frames[mid_idx]
        
        # Apply data augmentation with consistent seed
        augmentation_seed = random.randint(0, 2**32 - 1) if self.mode == 'train' else None
        
        # ========== Process Temporal Clip (VideoMAE normalization) ==========
        temporal_frames = []
        for frame in frames:
            if self.mode == 'train' and self.config['use_data_augmentation']:
                frame_aug, _, _ = self.data_aug(frame, None, None, augmentation_seed)
            else:
                frame_aug = frame
            
            # Convert to tensor and normalize for VideoMAE
            frame_tensor = self.to_tensor(frame_aug)  # [0, 1]
            frame_tensor = self.normalize_temporal(frame_tensor)  # ImageNet norm
            temporal_frames.append(frame_tensor)
        
        temporal_clip = torch.stack(temporal_frames, dim=0)  # (clip_size, C, H, W)
        
        # ========== Process Spatial Frame (CLIP/DINOv2 normalization) ==========
        if self.mode == 'train' and self.config['use_data_augmentation']:
            spatial_frame_aug, _, _ = self.data_aug(spatial_frame, None, None, augmentation_seed)
        else:
            spatial_frame_aug = spatial_frame
        
        spatial_tensor = self.to_tensor(spatial_frame_aug)  # [0, 1]
        spatial_tensor_normalized = self.normalize_spatial(spatial_tensor)  # CLIP/DINOv2 norm
        
        # ========== Process Frequency Frame (SRM normalization) ==========
        frequency_tensor = self.to_tensor(spatial_frame_aug)  # [0, 1]
        frequency_tensor_normalized = self.normalize_frequency(frequency_tensor)  # 0.5/0.5 or raw
        
        # ========== Keep Raw Frame for Semantic Grounding ==========
        raw_tensor = self.to_tensor(spatial_frame_aug)  # [0, 1] range, no normalization
        
        # Extract video name
        video_name = self._extract_video_name(clip_frame_paths[0], clip_info['video_idx'])
        
        return {
            'temporal_clip': temporal_clip,           # (clip_size, C, H, W) - VideoMAE norm
            'spatial_frame': spatial_tensor_normalized,  # (C, H, W) - CLIP/DINOv2 norm
            'frequency_frame': frequency_tensor_normalized,  # (C, H, W) - SRM norm
            'raw_frame': raw_tensor,                  # (C, H, W) - [0,1] for semantic grounding
            'label': label,
            'video_name': video_name
        }
    
    def _extract_video_name(self, frame_path: str, video_idx: int) -> str:
        """Extract video name from frame path."""
        try:
            if '\\' in frame_path:
                parts = frame_path.split('\\')
            else:
                parts = frame_path.split('/')
            
            # Find video name (parent directory of 'frames')
            if 'frames' in parts:
                frames_idx = parts.index('frames')
                return parts[frames_idx - 1]
            else:
                return parts[-2] if len(parts) > 1 else f"video_{video_idx}"
        except:
            return f"video_{video_idx}"
    
    def collate_fn(self, batch):
        """Custom collate function for batching."""
        temporal_clips = torch.stack([item['temporal_clip'] for item in batch])  # (B, clip_size, C, H, W)
        spatial_frames = torch.stack([item['spatial_frame'] for item in batch])  # (B, C, H, W)
        frequency_frames = torch.stack([item['frequency_frame'] for item in batch])  # (B, C, H, W)
        raw_frames = torch.stack([item['raw_frame'] for item in batch])  # (B, C, H, W)
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
        
        return {
            'temporal_clip': temporal_clips,      # For VideoMAE
            'spatial_frame': spatial_frames,      # For CLIP/DINOv2
            'frequency_frame': frequency_frames,  # For SRM
            'raw_frame': raw_frames,              # For semantic grounding
            'label': labels,
            'landmark': None,
            'mask': None,
        }
    
    def __len__(self):
        """Return the total number of clips (not videos)"""
        return len(self.clip_data)
    
    @staticmethod
    def prepare_data_loader(config, mode='train'):
        """
        Static method to create data loader for NeSyDeFake.
        """
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler
        
        # Create dataset
        dataset = NeSyDeFakeDataset(config, mode=mode)
        
        # Determine batch size based on mode
        batch_size = config['train_batchSize'] if mode == 'train' else config['test_batchSize']
        
        # Create sampler if using DDP
        sampler = None
        shuffle = (mode == 'train')
        
        if config.get('ddp', False) and mode == 'train':
            sampler = DistributedSampler(dataset)
            shuffle = False  # Sampler handles shuffling
        
        # Create data loader
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=int(config['workers']),
            collate_fn=dataset.collate_fn,
            sampler=sampler,
            pin_memory=True
        )
        
        return data_loader
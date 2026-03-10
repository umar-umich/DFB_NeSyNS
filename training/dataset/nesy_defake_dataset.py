"""
dataset/nesy_defake_dataset.py
==============================
NeSyDeFake Dataset — FRAME-LEVEL with SOURCE-PAIRED TRAINING

v2 changes (2026-03-10):
  - SOURCE-PAIRED TRAINING: For FF++, each fake frame is paired with a real
    frame from the same source video. Forces the model to learn manipulation
    artifacts, not identity/background shortcuts. (GenD, WACV 2026)
  - Batch structure: each __getitem__ returns a (real, fake) pair from the
    same source video. Batch size N -> 2N frames in forward pass.
  - Cross-dataset (non-FF++) falls back to random pairing.

Design principles:
  1. Single frame per sample — no temporal dimension.
  2. Two active branches (spatial, frequency) receive the SAME frame.
  3. Semantic attributes (73-d or 211-d) loaded from cache or computed online.
  4. Paired training: GenD-style source-matched real-fake pairs to prevent
     shortcut learning. The causal module benefits most — it must discover
     artifact-related causal edges, not identity-correlated ones.

Data flow per paired sample:
  Fake frame from video "802_885" + Real frame from source video "802"
      │
      ├─→ Both augmented independently
      ├─→ Both get spatial/freq/raw normalizations
      └─→ Collated as interleaved [real_0, fake_0, real_1, fake_1, ...]

Collated batch shapes (for batch_size N pairs = 2N frames):
  spatial_frames : (2N, C, H, W)
  freq_frames    : (2N, C, H, W)
  raw_frames     : (2N, C, H, W)
  semantic_attrs : (2N, semantic_dim)
  label          : (2N,)
"""

import json
import os
import random
from collections import defaultdict
from typing import Optional

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

    When paired_training=True (default for train mode on FF++), each sample
    returns a source-matched (real, fake) pair. The collator interleaves
    them so the batch contains paired real-fake frames from the same videos.
    """

    def __init__(self, config: dict, mode: str = "train"):
        self.resolution = config["resolution"]
        self.mode = mode

        # ── Force frame-level mode ────────────────────────────────────────
        config["video_mode"] = False

        # ── Semantic features path ────────────────────────────────────────
        self.semantic_features_root = config.get(
            'semantic_features_root',
            os.path.join(
                config.get('preprocess', {}).get('output_root_path',
                           config.get('dataset_root_path', '')),
            )
        )
        self.use_semantic = config.get('load_semantic_features', True)

        # ── Precomputed Face-LLaVA semantic features ─────────────────────
        sem_cfg = config.get('semantic_attributes', {})
        self.use_precomputed_semantic = (
            sem_cfg.get('backend') == 'precomputed'
            and sem_cfg.get('enabled', False)
        )
        self._precomputed_subdir = sem_cfg.get(
            'precomputed_dir', 'facellava_semantic')
        self._precomputed_dim = sem_cfg.get('precomputed_dim', 211)
        # Cache: video_base_dir -> {frame_idx: tensor}
        self._precomputed_cache = {} if self.use_precomputed_semantic else None

        # ── Parent handles JSON parsing, image_list/label_list ────────────
        super().__init__(config, mode)

        # ── Paired training setup ─────────────────────────────────────────
        self.paired_training = (
            mode == 'train'
            and config.get('paired_training', True)
        )

        if self.paired_training:
            self._build_pair_index()
        else:
            self._fake_indices = None
            self._source_to_real_frames = None

        # ── Sanity report ─────────────────────────────────────────────────
        real_count = sum(1 for la in self.label_list if la == 0)
        fake_count = sum(1 for la in self.label_list if la == 1)

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
            f"\n  Precomputed sem.  : {'ON (' + self._precomputed_subdir + ')' if self.use_precomputed_semantic else 'OFF'}"
            f"\n  Paired training   : {'ON' if self.paired_training else 'OFF'}"
            f"\n  Active branches   : spatial, frequency (no temporal)"
            f"\n{'='*60}\n"
        )

    # ------------------------------------------------------------------ #
    #  Paired training: build source-matched index                        #
    # ------------------------------------------------------------------ #

    def _extract_source_video(self, frame_path: str) -> Optional[str]:
        """
        Extract the source video ID from a frame path.

        For FF++ fake videos like "802_885", the source video is "802".
        For real videos like "929", returns "929".
        For non-FF++ datasets, returns None (pairing not available).
        """
        sep = "/" if "/" in frame_path else "\\"
        parts = frame_path.split(sep)
        if 'frames' not in parts:
            return None
        frames_idx = parts.index('frames')
        if frames_idx + 1 >= len(parts):
            return None
        video_name = parts[frames_idx + 1]

        # Fake video names have format "source_target" (e.g., "802_885")
        if '_' in video_name:
            return video_name.split('_')[0]
        # Real video names are just the ID (e.g., "802")
        return video_name

    def _build_pair_index(self):
        """
        Build index mapping: source_video_id -> list of real frame indices,
        and a list of fake frame indices with their source IDs.

        This enables O(1) lookup of real frames matching a given fake's source.
        """
        self._source_to_real_frames = defaultdict(list)
        self._fake_indices = []
        self._all_real_indices = []

        for idx, (path, label) in enumerate(
                zip(self.image_list, self.label_list)):
            if isinstance(path, list):
                path = path[0]
            source = self._extract_source_video(path)

            if label == 0:
                # Real frame — index by its video ID (which is the source)
                if source:
                    self._source_to_real_frames[source].append(idx)
                self._all_real_indices.append(idx)
            else:
                # Fake frame — store with its source video ID
                self._fake_indices.append((idx, source))

        n_paired = sum(
            1 for _, src in self._fake_indices
            if src and src in self._source_to_real_frames
        )
        n_unpaired = len(self._fake_indices) - n_paired

        print(f"[PairedTraining] {n_paired} fake frames have source-matched "
              f"real frames, {n_unpaired} will use random real pairing")

    # ------------------------------------------------------------------ #
    #  Semantic feature loading                                            #
    # ------------------------------------------------------------------ #

    def _load_semantic_for_frame(self, frame_path: str, frame_idx: int) -> np.ndarray:
        """
        Load per-frame semantic attributes from the cached .npz file.
        Returns (73,) float32 array. Zero-vector on any failure.
        """
        if not self.use_semantic:
            return np.zeros(SEMANTIC_DIM, dtype=np.float32)

        try:
            sep = "/" if "/" in frame_path else "\\"
            parts = frame_path.split(sep)

            if 'frames' in parts:
                frames_idx = parts.index('frames')
                video_name = parts[frames_idx + 1]
                npz_parts = parts[:frames_idx] + ['semantic_features', f'{video_name}.npz']
                npz_path = sep.join(npz_parts)
            else:
                return np.zeros(SEMANTIC_DIM, dtype=np.float32)

            frame_filename = parts[-1]
            frame_num = int(os.path.splitext(frame_filename)[0])

            if not os.path.exists(npz_path):
                return np.zeros(SEMANTIC_DIM, dtype=np.float32)

            data = np.load(npz_path, allow_pickle=True)
            per_frame = data['per_frame']  # (T, 73)

            if frame_num < per_frame.shape[0]:
                return per_frame[frame_num].astype(np.float32)
            else:
                return np.zeros(SEMANTIC_DIM, dtype=np.float32)

        except Exception:
            return np.zeros(SEMANTIC_DIM, dtype=np.float32)

    # ------------------------------------------------------------------ #
    #  Precomputed Face-LLaVA feature loading                              #
    # ------------------------------------------------------------------ #

    def _load_precomputed_semantic(self, frame_path: str) -> torch.Tensor:
        """
        Load precomputed Face-LLaVA 211-d attributes from .pt file.
        Returns (precomputed_dim,) tensor. Zero-vector on any failure.
        """
        try:
            sep = "/" if "/" in frame_path else "\\"
            parts = frame_path.split(sep)

            if 'frames' not in parts:
                return torch.zeros(self._precomputed_dim)

            frames_idx = parts.index('frames')
            video_name = parts[frames_idx + 1]
            base_dir = sep.join(parts[:frames_idx])

            # Cache key
            cache_key = f"{base_dir}/{video_name}"
            if cache_key not in self._precomputed_cache:
                pt_path = os.path.join(
                    base_dir, self._precomputed_subdir, f'{video_name}.pt')
                if os.path.exists(pt_path):
                    data = torch.load(pt_path, map_location='cpu',
                                      weights_only=False)
                    self._precomputed_cache[cache_key] = data
                else:
                    self._precomputed_cache[cache_key] = None

            cached = self._precomputed_cache[cache_key]
            if cached is None:
                return torch.zeros(self._precomputed_dim)

            features = cached['features']  # (n_frames, 211)
            frame_paths = cached.get('frame_paths', [])

            # Try to find exact frame match
            if frame_paths:
                frame_filename = parts[-1]
                for idx, fp in enumerate(frame_paths):
                    if fp.endswith(frame_filename):
                        return features[idx]

            # Fallback: index by frame number
            frame_filename = parts[-1]
            frame_num = int(os.path.splitext(frame_filename)[0])
            if frame_num < features.shape[0]:
                return features[frame_num]

            return torch.zeros(self._precomputed_dim)

        except Exception:
            return torch.zeros(self._precomputed_dim)

    # ------------------------------------------------------------------ #
    #  Single-frame loading helper                                         #
    # ------------------------------------------------------------------ #

    def _load_single_frame(self, index: int) -> dict:
        """Load and preprocess a single frame by index."""
        frame_path = self.image_list[index]
        label = self.label_list[index]

        if isinstance(frame_path, list):
            frame_path = frame_path[0]

        try:
            image = self.load_rgb(frame_path)
        except Exception as e:
            print(f"[NeSyDeFake] Failed to load {frame_path}: {e}")
            return self._load_single_frame(0)
        image = np.array(image)

        if self.mode == 'train' and self.config.get('use_data_augmentation', False):
            image, _, _ = self.data_aug(image, None, None)

        img_tensor = self.to_tensor(image)

        spatial_frames = self.normalize_spatial(img_tensor)
        freq_frames = self.normalize_frequency(img_tensor)
        raw_frames = img_tensor

        semantic_attrs = self._load_semantic_for_frame(frame_path, index)

        # Precomputed Face-LLaVA features (loaded alongside old semantic)
        if self.use_precomputed_semantic:
            precomputed_attrs = self._load_precomputed_semantic(frame_path)
        else:
            precomputed_attrs = torch.zeros(1)  # placeholder

        return {
            "spatial_frames":    spatial_frames,
            "freq_frames":       freq_frames,
            "raw_frames":        raw_frames,
            "semantic_attrs":    torch.from_numpy(semantic_attrs),
            "precomputed_attrs": precomputed_attrs,
            "label":             label,
            "name":              frame_path,
        }

    # ------------------------------------------------------------------ #
    #  Core dataset interface                                              #
    # ------------------------------------------------------------------ #

    def __getitem__(self, index: int) -> dict:
        """
        Return one sample. In paired mode, returns a dict with both
        a real and fake frame from the same source video.
        In unpaired mode, returns a single frame dict.
        """
        if not self.paired_training:
            return self._load_single_frame(index)

        # ── Paired mode: index into fake list, find matching real ──────
        fake_idx, source_id = self._fake_indices[index % len(self._fake_indices)]

        # Find a real frame from the same source video
        if source_id and source_id in self._source_to_real_frames:
            real_candidates = self._source_to_real_frames[source_id]
            real_idx = random.choice(real_candidates)
        else:
            # Fallback: random real frame (for non-FF++ datasets)
            real_idx = random.choice(self._all_real_indices)

        real_sample = self._load_single_frame(real_idx)
        fake_sample = self._load_single_frame(fake_idx)

        return {
            "real": real_sample,
            "fake": fake_sample,
        }

    def __len__(self) -> int:
        if self.paired_training and self._fake_indices:
            return len(self._fake_indices)
        assert len(self.image_list) == len(self.label_list)
        return len(self.image_list)

    # ------------------------------------------------------------------ #
    #  Collation                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def collate_fn(batch: list) -> dict:
        """
        Stack a list of per-sample dicts into batched tensors.

        Handles both paired and unpaired modes:
        - Paired: each item has 'real' and 'fake' sub-dicts. Interleave them
          so batch[0]=real_0, batch[1]=fake_0, batch[2]=real_1, etc.
          This ensures each consecutive pair shares the same source video.
        - Unpaired: each item is a flat dict with frame tensors.

        Output shapes (for N pairs = 2N frames, or N unpaired frames):
            spatial_frames : (2N or N, C, H, W)
            freq_frames    : (2N or N, C, H, W)
            raw_frames     : (2N or N, C, H, W)
            semantic_attrs : (2N or N, semantic_dim)
            label          : (2N or N,)
        """
        # Detect paired vs unpaired mode
        if 'real' in batch[0] and 'fake' in batch[0]:
            # Paired mode: interleave real and fake
            all_samples = []
            for item in batch:
                all_samples.append(item['real'])
                all_samples.append(item['fake'])
            batch = all_samples

        spatial_frames = torch.stack([s["spatial_frames"] for s in batch])
        freq_frames    = torch.stack([s["freq_frames"]    for s in batch])
        raw_frames     = torch.stack([s["raw_frames"]     for s in batch])
        semantic_attrs = torch.stack([s["semantic_attrs"]  for s in batch])
        labels         = torch.tensor([s["label"] for s in batch],
                                       dtype=torch.long)
        names          = [s["name"] for s in batch]

        # Precomputed Face-LLaVA attributes (if available)
        precomputed_attrs = None
        if "precomputed_attrs" in batch[0]:
            precomputed_attrs = torch.stack(
                [s["precomputed_attrs"] for s in batch])

        return {
            "spatial_frames":    spatial_frames,
            "freq_frames":       freq_frames,
            "raw_frames":        raw_frames,
            "semantic_attrs":    semantic_attrs,
            "precomputed_attrs": precomputed_attrs,
            "label":             labels,
            "name":              names,
            # Compatibility keys
            "landmark":          None,
            "mask":              None,
        }

    # ------------------------------------------------------------------ #
    #  Class-balance sampler                                               #
    # ------------------------------------------------------------------ #

    def get_weighted_sampler(self, target_real_fraction: float = 0.35) -> WeightedRandomSampler:
        """
        For unpaired mode: standard weighted random sampling.
        For paired mode: uniform sampling over fake indices (each fake
        automatically pulls its paired real).
        """
        if self.paired_training and self._fake_indices:
            # Paired mode: uniform over fakes (pairing handles balance)
            n = len(self._fake_indices)
            return WeightedRandomSampler(
                weights=torch.ones(n),
                num_samples=n,
                replacement=True,
            )

        labels = np.array(self.label_list)
        n_real = (labels == 0).sum()
        n_fake = (labels == 1).sum()

        w_real = target_real_fraction / n_real
        w_fake = (1.0 - target_real_fraction) / n_fake
        sample_weights = np.where(labels == 0, w_real, w_fake)

        print(f"[Sampler] target_real_fraction={target_real_fraction:.2f} | "
            f"real seen ~{w_real/w_fake * n_fake/n_real:.1f}x more than natural rate | "
            f"effective batch ratio real:fake ≈ "
            f"{target_real_fraction:.2f}:{1-target_real_fraction:.2f}")

        return WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).float(),
            num_samples=len(labels),
            replacement=True,
        )

    # ------------------------------------------------------------------ #
    #  DataLoader factory                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def prepare_data_loader(config: dict, mode: str = "train") -> DataLoader:
        """
        Factory method: create a DataLoader with correct sampler and settings.

        In paired training mode, batch_size refers to the number of pairs.
        The actual number of frames per batch is 2 * batch_size.
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
                ratio = config.get("balance_target_ratio", 0.5)
                sampler = dataset.get_weighted_sampler(target_real_fraction=ratio)
                shuffle = False

        n_workers = int(config["workers"])
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=n_workers,
            collate_fn=NeSyDeFakeDataset.collate_fn,
            pin_memory=True,
            drop_last=(mode == "train"),
            persistent_workers=(n_workers > 0),
            prefetch_factor=3 if n_workers > 0 else None,
        )

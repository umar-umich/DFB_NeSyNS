"""
dataset/nesy_defake_dataset.py
==============================
NeSyDeFake Dataset — FRAME-LEVEL with GenD-style PAIRED DATA

v3 changes (2026-03-23):
  - GenD-STYLE PAIRED TRAINING (WACV 2026):
    Each __getitem__ returns a SINGLE frame with source/video metadata.
    Pairing is at the DATA PREPARATION level: the training set must include
    both real and fake frames from the same source videos. Standard shuffled
    batching + UA loss on the hypersphere handles representation learning.
    This gives higher batch diversity vs explicit pair-forcing (N unique
    sources per batch instead of N/2), improving UA loss effectiveness.
  - Added source_uid, video_uid metadata per sample (GenD fields).
  - Batch size N = N frames (not N/2 pairs = N frames as before).
  - GenD paper result: paired data → 90.0% vs unpaired 85.3% cross-dataset.

Design principles:
  1. Single frame per sample — no temporal dimension.
  2. Two active branches (spatial, frequency) receive the SAME frame.
  3. Semantic attributes (211-d) loaded from precomputed Face-LLaVA cache.
  4. Paired data preparation: training set includes both real source video
     and its fake derivatives → UA loss prevents shortcut learning.
  5. source_uid / video_uid exposed in batch for optional source-aware losses.

Collated batch shapes (for batch_size N frames):
  spatial_frames : (N, C, H, W)
  freq_frames    : (N, C, H, W)
  raw_frames     : (N, C, H, W)
  semantic_attrs : (N, semantic_dim)
  label          : (N,)
  source_uid     : (N,)
  video_uid      : (N,)
"""

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

    GenD-style paired training (v3): each __getitem__ returns a single frame
    with source_uid / video_uid metadata. The training set must include
    source-matched real-fake pairs. Standard shuffled batching + UA loss
    on the hypersphere learns manipulation-specific representations.
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

        # ── Precomputed fast semantic features (InsightFace+MediaPipe+DeepFace)
        fast_cfg = config.get('fast_semantic', {})
        self.use_fast_semantic = fast_cfg.get('enabled', False)
        self._fast_semantic_subdir = fast_cfg.get(
            'precomputed_dir', 'fast_semantic')
        self._fast_semantic_dim = fast_cfg.get('output_dim', 58)
        self._fast_semantic_cache = {} if self.use_fast_semantic else None

        # ── VLM feature subset indices (select 64 from 211 FaceBench) ─────
        # When use_refined_features is True, we select the VLM subset from
        # the 211-d FaceBench vector and concatenate with fast features.
        self.use_refined_features = config.get('use_refined_features', False)
        self._vlm_indices = None
        self._vlm_dim = 0
        if self.use_refined_features:
            from networks.nesy_defake.semantic.refined_attributes import (
                VLM_INDICES_IN_FACEBENCH, NUM_VLM_FEATURES)
            self._vlm_dim = NUM_VLM_FEATURES
            if self.use_precomputed_semantic:
                self._vlm_indices = torch.LongTensor(VLM_INDICES_IN_FACEBENCH)

        # ── Precomputed Tier 2 forensic features ───────────────────────────
        ff_cfg = config.get('forensic_features', {})
        self.use_forensic_features = (
            ff_cfg.get('enabled', False)
            and bool(ff_cfg.get('precomputed_dir'))
        )
        self._forensic_subdir = ff_cfg.get('precomputed_dir', 'forensic_features')
        self._forensic_dim = ff_cfg.get('output_dim', 83)
        self._forensic_cache = {} if self.use_forensic_features else None

        # Hybrid OTF mode: keep indices 0..29 from the precomputed .pt (region
        # boundaries, regional blur, symmetry, color, DCT-by-region, quality
        # scores) but recompute indices 30..82 (PPNC / CCNC / SRM / noise /
        # FFT — all pixel-level, augmentation-sensitive) live on the augmented
        # image. Only meaningful when use_forensic_features is True. Requires
        # _forensic_dim == 83 to make the index layout match forensic_helpers.
        self.pixel_forensic_otf = (
            self.use_forensic_features
            and bool(ff_cfg.get('pixel_otf', False))
            and self._forensic_dim == 83
        )
        self._pixel_forensic_extractor = None
        if self.pixel_forensic_otf:
            from .pixel_forensic_extractor import (
                PixelForensicExtractor,
                PIXEL_SLICE,
            )
            self._pixel_forensic_extractor = PixelForensicExtractor()
            self._pixel_forensic_slice = PIXEL_SLICE

        # ── Parent handles JSON parsing, image_list/label_list ────────────
        super().__init__(config, mode)

        # ── K-augment: pre-augmented variant sampling ─────────────────────
        # When k_augment > 0 (train mode), each real frame randomly loads one
        # of K pre-augmented variants (image + matched precomputed features).
        # This replaces OTF augmentation with zero-mismatch precomputed views.
        self._k_augment = config.get('k_augment', 0) if mode == 'train' else 0
        self._augment_variant_paths = {}  # index → [path_aug1, ..., path_augK]
        if self._k_augment > 0:
            self._build_augment_variant_map()

        # ── GenD-style source/video UID maps ──────────────────────────────
        self.paired_training = (
            mode == 'train'
            and config.get('paired_training', True)
        )
        self._build_source_video_maps()

        # ── Sanity report ─────────────────────────────────────────────────
        real_count = sum(1 for la in self.label_list if la == 0)
        fake_count = sum(1 for la in self.label_list if la == 1)

        aug_active = (
            mode == 'train' and self.config.get('use_data_augmentation', False)
        )
        balance_active = (
            mode == 'train' and self.config.get('balance_classes', False)
        )

        # Count paired sources
        n_sources_with_both = 0
        if self.paired_training:
            for src in self._source_to_indices:
                labels_for_src = set(
                    self.label_list[i] for i in self._source_to_indices[src])
                if 0 in labels_for_src and 1 in labels_for_src:
                    n_sources_with_both += 1

        print(
            f"\n{'='*60}"
            f"\nNeSyDeFakeDataset [{mode}] — FRAME-LEVEL (GenD v3)"
            f"\n  Total frames        : {len(self.image_list)}"
            f"\n  Real / Fake         : {real_count} / {fake_count}"
            f"\n  Unique sources      : {len(self._source_to_indices)}"
            f"\n  Sources with R+F    : {n_sources_with_both}"
            f"\n  Unique videos       : {len(self._video_to_uid)}"
            f"\n  Resolution          : {self.resolution}x{self.resolution}"
            f"\n  Augmentation        : {'K-augment (K=' + str(self._k_augment) + ')' if self._k_augment > 0 else ('ON (OTF)' if aug_active else 'OFF')}"
            f"\n  Balanced sampling   : {'ON' if balance_active else 'OFF'}"
            f"\n  Semantic features   : {'ON' if self.use_semantic else 'OFF'}"
            f"\n  Precomputed sem.    : {'ON (' + self._precomputed_subdir + ')' if self.use_precomputed_semantic else 'OFF'}"
            f"\n  Paired data (GenD)  : {'ON' if self.paired_training else 'OFF'}"
            f"\n  Active branches     : spatial, frequency (no temporal)"
            f"\n{'='*60}\n"
        )

    # ------------------------------------------------------------------ #
    #  K-augment: pre-augmented variant mapping                            #
    # ------------------------------------------------------------------ #

    def _build_augment_variant_map(self):
        """Build per-sample mapping to K pre-augmented variant frame paths.

        For each real frame in image_list, generates K alternative paths by
        replacing 'frames/' with 'frames_aug_{k}/' (k=1..K). Only includes
        variants whose augmented image file actually exists on disk.

        Fake frames and frames with no existing augmented variants are skipped
        (they will not be augmented at training time).
        """
        K = self._k_augment
        n_with_variants = 0

        for idx, (path, label) in enumerate(
                zip(self.image_list, self.label_list)):
            if isinstance(path, list):
                path = path[0]

            # Only real frames get augmented variants
            if label != 0:
                continue

            # Only original frames (not already-augmented ones)
            if '/frames_aug_' in path or '\\frames_aug_' in path:
                continue

            variants = []
            for k in range(1, K + 1):
                aug_path = path.replace('/frames/', f'/frames_aug_{k}/')
                if os.path.exists(aug_path):
                    variants.append(aug_path)

            if variants:
                self._augment_variant_paths[idx] = variants
                n_with_variants += 1

        print(f"[K-augment] K={K}, {n_with_variants} real frames have "
              f"pre-augmented variants (avg {sum(len(v) for v in self._augment_variant_paths.values()) / max(n_with_variants, 1):.1f} per frame)")

    # ------------------------------------------------------------------ #
    #  GenD-style source/video UID maps                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_source_video(frame_path: str) -> Optional[str]:
        """
        Extract the source video ID from a frame path.

        For FF++ fake videos like "802_885", the source video is "802".
        For real videos like "929", returns "929".
        For augmented real videos like "929_aug1", returns "929".
        For non-FF++ datasets, returns None (pairing not available).
        """
        sep = "/" if "/" in frame_path else "\\"
        parts = frame_path.split(sep)

        # Find frames directory (handles 'frames' and 'frames_aug_N')
        frames_idx = None
        for pi, part in enumerate(parts):
            if part == 'frames' or part.startswith('frames_aug_'):
                frames_idx = pi
                break
        if frames_idx is None or frames_idx + 1 >= len(parts):
            return None
        video_name = parts[frames_idx + 1]

        # Augmented real video names: "929_aug1" -> source is "929"
        if '_aug' in video_name:
            return video_name.split('_aug')[0]
        # Fake video names have format "source_target" (e.g., "802_885")
        if '_' in video_name:
            return video_name.split('_')[0]
        # Real video names are just the ID (e.g., "802")
        return video_name

    @staticmethod
    def _extract_video_name(frame_path: str) -> Optional[str]:
        """Extract the video directory name from a frame path."""
        sep = "/" if "/" in frame_path else "\\"
        parts = frame_path.split(sep)
        for pi, part in enumerate(parts):
            if part == 'frames' or part.startswith('frames_aug_'):
                if pi + 1 < len(parts):
                    return parts[pi + 1]
        return None

    @staticmethod
    def _extract_source_name(frame_path: str) -> str:
        """
        Extract the manipulation source name from a frame path (GenD-style).
        E.g., '.../manipulated_sequences/Deepfakes/c23/frames/...' → 'Deepfakes'
              '.../original_sequences/youtube/c23/frames/...' → 'youtube'
        """
        sep = "/" if "/" in frame_path else "\\"
        parts = frame_path.split(sep)
        for pi, part in enumerate(parts):
            if part == 'frames' or part.startswith('frames_aug_'):
                # Source name is typically 2 levels up from frames dir
                # e.g. .../Deepfakes/c23/frames/... → parts[pi-2] = 'Deepfakes'
                if pi >= 2:
                    return parts[pi - 2]
        return "unknown"

    def _build_source_video_maps(self):
        """
        Build GenD-style mappings:
          - source_uid: unique ID per manipulation source (0 for all reals, 1+ for fakes)
          - video_uid: unique ID per video
          - source_to_indices: source_video_id → list of dataset indices (for sampling)
        """
        # Collect all source names and video names
        source_names = set()
        video_names = set()
        self._source_to_indices = defaultdict(list)

        self._per_sample_source_id = []   # source video ID per sample
        self._per_sample_video_name = []  # video name per sample
        self._per_sample_source_name = [] # manipulation source name per sample

        for idx, (path, label) in enumerate(
                zip(self.image_list, self.label_list)):
            if isinstance(path, list):
                path = path[0]

            source_id = self._extract_source_video(path)
            video_name = self._extract_video_name(path)
            source_name = self._extract_source_name(path)

            self._per_sample_source_id.append(source_id)
            self._per_sample_video_name.append(video_name)
            self._per_sample_source_name.append(source_name)

            if source_name:
                source_names.add(source_name)
            if video_name:
                video_names.add(video_name)
            if source_id:
                self._source_to_indices[source_id].append(idx)

        # Build source_uid map: all real sources → 0, fake sources → 1, 2, ...
        # (GenD convention)
        real_sources = sorted(s for s in source_names if 'real' in s.lower()
                              or 'youtube' in s.lower() or 'original' in s.lower())
        fake_sources = sorted(s for s in source_names if s not in real_sources)

        self._source_name_to_uid = {}
        for s in real_sources:
            self._source_name_to_uid[s] = 0
        for i, s in enumerate(fake_sources, start=1):
            self._source_name_to_uid[s] = i

        # Build video_uid map
        self._video_to_uid = {v: i for i, v in enumerate(sorted(video_names))}

        # Report paired data stats
        if self.paired_training:
            n_paired_fakes = sum(
                1 for idx, label in enumerate(self.label_list)
                if label == 1
                and self._per_sample_source_id[idx]
                and any(self.label_list[j] == 0
                        for j in self._source_to_indices.get(
                            self._per_sample_source_id[idx], []))
            )
            n_total_fakes = sum(1 for la in self.label_list if la == 1)
            print(f"[GenD PairedData] {n_paired_fakes}/{n_total_fakes} fake "
                  f"frames have source-matched real frames in training set")

    # ------------------------------------------------------------------ #
    #  Path helpers for precomputed feature loading                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_frame_path(frame_path: str):
        """Parse a frame path into (base_dir, pt_video_name, frame_filename, sep).

        For original frames:
          .../c23/frames/001/0000.png → base=.../c23, pt_name=001
        For augmented frames:
          .../c23/frames_aug_4/001/0000.png → base=.../c23, pt_name=001_aug4

        The pt_video_name matches the .pt file naming convention used by the
        precompute pipeline (e.g. fast_semantic/001_aug4.pt).
        """
        sep = "/" if "/" in frame_path else "\\"
        parts = frame_path.split(sep)

        frames_idx = None
        for pi, part in enumerate(parts):
            if part == 'frames' or part.startswith('frames_aug_'):
                frames_idx = pi
                break
        if frames_idx is None or frames_idx + 1 >= len(parts):
            return None, None, None, sep

        frames_dir = parts[frames_idx]       # 'frames' or 'frames_aug_4'
        video_name = parts[frames_idx + 1]   # '001'
        base_dir = sep.join(parts[:frames_idx])
        frame_filename = parts[-1]

        # Always load features from the ORIGINAL video's .pt file.
        # Augmented frames (frames_aug_N/) intentionally share the original's
        # precomputed features — the backbone learns augmentation-invariant
        # representations while features provide clean semantic/forensic signal.
        pt_video_name = video_name

        return base_dir, pt_video_name, frame_filename, sep

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

            # Find frames directory (handles 'frames' and 'frames_aug_N')
            frames_idx = None
            for pi, part in enumerate(parts):
                if part == 'frames' or part.startswith('frames_aug_'):
                    frames_idx = pi
                    break
            if frames_idx is not None:
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
        Handles both 'frames/' and 'frames_aug_N/' directories.
        Returns (precomputed_dim,) tensor. Zero-vector on any failure.
        """
        try:
            base_dir, pt_video_name, frame_filename, sep = \
                self._parse_frame_path(frame_path)
            if base_dir is None:
                return torch.zeros(self._precomputed_dim)

            cache_key = f"{base_dir}/{pt_video_name}"
            if cache_key not in self._precomputed_cache:
                pt_path = os.path.join(
                    base_dir, self._precomputed_subdir, f'{pt_video_name}.pt')
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
                for idx, fp in enumerate(frame_paths):
                    if fp.endswith(frame_filename):
                        return features[idx]

            # Fallback: index by frame number
            frame_filename = frame_filename
            frame_num = int(os.path.splitext(frame_filename)[0])
            if frame_num < features.shape[0]:
                return features[frame_num]

            return torch.zeros(self._precomputed_dim)

        except Exception:
            return torch.zeros(self._precomputed_dim)

    # ------------------------------------------------------------------ #
    #  Precomputed Tier 2 forensic feature loading                         #
    # ------------------------------------------------------------------ #

    def _load_forensic_features(self, frame_path: str) -> torch.Tensor:
        """
        Load precomputed forensic features from .pt file.
        Same caching pattern as _load_precomputed_semantic.
        Returns (forensic_dim,) tensor. Zero-vector on any failure.
        """
        try:
            base_dir, pt_video_name, frame_filename, sep = \
                self._parse_frame_path(frame_path)
            if base_dir is None:
                return torch.zeros(self._forensic_dim)

            cache_key = f"{base_dir}/{pt_video_name}"
            if cache_key not in self._forensic_cache:
                pt_path = os.path.join(
                    base_dir, self._forensic_subdir, f'{pt_video_name}.pt')
                if os.path.exists(pt_path):
                    data = torch.load(pt_path, map_location='cpu',
                                      weights_only=False)
                    self._forensic_cache[cache_key] = data
                else:
                    self._forensic_cache[cache_key] = None

            cached = self._forensic_cache[cache_key]
            if cached is None:
                return torch.zeros(self._forensic_dim)

            features = cached['features']  # (n_frames, D)
            frame_paths_cached = cached.get('frame_paths', [])

            # Try exact frame match
            if frame_paths_cached:
                for idx, fp in enumerate(frame_paths_cached):
                    if fp.endswith(frame_filename):
                        return features[idx]

            # Fallback: index by frame number
            frame_num = int(os.path.splitext(frame_filename)[0])
            if frame_num < features.shape[0]:
                return features[frame_num]

            return torch.zeros(self._forensic_dim)

        except Exception:
            return torch.zeros(self._forensic_dim)

    # ------------------------------------------------------------------ #
    #  Precomputed fast semantic feature loading (58-d)                     #
    # ------------------------------------------------------------------ #

    def _load_fast_semantic(self, frame_path: str) -> torch.Tensor:
        """
        Load precomputed fast semantic features (58-d) from .pt file.
        Same caching pattern as _load_precomputed_semantic / _load_forensic_features.
        Returns (fast_semantic_dim,) tensor. Zero-vector on any failure.
        """
        try:
            base_dir, pt_video_name, frame_filename, sep = \
                self._parse_frame_path(frame_path)
            if base_dir is None:
                return torch.zeros(self._fast_semantic_dim)

            cache_key = f"{base_dir}/{pt_video_name}"
            if cache_key not in self._fast_semantic_cache:
                pt_path = os.path.join(
                    base_dir, self._fast_semantic_subdir, f'{pt_video_name}.pt')
                if os.path.exists(pt_path):
                    data = torch.load(pt_path, map_location='cpu',
                                      weights_only=False)
                    self._fast_semantic_cache[cache_key] = data
                else:
                    self._fast_semantic_cache[cache_key] = None

            cached = self._fast_semantic_cache[cache_key]
            if cached is None:
                return torch.zeros(self._fast_semantic_dim)

            features = cached['features']  # (n_frames, 58)
            frame_paths_cached = cached.get('frame_paths', [])

            # Try exact frame match
            if frame_paths_cached:
                for idx, fp in enumerate(frame_paths_cached):
                    if fp.endswith(frame_filename):
                        return features[idx]

            # Fallback: index by frame number
            frame_num = int(os.path.splitext(frame_filename)[0])
            if frame_num < features.shape[0]:
                return features[frame_num]

            return torch.zeros(self._fast_semantic_dim)

        except Exception:
            return torch.zeros(self._fast_semantic_dim)

    # ------------------------------------------------------------------ #
    #  Single-frame loading helper                                         #
    # ------------------------------------------------------------------ #

    def _load_single_frame(self, index: int) -> dict:
        """Load and preprocess a single frame by index."""
        frame_path = self.image_list[index]
        label = self.label_list[index]

        if isinstance(frame_path, list):
            frame_path = frame_path[0]

        # K-augment: for real frames with pre-augmented variants, randomly
        # pick one variant. The augmented image + its precomputed features
        # are perfectly matched, so OTF augmentation is disabled entirely.
        # When K-augment is active, OTF aug is off for ALL frames (real
        # frames use pre-augmented variants; fake frames stay unaugmented
        # as is standard in deepfake detection).
        use_otf_aug = (
            self.mode == 'train'
            and self._k_augment == 0
            and self.config.get('use_data_augmentation', False)
        )
        if self._k_augment > 0 and index in self._augment_variant_paths:
            variants = self._augment_variant_paths[index]
            frame_path = random.choice(variants)

        try:
            image = self.load_rgb(frame_path)
        except Exception as e:
            print(f"[NeSyDeFake] Failed to load {frame_path}: {e}")
            return self._load_single_frame(0)
        image = np.array(image)

        if use_otf_aug:
            image, _, _ = self.data_aug(image, None, None)

        img_tensor = self.to_tensor(image)

        spatial_frames = self.normalize_spatial(img_tensor)
        freq_frames = self.normalize_frequency(img_tensor)
        raw_frames = img_tensor

        semantic_attrs = self._load_semantic_for_frame(frame_path, index)

        # Precomputed semantic features
        if self.use_refined_features and self.use_fast_semantic and self.use_precomputed_semantic:
            # Combined mode: [fast(58) || vlm(64)] = 122-d
            fast_feats = self._load_fast_semantic(frame_path)         # (58,)
            vlm_feats = self._load_precomputed_semantic(frame_path)   # (64,) or (211,)
            # If precomputed features are full 211-d FaceBench, select VLM subset
            if self._vlm_indices is not None and vlm_feats.shape[0] > self._vlm_dim:
                vlm_feats = vlm_feats[self._vlm_indices]              # (64,)
            precomputed_attrs = torch.cat([fast_feats, vlm_feats])    # (122,)
        elif self.use_refined_features and self.use_fast_semantic:
            # Fast-only mode: [fast(58) || zeros(64)] = 122-d
            # VLM features not yet available — zero-fill the VLM portion
            fast_feats = self._load_fast_semantic(frame_path)         # (58,)
            vlm_zeros = torch.zeros(self._vlm_dim)                    # (64,)
            precomputed_attrs = torch.cat([fast_feats, vlm_zeros])    # (122,)
        elif self.use_fast_semantic and not self.use_refined_features:
            # Fast-only mode without VLM: just fast(58)
            precomputed_attrs = self._load_fast_semantic(frame_path)   # (58,)
        elif self.use_precomputed_semantic:
            precomputed_attrs = self._load_precomputed_semantic(frame_path)
        else:
            precomputed_attrs = torch.zeros(1)  # placeholder

        # Tier 2 forensic features
        # Hybrid path (pixel_forensic_otf=True): load 83-d from precomputed
        # .pt, then overwrite indices 30..82 with features recomputed live on
        # the AUGMENTED numpy image. This matches the pixel-level forensic
        # signals (SRM / noise / FFT / PPNC / CCNC) to what the backbone
        # actually sees, while keeping the expensive region-based features
        # (0..29) from precompute since they need SegFormer parsing maps.
        # When K-augment is active, all 83 features are precomputed on the
        # augmented view, so skip OTF recompute (use_otf_aug is False).
        if self.use_forensic_features:
            forensic_features = self._load_forensic_features(frame_path)
            if self.pixel_forensic_otf and use_otf_aug and forensic_features.shape[0] == 83:
                pixel_np = self._pixel_forensic_extractor(image)
                forensic_features = forensic_features.clone()
                forensic_features[self._pixel_forensic_slice] = torch.from_numpy(pixel_np)
        else:
            forensic_features = torch.zeros(self._forensic_dim)

        # GenD-style source/video UIDs
        source_name = self._per_sample_source_name[index] if index < len(self._per_sample_source_name) else "unknown"
        source_uid = self._source_name_to_uid.get(source_name, 0)
        video_name = self._per_sample_video_name[index] if index < len(self._per_sample_video_name) else None
        video_uid = self._video_to_uid.get(video_name, 0) if video_name else 0

        return {
            "spatial_frames":    spatial_frames,
            "freq_frames":       freq_frames,
            "raw_frames":        raw_frames,
            "semantic_attrs":    torch.from_numpy(semantic_attrs),
            "precomputed_attrs": precomputed_attrs,
            "forensic_features": forensic_features,
            "label":             label,
            "source_uid":        source_uid,
            "video_uid":         video_uid,
            "name":              frame_path,
        }

    # ------------------------------------------------------------------ #
    #  Core dataset interface                                              #
    # ------------------------------------------------------------------ #

    def __getitem__(self, index: int) -> dict:
        """
        GenD-style: return a SINGLE frame with source/video metadata.
        Standard shuffled batching + UA loss handles representation learning.
        """
        return self._load_single_frame(index)

    def __len__(self) -> int:
        """GenD-style: total frame count (each frame is one sample)."""
        return len(self.image_list)

    # ------------------------------------------------------------------ #
    #  Collation                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def collate_fn(batch: list) -> dict:
        """
        GenD-style collation: stack N single-frame dicts into batched tensors.

        Output shapes (for batch_size N):
            spatial_frames : (N, C, H, W)
            freq_frames    : (N, C, H, W)
            raw_frames     : (N, C, H, W)
            semantic_attrs : (N, semantic_dim)
            label          : (N,)
            source_uid     : (N,)
            video_uid      : (N,)
        """
        spatial_frames = torch.stack([s["spatial_frames"] for s in batch])
        freq_frames    = torch.stack([s["freq_frames"]    for s in batch])
        raw_frames     = torch.stack([s["raw_frames"]     for s in batch])
        semantic_attrs = torch.stack([s["semantic_attrs"]  for s in batch])
        labels         = torch.tensor([s["label"] for s in batch],
                                       dtype=torch.long)
        source_uids    = torch.tensor([s["source_uid"] for s in batch],
                                       dtype=torch.long)
        video_uids     = torch.tensor([s["video_uid"] for s in batch],
                                       dtype=torch.long)
        names          = [s["name"] for s in batch]

        # Precomputed Face-LLaVA attributes (if available)
        precomputed_attrs = None
        if "precomputed_attrs" in batch[0]:
            precomputed_attrs = torch.stack(
                [s["precomputed_attrs"] for s in batch])

        # Precomputed Tier 2 forensic features (if available)
        forensic_features = None
        if "forensic_features" in batch[0]:
            forensic_features = torch.stack(
                [s["forensic_features"] for s in batch])

        return {
            "spatial_frames":    spatial_frames,
            "freq_frames":       freq_frames,
            "raw_frames":        raw_frames,
            "semantic_attrs":    semantic_attrs,
            "precomputed_attrs": precomputed_attrs,
            "forensic_features": forensic_features,
            "label":             labels,
            "source_uid":        source_uids,
            "video_uid":         video_uids,
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
        GenD-style: standard balanced real/fake sampling over all frames.
        Each sample is a single frame — no pair-specific logic needed.
        """
        labels = np.array(self.label_list)
        n_real = (labels == 0).sum()
        n_fake = (labels == 1).sum()

        w_real = target_real_fraction / max(n_real, 1)
        w_fake = (1.0 - target_real_fraction) / max(n_fake, 1)
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
        GenD-style: standard shuffled DataLoader. batch_size = number of frames.
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

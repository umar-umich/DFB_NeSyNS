"""
pixel_forensic_extractor.py
============================
On-the-fly pixel-level forensic feature extraction (features 30..82 of the
83-d forensic feature vector).

Why this exists
---------------
Forensic features 0..29 (region boundary gradients, regional blur, symmetry,
color, DCT by parsing region, and InsightFace/MediaPipe quality scores) depend
on a SegFormer face-parsing map and InsightFace/MediaPipe inference — none of
which can cheaply run inside DataLoader workers. So 0..29 stay precomputed.

Features 30..82 are the pixel-level forensic signals that actually change
under augmentation (blur, JPEG, brightness/contrast, noise):

    30..37  PPNC   — Paired Patch Noise Consistency
    38..45  CCNC   — Cross-Channel Noise Consistency
    46..60  SRM    — Spatial Rich Model filter bank (5 filters × mean/std/kurt)
    61..72  Multi-scale noise residuals (s1, s2, s4 × mean/std/kurt + xcorrs)
    73..82  FFT spectral features (8 radial bins + slope + HF ratio)

These are pure `cv2`/`numpy`/`scipy` ops — no GPU, no model state — and
recompute in ~15-25ms per image. Safe to run inside DataLoader workers.

Usage
-----
    ext = PixelForensicExtractor()
    feats_30_82 = ext(augmented_rgb_uint8)     # (53,) float32
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Ensure preprocessing/ is importable even when invoked from training/
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
_PREPROC_DIR = os.path.join(_REPO_ROOT, 'preprocessing')
if _PREPROC_DIR not in sys.path:
    sys.path.insert(0, _PREPROC_DIR)

import cv2  # noqa: E402
from forensic_helpers import (  # noqa: E402
    extract_ppnc,
    extract_ccnc,
    extract_srm,
    extract_multiscale_noise,
    extract_spectral,
)


# Index layout inside the 83-d forensic vector (from forensic_helpers.FEATURE_NAMES)
PIXEL_SLICE = slice(30, 83)                 # 53 features
PIXEL_DIM = PIXEL_SLICE.stop - PIXEL_SLICE.start  # 53

# Per-block offsets inside the 30..82 slice (relative to start=30)
_PPNC_LOCAL    = slice(0, 8)     # 30..37
_CCNC_LOCAL    = slice(8, 16)    # 38..45
_SRM_LOCAL     = slice(16, 31)   # 46..60
_NOISE_LOCAL   = slice(31, 43)   # 61..72
_SPECTRAL_LOCAL = slice(43, 53)  # 73..82


class PixelForensicExtractor:
    """Stateless extractor for pixel-level forensic features (30..82).

    Instances hold no model state, no CUDA context, no tensors — they are
    safe to construct per-worker or share across threads. The `__call__`
    operator takes a single HxWx3 uint8 RGB image and returns a (53,)
    float32 numpy array slotted to fill forensic indices 30..82.
    """

    __slots__ = ()

    def __call__(self, image_rgb: np.ndarray) -> np.ndarray:
        """Compute pixel forensic features (53-d) for one augmented RGB image.

        Args:
            image_rgb: uint8 numpy array of shape (H, W, 3), RGB ordering.

        Returns:
            (53,) float32 numpy array.
        """
        if image_rgb is None or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            return np.zeros(PIXEL_DIM, dtype=np.float32)

        # Contiguous uint8 is required by the cv2 color conversions below.
        if image_rgb.dtype != np.uint8:
            image_rgb = image_rgb.astype(np.uint8, copy=False)
        if not image_rgb.flags['C_CONTIGUOUS']:
            image_rgb = np.ascontiguousarray(image_rgb)

        try:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        except cv2.error:
            return np.zeros(PIXEL_DIM, dtype=np.float32)

        out = np.zeros(PIXEL_DIM, dtype=np.float32)
        try:
            out[_PPNC_LOCAL]     = extract_ppnc(gray)
            out[_CCNC_LOCAL]     = extract_ccnc(bgr)
            out[_SRM_LOCAL]      = extract_srm(gray)
            out[_NOISE_LOCAL]    = extract_multiscale_noise(gray)
            out[_SPECTRAL_LOCAL] = extract_spectral(gray)
        except Exception:
            # Any individual helper failure → return zeros for that block.
            # We deliberately do not log per-frame (hot path, worker-safe).
            pass
        # Replace any NaN/Inf from degenerate regions with zeros so downstream
        # linear layers stay well-conditioned.
        if not np.isfinite(out).all():
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out

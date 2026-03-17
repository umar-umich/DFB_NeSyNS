"""
networks/nesy_defake/semantic/forensic_features.py
===================================================
Tier 2: Pixel-Level Forensic Feature definitions.

This module provides feature names and configuration for precomputed
forensic features. The actual feature extraction happens in
preprocessing/precompute_forensic_features.py — features are loaded
from .pt files at training time (same pattern as FaceBench attributes).

30 features covering:
  - Boundary gradients (6): Sobel gradient magnitude at face parsing boundaries
  - Regional texture/blur (6): Laplacian variance per parsed region
  - Left-right symmetry (4): Intensity and landmark asymmetry
  - Color consistency (4): Chi-squared histogram distance between regions
  - Frequency anomaly (6): DCT high-frequency energy per region
  - Quality metrics (4): Antispoof, detection quality, blendshape symmetry
"""

from typing import List


FORENSIC_FEATURE_NAMES = [
    # Boundary gradients (6)
    'ff_grad_skin_bg',
    'ff_grad_eye_skin',
    'ff_grad_lip_skin',
    'ff_grad_nose_skin',
    'ff_grad_brow_skin',
    'ff_grad_mean_boundary',
    # Regional texture / blur (6)
    'ff_blur_skin',
    'ff_blur_eye',
    'ff_blur_mouth',
    'ff_blur_nose',
    'ff_blur_ratio_eye_skin',
    'ff_blur_ratio_mouth_skin',
    # Left-right symmetry (4)
    'ff_sym_eye',
    'ff_sym_mouth',
    'ff_sym_cheek',
    'ff_sym_jawline',
    # Color consistency (4)
    'ff_color_eye_skin',
    'ff_color_mouth_skin',
    'ff_color_nose_skin',
    'ff_color_lr_cheek',
    # Frequency anomaly (6)
    'ff_dct_hf_skin',
    'ff_dct_hf_eye',
    'ff_dct_hf_mouth',
    'ff_dct_hf_nose',
    'ff_dct_ratio_eye_skin',
    'ff_dct_ratio_mouth_skin',
    # Quality metrics (4)
    'ff_quality_antispoof',
    'ff_quality_det_score',
    'ff_quality_blendshape_sym',
    'ff_quality_landmark_jitter',
]

NUM_FORENSIC_FEATURES = len(FORENSIC_FEATURE_NAMES)
assert NUM_FORENSIC_FEATURES == 30, f"Expected 30 forensic features, got {NUM_FORENSIC_FEATURES}"


def get_forensic_feature_names() -> List[str]:
    """Interpretable names for causal graph visualization."""
    return list(FORENSIC_FEATURE_NAMES)

"""
networks/nesy_defake/semantic/forensic_features.py
===================================================
Tier 2: Pixel-Level Forensic Feature definitions.

This module provides feature names and configuration for precomputed
forensic features. The actual feature extraction happens in
preprocessing/precompute_forensic_features.py — features are loaded
from .pt files at training time (same pattern as FaceBench attributes).

83 features covering:
  - Boundary gradients (6): Sobel gradient magnitude at face parsing boundaries
  - Regional texture/blur (6): Laplacian variance per parsed region
  - Left-right symmetry (4): Intensity and landmark asymmetry
  - Color consistency (4): Chi-squared histogram distance between regions
  - Frequency anomaly (6): DCT high-frequency energy per region
  - Quality metrics (4): Antispoof, detection quality, blendshape symmetry
  - Pairwise patch noise consistency (8): PPNC region statistics
  - Cross-channel noise consistency (8): YCrCb noise energy/correlation
  - SRM noise residuals (15): 5 filter kernels × (mean, std, kurtosis)
  - Multi-scale noise analysis (12): 3 scales × (mean, std, kurt) + cross-corr
  - FFT spectral features (10): 8 radial bins + slope + HF ratio
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
    # Pairwise patch noise consistency — PPNC (8)
    'ff_ppnc_outer_eye_mean',
    'ff_ppnc_outer_eye_std',
    'ff_ppnc_inner_eye_mean',
    'ff_ppnc_inner_eye_std',
    'ff_ppnc_cheek_mean',
    'ff_ppnc_cheek_std',
    'ff_ppnc_jaw_mean',
    'ff_ppnc_jaw_std',
    # Cross-channel noise consistency — CCNC (8)
    'ff_ccnc_energy_y',
    'ff_ccnc_energy_cr',
    'ff_ccnc_energy_cb',
    'ff_ccnc_corr_y_cr',
    'ff_ccnc_corr_y_cb',
    'ff_ccnc_corr_cr_cb',
    'ff_ccnc_ratio_y_cr',
    'ff_ccnc_ratio_y_cb',
    # SRM noise residual filters (15): 5 kernels × (mean, std, kurtosis)
    'ff_srm_hedge_mean',
    'ff_srm_hedge_std',
    'ff_srm_hedge_kurt',
    'ff_srm_vedge_mean',
    'ff_srm_vedge_std',
    'ff_srm_vedge_kurt',
    'ff_srm_hlap_mean',
    'ff_srm_hlap_std',
    'ff_srm_hlap_kurt',
    'ff_srm_square_mean',
    'ff_srm_square_std',
    'ff_srm_square_kurt',
    'ff_srm_diag_mean',
    'ff_srm_diag_std',
    'ff_srm_diag_kurt',
    # Multi-scale noise analysis (12): scales 1,2,4 × (mean, std, kurt) + xcorr
    'ff_noise_s1_mean',
    'ff_noise_s1_std',
    'ff_noise_s1_kurt',
    'ff_noise_s2_mean',
    'ff_noise_s2_std',
    'ff_noise_s2_kurt',
    'ff_noise_s4_mean',
    'ff_noise_s4_std',
    'ff_noise_s4_kurt',
    'ff_noise_xcorr_s1s2',
    'ff_noise_xcorr_s2s4',
    'ff_noise_fine_coarse_ratio',
    # FFT spectral features (10): 8 radial bins + slope + HF ratio
    'ff_fft_bin0',
    'ff_fft_bin1',
    'ff_fft_bin2',
    'ff_fft_bin3',
    'ff_fft_bin4',
    'ff_fft_bin5',
    'ff_fft_bin6',
    'ff_fft_bin7',
    'ff_fft_slope',
    'ff_fft_hf_ratio',
]

NUM_FORENSIC_FEATURES = len(FORENSIC_FEATURE_NAMES)
assert NUM_FORENSIC_FEATURES == 83, f"Expected 83 forensic features, got {NUM_FORENSIC_FEATURES}"


def get_forensic_feature_names() -> List[str]:
    """Interpretable names for causal graph visualization."""
    return list(FORENSIC_FEATURE_NAMES)

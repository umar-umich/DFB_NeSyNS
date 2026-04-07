"""
forensic_helpers.py
===================
Pure CV functions for extracting pixel-level forensic features.
No GPU models or state — these run on CPU with numpy/cv2/scipy.

Feature groups (83-d total):
  [0-29]  Region-based forensics (SegFormer parsing): boundary gradients,
          blur, symmetry, color consistency, DCT HF energy, quality metrics
  [30-37] PPNC — Paired Patch Noise Consistency (symmetric face noise)
  [38-45] CCNC — Cross-Channel Noise Consistency (YCbCr noise correlations)
  [46-60] SRM — Spatial Rich Model high-pass filter bank (steganalysis)
  [61-72] Multi-Scale Noise Residuals (LoG at 3 scales)
  [73-82] Spectral Features (radial FFT spectrum)
"""

import cv2
import numpy as np
from scipy.fft import dctn


# SegFormer face-parsing labels (jonathandinu/face-parsing)
LABEL_BG = 0
LABEL_SKIN = 1
LABEL_L_BROW = 2
LABEL_R_BROW = 3
LABEL_L_EYE = 4
LABEL_R_EYE = 5
LABEL_GLASSES = 6
LABEL_L_EAR = 7
LABEL_R_EAR = 8
LABEL_NOSE = 10
LABEL_INNER_MOUTH = 11
LABEL_UPPER_LIP = 12
LABEL_LOWER_LIP = 13
LABEL_NECK = 14
LABEL_HAIR = 17

# Region groups
SKIN_LABELS = {LABEL_SKIN}
EYE_LABELS = {LABEL_L_EYE, LABEL_R_EYE}
BROW_LABELS = {LABEL_L_BROW, LABEL_R_BROW}
MOUTH_LABELS = {LABEL_INNER_MOUTH, LABEL_UPPER_LIP, LABEL_LOWER_LIP}
NOSE_LABELS = {LABEL_NOSE}
BG_LABELS = {LABEL_BG}
LEFT_EYE = {LABEL_L_EYE}
RIGHT_EYE = {LABEL_R_EYE}
LEFT_BROW = {LABEL_L_BROW}
RIGHT_BROW = {LABEL_R_BROW}

FEATURE_NAMES = [
    # Region-based forensics (0-29)
    'ff_grad_skin_bg', 'ff_grad_eye_skin', 'ff_grad_lip_skin',
    'ff_grad_nose_skin', 'ff_grad_brow_skin', 'ff_grad_mean_boundary',
    'ff_blur_skin', 'ff_blur_eye', 'ff_blur_mouth', 'ff_blur_nose',
    'ff_blur_ratio_eye_skin', 'ff_blur_ratio_mouth_skin',
    'ff_sym_eye', 'ff_sym_mouth', 'ff_sym_cheek', 'ff_sym_jawline',
    'ff_color_eye_skin', 'ff_color_mouth_skin', 'ff_color_nose_skin',
    'ff_color_lr_cheek',
    'ff_dct_hf_skin', 'ff_dct_hf_eye', 'ff_dct_hf_mouth', 'ff_dct_hf_nose',
    'ff_dct_ratio_eye_skin', 'ff_dct_ratio_mouth_skin',
    'ff_quality_antispoof', 'ff_quality_det_score',
    'ff_quality_blendshape_sym', 'ff_quality_landmark_jitter',
    # PPNC — Paired Patch Noise Consistency (30-37)
    'ff_ppnc_outer_eye_mean', 'ff_ppnc_outer_eye_std',
    'ff_ppnc_inner_eye_mean', 'ff_ppnc_inner_eye_std',
    'ff_ppnc_cheek_mean', 'ff_ppnc_cheek_std',
    'ff_ppnc_jaw_mean', 'ff_ppnc_jaw_std',
    # CCNC — Cross-Channel Noise Consistency (38-45)
    'ff_ccnc_energy_y', 'ff_ccnc_energy_cr', 'ff_ccnc_energy_cb',
    'ff_ccnc_corr_y_cr', 'ff_ccnc_corr_y_cb', 'ff_ccnc_corr_cr_cb',
    'ff_ccnc_ratio_y_cr', 'ff_ccnc_ratio_y_cb',
    # SRM — Spatial Rich Model filter bank (46-60)
    'ff_srm_hedge_mean', 'ff_srm_hedge_std', 'ff_srm_hedge_kurt',
    'ff_srm_vedge_mean', 'ff_srm_vedge_std', 'ff_srm_vedge_kurt',
    'ff_srm_hlap_mean', 'ff_srm_hlap_std', 'ff_srm_hlap_kurt',
    'ff_srm_square_mean', 'ff_srm_square_std', 'ff_srm_square_kurt',
    'ff_srm_diag_mean', 'ff_srm_diag_std', 'ff_srm_diag_kurt',
    # Multi-Scale Noise Residuals (61-72)
    'ff_noise_s1_mean', 'ff_noise_s1_std', 'ff_noise_s1_kurt',
    'ff_noise_s2_mean', 'ff_noise_s2_std', 'ff_noise_s2_kurt',
    'ff_noise_s4_mean', 'ff_noise_s4_std', 'ff_noise_s4_kurt',
    'ff_noise_xcorr_s1s2', 'ff_noise_xcorr_s2s4',
    'ff_noise_fine_coarse_ratio',
    # Spectral Features — FFT (73-82)
    'ff_fft_bin0', 'ff_fft_bin1', 'ff_fft_bin2', 'ff_fft_bin3',
    'ff_fft_bin4', 'ff_fft_bin5', 'ff_fft_bin6', 'ff_fft_bin7',
    'ff_fft_slope', 'ff_fft_hf_ratio',
]
NUM_FORENSIC_FEATURES = 83
assert len(FEATURE_NAMES) == NUM_FORENSIC_FEATURES


def region_mask(parsing_map: np.ndarray, labels: set) -> np.ndarray:
    mask = np.zeros_like(parsing_map, dtype=np.uint8)
    for l in labels:
        mask |= (parsing_map == l).astype(np.uint8)
    return mask


def boundary_gradient(gray: np.ndarray, mask_a: np.ndarray,
                      mask_b: np.ndarray) -> float:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_a = cv2.dilate(mask_a, kernel, iterations=1)
    dilated_b = cv2.dilate(mask_b, kernel, iterations=1)
    boundary = dilated_a & dilated_b
    if boundary.sum() < 10:
        return 0.0
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    return float(grad_mag[boundary > 0].mean()) / 255.0


def laplacian_variance(gray: np.ndarray, mask: np.ndarray,
                       lap: np.ndarray = None) -> float:
    if mask.sum() < 50:
        return 0.0
    if lap is None:
        lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap[mask > 0].var()) / (255.0 ** 2)


def region_mean_intensity(gray: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() < 10:
        return 0.0
    return float(gray[mask > 0].mean()) / 255.0


def safe_ratio(a: float, b: float) -> float:
    return a / (b + 1e-10)


def color_histogram_distance(lab_img: np.ndarray, mask_a: np.ndarray,
                              mask_b: np.ndarray) -> float:
    if mask_a.sum() < 50 or mask_b.sum() < 50:
        return 0.0
    hist_a, hist_b = [], []
    for c in range(3):
        channel = lab_img[:, :, c]
        ha, _ = np.histogram(channel[mask_a > 0], bins=32, range=(0, 256))
        hb, _ = np.histogram(channel[mask_b > 0], bins=32, range=(0, 256))
        hist_a.append(ha.astype(np.float64))
        hist_b.append(hb.astype(np.float64))
    hist_a = np.concatenate(hist_a)
    hist_b = np.concatenate(hist_b)
    hist_a = hist_a / (hist_a.sum() + 1e-10)
    hist_b = hist_b / (hist_b.sum() + 1e-10)
    chi2 = np.sum((hist_a - hist_b) ** 2 / (hist_a + hist_b + 1e-10))
    return min(float(chi2) / 2.0, 1.0)


def dct_hf_energy(gray: np.ndarray, mask: np.ndarray,
                  hf_threshold: float = 0.5) -> float:
    if mask.sum() < 100:
        return 0.0
    ys, xs = np.nonzero(mask)
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    if (y2 - y1) < 8 or (x2 - x1) < 8:
        return 0.0
    crop = gray[y1:y2, x1:x2].astype(np.float64)
    crop_mask = mask[y1:y2, x1:x2]
    crop = crop * crop_mask
    dct_coeff = dctn(crop, type=2, norm='ortho')
    energy = dct_coeff ** 2
    h, w = energy.shape
    hf_y = int(h * hf_threshold)
    hf_x = int(w * hf_threshold)
    total_energy = energy.sum()
    if total_energy < 1e-10:
        return 0.0
    hf_energy = energy[hf_y:, :].sum() + energy[:hf_y, hf_x:].sum()
    return min(float(hf_energy / total_energy), 1.0)


def left_right_symmetry(gray: np.ndarray, left_mask: np.ndarray,
                        right_mask: np.ndarray) -> float:
    left_mean = region_mean_intensity(gray, left_mask)
    right_mean = region_mean_intensity(gray, right_mask)
    return abs(left_mean - right_mean)


def cheek_masks(parsing_map: np.ndarray):
    skin_mask = region_mask(parsing_map, SKIN_LABELS)
    h, w = parsing_map.shape
    mid_x = w // 2
    left_cheek = skin_mask.copy()
    left_cheek[:, mid_x:] = 0
    right_cheek = skin_mask.copy()
    right_cheek[:, :mid_x] = 0
    return left_cheek, right_cheek


# ─────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────

def _kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis of a flattened array (optimized)."""
    x = x.ravel().astype(np.float64)
    m = x.mean()
    c = x - m
    s2 = np.dot(c, c) / len(c)  # variance via dot product
    if s2 < 1e-16:
        return 0.0
    m4 = np.dot(c * c, c * c) / len(c)  # 4th moment via dot product
    return float(m4 / (s2 * s2) - 3.0)


def _fast_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Fast Pearson correlation via dot product — O(n), no covariance matrix."""
    a_f, b_f = a.ravel(), b.ravel()
    a_c = a_f - a_f.mean()
    b_c = b_f - b_f.mean()
    num = np.dot(a_c, b_c)
    den = np.sqrt(np.dot(a_c, a_c) * np.dot(b_c, b_c))
    return float(num / den) if den > 1e-8 else 0.0


# ─────────────────────────────────────────────────────────────────────
# PPNC — Paired Patch Noise Consistency (8-d)
# ─────────────────────────────────────────────────────────────────────
# Compares noise statistics between symmetric face regions.
# Real cameras produce spatially correlated sensor noise — symmetric
# patches have similar noise distributions.  Deepfake generators don't
# model sensor noise correlation, so symmetric patches diverge.

# Fixed symmetric patch locations (fractions of image width/height)
# for face-cropped images.  No landmarks needed.
_PPNC_PAIRS = [
    # (left_cx_frac, cy_frac, right_cx_frac, cy_frac)  name
    (0.20, 0.35, 0.80, 0.35),   # outer eye regions
    (0.35, 0.35, 0.65, 0.35),   # inner eye regions
    (0.20, 0.55, 0.80, 0.55),   # cheek regions
    (0.25, 0.80, 0.75, 0.80),   # jaw regions
]


def extract_ppnc(gray: np.ndarray) -> np.ndarray:
    """
    Paired Patch Noise Consistency features (8-d).
    Works on face-cropped images using fixed symmetric patch locations.
    """
    h, w = gray.shape
    patch_r = max(int(min(h, w) * 0.08), 4)

    # Noise layer (medium-scale Gaussian)
    blurred = cv2.GaussianBlur(gray, (0, 0), 2.0).astype(np.float32)
    noise = gray.astype(np.float32) - blurred

    features = []
    for lcx_f, lcy_f, rcx_f, rcy_f in _PPNC_PAIRS:
        lx, ly = int(lcx_f * w), int(lcy_f * h)
        rx, ry = int(rcx_f * w), int(rcy_f * h)

        def _patch(cx, cy):
            y0, y1 = max(0, cy - patch_r), min(h, cy + patch_r)
            x0, x1 = max(0, cx - patch_r), min(w, cx + patch_r)
            p = noise[y0:y1, x0:x1]
            return p if p.size > 0 else np.zeros((1, 1), dtype=np.float32)

        lp, rp = _patch(lx, ly), _patch(rx, ry)
        features.append(abs(float(np.mean(lp)) - float(np.mean(rp))))
        features.append(abs(float(np.std(lp)) - float(np.std(rp))))

    return np.array(features, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────
# CCNC — Cross-Channel Noise Consistency (8-d)
# ─────────────────────────────────────────────────────────────────────
# Real cameras produce noise with specific cross-channel correlations
# (Bayer filter + ISP pipeline).  Deepfake generators process colour
# channels independently, breaking natural cross-channel structure.

def extract_ccnc(image_bgr: np.ndarray) -> np.ndarray:
    """
    Cross-channel noise consistency in YCrCb colour space (8-d).
    """
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    channels = [ycrcb[:, :, i].astype(np.float32) for i in range(3)]

    noises = []
    for ch in channels:
        blurred = cv2.GaussianBlur(ch, (0, 0), 2.0)
        noises.append(ch - blurred)

    features = []

    # Per-channel noise energy
    energies = []
    for n in noises:
        e = float(np.mean(n ** 2))
        energies.append(e)
        features.append(e)

    # Cross-channel noise correlations (fast dot-product Pearson)
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        features.append(_fast_corr(noises[i], noises[j]))

    # Cross-channel energy ratios (luma vs chroma)
    features.append(energies[0] / max(energies[1], 1e-8))
    features.append(energies[0] / max(energies[2], 1e-8))

    return np.array(features, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────
# SRM — Spatial Rich Model filter bank (15-d)
# ─────────────────────────────────────────────────────────────────────
# Five steganalysis kernels (Fridrich & Kodovský, IEEE TIFS 2012).
# Suppress semantic content, reveal residual manipulation traces.

_SRM_FILTERS = [
    np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=np.float32),   # h-edge
    np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=np.float32),   # v-edge
    np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float32),   # h-lap
    np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32),  # square
    np.array([[0, 0, -1], [0, 2, 0], [-1, 0, 0]], dtype=np.float32),  # diagonal
]


def extract_srm(gray: np.ndarray) -> np.ndarray:
    """
    SRM high-pass filter bank residual statistics (15-d).
    5 kernels × (mean_abs, std, kurtosis).
    """
    features = []
    img = gray.astype(np.float32)
    for kernel in _SRM_FILTERS:
        residual = cv2.filter2D(img, -1, kernel)
        features.append(float(np.mean(np.abs(residual))))
        features.append(float(np.std(residual)))
        features.append(_kurtosis(residual))

    return np.array(features, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────
# Multi-Scale Noise Residuals (12-d)
# ─────────────────────────────────────────────────────────────────────
# LoG at 3 scales: (mean_abs, std, kurtosis) per scale,
# cross-scale correlation, fine/coarse energy ratio.
# Captures the "structured noise vs homogeneous speckle" distinction.

def extract_multiscale_noise(gray: np.ndarray) -> np.ndarray:
    """
    Multi-scale noise residual statistics (12-d).
    """
    features = []
    residuals = []

    for sigma in [1.0, 2.0, 4.0]:
        blurred = cv2.GaussianBlur(gray, (0, 0), sigma).astype(np.float32)
        noise = gray.astype(np.float32) - blurred
        residuals.append(noise)
        features.append(float(np.mean(np.abs(noise))))
        features.append(float(np.std(noise)))
        features.append(_kurtosis(noise))

    # Cross-scale correlation (fast dot-product Pearson)
    for i in range(len(residuals) - 1):
        features.append(_fast_corr(residuals[i], residuals[i + 1]))

    # Fine/coarse energy ratio
    e_fine = np.mean(residuals[0] ** 2)
    e_coarse = np.mean(residuals[2] ** 2)
    features.append(float(e_fine / max(e_coarse, 1e-8)))

    return np.array(features, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────
# Spectral Features — FFT (10-d)
# ─────────────────────────────────────────────────────────────────────
# Radial average of FFT magnitude spectrum.  GAN checkerboard artifacts,
# diffusion models' different HF rolloff patterns.

def extract_spectral(gray: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """
    Radial FFT magnitude spectrum features (10-d).
    """
    h, w = gray.shape
    img_f = np.float32(gray) / 255.0

    fft = np.fft.fft2(img_f)
    fft_s = np.fft.fftshift(fft)
    mag = np.log1p(np.abs(fft_s))

    cy, cx = h // 2, w // 2
    max_r = min(cy, cx)
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # Vectorized radial binning via digitize + bincount (single pass)
    bin_edges = np.linspace(0, max_r, n_bins + 1)
    r_flat = r.ravel()
    mag_flat = mag.ravel()
    bin_idx = np.digitize(r_flat, bin_edges[1:], right=False)  # 0..n_bins-1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    bin_sums = np.bincount(bin_idx, weights=mag_flat, minlength=n_bins)
    bin_counts = np.bincount(bin_idx, minlength=n_bins)
    radial_profile = np.where(bin_counts > 0, bin_sums / bin_counts, 0.0)
    radial_profile = radial_profile[:n_bins].tolist()

    features = list(radial_profile)

    # Spectral slope
    freqs = np.arange(1, n_bins + 1, dtype=np.float64)
    profile = np.array(radial_profile, dtype=np.float64)
    if profile.std() > 1e-8:
        slope = float(np.polyfit(np.log(freqs), profile, 1)[0])
    else:
        slope = 0.0
    features.append(slope)

    # HF energy ratio (top 25%)
    total_e = sum(radial_profile) + 1e-8
    hf_e = sum(radial_profile[n_bins * 3 // 4:])
    features.append(float(hf_e / total_e))

    return np.array(features, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────
# Combined extraction
# ─────────────────────────────────────────────────────────────────────

def extract_forensic_features(image_rgb: np.ndarray,
                               parsing_map: np.ndarray) -> np.ndarray:
    """
    Extract 83 forensic features from an RGB image + its parsing map.
    Returns (NUM_FORENSIC_FEATURES,) float32 array.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    skin_mask = region_mask(parsing_map, SKIN_LABELS)
    eye_mask = region_mask(parsing_map, EYE_LABELS)
    mouth_mask = region_mask(parsing_map, MOUTH_LABELS)
    nose_mask = region_mask(parsing_map, NOSE_LABELS)
    brow_mask = region_mask(parsing_map, BROW_LABELS)
    bg_mask = region_mask(parsing_map, BG_LABELS)

    l_eye_mask = region_mask(parsing_map, LEFT_EYE)
    r_eye_mask = region_mask(parsing_map, RIGHT_EYE)
    l_cheek, r_cheek = cheek_masks(parsing_map)

    h = parsing_map.shape[0]
    mid_y = h // 2
    jaw_mask = skin_mask.copy()
    jaw_mask[:mid_y, :] = 0
    mid_x = parsing_map.shape[1] // 2
    l_jaw = jaw_mask.copy()
    l_jaw[:, mid_x:] = 0
    r_jaw = jaw_mask.copy()
    r_jaw[:, :mid_x] = 0

    l_mouth = mouth_mask.copy()
    l_mouth[:, mid_x:] = 0
    r_mouth = mouth_mask.copy()
    r_mouth[:, :mid_x] = 0

    feats = np.zeros(NUM_FORENSIC_FEATURES, dtype=np.float32)

    # Boundary gradients (6)
    feats[0] = boundary_gradient(gray, skin_mask, bg_mask)
    feats[1] = boundary_gradient(gray, eye_mask, skin_mask)
    feats[2] = boundary_gradient(gray, mouth_mask, skin_mask)
    feats[3] = boundary_gradient(gray, nose_mask, skin_mask)
    feats[4] = boundary_gradient(gray, brow_mask, skin_mask)
    feats[5] = np.mean(feats[:5])

    # Regional blur (6) — cached Laplacian
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    blur_skin = laplacian_variance(gray, skin_mask, lap)
    blur_eye = laplacian_variance(gray, eye_mask, lap)
    blur_mouth = laplacian_variance(gray, mouth_mask, lap)
    blur_nose = laplacian_variance(gray, nose_mask, lap)
    feats[6] = blur_skin
    feats[7] = blur_eye
    feats[8] = blur_mouth
    feats[9] = blur_nose
    feats[10] = safe_ratio(blur_eye, blur_skin)
    feats[11] = safe_ratio(blur_mouth, blur_skin)

    # Left-right symmetry (4)
    feats[12] = left_right_symmetry(gray, l_eye_mask, r_eye_mask)
    feats[13] = left_right_symmetry(gray, l_mouth, r_mouth)
    feats[14] = left_right_symmetry(gray, l_cheek, r_cheek)
    feats[15] = left_right_symmetry(gray, l_jaw, r_jaw)

    # Color consistency (4)
    feats[16] = color_histogram_distance(lab, eye_mask, skin_mask)
    feats[17] = color_histogram_distance(lab, mouth_mask, skin_mask)
    feats[18] = color_histogram_distance(lab, nose_mask, skin_mask)
    feats[19] = color_histogram_distance(lab, l_cheek, r_cheek)

    # Frequency anomaly (6)
    dct_skin = dct_hf_energy(gray, skin_mask)
    dct_eye = dct_hf_energy(gray, eye_mask)
    dct_mouth = dct_hf_energy(gray, mouth_mask)
    dct_nose = dct_hf_energy(gray, nose_mask)
    feats[20] = dct_skin
    feats[21] = dct_eye
    feats[22] = dct_mouth
    feats[23] = dct_nose
    feats[24] = safe_ratio(dct_eye, dct_skin)
    feats[25] = safe_ratio(dct_mouth, dct_skin)

    # Quality metrics (4) — filled by ForensicPrecomputer if models available
    feats[26] = 0.0  # ff_quality_antispoof
    feats[27] = 0.0  # ff_quality_det_score
    feats[28] = 0.0  # ff_quality_blendshape_sym
    feats[29] = 0.0  # ff_quality_landmark_jitter

    # ── New pixel-level forensic features (30-82) ───────────────────
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # PPNC — Paired Patch Noise Consistency (30-37)
    feats[30:38] = extract_ppnc(gray)

    # CCNC — Cross-Channel Noise Consistency (38-45)
    feats[38:46] = extract_ccnc(image_bgr)

    # SRM — Spatial Rich Model filter bank (46-60)
    feats[46:61] = extract_srm(gray)

    # Multi-Scale Noise Residuals (61-72)
    feats[61:73] = extract_multiscale_noise(gray)

    # Spectral Features — FFT (73-82)
    feats[73:83] = extract_spectral(gray)

    return feats

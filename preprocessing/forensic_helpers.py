"""
forensic_helpers.py
===================
Pure CV functions for extracting pixel-level forensic features.
No GPU models or state — these run on CPU with numpy/cv2/scipy.
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
]
assert len(FEATURE_NAMES) == 30


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


def laplacian_variance(gray: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() < 50:
        return 0.0
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


def extract_forensic_features(image_rgb: np.ndarray,
                               parsing_map: np.ndarray) -> np.ndarray:
    """
    Extract 30 forensic features from an RGB image + its parsing map.
    Returns (30,) float32 array.
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

    feats = np.zeros(30, dtype=np.float32)

    # Boundary gradients (6)
    feats[0] = boundary_gradient(gray, skin_mask, bg_mask)
    feats[1] = boundary_gradient(gray, eye_mask, skin_mask)
    feats[2] = boundary_gradient(gray, mouth_mask, skin_mask)
    feats[3] = boundary_gradient(gray, nose_mask, skin_mask)
    feats[4] = boundary_gradient(gray, brow_mask, skin_mask)
    feats[5] = np.mean(feats[:5])

    # Regional blur (6)
    blur_skin = laplacian_variance(gray, skin_mask)
    blur_eye = laplacian_variance(gray, eye_mask)
    blur_mouth = laplacian_variance(gray, mouth_mask)
    blur_nose = laplacian_variance(gray, nose_mask)
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

    return feats

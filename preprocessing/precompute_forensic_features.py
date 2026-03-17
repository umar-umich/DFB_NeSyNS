#!/usr/bin/env python3
"""
precompute_forensic_features.py
================================
Precompute Tier 2 pixel-level forensic features for all frames.

Uses SegFormer face parsing to extract 30 forensic features per frame:
  - Boundary gradients (6): Sobel gradient at region boundaries
  - Regional blur (6): Laplacian variance per face region
  - L/R symmetry (4): Intensity difference between left/right regions
  - Color consistency (4): Histogram distance between adjacent regions
  - Frequency anomaly (6): DCT high-frequency energy per region
  - Quality metrics (4): Antispoof, detection quality, blendshape symmetry

Output structure (mirrors frames/ directory):
  .../original_sequences/youtube/c23/forensic_features/929.pt
  .../manipulated_sequences/Deepfakes/c23/forensic_features/802_885.pt

Each .pt file:
  {'features': Tensor(n_frames, 30), 'frame_paths': List[str],
   'feature_names': List[str]}

Usage:
  python preprocessing/precompute_forensic_features.py \
      --detector_path training/config/detector/nesy_defake.yaml \
      --output_dir forensic_features
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from scipy.fft import dctn
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))


# ---------------------------------------------------------------------------
# Feature names (must match forensic_features.py)
# ---------------------------------------------------------------------------
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

# Left-side and right-side labels for symmetry
LEFT_EYE = {LABEL_L_EYE}
RIGHT_EYE = {LABEL_R_EYE}
LEFT_BROW = {LABEL_L_BROW}
RIGHT_BROW = {LABEL_R_BROW}


# ---------------------------------------------------------------------------
# Face parser (SegFormer) — lazy global initialization
# ---------------------------------------------------------------------------
_face_parser = None
_image_processor = None


def _init_face_parser(device='cuda:0'):
    """Lazy-load SegFormer face parser."""
    global _face_parser, _image_processor
    if _face_parser is not None:
        return

    from transformers import (SegformerImageProcessor,
                              SegformerForSemanticSegmentation)

    print("[ForensicFeatures] Loading SegFormer face parser...")
    # Try local cache first, fall back to HuggingFace hub
    model_name = "jonathandinu/face-parsing"
    _image_processor = SegformerImageProcessor.from_pretrained(model_name)
    _face_parser = SegformerForSemanticSegmentation.from_pretrained(
        model_name).to(device).eval()
    print(f"[ForensicFeatures] SegFormer loaded on {device}")


def get_parsing_map(image_rgb: np.ndarray, device='cuda:0') -> np.ndarray:
    """
    Get face parsing map from a single RGB image.

    Args:
        image_rgb: (H, W, 3) uint8 RGB image
    Returns:
        parsing_map: (H, W) int array with SegFormer labels
    """
    _init_face_parser(device)
    pil_img = Image.fromarray(image_rgb)
    inputs = _image_processor(images=pil_img, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = _face_parser(**inputs)
    logits = outputs.logits  # (1, n_classes, h, w)
    upsampled = F.interpolate(
        logits, size=image_rgb.shape[:2], mode='bilinear', align_corners=False)
    parsing_map = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()
    return parsing_map


def get_parsing_maps_batch(images_rgb: list, device='cuda:0') -> list:
    """
    Batch face parsing for multiple RGB images.

    Args:
        images_rgb: list of (H, W, 3) uint8 RGB images
    Returns:
        list of (H, W) int arrays with SegFormer labels
    """
    _init_face_parser(device)
    pil_imgs = [Image.fromarray(img) for img in images_rgb]
    inputs = _image_processor(images=pil_imgs, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = _face_parser(**inputs)
    logits = outputs.logits  # (B, n_classes, h', w')

    parsing_maps = []
    for i, img in enumerate(images_rgb):
        upsampled = F.interpolate(
            logits[i:i+1], size=img.shape[:2],
            mode='bilinear', align_corners=False)
        parsing_maps.append(upsampled.argmax(dim=1).squeeze(0).cpu().numpy())
    return parsing_maps


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _region_mask(parsing_map: np.ndarray, labels: set) -> np.ndarray:
    """Binary mask for a set of parsing labels."""
    mask = np.zeros_like(parsing_map, dtype=np.uint8)
    for l in labels:
        mask |= (parsing_map == l).astype(np.uint8)
    return mask


def _boundary_gradient(gray: np.ndarray, mask_a: np.ndarray,
                       mask_b: np.ndarray) -> float:
    """Mean Sobel gradient magnitude at the boundary between two regions."""
    # Dilate both masks slightly and find intersection (boundary strip)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_a = cv2.dilate(mask_a, kernel, iterations=1)
    dilated_b = cv2.dilate(mask_b, kernel, iterations=1)
    boundary = dilated_a & dilated_b

    if boundary.sum() < 10:
        return 0.0

    # Sobel gradient on grayscale image
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    return float(grad_mag[boundary > 0].mean()) / 255.0


def _laplacian_variance(gray: np.ndarray, mask: np.ndarray) -> float:
    """Laplacian variance (blur metric) for a masked region."""
    if mask.sum() < 50:
        return 0.0
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap[mask > 0].var()) / (255.0 ** 2)


def _region_mean_intensity(gray: np.ndarray, mask: np.ndarray) -> float:
    """Mean intensity in a masked region."""
    if mask.sum() < 10:
        return 0.0
    return float(gray[mask > 0].mean()) / 255.0


def _safe_ratio(a: float, b: float) -> float:
    """Safe ratio a/b, returns 0 if b is near zero."""
    return a / (b + 1e-10)


def _color_histogram_distance(lab_img: np.ndarray, mask_a: np.ndarray,
                               mask_b: np.ndarray) -> float:
    """Chi-squared distance between LAB color histograms of two regions."""
    if mask_a.sum() < 50 or mask_b.sum() < 50:
        return 0.0

    hist_a = []
    hist_b = []
    for c in range(3):
        channel = lab_img[:, :, c]
        ha, _ = np.histogram(channel[mask_a > 0], bins=32, range=(0, 256))
        hb, _ = np.histogram(channel[mask_b > 0], bins=32, range=(0, 256))
        hist_a.append(ha.astype(np.float64))
        hist_b.append(hb.astype(np.float64))

    hist_a = np.concatenate(hist_a)
    hist_b = np.concatenate(hist_b)

    # Normalize
    hist_a = hist_a / (hist_a.sum() + 1e-10)
    hist_b = hist_b / (hist_b.sum() + 1e-10)

    # Chi-squared distance
    chi2 = np.sum((hist_a - hist_b) ** 2 / (hist_a + hist_b + 1e-10))
    # Normalize to [0, 1] range (chi2 max is 2 for normalized histograms)
    return min(float(chi2) / 2.0, 1.0)


def _dct_hf_energy(gray: np.ndarray, mask: np.ndarray,
                   hf_threshold: float = 0.5) -> float:
    """
    Ratio of high-frequency DCT energy in a masked region.

    Crops a bounding box around the mask, computes 2D DCT, and measures
    the fraction of energy in the upper-frequency quadrants.
    """
    if mask.sum() < 100:
        return 0.0

    # Bounding box
    ys, xs = np.nonzero(mask)
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1

    # Ensure minimum size for DCT
    if (y2 - y1) < 8 or (x2 - x1) < 8:
        return 0.0

    crop = gray[y1:y2, x1:x2].astype(np.float64)
    crop_mask = mask[y1:y2, x1:x2]
    crop = crop * crop_mask  # Zero out non-region pixels

    # 2D DCT
    dct_coeff = dctn(crop, type=2, norm='ortho')
    energy = dct_coeff ** 2

    h, w = energy.shape
    # High-frequency: bottom-right quadrant beyond threshold
    hf_y = int(h * hf_threshold)
    hf_x = int(w * hf_threshold)

    total_energy = energy.sum()
    if total_energy < 1e-10:
        return 0.0

    hf_energy = energy[hf_y:, :].sum() + energy[:hf_y, hf_x:].sum()
    return min(float(hf_energy / total_energy), 1.0)


def _left_right_symmetry(gray: np.ndarray, left_mask: np.ndarray,
                          right_mask: np.ndarray) -> float:
    """
    Measure L/R symmetry by comparing mean intensity of left and right regions.
    Returns absolute difference in [0, 1].
    """
    left_mean = _region_mean_intensity(gray, left_mask)
    right_mean = _region_mean_intensity(gray, right_mask)
    return abs(left_mean - right_mean)


def _cheek_masks(parsing_map: np.ndarray):
    """
    Approximate left/right cheek masks from the skin region.
    Split the skin region at the horizontal center.
    """
    skin_mask = _region_mask(parsing_map, SKIN_LABELS)
    h, w = parsing_map.shape
    mid_x = w // 2

    left_cheek = skin_mask.copy()
    left_cheek[:, mid_x:] = 0
    right_cheek = skin_mask.copy()
    right_cheek[:, :mid_x] = 0

    return left_cheek, right_cheek


# ---------------------------------------------------------------------------
# Main per-frame feature extraction
# ---------------------------------------------------------------------------

def extract_forensic_features(image_rgb: np.ndarray,
                               parsing_map: np.ndarray) -> np.ndarray:
    """
    Extract 30 forensic features from an RGB image + its parsing map.

    Args:
        image_rgb: (H, W, 3) uint8 RGB image
        parsing_map: (H, W) int array from SegFormer

    Returns:
        features: (30,) float32 array, values in [0, 1]
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    # Build region masks
    skin_mask = _region_mask(parsing_map, SKIN_LABELS)
    eye_mask = _region_mask(parsing_map, EYE_LABELS)
    mouth_mask = _region_mask(parsing_map, MOUTH_LABELS)
    nose_mask = _region_mask(parsing_map, NOSE_LABELS)
    brow_mask = _region_mask(parsing_map, BROW_LABELS)
    bg_mask = _region_mask(parsing_map, BG_LABELS)

    l_eye_mask = _region_mask(parsing_map, LEFT_EYE)
    r_eye_mask = _region_mask(parsing_map, RIGHT_EYE)
    l_cheek, r_cheek = _cheek_masks(parsing_map)

    # Approximate jawline symmetry from skin bottom half
    h = parsing_map.shape[0]
    mid_y = h // 2
    jaw_mask = skin_mask.copy()
    jaw_mask[:mid_y, :] = 0
    mid_x = parsing_map.shape[1] // 2
    l_jaw = jaw_mask.copy()
    l_jaw[:, mid_x:] = 0
    r_jaw = jaw_mask.copy()
    r_jaw[:, :mid_x] = 0

    # Approximate left/right mouth
    l_mouth = mouth_mask.copy()
    l_mouth[:, mid_x:] = 0
    r_mouth = mouth_mask.copy()
    r_mouth[:, :mid_x] = 0

    feats = np.zeros(30, dtype=np.float32)

    # -- Boundary gradients (6) --
    feats[0] = _boundary_gradient(gray, skin_mask, bg_mask)
    feats[1] = _boundary_gradient(gray, eye_mask, skin_mask)
    feats[2] = _boundary_gradient(gray, mouth_mask, skin_mask)
    feats[3] = _boundary_gradient(gray, nose_mask, skin_mask)
    feats[4] = _boundary_gradient(gray, brow_mask, skin_mask)
    feats[5] = np.mean(feats[:5])  # mean boundary gradient

    # -- Regional blur (6) --
    blur_skin = _laplacian_variance(gray, skin_mask)
    blur_eye = _laplacian_variance(gray, eye_mask)
    blur_mouth = _laplacian_variance(gray, mouth_mask)
    blur_nose = _laplacian_variance(gray, nose_mask)
    feats[6] = blur_skin
    feats[7] = blur_eye
    feats[8] = blur_mouth
    feats[9] = blur_nose
    feats[10] = _safe_ratio(blur_eye, blur_skin)
    feats[11] = _safe_ratio(blur_mouth, blur_skin)

    # -- Left-right symmetry (4) --
    feats[12] = _left_right_symmetry(gray, l_eye_mask, r_eye_mask)
    feats[13] = _left_right_symmetry(gray, l_mouth, r_mouth)
    feats[14] = _left_right_symmetry(gray, l_cheek, r_cheek)
    feats[15] = _left_right_symmetry(gray, l_jaw, r_jaw)

    # -- Color consistency (4) --
    feats[16] = _color_histogram_distance(lab, eye_mask, skin_mask)
    feats[17] = _color_histogram_distance(lab, mouth_mask, skin_mask)
    feats[18] = _color_histogram_distance(lab, nose_mask, skin_mask)
    feats[19] = _color_histogram_distance(lab, l_cheek, r_cheek)

    # -- Frequency anomaly (6) --
    dct_skin = _dct_hf_energy(gray, skin_mask)
    dct_eye = _dct_hf_energy(gray, eye_mask)
    dct_mouth = _dct_hf_energy(gray, mouth_mask)
    dct_nose = _dct_hf_energy(gray, nose_mask)
    feats[20] = dct_skin
    feats[21] = dct_eye
    feats[22] = dct_mouth
    feats[23] = dct_nose
    feats[24] = _safe_ratio(dct_eye, dct_skin)
    feats[25] = _safe_ratio(dct_mouth, dct_skin)

    # -- Quality metrics (4) --
    # These require InsightFace/MediaPipe — set to 0 if not available
    # Will be filled by the optional quality extraction pass
    feats[26] = 0.0  # ff_quality_antispoof (filled below if available)
    feats[27] = 0.0  # ff_quality_det_score
    feats[28] = 0.0  # ff_quality_blendshape_sym
    feats[29] = 0.0  # ff_quality_landmark_jitter (temporal, always 0 for single frame)

    return feats


# ---------------------------------------------------------------------------
# Optional: InsightFace quality features
# ---------------------------------------------------------------------------
_insightface_app = None


def _init_insightface():
    """Lazy-load InsightFace for antispoof and quality scores."""
    global _insightface_app
    if _insightface_app is not None:
        return True
    try:
        from insightface.app import FaceAnalysis
        _insightface_app = FaceAnalysis(
            name='buffalo_l', providers=['CUDAExecutionProvider'])
        _insightface_app.prepare(ctx_id=0, det_size=(224, 224))
        print("[ForensicFeatures] InsightFace loaded")
        return True
    except Exception as e:
        print(f"[ForensicFeatures] InsightFace not available: {e}")
        return False


def _extract_insightface_quality(image_bgr: np.ndarray) -> tuple:
    """Extract antispoof and detection quality from InsightFace."""
    if _insightface_app is None:
        return 0.0, 0.0
    try:
        faces = _insightface_app.get(image_bgr)
        if len(faces) == 0:
            return 0.0, 0.0
        face = faces[0]
        antispoof = float(getattr(face, 'antispoof', 0.0) or 0.0)
        det_score = float(face.det_score)
        return antispoof, det_score
    except Exception:
        return 0.0, 0.0


# ---------------------------------------------------------------------------
# Optional: MediaPipe blendshape symmetry
# ---------------------------------------------------------------------------
_mediapipe_detector = None


def _init_mediapipe():
    """Lazy-load MediaPipe face landmark detector."""
    global _mediapipe_detector
    if _mediapipe_detector is not None:
        return True
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision as mp_vision

        # Find the task model file
        model_paths = [
            os.path.expanduser(
                '~/.mediapipe/models/face_landmarker.task'),
            '/data/umar/models/face_landmarker.task',
        ]
        model_path = None
        for p in model_paths:
            if os.path.exists(p):
                model_path = p
                break

        if model_path is None:
            print("[ForensicFeatures] MediaPipe model not found, skipping")
            return False

        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            output_face_blendshapes=True,
            num_faces=1,
        )
        _mediapipe_detector = mp_vision.FaceLandmarker.create_from_options(options)
        print("[ForensicFeatures] MediaPipe loaded")
        return True
    except Exception as e:
        print(f"[ForensicFeatures] MediaPipe not available: {e}")
        return False


def _extract_blendshape_symmetry(image_rgb: np.ndarray) -> float:
    """
    Compute bilateral blendshape symmetry deviation.
    Returns mean absolute difference between left/right blendshape pairs.
    """
    if _mediapipe_detector is None:
        return 0.0
    try:
        import mediapipe as mp
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = _mediapipe_detector.detect(mp_image)

        if not result.face_blendshapes or len(result.face_blendshapes) == 0:
            return 0.0

        bs = {b.category_name: b.score
              for b in result.face_blendshapes[0]}

        # Bilateral pairs (left vs right)
        bilateral_pairs = [
            ('browDownLeft', 'browDownRight'),
            ('browInnerUp', 'browInnerUp'),  # same, skip
            ('browOuterUpLeft', 'browOuterUpRight'),
            ('cheekSquintLeft', 'cheekSquintRight'),
            ('eyeBlinkLeft', 'eyeBlinkRight'),
            ('eyeLookDownLeft', 'eyeLookDownRight'),
            ('eyeLookInLeft', 'eyeLookInRight'),
            ('eyeLookOutLeft', 'eyeLookOutRight'),
            ('eyeLookUpLeft', 'eyeLookUpRight'),
            ('eyeSquintLeft', 'eyeSquintRight'),
            ('eyeWideLeft', 'eyeWideRight'),
            ('mouthSmileLeft', 'mouthSmileRight'),
            ('mouthFrownLeft', 'mouthFrownRight'),
            ('mouthDimpleLeft', 'mouthDimpleRight'),
            ('mouthStretchLeft', 'mouthStretchRight'),
            ('noseSneerLeft', 'noseSneerRight'),
        ]

        diffs = []
        for left_name, right_name in bilateral_pairs:
            if left_name == right_name:
                continue
            left_val = bs.get(left_name, 0.0)
            right_val = bs.get(right_name, 0.0)
            diffs.append(abs(left_val - right_val))

        return float(np.mean(diffs)) if diffs else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Video-level processing
# ---------------------------------------------------------------------------

def process_video_frames(frame_paths: list, device: str = 'cuda:0',
                         use_insightface: bool = True,
                         use_mediapipe: bool = True,
                         batch_size: int = 32) -> dict:
    """
    Process all frames of a video and return forensic features.

    SegFormer face parsing is batched for GPU efficiency; CPU-side feature
    extraction (Sobel, DCT, etc.) runs per-frame after each batch.

    Returns:
        {'features': Tensor(n_frames, 30), 'frame_paths': List[str],
         'feature_names': List[str]}
    """
    # Load all images first
    loaded = []  # list of (image_rgb, image_bgr) or None for failures
    for frame_path in frame_paths:
        try:
            image_bgr = cv2.imread(frame_path)
            if image_bgr is None:
                loaded.append(None)
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            loaded.append((image_rgb, image_bgr))
        except Exception as e:
            print(f"  [WARNING] Failed to load {frame_path}: {e}")
            loaded.append(None)

    all_features = [None] * len(frame_paths)

    # Batch SegFormer inference
    for batch_start in range(0, len(loaded), batch_size):
        batch_end = min(batch_start + batch_size, len(loaded))

        # Collect valid images for this batch
        batch_indices = []
        batch_images_rgb = []
        for i in range(batch_start, batch_end):
            if loaded[i] is not None:
                batch_indices.append(i)
                batch_images_rgb.append(loaded[i][0])

        if not batch_images_rgb:
            for i in range(batch_start, batch_end):
                all_features[i] = np.zeros(30, dtype=np.float32)
            continue

        # Batched face parsing on GPU
        try:
            parsing_maps = get_parsing_maps_batch(batch_images_rgb, device=device)
        except Exception as e:
            print(f"  [WARNING] Batch parsing failed: {e}, falling back to per-frame")
            parsing_maps = []
            for img_rgb in batch_images_rgb:
                try:
                    parsing_maps.append(get_parsing_map(img_rgb, device=device))
                except Exception:
                    parsing_maps.append(np.zeros(img_rgb.shape[:2], dtype=np.int64))

        # CPU-side feature extraction per frame
        for j, idx in enumerate(batch_indices):
            image_rgb, image_bgr = loaded[idx]
            try:
                feats = extract_forensic_features(image_rgb, parsing_maps[j])

                if use_insightface and _insightface_app is not None:
                    antispoof, det_score = _extract_insightface_quality(image_bgr)
                    feats[26] = antispoof
                    feats[27] = det_score

                if use_mediapipe and _mediapipe_detector is not None:
                    feats[28] = _extract_blendshape_symmetry(image_rgb)

                all_features[idx] = feats
            except Exception as e:
                print(f"  [WARNING] Failed on {frame_paths[idx]}: {e}")
                all_features[idx] = np.zeros(30, dtype=np.float32)

        # Free GPU memory between batches
        torch.cuda.empty_cache()

    # Fill any remaining Nones (failed loads)
    for i in range(len(all_features)):
        if all_features[i] is None:
            all_features[i] = np.zeros(30, dtype=np.float32)

    features_tensor = torch.from_numpy(np.stack(all_features))
    return {
        'features': features_tensor,
        'frame_paths': [os.path.basename(p) for p in frame_paths],
        'feature_names': FEATURE_NAMES,
    }


# ---------------------------------------------------------------------------
# Config loading and dataset discovery (matches precompute_semantic_features.py)
# ---------------------------------------------------------------------------

def load_config(detector_path):
    """Load and merge detector + train configs."""
    with open(detector_path) as f:
        config = yaml.safe_load(f)
    train_cfg_path = os.path.join(
        os.path.dirname(detector_path), '..', 'config', 'train_config.yaml')
    if not os.path.exists(train_cfg_path):
        train_cfg_path = './training/config/train_config.yaml'
    if os.path.exists(train_cfg_path):
        with open(train_cfg_path) as f:
            train_config = yaml.safe_load(f)
        if 'label_dict' in config:
            train_config['label_dict'] = config['label_dict']
        config.update(train_config)
    return config


def collect_videos_from_json(config):
    """
    Parse dataset JSONs and return a dict:
      video_id -> {'frames': [path1, path2, ...], 'output_dir': str,
                   'video_id': str}
    """
    json_folder = config['dataset_json_folder']
    compression = config.get('compression', 'c23')

    all_datasets = set()
    for key in ('train_dataset', 'test_dataset'):
        ds = config.get(key, [])
        if isinstance(ds, str):
            ds = [ds]
        all_datasets.update(ds)

    videos = {}
    total_frames = 0

    for dataset_name in sorted(all_datasets):
        json_path = os.path.join(json_folder, f'{dataset_name}.json')
        if not os.path.exists(json_path):
            print(f"  WARNING: JSON not found: {json_path}, skipping")
            continue

        with open(json_path) as f:
            data = json.load(f)

        for top_key, top_val in data.items():
            for label_key, label_val in top_val.items():
                for mode_key, mode_val in label_val.items():
                    first_val = next(iter(mode_val.values()), {})
                    if isinstance(first_val, dict) and 'frames' in first_val:
                        video_items = mode_val.items()
                    elif compression in mode_val:
                        video_items = mode_val[compression].items()
                    else:
                        continue

                    for video_id, video_data in video_items:
                        frames = video_data.get('frames', [])
                        if not frames:
                            continue

                        sample_frame = frames[0]
                        sep = '/' if '/' in sample_frame else '\\'
                        parts = sample_frame.split(sep)
                        frames_dir_idx = None
                        for pi, part in enumerate(parts):
                            if part == 'frames' or part.startswith('frames_aug_'):
                                frames_dir_idx = pi
                                break
                        if frames_dir_idx is not None:
                            base_dir = sep.join(parts[:frames_dir_idx])
                        else:
                            base_dir = os.path.dirname(
                                os.path.dirname(sample_frame))

                        vid_key = f"{base_dir}/{video_id}"
                        if vid_key not in videos:
                            videos[vid_key] = {
                                'frames': sorted(frames),
                                'output_dir': base_dir,
                                'video_id': video_id,
                            }
                            total_frames += len(frames)
                        else:
                            existing = set(videos[vid_key]['frames'])
                            for fp in frames:
                                if fp not in existing:
                                    videos[vid_key]['frames'].append(fp)
                                    total_frames += 1
                            videos[vid_key]['frames'].sort()

    print(f"\nCollected {len(videos)} videos, {total_frames} total frames")
    return videos


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Precompute Tier 2 forensic features for all frames')
    parser.add_argument('--detector_path', type=str, required=True,
                        help='Path to detector YAML config')
    parser.add_argument('--output_dir', type=str, default='forensic_features',
                        help='Output subdirectory name (placed alongside frames/)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run inference on')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip videos with existing .pt files')
    parser.add_argument('--compression', type=str, default=None,
                        help='Override compression level (e.g., c23)')
    parser.add_argument('--no_insightface', action='store_true',
                        help='Skip InsightFace quality extraction')
    parser.add_argument('--no_mediapipe', action='store_true',
                        help='Skip MediaPipe blendshape extraction')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for SegFormer face parsing (GPU)')
    parser.add_argument('--max_frames', type=int, default=0,
                        help='Max frames per video (0 = all)')
    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(args.detector_path)
    if args.compression:
        config['compression'] = args.compression

    print("=" * 60)
    print("Forensic Feature Precomputation (Tier 2, 30-d)")
    print("=" * 60)

    # Initialize models
    _init_face_parser(args.device)

    use_insightface = not args.no_insightface
    use_mediapipe = not args.no_mediapipe

    if use_insightface:
        _init_insightface()
    if use_mediapipe:
        _init_mediapipe()

    # Collect videos from dataset JSONs
    videos = collect_videos_from_json(config)

    if not videos:
        print("No videos found. Check your config and dataset JSONs.")
        return

    print(f"\n  Output subdir: {args.output_dir}")
    print(f"  Batch size: {args.batch_size}")

    # Process
    n_skipped = 0
    n_processed = 0
    n_failed = 0

    for vid_key, vid_info in tqdm(videos.items(), desc="Videos"):
        output_base = vid_info['output_dir']
        video_id = vid_info['video_id']

        output_dir = os.path.join(output_base, args.output_dir)
        output_path = os.path.join(output_dir, f"{video_id}.pt")

        if args.skip_existing and os.path.exists(output_path):
            n_skipped += 1
            continue

        frame_paths = vid_info['frames']
        if args.max_frames > 0:
            frame_paths = frame_paths[:args.max_frames]

        try:
            result = process_video_frames(
                frame_paths,
                device=args.device,
                use_insightface=use_insightface,
                use_mediapipe=use_mediapipe,
                batch_size=args.batch_size,
            )

            os.makedirs(output_dir, exist_ok=True)
            torch.save(result, output_path)
            n_processed += 1

        except Exception as e:
            print(f"  Error processing {vid_key}: {e}")
            n_failed += 1
            torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print(f"Precomputation complete!")
    print(f"  Processed: {n_processed}")
    print(f"  Skipped:   {n_skipped}")
    print(f"  Failed:    {n_failed}")
    print(f"  Output:    */{args.output_dir}/<video_id>.pt")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()

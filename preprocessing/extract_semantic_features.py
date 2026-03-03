"""
preprocessing/extract_semantic_features.py
==========================================
Extract identity-invariant semantic facial attributes from preprocessed face crops.

DESIGN PHILOSOPHY
-----------------
Every feature must answer a human-interpretable semantic question about the face,
NOT encode geometric coordinates or identity-specific information.

Identity features (ArcFace embeddings, identity drift) are EXCLUDED because they
make the causal graph identity-dependent — testing on unseen identities causes
every identity-connected edge to fire as a violation (false positives).

WHAT WE EXTRACT (per frame)
----------------------------
┌─────────────────────────────────────────────────────────────────────────┐
│ Source         │ Features                                  │ Count     │
├─────────────────────────────────────────────────────────────────────────┤
│ DeepFace       │ emotion (7), age (1), gender (2), race (6)│ 16        │
│ InsightFace    │ head pose (3), quality (1), antispoof (1) │  5        │
│ MediaPipe      │ 52 facial blendshape coefficients         │ 52        │
│                │ (ARKit-compatible muscle activations)      │           │
├─────────────────────────────────────────────────────────────────────────┤
│ Per-frame total│                                           │ 73        │
└─────────────────────────────────────────────────────────────────────────┘

WHY MEDIAPIPE BLENDSHAPES
--------------------------
The 52 blendshapes are continuous [0,1] activations of specific facial muscles.
They're deeply semantic — "how open is the jaw?", "how much is the left eyebrow
raised?", "is the mouth smiling on the right side?"

Key advantages for causal discovery:
  1. BILATERAL SYMMETRY: 20 left/right pairs (browDownLeft/Right, mouthSmileLeft/Right).
     Real faces have strong L↔R causal coupling (same nerve, same muscle group).
     Face-swaps break this at the blending boundary → causal violation.

  2. EXPRESSION COHERENCE: Real smiles activate mouthSmile + cheekSquint together
     (Duchenne pattern). Fakes may get mouth right but miss cheek muscles.
     The causal graph learns these multi-variable constraints.

  3. COMPLEMENTS DEEPFACE: DeepFace emotion gives 7 coarse categories.
     Blendshapes give the fine-grained muscle activations that PRODUCE emotions.
     Graph can model: blendshapes → emotion, violations = forgery signature.

VIDEO-LEVEL TEMPORAL SUMMARIES
-------------------------------
For each of the 73 attributes, compute:
  - mean: average value across frames
  - std: variability (fakes have inconsistent attributes)
  - delta: mean |consecutive-frame difference| (temporal smoothness)

Total causal variables: 73 × 3 = 219 per video

NOTE: 219 may be too many for DAGMA-DCE. We extract everything and select
subsets during ablation. Options: use all 219, PCA to ~60, or curated ~80.

OUTPUT FORMAT
-------------
  {preprocessed_root}/{dataset}/{sub_dataset}/semantic_features/{video_name}.npz

Each .npz contains:
  - 'per_frame':       (T, 73) float32 — raw per-frame attributes
  - 'summary':         (219,) float32 — video-level temporal summaries
  - 'feature_names':   (73,) — per-frame feature names
  - 'summary_names':   (219,) — summary feature names
  - 'extraction_info': dict with metadata

PREREQUISITES
-------------
  pip install deepface insightface onnxruntime-gpu mediapipe

  MediaPipe blendshapes require a model file:
    wget -O face_landmarker.task \\
      "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

  Place the model file in the same directory as this script, or pass
  --mediapipe_model_path to specify its location.

USAGE
-----
  python extract_semantic_features.py                              # config.yaml defaults
  python extract_semantic_features.py --dataset_name FaceForensics++ --comp c23
  python extract_semantic_features.py --skip_existing              # resume interrupted run
  python extract_semantic_features.py --deepface_only              # fastest, 16 features
  python extract_semantic_features.py --no_mediapipe               # skip blendshapes
"""

import os
import sys
import time
import logging
import datetime
import argparse
import glob
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm

# ── NumPy 2.0 compatibility for InsightFace ──
# InsightFace uses np.sctypes which was removed in NumPy 2.0.
# This shim restores it so InsightFace loads without downgrading NumPy.
if not hasattr(np, 'sctypes'):
    np.sctypes = {
        'int':       [np.int8, np.int16, np.int32, np.int64],
        'uint':      [np.uint8, np.uint16, np.uint32, np.uint64],
        'float':     [np.float16, np.float32, np.float64],
        'complex':   [np.complex64, np.complex128],
        'others':    [bool, object, bytes, str, np.void],
    }
# Also patch np.bool (another common NumPy 2.0 removal)
if not hasattr(np, 'bool'):
    np.bool = np.bool_
if not hasattr(np, 'int'):
    np.int = np.int_
if not hasattr(np, 'float'):
    np.float = np.float64
if not hasattr(np, 'complex'):
    np.complex = np.complex128
if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'str'):
    np.str = np.str_

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE NAME CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

DEEPFACE_FEATURE_NAMES = [
    # Emotion probabilities (7) — "what expression is this face showing?"
    'emo_angry', 'emo_disgust', 'emo_fear', 'emo_happy',
    'emo_sad', 'emo_surprise', 'emo_neutral',
    # Age (1) — "how old does this person appear?"
    'age',
    # Gender probabilities (2) — "perceived gender presentation?"
    'gender_man', 'gender_woman',
    # Race probabilities (6) — "perceived ethnicity distribution?"
    'race_asian', 'race_indian', 'race_black',
    'race_white', 'race_middle_eastern', 'race_latino',
]  # 16 total

INSIGHTFACE_FEATURE_NAMES = [
    # Head pose (3) — "where is the person facing?"
    'pose_yaw', 'pose_pitch', 'pose_roll',
    # Face quality (1) — "how clearly visible is the face?"
    'face_quality',
    # Anti-spoof (1) — "does this look like a real face?"
    'antispoof_score',
]  # 5 total

# MediaPipe 52 blendshape coefficients (ARKit-compatible)
# Each is [0, 1] intensity of a specific facial muscle activation.
# _neutral is index 0 in MediaPipe output but we include it as it measures
# "how neutral/relaxed is the face" — genuinely semantic.
MEDIAPIPE_BLENDSHAPE_NAMES = [
    'bs_neutral',
    # Brow (5) — "are the eyebrows raised/lowered?"
    'bs_browDownLeft', 'bs_browDownRight', 'bs_browInnerUp',
    'bs_browOuterUpLeft', 'bs_browOuterUpRight',
    # Cheek (3) — "are the cheeks puffed/squinted?"
    'bs_cheekPuff', 'bs_cheekSquintLeft', 'bs_cheekSquintRight',
    # Eye blink (2) — "are the eyes open/closed?"
    'bs_eyeBlinkLeft', 'bs_eyeBlinkRight',
    # Eye gaze direction (8) — "where are the eyes looking?"
    'bs_eyeLookDownLeft', 'bs_eyeLookDownRight',
    'bs_eyeLookInLeft', 'bs_eyeLookInRight',
    'bs_eyeLookOutLeft', 'bs_eyeLookOutRight',
    'bs_eyeLookUpLeft', 'bs_eyeLookUpRight',
    # Eye squint/wide (4) — "are the eyes squinting or wide?"
    'bs_eyeSquintLeft', 'bs_eyeSquintRight',
    'bs_eyeWideLeft', 'bs_eyeWideRight',
    # Jaw (4) — "is the jaw open/shifted?"
    'bs_jawForward', 'bs_jawLeft', 'bs_jawOpen', 'bs_jawRight',
    # Mouth shape (24) — "what is the mouth doing?"
    'bs_mouthClose', 'bs_mouthDimpleLeft', 'bs_mouthDimpleRight',
    'bs_mouthFrownLeft', 'bs_mouthFrownRight',
    'bs_mouthFunnel', 'bs_mouthLeft',
    'bs_mouthLowerDownLeft', 'bs_mouthLowerDownRight',
    'bs_mouthPressLeft', 'bs_mouthPressRight',
    'bs_mouthPucker', 'bs_mouthRight',
    'bs_mouthRollLower', 'bs_mouthRollUpper',
    'bs_mouthShrugLower', 'bs_mouthShrugUpper',
    'bs_mouthSmileLeft', 'bs_mouthSmileRight',
    'bs_mouthStretchLeft', 'bs_mouthStretchRight',
    'bs_mouthUpperUpLeft', 'bs_mouthUpperUpRight',
    # Nose (2) — "is the nose wrinkled?"
    'bs_noseSneerLeft', 'bs_noseSneerRight',
]  # 52 total

ALL_FEATURE_NAMES = (
    DEEPFACE_FEATURE_NAMES +       # 16
    INSIGHTFACE_FEATURE_NAMES +    #  5
    MEDIAPIPE_BLENDSHAPE_NAMES     # 52
)  # = 73 total

NUM_DEEPFACE = len(DEEPFACE_FEATURE_NAMES)       # 16
NUM_INSIGHTFACE = len(INSIGHTFACE_FEATURE_NAMES)  #  5
NUM_MEDIAPIPE = len(MEDIAPIPE_BLENDSHAPE_NAMES)   # 52
NUM_TOTAL = len(ALL_FEATURE_NAMES)                # 73

# Feature index ranges for slicing
IDX_DEEPFACE = (0, NUM_DEEPFACE)                                           # 0:16
IDX_INSIGHTFACE = (NUM_DEEPFACE, NUM_DEEPFACE + NUM_INSIGHTFACE)           # 16:21
IDX_MEDIAPIPE = (NUM_DEEPFACE + NUM_INSIGHTFACE, NUM_TOTAL)                # 21:73

SUMMARY_SUFFIXES = ['_mean', '_std', '_delta']
SUMMARY_NAMES = [f"{name}{suffix}" for suffix in SUMMARY_SUFFIXES for name in ALL_FEATURE_NAMES]
NUM_SUMMARY = len(SUMMARY_NAMES)  # 219


# ══════════════════════════════════════════════════════════════════════════════
# LOGGER
# ══════════════════════════════════════════════════════════════════════════════

def create_logger(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger('semantic_extraction')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path)
    sh = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADERS (lazy — loaded once, reused across all videos)
# ══════════════════════════════════════════════════════════════════════════════

_deepface_loaded = False
_insightface_app = None
_mediapipe_landmarker = None


def load_deepface(logger: logging.Logger):
    """Import DeepFace and warm up models."""
    global _deepface_loaded
    if _deepface_loaded:
        return True

    try:
        from deepface import DeepFace

        # Warm up — triggers model downloads if needed
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        dummy[80:180, 80:180] = 128
        try:
            DeepFace.analyze(
                dummy,
                actions=['age', 'gender', 'race', 'emotion'],
                detector_backend='skip',
                enforce_detection=False,
                silent=True,
            )
        except Exception:
            pass  # warmup on dummy, don't care about result

        _deepface_loaded = True
        logger.info("[DeepFace] Models loaded and warmed up")
        return True

    except ImportError:
        logger.error("[DeepFace] Not installed. Run: pip install deepface")
        return False


def load_insightface(logger: logging.Logger):
    """Load InsightFace FaceAnalysis app with buffalo_l model on GPU."""
    global _insightface_app
    if _insightface_app is not None:
        return _insightface_app

    try:
        from insightface.app import FaceAnalysis

        # Explicit CUDA provider with device selection
        # Respects CUDA_VISIBLE_DEVICES env variable
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,  # maps to CUDA_VISIBLE_DEVICES[0]
                'arena_extend_strategy': 'kSameAsRequested',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB cap
                'cudnn_conv_algo_search': 'HEURISTIC',
            }),
            'CPUExecutionProvider',
        ]

        app = FaceAnalysis(
            name='buffalo_l',
            providers=providers,
        )
        app.prepare(ctx_id=0, det_size=(256, 256))
        _insightface_app = app
        logger.info("[InsightFace] buffalo_l loaded (CUDA + CPU fallback)")
        return app

    except ImportError:
        logger.warning("[InsightFace] Not installed. Run: pip install insightface onnxruntime-gpu")
        return None
    except Exception as e:
        logger.warning(f"[InsightFace] Failed to load: {e}")
        return None


def load_mediapipe(model_path: str, logger: logging.Logger):
    """
    Load MediaPipe FaceLandmarker with blendshape output enabled.

    Requires the face_landmarker.task model file. Download:
      wget -O face_landmarker.task \\
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    """
    global _mediapipe_landmarker
    if _mediapipe_landmarker is not None:
        return _mediapipe_landmarker

    if not os.path.exists(model_path):
        logger.warning(
            f"[MediaPipe] Model not found at: {model_path}\n"
            f"  Download with:\n"
            f'  wget -O {model_path} '
            f'"https://storage.googleapis.com/mediapipe-models/face_landmarker/'
            f'face_landmarker/float16/latest/face_landmarker.task"'
        )
        return None

    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceLandmarker,
            FaceLandmarkerOptions,
            RunningMode,
        )

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.3,   # lower threshold for aligned crops
            min_face_presence_confidence=0.3,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        landmarker = FaceLandmarker.create_from_options(options)
        _mediapipe_landmarker = landmarker
        logger.info("[MediaPipe] FaceLandmarker loaded with blendshape output")
        return landmarker

    except ImportError:
        logger.warning("[MediaPipe] Not installed. Run: pip install mediapipe")
        return None
    except Exception as e:
        logger.warning(f"[MediaPipe] Failed to load: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PER-FRAME FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_deepface_features(face_img: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract DeepFace semantic features from a single aligned face crop.

    Args:
        face_img: (H, W, 3) BGR uint8 — aligned face crop

    Returns:
        (16,) float32 or None

    NOTE: detector_backend='skip' because input is already cropped/aligned.
    """
    from deepface import DeepFace

    try:
        result = DeepFace.analyze(
            face_img,
            actions=['age', 'gender', 'race', 'emotion'],
            detector_backend='skip',
            enforce_detection=False,
            silent=True,
        )
        if isinstance(result, list):
            result = result[0]

        f = np.zeros(NUM_DEEPFACE, dtype=np.float32)

        # Emotion (7)
        emo = result.get('emotion', {})
        f[0] = emo.get('angry', 0) / 100.0
        f[1] = emo.get('disgust', 0) / 100.0
        f[2] = emo.get('fear', 0) / 100.0
        f[3] = emo.get('happy', 0) / 100.0
        f[4] = emo.get('sad', 0) / 100.0
        f[5] = emo.get('surprise', 0) / 100.0
        f[6] = emo.get('neutral', 0) / 100.0

        # Age (1) — normalized to [0,1]
        f[7] = min(result.get('age', 0) / 100.0, 1.0)

        # Gender (2)
        gen = result.get('gender', {})
        f[8] = gen.get('Man', 0) / 100.0
        f[9] = gen.get('Woman', 0) / 100.0

        # Race (6)
        race = result.get('race', {})
        f[10] = race.get('asian', 0) / 100.0
        f[11] = race.get('indian', 0) / 100.0
        f[12] = race.get('black', 0) / 100.0
        f[13] = race.get('white', 0) / 100.0
        f[14] = race.get('middle eastern', 0) / 100.0
        f[15] = race.get('latino hispanic', 0) / 100.0

        return f

    except Exception:
        return None


def extract_insightface_features(face_img: np.ndarray, app) -> Optional[np.ndarray]:
    """
    Extract InsightFace semantic features from a single aligned face crop.

    Args:
        face_img: (H, W, 3) BGR uint8
        app: InsightFace FaceAnalysis app

    Returns:
        (5,) float32 or None
    """
    try:
        faces = app.get(face_img)
        if len(faces) == 0:
            return None

        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        f = np.zeros(NUM_INSIGHTFACE, dtype=np.float32)

        # Head pose — normalize degrees to [-1, 1]
        pose = getattr(face, 'pose', None)
        if pose is not None and len(pose) >= 3:
            f[0] = np.clip(pose[0] / 90.0, -1.0, 1.0)  # yaw
            f[1] = np.clip(pose[1] / 90.0, -1.0, 1.0)  # pitch
            f[2] = np.clip(pose[2] / 90.0, -1.0, 1.0)  # roll

        # Face quality
        det_score = getattr(face, 'det_score', None)
        f[3] = float(det_score) if det_score is not None else 0.5

        # Anti-spoof (may not be available in buffalo_l)
        antispoof = getattr(face, 'antispoof_score', None)
        f[4] = float(antispoof) if antispoof is not None else 0.5

        return f

    except Exception:
        return None


def extract_mediapipe_blendshapes(
    face_img: np.ndarray,
    landmarker,
) -> Optional[np.ndarray]:
    """
    Extract 52 MediaPipe blendshape coefficients from an aligned face crop.

    Args:
        face_img: (H, W, 3) BGR uint8
        landmarker: MediaPipe FaceLandmarker instance

    Returns:
        (52,) float32 or None

    MediaPipe FaceLandmarker returns blendshapes as a list of 52 Category
    objects, each with .score in [0, 1]. The order follows the ARKit
    Face Tracking standard:
      [0] _neutral, [1] browDownLeft, ..., [51] noseSneerRight

    Our feature vector preserves this exact ordering.
    """
    import mediapipe as mp

    try:
        # MediaPipe expects RGB
        rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect(mp_image)

        if not result.face_blendshapes or len(result.face_blendshapes) == 0:
            return None

        # Take first face's blendshapes
        blendshapes = result.face_blendshapes[0]

        if len(blendshapes) < 52:
            return None

        # Extract scores in canonical order
        f = np.zeros(NUM_MEDIAPIPE, dtype=np.float32)
        for i, bs in enumerate(blendshapes[:52]):
            f[i] = bs.score

        return f

    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO-LEVEL EXTRACTION AND SUMMARIZATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_temporal_summary(per_frame: np.ndarray) -> np.ndarray:
    """
    Compute video-level temporal summaries.

    Args:
        per_frame: (T, 73) float32

    Returns:
        (219,) float32 — [mean(73), std(73), delta(73)]
    """
    T, A = per_frame.shape

    mean = np.nanmean(per_frame, axis=0)
    std = np.nanstd(per_frame, axis=0)

    if T > 1:
        deltas = np.abs(np.diff(per_frame, axis=0))
        delta = np.nanmean(deltas, axis=0)
    else:
        delta = np.zeros(A, dtype=np.float32)

    mean = np.nan_to_num(mean, nan=0.0)
    std = np.nan_to_num(std, nan=0.0)
    delta = np.nan_to_num(delta, nan=0.0)

    return np.concatenate([mean, std, delta]).astype(np.float32)


def forward_fill_nan(arr: np.ndarray) -> np.ndarray:
    """Forward-fill then back-fill NaN values column-wise."""
    result = arr.copy()
    T, A = result.shape

    for j in range(A):
        col = result[:, j]
        mask = np.isnan(col)
        if not np.any(mask):
            continue
        if np.all(mask):
            col[:] = 0.0
            continue

        # Forward fill
        idx = np.where(~mask, np.arange(T), 0)
        np.maximum.accumulate(idx, out=idx)
        col[mask] = col[idx[mask]]

        # Back fill remaining
        mask = np.isnan(col)
        if np.any(mask):
            idx = np.where(~mask, np.arange(T), T - 1)
            idx = np.minimum.accumulate(idx[::-1])[::-1]
            col[mask] = col[idx[mask]]

    return result


def extract_video_features(
    frames_dir: Path,
    insightface_app,
    mediapipe_landmarker,
    use_deepface: bool = True,
    use_insightface: bool = True,
    use_mediapipe: bool = True,
    logger: logging.Logger = None,
) -> Optional[Dict]:
    """
    Extract all semantic features for one video's frames.

    Args:
        frames_dir: directory containing 000.png, 001.png, ...
        insightface_app: loaded InsightFace app (or None)
        mediapipe_landmarker: loaded MediaPipe FaceLandmarker (or None)
        use_deepface, use_insightface, use_mediapipe: flags
        logger: logging object

    Returns:
        dict with 'per_frame', 'summary', 'failed_frames', etc.
    """
    frame_paths = sorted(glob.glob(str(frames_dir / '*.png')))
    if not frame_paths:
        frame_paths = sorted(glob.glob(str(frames_dir / '*.jpg')))
    if not frame_paths:
        return None

    num_frames = len(frame_paths)
    per_frame = np.full((num_frames, NUM_TOTAL), np.nan, dtype=np.float32)
    failed_frames = []

    for i, fpath in enumerate(frame_paths):
        face_img = cv2.imread(fpath)
        if face_img is None:
            failed_frames.append(i)
            continue

        frame_failed = False

        # ── DeepFace (indices 0:16) ──
        if use_deepface:
            df = extract_deepface_features(face_img)
            if df is not None:
                per_frame[i, IDX_DEEPFACE[0]:IDX_DEEPFACE[1]] = df
            else:
                frame_failed = True

        # ── InsightFace (indices 16:21) ──
        if use_insightface and insightface_app is not None:
            isf = extract_insightface_features(face_img, insightface_app)
            if isf is not None:
                per_frame[i, IDX_INSIGHTFACE[0]:IDX_INSIGHTFACE[1]] = isf

        # ── MediaPipe blendshapes (indices 21:73) ──
        if use_mediapipe and mediapipe_landmarker is not None:
            mp_bs = extract_mediapipe_blendshapes(face_img, mediapipe_landmarker)
            if mp_bs is not None:
                per_frame[i, IDX_MEDIAPIPE[0]:IDX_MEDIAPIPE[1]] = mp_bs

        if frame_failed:
            failed_frames.append(i)

    # Check we got at least some valid data
    valid_mask = ~np.all(np.isnan(per_frame), axis=1)
    if not np.any(valid_mask):
        if logger:
            logger.warning(f"All {num_frames} frames failed for {frames_dir.name}")
        return None

    # Fill NaN frames (forward-fill, back-fill)
    per_frame_filled = forward_fill_nan(per_frame)
    summary = compute_temporal_summary(per_frame_filled)

    return {
        'per_frame': per_frame_filled,
        'summary': summary,
        'failed_frames': failed_frames,
        'num_frames': num_frames,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PROCESSING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def process_subdataset(
    frames_root: Path,
    output_root: Path,
    insightface_app,
    mediapipe_landmarker,
    use_deepface: bool,
    use_insightface: bool,
    use_mediapipe: bool,
    skip_existing: bool,
    logger: logging.Logger,
) -> Tuple[int, int]:
    """
    Process all videos in a sub-dataset.

    Input:  frames_root/frames/{video_name}/000.png ...
    Output: output_root/semantic_features/{video_name}.npz
    """
    frames_dir = frames_root / 'frames'
    if not frames_dir.exists():
        logger.warning(f"No 'frames' directory in {frames_root}")
        return 0, 0

    video_dirs = sorted([d for d in frames_dir.iterdir() if d.is_dir()])
    if not video_dirs:
        logger.warning(f"No video directories in {frames_dir}")
        return 0, 0

    output_dir = output_root / 'semantic_features'
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    failed = 0

    for video_dir in tqdm(video_dirs, desc=f"  {frames_root.name}", leave=False):
        video_name = video_dir.name
        output_path = output_dir / f"{video_name}.npz"

        if skip_existing and output_path.exists():
            processed += 1
            continue

        try:
            result = extract_video_features(
                video_dir,
                insightface_app,
                mediapipe_landmarker,
                use_deepface=use_deepface,
                use_insightface=use_insightface,
                use_mediapipe=use_mediapipe,
                logger=logger,
            )

            if result is None:
                failed += 1
                continue

            # Save — mirrors DeepfakeBench landmark storage pattern
            np.savez_compressed(
                str(output_path),
                per_frame=result['per_frame'],           # (T, 73)
                summary=result['summary'],               # (219,)
                feature_names=ALL_FEATURE_NAMES,          # (73,)
                summary_names=SUMMARY_NAMES,              # (219,)
                extraction_info={
                    'num_frames': result['num_frames'],
                    'failed_frames': result['failed_frames'],
                    'num_features_deepface': NUM_DEEPFACE,
                    'num_features_insightface': NUM_INSIGHTFACE,
                    'num_features_mediapipe': NUM_MEDIAPIPE,
                    'use_deepface': use_deepface,
                    'use_insightface': use_insightface,
                    'use_mediapipe': use_mediapipe,
                    'timestamp': datetime.datetime.now().isoformat(),
                },
            )
            processed += 1

        except Exception as e:
            failed += 1
            logger.error(f"Error processing {video_name}: {e}\n{traceback.format_exc()}")

    return processed, failed


def main():
    parser = argparse.ArgumentParser(
        description='Extract identity-invariant semantic facial attributes '
                    '(DeepFace + InsightFace + MediaPipe blendshapes)'
    )
    parser.add_argument('--config', type=str, default='./config.yaml')
    parser.add_argument('--dataset_name', type=str, default=None)
    parser.add_argument('--comp', type=str, default=None)
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip videos with existing .npz files')
    parser.add_argument('--deepface_only', action='store_true',
                        help='Only DeepFace (16 features)')
    parser.add_argument('--no_mediapipe', action='store_true',
                        help='Skip MediaPipe blendshapes')
    parser.add_argument('--no_insightface', action='store_true',
                        help='Skip InsightFace')
    parser.add_argument('--no_deepface', action='store_true',
                        help='Skip DeepFace')
    parser.add_argument('--mediapipe_model_path', type=str,
                        default='./face_landmarker.task',
                        help='Path to MediaPipe face_landmarker.task model')
    args = parser.parse_args()

    # ── Load config ──
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    dataset_name = args.dataset_name or config['preprocess']['dataset_name']['default']
    output_root_path = config['preprocess']['output_root_path']['default']
    comp = args.comp or config['preprocess']['comp']['default']

    # ── Setup logging ──
    log_path = f'./logs/semantic_extraction_{dataset_name}.log'
    logger = create_logger(log_path)
    logger.info("=" * 70)
    logger.info(f"Semantic Feature Extraction — {dataset_name}")
    logger.info(f"Output root: {output_root_path}")
    logger.info("=" * 70)

    # ── Determine extractors ──
    if args.deepface_only:
        use_deepface, use_insightface, use_mediapipe = True, False, False
    else:
        use_deepface = not args.no_deepface
        use_insightface = not args.no_insightface
        use_mediapipe = not args.no_mediapipe

    active = []
    if use_deepface:     active.append(f"DeepFace ({NUM_DEEPFACE})")
    if use_insightface:  active.append(f"InsightFace ({NUM_INSIGHTFACE})")
    if use_mediapipe:    active.append(f"MediaPipe ({NUM_MEDIAPIPE})")
    n_active = (NUM_DEEPFACE if use_deepface else 0) + \
               (NUM_INSIGHTFACE if use_insightface else 0) + \
               (NUM_MEDIAPIPE if use_mediapipe else 0)
    logger.info(f"Active extractors: {' + '.join(active)} = {n_active} per-frame features")
    logger.info(f"Summary variables: {n_active * 3}")

    # ── Load models ──
    if use_deepface:
        logger.info("Loading DeepFace...")
        if not load_deepface(logger):
            logger.error("DeepFace failed to load. Exiting.")
            sys.exit(1)

    insightface_app = None
    if use_insightface:
        logger.info("Loading InsightFace...")
        insightface_app = load_insightface(logger)
        if insightface_app is None:
            logger.warning("InsightFace unavailable — those features will be NaN-filled.")

    mediapipe_landmarker = None
    if use_mediapipe:
        logger.info("Loading MediaPipe FaceLandmarker...")
        mediapipe_landmarker = load_mediapipe(args.mediapipe_model_path, logger)
        if mediapipe_landmarker is None:
            logger.warning("MediaPipe unavailable — blendshapes will be NaN-filled.")

    # ── Resolve sub-dataset paths (mirrors preprocess.py exactly) ──
    output_base = Path(output_root_path) / dataset_name

    if dataset_name == 'FaceForensics++':
        sub_names = [
            "original_sequences/youtube",
            "original_sequences/actors",
            "manipulated_sequences/Deepfakes",
            "manipulated_sequences/Face2Face",
            "manipulated_sequences/FaceSwap",
            "manipulated_sequences/NeuralTextures",
            "manipulated_sequences/FaceShifter",
            "manipulated_sequences/DeepFakeDetection",
        ]
        sub_paths = [output_base / name / comp for name in sub_names]

    elif dataset_name in ('Celeb-DF-v1', 'Celeb-DF-v2'):
        sub_names = ['Celeb-real', 'Celeb-synthesis', 'YouTube-real']
        sub_paths = [output_base / name for name in sub_names]

    elif dataset_name == 'DFDCP':
        sub_names = ['original_videos', 'method_A', 'method_B']
        sub_paths = [output_base / name for name in sub_names]

    elif dataset_name == 'DFDC':
        sub_names = ['test'] + [f"train/dfdc_train_part_{i}" for i in range(50)]
        sub_paths = [output_base / name for name in sub_names]

    elif dataset_name == 'DeeperForensics-1.0':
        sub_names = []
        for top in ['source_videos', 'manipulated_videos']:
            top_path = output_base / top
            if top_path.exists():
                sub_names += [f"{top}/{d.name}" for d in sorted(top_path.iterdir()) if d.is_dir()]
        sub_paths = [output_base / name for name in sub_names]

    elif dataset_name == 'UADFV':
        sub_names = ['fake', 'real']
        sub_paths = [output_base / name for name in sub_names]

    else:
        logger.error(f"Unknown dataset: {dataset_name}")
        sys.exit(1)

    # ── Process ──
    start_time = time.monotonic()
    total_processed = 0
    total_failed = 0

    for sub_path in sub_paths:
        if not sub_path.exists():
            logger.warning(f"Path does not exist: {sub_path}")
            continue
        if not (sub_path / 'frames').exists():
            logger.warning(f"No 'frames' in {sub_path}, skipping")
            continue

        logger.info(f"Processing: {sub_path}")
        processed, failed = process_subdataset(
            frames_root=sub_path,
            output_root=sub_path,
            insightface_app=insightface_app,
            mediapipe_landmarker=mediapipe_landmarker,
            use_deepface=use_deepface,
            use_insightface=use_insightface,
            use_mediapipe=use_mediapipe,
            skip_existing=args.skip_existing,
            logger=logger,
        )
        total_processed += processed
        total_failed += failed
        logger.info(f"  Done: {processed} processed, {failed} failed")

    elapsed = (time.monotonic() - start_time) / 60
    logger.info("=" * 70)
    logger.info(f"COMPLETE — {total_processed} videos, {total_failed} failed, {elapsed:.1f} min")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
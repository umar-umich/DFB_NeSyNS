"""
networks/nesy_defake/semantic/refined_attributes.py
====================================================
Fast-only semantic feature taxonomy (v8, 2026-04-20).

VLM features were removed from the framework entirely. The combined
feature vector is now just the 58-d fast-extracted set produced by
InsightFace + MediaPipe + DeepFace. Downstream branches (concept,
ImprovedSCM, CCV) operate on this 58-d input.

Source: `preprocessing/precompute_fast_semantic.py` writes
`fast_semantic/{video_id}.pt` with shape (n_frames, 58).

Causal attributes (26) feed the SCM identity sub-graph as named nodes
alongside the compressed CLIP z-features. All chains are now
geometric/behavioural — the appearance-identity chains that required
VLM (gender→beard, age→wrinkles, hair mismatch, etc.) were dropped.
"""

from typing import List

# ═══════════════════════════════════════════════════════════════════════════
#  Fast-Extracted Features (InsightFace + MediaPipe + DeepFace)
# ═══════════════════════════════════════════════════════════════════════════

FAST_FEATURE_NAMES: List[str] = [
    # -- Demographics (3) --
    'fs_gender_score',           # [-1, 1]: -1=female, +1=male
    'fs_age_score',              # [0, 100] continuous age
    'fs_ethnicity_entropy',      # [0, 1]: low=confident, high=ambiguous

    # -- Expression (8) — calibrated softmax --
    'fs_expr_neutral', 'fs_expr_happy', 'fs_expr_sad', 'fs_expr_angry',
    'fs_expr_surprise', 'fs_expr_fear', 'fs_expr_disgust', 'fs_expr_contempt',

    # -- Action Units (23) --
    'fs_AU1', 'fs_AU2', 'fs_AU4', 'fs_AU5', 'fs_AU6', 'fs_AU9',
    'fs_AU10', 'fs_AU12', 'fs_AU14', 'fs_AU15', 'fs_AU17', 'fs_AU18',
    'fs_AU20', 'fs_AU23', 'fs_AU24', 'fs_AU25', 'fs_AU26', 'fs_AU28',
    'fs_AU43', 'fs_AU45', 'fs_AU7', 'fs_AU16', 'fs_AU22',

    # -- Head Pose (3) --
    'fs_pose_yaw', 'fs_pose_pitch', 'fs_pose_roll',

    # -- Gaze (4) --
    'fs_gaze_left_x', 'fs_gaze_left_y',
    'fs_gaze_right_x', 'fs_gaze_right_y',

    # -- Landmark Geometry (10) --
    'fs_eye_aspect_ratio_left', 'fs_eye_aspect_ratio_right',
    'fs_mouth_aspect_ratio', 'fs_interpupillary_ratio',
    'fs_nose_width_ratio', 'fs_jaw_width_ratio',
    'fs_face_width_height_ratio', 'fs_chin_angle',
    'fs_brow_height_ratio', 'fs_mouth_nose_ratio',

    # -- Quality / Symmetry (7) --
    'fs_det_confidence', 'fs_landmark_confidence', 'fs_face_area_ratio',
    'fs_blur_laplacian', 'fs_eye_lr_symmetry', 'fs_mouth_symmetry',
    'fs_jaw_symmetry',
]

NUM_FAST_FEATURES = len(FAST_FEATURE_NAMES)
assert NUM_FAST_FEATURES == 58, \
    f"Expected 58 fast features, got {NUM_FAST_FEATURES}"

_FAST_INDEX = {name: idx for idx, name in enumerate(FAST_FEATURE_NAMES)}


# ═══════════════════════════════════════════════════════════════════════════
#  Combined vector — identical to fast set (VLM removed)
# ═══════════════════════════════════════════════════════════════════════════

COMBINED_FEATURE_NAMES: List[str] = FAST_FEATURE_NAMES
NUM_COMBINED_FEATURES = NUM_FAST_FEATURES  # 58
_COMBINED_INDEX = _FAST_INDEX


def get_combined_index(name: str) -> int:
    """Return the index of ``name`` in the combined feature vector."""
    return _COMBINED_INDEX[name]


def ci(name: str) -> int:
    """Shorthand for ``get_combined_index`` — used in consistency rules."""
    return _COMBINED_INDEX[name]


# ═══════════════════════════════════════════════════════════════════════════
#  Causal Attributes (curated identity sub-graph nodes)
# ═══════════════════════════════════════════════════════════════════════════
# Fast-only causal chains preserved after VLM removal:
#   demographics anchor:    fs_gender_score, fs_age_score
#   expression ↔ AU:        fs_expr_* ↔ fs_AU*
#   pose ↔ gaze:            fs_pose_* ↔ fs_gaze_*
#   geometric symmetry:     fs_*_symmetry, landmark ratios
#   detection quality:      fs_det_confidence

CAUSAL_ATTRIBUTE_NAMES: List[str] = [
    # Demographic anchors (2)
    'fs_gender_score', 'fs_age_score',

    # Structural geometry (6)
    'fs_jaw_width_ratio', 'fs_nose_width_ratio', 'fs_face_width_height_ratio',
    'fs_interpupillary_ratio', 'fs_chin_angle', 'fs_brow_height_ratio',

    # Expression-AU coherence (10)
    'fs_expr_happy', 'fs_expr_sad', 'fs_expr_angry', 'fs_expr_surprise',
    'fs_AU6', 'fs_AU12', 'fs_AU1', 'fs_AU4', 'fs_AU15', 'fs_AU9',

    # Pose-gaze coherence (5)
    'fs_pose_yaw', 'fs_pose_pitch',
    'fs_gaze_left_x', 'fs_gaze_right_x', 'fs_gaze_left_y',

    # Symmetry + quality (3)
    'fs_eye_lr_symmetry', 'fs_jaw_symmetry', 'fs_det_confidence',
]

CAUSAL_ATTRIBUTE_INDICES: List[int] = [
    _COMBINED_INDEX[name] for name in CAUSAL_ATTRIBUTE_NAMES]
NUM_CAUSAL_ATTRIBUTES = len(CAUSAL_ATTRIBUTE_NAMES)

assert NUM_CAUSAL_ATTRIBUTES == 26, \
    f"Expected 26 causal attributes, got {NUM_CAUSAL_ATTRIBUTES}"

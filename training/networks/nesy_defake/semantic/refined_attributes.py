"""
networks/nesy_defake/semantic/refined_attributes.py
====================================================
Refined Semantic Feature Taxonomy for Deepfake Detection (v7, 2026-04-05).

Redesign goals:
  1. Remove duplicates and forensically useless attributes
  2. Move demographics/AUs/expression/geometry to fast extractors
     (InsightFace, MediaPipe, DeepFace) — 100x faster than FaceBench LLM
  3. Keep only VLM-requiring attributes in FaceBench
  4. Add missing features critical for deepfake detection
     (head pose, gaze, landmark geometry, eye/mouth aspect ratios)
  5. Design for scale=1.3 face crops where hair, ears, neck are visible

Three-source feature vector (precomputed, concatenated at load time):
  Source A: Fast models  (InsightFace + MediaPipe + DeepFace)  ~30ms/frame
  Source B: FaceBench VLM (Face-LLaVA 13B, teacher-forced)     ~500ms/frame
  Combined: [fast(58) || vlm(64)] = 122 features

Compared to original 211 FaceBench:
  - 54 irrelevant/duplicate attributes removed
  - 48 attributes moved to 100x faster extractors (with improvements)
  - 14 new forensically critical features added (pose, gaze, geometry)
  - Net: 211 slow features → 64 slow + 58 fast = 122 total, 6x faster
"""

from typing import List, Dict, Tuple

# ═══════════════════════════════════════════════════════════════════════════
#  SOURCE A: Fast-Extracted Features (InsightFace + MediaPipe + DeepFace)
# ═══════════════════════════════════════════════════════════════════════════
# Precomputed by preprocessing/precompute_fast_semantic.py
# Output: fast_semantic/{video_id}.pt → (n_frames, 58)
#
# These replace 48 FaceBench attributes with faster, more accurate versions
# and add 14 entirely new features not available from FaceBench.

FAST_FEATURE_NAMES = [
    # -- Demographics (InsightFace) — 3 features --
    # Continuous scores are more informative than binary FaceBench labels.
    # Replaces: male, female (2) → 1 continuous score
    # Replaces: baby_face, young_looking, middle_aged, elderly_looking,
    #           age_ambiguous (5) → 1 continuous score
    # Replaces: east_asian, african, caucasian, hispanic, south_asian (5) → 1
    'fs_gender_score',           # [-1, 1]: -1=female, +1=male (InsightFace)
    'fs_age_score',              # [0, 100]: continuous age (InsightFace)
    'fs_ethnicity_entropy',      # [0, 1]: low=confident, high=ambiguous (DeepFace)

    # -- Expression (DeepFace / HSEmotion) — 8 features --
    # Replaces FaceBench expression attributes with calibrated softmax scores.
    # DeepFace emotion model is specifically trained for this — LLaVA is not.
    # Replaces: neutral_expression, happy, sad, angry, surprised, fearful,
    #           disgusted, contemptuous (8)
    'fs_expr_neutral',           # softmax P(neutral)
    'fs_expr_happy',             # softmax P(happy)
    'fs_expr_sad',               # softmax P(sad)
    'fs_expr_angry',             # softmax P(angry)
    'fs_expr_surprise',          # softmax P(surprise)
    'fs_expr_fear',              # softmax P(fear)
    'fs_expr_disgust',           # softmax P(disgust)
    'fs_expr_contempt',          # softmax P(contempt) — 0 if model lacks it

    # -- Action Units (py-feat / LibreFace) — 20 features --
    # Dedicated AU models are FAR more accurate than LLaVA for AUs.
    # LLaVA was never trained on AU detection — it guesses from text semantics.
    # Replaces all 25 FaceBench AUs; we keep the 20 most forensically relevant.
    # Dropped: AU11 (nasolabial), AU13 (sharp_lip_puller) — rare, noisy.
    'fs_AU1',                    # inner brow raise
    'fs_AU2',                    # outer brow raise
    'fs_AU4',                    # brow lowerer
    'fs_AU5',                    # upper lid raise
    'fs_AU6',                    # cheek raise
    'fs_AU9',                    # nose wrinkler
    'fs_AU10',                   # upper lip raiser
    'fs_AU12',                   # lip corner puller (smile)
    'fs_AU14',                   # dimpler
    'fs_AU15',                   # lip corner depressor
    'fs_AU17',                   # chin raiser
    'fs_AU18',                   # lip pucker
    'fs_AU20',                   # lip stretcher
    'fs_AU23',                   # lip tightener
    'fs_AU24',                   # lip pressor
    'fs_AU25',                   # lips part
    'fs_AU26',                   # jaw drop
    'fs_AU28',                   # lip suck
    'fs_AU43',                   # eyes closed
    'fs_AU45',                   # blink
    'fs_AU7',                    # lid tightener
    'fs_AU16',                   # lower lip depressor
    'fs_AU22',                   # lip funneler

    # -- Head Pose (InsightFace / 6DRepNet) — 3 features --
    # NEW: Not in FaceBench. Critical for deepfake detection — faceswap
    # often introduces subtle pose inconsistencies between the pasted face
    # and the head/neck/hair context visible in the scale=1.3 crop.
    'fs_pose_yaw',               # [-90, 90] degrees
    'fs_pose_pitch',             # [-90, 90] degrees
    'fs_pose_roll',              # [-90, 90] degrees

    # -- Gaze Direction (MediaPipe) — 4 features --
    # NEW: Not in FaceBench. Deepfakes frequently have inconsistent gaze
    # (eyes look in slightly wrong direction after face replacement).
    'fs_gaze_left_x',            # left eye gaze x [-1, 1]
    'fs_gaze_left_y',            # left eye gaze y [-1, 1]
    'fs_gaze_right_x',           # right eye gaze x [-1, 1]
    'fs_gaze_right_y',           # right eye gaze y [-1, 1]

    # -- Landmark Geometry (MediaPipe 478 points) — 10 features --
    # NEW: Replaces vague FaceBench descriptions (large_nose, narrow_jaw)
    # with precise geometric ratios. More reliable for causal graphs because
    # these are continuous measurements, not noisy VLM yes/no guesses.
    # Replaces: large_nose/small_nose, pointed_nose/broad_nose,
    #           large_eyes/small_eyes, wide_set_eyes, narrow_eyes,
    #           wide_mouth/small_mouth, strong_jaw/narrow_jaw (12 attrs → 10)
    'fs_eye_aspect_ratio_left',  # eye height/width (blink proxy)
    'fs_eye_aspect_ratio_right', # eye height/width
    'fs_mouth_aspect_ratio',     # mouth height/width
    'fs_interpupillary_ratio',   # eye distance / face width
    'fs_nose_width_ratio',       # nose width / face width
    'fs_jaw_width_ratio',        # jaw width / face width
    'fs_face_width_height_ratio',# face width / face height
    'fs_chin_angle',             # angle of chin contour (degrees)
    'fs_brow_height_ratio',      # brow-to-eye distance / face height
    'fs_mouth_nose_ratio',       # nose-to-mouth dist / face height

    # -- Quality / Confidence — 7 features --
    # NEW: Detection and landmark confidence are strong deepfake signals.
    # Face detection models are trained on real faces — fakes score lower.
    'fs_det_confidence',         # face detection confidence [0, 1]
    'fs_landmark_confidence',    # MediaPipe face mesh confidence [0, 1]
    'fs_face_area_ratio',        # face bbox area / image area
    'fs_blur_laplacian',         # Laplacian variance (focus/sharpness)
    'fs_eye_lr_symmetry',        # |left_eye_area - right_eye_area| / avg
    'fs_mouth_symmetry',         # L/R mouth corner height difference
    'fs_jaw_symmetry',           # L/R jaw contour difference
]

NUM_FAST_FEATURES = len(FAST_FEATURE_NAMES)
assert NUM_FAST_FEATURES == 58, \
    f"Expected 58 fast features, got {NUM_FAST_FEATURES}"

# Demographics(3) + Expression(8) + AUs(23) + Pose(3) + Gaze(4) +
# Geometry(10) + Quality(7) = 58

# Index mapping for fast features
_FAST_INDEX = {name: idx for idx, name in enumerate(FAST_FEATURE_NAMES)}


# ═══════════════════════════════════════════════════════════════════════════
#  SOURCE B: VLM-Only Features (FaceBench Face-LLaVA — slow but necessary)
# ═══════════════════════════════════════════════════════════════════════════
# These attributes genuinely require Vision-Language Model understanding.
# Specialized models cannot reliably extract them.
# Precomputed by preprocessing/precompute_semantic_features.py (existing)
# but only the 65 attributes below are used (indexed from the 211-d output).
#
# Design for scale=1.3 crops: hair, ears, partial neck/shoulders visible.
# Hair color/length matter for identity consistency (long_hair + beard = sus).

VLM_FEATURE_NAMES = [
    # -- Hair (visible in crop, identity consistency) — 11 --
    # Hair color: faceswap can create mismatches between hair color/style
    # and facial attributes (e.g., gray_hair + young_smooth_skin).
    # Hair length: long_hair + beard/stubble is a strong identity mismatch.
    'black_hair', 'blonde_hair', 'brown_hair', 'gray_hair', 'red_hair',
    'long_hair', 'short_hair', 'bald',
    'straight_hair', 'wavy_hair', 'curly_hair',

    # -- Forehead — 1 --
    'forehead_wrinkles',         # age consistency (dropped large/small — use landmarks)

    # -- Eyebrows — 3 --
    # Kept distinct shape descriptors; dropped anti-correlated duplicates
    # (thin/sparse overlap, thick/bushy overlap)
    'arched_eyebrows', 'thick_eyebrows', 'unibrow',

    # -- Eyes — 5 --
    # Dropped eye colors (not forensic), dropped size (use landmark ratios).
    # Kept eyelid type (identity signal) and under-eye (age/fatigue signal).
    'double_eyelid', 'single_eyelid', 'hooded_eyes',
    'bags_under_eyes', 'dark_circles',

    # -- Mouth & Lips — 4 --
    # Kept only features not redundant with AU/expression/landmark data.
    # Dropped: full_lips/thin_lips (use landmark mouth_aspect_ratio),
    #          wide_mouth/small_mouth (use landmark), smiling/frowning (use expr)
    'mouth_open', 'mouth_closed',
    'teeth_visible', 'white_teeth',

    # -- Cheeks — 2 --
    'high_cheekbones', 'rosy_cheeks',

    # -- Chin & Jaw — 3 --
    # Dropped strong_jaw/narrow_jaw (use landmark jaw_width_ratio).
    # Kept VLM-only: double_chin (softness, not geometry), cleft (detail).
    'double_chin', 'cleft_chin', 'pointed_chin',

    # -- Face Shape — 3 --
    # Kept the most distinct shapes; landmark ratios cover the rest.
    'oval_face', 'round_face', 'square_face',

    # -- Ears (visible in crop) — 1 --
    'protruding_ears',           # identity signal, visible at scale=1.3

    # -- Skin — 8 --
    # Skin condition requires VLM-level holistic understanding.
    # Dropped pale_skin (≈ fair_skin), oily/dry (not forensic), freckled (fine detail).
    'fair_skin', 'medium_skin', 'dark_skin', 'olive_skin',
    'smooth_skin', 'wrinkled_skin',
    'acne', 'age_spots',

    # -- Facial Hair — 6 --
    # ALL kept: critical for gender-identity causal chains.
    # beard + female gender_score = strong fake signal.
    'beard', 'mustache', 'goatee', 'sideburns', 'stubble', 'clean_shaven',

    # -- Neck (visible in crop) — 2 --
    # Kept: neck length is an identity feature visible in scale=1.3 crops.
    # Mismatch between face age/build and neck can indicate manipulation.
    'long_neck', 'short_neck',

    # -- Symmetry — 2 --
    # VLM holistic symmetry assessment complements landmark-based measurements.
    'symmetrical_face', 'asymmetrical_face',

    # -- Accessories (occlusion detection) — 5 --
    # Only high-level categories that affect face visibility.
    # Dropped fine-grained types (round_glasses, stud_earrings, etc.)
    'eyeglasses', 'sunglasses', 'hat', 'face_mask', 'headphones',

    # -- Makeup — 4 --
    # Gender consistency: heavy_makeup + high male score = suspicious.
    # Dropped fine-grained types (red_lipstick, mascara, contour, etc.)
    'heavy_makeup', 'no_makeup', 'lipstick', 'eyeliner',

    # -- Surrounding / Image Quality — 4 --
    # Lighting affects both forgery visibility and LLaVA confidence.
    # Background simplicity affects blending artifact visibility.
    'bright_lighting', 'dim_lighting',
    'blurry_image', 'sharp_image',
]

NUM_VLM_FEATURES = len(VLM_FEATURE_NAMES)
assert NUM_VLM_FEATURES == 64, \
    f"Expected 64 VLM features, got {NUM_VLM_FEATURES}"

# Index mapping: VLM feature name → position in original 211-d FaceBench vector
# Used to select the 65 features from precomputed 211-d files without re-extraction
from .facial_semantic_extractor import FACEBENCH_ATTRIBUTES
_FB_INDEX = {name: idx for idx, name in enumerate(FACEBENCH_ATTRIBUTES)}

VLM_INDICES_IN_FACEBENCH = [_FB_INDEX[name] for name in VLM_FEATURE_NAMES]
"""Indices into the 211-d FaceBench vector to extract the 65 VLM features."""

# Index mapping for VLM features within the combined vector
_VLM_INDEX = {name: idx for idx, name in enumerate(VLM_FEATURE_NAMES)}


# ═══════════════════════════════════════════════════════════════════════════
#  COMBINED FEATURE VECTOR: [fast(58) || vlm(65)] = 123 total
# ═══════════════════════════════════════════════════════════════════════════

COMBINED_FEATURE_NAMES = FAST_FEATURE_NAMES + VLM_FEATURE_NAMES
NUM_COMBINED_FEATURES = NUM_FAST_FEATURES + NUM_VLM_FEATURES
assert NUM_COMBINED_FEATURES == 122

_COMBINED_INDEX = {name: idx for idx, name in enumerate(COMBINED_FEATURE_NAMES)}


def get_combined_index(name: str) -> int:
    """Get index of a feature in the combined [fast || vlm] vector."""
    return _COMBINED_INDEX[name]


def ci(name: str) -> int:
    """Shorthand for get_combined_index — used in consistency rules."""
    return _COMBINED_INDEX[name]


# ═══════════════════════════════════════════════════════════════════════════
#  CAUSAL GRAPH ATTRIBUTES (curated for identity-causal sub-graph)
# ═══════════════════════════════════════════════════════════════════════════
# These attributes form natural causal chains that faceswap breaks.
# They enter the causal graph as individual named nodes alongside
# compressed CLIP z-features.
#
# Selection criteria (same as v5, applied to new combined vector):
#   1. Forms a causal chain with other attributes (not standalone)
#   2. CLIP spatial features can visually verify the attribute
#   3. Faceswap plausibly disrupts the causal chain
#
# Key causal chains captured:
#   gender_score → beard/mustache/makeup/eyeliner (identity)
#   age_score → wrinkled_skin/gray_hair/smooth_skin/age_spots (aging)
#   expression → AU activations (muscle consistency)
#   pose → gaze direction (head-eye coordination)
#   hair_length/color → gender/age (cross-region identity)

CAUSAL_ATTRIBUTE_NAMES = [
    # -- Gender anchor + gender-linked (10) --
    # Fast: continuous gender score replaces binary male/female
    'fs_gender_score',
    # VLM: gender-linked appearance features (causal targets)
    'beard', 'mustache', 'goatee', 'sideburns', 'stubble', 'clean_shaven',
    'heavy_makeup', 'lipstick', 'eyeliner',

    # -- Age anchor + age-linked (8) --
    'fs_age_score',
    'smooth_skin', 'wrinkled_skin', 'age_spots', 'forehead_wrinkles',
    'gray_hair', 'bald', 'dark_circles',

    # -- Hair (cross-region identity) — (5) --
    # Visible in crop. Mismatch with face attributes = manipulation signal.
    'long_hair', 'short_hair',
    'black_hair', 'blonde_hair', 'brown_hair',

    # -- Structural geometry (continuous) — (6) --
    # Fast: precise landmark ratios for causal graph
    'fs_jaw_width_ratio', 'fs_nose_width_ratio', 'fs_face_width_height_ratio',
    'fs_interpupillary_ratio', 'fs_chin_angle', 'fs_brow_height_ratio',

    # -- Expression-AU coherence (10) --
    # Fast: expression + key AUs that must agree
    'fs_expr_happy', 'fs_expr_sad', 'fs_expr_angry', 'fs_expr_surprise',
    'fs_AU6', 'fs_AU12', 'fs_AU1', 'fs_AU4', 'fs_AU15', 'fs_AU9',

    # -- Pose-gaze coherence (5) --
    # NEW: Head pose should be consistent with gaze direction and
    # with the hair/neck context visible in the crop.
    'fs_pose_yaw', 'fs_pose_pitch',
    'fs_gaze_left_x', 'fs_gaze_right_x', 'fs_gaze_left_y',

    # -- Skin tone (3) --
    # Skin tone should be consistent across face regions and with
    # neck/ear color visible in the crop margin.
    'fair_skin', 'medium_skin', 'dark_skin',

    # -- Symmetry + quality (4) --
    'symmetrical_face',
    'fs_eye_lr_symmetry', 'fs_jaw_symmetry',
    'fs_det_confidence',
]

CAUSAL_ATTRIBUTE_INDICES = [_COMBINED_INDEX[name] for name in CAUSAL_ATTRIBUTE_NAMES]
NUM_CAUSAL_ATTRIBUTES = len(CAUSAL_ATTRIBUTE_NAMES)

assert NUM_CAUSAL_ATTRIBUTES == 51, \
    f"Expected 51 causal attributes, got {NUM_CAUSAL_ATTRIBUTES}"


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE REMOVAL / CHANGE LOG
# ═══════════════════════════════════════════════════════════════════════════
# Documenting every change from the original 211 FaceBench attributes.
#
# REMOVED (54 attrs — no forensic value or undetectable by LLaVA):
#   Hair:       white_hair, medium_length_hair, bangs, receding_hairline,
#               hair_parted, ponytail, bun_hair, braided_hair, dyed_hair (9)
#   Forehead:   large_forehead, small_forehead (2 — use landmark ratios)
#   Eyebrows:   straight_eyebrows, thin_eyebrows, bushy_eyebrows,
#               sparse_eyebrows (4 — redundant with thick + landmark)
#   Eyes:       brown_eyes, blue_eyes, green_eyes, hazel_eyes, black_eyes,
#               large_eyes, small_eyes, wide_set_eyes, puffy_eyes (9 — color
#               irrelevant, size from landmarks)
#   Eyelashes:  long_eyelashes, thick_eyelashes, sparse_eyelashes (3)
#   Nose:       small_nose, pointed_nose, broad_nose, upturned_nose,
#               long_nose, crooked_nose, flat_nose, large_nose (8 — landmarks)
#   Mouth:      full_lips, thin_lips, wide_mouth, small_mouth, smiling,
#               frowning (6 — lips from landmarks, smile/frown from expr)
#   Cheeks:     round_cheeks, hollow_cheeks (2 — landmarks)
#   Chin/Jaw:   round_chin, strong_jaw, narrow_jaw (3 — landmarks)
#   Face shape: heart_shaped_face, long_face, diamond_face (3 — landmarks)
#   Ears:       large_ears, small_ears (2 — low signal)
#   Skin:       pale_skin, freckled_skin, moles, scars, skin_blemishes,
#               oily_skin, dry_skin (7 — pale≈fair, rest too fine for LLaVA)
#   Neck:       (0 — kept both, visible in crop)
#   Age:        baby_face, young_looking, middle_aged, elderly_looking,
#               age_ambiguous (5 — replaced by continuous InsightFace age)
#   Other:      attractive (1 — subjective)
#   Accessories: reading_glasses, round_glasses, rectangular_glasses,
#               rimless_glasses, thick_frame_glasses, baseball_cap, beanie,
#               headband, turban, hood, stud_earrings, hoop_earrings,
#               dangling_earrings, necklace, choker, pendant_necklace,
#               nose_piercing, lip_piercing, ear_piercing, scarf, necktie,
#               bowtie, hair_clip, hair_band, veil (25 — too granular)
#   Makeup:     light_makeup, red_lipstick, pink_lipstick, nude_lipstick,
#               eyeshadow, mascara, blush, foundation, contour_makeup (9)
#   Surrounding: indoor_background, outdoor_background, plain_background,
#               complex_background, natural_lighting, artificial_lighting,
#               side_lighting, bokeh_background (8 — mostly irrelevant)
#   Identity:   male, female, east_asian, african, caucasian, hispanic,
#               south_asian (7 — replaced by continuous InsightFace scores)
#
# MOVED TO FAST EXTRACTORS (48 attrs → better versions):
#   Demographics:  male/female → fs_gender_score (continuous)
#                  5 age attrs → fs_age_score (continuous)
#                  5 ethnicity → fs_ethnicity_entropy
#   Expression:    8 expr → fs_expr_* (calibrated softmax)
#   Action Units:  25 AUs → 20 fs_AU* (dedicated AU model, much more accurate)
#   Geometry:      12 size/shape attrs → 10 landmark ratios
#
# ADDED (14 new features, all fast-extractable):
#   Head pose:     fs_pose_yaw, fs_pose_pitch, fs_pose_roll (3)
#   Gaze:          fs_gaze_left_x/y, fs_gaze_right_x/y (4)
#   Quality:       fs_det_confidence, fs_landmark_confidence,
#                  fs_face_area_ratio, fs_blur_laplacian (4)
#   Symmetry:      fs_eye_lr_symmetry, fs_mouth_symmetry,
#                  fs_jaw_symmetry (3)

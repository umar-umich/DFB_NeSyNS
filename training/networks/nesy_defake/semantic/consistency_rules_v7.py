"""
networks/nesy_defake/semantic/consistency_rules_v7.py
=====================================================
Cross-Attribute Consistency Rules v7 — operates on the COMBINED 122-d vector.

v7 changes (2026-04-05):
  - Operates on combined [fast(58) || vlm(64)] = 122-d feature vector
  - Expression-AU rules now use fast extractor scores (more accurate)
  - Gender rules use continuous fs_gender_score instead of binary male/female
  - NEW: Cross-region identity rules (hair+gender, hair+age, pose+gaze)
  - NEW: Landmark-based structural rules (jaw symmetry, eye symmetry)
  - Dropped: mutual_gender (binary male×female → replaced by continuous)
  - Dropped: mutual_smile_frown (smiling/frowning moved to expr, not in VLM)
  - Dropped: double_chin×narrow_jaw, square_face×narrow_jaw (narrow_jaw not in VLM,
             use landmark jaw_width_ratio instead)

22 training rules total (was 20).
"""

import logging
from typing import List, Tuple

import torch
import torch.nn as nn

from .refined_attributes import ci, COMBINED_FEATURE_NAMES, NUM_COMBINED_FEATURES

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
#  Training rule names (22 features — enter identity-causal graph)
# ──────────────────────────────────────────────────────────────────────────────
TRAINING_RULE_NAMES_V7 = [
    # -- Mutually exclusive / VLM conflicts (3) --
    'cr_mutual_mouth',           # mouth_open × mouth_closed
    'cr_lighting_conflict',      # bright_lighting × dim_lighting
    'cr_symmetry_conflict',      # symmetrical × asymmetrical

    # -- Expression-AU coherence (9) — now using fast extractor scores --
    'cr_happy_au6',              # |expr_happy - AU6|
    'cr_happy_au12',             # |expr_happy - AU12|
    'cr_surprise_au1au2',        # |expr_surprise - mean(AU1, AU2)|
    'cr_sad_au15',               # |expr_sad - AU15|
    'cr_angry_au4',              # |expr_angry - AU4|
    'cr_fear_au1au5',            # |expr_fear - mean(AU1, AU5)|
    'cr_disgust_au9',            # |expr_disgust - AU9|
    'cr_contempt_au14',          # |expr_contempt - AU14|
    'cr_neutral_any_au',         # expr_neutral × max(key AUs)

    # -- Cross-region identity mismatch (5) — NEW, key for deepfakes --
    # These capture face/context inconsistencies visible in scale=1.3 crops.
    'cr_gender_beard',           # female_score × beard (gender-facial_hair)
    'cr_gender_makeup',          # male_score × heavy_makeup
    'cr_hair_age',               # gray_hair × (1 - age_score) (gray but young)
    'cr_bald_longhair',          # bald × long_hair
    'cr_skin_tone_conflict',     # fair_skin × dark_skin

    # -- Skin/age coherence (2) --
    'cr_smooth_age',             # smooth_skin × high age_score
    'cr_wrinkle_age',            # wrinkled_skin × low age_score

    # -- Pose-gaze coherence (1) — NEW --
    'cr_gaze_lr_divergence',     # |left_gaze_x - right_gaze_x| (eyes should agree)

    # -- Landmark symmetry (2) — NEW, from fast features --
    'cr_eye_asymmetry',          # fs_eye_lr_symmetry (already computed, relay it)
    'cr_jaw_asymmetry',          # fs_jaw_symmetry

    # -- Image quality --
    'cr_image_quality',          # blurry × sharp
]

NUM_TRAINING_RULES_V7 = len(TRAINING_RULE_NAMES_V7)
assert NUM_TRAINING_RULES_V7 == 23, \
    f"Expected 23 training rules, got {NUM_TRAINING_RULES_V7}"


# ──────────────────────────────────────────────────────────────────────────────
#  Intervention rule definitions (v7 — updated for combined vector)
# ──────────────────────────────────────────────────────────────────────────────
INTERVENTION_RULE_DEFS_V7: List[Tuple[str, str, List[str], str]] = [
    # Gender → facial hair (using continuous gender_score)
    ('cr_inv_gender_beard',     'fs_gender_score', ['beard'],        'opposite'),
    ('cr_inv_gender_mustache',  'fs_gender_score', ['mustache'],     'opposite'),
    ('cr_inv_gender_stubble',   'fs_gender_score', ['stubble'],      'opposite'),
    ('cr_inv_gender_sideburns', 'fs_gender_score', ['sideburns'],    'opposite'),
    # Gender → makeup
    ('cr_inv_gender_makeup',    'fs_gender_score', ['heavy_makeup'], 'same'),
    ('cr_inv_gender_eyeliner',  'fs_gender_score', ['eyeliner'],     'same'),
    # Hair contradictions
    ('cr_inv_hair_bald_long',   'bald',            ['long_hair'],    'opposite'),
    # Age → appearance
    ('cr_inv_age_smooth_old',   'fs_age_score',    ['smooth_skin'],  'opposite'),
    ('cr_inv_age_wrinkle_young','fs_age_score',    ['wrinkled_skin'],'same'),
    ('cr_inv_age_gray_young',   'fs_age_score',    ['gray_hair'],    'same'),
    ('cr_inv_age_spots_young',  'fs_age_score',    ['age_spots'],    'same'),
    # Pose → gaze (head turn should match eye direction)
    ('cr_inv_pose_gaze',        'fs_pose_yaw',     ['fs_gaze_left_x', 'fs_gaze_right_x'], 'same'),
]


class CrossAttributeConsistencyRulesV7(nn.Module):
    """
    Compute 22 cross-attribute consistency violation scores from the
    combined 122-d [fast || vlm] feature vector.

    No trainable parameters. All operations are element-wise and
    fully differentiable for gradient flow.
    """

    def __init__(self):
        super().__init__()

        # -- VLM feature indices (in combined vector) --
        self._i_mouth_open = ci('mouth_open')
        self._i_mouth_closed = ci('mouth_closed')
        self._i_bright_lighting = ci('bright_lighting')
        self._i_dim_lighting = ci('dim_lighting')
        self._i_symmetrical = ci('symmetrical_face')
        self._i_asymmetrical = ci('asymmetrical_face')
        self._i_blurry = ci('blurry_image')
        self._i_sharp = ci('sharp_image')

        # Facial hair (VLM)
        self._i_beard = ci('beard')
        self._i_heavy_makeup = ci('heavy_makeup')
        self._i_gray_hair = ci('gray_hair')
        self._i_bald = ci('bald')
        self._i_long_hair = ci('long_hair')
        self._i_fair_skin = ci('fair_skin')
        self._i_dark_skin = ci('dark_skin')
        self._i_smooth_skin = ci('smooth_skin')
        self._i_wrinkled_skin = ci('wrinkled_skin')

        # -- Fast feature indices --
        self._i_gender = ci('fs_gender_score')
        self._i_age = ci('fs_age_score')
        self._i_expr_happy = ci('fs_expr_happy')
        self._i_expr_sad = ci('fs_expr_sad')
        self._i_expr_angry = ci('fs_expr_angry')
        self._i_expr_surprise = ci('fs_expr_surprise')
        self._i_expr_fear = ci('fs_expr_fear')
        self._i_expr_disgust = ci('fs_expr_disgust')
        self._i_expr_contempt = ci('fs_expr_contempt')
        self._i_expr_neutral = ci('fs_expr_neutral')
        self._i_au1 = ci('fs_AU1')
        self._i_au2 = ci('fs_AU2')
        self._i_au4 = ci('fs_AU4')
        self._i_au5 = ci('fs_AU5')
        self._i_au6 = ci('fs_AU6')
        self._i_au9 = ci('fs_AU9')
        self._i_au12 = ci('fs_AU12')
        self._i_au14 = ci('fs_AU14')
        self._i_au15 = ci('fs_AU15')
        self._i_gaze_lx = ci('fs_gaze_left_x')
        self._i_gaze_rx = ci('fs_gaze_right_x')
        self._i_eye_sym = ci('fs_eye_lr_symmetry')
        self._i_jaw_sym = ci('fs_jaw_symmetry')

        logger.info(
            f"[ConsistencyRulesV7] {NUM_TRAINING_RULES_V7} training rules, "
            f"{len(INTERVENTION_RULE_DEFS_V7)} intervention rules, "
            f"operating on {NUM_COMBINED_FEATURES}-d combined vector")

    def forward(self, attrs: torch.Tensor) -> torch.Tensor:
        """
        Compute training consistency violation scores.

        Args:
            attrs: (B, 122) combined [fast || vlm] feature vector
        Returns:
            (B, 22) violation scores
        """
        feats = []

        # -- Mutually exclusive / VLM conflicts (3) --
        feats.append(attrs[:, self._i_mouth_open] * attrs[:, self._i_mouth_closed])
        feats.append(attrs[:, self._i_bright_lighting] * attrs[:, self._i_dim_lighting])
        feats.append(attrs[:, self._i_symmetrical] * attrs[:, self._i_asymmetrical])

        # -- Expression-AU coherence (9) --
        # Now using fast extractor: expression = softmax [0,1], AU = intensity [0,1]
        feats.append(torch.abs(attrs[:, self._i_expr_happy] - attrs[:, self._i_au6]))
        feats.append(torch.abs(attrs[:, self._i_expr_happy] - attrs[:, self._i_au12]))
        feats.append(torch.abs(
            attrs[:, self._i_expr_surprise]
            - 0.5 * (attrs[:, self._i_au1] + attrs[:, self._i_au2])
        ))
        feats.append(torch.abs(attrs[:, self._i_expr_sad] - attrs[:, self._i_au15]))
        feats.append(torch.abs(attrs[:, self._i_expr_angry] - attrs[:, self._i_au4]))
        feats.append(torch.abs(
            attrs[:, self._i_expr_fear]
            - 0.5 * (attrs[:, self._i_au1] + attrs[:, self._i_au5])
        ))
        feats.append(torch.abs(attrs[:, self._i_expr_disgust] - attrs[:, self._i_au9]))
        feats.append(torch.abs(attrs[:, self._i_expr_contempt] - attrs[:, self._i_au14]))
        # Neutral should not have strong AUs
        key_aus = torch.stack([
            attrs[:, self._i_au6], attrs[:, self._i_au12],
            attrs[:, self._i_au1], attrs[:, self._i_au4],
            attrs[:, self._i_au15],
        ], dim=1)
        feats.append(attrs[:, self._i_expr_neutral] * key_aus.max(dim=1)[0])

        # -- Cross-region identity mismatch (5) --
        # gender_score: -1=female, +1=male. Convert to female_score for products.
        female_score = torch.clamp(0.5 - 0.5 * attrs[:, self._i_gender], 0, 1)
        male_score = torch.clamp(0.5 + 0.5 * attrs[:, self._i_gender], 0, 1)
        feats.append(female_score * attrs[:, self._i_beard])     # female + beard
        feats.append(male_score * attrs[:, self._i_heavy_makeup]) # male + makeup

        # Gray hair but young (low age_score)
        young_score = torch.clamp(1.0 - attrs[:, self._i_age], 0, 1)
        feats.append(attrs[:, self._i_gray_hair] * young_score)

        # Bald + long hair contradiction
        feats.append(attrs[:, self._i_bald] * attrs[:, self._i_long_hair])

        # Skin tone conflict
        feats.append(attrs[:, self._i_fair_skin] * attrs[:, self._i_dark_skin])

        # -- Skin/age coherence (2) --
        # Smooth skin but old
        old_score = torch.clamp(attrs[:, self._i_age], 0, 1)
        feats.append(attrs[:, self._i_smooth_skin] * old_score)
        # Wrinkled skin but young
        feats.append(attrs[:, self._i_wrinkled_skin] * young_score)

        # -- Pose-gaze coherence (1) --
        # Left and right eye gaze should agree (divergence = suspicious)
        feats.append(torch.abs(attrs[:, self._i_gaze_lx] - attrs[:, self._i_gaze_rx]))

        # -- Landmark symmetry relay (2) --
        # These are already computed by fast extractor but we relay them
        # as consistency features for the causal graph to use.
        feats.append(attrs[:, self._i_eye_sym])
        feats.append(attrs[:, self._i_jaw_sym])

        # -- Image quality --
        feats.append(attrs[:, self._i_blurry] * attrs[:, self._i_sharp])

        return torch.stack(feats, dim=1)  # (B, 22)

    @staticmethod
    def get_feature_names() -> List[str]:
        """Interpretable names for causal graph visualization."""
        return list(TRAINING_RULE_NAMES_V7)

    @staticmethod
    def get_intervention_rules() -> List[Tuple[str, str, List[str], str]]:
        """Return intervention rule definitions for CausalInterventionModule."""
        return list(INTERVENTION_RULE_DEFS_V7)

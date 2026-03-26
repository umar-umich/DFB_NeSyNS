"""
networks/nesy_defake/semantic/consistency_rules.py
===================================================
Cross-Attribute Consistency Rules for Deepfake Detection (v5, 2026-03-23).

Two categories of rules:

A. TRAINING RULES (20 features) — computed during forward pass, enter the
   identity-causal sub-graph as nodes. These fire often enough in training
   data to provide meaningful gradient signal.

B. INTERVENTION RULES (13 definitions) — too sparse during training (near-zero
   for >95% of samples because both constituent attributes are near-zero
   simultaneously). Instead, they define do-calculus interventions on the
   learned causal graph at inference time.

   Strong reason for the split: gender-specific rules like female×beard produce
   ~0 even on fakes because FaceBench correctly identifies the swapped face's
   gender. The violation is between face attributes and surrounding context
   (body/hair/clothing), which FaceBench doesn't capture. At inference,
   intervening on the gender node and observing causal cascades through the
   learned graph is far more powerful than checking a near-zero product.

All operations are element-wise — fully differentiable, no trainable parameters.
"""

import logging
from typing import List, Dict, Tuple

import torch
import torch.nn as nn

from .facial_semantic_extractor import FACEBENCH_ATTRIBUTES

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
#  Training rule names (20 features — enter identity-causal graph)
# ──────────────────────────────────────────────────────────────────────────────
TRAINING_RULE_NAMES = [
    # Mutually exclusive conflicts (4) — both high = LLaVA confused or manipulated
    'cr_mutual_mouth',           # mouth_open × mouth_closed
    'cr_mutual_gender',          # male × female
    'cr_mutual_smile_frown',     # smiling × frowning
    'cr_lighting_conflict',      # bright_lighting × dim_lighting
    # Expression-AU violations (5 original) — faceswap transfers expression
    # without correct muscle activations
    'cr_happy_au6',              # |happy - AU6_cheek_raise|
    'cr_happy_au12',             # |happy - AU12_lip_corner_puller|
    'cr_surprise_au1au2',        # |surprised - mean(AU1, AU2)|
    'cr_sad_au15',               # |sad - AU15_lip_corner_depressor|
    'cr_angry_au4',              # |angry - AU4_brow_lowerer|
    # Expression-AU violations (4 new) — extend to all basic emotions
    'cr_fear_au1au5',            # |fearful - mean(AU1, AU5)|
    'cr_disgust_au9',            # |disgusted - AU9_nose_wrinkler|
    'cr_contempt_au14',          # |contemptuous - AU14_dimpler|
    'cr_neutral_any_au',         # neutral × max(key AUs) — neutral shouldn't
                                 # have strong AU activations
    # Structural coherence (3 new) — bone structure contradictions
    'cr_double_chin_narrow_jaw', # double_chin × narrow_jaw
    'cr_square_face_narrow_jaw', # square_face × narrow_jaw
    'cr_round_face_pointed_chin',# round_face × pointed_chin
    # Skin coherence (2 new) — tone/age conflicts
    'cr_skin_tone_conflict',     # fair_skin × dark_skin (LLaVA confused by blending)
    'cr_skin_age_acne',          # acne × elderly_looking (rare in nature)
    # Symmetry/quality conflicts (2 original)
    'cr_symmetry_conflict',      # symmetrical × asymmetrical
    'cr_image_quality',          # blurry × sharp
]

NUM_TRAINING_RULES = len(TRAINING_RULE_NAMES)
assert NUM_TRAINING_RULES == 20, \
    f"Expected 20 training rules, got {NUM_TRAINING_RULES}"

# ──────────────────────────────────────────────────────────────────────────────
#  Intervention rule definitions (13 — used by CausalInterventionModule)
# ──────────────────────────────────────────────────────────────────────────────
# These are NOT computed as features. They define which attribute nodes to
# intervene on at inference time and what to observe.
# Format: (name, intervene_attr, observe_attrs, expected_direction)
#   expected_direction: 'same' = attributes should co-occur in reals,
#                       'opposite' = attributes should be mutually exclusive
INTERVENTION_RULE_DEFS: List[Tuple[str, str, List[str], str]] = [
    # Gender → facial hair (should not co-occur for females)
    ('cr_inv_gender_beard',      'female', ['beard'],        'opposite'),
    ('cr_inv_gender_mustache',   'female', ['mustache'],     'opposite'),
    ('cr_inv_gender_stubble',    'female', ['stubble'],      'opposite'),
    ('cr_inv_gender_sideburns',  'female', ['sideburns'],    'opposite'),
    # Gender → makeup (should not co-occur for males)
    ('cr_inv_gender_makeup',     'male',   ['heavy_makeup'], 'opposite'),
    ('cr_inv_gender_eyeliner',   'male',   ['eyeliner'],     'opposite'),
    # Gender → structure (statistical, not absolute)
    ('cr_inv_gender_strong_jaw', 'female', ['strong_jaw'],   'opposite'),
    # Contradictory hair
    ('cr_inv_hair_bald_long',    'bald',   ['long_hair'],    'opposite'),
    # Age → appearance (should not co-occur)
    ('cr_inv_age_smooth_elderly',   'elderly_looking', ['smooth_skin'],       'opposite'),
    ('cr_inv_age_wrinkle_young',    'young_looking',   ['wrinkled_skin'],     'opposite'),
    ('cr_inv_age_gray_young',       'young_looking',   ['gray_hair'],         'opposite'),
    ('cr_inv_age_receding_young',   'young_looking',   ['receding_hairline'], 'opposite'),
    ('cr_inv_age_spots_young',      'young_looking',   ['age_spots'],         'opposite'),
]

INTERVENTION_RULE_NAMES = [r[0] for r in INTERVENTION_RULE_DEFS]

# Legacy alias for backward compatibility
CONSISTENCY_RULE_NAMES = TRAINING_RULE_NAMES
NUM_CONSISTENCY_FEATURES = NUM_TRAINING_RULES


def _build_attr_index() -> Dict[str, int]:
    """Build name -> index mapping for FACEBENCH_ATTRIBUTES."""
    return {name: idx for idx, name in enumerate(FACEBENCH_ATTRIBUTES)}


class CrossAttributeConsistencyRules(nn.Module):
    """
    Compute 20 cross-attribute consistency violation scores (training rules).

    No trainable parameters. All operations are element-wise multiplications
    and absolute differences — fully differentiable for gradient flow.

    v5 changes (2026-03-23):
      - Removed 7 sparse rules (gender-facial_hair, age-appearance, bald+long)
        → moved to INTERVENTION_RULE_DEFS for inference-time do-calculus
      - Added 9 new rules: 4 expression-AU, 3 structural, 2 skin coherence
      - Total: 20 training rules (was 18)
    """

    def __init__(self):
        super().__init__()
        idx = _build_attr_index()

        # -- Mutually exclusive --
        self._i_mouth_open = idx['mouth_open']
        self._i_mouth_closed = idx['mouth_closed']
        self._i_male = idx['male']
        self._i_female = idx['female']
        self._i_smiling = idx['smiling']
        self._i_frowning = idx['frowning']
        self._i_bright_lighting = idx['bright_lighting']
        self._i_dim_lighting = idx['dim_lighting']

        # -- Expression-AU (original 5) --
        self._i_happy = idx['happy']
        self._i_au6 = idx['AU6_cheek_raise']
        self._i_au12 = idx['AU12_lip_corner_puller']
        self._i_surprised = idx['surprised']
        self._i_au1 = idx['AU1_inner_brow_raise']
        self._i_au2 = idx['AU2_outer_brow_raise']
        self._i_sad = idx['sad']
        self._i_au15 = idx['AU15_lip_corner_depressor']
        self._i_angry = idx['angry']
        self._i_au4 = idx['AU4_brow_lowerer']

        # -- Expression-AU (new 4) --
        self._i_fearful = idx['fearful']
        self._i_au5 = idx['AU5_upper_lid_raise']
        self._i_disgusted = idx['disgusted']
        self._i_au9 = idx['AU9_nose_wrinkler']
        self._i_contemptuous = idx['contemptuous']
        self._i_au14 = idx['AU14_dimpler']
        self._i_neutral = idx['neutral_expression']

        # -- Structural coherence (new 3) --
        self._i_double_chin = idx['double_chin']
        self._i_narrow_jaw = idx['narrow_jaw']
        self._i_square_face = idx['square_face']
        self._i_round_face = idx['round_face']
        self._i_pointed_chin = idx['pointed_chin']

        # -- Skin coherence (new 2) --
        self._i_fair_skin = idx['fair_skin']
        self._i_dark_skin = idx['dark_skin']
        self._i_acne = idx['acne']
        self._i_elderly = idx['elderly_looking']

        # -- Symmetry / quality (original 2) --
        self._i_symmetrical = idx['symmetrical_face']
        self._i_asymmetrical = idx['asymmetrical_face']
        self._i_blurry = idx['blurry_image']
        self._i_sharp = idx['sharp_image']

        logger.info(
            f"[ConsistencyRules] {NUM_TRAINING_RULES} training rules, "
            f"{len(INTERVENTION_RULE_DEFS)} intervention rules defined, "
            f"no trainable parameters")

    def forward(self, attrs: torch.Tensor) -> torch.Tensor:
        """
        Compute training consistency violation scores.

        Args:
            attrs: (B, 211) FaceBench attribute probabilities [0, 1]
        Returns:
            (B, 20) violation scores [0, 1]
        """
        feats = []

        # -- Mutually exclusive conflicts (4) --
        feats.append(attrs[:, self._i_mouth_open] * attrs[:, self._i_mouth_closed])
        feats.append(attrs[:, self._i_male] * attrs[:, self._i_female])
        feats.append(attrs[:, self._i_smiling] * attrs[:, self._i_frowning])
        feats.append(attrs[:, self._i_bright_lighting] * attrs[:, self._i_dim_lighting])

        # -- Expression-AU violations (5 original) --
        feats.append(torch.abs(attrs[:, self._i_happy] - attrs[:, self._i_au6]))
        feats.append(torch.abs(attrs[:, self._i_happy] - attrs[:, self._i_au12]))
        feats.append(torch.abs(
            attrs[:, self._i_surprised]
            - 0.5 * (attrs[:, self._i_au1] + attrs[:, self._i_au2])
        ))
        feats.append(torch.abs(attrs[:, self._i_sad] - attrs[:, self._i_au15]))
        feats.append(torch.abs(attrs[:, self._i_angry] - attrs[:, self._i_au4]))

        # -- Expression-AU violations (4 new) --
        # Fear: AU1 (inner brow raise) + AU5 (upper lid raise)
        feats.append(torch.abs(
            attrs[:, self._i_fearful]
            - 0.5 * (attrs[:, self._i_au1] + attrs[:, self._i_au5])
        ))
        # Disgust: AU9 (nose wrinkler)
        feats.append(torch.abs(attrs[:, self._i_disgusted] - attrs[:, self._i_au9]))
        # Contempt: AU14 (dimpler — unilateral lip corner)
        feats.append(torch.abs(attrs[:, self._i_contemptuous] - attrs[:, self._i_au14]))
        # Neutral should not have strong AUs
        key_aus = torch.stack([
            attrs[:, self._i_au6], attrs[:, self._i_au12],
            attrs[:, self._i_au1], attrs[:, self._i_au4],
            attrs[:, self._i_au15],
        ], dim=1)
        feats.append(attrs[:, self._i_neutral] * key_aus.max(dim=1)[0])

        # -- Structural coherence (3 new) --
        feats.append(attrs[:, self._i_double_chin] * attrs[:, self._i_narrow_jaw])
        feats.append(attrs[:, self._i_square_face] * attrs[:, self._i_narrow_jaw])
        feats.append(attrs[:, self._i_round_face] * attrs[:, self._i_pointed_chin])

        # -- Skin coherence (2 new) --
        feats.append(attrs[:, self._i_fair_skin] * attrs[:, self._i_dark_skin])
        feats.append(attrs[:, self._i_acne] * attrs[:, self._i_elderly])

        # -- Symmetry / quality conflicts (2) --
        feats.append(attrs[:, self._i_symmetrical] * attrs[:, self._i_asymmetrical])
        feats.append(attrs[:, self._i_blurry] * attrs[:, self._i_sharp])

        return torch.stack(feats, dim=1)  # (B, 20)

    @staticmethod
    def get_feature_names() -> List[str]:
        """Interpretable names for causal graph visualization."""
        return list(TRAINING_RULE_NAMES)

    @staticmethod
    def get_intervention_rules() -> List[Tuple[str, str, List[str], str]]:
        """Return intervention rule definitions for CausalInterventionModule."""
        return list(INTERVENTION_RULE_DEFS)

"""
networks/nesy_defake/semantic/consistency_rules.py
===================================================
Tier 1: Cross-Attribute Consistency Rules for Deepfake Detection.

Computes violation scores from the 211 FaceBench attributes that capture
forensic inconsistencies deepfakes introduce. These are deterministic,
differentiable functions with NO trainable parameters.

The key insight: deepfakes often create attribute combinations that violate
natural causal constraints (e.g., beard on a female face, happy expression
without the corresponding Action Units, elderly face with smooth skin).
These violations provide genuinely forensic signal for the causal graphs,
unlike the static identity attributes that are preserved by face manipulation.

Output: (B, 18) consistency violation scores appended to semantic vector.
"""

import logging
from typing import List, Dict

import torch
import torch.nn as nn

from .facial_semantic_extractor import FACEBENCH_ATTRIBUTES

logger = logging.getLogger(__name__)


# Feature names for interpretable causal graph nodes
CONSISTENCY_RULE_NAMES = [
    'cr_gender_beard',
    'cr_gender_mustache',
    'cr_gender_stubble',
    'cr_gender_makeup',
    'cr_mutual_mouth',
    'cr_mutual_gender',
    'cr_mutual_smile_frown',
    'cr_lighting_conflict',
    'cr_hair_bald_long',
    'cr_age_smooth_elderly',
    'cr_age_wrinkle_young',
    'cr_happy_au6',
    'cr_happy_au12',
    'cr_surprise_au1au2',
    'cr_sad_au15',
    'cr_angry_au4',
    'cr_symmetry_conflict',
    'cr_image_quality',
]

NUM_CONSISTENCY_FEATURES = len(CONSISTENCY_RULE_NAMES)


def _build_attr_index() -> Dict[str, int]:
    """Build name -> index mapping for FACEBENCH_ATTRIBUTES."""
    return {name: idx for idx, name in enumerate(FACEBENCH_ATTRIBUTES)}


class CrossAttributeConsistencyRules(nn.Module):
    """
    Compute 18 cross-attribute consistency violation scores.

    No trainable parameters. All operations are element-wise multiplications
    and absolute differences — fully differentiable for gradient flow.

    Categories:
      - Gender-attribute violations (4): beard/mustache/stubble on female, makeup on male
      - Mutually exclusive conflicts (4): mouth open+closed, male+female, smile+frown, lighting
      - Contradictory attributes (1): bald + long hair
      - Age-appearance inconsistency (2): elderly+smooth, young+wrinkled
      - Expression-AU violations (5): happy/surprise/sad/angry without corresponding AUs
      - Symmetry/quality conflicts (2): symmetric+asymmetric, blurry+sharp
    """

    def __init__(self):
        super().__init__()
        idx = _build_attr_index()

        # Store attribute indices as buffer (not parameters)
        # Gender-facial hair
        self._i_beard = idx['beard']
        self._i_mustache = idx['mustache']
        self._i_stubble = idx['stubble']
        self._i_male = idx['male']
        self._i_female = idx['female']
        self._i_heavy_makeup = idx['heavy_makeup']

        # Mutually exclusive
        self._i_mouth_open = idx['mouth_open']
        self._i_mouth_closed = idx['mouth_closed']
        self._i_smiling = idx['smiling']
        self._i_frowning = idx['frowning']
        self._i_bright_lighting = idx['bright_lighting']
        self._i_dim_lighting = idx['dim_lighting']

        # Contradictory
        self._i_bald = idx['bald']
        self._i_long_hair = idx['long_hair']

        # Age-appearance
        self._i_elderly = idx['elderly_looking']
        self._i_young = idx['young_looking']
        self._i_smooth_skin = idx['smooth_skin']
        self._i_wrinkled_skin = idx['wrinkled_skin']

        # Expression-AU
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

        # Symmetry / quality
        self._i_symmetrical = idx['symmetrical_face']
        self._i_asymmetrical = idx['asymmetrical_face']
        self._i_blurry = idx['blurry_image']
        self._i_sharp = idx['sharp_image']

        logger.info(
            f"[ConsistencyRules] {NUM_CONSISTENCY_FEATURES} rules, "
            f"no trainable parameters")

    def forward(self, attrs: torch.Tensor) -> torch.Tensor:
        """
        Compute consistency violation scores.

        Args:
            attrs: (B, 211) FaceBench attribute probabilities [0, 1]
        Returns:
            (B, 18) violation scores [0, 1]
        """
        feats = []

        # -- Gender-attribute violations (4) --
        # High score = suspicious combination
        feats.append(attrs[:, self._i_beard] * attrs[:, self._i_female])
        feats.append(attrs[:, self._i_mustache] * attrs[:, self._i_female])
        feats.append(attrs[:, self._i_stubble] * attrs[:, self._i_female])
        feats.append(
            attrs[:, self._i_heavy_makeup]
            * attrs[:, self._i_male]
            * (1.0 - attrs[:, self._i_female])
        )

        # -- Mutually exclusive conflicts (4) --
        # Both high simultaneously = model confused or manipulated
        feats.append(attrs[:, self._i_mouth_open] * attrs[:, self._i_mouth_closed])
        feats.append(attrs[:, self._i_male] * attrs[:, self._i_female])
        feats.append(attrs[:, self._i_smiling] * attrs[:, self._i_frowning])
        feats.append(attrs[:, self._i_bright_lighting] * attrs[:, self._i_dim_lighting])

        # -- Contradictory (1) --
        feats.append(attrs[:, self._i_bald] * attrs[:, self._i_long_hair])

        # -- Age-appearance inconsistency (2) --
        feats.append(attrs[:, self._i_elderly] * attrs[:, self._i_smooth_skin])
        feats.append(attrs[:, self._i_young] * attrs[:, self._i_wrinkled_skin])

        # -- Expression-AU violations (5) --
        # Duchenne smile: happy requires AU6 (cheek raise) + AU12 (lip corner pull)
        feats.append(torch.abs(attrs[:, self._i_happy] - attrs[:, self._i_au6]))
        feats.append(torch.abs(attrs[:, self._i_happy] - attrs[:, self._i_au12]))
        # Surprise: requires AU1 (inner brow raise) + AU2 (outer brow raise)
        feats.append(torch.abs(
            attrs[:, self._i_surprised]
            - 0.5 * (attrs[:, self._i_au1] + attrs[:, self._i_au2])
        ))
        # Sad: requires AU15 (lip corner depressor)
        feats.append(torch.abs(attrs[:, self._i_sad] - attrs[:, self._i_au15]))
        # Angry: requires AU4 (brow lowerer)
        feats.append(torch.abs(attrs[:, self._i_angry] - attrs[:, self._i_au4]))

        # -- Symmetry / quality conflicts (2) --
        feats.append(attrs[:, self._i_symmetrical] * attrs[:, self._i_asymmetrical])
        feats.append(attrs[:, self._i_blurry] * attrs[:, self._i_sharp])

        # Stack: (B, 18)
        return torch.stack(feats, dim=1)

    @staticmethod
    def get_feature_names() -> List[str]:
        """Interpretable names for causal graph visualization."""
        return list(CONSISTENCY_RULE_NAMES)

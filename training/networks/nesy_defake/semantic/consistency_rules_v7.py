"""
networks/nesy_defake/semantic/consistency_rules_v7.py
=====================================================
Cross-Attribute Consistency Rules — fast-only (VLM removed, 2026-04-20).

Operates on the 58-d fast feature vector. All VLM-dependent rules
(mouth_open/closed, lighting, skin_tone, gender↔beard, age↔wrinkles,
bald↔long_hair, image quality) were dropped when VLM features were
removed from the framework.

Remaining 12 rules cover:
  - Expression ↔ Action Unit coherence (9)
  - Left/right eye gaze agreement (1)
  - Landmark symmetry relay (2)
"""

import logging
from typing import List, Tuple

import torch
import torch.nn as nn

from .refined_attributes import ci, NUM_COMBINED_FEATURES

logger = logging.getLogger(__name__)


TRAINING_RULE_NAMES_V7 = [
    # Expression-AU coherence (9)
    'cr_happy_au6',              # |expr_happy - AU6|
    'cr_happy_au12',             # |expr_happy - AU12|
    'cr_surprise_au1au2',        # |expr_surprise - mean(AU1, AU2)|
    'cr_sad_au15',               # |expr_sad - AU15|
    'cr_angry_au4',              # |expr_angry - AU4|
    'cr_fear_au1au5',            # |expr_fear - mean(AU1, AU5)|
    'cr_disgust_au9',            # |expr_disgust - AU9|
    'cr_contempt_au14',          # |expr_contempt - AU14|
    'cr_neutral_any_au',         # expr_neutral × max(key AUs)

    # Pose-gaze coherence (1)
    'cr_gaze_lr_divergence',     # |left_gaze_x - right_gaze_x|

    # Landmark symmetry relay (2)
    'cr_eye_asymmetry',          # fs_eye_lr_symmetry
    'cr_jaw_asymmetry',          # fs_jaw_symmetry
]

NUM_TRAINING_RULES_V7 = len(TRAINING_RULE_NAMES_V7)
assert NUM_TRAINING_RULES_V7 == 12, \
    f"Expected 12 training rules, got {NUM_TRAINING_RULES_V7}"


# Intervention rule definitions (fast-only anchors)
INTERVENTION_RULE_DEFS_V7: List[Tuple[str, str, List[str], str]] = [
    ('cr_inv_pose_gaze', 'fs_pose_yaw',
     ['fs_gaze_left_x', 'fs_gaze_right_x'], 'same'),
]


class CrossAttributeConsistencyRulesV7(nn.Module):
    """
    Compute 12 cross-attribute consistency violation scores from the
    58-d fast feature vector.

    No trainable parameters; all ops are element-wise and differentiable.
    """

    def __init__(self):
        super().__init__()

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
            f"operating on {NUM_COMBINED_FEATURES}-d fast vector")

    def forward(self, attrs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attrs: (B, 58) fast feature vector
        Returns:
            (B, 12) violation scores
        """
        feats = []

        # Expression-AU coherence (9)
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
        key_aus = torch.stack([
            attrs[:, self._i_au6], attrs[:, self._i_au12],
            attrs[:, self._i_au1], attrs[:, self._i_au4],
            attrs[:, self._i_au15],
        ], dim=1)
        feats.append(attrs[:, self._i_expr_neutral] * key_aus.max(dim=1)[0])

        # Pose-gaze coherence (1)
        feats.append(torch.abs(
            attrs[:, self._i_gaze_lx] - attrs[:, self._i_gaze_rx]))

        # Landmark symmetry relay (2)
        feats.append(attrs[:, self._i_eye_sym])
        feats.append(attrs[:, self._i_jaw_sym])

        return torch.stack(feats, dim=1)  # (B, 12)

    @staticmethod
    def get_feature_names() -> List[str]:
        """Interpretable names for causal graph visualization."""
        return list(TRAINING_RULE_NAMES_V7)

    @staticmethod
    def get_intervention_rules() -> List[Tuple[str, str, List[str], str]]:
        """Return intervention rule definitions."""
        return list(INTERVENTION_RULE_DEFS_V7)

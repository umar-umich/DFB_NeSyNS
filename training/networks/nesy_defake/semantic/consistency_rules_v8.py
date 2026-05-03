"""
networks/nesy_defake/semantic/consistency_rules_v8.py
======================================================
Candidate consistency-rule set for the discriminative-gap selection
protocol (NeurIPS submission, 2026-05).

This module defines a *superset* of differentiable consistency
predicates from which a final retained set is selected by gap analysis
on a held-out slice of FF++ training data — see
``scripts/validate_predicates.py`` and
``configs/retained_predicates.yaml``.

Composition (28 candidates):
  - 12 predicates inherited from v7 (Expression-AU, gaze, symmetry).
  - 4 faceswap-specific predicates from README_april_28.md §5
    (yaw-gaze, pose-facewidth, lip-AU12, brow-eye-AU).
  - 12 FACS-grounded additions covering negative-emotion AU pairs,
    pitch/roll-gaze coherence, geometric ratios, bilateral symmetry,
    and Duchenne / genuine-expression AU clusters.

All predicates:
  - Operate on the 58-d fast-semantic substrate
    (``refined_attributes.FAST_FEATURE_NAMES``).
  - Are differentiable (abs / sigmoid / L2; no thresholding).
  - Output a violation score clamped to ``[0, 1]`` where higher = more
    inconsistent. The ``forward`` pass returns ``(B, 28)``.

The retained subset is selected post-hoc by
``scripts/validate_predicates.py`` and frozen in
``configs/retained_predicates.yaml``. After that point the symbolic
stream loads only the retained subset; all retained-set decisions are
made BEFORE any test or cross-dataset evaluation.
"""

import logging
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .refined_attributes import ci, NUM_COMBINED_FEATURES

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Candidate predicate names — order matches forward() output columns
# ═══════════════════════════════════════════════════════════════════════

CANDIDATE_RULE_NAMES_V8: List[str] = [
    # ── v7 inherited (12) ───────────────────────────────────────────────
    'cr_happy_au6',                # |happy - AU6|
    'cr_happy_au12',               # |happy - AU12|
    'cr_surprise_au1au2',          # |surprise - mean(AU1, AU2)|
    'cr_sad_au15',                 # |sad - AU15|
    'cr_angry_au4',                # |angry - AU4|
    'cr_fear_au1au5',              # |fear - mean(AU1, AU5)|
    'cr_disgust_au9',              # |disgust - AU9|
    'cr_contempt_au14',            # |contempt - AU14|
    'cr_neutral_any_au',           # neutral × max(key AUs)
    'cr_gaze_lr_divergence',       # |left_gaze_x - right_gaze_x|
    'cr_eye_asymmetry',            # fs_eye_lr_symmetry passthrough
    'cr_jaw_asymmetry',            # fs_jaw_symmetry passthrough

    # ── README §5 faceswap-specific (4) ────────────────────────────────
    'cr_yaw_gaze_misalign',        # |yaw - mean_gaze_x|
    'cr_pose_facewidth',           # face W/H should covary with |yaw|
    'cr_lip_geometry_au12',        # mouth-aspect ratio vs AU12 (smile)
    'cr_brow_eye_couple',          # brow-height covaries with AU1/AU4

    # ── FACS-grounded additions (12) ───────────────────────────────────
    # Negative-emotion AU coherence (3)
    'cr_sad_au4',                  # sadness should also fire AU4
    'cr_angry_au7',                # anger fires AU7 (lid tightener)
    'cr_disgust_au17',             # disgust fires AU17 (chin raiser)

    # Refined positive / surprise AU coherence (2)
    'cr_surprise_au2',             # surprise → AU2 outer-brow raiser
    'cr_sad_au1',                  # sadness → AU1 inner-brow raiser

    # Pose-gaze (additional axes) (2)
    'cr_pitch_gaze_y',             # |pitch - mean_gaze_y|
    'cr_roll_eye_sym',             # roll should track eye_lr_symmetry

    # Geometric ratios (2)
    'cr_ipd_facewidth',            # interpupillary / face_width — bounded
    'cr_jaw_cheekbone',            # jaw_width vs face_width_height

    # Bilateral symmetry (1)
    'cr_mouth_symmetry',           # fs_mouth_symmetry passthrough

    # AU pair coherence — Duchenne / genuine clusters (2)
    'cr_duchenne_smile',           # |AU6 - AU12| under high happy
    'cr_genuine_surprise',         # |AU1 - AU2| under high surprise
]

NUM_CANDIDATE_RULES_V8 = len(CANDIDATE_RULE_NAMES_V8)
assert NUM_CANDIDATE_RULES_V8 == 28, \
    f"Expected 28 candidate rules, got {NUM_CANDIDATE_RULES_V8}"


# Categories used by the validation script's plotting / table layer
RULE_CATEGORIES_V8: dict = {
    # v7 inherited
    'cr_happy_au6':            'expr-au',
    'cr_happy_au12':           'expr-au',
    'cr_surprise_au1au2':      'expr-au',
    'cr_sad_au15':             'expr-au',
    'cr_angry_au4':            'expr-au',
    'cr_fear_au1au5':          'expr-au',
    'cr_disgust_au9':          'expr-au',
    'cr_contempt_au14':        'expr-au',
    'cr_neutral_any_au':       'expr-au',
    'cr_gaze_lr_divergence':   'pose-gaze',
    'cr_eye_asymmetry':        'symmetry',
    'cr_jaw_asymmetry':        'symmetry',
    # README §5
    'cr_yaw_gaze_misalign':    'pose-gaze',
    'cr_pose_facewidth':       'geometry',
    'cr_lip_geometry_au12':    'expr-au',
    'cr_brow_eye_couple':      'expr-au',
    # FACS additions
    'cr_sad_au4':              'expr-au',
    'cr_angry_au7':            'expr-au',
    'cr_disgust_au17':         'expr-au',
    'cr_surprise_au2':         'expr-au',
    'cr_sad_au1':              'expr-au',
    'cr_pitch_gaze_y':         'pose-gaze',
    'cr_roll_eye_sym':         'pose-gaze',
    'cr_ipd_facewidth':        'geometry',
    'cr_jaw_cheekbone':        'geometry',
    'cr_mouth_symmetry':       'symmetry',
    'cr_duchenne_smile':       'au-pair',
    'cr_genuine_surprise':     'au-pair',
}


# Optional intervention rules (carried forward from v7)
INTERVENTION_RULE_DEFS_V8: List[Tuple[str, str, List[str], str]] = [
    ('cr_inv_pose_gaze', 'fs_pose_yaw',
     ['fs_gaze_left_x', 'fs_gaze_right_x'], 'same'),
]


def _clip01(x: torch.Tensor) -> torch.Tensor:
    """Clamp to [0, 1] without breaking autograd."""
    return torch.clamp(x, 0.0, 1.0)


class CrossAttributeConsistencyRulesV8(nn.Module):
    """Compute 28 candidate consistency violation scores.

    No trainable parameters; all ops are element-wise differentiable.
    The ``forward`` returns a ``(B, 28)`` tensor with values in [0, 1]
    where higher = more inconsistent under the rule.
    """

    def __init__(self):
        super().__init__()

        # --- expression / AU indices ---
        self._i_expr_neutral  = ci('fs_expr_neutral')
        self._i_expr_happy    = ci('fs_expr_happy')
        self._i_expr_sad      = ci('fs_expr_sad')
        self._i_expr_angry    = ci('fs_expr_angry')
        self._i_expr_surprise = ci('fs_expr_surprise')
        self._i_expr_fear     = ci('fs_expr_fear')
        self._i_expr_disgust  = ci('fs_expr_disgust')
        self._i_expr_contempt = ci('fs_expr_contempt')

        self._i_au1  = ci('fs_AU1')
        self._i_au2  = ci('fs_AU2')
        self._i_au4  = ci('fs_AU4')
        self._i_au5  = ci('fs_AU5')
        self._i_au6  = ci('fs_AU6')
        self._i_au7  = ci('fs_AU7')
        self._i_au9  = ci('fs_AU9')
        self._i_au12 = ci('fs_AU12')
        self._i_au14 = ci('fs_AU14')
        self._i_au15 = ci('fs_AU15')
        self._i_au17 = ci('fs_AU17')

        # --- pose / gaze ---
        self._i_yaw    = ci('fs_pose_yaw')
        self._i_pitch  = ci('fs_pose_pitch')
        self._i_roll   = ci('fs_pose_roll')
        self._i_gaze_lx = ci('fs_gaze_left_x')
        self._i_gaze_rx = ci('fs_gaze_right_x')
        self._i_gaze_ly = ci('fs_gaze_left_y')
        self._i_gaze_ry = ci('fs_gaze_right_y')

        # --- geometry ---
        self._i_face_wh    = ci('fs_face_width_height_ratio')
        self._i_jaw_w      = ci('fs_jaw_width_ratio')
        self._i_nose_w     = ci('fs_nose_width_ratio')
        self._i_ipd        = ci('fs_interpupillary_ratio')
        self._i_brow_h     = ci('fs_brow_height_ratio')
        self._i_mouth_ar   = ci('fs_mouth_aspect_ratio')
        self._i_mouth_nose = ci('fs_mouth_nose_ratio')

        # --- symmetry ---
        self._i_eye_sym    = ci('fs_eye_lr_symmetry')
        self._i_mouth_sym  = ci('fs_mouth_symmetry')
        self._i_jaw_sym    = ci('fs_jaw_symmetry')

        logger.info(
            f"[ConsistencyRulesV8] {NUM_CANDIDATE_RULES_V8} candidate "
            f"rules, {len(INTERVENTION_RULE_DEFS_V8)} intervention rules; "
            f"input substrate {NUM_COMBINED_FEATURES}-d.")

    # ── helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _coupling_violation(a: torch.Tensor, b: torch.Tensor,
                            anchor: torch.Tensor) -> torch.Tensor:
        """``|a - b| · anchor`` — higher when (a, b) disagree under high
        anchor signal. Clamped to [0, 1].
        """
        return _clip01(torch.abs(a - b) * _clip01(anchor))

    @staticmethod
    def _abs_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return _clip01(torch.abs(a - b))

    # ── main ────────────────────────────────────────────────────────────

    def forward(self, attrs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attrs: (B, 58) fast feature vector (expression / AU values
                already softmaxed / sigmoided into [0, 1]; pose & gaze
                are normalised to [-1, 1]).
        Returns:
            (B, 28) violation scores in [0, 1].
        """
        feats = []

        # ── 1-12: v7 inherited ──────────────────────────────────────────
        feats.append(self._abs_diff(attrs[:, self._i_expr_happy],
                                    attrs[:, self._i_au6]))                       # 1
        feats.append(self._abs_diff(attrs[:, self._i_expr_happy],
                                    attrs[:, self._i_au12]))                      # 2
        feats.append(self._abs_diff(
            attrs[:, self._i_expr_surprise],
            0.5 * (attrs[:, self._i_au1] + attrs[:, self._i_au2])))                # 3
        feats.append(self._abs_diff(attrs[:, self._i_expr_sad],
                                    attrs[:, self._i_au15]))                       # 4
        feats.append(self._abs_diff(attrs[:, self._i_expr_angry],
                                    attrs[:, self._i_au4]))                        # 5
        feats.append(self._abs_diff(
            attrs[:, self._i_expr_fear],
            0.5 * (attrs[:, self._i_au1] + attrs[:, self._i_au5])))                # 6
        feats.append(self._abs_diff(attrs[:, self._i_expr_disgust],
                                    attrs[:, self._i_au9]))                        # 7
        feats.append(self._abs_diff(attrs[:, self._i_expr_contempt],
                                    attrs[:, self._i_au14]))                       # 8
        key_aus = torch.stack([
            attrs[:, self._i_au6],  attrs[:, self._i_au12],
            attrs[:, self._i_au1],  attrs[:, self._i_au4],
            attrs[:, self._i_au15],
        ], dim=1)
        feats.append(_clip01(
            attrs[:, self._i_expr_neutral] * key_aus.max(dim=1)[0]))               # 9
        feats.append(_clip01(0.5 * torch.abs(
            attrs[:, self._i_gaze_lx] - attrs[:, self._i_gaze_rx])))               # 10
        feats.append(_clip01(attrs[:, self._i_eye_sym]))                           # 11
        feats.append(_clip01(attrs[:, self._i_jaw_sym]))                           # 12

        # ── 13-16: README §5 faceswap-specific ──────────────────────────
        # cr_yaw_gaze_misalign: a swap transplants source-face gaze onto
        # host head pose; on reals, head yaw and gaze direction co-vary.
        # Both yaw and gaze_x live on roughly comparable scales after
        # normalisation, so a direct |yaw - mean_gaze_x| is the score.
        mean_gaze_x = 0.5 * (attrs[:, self._i_gaze_lx]
                             + attrs[:, self._i_gaze_rx])
        feats.append(_clip01(0.5 * torch.abs(
            attrs[:, self._i_yaw] - mean_gaze_x)))                                 # 13

        # cr_pose_facewidth: face W/H ratio depends on yaw (faces look
        # narrower in profile). Faceswap flattens this dependency. The
        # violation is "how far the W/H is from what yaw predicts" —
        # we use a simple linear expectation: |W/H - (1 - 0.4·|yaw|)|.
        expected_wh = 1.0 - 0.4 * torch.abs(attrs[:, self._i_yaw])
        feats.append(_clip01(torch.abs(
            attrs[:, self._i_face_wh] - expected_wh)))                             # 14

        # cr_lip_geometry_au12: AU12 (smile) raises mouth corners and
        # widens the mouth → mouth_aspect_ratio should track AU12.
        feats.append(self._abs_diff(attrs[:, self._i_mouth_ar],
                                    attrs[:, self._i_au12]))                       # 15

        # cr_brow_eye_couple: brow_height should track AU1 (raise) - AU4
        # (lower). Texture-based swaps decouple them.
        brow_signed = attrs[:, self._i_au1] - attrs[:, self._i_au4]
        feats.append(_clip01(torch.abs(
            attrs[:, self._i_brow_h] - brow_signed)))                              # 16

        # ── 17-19: negative-emotion AU coherence ────────────────────────
        # Sadness fires both AU15 (lip-corner depressor) and AU4
        # (brow-lowerer); we already have AU15 in #4, here pair sadness
        # with AU4.
        feats.append(self._abs_diff(attrs[:, self._i_expr_sad],
                                    attrs[:, self._i_au4]))                        # 17
        # Anger commonly co-fires AU4 + AU7 (lid tightener); cover AU7.
        feats.append(self._abs_diff(attrs[:, self._i_expr_angry],
                                    attrs[:, self._i_au7]))                        # 18
        # Disgust + AU17 (chin-raiser)
        feats.append(self._abs_diff(attrs[:, self._i_expr_disgust],
                                    attrs[:, self._i_au17]))                       # 19

        # ── 20-21: refined positive / surprise AU coherence ─────────────
        # Surprise → AU2 (outer-brow raiser). Already paired w/ AU1+AU2
        # average in #3; this isolates AU2 alone for redundancy check.
        feats.append(self._abs_diff(attrs[:, self._i_expr_surprise],
                                    attrs[:, self._i_au2]))                        # 20
        # Sadness → AU1 (inner-brow raiser).
        feats.append(self._abs_diff(attrs[:, self._i_expr_sad],
                                    attrs[:, self._i_au1]))                        # 21

        # ── 22-23: pose-gaze (additional axes) ──────────────────────────
        # Pitch ↔ vertical gaze should covary on real faces.
        mean_gaze_y = 0.5 * (attrs[:, self._i_gaze_ly]
                             + attrs[:, self._i_gaze_ry])
        feats.append(_clip01(0.5 * torch.abs(
            attrs[:, self._i_pitch] - mean_gaze_y)))                               # 22
        # Head roll should leave eye_lr_symmetry small — tilt rotates the
        # eyeline but a sane geometry SHOULD measure low LR-symmetry mismatch
        # under controlled roll. Faceswaps with non-rigid blending often
        # leave residual asymmetry uncorrelated with roll. Score:
        # |eye_sym - |roll||  (high when symmetry doesn't track roll).
        feats.append(_clip01(torch.abs(
            attrs[:, self._i_eye_sym] - torch.abs(attrs[:, self._i_roll]))))       # 23

        # ── 24-25: geometric ratios ─────────────────────────────────────
        # Inter-pupillary distance vs face width: roughly constant ~0.45
        # on adult faces; deviation flags geometric distortion.
        feats.append(_clip01(torch.abs(
            attrs[:, self._i_ipd] - 0.45)))                                        # 24
        # Jaw vs cheekbone (face W/H proxy): jaw_width should track
        # face_width_height in a near-linear band on reals.
        feats.append(_clip01(torch.abs(
            attrs[:, self._i_jaw_w] - attrs[:, self._i_face_wh])))                 # 25

        # ── 26: bilateral mouth symmetry passthrough ───────────────────
        feats.append(_clip01(attrs[:, self._i_mouth_sym]))                         # 26

        # ── 27-28: Duchenne / genuine-expression AU pairs ──────────────
        # Duchenne smile: AU6 (cheek-raiser) AND AU12 (lip corner) co-fire.
        # Score is |AU6 - AU12| weighted by happy intensity.
        feats.append(self._coupling_violation(
            attrs[:, self._i_au6], attrs[:, self._i_au12],
            attrs[:, self._i_expr_happy]))                                          # 27
        # Genuine surprise: AU1 + AU2 co-fire under high surprise.
        feats.append(self._coupling_violation(
            attrs[:, self._i_au1], attrs[:, self._i_au2],
            attrs[:, self._i_expr_surprise]))                                       # 28

        return torch.stack(feats, dim=1)  # (B, 28)

    @staticmethod
    def get_feature_names() -> List[str]:
        return list(CANDIDATE_RULE_NAMES_V8)

    @staticmethod
    def get_categories() -> dict:
        return dict(RULE_CATEGORIES_V8)

    @staticmethod
    def get_intervention_rules() -> List[Tuple[str, str, List[str], str]]:
        return list(INTERVENTION_RULE_DEFS_V8)


# ═══════════════════════════════════════════════════════════════════════
#  Retained-set wrapper — reads configs/retained_predicates.yaml
# ═══════════════════════════════════════════════════════════════════════

class RetainedConsistencyRules(nn.Module):
    """Apply the v8 candidate set, then index out only the retained
    predicates listed in ``configs/retained_predicates.yaml``.

    The retained set was selected by ``scripts/validate_predicates.py``
    on the FF++ training-fold val slice (10 %, video-level, seed=42)
    and is frozen by the YAML. This wrapper enforces that the runtime
    output exactly matches the frozen choice — see
    ``RetainedConsistencyRules.verify_against_yaml``.

    The output shape is ``(B, k)`` where ``k = retained_count`` from
    the YAML.
    """

    def __init__(self, yaml_path: str):
        super().__init__()
        self.yaml_path = yaml_path
        spec = self._load_yaml(yaml_path)
        self._validate_yaml_against_v8(spec)

        self._retained_indices = [int(e['index']) for e in spec['retained_predicates']]
        self._retained_names = [str(e['name']) for e in spec['retained_predicates']]
        self.k = len(self._retained_indices)
        self.spec = spec

        self.full = CrossAttributeConsistencyRulesV8()
        # Register as buffer so it moves with .to(device) and gets saved
        # in state_dict for traceability.
        self.register_buffer(
            'retained_idx',
            torch.tensor(self._retained_indices, dtype=torch.long),
            persistent=True,
        )
        logger.info(
            f"[RetainedConsistencyRules] loaded {self.k} retained "
            f"predicates from {yaml_path}; rule={spec.get('selection_rule')}, "
            f"k={spec.get('selection_k')}.")

    @staticmethod
    def _load_yaml(path: str) -> dict:
        try:
            import yaml
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                'PyYAML required to load the retained-predicate config') from e
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'Retained predicates YAML not found: {path}')
        with open(path) as f:
            return yaml.safe_load(f)

    @staticmethod
    def _validate_yaml_against_v8(spec: dict) -> None:
        if spec.get('candidate_count') != NUM_CANDIDATE_RULES_V8:
            raise ValueError(
                f"YAML candidate_count={spec.get('candidate_count')} "
                f"≠ v8 module count {NUM_CANDIDATE_RULES_V8}")
        retained = spec.get('retained_predicates') or []
        if len(retained) != int(spec.get('retained_count', -1)):
            raise ValueError(
                'YAML retained_count does not match retained_predicates list')
        for entry in retained:
            idx = int(entry['index'])
            name = str(entry['name'])
            if not (0 <= idx < NUM_CANDIDATE_RULES_V8):
                raise ValueError(f'YAML index {idx} out of range')
            if CANDIDATE_RULE_NAMES_V8[idx] != name:
                raise ValueError(
                    f'YAML mismatch: index {idx} is '
                    f'{CANDIDATE_RULE_NAMES_V8[idx]!r} in v8 but YAML says '
                    f'{name!r}. Frozen retained set is out of sync with the '
                    f'v8 candidate module.')

    def verify_against_yaml(self) -> None:
        """Re-validate at runtime that the loaded retained set still
        matches the YAML and the v8 candidate module. Call this at
        training-script entry per the predicate-selection protocol.
        """
        # Re-run YAML validation in case the file changed since __init__
        self._validate_yaml_against_v8(self._load_yaml(self.yaml_path))
        # Also guard against accidental in-process mutation
        idx_now = self.retained_idx.tolist()
        names_now = [CANDIDATE_RULE_NAMES_V8[i] for i in idx_now]
        if names_now != self._retained_names:
            raise RuntimeError(
                'Retained predicate set drift detected at runtime. '
                'Names from idx buffer no longer match yaml load. '
                'Refusing to start training/eval.')

    def forward(self, attrs: torch.Tensor) -> torch.Tensor:
        full = self.full(attrs)                  # (B, 28)
        return full.index_select(1, self.retained_idx)  # (B, k)

    def get_feature_names(self) -> List[str]:
        return list(self._retained_names)


def load_retained_rules(
    yaml_path: Optional[str] = None,
) -> RetainedConsistencyRules:
    """Construct ``RetainedConsistencyRules`` from the default repo path
    if ``yaml_path`` is None.
    """
    if yaml_path is None:
        # Default: <repo_root>/configs/retained_predicates.yaml
        # __file__ = .../training/networks/nesy_defake/semantic/consistency_rules_v8.py
        here = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(here, '..', '..', '..', '..'))
        yaml_path = os.path.join(repo_root, 'configs',
                                 'retained_predicates.yaml')
    return RetainedConsistencyRules(yaml_path)

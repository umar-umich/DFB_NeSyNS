"""
networks/nesy_defake/concept_branch.py
======================================
Lightweight neuro-symbolic branch for ablation-based evidence fusion.

ConceptBranch (Ablation 3):
  58-d fast features -> consistency rules (12/18-d) -> MLP -> 2-d evidence

Produces a non-negative evidence vector that is fused with the spatial
branch's evidence via gated addition in the detector.

NOTE: SimplifiedCausalBranch (the legacy Ablation-4 'simple' linear SCM) was
ARCHIVED to attic/training/networks/nesy_defake/simplified_causal_branch.py.
The active causal branches are ImprovedCausalBranch ('improved_scm') and
CausalConstraintVerificationBranch ('ccv'); see causal_branch_factory.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .semantic.consistency_rules_v7 import CrossAttributeConsistencyRulesV7
from .semantic.consistency_rules_v8 import (
    RetainedConsistencyRules,
    load_retained_rules,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Ablation 3: Concept Evidence Branch
# ═══════════════════════════════════════════════════════════════════════════

class ConceptBranch(nn.Module):
    """
    Concept evidence branch operating on fast semantic features.

    Pipeline:
      combined_features (B, 58)
        -> CrossAttributeConsistencyRulesV7 -> violations (B, 12)
        -> concat [combined || violations] = (B, 70)
        -> LayerNorm -> Linear -> GELU -> Dropout -> Linear -> logits (B, 2)
        -> softplus -> evidence (B, 2)

    No connection to CLIP spatial features — fully independent signal.
    """

    def __init__(
        self,
        combined_dim: int = 58,
        rules_dim: int = 12,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.2,
        consistency_rules_version: str = 'v7',
        retained_predicates_yaml: str = None,
        substrate_mode: str = 'both',
    ):
        super().__init__()
        # substrate_mode controls what feeds the concept MLP:
        #   'both'           [x_sem || violations]  (default; input_dim = combined+rules)
        #   'rules_only'     violations only        (input_dim = rules)
        #   'substrate_only' x_sem only             (input_dim = combined). Violations
        #                    are still computed and returned (for logging / SCM), just
        #                    excluded from the MLP input.
        if substrate_mode not in ('both', 'rules_only', 'substrate_only'):
            raise ValueError(
                f"Unknown substrate_mode: {substrate_mode!r} "
                f"(expected 'both', 'rules_only', or 'substrate_only')")
        self.substrate_mode = substrate_mode
        # ── Predicate module selection ────────────────────────────────
        # 'v7'           — legacy 12 predicates (default, preserves
        #                   existing checkpoint dimensionality).
        # 'v8_retained'  — 18 predicates from the gap-selection protocol;
        #                   reads configs/retained_predicates.yaml. The
        #                   training script must call .verify_against_yaml()
        #                   at startup before the first forward.
        self.consistency_rules_version = consistency_rules_version
        if consistency_rules_version == 'v8_retained':
            self.consistency_rules = (
                RetainedConsistencyRules(retained_predicates_yaml)
                if retained_predicates_yaml
                else load_retained_rules())
            actual_rules_dim = self.consistency_rules.k
            if rules_dim != actual_rules_dim:
                logger.warning(
                    f"[ConceptBranch] config rules_dim={rules_dim} "
                    f"overridden to {actual_rules_dim} (frozen retained set)")
            rules_dim = actual_rules_dim
        elif consistency_rules_version == 'v7':
            self.consistency_rules = CrossAttributeConsistencyRulesV7()
        else:
            raise ValueError(
                f"Unknown consistency_rules_version: "
                f"{consistency_rules_version!r} "
                f"(expected 'v7' or 'v8_retained')")

        self._combined_dim = combined_dim
        self._rules_dim = rules_dim
        if substrate_mode == 'both':
            input_dim = combined_dim + rules_dim
        elif substrate_mode == 'rules_only':
            input_dim = rules_dim
        else:  # substrate_only
            input_dim = combined_dim
        self.concept_mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        nn.init.normal_(self.concept_mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.concept_mlp[-1].bias)

        logger.info(
            f"[ConceptBranch] {input_dim}-d input "
            f"(substrate_mode={substrate_mode}: {combined_dim} combined + "
            f"{rules_dim} rules, version={consistency_rules_version}) "
            f"-> {hidden_dim} hidden -> {num_classes} evidence")

    def forward(self, combined_features: torch.Tensor,
                predicate_mask: torch.Tensor = None) -> dict:
        """
        Args:
            combined_features: (B, 58) fast feature vector
            predicate_mask:    optional (B, K) or (1, K) tensor in {0, 1}.
                When supplied, violations are element-wise multiplied by it
                BEFORE concatenation with the substrate. Used by the
                faithfulness runner to zero specific retained predicates
                and observe the change in symbolic-stream evidence.
                ``None`` (default) is identical to the original forward.
        Returns:
            dict with 'logits', 'evidence', 'violations', 'concept_input'
        """
        violations = self.consistency_rules(combined_features)        # (B, K)
        if predicate_mask is not None:
            # Broadcast (1, K) to (B, K) if needed; same dtype as violations.
            # Applied in ALL substrate_modes so the returned violations (used by
            # logging / the SCM) reflect the intervention. In 'substrate_only'
            # they are not part of the MLP input, so masking has no effect on
            # evidence — but the interface stays consistent.
            mask = predicate_mask.to(
                dtype=violations.dtype, device=violations.device)
            violations = violations * mask
        # Build the MLP input per substrate_mode.
        if self.substrate_mode == 'both':
            concept_input = torch.cat(
                [combined_features, violations], dim=1)               # (B, 58+K)
        elif self.substrate_mode == 'rules_only':
            concept_input = violations                                # (B, K)
        else:  # substrate_only
            concept_input = combined_features                         # (B, 58)
        logits = self.concept_mlp(concept_input)                      # (B, 2)
        evidence = F.softplus(logits)                                 # (B, 2)

        return {
            'logits': logits,
            'evidence': evidence,
            'violations': violations,
            'concept_input': concept_input,
        }


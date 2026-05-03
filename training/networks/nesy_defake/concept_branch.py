"""
networks/nesy_defake/concept_branch.py
======================================
Lightweight neuro-symbolic branches for ablation-based evidence fusion.

ConceptBranch (Ablation 3):
  58-d fast features -> consistency rules v7 (12-d) -> MLP -> 2-d evidence

SimplifiedCausalBranch (Ablation 4):
  Spatial-only linear SCMs, identity (z32 + curated26 + rules12 = 70) +
  forensic (z32 + 83 = 115) sub-graphs -> differential residuals
  -> MLP -> 2-d evidence.

Both branches produce non-negative evidence vectors that are fused with
the spatial branch's evidence via gated addition in the detector.
"""

import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .semantic.consistency_rules_v7 import CrossAttributeConsistencyRulesV7
from .semantic.consistency_rules_v8 import (
    RetainedConsistencyRules,
    load_retained_rules,
)
from .semantic.refined_attributes import (
    CAUSAL_ATTRIBUTE_INDICES,
    NUM_CAUSAL_ATTRIBUTES,
    NUM_COMBINED_FEATURES,
)
from .causal.causal_discovery import LinearSCM, dagma_acyclicity

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
    ):
        super().__init__()
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

        input_dim = combined_dim + rules_dim
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
            f"({combined_dim} combined + {rules_dim} rules, "
            f"version={consistency_rules_version}) "
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
            mask = predicate_mask.to(
                dtype=violations.dtype, device=violations.device)
            violations = violations * mask
        concept_input = torch.cat(
            [combined_features, violations], dim=1)                   # (B, 58+K)
        logits = self.concept_mlp(concept_input)                      # (B, 2)
        evidence = F.softplus(logits)                                 # (B, 2)

        return {
            'logits': logits,
            'evidence': evidence,
            'violations': violations,
            'concept_input': concept_input,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Ablation 4: Simplified Causal Evidence Branch
# ═══════════════════════════════════════════════════════════════════════════

class SimplifiedCausalBranch(nn.Module):
    """
    Simplified causal branch: spatial-only, linear SCM, no SAE, no frequency.

    Sub-graphs:
      Identity:  z_causal(32) + curated(26) + rules(12) = 70 nodes
      Forensic:  z_causal(32) + forensic(83) = 115 nodes

    Real/fake SCM pairs learn different causal structures.
    Differential residuals (fake_residual - real_residual) encode what
    the manipulation "breaks" — this is the detection signal.

    Pipeline:
      spatial_raw (B, 1024) --detach--> compressor -> z (B, 32)
      x_identity = [z || curated || rules]  (B, 70)
      x_forensic = [z || forensic]          (B, 115)
      -> 4 LinearSCMs (id_real, id_fake, for_real, for_fake)
      -> differential residuals (B, 185)
      -> MLP -> logits (B, 2) -> softplus -> evidence (B, 2)
    """

    def __init__(
        self,
        backbone_dim: int = 1024,
        z_causal_dim: int = 32,
        curated_dim: int = 26,
        rules_dim: int = 12,
        forensic_dim: int = 83,
        hidden_dim: int = 64,
        num_classes: int = 2,
        sparsity_penalty: float = 0.01,
    ):
        super().__init__()
        self.z_causal_dim = z_causal_dim
        self.sparsity_penalty = sparsity_penalty

        # Curated attribute indices into the 58-d fast feature vector
        self._curated_indices = CAUSAL_ATTRIBUTE_INDICES

        # Compress spatial features -> z_causal (detached from CLIP)
        self.compressor = nn.Linear(backbone_dim, z_causal_dim, bias=False)
        nn.init.normal_(
            self.compressor.weight, std=1.0 / math.sqrt(backbone_dim))

        # Identity sub-graph
        d_id = z_causal_dim + curated_dim + rules_dim  # 70
        self.scm_identity_real = LinearSCM(d_id)
        self.scm_identity_fake = LinearSCM(d_id)
        self._d_identity = d_id

        # Forensic sub-graph
        d_for = z_causal_dim + forensic_dim  # 115
        self.scm_forensic_real = LinearSCM(d_for)
        self.scm_forensic_fake = LinearSCM(d_for)
        self._d_forensic = d_for

        # Differential residuals -> evidence
        residual_dim = d_id + d_for  # 185
        self.residual_mlp = nn.Sequential(
            nn.LayerNorm(residual_dim),
            nn.Linear(residual_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )
        nn.init.normal_(self.residual_mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.residual_mlp[-1].bias)

        logger.info(
            f"[SimplifiedCausalBranch] "
            f"identity={d_id} nodes, forensic={d_for} nodes, "
            f"residual_dim={residual_dim} -> {num_classes} evidence")

    def forward(
        self,
        spatial_raw: torch.Tensor,
        combined_features: torch.Tensor,
        violations: torch.Tensor,
        forensic_features: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            spatial_raw:       (B, 1024) CLIP features (will be detached)
            combined_features: (B, 58) fast feature vector
            violations:        (B, 12) consistency rule scores
            forensic_features: (B, 83) precomputed forensic features
        Returns:
            dict with 'logits', 'evidence', 'residuals',
            'A_identity_real/fake', 'A_forensic_real/fake',
            'dag_penalty'
        """
        # Detach: no gradient flows back to CLIP backbone
        z = self.compressor(spatial_raw.detach())  # (B, 32)

        # Extract curated attributes from fast feature vector
        curated = combined_features[:, self._curated_indices]  # (B, 26)

        # Build sub-graph inputs
        x_id = torch.cat([z, curated, violations], dim=1)     # (B, 70)
        x_for = torch.cat([z, forensic_features], dim=1)      # (B, 115)

        # SCM forward: x_hat = W @ x + b, residual = x - x_hat
        r_id_real = x_id - self.scm_identity_real(x_id)
        r_id_fake = x_id - self.scm_identity_fake(x_id)
        r_for_real = x_for - self.scm_forensic_real(x_for)
        r_for_fake = x_for - self.scm_forensic_fake(x_for)

        # Differential residual: what the fake SCM explains differently
        r_id = r_id_fake - r_id_real       # (B, 106)
        r_for = r_for_fake - r_for_real    # (B, 62)
        residuals = torch.cat([r_id, r_for], dim=1)  # (B, 168)

        logits = self.residual_mlp(residuals)  # (B, 2)
        evidence = F.softplus(logits)          # (B, 2)

        # Adjacency matrices for explainability + DAG penalty
        A_id_real = self.scm_identity_real.get_adjacency()
        A_id_fake = self.scm_identity_fake.get_adjacency()
        A_for_real = self.scm_forensic_real.get_adjacency()
        A_for_fake = self.scm_forensic_fake.get_adjacency()

        # DAG acyclicity penalty (sum over all 4 SCMs)
        dag_penalty = (
            dagma_acyclicity(A_id_real)
            + dagma_acyclicity(A_id_fake)
            + dagma_acyclicity(A_for_real)
            + dagma_acyclicity(A_for_fake)
        )
        # Sparsity penalty on adjacency
        sparsity = self.sparsity_penalty * (
            A_id_real.abs().sum() + A_id_fake.abs().sum()
            + A_for_real.abs().sum() + A_for_fake.abs().sum()
        )

        return {
            'logits': logits,
            'evidence': evidence,
            'residuals': residuals,
            'A_identity_real': A_id_real.detach(),
            'A_identity_fake': A_id_fake.detach(),
            'A_forensic_real': A_for_real.detach(),
            'A_forensic_fake': A_for_fake.detach(),
            'dag_penalty': dag_penalty + sparsity,
        }

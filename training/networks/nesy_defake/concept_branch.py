"""
networks/nesy_defake/concept_branch.py
======================================
Lightweight neuro-symbolic branches for ablation-based evidence fusion.

ConceptBranch (Ablation 3):
  122-d combined features -> consistency rules v7 (23-d) -> MLP -> 2-d evidence

SimplifiedCausalBranch (Ablation 4):
  Spatial-only linear SCMs, identity (106) + forensic (115) sub-graphs
  -> differential residuals -> MLP -> 2-d evidence

Both branches produce non-negative evidence vectors that are fused with
the spatial branch's evidence via gated addition in the detector.
"""

import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .semantic.consistency_rules_v7 import CrossAttributeConsistencyRulesV7
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
    Concept evidence branch operating on precomputed semantic features.

    Pipeline:
      combined_features (B, 122)
        -> CrossAttributeConsistencyRulesV7 -> violations (B, 23)
        -> concat [combined || violations] = (B, 145)
        -> LayerNorm -> Linear -> GELU -> Dropout -> Linear -> logits (B, 2)
        -> softplus -> evidence (B, 2)

    No connection to CLIP spatial features — fully independent signal.
    """

    def __init__(
        self,
        combined_dim: int = 122,
        rules_dim: int = 23,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.consistency_rules = CrossAttributeConsistencyRulesV7()

        input_dim = combined_dim + rules_dim  # 145
        self.concept_mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        # Small init so concept evidence starts near zero
        nn.init.normal_(self.concept_mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.concept_mlp[-1].bias)

        logger.info(
            f"[ConceptBranch] {input_dim}-d input "
            f"({combined_dim} combined + {rules_dim} rules) "
            f"-> {hidden_dim} hidden -> {num_classes} evidence")

    def forward(self, combined_features: torch.Tensor) -> dict:
        """
        Args:
            combined_features: (B, 122) precomputed [fast(58) || vlm(64)]
        Returns:
            dict with 'logits', 'evidence', 'violations', 'concept_input'
        """
        violations = self.consistency_rules(combined_features)  # (B, 23)
        concept_input = torch.cat(
            [combined_features, violations], dim=1)             # (B, 145)
        logits = self.concept_mlp(concept_input)                # (B, 2)
        evidence = F.softplus(logits)                           # (B, 2)

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
      Identity:  z_causal(32) + curated(51) + rules(23) = 106 nodes
      Forensic:  z_causal(32) + forensic(83) = 115 nodes

    Real/fake SCM pairs learn different causal structures.
    Differential residuals (fake_residual - real_residual) encode what
    the manipulation "breaks" — this is the detection signal.

    Pipeline:
      spatial_raw (B, 1024) --detach--> compressor -> z (B, 32)
      x_identity = [z || curated || rules]  (B, 106)
      x_forensic = [z || forensic]          (B, 115)
      -> 4 LinearSCMs (id_real, id_fake, for_real, for_fake)
      -> differential residuals (B, 221)
      -> MLP -> logits (B, 2) -> softplus -> evidence (B, 2)
    """

    def __init__(
        self,
        backbone_dim: int = 1024,
        z_causal_dim: int = 32,
        curated_dim: int = 51,
        rules_dim: int = 23,
        forensic_dim: int = 83,
        hidden_dim: int = 64,
        num_classes: int = 2,
        sparsity_penalty: float = 0.01,
    ):
        super().__init__()
        self.z_causal_dim = z_causal_dim
        self.sparsity_penalty = sparsity_penalty

        # Curated attribute indices in combined 122-d vector
        self._curated_indices = CAUSAL_ATTRIBUTE_INDICES

        # Compress spatial features -> z_causal (detached from CLIP)
        self.compressor = nn.Linear(backbone_dim, z_causal_dim, bias=False)
        nn.init.normal_(
            self.compressor.weight, std=1.0 / math.sqrt(backbone_dim))

        # Identity sub-graph
        d_id = z_causal_dim + curated_dim + rules_dim  # 106
        self.scm_identity_real = LinearSCM(d_id)
        self.scm_identity_fake = LinearSCM(d_id)
        self._d_identity = d_id

        # Forensic sub-graph
        d_for = z_causal_dim + forensic_dim  # 62
        self.scm_forensic_real = LinearSCM(d_for)
        self.scm_forensic_fake = LinearSCM(d_for)
        self._d_forensic = d_for

        # Differential residuals -> evidence
        residual_dim = d_id + d_for  # 168
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
            combined_features: (B, 122) precomputed [fast || vlm]
            violations:        (B, 23) consistency rule scores
            forensic_features: (B, 83) precomputed forensic features
        Returns:
            dict with 'logits', 'evidence', 'residuals',
            'A_identity_real/fake', 'A_forensic_real/fake',
            'dag_penalty'
        """
        # Detach: no gradient flows back to CLIP backbone
        z = self.compressor(spatial_raw.detach())  # (B, 32)

        # Extract curated attributes from combined vector
        curated = combined_features[:, self._curated_indices]  # (B, 51)

        # Build sub-graph inputs
        x_id = torch.cat([z, curated, violations], dim=1)     # (B, 106)
        x_for = torch.cat([z, forensic_features], dim=1)      # (B, 62)

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

"""
attic/.../simplified_causal_branch.py
=====================================
ARCHIVED (dead code). Split out of networks/nesy_defake/concept_branch.py.

SimplifiedCausalBranch was the Ablation-4 'simple' causal branch (legacy linear
SCM). It is superseded by ImprovedCausalBranch ('improved_scm') and
CausalConstraintVerificationBranch ('ccv'), which the reference config
(configs/ablations/full_defakenet_18rules.yaml) and all probe configs use.

Kept verbatim for reference / reproducibility of older checkpoints. Not on the
import path: causal_branch_factory raises a clear error for type='simple' and
points here. To resurrect, move this file back to
networks/nesy_defake/ and restore the factory import.
"""

import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.nesy_defake.semantic.refined_attributes import (
    CAUSAL_ATTRIBUTE_INDICES,
)
from networks.nesy_defake.causal.causal_discovery import (
    LinearSCM,
    dagma_acyclicity,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Ablation 4: Simplified Causal Evidence Branch  (ARCHIVED)
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

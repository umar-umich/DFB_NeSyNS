"""
networks/nesy_defake/improved_scm_branch.py
============================================
Ablation 4a: Improved Structural Causal Model Evidence Branch.

Fixes the fundamental issues in SimplifiedCausalBranch:
  A) Nonlinear SCMs (1-hidden-layer MLP) instead of linear Wx + b
  B) Explicit graph divergence loss pushing real ≠ fake structure
  C) Forensic features split into 3 semantically-meaningful sub-graphs
     (structural, noise-residual, spectral) instead of one 115-node graph
  D) Label-conditioned reconstruction loss: real SCM fits real samples only,
     fake SCM fits fake samples only — forces specialization

Sub-graphs (spatial-only, 4 total):
  Identity:             z(32) + curated(51) + rules(23)      = 106 nodes
  Forensic-structural:  z(32) + boundary/blur/sym/color (24)  =  56 nodes
  Forensic-noise:       z(32) + ppnc/ccnc/srm/noise (43)     =  75 nodes
  Forensic-spectral:    z(32) + dct/fft (16)                  =  48 nodes

Each sub-graph has real + fake nonlinear SCMs = 8 SCMs total.
Per-sub-graph differential residuals are projected to 8-d summaries,
concatenated (4 × 8 = 32-d), and passed through an MLP to produce
2-d evidence.
"""

import math
import logging
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .semantic.refined_attributes import (
    CAUSAL_ATTRIBUTE_INDICES,
    NUM_CAUSAL_ATTRIBUTES,
)
from .causal.causal_discovery import dagma_acyclicity

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Forensic feature group definitions (slice ranges into 83-d vector)
# ═══════════════════════════════════════════════════════════════════════════

FORENSIC_SUBGRAPH_GROUPS = [
    # (name, start_idx, end_idx, num_features)
    ('structural', 0, 30, 30),    # boundary(6)+blur(6)+sym(4)+color(4)+dct(6)+quality(4)
    ('noise',      30, 73, 43),   # ppnc(8)+ccnc(8)+srm(15)+multi-scale-noise(12)
    ('spectral',   73, 83, 10),   # fft(10)
]


# ═══════════════════════════════════════════════════════════════════════════
#  Nonlinear SCM (Fix A)
# ═══════════════════════════════════════════════════════════════════════════

class NonlinearSCM(nn.Module):
    """
    Nonlinear Structural Causal Model: x_hat = MLP(x).

    1-hidden-layer MLP with SiLU activation. The adjacency matrix is
    approximated as A ≈ |W2| @ |W1| (NOTEARS-MLP approximation),
    ignoring the nonlinearity. This is exact for small activations
    where SiLU ≈ linear.

    Compared to LinearSCM (W @ x + b):
      - Can capture threshold effects (age → wrinkles)
      - Can capture interaction effects (gender × makeup)
      - Adjacency approximation is less exact but still useful for
        DAG penalty and interpretability.
    """

    def __init__(self, d: int, hidden_dim: int = 64):
        super().__init__()
        self.d = d
        self.w1 = nn.Linear(d, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, d)
        # Small init — reconstruction starts near identity
        nn.init.normal_(self.w1.weight, std=0.02 / math.sqrt(d))
        nn.init.zeros_(self.w1.bias)
        nn.init.normal_(self.w2.weight, std=0.02 / math.sqrt(hidden_dim))
        nn.init.zeros_(self.w2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, d) → (B, d) reconstruction."""
        return self.w2(F.silu(self.w1(x)))

    def get_adjacency(self) -> torch.Tensor:
        """
        NOTEARS-MLP adjacency approximation: A ≈ |W2| @ |W1|.
        Diagonal zeroed (no self-loops).
        """
        # W1: (hidden, d), W2: (d, hidden)
        A = self.w2.weight.abs() @ self.w1.weight.abs()  # (d, d)
        return A * (1.0 - torch.eye(self.d, device=A.device))


# ═══════════════════════════════════════════════════════════════════════════
#  Sub-graph pair: real + fake nonlinear SCMs
# ═══════════════════════════════════════════════════════════════════════════

class SubGraphPair(nn.Module):
    """
    One sub-graph's real + fake nonlinear SCM pair.

    Produces:
      - Differential residual: r_fake - r_real (what manipulation breaks)
      - Projected summary: Linear(residual) → (B, summary_dim)
      - DAG penalty on both adjacency matrices
      - Sparsity penalty on both adjacencies
      - Graph divergence: negative L1 distance (maximize difference)
      - Label-conditioned reconstruction loss
    """

    def __init__(self, d: int, hidden_dim: int = 64,
                 summary_dim: int = 8,
                 sparsity_penalty: float = 0.01,
                 name: str = 'subgraph'):
        super().__init__()
        self.d = d
        self.name = name
        self.sparsity_penalty = sparsity_penalty

        self.scm_real = NonlinearSCM(d, hidden_dim)
        self.scm_fake = NonlinearSCM(d, hidden_dim)

        # Project differential residual to compact summary
        self.residual_proj = nn.Linear(d, summary_dim, bias=False)
        nn.init.normal_(self.residual_proj.weight, std=0.1 / math.sqrt(d))

        logger.info(f"[SubGraphPair:{name}] d={d}, hidden={hidden_dim}, "
                    f"summary={summary_dim}")

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, d) sub-graph input
            labels: (B,) optional — 0=real, 1=fake for reconstruction loss
        Returns:
            dict with 'summary', 'dag_penalty', 'divergence_loss',
            'recon_loss', 'A_real', 'A_fake'
        """
        device = x.device

        # SCM reconstructions
        x_hat_real = self.scm_real(x)
        x_hat_fake = self.scm_fake(x)

        # Residuals
        r_real = x - x_hat_real
        r_fake = x - x_hat_fake

        # Differential residual → projected summary
        r_diff = r_fake - r_real
        summary = self.residual_proj(r_diff)  # (B, summary_dim)

        # Adjacency matrices
        A_real = self.scm_real.get_adjacency()
        A_fake = self.scm_fake.get_adjacency()

        # --- Fix B: Graph divergence loss (maximize real ≠ fake) ---
        # Negative L1 distance: minimizing this maximizes the difference
        divergence_loss = -F.l1_loss(A_real, A_fake)

        # --- DAG + sparsity penalties ---
        dag_penalty = dagma_acyclicity(A_real) + dagma_acyclicity(A_fake)
        sparsity = self.sparsity_penalty * (
            A_real.abs().sum() + A_fake.abs().sum())

        # --- Fix D: Label-conditioned reconstruction loss ---
        recon_loss = torch.zeros(1, device=device).squeeze()
        if labels is not None and self.training:
            real_mask = (labels == 0)
            fake_mask = (labels == 1)
            if real_mask.sum() > 0:
                recon_loss = recon_loss + r_real[real_mask].pow(2).mean()
            if fake_mask.sum() > 0:
                recon_loss = recon_loss + r_fake[fake_mask].pow(2).mean()

        return {
            'summary': summary,
            'dag_penalty': dag_penalty + sparsity,
            'divergence_loss': divergence_loss,
            'recon_loss': recon_loss,
            'A_real': A_real.detach(),
            'A_fake': A_fake.detach(),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Improved Causal Evidence Branch (full pipeline)
# ═══════════════════════════════════════════════════════════════════════════

class ImprovedCausalBranch(nn.Module):
    """
    Improved causal evidence branch with fixes A-D.

    4 sub-graphs (spatial-only):
      1. Identity:            z(32) + curated(51) + rules(23) = 106
      2. Forensic-structural: z(32) + features(30) = 62
      3. Forensic-noise:      z(32) + features(43) = 75
      4. Forensic-spectral:   z(32) + features(10) = 42

    Each sub-graph has nonlinear real/fake SCM pairs → projected
    to 8-d summary → concatenated (4 × 8 = 32-d) → MLP → evidence.

    Losses (returned as dag_penalty for simplicity):
      - DAG acyclicity + sparsity on all 8 SCMs
      - Graph divergence: pushes real ≠ fake adjacency structure
      - Label-conditioned reconstruction: forces SCM specialization
    """

    def __init__(
        self,
        backbone_dim: int = 1024,
        z_causal_dim: int = 32,
        curated_dim: int = 51,
        rules_dim: int = 23,
        forensic_dim: int = 83,
        scm_hidden_dim: int = 64,
        summary_dim: int = 8,
        evidence_hidden: int = 64,
        num_classes: int = 2,
        sparsity_penalty: float = 0.01,
        divergence_weight: float = 0.1,
        recon_weight: float = 0.5,
    ):
        super().__init__()
        self.z_causal_dim = z_causal_dim
        self.divergence_weight = divergence_weight
        self.recon_weight = recon_weight

        # Curated attribute indices in combined 122-d vector
        self._curated_indices = CAUSAL_ATTRIBUTE_INDICES

        # Compress spatial features → z_causal (detached from CLIP)
        self.compressor = nn.Linear(backbone_dim, z_causal_dim, bias=False)
        nn.init.normal_(
            self.compressor.weight, std=1.0 / math.sqrt(backbone_dim))

        # --- Identity sub-graph ---
        d_identity = z_causal_dim + curated_dim + rules_dim  # 106
        self.identity_pair = SubGraphPair(
            d=d_identity, hidden_dim=scm_hidden_dim,
            summary_dim=summary_dim, sparsity_penalty=sparsity_penalty,
            name='identity')

        # --- Fix C: Split forensic into 3 sub-graphs ---
        self.forensic_groups = FORENSIC_SUBGRAPH_GROUPS
        self.forensic_pairs = nn.ModuleList()
        self._forensic_dims = []
        for name, start, end, n_feats in self.forensic_groups:
            d_for = z_causal_dim + n_feats
            self._forensic_dims.append(d_for)
            self.forensic_pairs.append(SubGraphPair(
                d=d_for, hidden_dim=min(scm_hidden_dim, d_for),
                summary_dim=summary_dim, sparsity_penalty=sparsity_penalty,
                name=f'forensic_{name}'))

        # Evidence MLP: all projected summaries → evidence
        num_subgraphs = 1 + len(self.forensic_groups)  # identity + forensic groups
        total_summary_dim = num_subgraphs * summary_dim  # 4 × 8 = 32
        self.evidence_mlp = nn.Sequential(
            nn.LayerNorm(total_summary_dim),
            nn.Linear(total_summary_dim, evidence_hidden),
            nn.GELU(),
            nn.Linear(evidence_hidden, num_classes),
        )
        nn.init.normal_(self.evidence_mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.evidence_mlp[-1].bias)

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"[ImprovedCausalBranch] {num_subgraphs} sub-graphs "
            f"(identity={d_identity}, forensic="
            f"{[d for d in self._forensic_dims]}), "
            f"scm_type=nonlinear(hidden={scm_hidden_dim}), "
            f"summary={summary_dim}×{num_subgraphs}={total_summary_dim} "
            f"→ {num_classes} evidence | {n_params:,} params")

    def forward(
        self,
        spatial_raw: torch.Tensor,
        combined_features: torch.Tensor,
        violations: torch.Tensor,
        forensic_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            spatial_raw:       (B, 1024) CLIP features (detached internally)
            combined_features: (B, 122) precomputed [fast || vlm]
            violations:        (B, 23) consistency rule scores
            forensic_features: (B, 83) precomputed forensic features
            labels:            (B,) optional — for label-conditioned recon loss
        Returns:
            dict with 'logits', 'evidence', 'dag_penalty',
            'A_identity_real/fake', plus per-forensic-group adjacencies
        """
        device = spatial_raw.device

        # Compress spatial features → z_causal (detached from CLIP)
        z = self.compressor(spatial_raw.detach())  # (B, 32)

        # --- Identity sub-graph ---
        curated = combined_features[:, self._curated_indices]  # (B, 51)
        x_identity = torch.cat([z, curated, violations], dim=1)  # (B, 106)
        identity_out = self.identity_pair(x_identity, labels)

        # --- Forensic sub-graphs (Fix C: split into 3 groups) ---
        forensic_outs = []
        for i, (name, start, end, n_feats) in enumerate(self.forensic_groups):
            x_for = torch.cat(
                [z, forensic_features[:, start:end]], dim=1)
            forensic_outs.append(self.forensic_pairs[i](x_for, labels))

        # Concatenate all projected summaries
        all_summaries = torch.cat(
            [identity_out['summary']]
            + [fo['summary'] for fo in forensic_outs],
            dim=1)  # (B, 32)

        logits = self.evidence_mlp(all_summaries)
        evidence = F.softplus(logits)

        # --- Aggregate auxiliary losses ---
        total_dag = identity_out['dag_penalty']
        total_div = identity_out['divergence_loss']
        total_recon = identity_out['recon_loss']
        for fo in forensic_outs:
            total_dag = total_dag + fo['dag_penalty']
            total_div = total_div + fo['divergence_loss']
            total_recon = total_recon + fo['recon_loss']

        # Combined penalty (all passed through dag_penalty for simplicity)
        combined_penalty = (
            total_dag
            + self.divergence_weight * total_div
            + self.recon_weight * total_recon
        )

        result = {
            'logits': logits,
            'evidence': evidence,
            'dag_penalty': combined_penalty,
            'dag_penalty_raw': total_dag.detach(),
            'divergence_loss': total_div.detach(),
            'recon_loss': total_recon.detach(),
            # Adjacencies for interpretability
            'A_identity_real': identity_out['A_real'],
            'A_identity_fake': identity_out['A_fake'],
        }
        for i, (name, _, _, _) in enumerate(self.forensic_groups):
            result[f'A_forensic_{name}_real'] = forensic_outs[i]['A_real']
            result[f'A_forensic_{name}_fake'] = forensic_outs[i]['A_fake']

        return result

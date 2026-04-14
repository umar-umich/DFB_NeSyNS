"""
Multi-Modal Fusion Module — Ablation-aware version
====================================================
Stays architecturally identical to the original (same layer structure,
same weighted-sum logic, same CrossModalAttention delegation) while
adding support for inactive branches (passed as None) for ablation runs.

Changes vs. original:
  1. __init__: branch dims and fused_dim are computed from active_branches
     so the Linear input size is always correct.
  2. forward: None-branch inputs are skipped gracefully.
  3. weighted fusion: still a weighted SUM over projected branches
     (not concat — original behaviour preserved).
  4. CrossModalAttention: still delegated to your existing class.
"""

import torch
import torch.nn as nn
from .cross_modal_attention import CrossModalAttention


class MultiModalFusion(nn.Module):
    """Fuse features from multiple active streams."""

    def __init__(self, config: dict):
        super().__init__()

        self.fusion_type    = config['fusion']['type']
        self.projection_dim = config['fusion']['projection_dim']
        self.dropout_rate   = config['fusion']['dropout']

        # ── Per-branch dims (always read all three from config) ────────────
        fm = config['foundation_models']
        self._all_branch_dims = {
            'spatial':   fm['spatial']['output_dim'],
            'frequency': fm['frequency']['output_dim'],
        }

        # ── Active branch set ─────────────────────────────────────────────
        all_branches = ('spatial', 'frequency')
        self.active_branches = list(config.get('active_branches', all_branches))

        # Dims for active branches only (in fixed order)
        self.active_dims = [
            self._all_branch_dims[b] for b in all_branches
            if b in self.active_branches
        ]

        # fused_dim: use config value if all branches active (backwards
        # compatible), otherwise compute from active dims
        # if len(self.active_branches) == 3:
        #     self.fused_dim = config['fusion']['fused_dim']
        # else:
        #     # For ablation: fused_dim = sum of active branch output dims
        #     # This matches what train.py / build_backbone() also computes.
        #     self.fused_dim = sum(self.active_dims)
        self.fused_dim = self.projection_dim * len(self.active_branches)

        # ── Build fusion layers ───────────────────────────────────────────
        if self.fusion_type == 'concat':
            self._build_concat_fusion()

        elif self.fusion_type == 'attention':
            self._build_attention_fusion()

        elif self.fusion_type == 'weighted':
            self._build_weighted_fusion()

        else:
            raise NotImplementedError(
                f"Fusion type '{self.fusion_type}' not implemented")

    # ------------------------------------------------------------------ #
    #  Builder helpers                                                     #
    # ------------------------------------------------------------------ #

    def _build_concat_fusion(self) -> None:
        """
        Original two-stage MLP:
          concat(active feats) → fused_dim  →  projection_dim
        """
        # Input dim = sum of active branch output dims
        concat_input_dim = self.projection_dim * len(self.active_branches)

        self.fusion = nn.Sequential(
            nn.Linear(concat_input_dim, self.fused_dim),
            nn.LayerNorm(self.fused_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.fused_dim, self.projection_dim),
            nn.LayerNorm(self.projection_dim),
        )

    def _build_attention_fusion(self) -> None:
        """
        Delegate to your CrossModalAttention class.
        When fewer than 3 branches are active we fall back to a simple
        linear projection (CrossModalAttention expects all three dims).
        """
        if len(self.active_branches) == 3:
            t, s, f = (self._all_branch_dims[b]
                       for b in ('spatial', 'frequency'))
            self.fusion = CrossModalAttention(t, s, f, self.projection_dim)
            self._attn_fallback = False
        else:
            # Fallback: treat as concat → linear for partial-branch runs
            concat_input_dim = sum(self.active_dims)
            self.fusion = nn.Sequential(
                nn.Linear(concat_input_dim, self.projection_dim),
                nn.LayerNorm(self.projection_dim),
                nn.ReLU(),
            )
            self._attn_fallback = True

    def _build_weighted_fusion(self) -> None:
        """
        Original: project each active branch to projection_dim,
        then compute a learnable weighted SUM.
        """
        branch_order = ('spatial', 'frequency')

        # One linear projection per ACTIVE branch
        self.branch_projs = nn.ModuleDict()
        for b in branch_order:
            if b in self.active_branches:
                self.branch_projs[b] = nn.Linear(
                    self._all_branch_dims[b], self.projection_dim)

        # Learnable scalar weight per active branch
        n = len(self.active_branches)
        self.fusion_weights = nn.Parameter(torch.ones(n) / n)

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        spatial_feat:   torch.Tensor | None,
        frequency_feat: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Args:
            spatial_feat:   (B, D_spatial)   — pass None if branch inactive
            frequency_feat: (B, D_frequency) — pass None if branch inactive
        Returns:
            fused: (B, projection_dim)
        """
        # Map branch name → tensor (None for inactive)
        feat_map = {
            'spatial':   spatial_feat,
            'frequency': frequency_feat,
        }

        # Collect active tensors in canonical order
        branch_order = ('spatial', 'frequency')
        active_feats = [feat_map[b] for b in branch_order
                        if b in self.active_branches]

        if not active_feats:
            raise ValueError(
                "MultiModalFusion.forward(): all branch inputs are None. "
                "At least one branch must be active.")

        # ── Concat ────────────────────────────────────────────────────────
        if self.fusion_type == 'concat':
            combined = torch.cat(active_feats, dim=1)
            return self.fusion(combined)

        # ── Attention ─────────────────────────────────────────────────────
        elif self.fusion_type == 'attention':
            if not self._attn_fallback:
                # Full three-branch CrossModalAttention (original path)
                return self.fusion(spatial_feat, frequency_feat)
            else:
                # Partial-branch fallback: concat → linear
                combined = torch.cat(active_feats, dim=1)
                return self.fusion(combined)

        # ── Weighted sum ──────────────────────────────────────────────────
        elif self.fusion_type == 'weighted':
            weights = torch.softmax(self.fusion_weights, dim=0)  # (n,)
            projected = [
                self.branch_projs[b](feat_map[b])
                for b in branch_order if b in self.active_branches
            ]
            # Weighted sum in projection space (matches original behaviour)
            fused = sum(w * p for w, p in zip(weights, projected))
            return fused

        else:
            raise NotImplementedError(
                f"Fusion type '{self.fusion_type}' not implemented")
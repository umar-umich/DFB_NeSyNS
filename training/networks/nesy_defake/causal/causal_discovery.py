"""
networks/nesy_defake/causal/causal_discovery_module.py
=======================================================
Module 3: Dual Sub-Graph Causal Discovery — DAGMA-DCE v3

ARCHITECTURE: 8 causal graphs (2 branches × 2 sub-graphs × 2 distributions)
-----------------------------------------------------------------------------
v3 redesign (2026-03-23):
  - Dual sub-graphs per branch:
    A) Identity-Causal: z_causal(32) + curated_attrs(51) + rules(20) = 103 nodes
       Discovers: gender→facial_hair, age→skin, expression→AU causal chains
       that faceswap breaks by pasting a face with different identity attributes.
    B) Forensic-Pixel: z_causal(32) + forensic_features(30) = 62 nodes
       Discovers: blur→boundary→frequency artifact causal chains.
  - Two-stage compression: backbone(1024) → z_feature(128) → z_causal(32)
    128-d features used for fusion/classifier; 32-d for graph tractability.
  - Curated 51 FaceBench attributes (from 211) selected for causal chains:
    gender-linked(11), age-linked(9), structural(10), expression-AU(15),
    skin-tone(4), symmetry(2).
  - Expanded consistency rules: 20 training + 13 inference-intervention.
  - Linear SCM for structure discovery (Peters et al., 2014).

Detection signal: residuals from both sub-graphs concatenated per branch.
  - Identity residuals: captures identity-constraint violations
  - Forensic residuals: captures pixel-level artifact patterns
  - Combined → CausalViolationAttentionFusion → classifier
"""

import logging
import math
from typing import Optional, Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DAGMA acyclicity constraint (Bello et al., NeurIPS 2022)
# ---------------------------------------------------------------------------

def dagma_acyclicity(A: torch.Tensor, s: float = 1.0) -> torch.Tensor:
    """
    DAGMA M-matrix acyclicity constraint: h(A) = -log det(sI - A∘A) + d*log(s)

    h(A) = 0 iff the graph represented by A is a DAG.

    Args:
        A: (d, d) weighted adjacency matrix
        s: scalar > spectral_radius(A∘A). Default s=1.0 works when A is small.
    Returns:
        Scalar h(A) >= 0, equals 0 iff A is a DAG.
    """
    d = A.shape[0]
    A_sq = A * A
    M = s * torch.eye(d, device=A.device, dtype=A.dtype) - A_sq
    _, logabsdet = torch.linalg.slogdet(M)
    h = -logabsdet + d * math.log(s)
    return h.clamp(min=0.0)


# ---------------------------------------------------------------------------
# Linear SCM: identifiable causal structure discovery
# ---------------------------------------------------------------------------

class LinearSCM(nn.Module):
    """
    Linear Structural Causal Model for d variables.

    x_hat = W @ x + b

    Linear SCMs have stronger identifiability guarantees than nonlinear ones
    (Peters et al., 2014). For causal graph discovery, the weight matrix W
    directly encodes causal effects — no Jacobian computation needed.

    The Jacobian of a linear model IS the weight matrix: df/dx = W.
    This makes graph discovery exact, not an EMA approximation.
    """

    def __init__(self, d: int, **kwargs):
        super().__init__()
        self.d = d
        self.W = nn.Parameter(torch.zeros(d, d))
        self.b = nn.Parameter(torch.zeros(d))
        # Small random init — zero-centered for sparsity
        nn.init.normal_(self.W, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, d) -> (B, d)"""
        return x @ self.W.t() + self.b

    def get_adjacency(self) -> torch.Tensor:
        """Adjacency = |W| with diagonal zeroed (no self-loops)."""
        A = self.W.abs()
        return A * (1.0 - torch.eye(self.d, device=A.device))


# ---------------------------------------------------------------------------
# Nonlinear SCM: expressive residual computation for detection signal
# ---------------------------------------------------------------------------

class BatchedSCM(nn.Module):
    """
    Nonlinear Structural Causal Model for d variables.

    Shared trunk MLP + d output heads. Sigmoid activations for
    everywhere-differentiable Jacobians.

    Architecture:
        x (B, d) -> trunk [Linear->Sigmoid]x(L-1) -> h (B, hidden)
        -> heads Linear(hidden, d) -> x_hat (B, d)
    """

    def __init__(self, d: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.d = d

        trunk_layers = []
        d_in = d
        for _ in range(num_layers - 1):
            trunk_layers.append(nn.Linear(d_in, hidden_dim))
            trunk_layers.append(nn.Sigmoid())
            d_in = hidden_dim
        self.trunk = nn.Sequential(*trunk_layers)

        self.heads = nn.Linear(hidden_dim, d)
        nn.init.normal_(self.heads.weight, std=0.01)
        nn.init.zeros_(self.heads.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, d) -> (B, d)"""
        return self.heads(self.trunk(x))


# ---------------------------------------------------------------------------
# DAGMA-DCE graph learner (supports linear and nonlinear SCMs)
# ---------------------------------------------------------------------------

class DAGMADCELearner(nn.Module):
    """
    DAGMA-DCE causal graph learner for one distribution (real OR fake).

    With linear SCM: A = |W| directly (exact, no EMA needed).
    With nonlinear SCM: A_ij = sqrt(E_x[(df_j/dx_i)^2]) via factored Jacobian.

    The EMA buffer accumulates graph estimates for stable visualization.
    """

    def __init__(self, d: int, hidden_dim: int = 64, num_layers: int = 2,
                 sparsity_penalty: float = 0.01, s: float = 1.0,
                 scm_type: str = 'linear'):
        super().__init__()
        self.d = d
        self.s = s
        self.sparsity_penalty = sparsity_penalty
        self.scm_type = scm_type

        if scm_type == 'linear':
            self.scm = LinearSCM(d)
        else:
            self.scm = BatchedSCM(d, hidden_dim, num_layers)

        self.register_buffer('_A_dce_ema', torch.zeros(d, d))
        self._ema_decay = 0.99
        self._ema_initialized = False

    def set_direction_mask(self, mask: torch.Tensor) -> None:
        """Register a directionality mask to enforce structural priors."""
        self.register_buffer('_direction_mask', mask)

    def compute_adjacency_dce(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute adjacency matrix.

        Linear: A = |W| (exact, O(d²) — no backward passes).
        Nonlinear: factored Jacobian (O(hidden_dim) backward passes).

        Args:
            x: (B, d) input from a single distribution
        Returns:
            A_dce: (d, d) adjacency matrix; EMA buffer updated in-place.
        """
        if self.scm_type == 'linear':
            A_dce = self.scm.get_adjacency()
        else:
            # Nonlinear: factored Jacobian (same as v1)
            _, d = x.shape
            assert d == self.d

            if not x.requires_grad:
                x = x.detach().requires_grad_(True)

            h = self.scm.trunk(x)
            hidden_dim = h.shape[1]

            J_trunk_sq = torch.zeros(d, hidden_dim, device=x.device, dtype=x.dtype)
            for k in range(hidden_dim):
                grad_k = torch.autograd.grad(
                    outputs=h[:, k].sum(),
                    inputs=x,
                    create_graph=self.training,
                    retain_graph=True,
                )[0]
                J_trunk_sq[:, k] = (grad_k ** 2).mean(dim=0)

            W_sq = self.scm.heads.weight ** 2
            A_sq = J_trunk_sq @ W_sq.t()
            A_dce = torch.sqrt(A_sq + 1e-10)
            A_dce = A_dce * (1.0 - torch.eye(d, device=A_dce.device))

        # Apply directionality mask
        if hasattr(self, '_direction_mask') and self._direction_mask is not None:
            A_dce = A_dce * self._direction_mask

        if self.training:
            with torch.no_grad():
                if not self._ema_initialized:
                    self._A_dce_ema.copy_(A_dce.detach())
                    self._ema_initialized = True
                else:
                    self._A_dce_ema.mul_(self._ema_decay).add_(
                        A_dce.detach(), alpha=1.0 - self._ema_decay)

        return A_dce

    def compute_dag_penalty(self) -> torch.Tensor:
        """DAGMA acyclicity penalty on EMA adjacency."""
        return dagma_acyclicity(self._A_dce_ema, self.s)


# ---------------------------------------------------------------------------
# Feature compression: backbone features -> compact causal z-features
# ---------------------------------------------------------------------------

class SparseFeatureSelector(nn.Module):
    """
    Linear projection: compresses backbone features to compact causal nodes.

    No activation — keeps the projection interpretable as a linear combination
    so each z_active dimension maps to a fixed set of backbone features.
    """

    def __init__(self, sae_dict_size: int, output_dim: int, num_branches: int = 1):
        super().__init__()
        total_sae_dim = sae_dict_size * num_branches
        self.proj = nn.Linear(total_sae_dim, output_dim, bias=False)
        nn.init.normal_(self.proj.weight, std=1.0 / math.sqrt(total_sae_dim))
        logger.info(f"[SparseFeatureSelector] {total_sae_dim} -> {output_dim} (linear, no bias)")

    def forward(self, z_sae: torch.Tensor) -> torch.Tensor:
        return self.proj(z_sae)


# ---------------------------------------------------------------------------
# Directionality mask builder
# ---------------------------------------------------------------------------

def build_directionality_mask(z_dim: int, s_dim: int) -> torch.Tensor:
    """
    Build a (d, d) binary mask that blocks latent→semantic edges.

    Encodes the forensic prior: semantic/forensic concepts cause visual
    patterns, not vice versa. Allowed edges:
      - forensic → latent   (anomalies cause feature patterns)
      - forensic → forensic (inter-anomaly causality)
      - latent → latent     (intra-feature relationships)
      - latent → forensic   BLOCKED
    """
    d = z_dim + s_dim
    mask = torch.ones(d, d)
    mask[:z_dim, z_dim:d] = 0.0
    return mask


# ---------------------------------------------------------------------------
# Per-branch causal graph pair (real + fake)
# ---------------------------------------------------------------------------

class BranchCausalPair(nn.Module):
    """
    One branch's real + fake causal graph pair.

    Contains:
      - causal_learner_real:  DAGMA-DCE for real-face distribution
      - causal_learner_fake:  DAGMA-DCE for fake-face distribution

    Detection signal: SCM residuals (x - x_hat).
    """

    def __init__(self, d: int, dag_cfg: dict, disc_cfg: dict, branch_name: str,
                 direction_mask: Optional[torch.Tensor] = None):
        super().__init__()
        self.d = d
        self.branch_name = branch_name
        self._jacobian_every_n = disc_cfg.get('jacobian_every_n', 1)
        self._step_counter = 0

        scm_type = disc_cfg.get('scm_type', 'linear')

        self.causal_learner_real = DAGMADCELearner(
            d=d,
            hidden_dim=dag_cfg['hidden_dim'],
            num_layers=dag_cfg['num_layers'],
            sparsity_penalty=disc_cfg.get('sparsity_penalty', 0.01),
            s=disc_cfg.get('dagma_s', 1.0),
            scm_type=scm_type,
        )

        self.causal_learner_fake = DAGMADCELearner(
            d=d,
            hidden_dim=dag_cfg['hidden_dim'],
            num_layers=dag_cfg['num_layers'],
            sparsity_penalty=disc_cfg.get('sparsity_penalty_fake',
                                          disc_cfg.get('sparsity_penalty', 0.01) * 2),
            s=disc_cfg.get('dagma_s', 1.0),
            scm_type=scm_type,
        )

        if direction_mask is not None:
            self.causal_learner_real.set_direction_mask(direction_mask)
            self.causal_learner_fake.set_direction_mask(direction_mask)

    def forward(
        self,
        x: torch.Tensor,
        label: Optional[torch.Tensor],
        return_graph: bool = False,
    ) -> Dict:
        """
        Run real + fake SCM on input x, return residuals and optionally graphs.
        """
        # SCM reconstructions (all frames)
        x_hat_real = self.causal_learner_real.scm(x)
        x_hat_fake = self.causal_learner_fake.scm(x)
        residuals_real = x - x_hat_real
        residuals_fake = x - x_hat_fake

        # Graph discovery (label-restricted)
        A_real = self.causal_learner_real._A_dce_ema
        A_fake = self.causal_learner_fake._A_dce_ema

        compute_jacobian = (
            not self.training
            or self._jacobian_every_n <= 1
            or (self._step_counter % self._jacobian_every_n == 0)
        )
        if self.training:
            self._step_counter += 1

        if self.training and label is not None and compute_jacobian:
            real_mask = (label == 0)
            fake_mask = (label == 1)

            if real_mask.sum() > 1:
                x_real = x[real_mask].detach().requires_grad_(True)
                A_real = self.causal_learner_real.compute_adjacency_dce(x_real)

            if fake_mask.sum() > 1:
                x_fake = x[fake_mask].detach().requires_grad_(True)
                A_fake = self.causal_learner_fake.compute_adjacency_dce(x_fake)

        out = {
            'residuals_real': residuals_real,
            'residuals_fake': residuals_fake,
        }
        if return_graph:
            out['A_real'] = A_real
            out['A_fake'] = A_fake
        return out


# ---------------------------------------------------------------------------
# Main module: CausalDiscoveryModule (compact forensic-focused)
# ---------------------------------------------------------------------------

class CausalDiscoveryModule(nn.Module):
    """
    Module 3: Dual Sub-Graph Causal Discovery (v3).

    8 graphs total: 2 branches × 2 sub-graphs × 2 distributions.

    Per branch:
      A) Identity-Causal sub-graph (~103 nodes):
         z_causal(32) + curated_attrs(51) + training_rules(20)
         Discovers: gender→facial_hair, age→skin, expression→AU chains
      B) Forensic-Pixel sub-graph (~62 nodes):
         z_causal(32) + forensic_features(30)
         Discovers: blur→boundary→frequency artifact chains

    Two-stage compression: backbone(1024) → z_feature(128) → z_causal(32).
    128-d features go to fusion/classifier; 32-d for graph tractability.
    Residuals from both sub-graphs concatenated per branch for attention fusion.
    """

    def __init__(self, config: dict, semantic_attr_names: Optional[list] = None):
        super().__init__()

        from ..semantic.facial_semantic_extractor import (
            CAUSAL_ATTRIBUTE_INDICES, CAUSAL_ATTRIBUTE_NAMES,
            NUM_CAUSAL_ATTRIBUTES,
        )

        causal_cfg = config['causal_module']
        lv_cfg = causal_cfg['latent_variables']
        dag_cfg = causal_cfg['dag_learning']
        disc_cfg = causal_cfg['discovery']

        # -- z-dimensions: two-stage compression --------------------------------
        # Stage 1: backbone(1024) → z_feature via SparseFeatureSelector
        self.z_spatial_dim = lv_cfg['z_spatial_dim']       # 128
        self.z_freq_dim = lv_cfg['z_frequency_dim']         # 128
        # Stage 2: z_feature → z_causal for graph nodes
        self.z_causal_dim = lv_cfg.get('z_causal_dim', 32)  # 32

        # -- Semantic dimension parsing -----------------------------------------
        full_s_dim = causal_cfg['semantic_dim']  # 261 (211+20+30) from detector
        base_attr_dim = config.get('semantic_attributes', {}).get('precomputed_dim', 211)
        tier1_dim = config.get('consistency_rules', {}).get('output_dim', 20)
        tier2_dim = config.get('forensic_features', {}).get('output_dim', 30)

        # Store for external consumers
        self._full_s_dim = full_s_dim
        self._base_attr_dim = base_attr_dim
        self._tier1_dim = tier1_dim
        self._tier2_dim = tier2_dim

        # -- Curated attribute indices for identity sub-graph -------------------
        self._causal_attr_indices = CAUSAL_ATTRIBUTE_INDICES  # 51 indices into 211
        self._curated_dim = NUM_CAUSAL_ATTRIBUTES             # 51
        self._identity_s_dim = self._curated_dim + tier1_dim  # 51 + 20 = 71
        self._forensic_s_dim = tier2_dim                       # 30

        # -- Graph node dimensions per sub-graph --------------------------------
        self.d_identity = self.z_causal_dim + self._identity_s_dim  # 32+71 = 103
        self.d_forensic = self.z_causal_dim + self._forensic_s_dim  # 32+30 = 62

        # Combined residual dim per branch (for attention fusion interface)
        self.d_spatial = self.d_identity + self.d_forensic  # 103+62 = 165
        self.d_freq = self.d_identity + self.d_forensic     # same

        # -- Feature selectors: backbone → z_feature (128) ----------------------
        sae_cfg = config.get('sparse_features', {})
        self.sae_dict_size = sae_cfg.get(
            'dict_size',
            sae_cfg.get('sparse_autoencoder', {}).get('input_dim', 1024) * 4,
        )

        if self.z_spatial_dim == self.sae_dict_size:
            self.spatial_selector = nn.Identity()
        else:
            self.spatial_selector = SparseFeatureSelector(
                sae_dict_size=self.sae_dict_size,
                output_dim=self.z_spatial_dim, num_branches=1)

        if self.z_freq_dim == self.sae_dict_size:
            self.freq_selector = nn.Identity()
        else:
            self.freq_selector = SparseFeatureSelector(
                sae_dict_size=self.sae_dict_size,
                output_dim=self.z_freq_dim, num_branches=1)

        # -- Causal compressors: z_feature(128) → z_causal(32) -----------------
        # Linear projection keeps interpretability (each z_causal node is a
        # fixed linear combination of z_feature dims).
        self.spatial_compressor = nn.Linear(
            self.z_spatial_dim, self.z_causal_dim, bias=False)
        self.freq_compressor = nn.Linear(
            self.z_freq_dim, self.z_causal_dim, bias=False)
        nn.init.normal_(self.spatial_compressor.weight,
                        std=1.0 / math.sqrt(self.z_spatial_dim))
        nn.init.normal_(self.freq_compressor.weight,
                        std=1.0 / math.sqrt(self.z_freq_dim))

        # -- Normalization layers -----------------------------------------------
        self.z_feature_spatial_norm = nn.LayerNorm(self.z_spatial_dim)
        self.z_feature_freq_norm = nn.LayerNorm(self.z_freq_dim)
        self.z_causal_norm = nn.LayerNorm(self.z_causal_dim)
        self.identity_s_norm = nn.LayerNorm(self._identity_s_dim)
        self.forensic_s_norm = nn.LayerNorm(self._forensic_s_dim)

        # -- Directionality masks -----------------------------------------------
        self.enforce_directionality = disc_cfg.get('enforce_directionality', False)
        identity_dir_mask = None
        forensic_dir_mask = None
        if self.enforce_directionality:
            identity_dir_mask = build_directionality_mask(
                self.z_causal_dim, self._identity_s_dim)
            forensic_dir_mask = build_directionality_mask(
                self.z_causal_dim, self._forensic_s_dim)

        # -- Identity-Causal sub-graphs (per branch) ----------------------------
        self.identity_spatial = BranchCausalPair(
            d=self.d_identity, dag_cfg=dag_cfg, disc_cfg=disc_cfg,
            branch_name='spatial_identity', direction_mask=identity_dir_mask)
        self.identity_freq = BranchCausalPair(
            d=self.d_identity, dag_cfg=dag_cfg, disc_cfg=disc_cfg,
            branch_name='freq_identity', direction_mask=identity_dir_mask)

        # -- Forensic-Pixel sub-graphs (per branch) -----------------------------
        self.forensic_spatial = BranchCausalPair(
            d=self.d_forensic, dag_cfg=dag_cfg, disc_cfg=disc_cfg,
            branch_name='spatial_forensic', direction_mask=forensic_dir_mask)
        self.forensic_freq = BranchCausalPair(
            d=self.d_forensic, dag_cfg=dag_cfg, disc_cfg=disc_cfg,
            branch_name='freq_forensic', direction_mask=forensic_dir_mask)

        # Legacy compat
        self.use_semantic_graph = False
        self.causal_semantic = None
        self.sparsity_weight = disc_cfg.get('sparsity_penalty', 0.01)

        # -- Node names for interpretability ------------------------------------
        self._identity_node_names = (
            [f'z_causal_{i}' for i in range(self.z_causal_dim)]
            + list(CAUSAL_ATTRIBUTE_NAMES)
            + list(config.get('_training_rule_names', [f'cr_{i}' for i in range(tier1_dim)]))
        )
        from ..semantic.forensic_features import get_forensic_feature_names
        self._forensic_node_names = (
            [f'z_causal_{i}' for i in range(self.z_causal_dim)]
            + get_forensic_feature_names()
        )

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        scm_type = disc_cfg.get('scm_type', 'linear')
        logger.info(
            f"[CausalDiscoveryModule] Dual sub-graph (v3): "
            f"d_identity={self.d_identity} (z_causal={self.z_causal_dim}, "
            f"curated={self._curated_dim}, rules={tier1_dim}), "
            f"d_forensic={self.d_forensic} (z_causal={self.z_causal_dim}, "
            f"forensic={tier2_dim}), "
            f"z_feature={self.z_spatial_dim}, scm_type={scm_type}, "
            f"graphs=8, trainable={n_params:,}"
        )

    # ------------------------------------------------------------------ #
    #  Input construction                                                  #
    # ------------------------------------------------------------------ #

    def _compress_z(
        self,
        z_branch: Optional[torch.Tensor],
        selector: nn.Module,
        compressor: nn.Module,
        z_feature_norm: nn.LayerNorm,
        B: int, device: torch.device,
    ) -> torch.Tensor:
        """backbone(1024) → selector → z_feature(128) → compressor → z_causal(32)"""
        if z_branch is not None:
            z_feature = z_feature_norm(selector(z_branch))
            z_causal = self.z_causal_norm(compressor(z_feature))
        else:
            z_causal = torch.zeros(B, self.z_causal_dim, device=device)
        return z_causal

    def _build_identity_input(
        self, z_causal: torch.Tensor, semantic_attrs: torch.Tensor,
    ) -> torch.Tensor:
        """Build identity sub-graph input: [z_causal(32), curated(51), rules(20)]."""
        B = z_causal.shape[0]
        device = z_causal.device

        # Extract curated attributes (51 selected from base 211)
        curated = semantic_attrs[:, self._causal_attr_indices].float()
        # Extract training rules (tier1, after base 211)
        rules = semantic_attrs[:, self._base_attr_dim:
                               self._base_attr_dim + self._tier1_dim].float()
        # Concatenate and normalize
        s_identity = torch.cat([curated, rules], dim=1)  # (B, 71)
        s_identity = self.identity_s_norm(s_identity)
        return torch.cat([z_causal, s_identity], dim=1)  # (B, 103)

    def _build_forensic_input(
        self, z_causal: torch.Tensor, semantic_attrs: torch.Tensor,
    ) -> torch.Tensor:
        """Build forensic sub-graph input: [z_causal(32), forensic(30)]."""
        # Extract forensic features (tier2, after base 211 + tier1)
        start = self._base_attr_dim + self._tier1_dim
        forensic = semantic_attrs[:, start:start + self._tier2_dim].float()
        forensic = self.forensic_s_norm(forensic)
        return torch.cat([z_causal, forensic], dim=1)  # (B, 62)

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        z_spatial: Optional[torch.Tensor] = None,
        z_freq: Optional[torch.Tensor] = None,
        semantic_attrs: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        return_graph: bool = False,
    ) -> Dict:
        """
        Dual sub-graph causal discovery (v3).

        Returns: dict with keys:
            residuals_spatial_real/fake  (B, d_identity + d_forensic)
            residuals_freq_real/fake     (B, d_identity + d_forensic)
            A_*_identity_real/fake       (d_identity, d_identity)  [if return_graph]
            A_*_forensic_real/fake       (d_forensic, d_forensic)  [if return_graph]
        """
        B = (z_spatial.shape[0] if z_spatial is not None
             else semantic_attrs.shape[0])
        device = (z_spatial.device if z_spatial is not None
                  else semantic_attrs.device)

        # Stage 1+2: compress to z_causal (32-d)
        z_causal_spatial = self._compress_z(
            z_spatial, self.spatial_selector, self.spatial_compressor,
            self.z_feature_spatial_norm, B, device)
        z_causal_freq = self._compress_z(
            z_freq, self.freq_selector, self.freq_compressor,
            self.z_feature_freq_norm, B, device)

        # Build sub-graph inputs
        x_spatial_identity = self._build_identity_input(z_causal_spatial, semantic_attrs)
        x_spatial_forensic = self._build_forensic_input(z_causal_spatial, semantic_attrs)
        x_freq_identity = self._build_identity_input(z_causal_freq, semantic_attrs)
        x_freq_forensic = self._build_forensic_input(z_causal_freq, semantic_attrs)

        # Enable grad for potential Jacobian computation
        for t in [x_spatial_identity, x_spatial_forensic,
                  x_freq_identity, x_freq_forensic]:
            if not t.requires_grad:
                t.requires_grad_(True)

        # Run all 8 sub-graphs
        si_out = self.identity_spatial(x_spatial_identity, label, return_graph)
        sf_out = self.forensic_spatial(x_spatial_forensic, label, return_graph)
        fi_out = self.identity_freq(x_freq_identity, label, return_graph)
        ff_out = self.forensic_freq(x_freq_forensic, label, return_graph)

        # Concatenate residuals per branch (identity + forensic)
        out = {
            'residuals_spatial_real': torch.cat([
                si_out['residuals_real'], sf_out['residuals_real']], dim=1),
            'residuals_spatial_fake': torch.cat([
                si_out['residuals_fake'], sf_out['residuals_fake']], dim=1),
            'residuals_freq_real': torch.cat([
                fi_out['residuals_real'], ff_out['residuals_real']], dim=1),
            'residuals_freq_fake': torch.cat([
                fi_out['residuals_fake'], ff_out['residuals_fake']], dim=1),
        }

        if return_graph:
            out['A_spatial_identity_real'] = si_out.get('A_real')
            out['A_spatial_identity_fake'] = si_out.get('A_fake')
            out['A_spatial_forensic_real'] = sf_out.get('A_real')
            out['A_spatial_forensic_fake'] = sf_out.get('A_fake')
            out['A_freq_identity_real'] = fi_out.get('A_real')
            out['A_freq_identity_fake'] = fi_out.get('A_fake')
            out['A_freq_forensic_real'] = ff_out.get('A_real')
            out['A_freq_forensic_fake'] = ff_out.get('A_fake')

        return out

    # ------------------------------------------------------------------ #
    #  Interpretability helpers                                            #
    # ------------------------------------------------------------------ #

    def get_graph_divergence(self, branch: str = 'spatial',
                             subgraph: str = 'identity') -> Dict[str, torch.Tensor]:
        """Interpretable comparison of real vs fake causal graphs."""
        pair = self._get_pair(branch, subgraph)
        A_r = pair.causal_learner_real._A_dce_ema
        A_f = pair.causal_learner_fake._A_dce_ema
        return {
            'A_real':           A_r.clone(),
            'A_fake':           A_f.clone(),
            'divergence':       (A_r - A_f).clone(),
            'broken_by_fakes':  (A_r - A_f).clamp(min=0).clone(),
            'created_by_fakes': (A_f - A_r).clamp(min=0).clone(),
        }

    def get_causal_graph(self, branch: str = 'spatial',
                          subgraph: str = 'identity') -> torch.Tensor:
        """Return real-face graph EMA."""
        pair = self._get_pair(branch, subgraph)
        return pair.causal_learner_real._A_dce_ema.clone()

    def _get_pair(self, branch: str, subgraph: str) -> BranchCausalPair:
        if branch == 'spatial':
            return self.identity_spatial if subgraph == 'identity' else self.forensic_spatial
        else:
            return self.identity_freq if subgraph == 'identity' else self.forensic_freq

    def get_node_names(self, branch: str = 'spatial',
                        subgraph: str = 'identity') -> list:
        """Human-readable names for causal graph nodes."""
        if subgraph == 'identity':
            prefix = 'z_spatial' if branch == 'spatial' else 'z_freq'
            return ([f'{prefix}_{i}' for i in range(self.z_causal_dim)]
                    + self._identity_node_names[self.z_causal_dim:])
        else:
            prefix = 'z_spatial' if branch == 'spatial' else 'z_freq'
            return ([f'{prefix}_{i}' for i in range(self.z_causal_dim)]
                    + self._forensic_node_names[self.z_causal_dim:])

    def get_semantic_node_names(self, subgraph: str = 'identity') -> list:
        """Semantic node names used in a sub-graph."""
        if subgraph == 'identity':
            return self._identity_node_names[self.z_causal_dim:]
        return self._forensic_node_names[self.z_causal_dim:]

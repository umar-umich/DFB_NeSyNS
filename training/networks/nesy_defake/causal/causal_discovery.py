"""
networks/nesy_defake/causal/causal_discovery_module.py
=======================================================
Module 3: Compact Forensic-Focused Causal Discovery — DAGMA-DCE v2

ARCHITECTURE: 4 compact causal graphs (2 branches × 2 distributions)
-------------------------------------------------------------------------
v2 redesign rationale:
  - Previous 6 graphs at 387 nodes each were far beyond DAGMA's validated
    range (20-100 nodes), leading to demographic-correlation collapse.
  - Now: 4 graphs at ~64 nodes each — well within tractable range.
  - Forensic-focused: only consistency rules (18) + forensic features (30)
    enter the causal graph, NOT the base 211 demographic attributes.
  - Linear SCM for structure discovery (better identifiability per
    Peters et al., 2014), nonlinear residuals for detection signal.

Node composition per branch graph (d ≈ 64):
  [z_branch_0..15]  16 compressed latent visual features
  [cr_*]            18 cross-attribute consistency rules
  [ff_*]            30 pixel-level forensic features
  Total:            64 interpretable, forensically relevant nodes

Detection signal: linear SCM residuals.
  - Real SCM trained on reals → high residuals on fakes
  - Fake SCM trained on fakes → high residuals on reals
  - Residual difference patterns → CausalViolationAttentionFusion → classifier

Algorithm: Hybrid Linear DAGMA + Nonlinear Residual (CDNOD-inspired)
  - Linear SCM discovers identifiable causal structure
  - The linear causal structure is shared between real/fake (faces follow
    same physics); the residual distributions differ (fakes add artifacts)
  - This handles highly overlapping distributions better than nonlinear
    SCMs which memorize correlations instead of discovering causation.
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
    Module 3: Compact Forensic-Focused Dual-Graph Causal Discovery.

    4 graphs total: 2 branches (spatial, freq) × 2 distributions (real, fake).
    Each graph has ~64 nodes: 16 z-features + 48 forensic/consistency features.

    The base 211 FaceBench demographic attributes are EXCLUDED from the causal
    graph — they carry identity/demographic signal, not forensic signal.
    Demographics flow through the semantic gate directly to the classifier.

    The causal graph focuses on: how do visual features (z) relate to
    forensic anomalies (consistency rules, pixel-level forensics)?
    """

    def __init__(self, config: dict, semantic_attr_names: Optional[list] = None):
        super().__init__()

        causal_cfg = config['causal_module']
        lv_cfg = causal_cfg['latent_variables']
        dag_cfg = causal_cfg['dag_learning']
        disc_cfg = causal_cfg['discovery']

        # -- z-dimensions for causal graph ------------------------------------
        # When sparse_features is disabled, z_dim should match backbone output
        # so raw features flow directly into the causal graph (no compression).
        self.z_spatial_dim = lv_cfg['z_spatial_dim']
        self.z_freq_dim = lv_cfg['z_frequency_dim']

        # -- Forensic-focused semantic selection ----------------------------
        # Only consistency rules + forensic features enter the causal graph.
        # The base 211 attributes are demographics, not forensics.
        self._forensic_only = causal_cfg.get('forensic_only', True)
        full_s_dim = causal_cfg['semantic_dim']  # 259 (211+18+30) from detector

        # Parse which portion is forensic (tier1 + tier2)
        base_attr_dim = config.get('semantic_attributes', {}).get('precomputed_dim', 211)
        tier1_dim = config.get('consistency_rules', {}).get('output_dim', 18)
        tier2_dim = config.get('forensic_features', {}).get('output_dim', 30)

        if self._forensic_only:
            # Only tier1 + tier2 go into causal graph
            self.s_dim = tier1_dim + tier2_dim  # 48
            self._sem_slice_start = base_attr_dim  # skip first 211
            self._sem_slice_end = full_s_dim       # take rest
        else:
            # Everything goes in (legacy behavior)
            self.s_dim = full_s_dim
            self._sem_slice_start = 0
            self._sem_slice_end = full_s_dim

        self.d_spatial = self.z_spatial_dim + self.s_dim   # 16 + 48 = 64
        self.d_freq = self.z_freq_dim + self.s_dim         # 16 + 48 = 64

        # -- Store full semantic dim for external consumers -----------------
        # CausalInterventionModule and detector need the full semantic dim
        self._full_s_dim = full_s_dim

        # -- Store semantic attribute names for interpretable graphs --------
        self._semantic_attr_names = semantic_attr_names
        # Build forensic-only names
        if semantic_attr_names and self._forensic_only:
            self._causal_node_sem_names = semantic_attr_names[base_attr_dim:]
        elif semantic_attr_names:
            self._causal_node_sem_names = semantic_attr_names
        else:
            self._causal_node_sem_names = [f's_{i}' for i in range(self.s_dim)]

        # -- Feature selectors (backbone -> causal z) --------------------------
        # When z_dim matches input dim (raw features mode), use Identity
        # instead of a learned projection — no wasted parameters.
        sae_cfg = config.get('sparse_features', {})
        self.sae_dict_size = sae_cfg.get(
            'dict_size',
            sae_cfg.get('sparse_autoencoder', {}).get('input_dim', 1024) * 4,
        )

        if self.z_spatial_dim == self.sae_dict_size:
            self.spatial_selector = nn.Identity()
            logger.info(f"[CausalDiscovery] spatial selector: Identity (raw {self.sae_dict_size}-d)")
        else:
            self.spatial_selector = SparseFeatureSelector(
                sae_dict_size=self.sae_dict_size,
                output_dim=self.z_spatial_dim,
                num_branches=1,
            )

        if self.z_freq_dim == self.sae_dict_size:
            self.freq_selector = nn.Identity()
            logger.info(f"[CausalDiscovery] freq selector: Identity (raw {self.sae_dict_size}-d)")
        else:
            self.freq_selector = SparseFeatureSelector(
                sae_dict_size=self.sae_dict_size,
                output_dim=self.z_freq_dim,
                num_branches=1,
            )

        # -- Normalization layers -------------------------------------------
        self.z_spatial_norm = nn.LayerNorm(self.z_spatial_dim)
        self.z_freq_norm = nn.LayerNorm(self.z_freq_dim)
        self.s_norm = nn.LayerNorm(self.s_dim)

        # -- Directionality masks (latent→forensic blocked) -----------------
        self.enforce_directionality = disc_cfg.get('enforce_directionality', False)
        spatial_dir_mask = None
        freq_dir_mask = None
        if self.enforce_directionality:
            spatial_dir_mask = build_directionality_mask(self.z_spatial_dim, self.s_dim)
            freq_dir_mask = build_directionality_mask(self.z_freq_dim, self.s_dim)
            logger.info(
                f"[CausalDiscoveryModule] Directionality mask: "
                f"blocking latent→forensic edges (z→s block zeroed)")

        # -- Spatial branch: real + fake causal graphs ----------------------
        self.causal_spatial = BranchCausalPair(
            d=self.d_spatial, dag_cfg=dag_cfg, disc_cfg=disc_cfg,
            branch_name='spatial', direction_mask=spatial_dir_mask)

        # -- Frequency branch: real + fake causal graphs --------------------
        self.causal_freq = BranchCausalPair(
            d=self.d_freq, dag_cfg=dag_cfg, disc_cfg=disc_cfg,
            branch_name='frequency', direction_mask=freq_dir_mask)

        # -- Semantic-only graph removed (v2) -------------------------------
        # Demographics (211 base attrs) don't belong in causal graph.
        # Forensic features already capture inter-attribute relationships
        # via consistency rules (e.g., cr_happy_au6, cr_gender_beard).
        self.use_semantic_graph = False
        self.causal_semantic = None

        self.sparsity_weight = disc_cfg.get('sparsity_penalty', 0.01)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        scm_type = disc_cfg.get('scm_type', 'linear')
        logger.info(
            f"[CausalDiscoveryModule] Compact forensic-focused (v2): "
            f"d_spatial={self.d_spatial} (z={self.z_spatial_dim}, s={self.s_dim}), "
            f"d_freq={self.d_freq} (z={self.z_freq_dim}, s={self.s_dim}), "
            f"forensic_only={self._forensic_only}, scm_type={scm_type}, "
            f"graphs=4, trainable={n_params:,}"
        )

    # ------------------------------------------------------------------ #
    #  Input construction                                                  #
    # ------------------------------------------------------------------ #

    def _build_branch_input(
        self,
        z_branch: Optional[torch.Tensor],
        semantic_attrs: Optional[torch.Tensor],
        selector: nn.Module,
        z_norm: nn.LayerNorm,
        z_dim: int,
    ) -> torch.Tensor:
        """
        Build causal input: [z_active(z_dim), forensic_attrs(s_dim)].
        """
        B = (z_branch.shape[0] if z_branch is not None
             else semantic_attrs.shape[0])
        device = (z_branch.device if z_branch is not None
                  else semantic_attrs.device)

        if z_branch is not None:
            z_active = z_norm(selector(z_branch))
        else:
            z_active = torch.zeros(B, z_dim, device=device)

        if semantic_attrs is not None:
            # Slice to forensic-only features if configured
            s_input = semantic_attrs[:, self._sem_slice_start:self._sem_slice_end]
            s_sem = self.s_norm(s_input.float())
        else:
            s_sem = torch.zeros(B, self.s_dim, device=device)

        return torch.cat([z_active, s_sem], dim=1)

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
        Compact forensic-focused causal discovery.

        Returns: dict with keys:
            residuals_spatial_real/fake (B, d_spatial) per-branch residuals
            residuals_freq_real/fake    (B, d_freq)   per-branch residuals
            A_spatial_real/fake     (d_spatial, d_spatial) [if return_graph]
            A_freq_real/fake        (d_freq, d_freq)      [if return_graph]
        """
        # Build compact causal inputs
        x_spatial = self._build_branch_input(
            z_spatial, semantic_attrs,
            self.spatial_selector, self.z_spatial_norm, self.z_spatial_dim)
        x_freq = self._build_branch_input(
            z_freq, semantic_attrs,
            self.freq_selector, self.z_freq_norm, self.z_freq_dim)

        # Enable grad for Jacobian computation (nonlinear SCM only)
        if not x_spatial.requires_grad:
            x_spatial = x_spatial.requires_grad_(True)
        if not x_freq.requires_grad:
            x_freq = x_freq.requires_grad_(True)

        # Per-branch causal discovery
        spatial_out = self.causal_spatial(x_spatial, label, return_graph)
        freq_out = self.causal_freq(x_freq, label, return_graph)

        out = {
            'residuals_spatial_real': spatial_out['residuals_real'],
            'residuals_spatial_fake': spatial_out['residuals_fake'],
            'residuals_freq_real': freq_out['residuals_real'],
            'residuals_freq_fake': freq_out['residuals_fake'],
        }

        if return_graph:
            out['A_spatial_real'] = spatial_out['A_real']
            out['A_spatial_fake'] = spatial_out['A_fake']
            out['A_freq_real'] = freq_out['A_real']
            out['A_freq_fake'] = freq_out['A_fake']

        return out

    # ------------------------------------------------------------------ #
    #  Interpretability helpers                                            #
    # ------------------------------------------------------------------ #

    def get_graph_divergence(self, branch: str = 'spatial') -> Dict[str, torch.Tensor]:
        """Interpretable comparison of real vs fake causal graphs."""
        pair = self.causal_spatial if branch == 'spatial' else self.causal_freq
        A_r = pair.causal_learner_real._A_dce_ema
        A_f = pair.causal_learner_fake._A_dce_ema
        return {
            'A_real':           A_r.clone(),
            'A_fake':           A_f.clone(),
            'divergence':       (A_r - A_f).clone(),
            'broken_by_fakes':  (A_r - A_f).clamp(min=0).clone(),
            'created_by_fakes': (A_f - A_r).clamp(min=0).clone(),
        }

    def get_causal_graph(self, branch: str = 'spatial') -> torch.Tensor:
        """Return real-face graph EMA for a branch."""
        pair = self.causal_spatial if branch == 'spatial' else self.causal_freq
        return pair.causal_learner_real._A_dce_ema.clone()

    def get_node_names(self, branch: str = 'spatial') -> list:
        """Human-readable names for the compact causal graph nodes."""
        names = []
        if branch == 'spatial':
            for i in range(self.z_spatial_dim):
                names.append(f'z_spatial_{i}')
        else:
            for i in range(self.z_freq_dim):
                names.append(f'z_freq_{i}')

        names.extend(self._causal_node_sem_names)
        return names

    def get_semantic_node_names(self) -> list:
        """Forensic node names used in the causal graph."""
        return list(self._causal_node_sem_names)

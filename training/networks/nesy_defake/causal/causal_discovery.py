"""
networks/nesy_defake/causal/causal_discovery_module.py
=======================================================
Module 3: Per-Branch Dual-Graph Causal Discovery — DAGMA-DCE for Deepfake Detection

ARCHITECTURE: CAUSAL GRAPHS (2 branches × 2 distributions + 1 semantic pair)
-------------------------------------------------------------------------
Each latent branch (spatial, frequency) has its own pair of real/fake causal
graphs that discover relationships between that branch's SAE features and
the shared semantic facial attributes.

Detection signal comes from SCM RESIDUALS:
  - Real SCM trained on reals → high residuals on fakes (can't explain them)
  - Fake SCM trained on fakes → high residuals on reals (can't explain them)
  - Residual patterns fed to attention fusion → classifier

Additionally, an intra-semantic causal graph (Part A) discovers inter-attribute
causal relationships (e.g., happy → AU6 + AU12) that hold in reals but break
in fakes.

Directionality masks (Part B) encode the forensic prior: semantic concepts
cause visual patterns, not vice versa.
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
# Shared SCM: all d nodes modeled by a single batched MLP
# ---------------------------------------------------------------------------

class BatchedSCM(nn.Module):
    """
    Structural Causal Model for d variables.

    Shared trunk MLP + d output heads. More efficient than d separate NodeMLPs.
    Sigmoid activations (not ReLU) for everywhere-differentiable Jacobians.

    Architecture:
        x (B, d) -> trunk [Linear->Sigmoid]x(L-1) -> h (B, hidden) -> heads Linear(hidden, d) -> x_hat (B, d)
    """

    def __init__(self, d: int, hidden_dim: int = 128, num_layers: int = 3):
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
# DAGMA-DCE graph learner
# ---------------------------------------------------------------------------

class DAGMADCELearner(nn.Module):
    """
    DAGMA-DCE causal graph learner for one distribution (real OR fake).

    Learns A_dce where A_ij = sqrt( E_x[(df_j/dx_i)^2] ) -- the RMS Jacobian
    (Differential Causal Effect) of the SCM, interpretable as causal strength.

    The EMA buffer _A_dce_ema accumulates Jacobian estimates from labeled subsets:
    - DAGMADCELearner for real: updated only from real frames
    - DAGMADCELearner for fake: updated only from fake frames
    """

    def __init__(self, d: int, hidden_dim: int = 128, num_layers: int = 3,
                 sparsity_penalty: float = 0.01, s: float = 1.0):
        super().__init__()
        self.d = d
        self.s = s
        self.sparsity_penalty = sparsity_penalty

        self.scm = BatchedSCM(d, hidden_dim, num_layers)

        self.register_buffer('_A_dce_ema', torch.zeros(d, d))
        self._ema_decay = 0.99
        self._ema_initialized = False

    def set_direction_mask(self, mask: torch.Tensor) -> None:
        """Register a directionality mask to enforce structural priors on A_dce."""
        self.register_buffer('_direction_mask', mask)

    def compute_adjacency_dce(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute DAGMA-DCE adjacency matrix via factored Jacobian.

        Factored via shared trunk: J_full = heads.weight @ J_trunk, reducing
        cost from d full backward passes to hidden_dim trunk passes + matmul.

        Args:
            x: (B, d) input from a single distribution (real or fake subset)
        Returns:
            A_dce: (d, d) DCE adjacency matrix; EMA buffer updated in-place.
        """
        _, d = x.shape
        assert d == self.d

        if not x.requires_grad:
            x = x.detach().requires_grad_(True)

        h = self.scm.trunk(x)       # (B, hidden)
        hidden_dim = h.shape[1]

        CHUNK = 64
        J_trunk_sq = torch.zeros(d, hidden_dim, device=x.device, dtype=x.dtype)

        for k_start in range(0, hidden_dim, CHUNK):
            k_end = min(k_start + CHUNK, hidden_dim)
            for k in range(k_start, k_end):
                grad_k = torch.autograd.grad(
                    outputs=h[:, k].sum(),
                    inputs=x,
                    create_graph=self.training,
                    retain_graph=True,
                )[0]                # (B, d)
                J_trunk_sq[:, k] = (grad_k ** 2).mean(dim=0)

        # Factored DCE: A_sq[i,j] ~ sum_k W[j,k]^2 * E[(dh_k/dx_i)^2]
        W_sq = self.scm.heads.weight ** 2   # (d, hidden)
        A_sq = J_trunk_sq @ W_sq.t()        # (d, d)
        A_dce = torch.sqrt(A_sq + 1e-10)
        A_dce = A_dce * (1.0 - torch.eye(d, device=A_dce.device))

        # Apply directionality mask (Part B: block latent→semantic edges)
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
        """DAGMA acyclicity penalty on EMA adjacency. Called from get_losses()."""
        return dagma_acyclicity(self._A_dce_ema, self.s)


# ---------------------------------------------------------------------------
# Feature compression: per-branch SAE sparse -> dense active features
# ---------------------------------------------------------------------------

class SparseFeatureSelector(nn.Module):
    """
    Linear projection: picks out causally relevant SAE dictionary elements.

    No activation -- keeps the projection interpretable as a linear combination
    of monosemantic features, so each z_active dimension maps to a fixed set
    of dictionary atoms that the causal graph can reason about consistently.
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
# Directionality mask builder (Part B)
# ---------------------------------------------------------------------------

def build_directionality_mask(z_dim: int, s_dim: int) -> torch.Tensor:
    """
    Build a (d, d) binary mask that blocks latent→semantic edges.

    Encodes the forensic prior: semantic concepts cause visual patterns,
    not vice versa. After masking, allowed edges are:
      - semantic → latent   (concepts cause features)
      - semantic → semantic (inter-attribute causality)
      - latent → latent     (intra-feature relationships)
      - latent → semantic   BLOCKED

    Args:
        z_dim: number of latent feature dimensions (first z_dim rows/cols)
        s_dim: number of semantic attribute dimensions (next s_dim rows/cols)
    Returns:
        mask: (d, d) binary tensor, 0 in the latent→semantic block
    """
    d = z_dim + s_dim
    mask = torch.ones(d, d)
    # Zero out block [0:z_dim, z_dim:d] — latent rows, semantic columns
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

    Detection signal: SCM residuals (x - x_hat). Real SCM has high residuals
    on fakes (can't explain generator artifacts), fake SCM has high residuals
    on reals (can't explain natural structure). These residuals are projected
    into the classifier via CausalViolationAttentionFusion.
    """

    def __init__(self, d: int, dag_cfg: dict, disc_cfg: dict, branch_name: str,
                 direction_mask: Optional[torch.Tensor] = None):
        super().__init__()
        self.d = d
        self.branch_name = branch_name
        self._jacobian_every_n = disc_cfg.get('jacobian_every_n', 1)
        self._step_counter = 0

        self.causal_learner_real = DAGMADCELearner(
            d=d,
            hidden_dim=dag_cfg['hidden_dim'],
            num_layers=dag_cfg['num_layers'],
            sparsity_penalty=disc_cfg.get('sparsity_penalty', 0.01),
            s=disc_cfg.get('dagma_s', 1.0),
        )

        self.causal_learner_fake = DAGMADCELearner(
            d=d,
            hidden_dim=dag_cfg['hidden_dim'],
            num_layers=dag_cfg['num_layers'],
            sparsity_penalty=disc_cfg.get('sparsity_penalty_fake',
                                          disc_cfg.get('sparsity_penalty', 0.01) * 2),
            s=disc_cfg.get('dagma_s', 1.0),
        )

        # Apply directionality mask if provided
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

        Args:
            x:     (B, d) causal variables for this branch
            label: (B,) 0=real, 1=fake for label-restricted Jacobian
            return_graph: include adjacency matrices in output
        Returns:
            dict with residuals_real, residuals_fake,
            and optionally A_real, A_fake.
        """
        # SCM reconstructions (all frames)
        x_hat_real = self.causal_learner_real.scm(x)
        x_hat_fake = self.causal_learner_fake.scm(x)
        residuals_real = x - x_hat_real
        residuals_fake = x - x_hat_fake

        # Graph discovery (label-restricted Jacobians)
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
# Main module: CausalDiscoveryModule (per-branch dual-graph)
# ---------------------------------------------------------------------------

class CausalDiscoveryModule(nn.Module):
    """
    Module 3: Per-Branch Dual-Graph Causal Discovery for deepfake detection.

    Two branches (spatial, frequency), each with its own pair of real/fake
    DAGMA-DCE causal graphs operating on [z_branch_active, semantic_attrs].

    Optionally, an intra-semantic causal graph (Part A) operates on just
    the 211 semantic attributes to discover inter-attribute relationships.

    Forward returns per-branch residuals and graphs.
    """

    def __init__(self, config: dict, semantic_attr_names: Optional[list] = None):
        super().__init__()

        causal_cfg = config['causal_module']
        lv_cfg = causal_cfg['latent_variables']
        dag_cfg = causal_cfg['dag_learning']
        disc_cfg = causal_cfg['discovery']

        self.z_spatial_dim = lv_cfg['z_spatial_dim']      # 128
        self.z_freq_dim = lv_cfg['z_frequency_dim']       # 128
        self.s_dim = causal_cfg['semantic_dim']            # 211 (FaceBench) or 128 (vision-only)
        self.d_spatial = self.z_spatial_dim + self.s_dim   # 339 (with LLM) or 256
        self.d_freq = self.z_freq_dim + self.s_dim        # 339 (with LLM) or 256

        # Store semantic attribute names for interpretable causal graphs
        self._semantic_attr_names = semantic_attr_names

        sae_cfg = config.get('sparse_features', {})
        self.sae_dict_size = sae_cfg.get(
            'dict_size',
            sae_cfg.get('sparse_autoencoder', {}).get('input_dim', 1024) * 4,
        )

        # -- Per-branch feature selectors (SAE sparse -> dense active) --------
        self.spatial_selector = SparseFeatureSelector(
            sae_dict_size=self.sae_dict_size,
            output_dim=self.z_spatial_dim,
            num_branches=1,
        )
        self.freq_selector = SparseFeatureSelector(
            sae_dict_size=self.sae_dict_size,
            output_dim=self.z_freq_dim,
            num_branches=1,
        )

        # -- Normalization layers ---------------------------------------------
        self.z_spatial_norm = nn.LayerNorm(self.z_spatial_dim)
        self.z_freq_norm = nn.LayerNorm(self.z_freq_dim)
        self.s_norm = nn.LayerNorm(self.s_dim)  # shared for both branches

        # -- Part B: Directionality masks (latent→semantic blocked) -----------
        self.enforce_directionality = disc_cfg.get('enforce_directionality', False)
        spatial_dir_mask = None
        freq_dir_mask = None
        if self.enforce_directionality:
            spatial_dir_mask = build_directionality_mask(self.z_spatial_dim, self.s_dim)
            freq_dir_mask = build_directionality_mask(self.z_freq_dim, self.s_dim)
            logger.info(
                f"[CausalDiscoveryModule] Directionality mask: "
                f"blocking latent→semantic edges (z→s block zeroed)")

        # -- Spatial branch: real + fake causal graphs -----------------------
        self.causal_spatial = BranchCausalPair(
            d=self.d_spatial, dag_cfg=dag_cfg, disc_cfg=disc_cfg,
            branch_name='spatial', direction_mask=spatial_dir_mask)

        # -- Frequency branch: real + fake causal graphs ---------------------
        self.causal_freq = BranchCausalPair(
            d=self.d_freq, dag_cfg=dag_cfg, disc_cfg=disc_cfg,
            branch_name='frequency', direction_mask=freq_dir_mask)

        # -- Part A: Intra-semantic causal graph (211-node SCM pair) ----------
        sem_graph_cfg = causal_cfg.get('semantic_graph', {})
        self.use_semantic_graph = sem_graph_cfg.get('enabled', False)
        self.causal_semantic = None
        if self.use_semantic_graph:
            sem_dag_cfg = {
                'hidden_dim': sem_graph_cfg.get('hidden_dim', 64),
                'num_layers': sem_graph_cfg.get('num_layers', 2),
            }
            self.causal_semantic = BranchCausalPair(
                d=self.s_dim, dag_cfg=sem_dag_cfg, disc_cfg=disc_cfg,
                branch_name='semantic')
            logger.info(
                f"[CausalDiscoveryModule] Semantic graph: d={self.s_dim}, "
                f"hidden={sem_dag_cfg['hidden_dim']}, layers={sem_dag_cfg['num_layers']}")

        self.sparsity_weight = disc_cfg.get('sparsity_penalty', 0.01)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"[CausalDiscoveryModule] Per-branch dual-graph: "
            f"d_spatial={self.d_spatial} (z={self.z_spatial_dim}, s={self.s_dim}), "
            f"d_freq={self.d_freq} (z={self.z_freq_dim}, s={self.s_dim}), "
            f"SCM hidden={dag_cfg['hidden_dim']}, layers={dag_cfg['num_layers']}, "
            f"trainable={n_params:,}"
        )

    # ------------------------------------------------------------------ #
    #  Input construction                                                  #
    # ------------------------------------------------------------------ #

    def _build_branch_input(
        self,
        z_branch: Optional[torch.Tensor],
        semantic_attrs: Optional[torch.Tensor],
        selector: SparseFeatureSelector,
        z_norm: nn.LayerNorm,
        z_dim: int,
    ) -> torch.Tensor:
        """
        Build causal input for one branch: [z_active, s_semantic].
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
            s_sem = self.s_norm(semantic_attrs.float())
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
        Per-branch dual-graph causal discovery.

        Returns: dict with keys (prefixed by branch):
            residuals_spatial_real/fake (B, d_spatial) per-branch residuals
            residuals_freq_real/fake    (B, d_freq)   per-branch residuals
            residuals_sem_real/fake     (B, s_dim)    semantic residuals [if enabled]
            A_spatial_real/fake     (d_spatial, d_spatial) [if return_graph]
            A_freq_real/fake        (d_freq, d_freq)      [if return_graph]
            A_sem_real/fake         (s_dim, s_dim)        [if return_graph + semantic]
        """
        # Build per-branch causal inputs
        x_spatial = self._build_branch_input(
            z_spatial, semantic_attrs,
            self.spatial_selector, self.z_spatial_norm, self.z_spatial_dim)
        x_freq = self._build_branch_input(
            z_freq, semantic_attrs,
            self.freq_selector, self.z_freq_norm, self.z_freq_dim)

        # Enable grad for Jacobian computation
        if not x_spatial.requires_grad:
            x_spatial = x_spatial.requires_grad_(True)
        if not x_freq.requires_grad:
            x_freq = x_freq.requires_grad_(True)

        # Per-branch causal discovery
        spatial_out = self.causal_spatial(x_spatial, label, return_graph)
        freq_out = self.causal_freq(x_freq, label, return_graph)

        out = {
            # Per-branch residuals (used by CausalViolationAttentionFusion)
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

        # -- Part A: Intra-semantic causal graph (211-node SCM pair) ----------
        if self.use_semantic_graph and self.causal_semantic is not None and semantic_attrs is not None:
            s_normed = self.s_norm(semantic_attrs.float())
            if not s_normed.requires_grad:
                s_normed = s_normed.requires_grad_(True)
            sem_out = self.causal_semantic(s_normed, label, return_graph)
            out['residuals_sem_real'] = sem_out['residuals_real']
            out['residuals_sem_fake'] = sem_out['residuals_fake']
            if return_graph:
                out['A_sem_real'] = sem_out['A_real']
                out['A_sem_fake'] = sem_out['A_fake']

        return out

    # ------------------------------------------------------------------ #
    #  Interpretability helpers                                            #
    # ------------------------------------------------------------------ #

    def get_graph_divergence(self, branch: str = 'spatial') -> Dict[str, torch.Tensor]:
        """Interpretable comparison of real vs fake causal graphs for a branch."""
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
        """Human-readable names for the causal variable nodes of a branch."""
        names = []
        if branch == 'spatial':
            for i in range(self.z_spatial_dim):
                names.append(f'z_spatial_{i}')
        else:
            for i in range(self.z_freq_dim):
                names.append(f'z_freq_{i}')

        if self._semantic_attr_names and len(self._semantic_attr_names) == self.s_dim:
            names.extend(self._semantic_attr_names)
        else:
            for i in range(self.s_dim):
                names.append(f's_{i}')
        return names

    def get_semantic_node_names(self) -> list:
        """Human-readable names for the 211 semantic-only causal graph nodes."""
        if self._semantic_attr_names and len(self._semantic_attr_names) == self.s_dim:
            return list(self._semantic_attr_names)
        return [f's_{i}' for i in range(self.s_dim)]

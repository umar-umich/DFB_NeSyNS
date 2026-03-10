"""
networks/nesy_defake/causal/causal_discovery_module.py
=======================================================
Module 3: Per-Branch Dual-Graph Causal Discovery — DAGMA-DCE for Deepfake Detection

ARCHITECTURE: FOUR CAUSAL GRAPHS (2 branches × 2 distributions)
----------------------------------------------------------------
Each latent branch (spatial, frequency) has its own pair of real/fake causal
graphs that discover relationships between that branch's SAE features and
the shared semantic facial attributes:

  Spatial branch (d_spatial = z_spatial_dim + semantic_dim):
    - A_spatial_real: spatial ↔ semantic causality in REAL faces
      (genuine spatial-semantic biomechanics, e.g. smooth_skin → no_wrinkles)
    - A_spatial_fake: spatial ↔ semantic causality in FAKE faces
      (spatial generator artifacts, e.g. blending boundary → skin texture mismatch)

  Frequency branch (d_freq = z_freq_dim + semantic_dim):
    - A_freq_real: frequency ↔ semantic causality in REAL faces
      (natural frequency-semantic structure, e.g. high_freq → hair_texture)
    - A_freq_fake: frequency ↔ semantic causality in FAKE faces
      (frequency artifacts, e.g. GAN spectral peak → face_region)

WHY PER-BRANCH GRAPHS
---------------------
1. Interpretability: see exactly what each generator breaks in spatial vs
   frequency domains, and how those relate to facial semantics.
2. Smaller causal spaces: d=256 per branch (vs d=384 combined), making
   DAGMA Jacobian computation cheaper per graph.
3. Orthogonal signals: spatial and frequency artifacts manifest differently;
   separate graphs let the SCMs specialize.

DETECTION SIGNAL
----------------
Four violation/conformance scores, all pushing high for fakes:
  v_spatial_real: fakes violate spatial-semantic biomechanics
  v_spatial_fake: fakes conform to spatial generator artifact patterns
  v_freq_real:    fakes violate frequency-semantic structure
  v_freq_fake:    fakes conform to frequency artifact patterns

Aggregated: v_real = v_spatial_real + v_freq_real
            v_fake = v_spatial_fake + v_freq_fake

Per-branch residuals → violation_proj_{spatial,freq}_{real,fake} → classifier.

GRAPH DIVERGENCE (interpretability)
-------------------------------------
Per-branch: A_spatial_real - A_spatial_fake shows what generators break/create
in the spatial domain. A_freq_real - A_freq_fake for frequency domain.

WHY DAGMA-DCE (Waxman et al., OJSP 2024)
-----------------------------------------
Edge weights = RMS Jacobian ∂f_j/∂x_i over the data distribution.
Interpretable as actual causal strength, not arbitrary network weights.

INTEGRATION WITH DETECTOR
--------------------------
causal_out = self.causal_module(
    z_spatial=z_spatial,       # (B, 4096) spatial SAE sparse features
    z_freq=z_freq,             # (B, 4096) frequency SAE sparse features
    semantic_attrs=s_sem,      # (B, 128) semantic attributes
    label=label,               # (B,) used to route real/fake Jacobians
    return_graph=True,
)
# causal_out keys: v_spatial_real, v_spatial_fake, v_freq_real, v_freq_fake,
#   residuals_spatial_real/fake, residuals_freq_real/fake,
#   A_spatial_real/fake, A_freq_real/fake, v_real, v_fake (aggregated)
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

    Uses the log-determinant characterization from:
        Bello et al., "DAGMA: Learning DAGs via M-matrices and a Log-Determinant
        Acyclicity Characterization", NeurIPS 2022.

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

        CHUNK = 32
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
# Violation / Conformance Scorer
# ---------------------------------------------------------------------------

class ViolationScorer(nn.Module):
    """
    Maps per-node SCM residuals -> scalar score (B,).

    Used for two purposes:
      1. violation_scorer: high when sample VIOLATES real-face causal structure
      2. conformance_scorer: high when sample CONFORMS to fake-face artifact structure

    Both learn to output high for fakes and low for reals, supervised by
    their respective contrastive losses. The in-degree weighting focuses
    attention on highly-connected nodes where violations are most informative.
    """

    def __init__(self, d: int, hidden_dim: int = 64):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor,
                A_dce: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:     (B, d) true causal variable values
            x_hat: (B, d) SCM reconstruction
            A_dce: (d, d) adjacency for in-degree weighting (optional)
        Returns:
            score: (B,) scalar score per sample
        """
        node_errors = (x - x_hat) ** 2     # (B, d)

        if A_dce is not None:
            in_degree = A_dce.sum(dim=0).detach()           # (d,)
            in_degree = in_degree / (in_degree.mean() + 1e-8)
            node_errors = node_errors * in_degree.unsqueeze(0)

        return self.scorer(node_errors).squeeze(-1)         # (B,)


# ---------------------------------------------------------------------------
# Per-branch causal graph pair (real + fake)
# ---------------------------------------------------------------------------

class BranchCausalPair(nn.Module):
    """
    One branch's real + fake causal graph pair.

    Contains:
      - causal_learner_real:  DAGMA-DCE for real-face distribution
      - causal_learner_fake:  DAGMA-DCE for fake-face distribution
      - violation_scorer_real: scores violation of real-face structure
      - conformance_scorer_fake: scores conformance to fake artifact structure
    """

    def __init__(self, d: int, dag_cfg: dict, disc_cfg: dict, branch_name: str):
        super().__init__()
        self.d = d
        self.branch_name = branch_name

        self.causal_learner_real = DAGMADCELearner(
            d=d,
            hidden_dim=dag_cfg['hidden_dim'],
            num_layers=dag_cfg['num_layers'],
            sparsity_penalty=disc_cfg.get('sparsity_penalty', 0.01),
            s=disc_cfg.get('dagma_s', 1.0),
        )
        self.violation_scorer_real = ViolationScorer(
            d=d,
            hidden_dim=dag_cfg.get('violation_hidden_dim', 64),
        )

        self.causal_learner_fake = DAGMADCELearner(
            d=d,
            hidden_dim=dag_cfg['hidden_dim'],
            num_layers=dag_cfg['num_layers'],
            sparsity_penalty=disc_cfg.get('sparsity_penalty_fake',
                                          disc_cfg.get('sparsity_penalty', 0.01) * 2),
            s=disc_cfg.get('dagma_s', 1.0),
        )
        self.conformance_scorer_fake = ViolationScorer(
            d=d,
            hidden_dim=dag_cfg.get('violation_hidden_dim', 64),
        )

    def forward(
        self,
        x: torch.Tensor,
        label: Optional[torch.Tensor],
        return_graph: bool = False,
    ) -> Dict:
        """
        Run real + fake SCM on input x, compute violation/conformance scores.

        Args:
            x:     (B, d) causal variables for this branch
            label: (B,) 0=real, 1=fake for label-restricted Jacobian
            return_graph: include adjacency matrices in output
        Returns:
            dict with v_real, v_fake, residuals_real, residuals_fake,
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

        if self.training and label is not None:
            real_mask = (label == 0)
            fake_mask = (label == 1)

            if real_mask.sum() > 1:
                x_real = x[real_mask].detach().requires_grad_(True)
                A_real = self.causal_learner_real.compute_adjacency_dce(x_real)

            if fake_mask.sum() > 1:
                x_fake = x[fake_mask].detach().requires_grad_(True)
                A_fake = self.causal_learner_fake.compute_adjacency_dce(x_fake)

        # Scoring
        v_real = self.violation_scorer_real(x, x_hat_real, A_real)
        v_fake = self.conformance_scorer_fake(x, x_hat_fake, A_fake)

        out = {
            'v_real': v_real,
            'v_fake': v_fake,
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
    DAGMA-DCE causal graphs operating on [z_branch_active, semantic_attrs]:

      Spatial branch (d_spatial = z_spatial_dim + semantic_dim):
        causal_spatial.causal_learner_real -> A_spatial_real
        causal_spatial.causal_learner_fake -> A_spatial_fake

      Frequency branch (d_freq = z_freq_dim + semantic_dim):
        causal_freq.causal_learner_real -> A_freq_real
        causal_freq.causal_learner_fake -> A_freq_fake

    Semantic features are shared across both branches but each branch
    discovers its own causal structure with the semantics.

    Forward returns per-branch violation scores, residuals, and graphs.
    Aggregated v_real/v_fake are also provided for compatibility.
    """

    def __init__(self, config: dict):
        super().__init__()

        causal_cfg = config['causal_module']
        lv_cfg = causal_cfg['latent_variables']
        dag_cfg = causal_cfg['dag_learning']
        disc_cfg = causal_cfg['discovery']

        self.z_spatial_dim = lv_cfg['z_spatial_dim']      # 128
        self.z_freq_dim = lv_cfg['z_frequency_dim']       # 128
        self.s_dim = causal_cfg['semantic_dim']            # 128
        self.d_spatial = self.z_spatial_dim + self.s_dim   # 256
        self.d_freq = self.z_freq_dim + self.s_dim        # 256

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

        # -- Spatial branch: real + fake causal graphs -----------------------
        self.causal_spatial = BranchCausalPair(
            d=self.d_spatial, dag_cfg=dag_cfg, disc_cfg=disc_cfg,
            branch_name='spatial')

        # -- Frequency branch: real + fake causal graphs ---------------------
        self.causal_freq = BranchCausalPair(
            d=self.d_freq, dag_cfg=dag_cfg, disc_cfg=disc_cfg,
            branch_name='frequency')

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

        Args:
            z_spatial:      (B, 4096) spatial SAE sparse features
            z_freq:         (B, 4096) frequency SAE sparse features
            semantic_attrs: (B, semantic_dim) facial semantic attributes
            label:          (B,) 0=real, 1=fake for label-restricted Jacobians
            return_graph:   include adjacency matrices in output

        Returns: dict with keys (prefixed by branch):
            v_spatial_real/fake     (B,)   per-branch violation/conformance
            v_freq_real/fake        (B,)   per-branch violation/conformance
            residuals_spatial_real/fake (B, d_spatial) per-branch residuals
            residuals_freq_real/fake    (B, d_freq)   per-branch residuals
            A_spatial_real/fake     (d_spatial, d_spatial) [if return_graph]
            A_freq_real/fake        (d_freq, d_freq)      [if return_graph]
            v_real, v_fake          (B,)   aggregated scores (backward compat)
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

        # Aggregated scores (backward compatible)
        v_real = spatial_out['v_real'] + freq_out['v_real']
        v_fake = spatial_out['v_fake'] + freq_out['v_fake']

        out = {
            # Per-branch spatial
            'v_spatial_real': spatial_out['v_real'],
            'v_spatial_fake': spatial_out['v_fake'],
            'residuals_spatial_real': spatial_out['residuals_real'],
            'residuals_spatial_fake': spatial_out['residuals_fake'],
            # Per-branch frequency
            'v_freq_real': freq_out['v_real'],
            'v_freq_fake': freq_out['v_fake'],
            'residuals_freq_real': freq_out['residuals_real'],
            'residuals_freq_fake': freq_out['residuals_fake'],
            # Aggregated (backward compat)
            'v_real': v_real,
            'v_fake': v_fake,
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
        """
        Interpretable comparison of real vs fake causal graphs for a branch.

        Args:
            branch: 'spatial' or 'frequency'
        Returns:
            A_real, A_fake, divergence, broken_by_fakes, created_by_fakes
        """
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
        """
        Human-readable names for the causal variable nodes of a branch.

        Spatial branch:  [z_spatial_0..N, s_0..s_M]
        Frequency branch: [z_freq_0..N, s_0..s_M]
        """
        names = []
        if branch == 'spatial':
            for i in range(self.z_spatial_dim):
                names.append(f'z_spatial_{i}')
        else:
            for i in range(self.z_freq_dim):
                names.append(f'z_freq_{i}')
        for i in range(self.s_dim):
            names.append(f's_{i}')
        return names

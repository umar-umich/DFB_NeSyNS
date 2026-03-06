"""
networks/nesy_defake/causal/causal_discovery_module.py
=======================================================
Module 3: Dual-Graph Causal Discovery — DAGMA-DCE for Deepfake Detection

ARCHITECTURE: TWO COMPLEMENTARY GRAPHS
---------------------------------------
Two separate DAGMA-DCE learners operate on the same 329-d causal variable space:

  Graph_real (A_real):
    - SCM_real trained to reconstruct real faces
    - A_real EMA updated from REAL frames only
    - Captures genuine facial biomechanics
      e.g. "smileLeft → cheekSquintLeft → eyeNarrow" holds universally for reals
    - Violation score v_real = how much this sample BREAKS real-face structure
      → HIGH for fakes (they violate biomechanical constraints)
      → LOW  for reals (they follow their own generating process)

  Graph_fake (A_fake):
    - SCM_fake trained to reconstruct fake faces
    - A_fake EMA updated from FAKE frames only
    - Captures systematic generator artifacts
      e.g. "z_freq_23 → jawOpen" (blending artifact spuriously correlated with mouth)
    - Conformance score v_fake = how much this sample CONFORMS to fake structure
      → HIGH for fakes (they exhibit generator-specific patterns)
      → LOW  for reals (they don't follow generator artifact patterns)

DETECTION SIGNAL
----------------
Both scores push in the same direction for fakes (high) vs reals (low),
but through orthogonal mechanisms:

  v_real: absence of biomechanical structure (breaks what reals must satisfy)
  v_fake: presence of generator artifact structure (matches what generators produce)

Combined via node_residuals → violation_proj_{real,fake} → additive fusion into classifier.

GRAPH DIVERGENCE (interpretability)
-------------------------------------
  A_real - A_fake > 0:  edges generators BREAK  (biomechanical constraints)
  A_fake - A_real > 0:  edges generators CREATE  (artificial artifact correlations)

This pinpoints exactly what each deepfake method breaks and introduces.

WHY DAGMA-DCE (Waxman et al., OJSP 2024)
-----------------------------------------
Edge weights = RMS Jacobian ∂f_j/∂x_i over the data distribution.
Interpretable as actual causal strength, not arbitrary network weights.

INTEGRATION WITH DETECTOR
--------------------------
causal_out = self.causal_module(
    z_sae=z_sae,           # (B, 8192) sparse SAE features
    semantic_attrs=s_sem,  # (B, 73)
    label=label,           # (B,) used to route real/fake Jacobians
    return_graph=True,
)
# causal_out is a dict with keys:
#   v_real, v_fake, residuals_real, residuals_fake, A_real, A_fake
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
# Per-node MLP (unused directly, kept for reference)
# ---------------------------------------------------------------------------

class NodeMLP(nn.Module):
    """Single-node MLP: f_j: R^d → R. Kept for reference; BatchedSCM is used instead."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        layers = []
        d_in = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(d_in, hidden_dim))
            layers.append(nn.Sigmoid())
            d_in = hidden_dim
        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Shared SCM: all d nodes modeled by a single batched MLP
# ---------------------------------------------------------------------------

class BatchedSCM(nn.Module):
    """
    Structural Causal Model for d variables.

    Shared trunk MLP + d output heads. More efficient than d separate NodeMLPs.
    Sigmoid activations (not ReLU) for everywhere-differentiable Jacobians.

    Architecture:
        x (B, d) → trunk [Linear→Sigmoid]×(L-1) → h (B, hidden) → heads Linear(hidden, d) → x_hat (B, d)
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
        """(B, d) → (B, d)"""
        return self.heads(self.trunk(x))


# ---------------------------------------------------------------------------
# DAGMA-DCE graph learner
# ---------------------------------------------------------------------------

class DAGMADCELearner(nn.Module):
    """
    DAGMA-DCE causal graph learner for one distribution (real OR fake).

    Learns A_dce where A_ij = sqrt( E_x[(∂f_j/∂x_i)²] ) — the RMS Jacobian
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

        # Factored DCE: A_sq[i,j] ≈ Σ_k W[j,k]² · E[(∂h_k/∂x_i)²]
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
# Feature compression: Z_sae (8192 sparse) → Z_active (256 dense)
# ---------------------------------------------------------------------------

class SparseFeatureSelector(nn.Module):
    """
    Linear 8192→256 projection: picks out causally relevant SAE dictionary elements.

    No activation — keeps the projection interpretable as a linear combination
    of monosemantic features, so each z_active dimension maps to a fixed set
    of dictionary atoms that the causal graph can reason about consistently.
    """

    def __init__(self, sae_dict_size: int, output_dim: int, num_branches: int = 2):
        super().__init__()
        total_sae_dim = sae_dict_size * num_branches
        self.proj = nn.Linear(total_sae_dim, output_dim, bias=False)
        nn.init.normal_(self.proj.weight, std=1.0 / math.sqrt(total_sae_dim))
        logger.info(f"[SparseFeatureSelector] {total_sae_dim} → {output_dim} (linear, no bias)")

    def forward(self, z_sae: torch.Tensor) -> torch.Tensor:
        return self.proj(z_sae)


# ---------------------------------------------------------------------------
# Violation / Conformance Scorer
# ---------------------------------------------------------------------------

class ViolationScorer(nn.Module):
    """
    Maps per-node SCM residuals → scalar score (B,).

    Used for two purposes:
      1. violation_scorer_real: high when sample VIOLATES real-face causal structure
      2. conformance_scorer_fake: high when sample CONFORMS to fake-face artifact structure

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
# Main module: CausalDiscoveryModule (dual-graph)
# ---------------------------------------------------------------------------

class CausalDiscoveryModule(nn.Module):
    """
    Module 3: Dual-Graph Causal Discovery for deepfake detection.

    Two parallel DAGMA-DCE learners operating on the same 329-d causal
    variable space (256 active SAE features + 73 semantic attributes):

      causal_learner_real  → A_real: real-face biomechanical DAG
      causal_learner_fake  → A_fake: fake-face generator artifact DAG

    Forward returns:
      v_real       (B,) — violation of real-face causal structure  [high=fake]
      v_fake       (B,) — conformance to fake-face artifact structure [high=fake]
      residuals_real (B, d) — x - SCM_real(x), fed to violation_proj_real
      residuals_fake (B, d) — x - SCM_fake(x), fed to violation_proj_fake
      A_real / A_fake       — DCE adjacency matrices (interpretable)

    Graph divergence analysis:
      get_graph_divergence() returns A_real - A_fake (broken edges)
                             and A_fake - A_real (created edges)
    """

    def __init__(self, config: dict):
        super().__init__()

        causal_cfg = config['causal_module']
        lv_cfg = causal_cfg['latent_variables']
        dag_cfg = causal_cfg['dag_learning']
        disc_cfg = causal_cfg['discovery']

        self.z_active_dim = lv_cfg['total_sae_dim']   # 256
        self.s_dim = causal_cfg['semantic_dim']        # 73
        self.d = self.z_active_dim + self.s_dim        # 329

        sae_cfg = config.get('sparse_features', {})
        self.sae_dict_size = sae_cfg.get(
            'dict_size',
            sae_cfg.get('sparse_autoencoder', {}).get('input_dim', 1024) * 4,
        )

        # ── Shared input pipeline (both graphs see the same x) ────────────
        self.feature_selector = SparseFeatureSelector(
            sae_dict_size=self.sae_dict_size,
            output_dim=self.z_active_dim,
            num_branches=2,
        )
        self.z_norm = nn.LayerNorm(self.z_active_dim)
        self.s_norm = nn.LayerNorm(self.s_dim)

        # ── Graph 1: Real-face biomechanical DAG ──────────────────────────
        # SCM_real trained on real frames; A_real EMA from real frames only.
        # v_real is high when a sample VIOLATES real-face causal structure.
        self.causal_learner_real = DAGMADCELearner(
            d=self.d,
            hidden_dim=dag_cfg['hidden_dim'],
            num_layers=dag_cfg['num_layers'],
            sparsity_penalty=disc_cfg.get('sparsity_penalty', 0.01),
            s=disc_cfg.get('dagma_s', 1.0),
        )
        self.violation_scorer_real = ViolationScorer(
            d=self.d,
            hidden_dim=dag_cfg.get('violation_hidden_dim', 64),
        )

        # ── Graph 2: Fake-face generator artifact DAG ─────────────────────
        # SCM_fake trained on fake frames; A_fake EMA from fake frames only.
        # v_fake is high when a sample CONFORMS to fake-face artifact structure.
        # Lower dag_penalty_weight than real graph: fake graph is less universal
        # (trained fakes are a proxy for generator artifacts, not ground truth).
        self.causal_learner_fake = DAGMADCELearner(
            d=self.d,
            hidden_dim=dag_cfg['hidden_dim'],
            num_layers=dag_cfg['num_layers'],
            sparsity_penalty=disc_cfg.get('sparsity_penalty_fake',
                                          disc_cfg.get('sparsity_penalty', 0.01) * 2),
            s=disc_cfg.get('dagma_s', 1.0),
        )
        self.conformance_scorer_fake = ViolationScorer(
            d=self.d,
            hidden_dim=dag_cfg.get('violation_hidden_dim', 64),
        )

        self.sparsity_weight = disc_cfg.get('sparsity_penalty', 0.01)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"[CausalDiscoveryModule] Dual-graph: d={self.d} "
            f"(z={self.z_active_dim}, s={self.s_dim}), "
            f"SCM hidden={dag_cfg['hidden_dim']}, layers={dag_cfg['num_layers']}, "
            f"trainable={n_params:,}"
        )

    # ------------------------------------------------------------------ #
    #  Input construction                                                  #
    # ------------------------------------------------------------------ #

    def _build_causal_input(
        self,
        z_sae: Optional[torch.Tensor],
        semantic_attrs: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Build x (B, d): LayerNorm-normalised [z_active (256), s_semantic (73)].
        """
        B = z_sae.shape[0] if z_sae is not None else semantic_attrs.shape[0]
        device = z_sae.device if z_sae is not None else semantic_attrs.device

        if z_sae is not None:
            z_active = self.z_norm(self.feature_selector(z_sae))
        else:
            z_active = torch.zeros(B, self.z_active_dim, device=device)

        if semantic_attrs is not None:
            s_sem = self.s_norm(semantic_attrs.float())
        else:
            s_sem = torch.zeros(B, self.s_dim, device=device)

        return torch.cat([z_active, s_sem], dim=1)     # (B, d)

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        z_sae: Optional[torch.Tensor] = None,
        semantic_attrs: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        return_graph: bool = False,
    ) -> Dict:
        """
        Args:
            z_sae:          (B, 8192) sparse SAE features
            semantic_attrs: (B, 73) per-frame attributes
            label:          (B,) integer labels (0=real, 1=fake).
                            Required for label-restricted Jacobian computation
                            and EMA routing during training.
            return_graph:   if True, include A_real / A_fake in returned dict.

        Returns: dict with keys
            v_real        (B,)    — violation score on real graph  [high=fake]
            v_fake        (B,)    — conformance score on fake graph [high=fake]
            residuals_real (B, d) — x − SCM_real(x)
            residuals_fake (B, d) — x − SCM_fake(x)
            A_real         (d,d)  — real-graph adjacency [only if return_graph]
            A_fake         (d,d)  — fake-graph adjacency [only if return_graph]
        """
        x = self._build_causal_input(z_sae, semantic_attrs)    # (B, d)
        # Enable grad for Jacobian computation. We do NOT detach from upstream
        # so that gradients flow back through the SAE and backbone, enabling
        # end-to-end learning of causally informative features.
        if not x.requires_grad:
            x = x.requires_grad_(True)

        # ── SCM reconstructions (all frames for full-batch scoring) ───────
        x_hat_real = self.causal_learner_real.scm(x)       # (B, d)
        x_hat_fake = self.causal_learner_fake.scm(x)       # (B, d)
        residuals_real = x - x_hat_real                    # (B, d)
        residuals_fake = x - x_hat_fake                    # (B, d)

        # ── Graph discovery (label-restricted Jacobians) ──────────────────
        # A_real updated from real frames only: learns pure biomechanical structure.
        # A_fake updated from fake frames only: learns pure generator artifact structure.
        # Falls back to EMA when subset is too small (< 2 samples) or at inference.
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

        # ── Scoring ───────────────────────────────────────────────────────
        # v_real: ViolationScorer on real-graph residuals.
        #   Contrastive supervision ensures: high for fakes, low for reals.
        v_real = self.violation_scorer_real(x, x_hat_real, A_real)     # (B,)

        # v_fake: ConformanceScorer on fake-graph residuals.
        #   Same scorer architecture; contrastive supervision trains it to output
        #   HIGH for fakes (they have LOW residuals on the fake SCM → conform to
        #   generator patterns) and LOW for reals (high fake-SCM residuals).
        v_fake = self.conformance_scorer_fake(x, x_hat_fake, A_fake)   # (B,)

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

    # ------------------------------------------------------------------ #
    #  Interpretability helpers                                            #
    # ------------------------------------------------------------------ #

    def get_graph_divergence(self) -> Dict[str, torch.Tensor]:
        """
        Interpretable comparison of real vs fake causal graphs.

        Returns:
            A_real:          (d, d) real-face biomechanical DAG
            A_fake:          (d, d) fake-face generator artifact DAG
            divergence:      (d, d) signed A_real − A_fake
            broken_by_fakes: (d, d) edges strong in reals, absent in fakes
                             = causal relationships generators systematically violate
            created_by_fakes:(d, d) edges absent in reals, present in fakes
                             = artificial correlations generators introduce
        """
        A_r = self.causal_learner_real._A_dce_ema
        A_f = self.causal_learner_fake._A_dce_ema
        return {
            'A_real':           A_r.clone(),
            'A_fake':           A_f.clone(),
            'divergence':       (A_r - A_f).clone(),
            'broken_by_fakes':  (A_r - A_f).clamp(min=0).clone(),
            'created_by_fakes': (A_f - A_r).clamp(min=0).clone(),
        }

    def get_causal_graph(self) -> torch.Tensor:
        """Return real-face graph EMA (primary graph, backward compat)."""
        return self.causal_learner_real._A_dce_ema.clone()

    def get_node_names(self) -> list:
        """
        Human-readable names for the 329 causal variable nodes.
          [0:128]   z_spatial_0..127
          [128:256] z_freq_0..127
          [256:329] s_0..s_72
        """
        names = []
        half = self.z_active_dim // 2
        for i in range(half):
            names.append(f'z_spatial_{i}')
        for i in range(half):
            names.append(f'z_freq_{i}')
        for i in range(self.s_dim):
            names.append(f's_{i}')
        return names

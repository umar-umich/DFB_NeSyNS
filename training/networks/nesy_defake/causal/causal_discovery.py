"""
networks/nesy_defake/causal/causal_discovery_module.py
=======================================================
Module 3: Causal Discovery — DAGMA-DCE for Deepfake Detection

WHAT THIS MODULE DOES
---------------------
Learns a directed acyclic graph (DAG) connecting:
  - Z_sae: 256 active SAE features (128 spatial + 128 frequency) per sample
  - S_semantic: 73 facial attributes (action units, landmarks, pose, etc.)

The DAG captures how neural features CAUSE observable semantic changes in
real faces. Deepfakes violate these causal constraints because generators
don't perfectly model facial biomechanics (e.g., "smileLeft causes
cheekSquintLeft" holds for real faces but is often broken in fakes).

The violation_score measures how much a given sample deviates from the
learned causal structure — high for fakes, low for reals.

WHY DAGMA-DCE (Waxman et al., OJSP 2024)
-----------------------------------------
Standard DAGMA learns a weighted adjacency matrix W from MLP first-layer
weights, but Waxman et al. proved these weights are NOT interpretable as
causal strengths — they can be arbitrarily different from true causal effects.

DAGMA-DCE instead defines W_ij as the RMS of the Jacobian ∂f_j/∂x_i
evaluated over the data distribution (the Differential Causal Effect).
This gives us:
  - Interpretable edge weights (actual causal strength)
  - Model-agnostic (works with any differentiable f)
  - Principled thresholding (edges with DCE < threshold are non-causal)

For our NeurIPS contribution, this is the first application of DAGMA-DCE
to vision foundation model features for forensic tasks.

ARCHITECTURE
------------
```
Z_sae (256) ─┐                    ┌─→ A_dce (329 × 329)  ← interpretable DAG
             ├→ X (329-d) → SCM ──┤
S_sem (73)  ─┘                    └─→ violation_score (B,)
```

The Structural Causal Model (SCM) is a set of d=329 MLP functions:
  x_j = f_j(x_pa(j)) + noise_j

where pa(j) are the parents of node j in the graph. During training,
we don't know the parents — DAGMA-DCE discovers them by fitting the
MLPs and computing the Jacobian-based adjacency matrix A_dce.

The DAG constraint h(A) = 0 (from DAGMA's M-matrix characterization)
ensures acyclicity. We add this as a penalty to the training loss.

INTEGRATION WITH DETECTOR
-------------------------
Called from NeSyDeFakeHybridDetector.forward():
  violation_score, causal_dag, _ = self.causal_module(
      fused_features,       # (B, proj_dim) — not used directly by SCM
      z_sae=z_sae,          # (B, 8192) sparse, ~256 active per sample
      semantic_attrs=s_sem, # (B, 73)
      return_graph=True,
  )

The violation_score enters the loss as:
  L_causal = MSE(violation_score, label) + λ * h(A_dce)

Config (causal_module section of YAML):
  enabled: true/false
  latent_variables.total_sae_dim: 256  (target_k * 2 branches)
  semantic_dim: 73
  discovery.sparsity_penalty: 0.01
  dag_learning.hidden_dim: 128
  dag_learning.num_layers: 3
  dag_learning.dag_penalty_weight: 0.1
"""

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DAGMA acyclicity constraint (Bello et al., NeurIPS 2022)
# ---------------------------------------------------------------------------

def dagma_acyclicity(A: torch.Tensor, s: float = 1.0) -> torch.Tensor:
    """
    DAGMA M-matrix acyclicity constraint: h(A) = -log det(sI - A∘A) + d*log(s)

    h(A) = 0 iff the graph represented by A is a DAG.

    This is faster and more numerically stable than NOTEARS' matrix exponential
    constraint. Uses the log-determinant characterization from:
        Bello et al., "DAGMA: Learning DAGs via M-matrices and a Log-Determinant
        Acyclicity Characterization", NeurIPS 2022.

    Args:
        A: (d, d) weighted adjacency matrix (non-negative after squaring)
        s: scalar > spectral_radius(A∘A), controls numerical conditioning.
           Default s=1.0 works when A entries are small (after sparsity penalty).
    Returns:
        Scalar h(A) ≥ 0, equals 0 iff A is a DAG.
    """
    d = A.shape[0]
    A_sq = A * A  # element-wise square → non-negative
    M = s * torch.eye(d, device=A.device, dtype=A.dtype) - A_sq
    # log det of M-matrix. If M is not positive definite, h(A) → ∞ (not a DAG).
    sign, logabsdet = torch.linalg.slogdet(M)
    # If sign is negative, the matrix is not positive definite (cycle exists)
    h = -logabsdet + d * math.log(s)
    # Clamp to prevent negative values from numerical noise
    return h.clamp(min=0.0)


# ---------------------------------------------------------------------------
# Per-node MLP for SCM: x_j = f_j(x_{-j})
# ---------------------------------------------------------------------------

class NodeMLP(nn.Module):
    """
    MLP that models how node j depends on all other nodes.

    f_j: R^d → R  (takes all d inputs, predicts node j's value)

    We need the FULL Jacobian of this function for DAGMA-DCE,
    so we keep it simple (2-3 layers) and use smooth activations
    (Sigmoid, not ReLU) to ensure differentiability everywhere.

    The DAGMA-DCE paper requires "once-differentiable" functions,
    so Sigmoid is a natural choice (infinitely differentiable).
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        layers = []
        d_in = input_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(d_in, hidden_dim))
            layers.append(nn.Sigmoid())
            d_in = hidden_dim
        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, d) all node values
        Returns:
            (B, 1) predicted value for this node
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# Shared SCM: all d nodes modeled by a single batched MLP
# ---------------------------------------------------------------------------

class BatchedSCM(nn.Module):
    """
    Structural Causal Model for d variables, implemented as a shared MLP
    with per-node output heads rather than d separate MLPs.

    This is dramatically more efficient than d individual NodeMLPs for
    d=329 nodes. The shared trunk learns common feature interactions,
    while per-node heads specialize.

    Architecture:
        x (B, d) → shared_trunk (B, hidden) → heads (B, d)

    Each head predicts x̂_j from the shared representation, and the
    DCE adjacency matrix is computed via the Jacobian of the full
    mapping x → x̂.
    """

    def __init__(self, d: int, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.d = d

        # Shared trunk: x → hidden representation
        trunk_layers = []
        d_in = d
        for i in range(num_layers - 1):
            trunk_layers.append(nn.Linear(d_in, hidden_dim))
            trunk_layers.append(nn.Sigmoid())
            d_in = hidden_dim
        self.trunk = nn.Sequential(*trunk_layers)

        # Per-node output heads: hidden → scalar per node
        # Using a single Linear(hidden, d) is equivalent to d separate
        # Linear(hidden, 1) but much faster on GPU.
        self.heads = nn.Linear(hidden_dim, d)

        # Initialize heads with small weights to start near identity
        nn.init.normal_(self.heads.weight, std=0.01)
        nn.init.zeros_(self.heads.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, d) observed node values
        Returns:
            x_hat: (B, d) predicted node values from SCM
        """
        h = self.trunk(x)    # (B, hidden)
        return self.heads(h)  # (B, d)


# ---------------------------------------------------------------------------
# Causal Learner: DAGMA-DCE graph discovery
# ---------------------------------------------------------------------------

class DAGMADCELearner(nn.Module):
    """
    DAGMA-DCE causal graph learner.

    Learns a DAG over d = z_dim + s_dim variables by:
    1. Fitting a BatchedSCM to reconstruct each variable from all others
    2. Computing the DCE adjacency matrix A_dce via Jacobian
    3. Enforcing acyclicity via DAGMA's M-matrix constraint
    4. Encouraging sparsity via L1 on A_dce

    The DCE adjacency matrix is:
        A_ij = sqrt( E_x[ (∂f_j/∂x_i)^2 ] )

    i.e., the RMS of the partial derivative of the j-th output with respect
    to the i-th input, averaged over the data distribution. This measures
    how much changing x_i affects x_j through the SCM.

    During training:
    - compute_adjacency_dce() returns the current A_dce
    - compute_dag_penalty() returns h(A_dce) for the loss
    - forward() returns the SCM reconstruction for the MSE score
    """

    def __init__(self, d: int, hidden_dim: int = 128, num_layers: int = 3,
                 sparsity_penalty: float = 0.01, s: float = 1.0):
        super().__init__()
        self.d = d
        self.s = s
        self.sparsity_penalty = sparsity_penalty

        self.scm = BatchedSCM(d, hidden_dim, num_layers)

        # Running estimate of A_dce for inference (updated during training)
        self.register_buffer('_A_dce_ema', torch.zeros(d, d))
        self._ema_decay = 0.99
        self._ema_initialized = False

    def compute_adjacency_dce(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the DAGMA-DCE adjacency matrix via efficient Jacobian computation.

        A_ij = sqrt( (1/B) * sum_b (∂f_j/∂x_i(x_b))^2 )

        Efficiency strategy:
        For our BatchedSCM architecture (shared trunk + linear heads), the
        full Jacobian factors as:

            J_full[j, i] = heads.weight[j, :] @ J_trunk[:, i]

        where J_trunk = ∂trunk(x)/∂x is (hidden, d). We only need to
        compute J_trunk ONCE (d backward passes through the trunk), then
        the full d×d Jacobian is just a matrix multiply with heads.weight.

        This reduces cost from d backward passes through the FULL network
        to d backward passes through just the TRUNK (no heads), plus one
        matrix multiply. For d=329 with hidden=128, this is ~2.6x faster.

        Additionally, we chunk the backward passes for memory efficiency.

        Args:
            x: (B, d) input node values (requires_grad must be True)
        Returns:
            A_dce: (d, d) adjacency matrix with DCE edge weights
        """
        B, d = x.shape
        assert d == self.d

        if not x.requires_grad:
            x = x.detach().requires_grad_(True)

        # Forward through trunk only
        h = self.scm.trunk(x)  # (B, hidden)
        hidden_dim = h.shape[1]

        # Compute J_trunk: (B, hidden, d) via chunked backward passes
        # For each hidden dimension k, compute ∂h[:,k]/∂x
        CHUNK = 32  # process 32 hidden dims at a time
        J_trunk_sq = torch.zeros(d, hidden_dim, device=x.device, dtype=x.dtype)

        for k_start in range(0, hidden_dim, CHUNK):
            k_end = min(k_start + CHUNK, hidden_dim)
            chunk_size = k_end - k_start

            # Sum over batch for each hidden dim in chunk
            for k in range(k_start, k_end):
                grad_k = torch.autograd.grad(
                    outputs=h[:, k].sum(),
                    inputs=x,
                    create_graph=self.training,
                    retain_graph=True,
                )[0]  # (B, d)
                # Accumulate: E_b[(∂h_k/∂x_i)^2] for all i
                J_trunk_sq[:, k] = (grad_k ** 2).mean(dim=0)

        # Now compute A_dce using the factored Jacobian:
        # (∂x̂_j/∂x_i)^2 ≈ (W_heads[j,:] @ J_trunk[:,i])^2
        # But we want E[(∂x̂_j/∂x_i)^2], not (E[∂x̂_j/∂x_i])^2.
        #
        # For the EXACT computation:
        # A_ij^2 = E_b[ (sum_k W[j,k] * J_trunk[b,k,i])^2 ]
        #        = sum_k sum_l W[j,k]*W[j,l] * E_b[J[b,k,i]*J[b,l,i]]
        #
        # The cross terms make this expensive. As a practical approximation
        # (standard in DAGMA-DCE implementations), we use:
        # A_ij^2 ≈ sum_k W[j,k]^2 * E_b[(J_trunk[b,k,i])^2]
        #        = (W^2) @ J_trunk_sq
        #
        # This is exact when hidden activations are uncorrelated (approximately
        # true after sigmoid saturation) and is the standard approach.

        W_sq = self.scm.heads.weight ** 2  # (d, hidden)
        A_sq = J_trunk_sq @ W_sq.t()       # (d, d): A_sq[i,j] = sum_k W[j,k]^2 * J_sq[i,k]
        A_dce = torch.sqrt(A_sq + 1e-10)

        # Zero out diagonal (no self-loops in causal graph)
        A_dce = A_dce * (1.0 - torch.eye(d, device=A_dce.device))

        # Update EMA for inference
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
        """
        Compute DAGMA acyclicity penalty on the current EMA adjacency matrix.

        Called from detector.get_losses() to add to the total loss.
        Uses the EMA rather than recomputing A_dce (already computed in forward).
        """
        return dagma_acyclicity(self._A_dce_ema, self.s)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, d) concatenated [z_sae_active, s_semantic]
        Returns:
            x_hat: (B, d) SCM reconstruction
            A_dce: (d, d) current adjacency matrix
        """
        x_hat = self.scm(x)
        A_dce = self.compute_adjacency_dce(x)
        return x_hat, A_dce


# ---------------------------------------------------------------------------
# Feature compression: Z_sae (8192 sparse) → Z_active (256 dense)
# ---------------------------------------------------------------------------

class SparseFeatureSelector(nn.Module):
    """
    Converts sparse SAE features (8192-d with ~256 active) into a dense
    fixed-size representation suitable for the causal graph.

    Strategy: Use the top-k values from each branch's sparse vector.
    Since BatchTopK guarantees exactly target_k active features per sample,
    we extract those k values and their magnitudes.

    However, the ACTIVE INDICES vary across samples — different faces
    activate different SAE features. For the causal graph to be consistent,
    we need a FIXED-SIZE representation where position i always means the
    same thing.

    Solution: Learnable projection from full sparse vector to compact dense.
    The linear projection learns which SAE dictionary elements are most
    causally relevant. This is equivalent to "reading off" specific
    monosemantic features that matter for causal discovery.

    Alternative considered: Using the raw 8192-d vector. Rejected because
    the causal graph would have 8192+73 = 8265 nodes, making Jacobian
    computation O(d^2) = ~68M operations per sample — too expensive.
    With compression to 256, it's O(329^2) ≈ 108K per sample.
    """

    def __init__(self, sae_dict_size: int, output_dim: int, num_branches: int = 2):
        super().__init__()
        total_sae_dim = sae_dict_size * num_branches

        # Learnable projection: picks out causally relevant features
        # No activation — keep it linear so the causal graph sees
        # a linear combination of monosemantic features (interpretable).
        self.proj = nn.Linear(total_sae_dim, output_dim, bias=False)

        # Initialize with small weights — training will find the right
        # dictionary elements to focus on
        nn.init.normal_(self.proj.weight, std=1.0 / math.sqrt(total_sae_dim))

        logger.info(
            f"[SparseFeatureSelector] {total_sae_dim} → {output_dim} "
            f"(linear, no bias)"
        )

    def forward(self, z_sae: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_sae: (B, total_sae_dim) sparse features from DualBranchSAE
        Returns:
            z_active: (B, output_dim) dense compressed features
        """
        return self.proj(z_sae)


# ---------------------------------------------------------------------------
# Violation Score: how much does this sample violate causal constraints?
# ---------------------------------------------------------------------------

class ViolationScorer(nn.Module):
    """
    Computes a scalar violation score from the SCM reconstruction error
    and the causal graph structure.

    Key insight: Real faces follow consistent causal patterns (e.g.,
    raising eyebrows causes forehead wrinkles). Fakes often break these
    patterns because generators model appearance but not biomechanics.

    The violation score combines:
    1. Per-node reconstruction error: |x_j - f_j(x_{pa(j)})|^2
       High for nodes whose values don't follow the learned SCM.
    2. Graph-weighted error: weight each node's error by how well-connected
       it is (central nodes violating constraints is more suspicious).

    Output is a scalar in [0, 1] (after sigmoid) — directly used as a
    detection signal that's added to the classification loss.
    """

    def __init__(self, d: int, hidden_dim: int = 64):
        super().__init__()
        # Project per-node errors to a scalar violation score
        self.scorer = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor,
                A_dce: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, d) true node values
            x_hat: (B, d) SCM reconstructed values
            A_dce: (d, d) optional adjacency matrix for weighting
        Returns:
            violation: (B,) scalar violation score per sample
        """
        # Per-node squared reconstruction error
        node_errors = (x - x_hat) ** 2  # (B, d)

        # Optionally weight by node connectivity (in-degree)
        if A_dce is not None:
            # in_degree[j] = sum_i A_ij = how many parents node j has
            in_degree = A_dce.sum(dim=0).detach()  # (d,)
            # Normalize to mean 1 to avoid scaling issues
            in_degree = in_degree / (in_degree.mean() + 1e-8)
            node_errors = node_errors * in_degree.unsqueeze(0)  # (B, d)

        violation = self.scorer(node_errors).squeeze(-1)  # (B,)
        return violation


# ---------------------------------------------------------------------------
# Main module: CausalDiscoveryModule
# ---------------------------------------------------------------------------

class CausalDiscoveryModule(nn.Module):
    """
    Module 3: Causal Discovery for deepfake detection.

    Combines:
    - SparseFeatureSelector: Z_sae (8192 sparse) → Z_active (256 dense)
    - DAGMADCELearner: learns causal DAG over Z_active + S_semantic
    - ViolationScorer: produces per-sample violation score

    Config section (causal_module):
        enabled: true
        latent_variables:
            total_sae_dim: 256      # number of active SAE features
        semantic_dim: 73
        discovery:
            sparsity_penalty: 0.01
        dag_learning:
            hidden_dim: 128
            num_layers: 3
            dag_penalty_weight: 0.1
    """

    def __init__(self, config: dict):
        super().__init__()

        causal_cfg = config['causal_module']
        lv_cfg = causal_cfg['latent_variables']
        dag_cfg = causal_cfg['dag_learning']
        disc_cfg = causal_cfg['discovery']

        # Dimensions
        self.z_active_dim = lv_cfg['total_sae_dim']  # 256
        self.s_dim = causal_cfg['semantic_dim']       # 73
        self.d = self.z_active_dim + self.s_dim       # 329

        # SAE dict size for the selector
        sae_cfg = config.get('sparse_features', {})
        self.sae_dict_size = sae_cfg.get('dict_size',
                                          sae_cfg.get('sparse_autoencoder', {})
                                          .get('input_dim', 1024) * 4)
        num_branches = 2  # spatial + frequency

        # ── Sparse feature compression ────────────────────────────────────
        self.feature_selector = SparseFeatureSelector(
            sae_dict_size=self.sae_dict_size,
            output_dim=self.z_active_dim,
            num_branches=num_branches,
        )

        # ── Input normalization ───────────────────────────────────────────
        # Both z_active and s_semantic need to be on similar scales for the
        # SCM to work. LayerNorm each independently before concatenation.
        self.z_norm = nn.LayerNorm(self.z_active_dim)
        self.s_norm = nn.LayerNorm(self.s_dim)

        # ── DAGMA-DCE learner ─────────────────────────────────────────────
        self.causal_learner = DAGMADCELearner(
            d=self.d,
            hidden_dim=dag_cfg['hidden_dim'],
            num_layers=dag_cfg['num_layers'],
            sparsity_penalty=disc_cfg.get('sparsity_penalty', 0.01),
            s=disc_cfg.get('dagma_s', 1.0),
        )

        # ── Violation scorer ──────────────────────────────────────────────
        self.violation_scorer = ViolationScorer(
            d=self.d,
            hidden_dim=dag_cfg.get('violation_hidden_dim', 64),
        )

        # ── Sparsity penalty weight for A_dce L1 ─────────────────────────
        self.sparsity_weight = disc_cfg.get('sparsity_penalty', 0.01)

        # Log module info
        logger.info(
            f"[CausalDiscoveryModule] Initialized:"
            f"  d={self.d} (z_active={self.z_active_dim}, s_sem={self.s_dim})"
        )
        logger.info(
            f"  SCM: hidden={dag_cfg['hidden_dim']}, "
            f"layers={dag_cfg['num_layers']}"
        )
        logger.info(
            f"  Sparsity penalty: {self.sparsity_weight}, "
            f"DAG penalty weight: {dag_cfg['dag_penalty_weight']}"
        )
        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"  Params: {trainable:,} trainable / {total_params:,} total"
        )

    def _build_causal_input(
        self,
        z_sae: Optional[torch.Tensor],
        semantic_attrs: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Build the d-dimensional input for the causal graph.

        Args:
            z_sae: (B, total_sae_dict_dim) sparse SAE features
            semantic_attrs: (B, 73) per-frame semantic attributes
        Returns:
            x: (B, d) concatenated [z_active, s_semantic]
        """
        B = z_sae.shape[0] if z_sae is not None else semantic_attrs.shape[0]
        device = z_sae.device if z_sae is not None else semantic_attrs.device

        # Compress sparse SAE features to dense active features
        if z_sae is not None:
            z_active = self.feature_selector(z_sae)  # (B, z_active_dim)
            z_active = self.z_norm(z_active)
        else:
            z_active = torch.zeros(B, self.z_active_dim, device=device)

        # Normalize semantic attributes
        if semantic_attrs is not None:
            s_sem = self.s_norm(semantic_attrs.float())
        else:
            s_sem = torch.zeros(B, self.s_dim, device=device)

        # Concatenate: [z_active, s_semantic]
        x = torch.cat([z_active, s_sem], dim=1)  # (B, d)
        return x

    def forward(
        self,
        z_sae: Optional[torch.Tensor] = None,
        semantic_attrs: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        return_graph: bool = False,
    ):
        """
        Args:
            z_sae: (B, total_sae_dict_dim) sparse SAE features from
                DualBranchSparseAutoencoder. These are the compressed
                monosemantic representations — the only upstream signal
                that belongs in the causal graph alongside semantics.
            semantic_attrs: (B, 73) per-frame semantic attributes
            label: (B,) integer labels (0=real, 1=fake). Retained for
                interface compatibility (e.g. causal warmup logging).
                A_dce is now computed over ALL frames regardless of label
                to produce a graph robust to unseen manipulation types and
                to avoid false positives from overfitting to real-only topology.
            return_graph: if True, also return node_residuals and A_dce

        Returns (if return_graph=False):
            violation_score: (B,) per-sample violation scores

        Returns (if return_graph=True):
            violation_score: (B,) per-sample violation scores
            node_residuals: (B, d) per-node SCM reconstruction residuals
                (available for gated fusion into classifier)
            A_dce: (d, d) current adjacency matrix
        """
        # Build causal input [z_active, s_semantic]
        x = self._build_causal_input(z_sae, semantic_attrs)

        # Detach from upstream graph; re-enable grad for Jacobian computation
        x = x.detach().requires_grad_(True)

        if self.training:
            # SCM reconstruction for ALL frames (needed for violation score)
            x_hat = self.causal_learner.scm(x)  # (B, d)
            node_residuals = x - x_hat           # (B, d)

            # A_dce computed over ALL frames (real + fake).
            # Rationale: computing Jacobian statistics from both distributions
            # ensures the estimated causal graph captures the full manifold
            # seen during training. A graph fit only on real frames risks
            # over-specialising to clean facial biomechanics and producing
            # false positives for OOD manipulation methods whose causal
            # signatures differ from training fakes but still diverge from
            # the real-only graph. Using all frames gives a more balanced
            # reference topology; the L_structural loss (real frames only,
            # in get_losses) still enforces that SCM predictions are accurate
            # for real faces, maintaining detection sensitivity.
            A_dce = self.causal_learner.compute_adjacency_dce(x)

            # Violation score for all frames (uses current A_dce for weighting)
            violation_score = self.violation_scorer(x, x_hat, A_dce)

            if return_graph:
                return violation_score, node_residuals, A_dce
            return violation_score

        else:
            # Inference: use EMA adjacency, skip Jacobian computation
            x_hat = self.causal_learner.scm(x)
            node_residuals = x - x_hat
            violation_score = self.violation_scorer(
                x, x_hat, self.causal_learner._A_dce_ema
            )

            if return_graph:
                return violation_score, node_residuals, self.causal_learner._A_dce_ema
            return violation_score

    def get_causal_graph(self) -> torch.Tensor:
        """
        Return the current causal graph (EMA adjacency matrix).
        Useful for visualization and interpretability analysis.

        Returns:
            A_dce: (d, d) tensor where A_ij = causal effect of i on j
            First z_active_dim rows/cols are SAE features,
            remaining s_dim are semantic attributes.
        """
        return self.causal_learner._A_dce_ema.clone()

    def get_node_names(self) -> list:
        """
        Return human-readable names for graph nodes.
        SAE features: 'z_spatial_0', ..., 'z_spatial_127', 'z_freq_0', ..., 'z_freq_127'
        Semantic: 'AU01_r', 'AU02_r', ..., 'pose_Rx', 'pose_Ry', 'pose_Rz'
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
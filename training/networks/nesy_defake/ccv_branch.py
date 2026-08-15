"""
networks/nesy_defake/ccv_branch.py
===================================
Ablation 4b: Causal Constraint Verification (CCV) Evidence Branch.

Neuro-symbolic approach: encode domain knowledge as differentiable
constraints, learn additional constraints from data, detect forensic
anomalies, and verify counterfactual consistency.

Three components produce interpretable violation/anomaly signals that
are fused into 2-d evidence via a small MLP:
 
  1. LearnedConstraintFunctions: K learned soft constraints over combined
     features (extends the 23 hand-coded ConsistencyRulesV7).
  2. ForensicAnomalyDetector: per-group autoencoders on 83-d forensic
     features → 5 grouped reconstruction-error anomaly scores.
  3. CounterfactualPredictor: "given these attributes, what should CLIP
     features look like?" — mismatch = manipulation signal.

Total input to evidence MLP: 23 + K + 5 + 1 = ~45 features.
Total trainable params: ~15-20K (vs ~180K for 4 linear SCMs at 115 nodes).
"""

import math
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Forensic Feature Groups (semantic grouping of 83-d forensic features)
# ═══════════════════════════════════════════════════════════════════════════

FORENSIC_GROUPS = [
    ('boundary_texture', 0, 12),    # ff_grad_* (6) + ff_blur_* (6)
    ('symmetry_color',   12, 30),   # ff_sym_* (4) + ff_color_* (4) + ff_dct_* (6) + ff_quality_* (4) = 18
    ('patch_noise',      30, 46),   # ff_ppnc_* (8) + ff_ccnc_* (8) = 16
    ('srm_noise',        46, 73),   # ff_srm_* (15) + ff_noise_* (12) = 27
    ('fft_spectral',     73, 83),   # ff_fft_* (10)
]

NUM_FORENSIC_GROUPS = len(FORENSIC_GROUPS)


# ═══════════════════════════════════════════════════════════════════════════
#  Component 1: Learned Constraint Functions
# ═══════════════════════════════════════════════════════════════════════════

class LearnedConstraintFunctions(nn.Module):
    """
    Learn K additional soft constraint functions over the combined feature
    space. Each constraint discovers an attribute interaction pattern that
    the 23 hand-coded rules miss.

    Shared trunk + K independent heads. Output is sigmoid-bounded [0, 1]
    where 1 = maximum violation.

    Architecture:
      combined_features (B, 122) → LayerNorm → Linear → GELU → Linear
      → K sigmoid heads → violation_scores (B, K)
    """

    def __init__(self, input_dim: int = 122, num_constraints: int = 16,
                 hidden_dim: int = 48):
        super().__init__()
        self.num_constraints = num_constraints
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )
        self.heads = nn.Linear(hidden_dim, num_constraints)
        # Small init so learned constraints start near 0.5 (uncertain)
        nn.init.normal_(self.heads.weight, std=0.02)
        nn.init.zeros_(self.heads.bias)

        logger.info(f"[LearnedConstraintFunctions] {input_dim} → {hidden_dim} "
                    f"→ {num_constraints} learned constraints")

    def forward(self, combined_features: torch.Tensor) -> torch.Tensor:
        """(B, 122) → (B, K) violation scores in [0, 1]."""
        return torch.sigmoid(self.heads(self.trunk(combined_features)))


# ═══════════════════════════════════════════════════════════════════════════
#  Component 2: Forensic Anomaly Detector
# ═══════════════════════════════════════════════════════════════════════════

class ForensicAnomalyDetector(nn.Module):
    """
    Per-group autoencoders on 83-d forensic features. Reconstruction error
    per group is the anomaly score — real faces should reconstruct well
    (consistent forensic patterns), fakes should not.

    5 groups (semantically meaningful):
      boundary_texture (12), symmetry_color (18), patch_noise (16),
      srm_noise (27), fft_spectral (10)

    Each group: encoder (d→bottleneck) + decoder (bottleneck→d).
    Output: (B, 5) per-group anomaly scores.
    """

    def __init__(self, bottleneck: int = 8):
        super().__init__()
        self.groups = FORENSIC_GROUPS

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for name, start, end in self.groups:
            dim = end - start
            self.encoders.append(nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, bottleneck),
                nn.GELU(),
            ))
            self.decoders.append(nn.Linear(bottleneck, dim))

        logger.info(f"[ForensicAnomalyDetector] {len(self.groups)} groups, "
                    f"bottleneck={bottleneck}")

    def forward(self, forensic_features: torch.Tensor) -> torch.Tensor:
        """(B, 83) → (B, 5) per-group anomaly scores."""
        scores = []
        for i, (name, start, end) in enumerate(self.groups):
            x = forensic_features[:, start:end]
            z = self.encoders[i](x)
            x_hat = self.decoders[i](z)
            # Normalized reconstruction error per group
            score = (x - x_hat).pow(2).mean(dim=1, keepdim=True)
            scores.append(score)
        return torch.cat(scores, dim=1)


# ═══════════════════════════════════════════════════════════════════════════
#  Component 3: Counterfactual Predictor
# ═══════════════════════════════════════════════════════════════════════════

class CounterfactualPredictor(nn.Module):
    """
    Predicts what CLIP spatial features SHOULD look like given the
    semantic + forensic attribute profile. For real faces, the predicted
    features match the actual ones (attributes are consistent with
    appearance). For fakes, there is a mismatch.

    This is a *counterfactual* question: "If a face truly had these
    attributes, what would it look like in CLIP space?"

    Architecture:
      semantic_summary (B, D) → MLP → predicted_z (B, z_dim)
      actual_z = compress(spatial_raw.detach())  (B, z_dim)
      residual = per-dim squared error (B, z_dim)
      mismatch = log1p(mean) → (B, 1)   # bounded, OOD-safe

    Output: (B, 1) counterfactual mismatch score.
    """

    def __init__(self, semantic_dim: int, backbone_dim: int = 1024,
                 z_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.z_dim = z_dim
        self.compressor = nn.Linear(backbone_dim, z_dim, bias=False)
        nn.init.normal_(self.compressor.weight,
                        std=1.0 / math.sqrt(backbone_dim))

        self.predictor = nn.Sequential(
            nn.LayerNorm(semantic_dim),
            nn.Linear(semantic_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, z_dim),
        )

        logger.info(f"[CounterfactualPredictor] semantic({semantic_dim}) "
                    f"→ z({z_dim}), backbone({backbone_dim}) → z({z_dim})")

    def forward(self, spatial_raw: torch.Tensor,
                semantic_summary: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spatial_raw: (B, 1024) CLIP features (will be detached)
            semantic_summary: (B, D) concatenated semantic signals
        Returns:
            (B, 1) counterfactual mismatch score
        """
        actual_z = self.compressor(spatial_raw.detach())
        predicted_z = self.predictor(semantic_summary)
        # Bounded mismatch: mean squared error over z-dims, compressed by log1p.
        # The former `.sum()` over 32 dims was unbounded and produced inf/NaN
        # evidence on extreme OOD samples (DFDC / DeepFakeDetection). log1p of
        # the *mean* keeps this in a small, finite range without changing the
        # monotone real-vs-fake ordering on in-distribution inputs.
        mse = (actual_z - predicted_z).pow(2).mean(dim=1, keepdim=True)
        mismatch = torch.log1p(mse)
        return mismatch


# ═══════════════════════════════════════════════════════════════════════════
#  CCV Evidence Branch (full pipeline)
# ═══════════════════════════════════════════════════════════════════════════

class CausalConstraintVerificationBranch(nn.Module):
    """
    Causal Constraint Verification (CCV) evidence branch.

    Fuses three complementary NeSy signals:
      1. Learned constraint violations (K features)
      2. Forensic anomaly scores (5 features)
      3. Counterfactual predictor mismatch (1 feature)

    Combined with 12 hand-coded consistency rule violations from the
    ConceptBranch, the total signal is 12 + K + 5 + 1 = ~34-d.

    Pipeline:
      [violations(12) || learned(K) || anomaly(5) || cf(1)] → (B, ~34)
      → LayerNorm → Linear → GELU → Dropout → Linear → (B, 2)
      → softplus → evidence (B, 2)
    """

    def __init__(
        self,
        combined_dim: int = 58,
        rules_dim: int = 12,
        forensic_dim: int = 83,
        backbone_dim: int = 1024,
        num_constraints: int = 16,
        constraint_hidden: int = 48,
        forensic_bottleneck: int = 8,
        cf_z_dim: int = 32,
        cf_hidden_dim: int = 64,
        evidence_hidden: int = 64,
        num_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Component 1: learned constraint functions
        self.learned_constraints = LearnedConstraintFunctions(
            input_dim=combined_dim,
            num_constraints=num_constraints,
            hidden_dim=constraint_hidden,
        )

        # Component 2: forensic anomaly detector
        self.forensic_anomaly = ForensicAnomalyDetector(
            bottleneck=forensic_bottleneck,
        )

        # Component 3: counterfactual predictor
        # Input: combined_features + violations + forensic_features
        cf_input_dim = combined_dim + rules_dim + forensic_dim
        self.counterfactual = CounterfactualPredictor(
            semantic_dim=cf_input_dim,
            backbone_dim=backbone_dim,
            z_dim=cf_z_dim,
            hidden_dim=cf_hidden_dim,
        )

        # Evidence fusion MLP
        total_signal_dim = (rules_dim + num_constraints
                           + NUM_FORENSIC_GROUPS + 1)
        self.evidence_mlp = nn.Sequential(
            nn.LayerNorm(total_signal_dim),
            nn.Linear(total_signal_dim, evidence_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(evidence_hidden, num_classes),
        )
        nn.init.normal_(self.evidence_mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.evidence_mlp[-1].bias)

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"[CCV Branch] signals: {rules_dim} rules + {num_constraints} "
            f"learned + {NUM_FORENSIC_GROUPS} anomaly + 1 counterfactual "
            f"= {total_signal_dim}-d → {evidence_hidden} → {num_classes} "
            f"evidence | {n_params:,} params")

    def forward(
        self,
        spatial_raw: torch.Tensor,
        combined_features: torch.Tensor,
        violations: torch.Tensor,
        forensic_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        dataset: Optional[str] = None,
    ) -> dict:
        """
        Args:
            spatial_raw:       (B, 1024) CLIP features (detached internally)
            combined_features: (B, 58) fast feature vector
            violations:        (B, 12) hand-coded consistency rule scores
            forensic_features: (B, 83) precomputed forensic features
            labels:            (B,) optional — unused by CCV but accepted
                               for interface compatibility with improved SCM.
            dataset:           optional dataset name, used only to annotate the
                               non-finite-evidence warning (TASK 1d).
        Returns:
            dict with 'logits', 'evidence', 'dag_penalty',
            'violation_scores', 'anomaly_scores', 'counterfactual_residual'
        """
        device = spatial_raw.device

        # Component 1: learned constraint violations
        learned_violations = self.learned_constraints(combined_features)
        # sigmoid-bounded to [0, 1] already, but guard against NaN propagating
        # from an upstream non-finite combined_features on extreme OOD inputs.
        learned_violations = torch.nan_to_num(
            learned_violations, nan=0.0, posinf=50.0, neginf=0.0
        ).clamp(max=50.0)

        # Component 2: forensic anomaly scores
        anomaly_scores = self.forensic_anomaly(forensic_features)
        # Reconstruction MSE is unbounded above; extreme OOD forensic features
        # can blow it up to inf/NaN. Sanitize and cap before it reaches the MLP.
        anomaly_scores = torch.nan_to_num(
            anomaly_scores, nan=0.0, posinf=50.0, neginf=0.0
        ).clamp(max=50.0)

        # Component 3: counterfactual mismatch
        semantic_summary = torch.cat(
            [combined_features, violations, forensic_features], dim=1)
        cf_residual = self.counterfactual(spatial_raw, semantic_summary)

        # Fuse all signals
        all_signals = torch.cat([
            violations,            # (B, 12) hand-coded
            learned_violations,    # (B, K)  learned
            anomaly_scores,        # (B, 5)  forensic groups
            cf_residual,           # (B, 1)  counterfactual
        ], dim=1)

        # Clamp evidence logits before softplus: softplus(x) grows ~linearly for
        # large x, so a logit of 1e4 yields evidence 1e4 → alpha 1e4 → downstream
        # 1/S underflow and NaN. [-30, 30] keeps softplus in a safe finite range
        # (softplus(30) ≈ 30, softplus(-30) ≈ 1e-13) without touching normal
        # logits, which sit well within this band.
        logits = self.evidence_mlp(all_signals).clamp(min=-30.0, max=30.0)
        evidence = F.softplus(logits)

        # TASK 1d: warn (don't crash) if any non-finite evidence slips through,
        # sanitize it, and log the dataset + count so the offending split is
        # identifiable in the run log.
        if not torch.isfinite(evidence).all():
            n_bad = int((~torch.isfinite(evidence)).sum().item())
            logger.warning(
                "[CCV Branch] %d non-finite evidence entries on dataset=%s; "
                "sanitizing with nan_to_num (this batch's causal evidence is "
                "degraded, not trusted).", n_bad, dataset or 'unknown')
            evidence = torch.nan_to_num(
                evidence, nan=0.0, posinf=30.0, neginf=0.0)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)

        return {
            'logits': logits,
            'evidence': evidence,
            'dag_penalty': torch.zeros(1, device=device).squeeze(),
            'violation_scores': learned_violations.detach(),
            'anomaly_scores': anomaly_scores.detach(),
            'counterfactual_residual': cf_residual.detach(),
        }

"""
networks/nesy_defake/semantic/causal_intervention.py
=====================================================
Tier 3: Do-Calculus Causal Intervention Module (Inference Only).

At inference, uses the learned SCMs and causal graphs to perform
counterfactual interventions: "if we flip attribute X, how do the
causal predictions change?"

Real faces show expected causal cascades (flip 'happy' → AU6/AU12 change).
Fake faces show anomalous responses (manipulation breaks causal structure).

Produces 16 features (4 graphs × 4 statistics) added to classifier_input.
No trainable parameters — reads frozen graph state from CausalDiscoveryModule.
"""

import logging
from typing import List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

CAUSAL_INTERVENTION_NAMES = [
    # Spatial real graph
    'ci_spatial_real_mean_disc',
    'ci_spatial_real_max_disc',
    'ci_spatial_real_disc_entropy',
    'ci_spatial_real_structural',
    # Spatial fake graph
    'ci_spatial_fake_mean_disc',
    'ci_spatial_fake_max_disc',
    'ci_spatial_fake_disc_entropy',
    'ci_spatial_fake_structural',
    # Frequency real graph
    'ci_freq_real_mean_disc',
    'ci_freq_real_max_disc',
    'ci_freq_real_disc_entropy',
    'ci_freq_real_structural',
    # Frequency fake graph
    'ci_freq_fake_mean_disc',
    'ci_freq_fake_max_disc',
    'ci_freq_fake_disc_entropy',
    'ci_freq_fake_structural',
]

NUM_INTERVENTION_FEATURES = len(CAUSAL_INTERVENTION_NAMES)


class CausalInterventionModule(nn.Module):
    """
    Perform do-calculus interventions on learned causal graphs at inference.

    For each of 4 SCMs (spatial_real, spatial_fake, freq_real, freq_fake):
      1. Compute baseline reconstruction: x_hat = SCM(x)
      2. For top-k most confident semantic attributes, flip each one
      3. Measure how much downstream predictions change (discrepancy)
      4. Aggregate into 4 summary statistics

    Real faces: predictable causal responses (high structural consistency)
    Fake faces: anomalous responses (broken causal structure)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.top_k = config.get('top_k_interventions', 10)

    @torch.no_grad()
    def forward(
        self,
        causal_module: nn.Module,
        augmented_semantic: torch.Tensor,
        z_spatial: torch.Tensor = None,
        z_freq: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute intervention features for each sample.

        Args:
            causal_module: CausalDiscoveryModule with trained SCMs
            augmented_semantic: (B, s_dim) augmented semantic vector
            z_spatial: (B, sae_dim) spatial SAE features (detached)
            z_freq: (B, sae_dim) frequency SAE features (detached)
        Returns:
            (B, 16) intervention discrepancy features
        """
        B = augmented_semantic.shape[0]
        device = augmented_semantic.device

        # Build per-branch causal inputs (same as CausalDiscoveryModule._build_branch_input)
        x_spatial = causal_module._build_branch_input(
            z_spatial, augmented_semantic,
            causal_module.spatial_selector, causal_module.z_spatial_norm,
            causal_module.z_spatial_dim)
        x_freq = causal_module._build_branch_input(
            z_freq, augmented_semantic,
            causal_module.freq_selector, causal_module.z_freq_norm,
            causal_module.z_freq_dim)

        # Identify top-k semantic node indices to intervene on
        # Semantic nodes are the last s_dim dimensions of each branch input
        s_dim = causal_module.s_dim
        z_spatial_dim = causal_module.z_spatial_dim

        # Use the semantic portion (after z_active) for confidence ranking
        sem_portion = augmented_semantic  # (B, s_dim)
        # Confidence = how far from 0.5 (most decisive attributes)
        confidence = torch.abs(sem_portion - 0.5)  # (B, s_dim)
        # Top-k per sample
        k = min(self.top_k, s_dim)
        _, top_indices = confidence.topk(k, dim=1)  # (B, k)

        all_feats = []

        # Process each graph
        scm_pairs = [
            ('spatial', 'real', causal_module.causal_spatial.causal_learner_real.scm, x_spatial, z_spatial_dim),
            ('spatial', 'fake', causal_module.causal_spatial.causal_learner_fake.scm, x_spatial, z_spatial_dim),
            ('freq', 'real', causal_module.causal_freq.causal_learner_real.scm, x_freq, causal_module.z_freq_dim),
            ('freq', 'fake', causal_module.causal_freq.causal_learner_fake.scm, x_freq, causal_module.z_freq_dim),
        ]

        for branch, dist, scm, x_input, z_dim in scm_pairs:
            # Baseline reconstruction
            x_hat_baseline = scm(x_input)  # (B, d)

            # Collect per-intervention discrepancies
            discrepancies = []

            for ki in range(k):
                # Semantic node index in the full causal vector
                # Semantic attrs start at index z_dim
                sem_node_idx = z_dim + top_indices[:, ki]  # (B,)

                # Flip the attribute: 1 - current value
                x_perturbed = x_input.clone()
                batch_idx = torch.arange(B, device=device)
                current_val = x_perturbed[batch_idx, sem_node_idx]
                x_perturbed[batch_idx, sem_node_idx] = 1.0 - current_val

                # Intervened reconstruction
                x_hat_perturbed = scm(x_perturbed)  # (B, d)

                # Per-sample discrepancy
                disc = (x_hat_perturbed - x_hat_baseline).pow(2).sum(dim=1).sqrt()  # (B,)
                discrepancies.append(disc)

            # Stack: (B, k)
            disc_matrix = torch.stack(discrepancies, dim=1)

            # Summary statistics per sample
            mean_disc = disc_matrix.mean(dim=1)     # (B,)
            max_disc = disc_matrix.max(dim=1)[0]    # (B,)

            # Entropy of discrepancy distribution (how spread out)
            disc_probs = disc_matrix / (disc_matrix.sum(dim=1, keepdim=True) + 1e-10)
            disc_entropy = -(disc_probs * (disc_probs + 1e-10).log()).sum(dim=1)  # (B,)

            # Structural anomaly: coefficient of variation
            # High CV = inconsistent causal responses = suspicious
            disc_std = disc_matrix.std(dim=1)       # (B,)
            structural = disc_std / (mean_disc + 1e-10)  # (B,)

            all_feats.extend([mean_disc, max_disc, disc_entropy, structural])

        # Stack all 16 features: (B, 16)
        return torch.stack(all_feats, dim=1)

    @staticmethod
    def get_feature_names() -> List[str]:
        """Interpretable names for logging."""
        return list(CAUSAL_INTERVENTION_NAMES)

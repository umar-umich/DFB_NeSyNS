"""
networks/nesy_defake/semantic/causal_intervention.py
=====================================================
Tier 3: Neuro-Symbolic Causal Intervention Module.

Active at both train and inference. Uses learned SCMs and causal graphs
for two complementary reasoning tasks:

A. Cross-graph consistency scoring (8 features, zero extra SCM forwards):
   Compares how well each sample fits the real vs fake SCMs using
   already-computed residuals from the causal module.

B. Do-calculus interventions (20 features):
   Flips top-k semantically important nodes and measures how SCM
   predictions change. Graph-guided node selection prioritizes
   causally important nodes. Differential response across real/fake
   SCMs provides additional discrimination.

Total: 28 interpretable features projected into classifier space.
All computed under torch.no_grad() — no gradient flows to SCMs.
The ci_projection layer trains via classification loss backprop.
"""

import logging
from typing import Dict, List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# -- Cross-graph consistency features (8) ------------------------------------
CROSS_GRAPH_NAMES = [
    'cg_spatial_real_fit',
    'cg_spatial_fake_fit',
    'cg_spatial_consistency',
    'cg_spatial_z_sem_ratio',
    'cg_freq_real_fit',
    'cg_freq_fake_fit',
    'cg_freq_consistency',
    'cg_freq_z_sem_ratio',
]

# -- Do-calculus intervention features (16) ----------------------------------
INTERVENTION_NAMES = [
    'ci_spatial_real_mean_disc',
    'ci_spatial_real_max_disc',
    'ci_spatial_real_disc_entropy',
    'ci_spatial_real_structural',
    'ci_spatial_fake_mean_disc',
    'ci_spatial_fake_max_disc',
    'ci_spatial_fake_disc_entropy',
    'ci_spatial_fake_structural',
    'ci_freq_real_mean_disc',
    'ci_freq_real_max_disc',
    'ci_freq_real_disc_entropy',
    'ci_freq_real_structural',
    'ci_freq_fake_mean_disc',
    'ci_freq_fake_max_disc',
    'ci_freq_fake_disc_entropy',
    'ci_freq_fake_structural',
]

# -- Differential intervention features (4) ---------------------------------
DIFFERENTIAL_NAMES = [
    'ci_spatial_diff_mean',
    'ci_spatial_diff_max',
    'ci_freq_diff_mean',
    'ci_freq_diff_max',
]

# All feature names in order
CAUSAL_INTERVENTION_NAMES = CROSS_GRAPH_NAMES + INTERVENTION_NAMES + DIFFERENTIAL_NAMES
NUM_INTERVENTION_FEATURES = len(CAUSAL_INTERVENTION_NAMES)  # 28


def _disc_stats(disc_matrix: torch.Tensor):
    """Compute 4 summary statistics from a (B, k) discrepancy matrix."""
    mean_disc = disc_matrix.mean(dim=1)                         # (B,)
    max_disc = disc_matrix.max(dim=1)[0]                        # (B,)
    # Entropy of discrepancy distribution
    disc_probs = disc_matrix / (disc_matrix.sum(dim=1, keepdim=True) + 1e-10)
    disc_entropy = -(disc_probs * (disc_probs + 1e-10).log()).sum(dim=1)
    # Coefficient of variation (high = inconsistent causal responses)
    disc_std = disc_matrix.std(dim=1)
    structural = disc_std / (mean_disc + 1e-10)
    return mean_disc, max_disc, disc_entropy, structural


class CausalInterventionModule(nn.Module):
    """
    Neuro-symbolic causal reasoning via cross-graph scoring and do-calculus.

    Produces 28 interpretable features:
      - 8 cross-graph consistency scores (from already-computed residuals)
      - 16 do-calculus intervention statistics (4 graphs x 4 stats)
      - 4 differential intervention responses (real vs fake SCM comparison)

    No trainable parameters — reads frozen graph state from CausalDiscoveryModule.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.top_k = config.get('top_k_interventions', 10)
        self.graph_guided_alpha = config.get('graph_guided_alpha', 0.5)

    # ------------------------------------------------------------------ #
    #  A. Cross-graph consistency scoring (8 features)                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    @torch.no_grad()
    def compute_cross_graph_scores(
        causal_out: Dict[str, torch.Tensor],
        z_causal_dim: int,
        d_identity: int,
    ) -> torch.Tensor:
        """
        Compare how well each sample fits real vs fake SCMs.

        Uses residuals already in causal_out — zero extra SCM forwards.
        v3: residuals are concatenated [identity(d_identity), forensic(d_forensic)].

        Returns:
            (B, 8) cross-graph consistency features.
        """
        feats = []

        for branch in ('spatial', 'freq'):
            res_real = causal_out[f'residuals_{branch}_real']   # (B, d_identity+d_forensic)
            res_fake = causal_out[f'residuals_{branch}_fake']

            # How well sample fits each SCM (lower residual = better fit)
            real_fit = res_real.pow(2).sum(dim=1)               # (B,)
            fake_fit = res_fake.pow(2).sum(dim=1)               # (B,)

            # Consistency: positive = more real-like structure
            consistency = real_fit - fake_fit                    # (B,)

            # Z-feature vs semantic residual ratio (identity sub-graph only)
            # Identity layout: z_causal(0:32) + s_identity(32:103)
            z_res_real = res_real[:, :z_causal_dim].pow(2).sum(dim=1)
            s_res_real = res_real[:, z_causal_dim:d_identity].pow(2).sum(dim=1)
            z_sem_ratio = z_res_real / (s_res_real + 1e-10)     # (B,)

            feats.extend([real_fit, fake_fit, consistency, z_sem_ratio])

        return torch.stack(feats, dim=1)  # (B, 8)

    # ------------------------------------------------------------------ #
    #  B+C. Do-calculus interventions + differential response (20 features)#
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def compute_intervention_features(
        self,
        causal_module: nn.Module,
        augmented_semantic: torch.Tensor,
        z_spatial: torch.Tensor = None,
        z_freq: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Do-calculus interventions with graph-guided node selection.

        For each branch (spatial, freq):
          1. Build causal input, select top-k nodes by blended score
          2. For each intervention, run both real and fake SCMs
          3. Compute per-graph stats (16) + differential response (4)

        Returns:
            (B, 20) intervention + differential features.
        """
        B = augmented_semantic.shape[0]
        device = augmented_semantic.device

        # Build per-branch identity sub-graph inputs (where semantic nodes live)
        # v3: two-stage compression, so we need to replicate the forward path
        z_causal_spatial = causal_module._compress_z(
            z_spatial, causal_module.spatial_selector,
            causal_module.spatial_compressor,
            causal_module.z_feature_spatial_norm,
            B, device)
        z_causal_freq = causal_module._compress_z(
            z_freq, causal_module.freq_selector,
            causal_module.freq_compressor,
            causal_module.z_feature_freq_norm,
            B, device)
        x_spatial = causal_module._build_identity_input(z_causal_spatial, augmented_semantic)
        x_freq = causal_module._build_identity_input(z_causal_freq, augmented_semantic)

        # Identity sub-graph semantic dim = curated(51) + tier1(20) = 71
        s_dim = causal_module._identity_s_dim
        z_dim = causal_module.z_causal_dim

        # Confidence ranking of identity semantic nodes
        # The semantic portion starts after z_causal in the identity input
        sem_portion = x_spatial[:, z_dim:]  # (B, s_dim=71)
        confidence = torch.abs(sem_portion - 0.5)  # (B, s_dim)

        k = min(self.top_k, s_dim)
        alpha = self.graph_guided_alpha

        all_feats = []

        # Process per-branch — use identity sub-graphs (which have semantic nodes)
        branch_configs = [
            ('spatial', x_spatial, z_dim,
             causal_module.identity_spatial.causal_learner_real,
             causal_module.identity_spatial.causal_learner_fake),
            ('freq', x_freq, z_dim,
             causal_module.identity_freq.causal_learner_real,
             causal_module.identity_freq.causal_learner_fake),
        ]

        for branch, x_input, z_dim, learner_real, learner_fake in branch_configs:
            scm_real = learner_real.scm
            scm_fake = learner_fake.scm

            # -- Graph-guided node selection ----------------------------------
            # Blend confidence with structural importance from EMA adjacency
            if alpha < 1.0 and hasattr(learner_real, '_A_dce_ema'):
                # Average importance from both real and fake graphs
                A_real = learner_real._A_dce_ema  # (d, d)
                A_fake = learner_fake._A_dce_ema  # (d, d)
                # Out-degree of semantic nodes (sum of outgoing edge weights)
                sem_importance_real = A_real[z_dim:, :].sum(dim=1)  # (s_dim,)
                sem_importance_fake = A_fake[z_dim:, :].sum(dim=1)  # (s_dim,)
                sem_importance = (sem_importance_real + sem_importance_fake) / 2
                # Normalize to [0, 1]
                imp_max = sem_importance.max()
                if imp_max > 1e-10:
                    sem_importance = sem_importance / imp_max
                # Blend: (B, s_dim)
                combined_score = (alpha * confidence
                                  + (1 - alpha) * sem_importance.unsqueeze(0))
            else:
                combined_score = confidence

            _, top_indices = combined_score.topk(k, dim=1)  # (B, k)
            batch_idx = torch.arange(B, device=device)

            # -- Baseline reconstructions -------------------------------------
            x_hat_base_real = scm_real(x_input)  # (B, d)
            x_hat_base_fake = scm_fake(x_input)  # (B, d)

            disc_real_list = []
            disc_fake_list = []

            for ki in range(k):
                sem_node_idx = z_dim + top_indices[:, ki]  # (B,)

                # Flip the attribute
                x_perturbed = x_input.clone()
                x_perturbed[batch_idx, sem_node_idx] = (
                    1.0 - x_perturbed[batch_idx, sem_node_idx])

                # Run both SCMs on same perturbation
                x_hat_pert_real = scm_real(x_perturbed)
                x_hat_pert_fake = scm_fake(x_perturbed)

                disc_r = (x_hat_pert_real - x_hat_base_real).pow(2).sum(dim=1).sqrt()
                disc_f = (x_hat_pert_fake - x_hat_base_fake).pow(2).sum(dim=1).sqrt()
                disc_real_list.append(disc_r)
                disc_fake_list.append(disc_f)

            disc_real_mat = torch.stack(disc_real_list, dim=1)  # (B, k)
            disc_fake_mat = torch.stack(disc_fake_list, dim=1)  # (B, k)

            # Per-graph statistics (4 stats × 2 distributions = 8 per branch)
            all_feats.extend(_disc_stats(disc_real_mat))   # 4: real graph
            all_feats.extend(_disc_stats(disc_fake_mat))   # 4: fake graph

            # Differential response: how differently real vs fake SCMs react
            diff_mat = (disc_real_mat - disc_fake_mat).abs()   # (B, k)
            diff_mean = diff_mat.mean(dim=1)                   # (B,)
            diff_max = diff_mat.max(dim=1)[0]                  # (B,)
            all_feats.extend([diff_mean, diff_max])

        # Stack: 2 branches × (8 per-graph + 2 differential) = 20
        return torch.stack(all_feats, dim=1)

    # ------------------------------------------------------------------ #
    #  Legacy forward (calls both methods)                                 #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def forward(
        self,
        causal_module: nn.Module,
        causal_out: Dict[str, torch.Tensor],
        augmented_semantic: torch.Tensor,
        z_spatial: torch.Tensor = None,
        z_freq: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute all 28 intervention features.

        Returns:
            (B, 28) concatenated cross-graph + intervention + differential.
        """
        cross_graph = self.compute_cross_graph_scores(
            causal_out, causal_module.z_causal_dim, causal_module.d_identity)
        intervention = self.compute_intervention_features(
            causal_module, augmented_semantic, z_spatial, z_freq)
        return torch.cat([cross_graph, intervention], dim=1)

    @staticmethod
    def get_feature_names() -> List[str]:
        """Interpretable names for logging."""
        return list(CAUSAL_INTERVENTION_NAMES)

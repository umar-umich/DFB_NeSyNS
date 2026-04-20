"""
Factory for Ablation-4 causal branches.

Selects one of three implementations based on config['causal_branch']['type']:
  - 'ccv'          : CausalConstraintVerificationBranch (default, novel)
  - 'improved_scm' : ImprovedCausalBranch (nonlinear SCM with 4 sub-graphs)
  - 'simple'       : SimplifiedCausalBranch (legacy linear SCM)
"""

import torch.nn as nn


def build_causal_branch(sc_cfg: dict) -> nn.Module:
    ctype = sc_cfg.get('type', 'simple')

    if ctype == 'ccv':
        from networks.nesy_defake.ccv_branch import (
            CausalConstraintVerificationBranch)
        return CausalConstraintVerificationBranch(
            combined_dim=sc_cfg.get('combined_dim', 122),
            rules_dim=sc_cfg.get('rules_dim', 23),
            forensic_dim=sc_cfg.get('forensic_dim', 83),
            backbone_dim=sc_cfg.get('backbone_dim', 1024),
            num_constraints=sc_cfg.get('num_constraints', 16),
            constraint_hidden=sc_cfg.get('constraint_hidden', 48),
            forensic_bottleneck=sc_cfg.get('forensic_bottleneck', 8),
            cf_z_dim=sc_cfg.get('cf_z_dim', 32),
            cf_hidden_dim=sc_cfg.get('cf_hidden_dim', 64),
            evidence_hidden=sc_cfg.get('evidence_hidden', 64),
            dropout=sc_cfg.get('dropout', 0.2),
        )

    if ctype == 'improved_scm':
        from networks.nesy_defake.improved_scm_branch import (
            ImprovedCausalBranch)
        return ImprovedCausalBranch(
            backbone_dim=sc_cfg.get('backbone_dim', 1024),
            z_causal_dim=sc_cfg.get('z_causal_dim', 32),
            curated_dim=sc_cfg.get('curated_dim', 51),
            rules_dim=sc_cfg.get('rules_dim', 23),
            forensic_dim=sc_cfg.get('forensic_dim', 83),
            scm_hidden_dim=sc_cfg.get('scm_hidden_dim', 64),
            summary_dim=sc_cfg.get('summary_dim', 8),
            evidence_hidden=sc_cfg.get('evidence_hidden', 64),
            sparsity_penalty=sc_cfg.get('sparsity_penalty', 0.01),
            divergence_weight=sc_cfg.get('divergence_weight', 0.1),
            recon_weight=sc_cfg.get('recon_weight', 0.5),
        )

    from networks.nesy_defake.concept_branch import SimplifiedCausalBranch
    return SimplifiedCausalBranch(
        backbone_dim=sc_cfg.get('backbone_dim', 1024),
        z_causal_dim=sc_cfg.get('z_causal_dim', 32),
        curated_dim=sc_cfg.get('curated_dim', 51),
        rules_dim=sc_cfg.get('rules_dim', 23),
        forensic_dim=sc_cfg.get('forensic_dim', 83),
        hidden_dim=sc_cfg.get('hidden_dim', 64),
        sparsity_penalty=sc_cfg.get('sparsity_penalty', 0.01),
    )

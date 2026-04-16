"""
interpretability/analyzers/branch_evidence.py
=============================================
Level 2: Branch-Level Evidence Decomposition — shows which reasoning pathway
(spatial / concept / causal) drove each prediction.
"""

import os

import numpy as np

from .base import BaseAnalyzer
from .. import visualization as viz


class BranchEvidenceAnalyzer(BaseAnalyzer):
    name = 'branch_evidence'
    required_keys = ('spatial_evidence', 'concept_evidence', 'causal_evidence',
                     'total_evidence')

    def analyze(self) -> dict:
        spatial = self._stack('spatial_evidence')
        concept = self._stack('concept_evidence')
        causal = self._stack('causal_evidence')
        total = self._stack('total_evidence')
        labels = self._all_labels
        if spatial is None or total is None:
            return {}

        s_sum = spatial.sum(axis=1)
        t_sum = total.sum(axis=1)
        branches = {'spatial': s_sum}
        if concept is not None:
            branches['concept'] = concept.sum(axis=1)
        if causal is not None:
            branches['causal'] = causal.sum(axis=1)

        total_safe = np.maximum(t_sum, 1e-8)
        ratios = {k: float((v / total_safe).mean()) for k, v in branches.items()}

        real_mask = labels == 0
        fake_mask = labels == 1
        per_class = {}
        for cls_name, mask in [('Real', real_mask), ('Fake', fake_mask)]:
            if mask.any():
                per_class[cls_name] = {k: float(v[mask].mean()) for k, v in branches.items()}

        self._branches = branches
        self._ratios = ratios
        self._per_class = per_class
        return {'contribution_ratios': ratios, 'per_class_evidence': per_class}

    def visualize(self, save_dir: str) -> None:
        if not hasattr(self, '_ratios'):
            return
        # Single plot: per-class evidence decomposition (includes overall ratios)
        if self._per_class:
            viz.plot_evidence_by_class(
                self._per_class,
                title='Branch Evidence Decomposition (Real vs Fake)',
                save_path=os.path.join(save_dir, 'branch_evidence.png'),
            )
        else:
            viz.plot_stacked_bar(
                self._ratios,
                title='Branch Evidence Contribution Ratios',
                save_path=os.path.join(save_dir, 'branch_evidence.png'),
                ylabel='Fraction of Total Evidence',
            )

    def explain_sample(self, idx: int) -> str:
        if not hasattr(self, '_branches'):
            return ''
        total = sum(v[idx] for v in self._branches.values())
        total = max(total, 1e-8)
        parts = [f"{name}={vals[idx]:.2f} ({vals[idx] / total * 100:.0f}%)"
                 for name, vals in self._branches.items()]
        return f"[Evidence] {', '.join(parts)}"

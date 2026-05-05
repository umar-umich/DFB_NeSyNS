"""
interpretability/analyzers/branch_evidence.py
=============================================
Level 2: Branch-Level Evidence Decomposition — shows which reasoning pathway
(spatial / concept / causal) drove each prediction.

2026-05-04 — Goal 1: data exports added.
  * branch_evidence.csv             (per-class, per-branch mean evidence + share)
  * branch_evidence.json            (n_samples / dataset_name / contribution ratios)
  * branch_evidence_per_sample.npz  (full per-sample populations)
"""

import os

import numpy as np

from .base import BaseAnalyzer
from .. import visualization as viz
from ..data_export import (
    analyzer_metadata, dump_csv, dump_json, dump_npz, sibling_path,
    with_basename,
)


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
        # Stash full per-sample populations so visualize() can dump NPZ.
        self._per_sample = {'spatial': s_sum}
        if concept is not None:
            self._per_sample['concept'] = concept.sum(axis=1)
        if causal is not None:
            self._per_sample['causal'] = causal.sum(axis=1)
        self._labels_flat = labels.astype(int).reshape(-1)
        return {'contribution_ratios': ratios, 'per_class_evidence': per_class}

    def visualize(self, save_dir: str) -> None:
        if not hasattr(self, '_ratios'):
            return
        png_path = os.path.join(save_dir, 'branch_evidence.png')
        # Single plot: per-class evidence decomposition (includes overall ratios)
        if self._per_class:
            viz.plot_evidence_by_class(
                self._per_class,
                title='Branch Evidence Decomposition (Real vs Fake)',
                save_path=png_path,
            )
        else:
            viz.plot_stacked_bar(
                self._ratios,
                title='Branch Evidence Contribution Ratios',
                save_path=png_path,
                ylabel='Fraction of Total Evidence',
            )

        # ── Data exports ────────────────────────────────────────────────
        # CSV — per-class, per-branch mean evidence + within-class share.
        rows = []
        for cls_name, branch_means in (self._per_class or {}).items():
            total_cls = sum(branch_means.values()) or 1.0
            for br, val in branch_means.items():
                rows.append({
                    'class': cls_name,
                    'branch': br,
                    'mean_evidence': float(val),
                    'share_pct': float(100.0 * val / total_cls),
                })
        if rows:
            dump_csv(
                rows, ['class', 'branch', 'mean_evidence', 'share_pct'],
                sibling_path(png_path, '.csv'),
            )

        # NPZ — full per-sample evidence per branch + labels.
        per_sample = getattr(self, '_per_sample', {})
        if per_sample:
            arrays = {k: v for k, v in per_sample.items()}
            arrays['label'] = self._labels_flat
            dump_npz(arrays, with_basename(png_path,
                                           'branch_evidence_per_sample.npz'))

        dump_json(
            analyzer_metadata(
                n_samples=int(len(self._labels_flat))
                    if hasattr(self, '_labels_flat') else 0,
                dataset_name=self.dataset_name,
                contribution_ratios=self._ratios,
                per_class_evidence=self._per_class,
                branches_present=list(self._per_sample.keys())
                    if per_sample else [],
            ),
            sibling_path(png_path, '.json'),
        )

    def explain_sample(self, idx: int) -> str:
        if not hasattr(self, '_branches'):
            return ''
        total = sum(v[idx] for v in self._branches.values())
        total = max(total, 1e-8)
        parts = [f"{name}={vals[idx]:.2f} ({vals[idx] / total * 100:.0f}%)"
                 for name, vals in self._branches.items()]
        return f"[Evidence] {', '.join(parts)}"

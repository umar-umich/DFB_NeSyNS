"""
interpretability/analyzers/edl_uncertainty.py
=============================================
Level 1: EDL Uncertainty Quantification — per-prediction Dirichlet analysis,
ECE calibration, and uncertainty histograms (real vs fake).
"""

import os

import numpy as np

from .base import BaseAnalyzer
from .. import visualization as viz


class EDLUncertaintyAnalyzer(BaseAnalyzer):
    name = 'edl_uncertainty'
    required_keys = ('alpha', 'uncertainty', 'prob')

    def __init__(self, n_bins: int = 10):
        super().__init__()
        self.n_bins = n_bins

    # ── ECE computation ───────────────────────────────────────────────────

    @staticmethod
    def _compute_ece(probs, labels, n_bins):
        confidences = np.maximum(probs, 1 - probs)
        predictions = (probs >= 0.5).astype(int)
        correct = (predictions == labels).astype(float)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_accs = np.zeros(n_bins)
        bin_confs = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)

        for i in range(n_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            mask = (confidences > lo) & (confidences <= hi)
            if mask.sum() > 0:
                bin_accs[i] = correct[mask].mean()
                bin_confs[i] = confidences[mask].mean()
                bin_counts[i] = mask.sum()

        total = bin_counts.sum()
        ece = (bin_counts / max(total, 1) * np.abs(bin_accs - bin_confs)).sum()
        return ece, bin_accs, bin_confs, bin_counts

    # ── interface ──────────────────────────────────────────────────────────

    def analyze(self) -> dict:
        uncertainty = self._stack('uncertainty')
        prob = self._stack('prob')
        alpha = self._stack('alpha')
        labels = self._all_labels
        if uncertainty is None or prob is None:
            return {}

        ece, bin_accs, bin_confs, bin_counts = self._compute_ece(
            prob, labels, self.n_bins)

        real_mask = labels == 0
        fake_mask = labels == 1

        results = {
            'ece': float(ece),
            'mean_uncertainty': float(uncertainty.mean()),
            'mean_uncertainty_real': float(uncertainty[real_mask].mean()) if real_mask.any() else 0.0,
            'mean_uncertainty_fake': float(uncertainty[fake_mask].mean()) if fake_mask.any() else 0.0,
            'bin_accuracies': bin_accs.tolist(),
            'bin_confidences': bin_confs.tolist(),
            'bin_counts': bin_counts.tolist(),
        }

        self._results = results
        self._uncertainty = uncertainty
        self._prob = prob
        self._alpha = alpha
        self._cached_labels = labels
        return results

    def visualize(self, save_dir: str) -> None:
        if not hasattr(self, '_results'):
            return
        r = self._results
        labels = self._cached_labels
        real_mask = labels == 0
        fake_mask = labels == 1

        viz.plot_histogram(
            {'Real': self._uncertainty[real_mask],
             'Fake': self._uncertainty[fake_mask]},
            title='EDL Uncertainty Distribution',
            xlabel='Epistemic Uncertainty (K/S)',
            save_path=os.path.join(save_dir, 'uncertainty_histogram.png'),
        )
        viz.plot_reliability_diagram(
            confidences=np.array(r['bin_confidences']),
            accuracies=np.array(r['bin_accuracies']),
            bin_counts=np.array(r['bin_counts']),
            ece=r['ece'],
            save_path=os.path.join(save_dir, 'reliability_diagram.png'),
            n_bins=self.n_bins,
        )

    def explain_sample(self, idx: int) -> str:
        if not hasattr(self, '_uncertainty'):
            return ''
        u = self._uncertainty[idx]
        p = self._prob[idx]
        a = self._alpha[idx]
        label = 'FAKE' if p >= 0.5 else 'REAL'
        level = 'high' if u > 0.5 else ('moderate' if u > 0.25 else 'low')
        return (f"[EDL] Predicted {label} (prob={p:.3f}), "
                f"uncertainty={u:.3f} ({level}), "
                f"alpha=[{a[0]:.2f}, {a[1]:.2f}]")

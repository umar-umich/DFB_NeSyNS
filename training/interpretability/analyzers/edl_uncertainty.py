"""
interpretability/analyzers/edl_uncertainty.py
=============================================
Level 1: EDL Uncertainty Quantification — per-prediction Dirichlet analysis,
ECE calibration, and uncertainty histograms (real vs fake).

2026-05-04 — Goals 1/2/3:
  * Histogram is rendered over the full population (no sub-sample).
  * Reliability diagram switched to a two-panel composite with
    equal-mass binning by default; equal-width ECE (10 bins) and ACE
    are also computed and reported in the title and JSON.
  * x-axis auto-cropped to the empirical confidence range when
    ``reliability_xlim_auto`` is True (default).
  * Optional temperature scaling (deterministic 90/10 split, seed 42)
    contributes ``ece_post_temperature_scaling`` to the JSON only;
    predictions and the saved figure are unchanged.
  * Per-bin and per-sample tables exported alongside every PNG.
"""

import os
from typing import Dict, Optional, Tuple

import numpy as np

from .base import BaseAnalyzer
from .. import visualization as viz
from ..data_export import (
    analyzer_metadata, dump_csv, dump_csv_columns, dump_json, dump_npz,
    sibling_path, with_basename,
)


class EDLUncertaintyAnalyzer(BaseAnalyzer):
    name = 'edl_uncertainty'
    required_keys = ('alpha', 'uncertainty', 'prob')

    def __init__(
        self,
        n_bins: int = 10,
        reliability_binning: str = 'equal_mass',
        reliability_n_bins_main: int = 15,
        reliability_xlim_auto: bool = True,
        temperature_scaling: bool = False,
    ):
        super().__init__()
        # Equal-width ECE is conventionally reported with 10 bins; keep
        # that as the canonical bin count.
        self.n_bins = int(n_bins)
        self.reliability_binning = str(reliability_binning)  # 'equal_width' | 'equal_mass'
        self.reliability_n_bins_main = int(reliability_n_bins_main)
        self.reliability_xlim_auto = bool(reliability_xlim_auto)
        self.temperature_scaling = bool(temperature_scaling)

    # ── ECE / ACE / binning primitives ────────────────────────────────────

    @staticmethod
    def _equal_width_edges(n_bins: int) -> np.ndarray:
        return np.linspace(0.0, 1.0, n_bins + 1)

    @staticmethod
    def _equal_mass_edges(confidences: np.ndarray, n_bins: int) -> np.ndarray:
        """Quantile-based bin edges so every bin gets ~equal sample mass."""
        if confidences.size == 0:
            return np.linspace(0.0, 1.0, n_bins + 1)
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(confidences, qs)
        # Guard against degenerate ties — make edges strictly monotonic.
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-9
        edges[0] = min(edges[0], 0.0)
        edges[-1] = max(edges[-1], 1.0)
        return edges

    @staticmethod
    def _bin_calibration(
        confidences: np.ndarray,
        correct: np.ndarray,
        edges: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (bin_acc, bin_conf, bin_count) over the supplied edges."""
        n_bins = len(edges) - 1
        accs = np.zeros(n_bins, dtype=float)
        confs = np.zeros(n_bins, dtype=float)
        counts = np.zeros(n_bins, dtype=int)
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            if i == 0:
                mask = (confidences >= lo) & (confidences <= hi)
            else:
                mask = (confidences > lo) & (confidences <= hi)
            n = int(mask.sum())
            if n > 0:
                accs[i] = float(correct[mask].mean())
                confs[i] = float(confidences[mask].mean())
                counts[i] = n
        return accs, confs, counts

    @classmethod
    def _ece_from_bins(
        cls,
        accs: np.ndarray,
        confs: np.ndarray,
        counts: np.ndarray,
    ) -> float:
        total = max(int(counts.sum()), 1)
        return float((counts / total * np.abs(accs - confs)).sum())

    def _binning_pack(
        self,
        confidences: np.ndarray,
        correct: np.ndarray,
        scheme: str,
        n_bins: int,
    ) -> Dict[str, np.ndarray]:
        if scheme == 'equal_width':
            edges = self._equal_width_edges(n_bins)
        elif scheme == 'equal_mass':
            edges = self._equal_mass_edges(confidences, n_bins)
        else:
            raise ValueError(f'unknown binning scheme: {scheme}')
        accs, confs, counts = self._bin_calibration(confidences, correct, edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ece = self._ece_from_bins(accs, confs, counts)
        return {
            'scheme': scheme,
            'edges': edges,
            'centers': centers,
            'bin_acc': accs,
            'bin_conf': confs,
            'bin_count': counts,
            'ece': ece,
        }

    @staticmethod
    def _ace(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
        """Adaptive Calibration Error: equal-mass binning, *unweighted* mean
        of per-bin |acc - conf|. Penalises mis-calibration uniformly across
        the score distribution rather than weighting by bin mass.
        """
        if confidences.size == 0:
            return 0.0
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(confidences, qs)
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-9
        accs, confs, counts = EDLUncertaintyAnalyzer._bin_calibration(
            confidences, correct, edges)
        active = counts > 0
        if not active.any():
            return 0.0
        return float(np.abs(accs[active] - confs[active]).mean())

    @staticmethod
    def _temperature_scaled_ece(
        prob: np.ndarray,
        labels: np.ndarray,
        seed: int = 42,
    ) -> Optional[float]:
        """Fit a single temperature ``T`` on a held-out 10% slice, then
        report the equal-width 10-bin ECE on the remaining 90%. Returns
        None if scipy is unavailable or the slice is too small.

        NB: this is a REPORT-ONLY number — the saved predictions and the
        saved reliability figure are unchanged.
        """
        try:
            from scipy.optimize import minimize_scalar
        except ImportError:
            return None
        n = len(prob)
        if n < 50:
            return None
        rng = np.random.default_rng(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_val = max(1, int(round(0.10 * n)))
        val_idx, eval_idx = idx[:n_val], idx[n_val:]

        prob_val = np.clip(prob[val_idx], 1e-6, 1 - 1e-6)
        # Recover binary logits from prob (P(fake) = sigmoid(z) → z = logit p).
        z_val = np.log(prob_val) - np.log(1 - prob_val)
        y_val = labels[val_idx].astype(int)

        def _nll(T: float) -> float:
            T = max(T, 1e-3)
            p = 1.0 / (1.0 + np.exp(-z_val / T))
            p = np.clip(p, 1e-9, 1 - 1e-9)
            return float(-np.mean(y_val * np.log(p) + (1 - y_val) * np.log(1 - p)))

        res = minimize_scalar(_nll, bounds=(0.05, 10.0), method='bounded')
        T = float(res.x)
        prob_eval = np.clip(prob[eval_idx], 1e-6, 1 - 1e-6)
        z_eval = np.log(prob_eval) - np.log(1 - prob_eval)
        prob_eval_T = 1.0 / (1.0 + np.exp(-z_eval / T))
        conf_eval = np.maximum(prob_eval_T, 1 - prob_eval_T)
        pred_eval = (prob_eval_T >= 0.5).astype(int)
        correct_eval = (pred_eval == labels[eval_idx]).astype(float)
        edges = np.linspace(0.0, 1.0, 11)
        accs, confs, counts = EDLUncertaintyAnalyzer._bin_calibration(
            conf_eval, correct_eval, edges)
        return EDLUncertaintyAnalyzer._ece_from_bins(accs, confs, counts)

    # ── interface ──────────────────────────────────────────────────────────

    def analyze(self) -> dict:
        uncertainty = self._stack('uncertainty')
        prob = self._stack('prob')
        alpha = self._stack('alpha')
        labels = self._all_labels
        if uncertainty is None or prob is None:
            return {}
        # Full population — no sub-sampling for distributions.
        prob = prob.reshape(-1)
        labels = labels.astype(int).reshape(-1)
        confidences = np.maximum(prob, 1.0 - prob)
        predictions = (prob >= 0.5).astype(int)
        correct = (predictions == labels).astype(float)

        # Equal-width ECE (10 bins, conventional)
        pack_eqw = self._binning_pack(confidences, correct, 'equal_width', self.n_bins)
        # Equal-mass ECE (default 15 bins)
        pack_eqm = self._binning_pack(
            confidences, correct, 'equal_mass', self.reliability_n_bins_main)
        # ACE (15 equal-mass bins, unweighted)
        ace = self._ace(confidences, correct, n_bins=15)

        results = {
            'ece': pack_eqw['ece'],            # canonical (back-compat)
            'ece_equal_width_10': pack_eqw['ece'],
            'ece_equal_mass_15': pack_eqm['ece'],
            'ace_equal_mass_15': ace,
            'mean_uncertainty': float(uncertainty.mean()),
            'mean_uncertainty_real': float(uncertainty[labels == 0].mean())
                if (labels == 0).any() else 0.0,
            'mean_uncertainty_fake': float(uncertainty[labels == 1].mean())
                if (labels == 1).any() else 0.0,
            'bin_accuracies': pack_eqw['bin_acc'].tolist(),
            'bin_confidences': pack_eqw['bin_conf'].tolist(),
            'bin_counts': pack_eqw['bin_count'].tolist(),
        }

        # Optional temperature-scaled ECE (report-only)
        if self.temperature_scaling:
            ts_ece = self._temperature_scaled_ece(prob, labels)
            results['ece_post_temperature_scaling'] = (
                None if ts_ece is None else float(ts_ece))

        self._results = results
        self._uncertainty = uncertainty
        self._prob = prob
        self._alpha = alpha
        self._cached_labels = labels
        self._confidences = confidences
        self._correct = correct
        self._pack_eqw = pack_eqw
        self._pack_eqm = pack_eqm
        self._ace_value = ace
        return results

    def visualize(self, save_dir: str) -> None:
        if not hasattr(self, '_results'):
            return
        labels = self._cached_labels
        real_mask = labels == 0
        fake_mask = labels == 1

        # ── Uncertainty histogram (full population, no sub-sample) ─────
        u_real = self._uncertainty[real_mask]
        u_fake = self._uncertainty[fake_mask]
        hist_path = os.path.join(save_dir, 'uncertainty_histogram.png')
        viz.plot_histogram(
            {'Real': u_real, 'Fake': u_fake},
            title='EDL Uncertainty Distribution',
            xlabel='Epistemic Uncertainty (K/S)',
            save_path=hist_path,
            balance=True,
        )
        # NPZ — full per-sample populations.
        dump_npz(
            {'u_real': u_real, 'u_fake': u_fake},
            sibling_path(hist_path, '.npz'),
        )
        dump_json(
            analyzer_metadata(
                n_samples=int(len(self._uncertainty)),
                dataset_name=self.dataset_name,
                n_real=int(u_real.size),
                n_fake=int(u_fake.size),
                mean_u_real=float(u_real.mean()) if u_real.size else 0.0,
                mean_u_fake=float(u_fake.mean()) if u_fake.size else 0.0,
                median_u_real=float(np.median(u_real)) if u_real.size else 0.0,
                median_u_fake=float(np.median(u_fake)) if u_fake.size else 0.0,
            ),
            sibling_path(hist_path, '.json'),
        )

        # ── Reliability diagram — two-panel composite ──────────────────
        rel_path = os.path.join(save_dir, 'reliability_diagram.png')
        if self.reliability_binning == 'equal_mass':
            main_pack = self._pack_eqm
            main_label = f'equal-mass ({self.reliability_n_bins_main} bins)'
        else:
            main_pack = self._pack_eqw
            main_label = f'equal-width ({self.n_bins} bins)'

        # Auto-crop x-axis to the empirical confidence range
        if self.reliability_xlim_auto and self._confidences.size:
            xlim = (max(0.0, float(self._confidences.min()) - 0.05), 1.0)
        else:
            xlim = (0.0, 1.0)

        viz.plot_reliability_diagram_v2(
            confidences_per_sample=self._confidences,
            accuracies_per_sample=self._correct,
            bin_edges_main=main_pack['edges'],
            bin_centers_main=main_pack['centers'],
            bin_confidence_main=main_pack['bin_conf'],
            bin_accuracy_main=main_pack['bin_acc'],
            bin_count_main=main_pack['bin_count'],
            ece_equal_width=self._pack_eqw['ece'],
            ece_equal_mass=self._pack_eqm['ece'],
            ace=self._ace_value,
            main_scheme_label=main_label,
            save_path=rel_path,
            xlim=xlim,
        )

        # CSV — main scheme's per-bin table.
        bin_rows = []
        edges = main_pack['edges']
        for i in range(len(main_pack['centers'])):
            bin_rows.append({
                'bin_lower': float(edges[i]),
                'bin_upper': float(edges[i + 1]),
                'bin_center': float(main_pack['centers'][i]),
                'bin_count': int(main_pack['bin_count'][i]),
                'bin_confidence': float(main_pack['bin_conf'][i]),
                'bin_accuracy': float(main_pack['bin_acc'][i]),
                'gap': float(abs(main_pack['bin_acc'][i]
                                 - main_pack['bin_conf'][i])),
            })
        dump_csv(
            bin_rows,
            ['bin_lower', 'bin_upper', 'bin_center', 'bin_count',
             'bin_confidence', 'bin_accuracy', 'gap'],
            sibling_path(rel_path, '.csv'),
        )

        # CSV — extended (both schemes side-by-side for offline re-binning).
        ext_rows = []
        for scheme_pack in (self._pack_eqw, self._pack_eqm):
            scheme = scheme_pack['scheme']
            edges_s = scheme_pack['edges']
            for i in range(len(scheme_pack['centers'])):
                ext_rows.append({
                    'scheme': scheme,
                    'bin_idx': i,
                    'bin_lower': float(edges_s[i]),
                    'bin_upper': float(edges_s[i + 1]),
                    'bin_center': float(scheme_pack['centers'][i]),
                    'bin_count': int(scheme_pack['bin_count'][i]),
                    'bin_confidence': float(scheme_pack['bin_conf'][i]),
                    'bin_accuracy': float(scheme_pack['bin_acc'][i]),
                    'gap': float(abs(scheme_pack['bin_acc'][i]
                                     - scheme_pack['bin_conf'][i])),
                })
        dump_csv(
            ext_rows,
            ['scheme', 'bin_idx', 'bin_lower', 'bin_upper', 'bin_center',
             'bin_count', 'bin_confidence', 'bin_accuracy', 'gap'],
            with_basename(rel_path, 'reliability_diagram_extended.csv'),
        )

        # Per-sample CSV — full population, every prediction, for offline
        # re-binning under any scheme.
        n = int(len(self._prob))
        dump_csv_columns(
            {
                'sample_idx': np.arange(n, dtype=np.int64),
                'confidence': self._confidences,
                'predicted_class': (self._prob >= 0.5).astype(int),
                'true_class': labels.astype(int),
                'correct': self._correct.astype(int),
            },
            with_basename(rel_path, 'reliability_per_sample.csv'),
        )

        # JSON — scalar metadata for the reliability figure.
        dump_json(
            analyzer_metadata(
                n_samples=n,
                dataset_name=self.dataset_name,
                ece=float(self._results['ece']),
                ece_equal_width_10=float(self._pack_eqw['ece']),
                ece_equal_mass_15=float(self._pack_eqm['ece']),
                ace_equal_mass_15=float(self._ace_value),
                ece_post_temperature_scaling=self._results.get(
                    'ece_post_temperature_scaling'),
                n_bins_equal_width=self.n_bins,
                n_bins_equal_mass=self.reliability_n_bins_main,
                binning_scheme=self.reliability_binning,
                xlim_auto=self.reliability_xlim_auto,
                confidence_min=float(self._confidences.min())
                    if self._confidences.size else 0.0,
                confidence_max=float(self._confidences.max())
                    if self._confidences.size else 0.0,
            ),
            sibling_path(rel_path, '.json'),
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

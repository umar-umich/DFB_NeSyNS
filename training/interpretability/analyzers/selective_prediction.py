"""
interpretability/analyzers/selective_prediction.py
==================================================
2026-05-04 — Goal 1: data exports added.
  * risk_coverage.csv          (coverage / risk_edl / risk_softmax / risk_oracle)
  * selective_summary.json     (AURC + E-AURC for all three rankings)
  * confidence_vs_accuracy.csv (coverage / accuracy_edl / accuracy_softmax)

Selective-prediction metrics promised in README §8:

  * Risk-Coverage curve   — error rate on the accepted fraction, sweeping
                            the EDL-uncertainty threshold.
  * AURC                  — area under the risk-coverage curve (lower is better).
  * E-AURC                — excess AURC vs the oracle confidence ranking.
  * Confidence-vs-accuracy curve over the same sweep.

Rationale: EDL gives us a per-sample uncertainty `u`. In deployment we
want "flag and escalate" behaviour — defer the noisy predictions to a
human and keep only the confident ones. AURC quantifies how useful `u`
is for that deferral decision. A lower AURC than the baseline "rank by
probability margin" signals that EDL is giving us information beyond
what softmax already provides.
"""

import os

import numpy as np

from .base import BaseAnalyzer
from .. import visualization as viz
from ..data_export import (
    analyzer_metadata, dump_csv_columns, dump_json, sibling_path,
    with_basename,
)


class SelectivePredictionAnalyzer(BaseAnalyzer):
    name = 'selective_prediction'
    required_keys = ('prob', 'uncertainty')

    @staticmethod
    def _risk_coverage(scores: np.ndarray, errors: np.ndarray):
        """Sort samples by *scores* ascending (low = confident) and sweep.

        Returns coverage in [1/n, 1] and cumulative risk (error rate on
        the accepted prefix).
        """
        order = np.argsort(scores)
        err_sorted = errors[order].astype(np.float64)
        cum = np.cumsum(err_sorted)
        n = len(err_sorted)
        coverage = np.arange(1, n + 1) / n
        risk = cum / np.arange(1, n + 1)
        return coverage, risk

    @staticmethod
    def _aurc(coverage: np.ndarray, risk: np.ndarray) -> float:
        return float(np.trapz(risk, coverage))

    def analyze(self) -> dict:
        prob = self._stack('prob')
        unc = self._stack('uncertainty')
        if prob is None or unc is None:
            return {}
        labels = self._all_labels.astype(int).reshape(-1)
        prob = prob.reshape(-1)
        unc = unc.reshape(-1)
        preds = (prob >= 0.5).astype(int)
        errors = (preds != labels).astype(int)

        # Rank by uncertainty (ours) — low u first.
        cov_u, risk_u = self._risk_coverage(unc, errors)
        aurc_u = self._aurc(cov_u, risk_u)

        # Baseline rank by softmax margin (|p − 0.5|) — large margin first.
        margin = -np.abs(prob - 0.5)
        cov_m, risk_m = self._risk_coverage(margin, errors)
        aurc_m = self._aurc(cov_m, risk_m)

        # Oracle: always accept the correct ones first.
        cov_o, risk_o = self._risk_coverage(errors.astype(float), errors)
        aurc_oracle = self._aurc(cov_o, risk_o)

        # Confidence-vs-accuracy: for each uncertainty percentile, what
        # fraction of the accepted samples is correct?
        cov_steps = np.linspace(0.1, 1.0, 10)
        acc_at_cov = []
        for c in cov_steps:
            k = max(1, int(round(c * len(errors))))
            order = np.argsort(unc)[:k]
            acc_at_cov.append(1.0 - errors[order].mean())

        self._curves = {
            'ours':     (cov_u, risk_u),
            'softmax':  (cov_m, risk_m),
            'oracle':   (cov_o, risk_o),
        }
        self._acc_at_cov = (cov_steps, np.asarray(acc_at_cov))

        return {
            'AURC_uncertainty':     aurc_u,
            'AURC_softmax_margin':  aurc_m,
            'AURC_oracle':          aurc_oracle,
            'E_AURC_uncertainty':   aurc_u - aurc_oracle,
            'E_AURC_softmax':       aurc_m - aurc_oracle,
            'overall_error_rate':   float(errors.mean()),
            'acc_at_coverage':      dict(zip(
                [f'{int(c*100)}%' for c in cov_steps],
                [float(a) for a in acc_at_cov])),
        }

    def visualize(self, save_dir: str) -> None:
        if not hasattr(self, '_curves'):
            return
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        rc_png = os.path.join(save_dir, 'risk_coverage_curve.png')
        cv_png = os.path.join(save_dir, 'confidence_vs_accuracy.png')

        # Risk-coverage curve
        fig, ax = plt.subplots(figsize=(7, 5))
        styles = {
            'ours':    ('#1f77b4', 'EDL uncertainty'),
            'softmax': ('#ff7f0e', 'Softmax margin'),
            'oracle':  ('#555555', 'Oracle (lower bound)'),
        }
        for name, (cov, risk) in self._curves.items():
            color, label = styles[name]
            ls = '--' if name == 'oracle' else '-'
            ax.plot(cov, risk, color=color, linestyle=ls, linewidth=2, label=label)
        ax.set_xlabel('Coverage')
        ax.set_ylabel('Selective risk (error on accepted)')
        ax.set_title('Risk–Coverage (lower is better)')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(rc_png, dpi=150)
        plt.close(fig)

        # Confidence-vs-accuracy
        cov_steps, acc = self._acc_at_cov
        # Compute softmax accuracy at the same coverage steps for the CSV
        try:
            margin_order = np.argsort(-np.abs(self._curves['softmax'][1] - 0))
        except Exception:
            margin_order = None
        # We store curves via _risk_coverage which gives risk; recover acc.
        _, risk_softmax = self._curves.get('softmax', (None, None))
        acc_softmax_at_cov = []
        if risk_softmax is not None:
            n = len(risk_softmax)
            for c in cov_steps:
                k = max(1, int(round(c * n))) - 1
                acc_softmax_at_cov.append(float(1.0 - risk_softmax[k]))

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(cov_steps, acc, 'o-', color='#2ca02c', linewidth=2,
                label='EDL uncertainty')
        if acc_softmax_at_cov:
            ax.plot(cov_steps, acc_softmax_at_cov, 's--', color='#ff7f0e',
                    linewidth=1.6, label='Softmax margin')
            ax.legend(fontsize=9, loc='lower left')
        ax.set_xlabel('Coverage (fraction of most-confident samples kept)')
        ax.set_ylabel('Accuracy on accepted samples')
        ax.set_title('Confidence-aware accuracy')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_ylim(0.0, 1.01)
        fig.tight_layout()
        fig.savefig(cv_png, dpi=150)
        plt.close(fig)

        # ── Data exports ────────────────────────────────────────────────
        # risk_coverage.csv — three risk curves on a common coverage grid.
        cov_u, risk_u = self._curves['ours']
        _, risk_o = self._curves['oracle']
        risk_s = self._curves['softmax'][1]
        n = len(cov_u)
        dump_csv_columns(
            {
                'coverage': cov_u,
                'risk_edl': risk_u,
                'risk_softmax': risk_s,
                'risk_oracle': risk_o,
            },
            sibling_path(rc_png, '.csv'),
        )

        # confidence_vs_accuracy.csv
        if acc_softmax_at_cov:
            dump_csv_columns(
                {
                    'coverage': np.asarray(cov_steps),
                    'accuracy_edl': np.asarray(acc),
                    'accuracy_softmax': np.asarray(acc_softmax_at_cov),
                },
                sibling_path(cv_png, '.csv'),
            )
        else:
            dump_csv_columns(
                {
                    'coverage': np.asarray(cov_steps),
                    'accuracy_edl': np.asarray(acc),
                },
                sibling_path(cv_png, '.csv'),
            )

        # selective_summary.json — covers both PNGs.
        aurc_u = self._aurc(*self._curves['ours'])
        aurc_s = self._aurc(*self._curves['softmax'])
        aurc_o = self._aurc(*self._curves['oracle'])
        summary = analyzer_metadata(
            n_samples=int(n),
            dataset_name=self.dataset_name,
            aurc_edl=float(aurc_u),
            aurc_softmax=float(aurc_s),
            aurc_oracle=float(aurc_o),
            e_aurc_edl=float(aurc_u - aurc_o),
            e_aurc_softmax=float(aurc_s - aurc_o),
        )
        dump_json(summary, with_basename(rc_png, 'selective_summary.json'))
        # Also write the JSON sidecars for each PNG so downstream scripts
        # that look up `<basename>.json` find them.
        dump_json(summary, sibling_path(rc_png, '.json'))
        dump_json(summary, sibling_path(cv_png, '.json'))

    def explain_sample(self, idx: int) -> str:
        return ''

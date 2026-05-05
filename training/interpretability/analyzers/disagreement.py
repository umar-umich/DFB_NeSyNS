"""
interpretability/analyzers/disagreement.py
==========================================
Level 6/7: Inter-Branch Disagreement — measures reasoning conflict between
neural (spatial) and symbolic (concept + causal) branches, and correlates
disagreement with EDL uncertainty.

2026-05-04 — Goals 1 + 2:
  * Histograms / scatters use the FULL test population (no sub-sample);
    if a u-vs-d scatter would be too dense, switch to ``plt.hexbin``
    (gridsize=60, bins='log') instead of subsampling.
  * disagreement_per_sample.npz, disagreement_summary.json,
    uncertainty_by_agreement.csv exported as siblings of the PNGs.
"""

import os
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .base import BaseAnalyzer
from .. import visualization as viz
from ..data_export import (
    analyzer_metadata, dump_csv_columns, dump_json, dump_npz, sibling_path,
    with_basename,
)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2 or a.size != b.size:
        return float('nan')
    ar = np.argsort(np.argsort(a))
    br = np.argsort(np.argsort(b))
    return float(np.corrcoef(ar, br)[0, 1])


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2 or a.size != b.size:
        return float('nan')
    if a.std() == 0 or b.std() == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def _cosine_disagreement(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Per-sample 1 - cosine_similarity over (B, K) probability rows."""
    pn = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-12)
    qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    return 1.0 - np.clip((pn * qn).sum(axis=1), -1.0, 1.0)


class DisagreementAnalyzer(BaseAnalyzer):
    name = 'disagreement'
    required_keys = ('spatial_evidence', 'concept_evidence', 'causal_evidence',
                     'uncertainty')

    def analyze(self) -> dict:
        spatial = self._stack('spatial_evidence')
        concept = self._stack('concept_evidence')
        causal = self._stack('causal_evidence')
        uncertainty = self._stack('uncertainty')
        labels = self._all_labels
        if spatial is None:
            return {}

        spatial_pred = spatial.argmax(axis=1)
        branches = {'spatial': spatial_pred}
        if concept is not None:
            branches['concept'] = concept.argmax(axis=1)
        if causal is not None:
            branches['causal'] = causal.argmax(axis=1)

        # Pairwise disagreement (hard prediction)
        branch_names = list(branches.keys())
        disagreement_rates = {}
        for i in range(len(branch_names)):
            for j in range(i + 1, len(branch_names)):
                n1, n2 = branch_names[i], branch_names[j]
                rate = float((branches[n1] != branches[n2]).mean())
                disagreement_rates[f'{n1}_vs_{n2}'] = rate

        results = {'disagreement_rates': disagreement_rates}

        # Continuous disagreement: 1 - cos(branch_evidence rows). This is
        # the IBDC-style soft signal used for the u-vs-d correlation
        # downstream — kept as full per-sample arrays.
        soft = {}
        if concept is not None:
            soft['spatial_concept'] = _cosine_disagreement(spatial, concept)
        if causal is not None:
            soft['spatial_causal'] = _cosine_disagreement(spatial, causal)
        if concept is not None and causal is not None:
            soft['concept_causal'] = _cosine_disagreement(concept, causal)
        if soft:
            d_avg = np.mean(np.stack(list(soft.values()), axis=0), axis=0)
        else:
            d_avg = np.zeros(spatial.shape[0], dtype=float)

        agree_flag = None
        if len(branch_names) >= 2 and uncertainty is not None:
            all_preds = np.stack(list(branches.values()), axis=0)
            any_disagree = (all_preds.max(axis=0) != all_preds.min(axis=0))
            agree_mask = ~any_disagree
            disagree_mask = any_disagree
            agree_flag = (~any_disagree).astype(int)  # 1 = all branches agree
            results['overall_disagreement_rate'] = float(any_disagree.mean())
            results['mean_uncertainty_agree'] = float(uncertainty[agree_mask].mean()) if agree_mask.any() else 0.0
            results['mean_uncertainty_disagree'] = float(uncertainty[disagree_mask].mean()) if disagree_mask.any() else 0.0

        self._branches = branches
        self._uncertainty = uncertainty
        self._cached_labels = labels
        self._results = results
        self._d_avg = d_avg                 # (N,)
        self._d_pairs = soft                # dict of (N,)
        self._agree_flag = agree_flag       # (N,) or None
        # `correct` based on the spatial branch matching the ground-truth
        # label — kept for the per-sample NPZ.
        self._correct = (spatial_pred == labels.astype(int)).astype(int)
        return results

    def visualize(self, save_dir: str) -> None:
        if not hasattr(self, '_results'):
            return

        rates = self._results.get('disagreement_rates', {})
        rates_png = os.path.join(save_dir, 'disagreement_rates.png')
        if rates:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(range(len(rates)), list(rates.values()),
                   tick_label=list(rates.keys()))
            ax.set_ylabel('Disagreement Rate')
            ax.set_title('Inter-Branch Disagreement Rates')
            ax.set_ylim(0, 1)
            fig.tight_layout()
            fig.savefig(rates_png, dpi=150)
            plt.close(fig)
            # JSON sidecar — pair-wise rates + counts.
            dump_json(
                analyzer_metadata(
                    n_samples=int(len(self._cached_labels)),
                    dataset_name=self.dataset_name,
                    disagreement_rates=rates,
                    overall_disagreement_rate=self._results.get(
                        'overall_disagreement_rate'),
                ),
                sibling_path(rates_png, '.json'),
            )

        # ── uncertainty_by_agreement.png — full sample, no subset ───────
        png_path = os.path.join(save_dir, 'uncertainty_by_agreement.png')
        if (self._uncertainty is not None and len(self._branches) >= 2):
            all_preds = np.stack(list(self._branches.values()), axis=0)
            any_disagree = (all_preds.max(axis=0) != all_preds.min(axis=0))
            agree_unc = self._uncertainty[~any_disagree]
            disagree_unc = self._uncertainty[any_disagree]
            if agree_unc.size > 0 and disagree_unc.size > 0:
                # full population, no subsample
                viz.plot_histogram(
                    {'Agree': agree_unc, 'Disagree': disagree_unc},
                    title='Uncertainty by Branch Agreement',
                    xlabel='Uncertainty',
                    save_path=png_path,
                    balance=True,
                )

        # ── u-vs-d scatter (hexbin if too dense) ────────────────────────
        if self._uncertainty is not None and self._d_avg.size:
            uvd_png = os.path.join(save_dir, 'uncertainty_vs_disagreement.png')
            n = int(self._uncertainty.size)
            fig, ax = plt.subplots(figsize=(7, 5))
            if n > 5000:
                # hexbin instead of scatter — full population, no subsample
                hb = ax.hexbin(self._d_avg, self._uncertainty,
                               gridsize=60, bins='log',
                               cmap='viridis', mincnt=1)
                fig.colorbar(hb, ax=ax, label='log(count)')
            else:
                ax.scatter(self._d_avg, self._uncertainty, s=4, alpha=0.4,
                           color='#1f77b4')
            ax.set_xlabel('Inter-branch disagreement (mean 1 - cos)')
            ax.set_ylabel('Uncertainty u = K/S')
            ax.set_title('Uncertainty vs. inter-branch disagreement')
            ax.grid(True, linestyle='--', alpha=0.3)
            fig.tight_layout()
            fig.savefig(uvd_png, dpi=150)
            plt.close(fig)

        # ── Per-sample NPZ ──────────────────────────────────────────────
        npz_arrays = {
            'd_pairwise_avg': self._d_avg.astype(float),
            'u': (self._uncertainty.astype(float)
                  if self._uncertainty is not None
                  else np.zeros_like(self._d_avg)),
            'labels': self._cached_labels.astype(int).reshape(-1),
            'correct': self._correct.astype(int),
        }
        for k in ('spatial_concept', 'spatial_causal', 'concept_causal'):
            if k in self._d_pairs:
                npz_arrays[f'd_{k}'] = self._d_pairs[k].astype(float)
        # Also export the agreement flag (1 = all branches agree).
        if self._agree_flag is not None:
            npz_arrays['agree_flag'] = self._agree_flag.astype(int)
        dump_npz(
            npz_arrays,
            with_basename(png_path, 'disagreement_per_sample.npz'),
        )

        # ── uncertainty_by_agreement.csv ────────────────────────────────
        if self._agree_flag is not None and self._uncertainty is not None:
            n = int(self._uncertainty.size)
            dump_csv_columns(
                {
                    'sample_idx': np.arange(n, dtype=np.int64),
                    'u': self._uncertainty.astype(float),
                    'agree_flag': self._agree_flag.astype(int),
                    'branch_pair_threshold': np.full(n, 'argmax_match',
                                                     dtype=object),
                },
                with_basename(png_path, 'uncertainty_by_agreement.csv'),
            )

        # ── disagreement_summary.json ──────────────────────────────────
        u_arr = self._uncertainty if self._uncertainty is not None \
            else np.zeros_like(self._d_avg)
        spear = _spearman(u_arr, self._d_avg)
        pear = _pearson(u_arr, self._d_avg)
        dump_json(
            analyzer_metadata(
                n_samples=int(self._d_avg.size),
                dataset_name=self.dataset_name,
                spearman_u_d=spear,
                pearson_u_d=pear,
                mean_u_agree=self._results.get('mean_uncertainty_agree'),
                mean_u_disagree=self._results.get('mean_uncertainty_disagree'),
                agreement_threshold_used='argmax_match',
                overall_disagreement_rate=self._results.get(
                    'overall_disagreement_rate'),
                disagreement_rates=self._results.get('disagreement_rates'),
            ),
            sibling_path(png_path, '.json'),
        )

    def explain_sample(self, idx: int) -> str:
        if not hasattr(self, '_branches'):
            return ''
        preds = {k: int(v[idx]) for k, v in self._branches.items()}
        class_labels = ['REAL', 'FAKE']
        parts = [f"{k}={class_labels[v]}" for k, v in preds.items()]
        status = 'AGREE' if len(set(preds.values())) == 1 else 'DISAGREE'
        unc_str = ''
        if self._uncertainty is not None:
            unc_str = f', uncertainty={self._uncertainty[idx]:.3f}'
        return f"[Disagreement] {', '.join(parts)} -- {status}{unc_str}"

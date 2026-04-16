"""
interpretability/analyzers/disagreement.py
==========================================
Level 6/7: Inter-Branch Disagreement — measures reasoning conflict between
neural (spatial) and symbolic (concept + causal) branches, and correlates
disagreement with EDL uncertainty.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .base import BaseAnalyzer
from .. import visualization as viz


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

        # Pairwise disagreement
        branch_names = list(branches.keys())
        disagreement_rates = {}
        for i in range(len(branch_names)):
            for j in range(i + 1, len(branch_names)):
                n1, n2 = branch_names[i], branch_names[j]
                rate = float((branches[n1] != branches[n2]).mean())
                disagreement_rates[f'{n1}_vs_{n2}'] = rate

        results = {'disagreement_rates': disagreement_rates}

        if len(branch_names) >= 2 and uncertainty is not None:
            all_preds = np.stack(list(branches.values()), axis=0)
            any_disagree = (all_preds.max(axis=0) != all_preds.min(axis=0)).astype(float)
            agree_mask = any_disagree == 0
            disagree_mask = any_disagree == 1
            results['overall_disagreement_rate'] = float(any_disagree.mean())
            results['mean_uncertainty_agree'] = float(uncertainty[agree_mask].mean()) if agree_mask.any() else 0.0
            results['mean_uncertainty_disagree'] = float(uncertainty[disagree_mask].mean()) if disagree_mask.any() else 0.0

        self._branches = branches
        self._uncertainty = uncertainty
        self._cached_labels = labels
        self._results = results
        return results

    def visualize(self, save_dir: str) -> None:
        if not hasattr(self, '_results'):
            return

        rates = self._results.get('disagreement_rates', {})
        if rates:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(range(len(rates)), list(rates.values()), tick_label=list(rates.keys()))
            ax.set_ylabel('Disagreement Rate')
            ax.set_title('Inter-Branch Disagreement Rates')
            ax.set_ylim(0, 1)
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, 'disagreement_rates.png'), dpi=150)
            plt.close(fig)

        if self._uncertainty is not None and len(self._branches) >= 2:
            all_preds = np.stack(list(self._branches.values()), axis=0)
            any_disagree = (all_preds.max(axis=0) != all_preds.min(axis=0))
            agree_unc = self._uncertainty[~any_disagree]
            disagree_unc = self._uncertainty[any_disagree]
            if agree_unc.size > 0 and disagree_unc.size > 0:
                viz.plot_histogram(
                    {'Agree': agree_unc, 'Disagree': disagree_unc},
                    title='Uncertainty by Branch Agreement',
                    xlabel='Uncertainty',
                    save_path=os.path.join(save_dir, 'uncertainty_by_agreement.png'),
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

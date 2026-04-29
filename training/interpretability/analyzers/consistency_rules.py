"""
interpretability/analyzers/consistency_rules.py
===============================================
Level 3: Named Consistency Rule Violations — per-sample top violated rules
with human-readable FACS names, aggregate firing rates by class, and
optional misclassification-slice report.
"""

import csv
import os

import numpy as np

from .base import BaseAnalyzer
from .. import visualization as viz


def _load_rule_names(n_rules: int):
    """Resolve rule names to match the detector's active rule set.

    v7 (23 rules) is the current production set; older v5 had 20.
    """
    if n_rules == 23:
        try:
            from networks.nesy_defake.semantic.consistency_rules_v7 import (
                TRAINING_RULE_NAMES_V7,
            )
            return list(TRAINING_RULE_NAMES_V7)
        except ImportError:
            pass
    if n_rules == 20:
        try:
            from networks.nesy_defake.semantic.consistency_rules import (
                TRAINING_RULE_NAMES,
            )
            return list(TRAINING_RULE_NAMES)
        except ImportError:
            pass
    return [f'rule_{i}' for i in range(n_rules)]


class ConsistencyRuleAnalyzer(BaseAnalyzer):
    name = 'consistency_rules'
    required_keys = ('violations',)

    def __init__(self):
        super().__init__()
        self.rule_names = _load_rule_names(23)

    def analyze(self) -> dict:
        violations = self._stack('violations')
        labels = self._all_labels
        if violations is None:
            return {}

        real_mask = labels == 0
        fake_mask = labels == 1
        n_rules = violations.shape[1]
        if len(self.rule_names) != n_rules:
            self.rule_names = _load_rule_names(n_rules)
        names = self.rule_names[:n_rules]

        real_means = violations[real_mask].mean(axis=0) if real_mask.any() else np.zeros(n_rules)
        fake_means = violations[fake_mask].mean(axis=0) if fake_mask.any() else np.zeros(n_rules)

        gaps = fake_means - real_means
        sorted_indices = np.argsort(-np.abs(gaps))
        top_rules = [
            {'rule': names[i], 'gap': float(gaps[i]),
             'real_mean': float(real_means[i]), 'fake_mean': float(fake_means[i])}
            for i in sorted_indices[:10]
        ]

        self._violations = violations
        self._names = names
        self._real_means = real_means
        self._fake_means = fake_means
        return {'top_discriminative_rules': top_rules}

    def visualize(self, save_dir: str) -> None:
        if not hasattr(self, '_names'):
            return
        # Side-by-side firing rates (kept for completeness; uses pretty labels)
        from ..pretty_names import pretty_many
        viz.plot_grouped_bar(
            {'Real': self._real_means.tolist(),
             'Fake': self._fake_means.tolist()},
            pretty_many(self._names),
            title='Consistency rule firing rates by class',
            save_path=os.path.join(save_dir, 'rule_firing_rates.png'),
            ylabel='Mean violation score',
        )
        # Discriminative-gap view — usually the figure to put in the paper
        viz.plot_discriminative_gap(
            real_values=self._real_means.tolist(),
            fake_values=self._fake_means.tolist(),
            category_names=list(self._names),
            title=(r'Rules ranked by class-discriminative gap '
                   r'(top-12 by $|\mathrm{fake}-\mathrm{real}|$)'),
            save_path=os.path.join(save_dir, 'rule_discriminative_gap.png'),
            top_k=12,
        )

    def explain_sample(self, idx: int) -> str:
        if not hasattr(self, '_violations'):
            return ''
        v = self._violations[idx]
        top_k = np.argsort(-v)[:5]
        parts = [f"{self._names[i]}={v[i]:.3f}" for i in top_k if v[i] > 0.01]
        if not parts:
            return "[Rules] No significant violations"
        return f"[Rules] Top violations: {', '.join(parts)}"

    def slice_report(self, preds: np.ndarray, save_dir: str,
                     top_k_plot: int = 15) -> dict:
        """Per-rule firing rates split by confusion-matrix slice.

        Slices: TN (real, pred real), FP (real, pred fake),
                TP (fake, pred fake), FN (fake, pred real).

        Writes:
          - rule_slice_means.csv : per-rule mean + std + count per slice,
                                   plus error gaps (FP-TN, FN-TP).
          - rule_error_gap.png   : top-|gap| rules side-by-side.
          - rule_slice_summary.txt : short narrative of the worst offenders.
        """
        violations = self._stack('violations')
        if violations is None:
            return {}
        labels = self._all_labels
        preds = np.asarray(preds).astype(int).reshape(-1)
        if preds.shape[0] != violations.shape[0]:
            raise ValueError(
                f"slice_report: preds ({preds.shape[0]}) and violations "
                f"({violations.shape[0]}) must have the same length")

        n_rules = violations.shape[1]
        if len(self.rule_names) != n_rules:
            self.rule_names = _load_rule_names(n_rules)
        names = self.rule_names[:n_rules]

        masks = {
            'TN': (labels == 0) & (preds == 0),
            'FP': (labels == 0) & (preds == 1),
            'TP': (labels == 1) & (preds == 1),
            'FN': (labels == 1) & (preds == 0),
        }
        counts = {k: int(m.sum()) for k, m in masks.items()}

        def _mean(m):
            return violations[m].mean(axis=0) if m.any() else np.zeros(n_rules)

        def _std(m):
            return violations[m].std(axis=0) if m.any() else np.zeros(n_rules)

        slice_means = {k: _mean(m) for k, m in masks.items()}
        slice_stds = {k: _std(m) for k, m in masks.items()}
        fp_gap = slice_means['FP'] - slice_means['TN']  # real misclassified
        fn_gap = slice_means['FN'] - slice_means['TP']  # fake missed

        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, 'rule_slice_means.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                'rule',
                'TN_mean', 'TN_std',
                'FP_mean', 'FP_std',
                'TP_mean', 'TP_std',
                'FN_mean', 'FN_std',
                'FP_minus_TN', 'FN_minus_TP',
            ])
            for i, name in enumerate(names):
                w.writerow([
                    name,
                    f'{slice_means["TN"][i]:.5f}', f'{slice_stds["TN"][i]:.5f}',
                    f'{slice_means["FP"][i]:.5f}', f'{slice_stds["FP"][i]:.5f}',
                    f'{slice_means["TP"][i]:.5f}', f'{slice_stds["TP"][i]:.5f}',
                    f'{slice_means["FN"][i]:.5f}', f'{slice_stds["FN"][i]:.5f}',
                    f'{fp_gap[i]:+.5f}', f'{fn_gap[i]:+.5f}',
                ])

        if counts['FP'] > 0 or counts['FN'] > 0:
            combined_gap = np.abs(fp_gap) + np.abs(fn_gap)
            top_idx = np.argsort(-combined_gap)[:top_k_plot]
            viz.plot_grouped_bar(
                {
                    'TN (real ok)':    slice_means['TN'][top_idx].tolist(),
                    'FP (real wrong)': slice_means['FP'][top_idx].tolist(),
                    'TP (fake ok)':    slice_means['TP'][top_idx].tolist(),
                    'FN (fake wrong)': slice_means['FN'][top_idx].tolist(),
                },
                [names[i] for i in top_idx],
                title=(f'Rule firing by confusion slice — top {top_k_plot} by '
                       f'|FP-TN|+|FN-TP|'),
                save_path=os.path.join(save_dir, 'rule_error_gap.png'),
                ylabel='Mean violation score',
            )

        # Narrative: the rules where wrong-slice firing diverges most from
        # the correct slice are the ones misleading the classifier.
        def _top(gap, k=5):
            order = np.argsort(-np.abs(gap))
            return [(names[i], float(gap[i])) for i in order[:k]
                    if abs(gap[i]) > 1e-6]

        top_fp = _top(fp_gap)
        top_fn = _top(fn_gap)
        lines = [
            'Consistency-rule error-slice report',
            (f'Counts — TN={counts["TN"]} FP={counts["FP"]} '
             f'TP={counts["TP"]} FN={counts["FN"]}'),
            '',
            'False-positive gap (real wrongly called fake): FP mean - TN mean',
            '  Positive → rule fires on misclassified reals; may trigger false fakes.',
        ]
        for name, gap in top_fp:
            lines.append(f'    {name:<28s}  {gap:+.4f}')
        lines.append('')
        lines.append('False-negative gap (fake wrongly called real): FN mean - TP mean')
        lines.append('  Negative → rule is quieter on missed fakes; may be the signal we lose.')
        for name, gap in top_fn:
            lines.append(f'    {name:<28s}  {gap:+.4f}')

        txt_path = os.path.join(save_dir, 'rule_slice_summary.txt')
        with open(txt_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')

        return {
            'counts': counts,
            'slice_means': {k: v.tolist() for k, v in slice_means.items()},
            'fp_gap_top': top_fp,
            'fn_gap_top': top_fn,
            'csv': csv_path,
        }

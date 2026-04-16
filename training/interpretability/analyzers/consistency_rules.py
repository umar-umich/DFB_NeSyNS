"""
interpretability/analyzers/consistency_rules.py
===============================================
Level 3: Named Consistency Rule Violations — per-sample top violated rules
with human-readable FACS names, aggregate firing rates by class.
"""

import os

import numpy as np

from .base import BaseAnalyzer
from .. import visualization as viz


class ConsistencyRuleAnalyzer(BaseAnalyzer):
    name = 'consistency_rules'
    required_keys = ('violations',)

    def __init__(self):
        super().__init__()
        try:
            from networks.nesy_defake.semantic.consistency_rules import (
                TRAINING_RULE_NAMES,
            )
            self.rule_names = list(TRAINING_RULE_NAMES)
        except ImportError:
            self.rule_names = [f'rule_{i}' for i in range(20)]

    def analyze(self) -> dict:
        violations = self._stack('violations')
        labels = self._all_labels
        if violations is None:
            return {}

        real_mask = labels == 0
        fake_mask = labels == 1
        n_rules = violations.shape[1]
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
        viz.plot_grouped_bar(
            {'Real': self._real_means.tolist(), 'Fake': self._fake_means.tolist()},
            self._names,
            title='Consistency Rule Firing Rates by Class',
            save_path=os.path.join(save_dir, 'rule_firing_rates.png'),
            ylabel='Mean Violation Score',
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

"""
interpretability/analyzers/gate_analysis.py
===========================================
Level 5: Evidence Gate Analysis — per-image trust allocation between
concept and causal branches (static or feature-conditioned gates).
"""

import os

import numpy as np

from .base import BaseAnalyzer
from .. import visualization as viz


class GateAnalyzer(BaseAnalyzer):
    name = 'gate_analysis'
    required_keys = ('concept_gate', 'causal_gate', 'uncertainty')

    def __init__(self):
        super().__init__()
        self._concept_gates: list = []
        self._causal_gates: list = []

    def reset(self):
        super().reset()
        self._concept_gates = []
        self._causal_gates = []

    def collect(self, preds: dict, labels) -> None:
        self._labels.append(self._to_numpy(labels))
        for key, buf in [('concept_gate', self._concept_gates),
                         ('causal_gate', self._causal_gates)]:
            val = preds.get(key)
            if val is not None:
                buf.append(self._to_numpy(val))
        unc = preds.get('uncertainty')
        if unc is not None:
            self._buffers.setdefault('uncertainty', []).append(self._to_numpy(unc))

    def analyze(self) -> dict:
        results = {}
        for name, buf in [('concept_gate', self._concept_gates),
                          ('causal_gate', self._causal_gates)]:
            if buf:
                vals = np.concatenate([np.atleast_1d(v) for v in buf])
                results[name] = {
                    'mean': float(vals.mean()),
                    'std': float(vals.std()),
                    'min': float(vals.min()),
                    'max': float(vals.max()),
                }
        self._results = results
        return results

    def visualize(self, save_dir: str) -> None:
        # Single combined histogram for all gates
        data = {}
        for name, buf in [('concept_gate', self._concept_gates),
                          ('causal_gate', self._causal_gates)]:
            if buf:
                data[name] = np.concatenate([np.atleast_1d(v) for v in buf])
        if data:
            viz.plot_histogram(
                data,
                title='Evidence Gate Distributions',
                xlabel='Gate Value',
                save_path=os.path.join(save_dir, 'gate_distributions.png'),
            )

    def explain_sample(self, idx: int) -> str:
        parts = []
        cg_val, ca_val = None, None
        for name, buf, attr in [('concept_gate', self._concept_gates, 'cg'),
                                ('causal_gate', self._causal_gates, 'ca')]:
            if buf:
                vals = np.concatenate([np.atleast_1d(v) for v in buf])
                val = vals[min(idx, len(vals) - 1)]
                is_static = ' (static)' if len(vals) == 1 else ''
                parts.append(f"{name}={val:.3f}{is_static}")
                if attr == 'cg':
                    cg_val = val
                else:
                    ca_val = val
        if not parts:
            return ''
        dominant = ''
        if cg_val is not None and ca_val is not None:
            dominant = f" -- {'concept' if cg_val > ca_val else 'causal'} branch dominant"
        return f"[Gates] {', '.join(parts)}{dominant}"

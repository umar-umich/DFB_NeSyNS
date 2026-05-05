"""
interpretability/analyzers/gate_analysis.py
===========================================
Level 5: Evidence Gate Analysis — per-image trust allocation between
concept and causal branches (static or feature-conditioned gates).

2026-05-04 — Goal 1 + 2:
  * Histogram uses the FULL gate-value population (no sub-sample).
  * gate_values.csv  — per-sample concept_gate / causal_gate.
  * gate_summary.json — mean/std/min/max + ``is_constant`` flag
    (std < 1e-6) per gate. Case-study panel uses ``is_constant`` to
    decide whether to display the gate footer.
"""

import os

import numpy as np

from .base import BaseAnalyzer
from .. import visualization as viz
from ..data_export import (
    analyzer_metadata, dump_csv_columns, dump_json, sibling_path,
    with_basename,
)


_CONST_TOL = 1e-6


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
        # Full-population stack — no sub-sample.
        data = {}
        for name, buf in [('concept_gate', self._concept_gates),
                          ('causal_gate', self._causal_gates)]:
            if buf:
                data[name] = np.concatenate([np.atleast_1d(v) for v in buf])
        png_path = os.path.join(save_dir, 'gate_distributions.png')
        if data:
            viz.plot_histogram(
                data,
                title='Evidence Gate Distributions',
                xlabel='Gate Value',
                save_path=png_path,
            )

        # ── Data exports ────────────────────────────────────────────────
        cg = data.get('concept_gate')
        ca = data.get('causal_gate')
        n_samples = int(max(
            cg.size if cg is not None else 0,
            ca.size if ca is not None else 0,
        ))

        # gate_values.csv — per-sample side-by-side. Pad shorter columns
        # with NaN so column-major dump can interleave them safely.
        if n_samples > 0:
            def _padded(arr):
                if arr is None:
                    return np.full(n_samples, np.nan, dtype=float)
                if arr.size == n_samples:
                    return arr.astype(float)
                # Static gate: replicate to per-sample length so every row
                # carries the gate value the model actually applied.
                if arr.size == 1:
                    return np.full(n_samples, float(arr.item()), dtype=float)
                # Length mismatch we don't know how to align — pad with NaN.
                out = np.full(n_samples, np.nan, dtype=float)
                out[:min(n_samples, arr.size)] = arr[:n_samples]
                return out

            dump_csv_columns(
                {
                    'sample_idx': np.arange(n_samples, dtype=np.int64),
                    'concept_gate': _padded(cg),
                    'causal_gate': _padded(ca),
                },
                with_basename(png_path, 'gate_values.csv'),
            )

        def _stats(arr):
            if arr is None or arr.size == 0:
                return None
            return {
                'mean': float(arr.mean()),
                'std': float(arr.std()),
                'min': float(arr.min()),
                'max': float(arr.max()),
                'is_constant': bool(arr.std() < _CONST_TOL),
            }

        cg_stats = _stats(cg)
        ca_stats = _stats(ca)
        dump_json(
            analyzer_metadata(
                n_samples=n_samples,
                dataset_name=self.dataset_name,
                concept_gate_mean=(cg_stats or {}).get('mean'),
                concept_gate_std=(cg_stats or {}).get('std'),
                causal_gate_mean=(ca_stats or {}).get('mean'),
                causal_gate_std=(ca_stats or {}).get('std'),
                concept_gate_is_constant=(cg_stats or {}).get('is_constant'),
                causal_gate_is_constant=(ca_stats or {}).get('is_constant'),
                # legacy convenience: a single boolean is True only if
                # BOTH are constant. Per-gate flags above are authoritative.
                is_constant=(
                    bool((cg_stats or {}).get('is_constant'))
                    and bool((ca_stats or {}).get('is_constant'))
                ) if (cg_stats and ca_stats) else False,
                concept_gate_full=cg_stats,
                causal_gate_full=ca_stats,
            ),
            with_basename(png_path, 'gate_summary.json'),
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

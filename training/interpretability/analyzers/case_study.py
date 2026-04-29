"""
interpretability/analyzers/case_study.py
========================================
Per-sample "paper-figure" gallery: for the top-K most-uncertain, top-K
confidently-correct, and top-K confidently-wrong samples, render one
PNG per sample that stitches together:

  [frame]  |  p(fake) + uncertainty gauges
           |  stacked branch evidence bar
           |  top-5 consistency-rule violations
           |  r_diff radar across the 4 SCM sub-graphs

Image paths are supplied via `set_image_paths()` after inference so the
analyzer does not have to keep tensors resident in RAM during the eval
loop.
"""

import os
from typing import List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .base import BaseAnalyzer
try:
    from ..pretty_names import pretty_many
except ImportError:  # flat-import fallback
    from interpretability.pretty_names import pretty_many  # type: ignore


def _load_rule_names(n_rules: int):
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


class CaseStudyAnalyzer(BaseAnalyzer):
    name = 'case_study'
    required_keys = (
        'prob', 'uncertainty', 'alpha',
        'spatial_evidence', 'concept_evidence', 'causal_evidence',
        'violations', 'r_diff_g',
        'concept_gate', 'causal_gate',
    )

    SUBGRAPH_NAMES = [
        'identity', 'structural', 'noise', 'spectral',
    ]

    def __init__(self, num_samples: int = 6):
        super().__init__()
        self.num_samples = int(num_samples)
        self._image_paths: List[str] = []

    # ── external inputs ───────────────────────────────────────────────────

    def set_image_paths(self, paths: List[str]) -> None:
        self._image_paths = list(paths)

    # ── sample selection ──────────────────────────────────────────────────

    def _select_indices(self, prob, unc, labels) -> dict:
        preds = (prob >= 0.5).astype(int)
        correct = preds == labels
        n = len(labels)
        k = min(self.num_samples, max(1, n // 4))

        groups = {}
        if n > 0:
            correct_idx = np.where(correct)[0]
            wrong_idx = np.where(~correct)[0]
            # Confident: lowest uncertainty among correct/wrong respectively
            if correct_idx.size:
                order = np.argsort(unc[correct_idx])[:k]
                groups['confident_correct'] = correct_idx[order]
            if wrong_idx.size:
                order = np.argsort(unc[wrong_idx])[:k]
                groups['confident_wrong'] = wrong_idx[order]
            # Uncertain split by correctness — replaces the old single
            # 'uncertain' bucket so the gallery distinguishes
            # "model hesitated but got it right" from "model hesitated
            # and missed".
            if correct_idx.size:
                order = np.argsort(-unc[correct_idx])[:k]
                groups['uncertain_correct'] = correct_idx[order]
            if wrong_idx.size:
                order = np.argsort(-unc[wrong_idx])[:k]
                groups['uncertain_wrong'] = wrong_idx[order]
        return groups

    # ── rendering ─────────────────────────────────────────────────────────

    @staticmethod
    def _draw_gauge(ax, value, label, color):
        ax.barh([0], [value], color=color, edgecolor='black', height=0.5)
        ax.barh([0], [1.0], color='none', edgecolor='#888', height=0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xticks([0, 0.5, 1.0])
        ax.tick_params(axis='x', labelsize=7)
        ax.set_title(f'{label}: {value:.3f}', fontsize=9)

    @staticmethod
    def _draw_evidence_stack(ax, parts):
        names = list(parts.keys())
        vals = [parts[n] for n in names]
        colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
        bottom = 0.0
        for name, v, c in zip(names, vals, colors):
            ax.bar(['evidence'], [v], bottom=bottom, color=c,
                   label=f'{name}: {v:.2f}', edgecolor='white')
            bottom += v
        ax.set_title('Branch evidence', fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc='upper right')

    @staticmethod
    def _draw_top_rules(ax, violations, rule_names, top_k=5):
        order = np.argsort(-violations)[:top_k]
        shown = [(rule_names[i], violations[i]) for i in order if violations[i] > 0.01]
        if not shown:
            ax.text(0.5, 0.5, 'No significant rule violations',
                    ha='center', va='center', transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return
        raw_names = [s[0] for s in shown]
        names = pretty_many(raw_names)  # cr_mutual_mouth → R_{mouth open×closed}
        vals = [s[1] for s in shown]
        y = np.arange(len(names))
        ax.barh(y, vals, color='#d62728', alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0, max(1.0, max(vals) * 1.05))
        ax.set_title('Top rule violations', fontsize=9)
        ax.tick_params(axis='x', labelsize=7)

    @staticmethod
    def _draw_radar(ax, values, names):
        n = len(names)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles += angles[:1]
        vals = list(values) + [values[0]]
        ax.plot(angles, vals, 'o-', color='#1f77b4', linewidth=1.5)
        ax.fill(angles, vals, alpha=0.2, color='#1f77b4')
        ax.set_thetagrids(np.degrees(angles[:-1]), names, fontsize=7)
        ax.set_title('r_diff per sub-graph', fontsize=9, pad=12)
        ax.tick_params(axis='y', labelsize=6)

    def _render_one(self, idx, img_path, prob, unc, spatial_ev,
                    concept_ev, causal_ev, violations, r_diff, gates,
                    rule_names, label, pred_label, save_path):
        fig = plt.figure(figsize=(15, 5.5))
        gs = fig.add_gridspec(
            2, 5, width_ratios=[2.2, 0.9, 1.1, 1.6, 1.4],
            height_ratios=[1, 1], wspace=0.45, hspace=0.55,
        )

        # Panel 1: frame (spans 2 rows)
        ax_img = fig.add_subplot(gs[:, 0])
        try:
            img = plt.imread(img_path)
            ax_img.imshow(img)
        except Exception as e:
            ax_img.text(0.5, 0.5, f'<image load failed>\n{e}',
                        ha='center', va='center', transform=ax_img.transAxes,
                        fontsize=9)
        ax_img.axis('off')
        ok_mark = 'OK' if label == pred_label else 'MISS'
        ax_img.set_title(
            f'idx {idx} — label={"FAKE" if label else "REAL"}, '
            f'pred={"FAKE" if pred_label else "REAL"} [{ok_mark}]',
            fontsize=10)

        # Panel 2: p(fake) and uncertainty gauges
        ax_p = fig.add_subplot(gs[0, 1])
        self._draw_gauge(ax_p, float(prob), 'p(fake)', '#d62728')
        ax_u = fig.add_subplot(gs[1, 1])
        self._draw_gauge(ax_u, float(unc), 'uncertainty u', '#9467bd')

        # Panel 3: evidence stack
        ev_parts = {}
        if spatial_ev is not None:
            ev_parts['spatial'] = float(spatial_ev.sum())
        if concept_ev is not None:
            ev_parts['concept'] = float(concept_ev.sum())
        if causal_ev is not None:
            ev_parts['causal'] = float(causal_ev.sum())
        ax_ev = fig.add_subplot(gs[:, 2])
        if ev_parts:
            self._draw_evidence_stack(ax_ev, ev_parts)
        else:
            ax_ev.set_axis_off()

        # Panel 4: top rules
        ax_rules = fig.add_subplot(gs[:, 3])
        if violations is not None:
            self._draw_top_rules(ax_rules, violations, rule_names)
        else:
            ax_rules.set_axis_off()

        # Panel 5: radar (polar)
        if r_diff is not None:
            ax_rad = fig.add_subplot(gs[:, 4], projection='polar')
            sg_names = self.SUBGRAPH_NAMES[:len(r_diff)]
            self._draw_radar(ax_rad, r_diff, sg_names)
        else:
            ax_rad = fig.add_subplot(gs[:, 4])
            ax_rad.set_axis_off()

        # Caption strip with gates
        if gates:
            gate_str = '  '.join(f'{k}={v:.2f}' for k, v in gates.items())
            fig.text(0.02, 0.02, gate_str, fontsize=8, color='#333')

        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    # ── interface ──────────────────────────────────────────────────────────

    def analyze(self) -> dict:
        prob = self._stack('prob')
        unc = self._stack('uncertainty')
        if prob is None or unc is None:
            return {}
        prob = prob.reshape(-1)
        unc = unc.reshape(-1)
        labels = self._all_labels.astype(int).reshape(-1)

        self._prob = prob
        self._unc = unc
        self._labels_flat = labels
        self._spatial_ev = self._stack('spatial_evidence')
        self._concept_ev = self._stack('concept_evidence')
        self._causal_ev = self._stack('causal_evidence')
        self._violations = self._stack('violations')
        self._r_diff = self._stack('r_diff_g')
        # Gates: may be scalar (static) or per-sample — normalise
        cg = self._stack('concept_gate')
        ca = self._stack('causal_gate')
        self._concept_gate = cg.reshape(-1) if cg is not None else None
        self._causal_gate = ca.reshape(-1) if ca is not None else None

        n_rules = self._violations.shape[1] if self._violations is not None else 0
        self._rule_names = _load_rule_names(n_rules)
        self._groups = self._select_indices(prob, unc, labels)

        return {
            'group_sizes': {g: int(len(idx)) for g, idx in self._groups.items()},
            'num_samples_requested': self.num_samples,
            'has_images': bool(self._image_paths),
        }

    def visualize(self, save_dir: str) -> None:
        if not hasattr(self, '_groups'):
            return
        if not self._image_paths:
            # Write a tiny marker so users know this analyzer ran but
            # could not render images.
            with open(os.path.join(save_dir, 'case_study_no_images.txt'), 'w') as f:
                f.write('CaseStudyAnalyzer: image_paths not provided; '
                        'skipping gallery render.\n')
            return

        gallery_dir = os.path.join(save_dir, 'case_study')
        os.makedirs(gallery_dir, exist_ok=True)

        preds = (self._prob >= 0.5).astype(int)
        for group_name, indices in self._groups.items():
            group_dir = os.path.join(gallery_dir, group_name)
            os.makedirs(group_dir, exist_ok=True)
            for rank, idx in enumerate(indices):
                idx = int(idx)
                if idx >= len(self._image_paths):
                    continue
                img_path = self._image_paths[idx]
                spatial_ev = (self._spatial_ev[idx]
                              if self._spatial_ev is not None else None)
                concept_ev = (self._concept_ev[idx]
                              if self._concept_ev is not None else None)
                causal_ev = (self._causal_ev[idx]
                             if self._causal_ev is not None else None)
                violations = (self._violations[idx]
                              if self._violations is not None else None)
                r_diff = (self._r_diff[idx]
                          if self._r_diff is not None else None)

                # Gates may be per-sample (len == N) or scalar/static
                # (any shorter length — collected once per batch). Only
                # index per-sample if the length matches; otherwise use
                # the mean as the single "global" gate value.
                n_samples = len(self._labels_flat)
                gates = {}
                for name, buf in [('concept_gate', self._concept_gate),
                                  ('causal_gate', self._causal_gate)]:
                    if buf is None:
                        continue
                    gates[name] = float(buf[idx]) if len(buf) == n_samples \
                        else float(buf.mean())

                save_path = os.path.join(
                    group_dir, f'{rank:02d}_idx{idx}.png')
                try:
                    self._render_one(
                        idx=idx,
                        img_path=img_path,
                        prob=self._prob[idx],
                        unc=self._unc[idx],
                        spatial_ev=spatial_ev,
                        concept_ev=concept_ev,
                        causal_ev=causal_ev,
                        violations=violations,
                        r_diff=r_diff,
                        gates=gates,
                        rule_names=self._rule_names,
                        label=int(self._labels_flat[idx]),
                        pred_label=int(preds[idx]),
                        save_path=save_path,
                    )
                except Exception:
                    plt.close('all')
                    continue

    def explain_sample(self, idx: int) -> str:
        return ''

"""
interpretability/analyzers/case_study.py
========================================
Per-sample "paper-figure" gallery.

2026-05-04 — Goal 4 expansion:
  * Default samples-per-group raised 6 → 10 (config knob
    ``interpretability.case_study_samples``; respects values up to 20).
  * Five groups now emitted, each in its own subdirectory under
    ``case_study/``:
        confident_correct  (lowest u  among correct)
        confident_wrong    (lowest u  among incorrect)
        uncertain_correct  (highest u among correct)
        uncertain_wrong    (highest u among incorrect)
        borderline         (smallest |p − 0.5| regardless of correctness)
  * Per-sample sidecar JSON  ``<rank>_idx<idx>.json`` written alongside
    every PNG so authors can filter offline (e.g. "uncertain_wrong with
    u > 0.20 from CDFv2 with the noise sub-graph dominant").
  * Rule labels in the panel are pulled from the active rule registry
    (consistency_rules_v8 retained set when the model is trained with
    that version, else v7, else v5 fallback). Falls back to
    ``rule_<i>`` integer labels with a warning rather than crashing.
  * The constant-gate footer (``concept_gate=X.XX causal_gate=Y.YY``)
    is **suppressed when the gates are constant across the run**. The
    decision is taken from the per-gate std at panel render time
    (matches what gate_summary.json reports). When non-constant, the
    footer is rendered as before.
  * The radar inside the panel is replaced with a horizontal 4-bar
    mini-plot of Δr_g per sub-graph (identity / structural / noise /
    spectral) — bars are readable at print size, radars are not.
    A separate ``<rank>_idx<idx>_radar.png`` is also written for
    slide-deck use.

Image paths are supplied via :py:meth:`set_image_paths` after inference
so the analyzer does not have to keep tensors resident in RAM during the
eval loop.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .base import BaseAnalyzer
from ..data_export import dump_json
try:
    from ..pretty_names import pretty_many
except ImportError:  # flat-import fallback
    from interpretability.pretty_names import pretty_many  # type: ignore

logger = logging.getLogger(__name__)


# ── Constant-gate detection threshold (matches gate_analysis.py). ──────
_GATE_CONST_TOL = 1e-6

# ── Maximum per-group sample count we'll honour from config. ───────────
MAX_SAMPLES_PER_GROUP = 20

# ── Group ordering for the gallery directory tree. ─────────────────────
GROUP_NAMES = (
    'confident_correct',
    'confident_wrong',
    'uncertain_correct',
    'uncertain_wrong',
    'borderline',
)


def _load_rule_names(n_rules: int) -> Tuple[List[str], str]:
    """Resolve rule names. Returns (names, source_tag).

    source_tag is one of {'v8_retained', 'v7', 'v5', 'fallback_integer'}
    so the caller can log which registry was hit.
    """
    if n_rules == 18:
        # v8 retained set comes from the frozen YAML.
        try:
            import yaml
            here = os.path.abspath(__file__)
            repo_root = os.path.abspath(os.path.join(
                os.path.dirname(here), '..', '..', '..'))
            yaml_path = os.path.join(repo_root, 'configs',
                                     'retained_predicates.yaml')
            if os.path.exists(yaml_path):
                with open(yaml_path) as f:
                    spec = yaml.safe_load(f) or {}
                retained = spec.get('retained_predicates') or []
                names = [str(e['name']) for e in retained]
                if len(names) == n_rules:
                    return names, 'v8_retained'
        except Exception as exc:  # pragma: no cover - best-effort
            logger.warning(f'v8_retained name lookup failed: {exc}')
    if n_rules == 23:
        try:
            from networks.nesy_defake.semantic.consistency_rules_v7 import (
                TRAINING_RULE_NAMES_V7,
            )
            return list(TRAINING_RULE_NAMES_V7), 'v7'
        except ImportError:
            pass
    if n_rules == 20:
        try:
            from networks.nesy_defake.semantic.consistency_rules import (
                TRAINING_RULE_NAMES,
            )
            return list(TRAINING_RULE_NAMES), 'v5'
        except ImportError:
            pass
    logger.warning(
        f'CaseStudy: no named rule registry matches n_rules={n_rules}; '
        f'falling back to integer labels rule_<i>.')
    return [f'rule_{i}' for i in range(n_rules)], 'fallback_integer'


class CaseStudyAnalyzer(BaseAnalyzer):
    name = 'case_study'
    required_keys = (
        'prob', 'uncertainty', 'alpha',
        'spatial_evidence', 'concept_evidence', 'causal_evidence',
        'violations', 'r_diff_g',
        'concept_gate', 'causal_gate',
    )

    SUBGRAPH_NAMES = ('identity', 'structural', 'noise', 'spectral')

    def __init__(self, num_samples: int = 10):
        super().__init__()
        # Cap upward so a stray YAML knob can't spawn 100 panels per group.
        self.num_samples = int(min(max(int(num_samples), 1),
                                   MAX_SAMPLES_PER_GROUP))
        self._image_paths: List[str] = []
        self._rule_source: str = 'unset'

    # ── external inputs ───────────────────────────────────────────────────

    def set_image_paths(self, paths: List[str]) -> None:
        self._image_paths = list(paths)

    # ── sample selection ──────────────────────────────────────────────────

    def _select_indices(self, prob, unc, labels) -> Dict[str, np.ndarray]:
        preds = (prob >= 0.5).astype(int)
        correct = preds == labels
        n = len(labels)
        # Don't ask for more samples than 1/4 of the population per group;
        # otherwise tiny test slices end up with overlapping picks.
        k = min(self.num_samples, max(1, n // 4))

        groups: Dict[str, np.ndarray] = {}
        if n == 0:
            return groups

        correct_idx = np.where(correct)[0]
        wrong_idx = np.where(~correct)[0]

        # Confident: lowest u within the (correct | wrong) sub-population
        if correct_idx.size:
            order = np.argsort(unc[correct_idx])[:k]
            groups['confident_correct'] = correct_idx[order]
        if wrong_idx.size:
            order = np.argsort(unc[wrong_idx])[:k]
            groups['confident_wrong'] = wrong_idx[order]

        # Uncertain: highest u within the (correct | wrong) sub-population
        if correct_idx.size:
            order = np.argsort(-unc[correct_idx])[:k]
            groups['uncertain_correct'] = correct_idx[order]
        if wrong_idx.size:
            order = np.argsort(-unc[wrong_idx])[:k]
            groups['uncertain_wrong'] = wrong_idx[order]

        # Borderline: smallest |p − 0.5| regardless of correctness
        margin = np.abs(prob - 0.5)
        order = np.argsort(margin)[:k]
        groups['borderline'] = order

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
        shown = [(rule_names[i], violations[i]) for i in order
                 if violations[i] > 0.01]
        if not shown:
            ax.text(0.5, 0.5, 'No significant rule violations',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=9)
            ax.set_axis_off()
            return
        raw_names = [s[0] for s in shown]
        names = pretty_many(raw_names)
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
    def _draw_rdiff_minibar(ax, values: np.ndarray, names: Tuple[str, ...]):
        """Horizontal 4-bar Δr_g per sub-graph — replaces the radar
        inside the panel because bars stay readable at print size.
        """
        names = list(names[:len(values)])
        y = np.arange(len(names))
        bars = ax.barh(y, values, color='#1f77b4', alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=7)
        ax.invert_yaxis()
        # Annotate values to the right of each bar.
        vmax = float(np.max(values)) if len(values) else 1.0
        pad = 0.02 * (vmax + 1e-6)
        for b, v in zip(bars, values):
            ax.text(b.get_width() + pad, b.get_y() + b.get_height() / 2,
                    f'{float(v):.2f}', va='center', ha='left', fontsize=7,
                    color='#222')
        ax.set_xlim(0, vmax * 1.20 + 1e-6)
        ax.set_title(r'$\Delta r_g$ per sub-graph', fontsize=9)
        ax.tick_params(axis='x', labelsize=7)

    @staticmethod
    def _draw_radar(ax, values, names):
        n = len(names)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles += angles[:1]
        vals = list(values) + [values[0]]
        ax.plot(angles, vals, 'o-', color='#1f77b4', linewidth=1.5)
        ax.fill(angles, vals, alpha=0.2, color='#1f77b4')
        ax.set_thetagrids(np.degrees(angles[:-1]), list(names), fontsize=7)
        ax.set_title('r_diff per sub-graph', fontsize=9, pad=12)
        ax.tick_params(axis='y', labelsize=6)

    # ── panel composition ────────────────────────────────────────────────

    def _render_one(self, idx, img_path, prob, unc, spatial_ev,
                    concept_ev, causal_ev, violations, r_diff, gates,
                    show_gate_footer, rule_names, label, pred_label,
                    save_path):
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

        # Panel 5: Δr_g mini-bar (replaces radar at print size)
        ax_bar = fig.add_subplot(gs[:, 4])
        if r_diff is not None and len(r_diff):
            sg_names = self.SUBGRAPH_NAMES[:len(r_diff)]
            self._draw_rdiff_minibar(ax_bar, np.asarray(r_diff), sg_names)
        else:
            ax_bar.set_axis_off()

        # Caption strip with gates — suppressed when gates are constant.
        if gates and show_gate_footer:
            gate_str = '  '.join(f'{k}={v:.2f}' for k, v in gates.items())
            fig.text(0.02, 0.02, gate_str, fontsize=8, color='#333')

        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _render_radar_only(self, r_diff, save_path):
        """Slide-deck radar — kept available outside the print panel."""
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='polar')
        sg_names = self.SUBGRAPH_NAMES[:len(r_diff)]
        self._draw_radar(ax, np.asarray(r_diff), sg_names)
        fig.tight_layout()
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
        cg = self._stack('concept_gate')
        ca = self._stack('causal_gate')
        self._concept_gate = cg.reshape(-1) if cg is not None else None
        self._causal_gate = ca.reshape(-1) if ca is not None else None

        # Gate-constancy decision — drives footer suppression in the panel.
        self._gates_constant = self._detect_constant_gates()

        n_rules = (self._violations.shape[1]
                   if self._violations is not None else 0)
        self._rule_names, self._rule_source = _load_rule_names(n_rules)
        if self._rule_source != 'unset':
            logger.info(
                f'CaseStudy: rule registry = {self._rule_source} '
                f'({n_rules} rules)')

        self._groups = self._select_indices(prob, unc, labels)

        return {
            'group_sizes': {g: int(len(idx))
                            for g, idx in self._groups.items()},
            'num_samples_requested': self.num_samples,
            'has_images': bool(self._image_paths),
            'rule_registry': self._rule_source,
            'gates_constant': self._gates_constant,
        }

    def _detect_constant_gates(self) -> Dict[str, bool]:
        out: Dict[str, bool] = {'concept': False, 'causal': False}
        for name, arr in [('concept', self._concept_gate),
                          ('causal', self._causal_gate)]:
            if arr is None or arr.size == 0:
                out[name] = False
                continue
            if arr.size == 1:
                out[name] = True
            else:
                out[name] = bool(arr.std() < _GATE_CONST_TOL)
        return out

    def visualize(self, save_dir: str) -> None:
        if not hasattr(self, '_groups'):
            return
        if not self._image_paths:
            with open(os.path.join(save_dir, 'case_study_no_images.txt'), 'w') as f:
                f.write('CaseStudyAnalyzer: image_paths not provided; '
                        'skipping gallery render.\n')
            return

        gallery_dir = os.path.join(save_dir, 'case_study')
        os.makedirs(gallery_dir, exist_ok=True)

        # Footer suppression: only render gate strings when at least one
        # gate is non-constant. Per-gate flag is consulted again at the
        # render call so the footer drops the constant gate but keeps the
        # other one if mixed.
        any_non_constant = any(not v for v in self._gates_constant.values())

        preds = (self._prob >= 0.5).astype(int)
        for group_name in GROUP_NAMES:
            indices = self._groups.get(group_name)
            if indices is None or len(indices) == 0:
                continue
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

                # Gates may be per-sample (len == N) or scalar/static.
                n_samples = len(self._labels_flat)
                gates: Dict[str, float] = {}
                gate_per_sample: Dict[str, bool] = {}
                for name, buf in [('concept_gate', self._concept_gate),
                                  ('causal_gate', self._causal_gate)]:
                    if buf is None or buf.size == 0:
                        continue
                    if len(buf) == n_samples:
                        gates[name] = float(buf[idx])
                        gate_per_sample[name] = True
                    else:
                        gates[name] = float(buf.mean())
                        gate_per_sample[name] = False

                # Drop gates flagged constant from the footer entirely;
                # render the remaining ones if any are non-constant.
                gates_for_footer: Dict[str, float] = {}
                for k, v in gates.items():
                    short = k.replace('_gate', '')
                    if not self._gates_constant.get(short, False):
                        gates_for_footer[k] = v
                show_gate_footer = bool(gates_for_footer) and any_non_constant

                save_path = os.path.join(
                    group_dir, f'{rank:02d}_idx{idx}.png')
                json_path = os.path.join(
                    group_dir, f'{rank:02d}_idx{idx}.json')
                radar_path = os.path.join(
                    group_dir, f'{rank:02d}_idx{idx}_radar.png')

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
                        gates=gates_for_footer,
                        show_gate_footer=show_gate_footer,
                        rule_names=self._rule_names,
                        label=int(self._labels_flat[idx]),
                        pred_label=int(preds[idx]),
                        save_path=save_path,
                    )
                    if r_diff is not None and len(r_diff):
                        self._render_radar_only(r_diff, radar_path)
                except Exception as exc:
                    logger.warning(
                        f'CaseStudy render {group_name}/{rank}/{idx} '
                        f'failed: {exc}')
                    plt.close('all')
                    continue

                # ── Sidecar metadata JSON ─────────────────────────────
                meta = self._build_sidecar(
                    idx=idx, group_name=group_name, img_path=img_path,
                    spatial_ev=spatial_ev, concept_ev=concept_ev,
                    causal_ev=causal_ev, violations=violations,
                    r_diff=r_diff, gates=gates,
                    label=int(self._labels_flat[idx]),
                    pred_label=int(preds[idx]),
                )
                try:
                    dump_json(meta, json_path)
                except Exception as exc:
                    logger.warning(
                        f'CaseStudy sidecar JSON {json_path} failed: {exc}')

    def _build_sidecar(
        self,
        *,
        idx: int,
        group_name: str,
        img_path: str,
        spatial_ev,
        concept_ev,
        causal_ev,
        violations,
        r_diff,
        gates,
        label,
        pred_label,
    ) -> dict:
        # Top-rule violations sorted desc, with named labels.
        top_rule_violations: List[dict] = []
        if violations is not None:
            v = np.asarray(violations).reshape(-1)
            order = np.argsort(-v)
            for j in order[:5]:
                if v[j] <= 0.01:
                    break
                rule_name = (self._rule_names[j]
                             if j < len(self._rule_names)
                             else f'rule_{int(j)}')
                top_rule_violations.append(
                    {'rule_name': str(rule_name),
                     'score': float(v[j])})
        # Branch evidence summary (sums to one number per branch).
        branch_evidence = {}
        if spatial_ev is not None:
            branch_evidence['spatial'] = float(np.asarray(spatial_ev).sum())
        if concept_ev is not None:
            branch_evidence['concept'] = float(np.asarray(concept_ev).sum())
        if causal_ev is not None:
            branch_evidence['causal'] = float(np.asarray(causal_ev).sum())
        # r_diff per named sub-graph.
        rd: Dict[str, float] = {}
        if r_diff is not None:
            arr = np.asarray(r_diff).reshape(-1)
            for i, name in enumerate(self.SUBGRAPH_NAMES[:len(arr)]):
                rd[name] = float(arr[i])
        return {
            'sample_idx': int(idx),
            'true_label': int(label),
            'pred_label': int(pred_label),
            'p_fake': float(self._prob[idx]),
            'u': float(self._unc[idx]),
            'branch_evidence': branch_evidence,
            'top_rule_violations': top_rule_violations,
            'r_diff': rd,
            'concept_gate': gates.get('concept_gate'),
            'causal_gate': gates.get('causal_gate'),
            'image_path': str(img_path),
            'group': group_name,
            'dataset_name': self.dataset_name,
            'rule_registry': self._rule_source,
        }

    def explain_sample(self, idx: int) -> str:
        return ''

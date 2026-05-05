"""
interpretability/analyzers/scm_analysis.py
==========================================
Level 4-SCM: Structural Causal Model analysis — adjacency matrix divergence,
top divergent edges with named nodes, per-subgraph divergence profiles,
DAG sparsity metrics.

2026-05-04 — Goals 1 + 2:
  * r_diff_radar.csv             (per-sub-graph mean / std, real vs fake)
  * r_diff_per_sample.npz        (full (N, 4) r_diff matrix + labels +
                                  method labels when available)
  * r_diff_radar_per_method.csv  (per-method mean per sub-graph)
  * dominant_subgraph_frequency.csv
  * per sub-graph: ``<sub>_adjacency.npz`` + ``<sub>_top_edges.csv``
  * Per-sub-graph r_diff histograms use the FULL population.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from .base import BaseAnalyzer
from .. import visualization as viz
from ..data_export import (
    analyzer_metadata, dump_csv, dump_csv_columns, dump_json, dump_npz,
    sibling_path, with_basename,
)

logger = logging.getLogger(__name__)


class SCMAnalyzer(BaseAnalyzer):
    name = 'scm_analysis'
    # Per-sample residual magnitudes (B, 4) and per-sub-graph (B,).
    # scm_input_*: per-sample SCM input vectors used to compute
    # activation-weighted edge scores (data-driven divergence).
    required_keys = (
        'r_diff_g',
        'r_diff_identity',
        'r_diff_forensic_structural',
        'r_diff_forensic_noise',
        'r_diff_forensic_spectral',
        'scm_input_identity',
        'scm_input_forensic_structural',
        'scm_input_forensic_noise',
        'scm_input_forensic_spectral',
    )

    # Stable ordering matches ImprovedCausalBranch.forward stack.
    SUBGRAPH_NAMES = [
        'identity', 'forensic_structural', 'forensic_noise', 'forensic_spectral',
    ]

    def __init__(
        self,
        top_k: int = 20,
        top_k_levels: Tuple[int, ...] = (20, 30),
        distinctive_keep_ratio: float = 0.7,
    ):
        super().__init__()
        self.top_k = top_k
        # Render the per-class graphs / paired-edge bars / top-edge tables
        # at multiple K levels so the paper can show how the picture
        # evolves as more contributing edges are included.
        self.top_k_levels = tuple(sorted(set(top_k_levels)))
        # plot_class_distinctive_graphs: edges where A_fake/A_real falls
        # below this ratio are rendered as 'missing in fake' (faded
        # dashed). Lower ratio → fewer edges marked missing.
        self.distinctive_keep_ratio = float(distinctive_keep_ratio)
        self._adjacencies: Dict[str, np.ndarray] = {}
        self._node_names: Dict[str, List[str]] = {}
        self._method_labels: Optional[np.ndarray] = None
        self._load_node_names()

    def set_method_labels(self, label_spe) -> None:
        """Attach a 1-D array of per-sample specific-method labels."""
        arr = self._to_numpy(label_spe).reshape(-1)
        self._method_labels = arr

    # ── node name construction ─────────────────────────────────────────────

    def _load_node_names(self):
        z_names = [f'z_{i}' for i in range(32)]

        # Identity: z(32) + curated(51) + rules(20)
        try:
            from networks.nesy_defake.semantic.refined_attributes import CAUSAL_ATTRIBUTE_NAMES
            curated = list(CAUSAL_ATTRIBUTE_NAMES)
        except ImportError:
            curated = [f'attr_{i}' for i in range(51)]
        try:
            from networks.nesy_defake.semantic.consistency_rules import TRAINING_RULE_NAMES
            rules = list(TRAINING_RULE_NAMES)
        except ImportError:
            rules = [f'rule_{i}' for i in range(20)]
        self._node_names['identity'] = z_names + curated + rules

        # Forensic sub-graphs: z(32) + forensic features (sliced)
        try:
            from networks.nesy_defake.semantic.forensic_features import FORENSIC_FEATURE_NAMES
            ff_names = list(FORENSIC_FEATURE_NAMES)
        except ImportError:
            ff_names = [f'ff_{i}' for i in range(83)]

        try:
            from networks.nesy_defake.improved_scm_branch import FORENSIC_SUBGRAPH_GROUPS
            groups = list(FORENSIC_SUBGRAPH_GROUPS)
        except ImportError:
            groups = [
                ('structural', 0, 30, 30),
                ('noise', 30, 73, 43),
                ('spectral', 73, 83, 10),
            ]

        for gname, start, end, _ in groups:
            self._node_names[f'forensic_{gname}'] = z_names + ff_names[start:end]

    # ── collection ─────────────────────────────────────────────────────────

    def collect_adjacencies(self, model) -> None:
        """One-time collection of adjacency matrices from model parameters."""
        branch = getattr(model, 'causal_branch', None)
        if branch is None:
            return
        for sg_name in ['identity', 'forensic_structural', 'forensic_noise', 'forensic_spectral']:
            for label in ['real', 'fake']:
                key = f'A_{sg_name}_{label}'
                pair = getattr(branch, f'sg_{sg_name}', None)
                if pair is None:
                    for attr_name in dir(branch):
                        if sg_name.replace('forensic_', '') in attr_name and 'sg' in attr_name:
                            pair = getattr(branch, attr_name, None)
                            break
                if pair is not None:
                    scm = getattr(pair, f'scm_{label}', None)
                    if scm is not None and hasattr(scm, 'get_adjacency'):
                        self._adjacencies[key] = scm.get_adjacency().detach().cpu().numpy()
                    elif scm is not None:
                        try:
                            w1 = scm.net[0].weight.detach().cpu().abs()
                            w2 = scm.net[2].weight.detach().cpu().abs()
                            self._adjacencies[key] = (w2 @ w1).numpy()
                        except (IndexError, AttributeError):
                            pass

    def collect(self, preds: dict, labels) -> None:
        # Accumulate per-sample r_diff signals (via BaseAnalyzer buffers)
        super().collect(preds, labels)
        # Also pick up adjacency matrices returned in the pred dict.
        # These are single (d, d) tensors per branch — stash only once.
        for k, v in preds.items():
            if k.startswith('A_') and isinstance(v, (torch.Tensor, np.ndarray)):
                if k not in self._adjacencies:
                    self._adjacencies[k] = self._to_numpy(v)

    # ── edge helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _top_edges(A: np.ndarray, node_names: List[str], top_k: int
                   ) -> List[Tuple[str, str, float]]:
        n = min(A.shape[0], len(node_names))
        flat = A[:n, :n].flatten()
        top_idx = np.argsort(-np.abs(flat))[:top_k]
        return [(node_names[divmod(i, n)[0]], node_names[divmod(i, n)[1]], float(flat[i]))
                for i in top_idx if abs(flat[i]) > 1e-6]

    @staticmethod
    def _divergent_edges(A_real: np.ndarray, A_fake: np.ndarray,
                         node_names: List[str], top_k: int
                         ) -> List[Tuple[str, str, float]]:
        n = min(A_real.shape[0], A_fake.shape[0], len(node_names))
        diff = np.abs(A_real[:n, :n] - A_fake[:n, :n])
        flat = diff.flatten()
        top_idx = np.argsort(-flat)[:top_k]
        return [(node_names[divmod(i, n)[0]], node_names[divmod(i, n)[1]], float(flat[i]))
                for i in top_idx if flat[i] > 1e-6]

    # ── analysis ───────────────────────────────────────────────────────────

    def analyze(self) -> dict:
        results = {
            'num_adjacencies': len(self._adjacencies),
            'subgraph_divergence': {},
            'top_divergent_edges': {},
            'dag_sparsity': {},
        }

        # ── Per-sample r_diff_g signal ─────────────────────────────────────
        r_diff = self._stack('r_diff_g')
        if r_diff is None:
            # Fallback: rebuild (B, 4) from individual per-sub-graph keys
            cols = [self._stack(f'r_diff_{sg}') for sg in self.SUBGRAPH_NAMES]
            if all(c is not None for c in cols):
                r_diff = np.stack([c.reshape(-1) for c in cols], axis=1)

        labels = None
        if self._labels:
            labels = self._all_labels
        if r_diff is not None and labels is not None:
            r_diff = r_diff.reshape(len(labels), -1)
            real_mask = labels == 0
            fake_mask = labels == 1
            per_sg_mean = {}
            for i, sg in enumerate(self.SUBGRAPH_NAMES[:r_diff.shape[1]]):
                per_sg_mean[sg] = {
                    'mean_real': float(r_diff[real_mask, i].mean()) if real_mask.any() else 0.0,
                    'mean_fake': float(r_diff[fake_mask, i].mean()) if fake_mask.any() else 0.0,
                    'gap': float(
                        (r_diff[fake_mask, i].mean() if fake_mask.any() else 0.0)
                        - (r_diff[real_mask, i].mean() if real_mask.any() else 0.0)
                    ),
                }
            results['r_diff_per_subgraph'] = per_sg_mean
            dominant = np.argmax(r_diff, axis=1) if r_diff.size else None
            if dominant is not None:
                results['dominant_subgraph_frequency'] = {
                    sg: float((dominant == i).mean())
                    for i, sg in enumerate(self.SUBGRAPH_NAMES[:r_diff.shape[1]])
                }
            self._r_diff = r_diff
            self._r_diff_labels = labels

            # ── Per-branch, per-class mean |x_j| activations ─────────────
            # Used for activation-weighted edge scoring in
            # plot_class_distinctive_graphs. Shape per branch: (d,) where
            # d = SCM input dim for that sub-graph.
            self._mean_act_real: Dict[str, np.ndarray] = {}
            self._mean_act_fake: Dict[str, np.ndarray] = {}
            for sg in self.SUBGRAPH_NAMES:
                x_sg = self._stack(f'scm_input_{sg}')
                if x_sg is None:
                    continue
                x_sg = x_sg.reshape(len(labels), -1)
                if real_mask.any():
                    self._mean_act_real[sg] = np.abs(x_sg[real_mask]).mean(axis=0)
                if fake_mask.any():
                    self._mean_act_fake[sg] = np.abs(x_sg[fake_mask]).mean(axis=0)

        # ── Adjacency-matrix divergence (unchanged) ───────────────────────
        for sg_name in self.SUBGRAPH_NAMES:
            A_real = self._adjacencies.get(f'A_{sg_name}_real')
            A_fake = self._adjacencies.get(f'A_{sg_name}_fake')
            names = self._node_names.get(sg_name, [])
            if A_real is None or A_fake is None:
                continue

            n = min(A_real.shape[0], A_fake.shape[0])
            div = float(np.abs(A_real[:n, :n] - A_fake[:n, :n]).sum())
            results['subgraph_divergence'][sg_name] = div

            threshold = 0.01
            results['dag_sparsity'][sg_name] = {
                'real': float((np.abs(A_real[:n, :n]) > threshold).mean()),
                'fake': float((np.abs(A_fake[:n, :n]) > threshold).mean()),
            }

            div_edges = self._divergent_edges(A_real, A_fake, names, self.top_k)
            results['top_divergent_edges'][sg_name] = [
                {'src': e[0], 'tgt': e[1], 'divergence': e[2]} for e in div_edges
            ]

        self._results = results
        return results

    # ── visualization ──────────────────────────────────────────────────────

    def visualize(self, save_dir: str) -> None:
        if not self._adjacencies and not hasattr(self, '_r_diff'):
            return

        scm_dir = os.path.join(save_dir, 'scm')
        os.makedirs(scm_dir, exist_ok=True)

        # ── Per-sample r_diff_g views (Level 4) ───────────────────────────
        if hasattr(self, '_r_diff') and self._r_diff is not None:
            r_diff = self._r_diff
            labels = self._r_diff_labels
            real_mask = labels == 0
            fake_mask = labels == 1
            n_sg = r_diff.shape[1]
            sg_names = self.SUBGRAPH_NAMES[:n_sg]

            # Radar: mean r_diff magnitude per sub-graph, real vs fake
            real_means = (r_diff[real_mask].mean(axis=0)
                          if real_mask.any() else np.zeros(n_sg))
            fake_means = (r_diff[fake_mask].mean(axis=0)
                          if fake_mask.any() else np.zeros(n_sg))
            real_stds = (r_diff[real_mask].std(axis=0)
                         if real_mask.any() else np.zeros(n_sg))
            fake_stds = (r_diff[fake_mask].std(axis=0)
                         if fake_mask.any() else np.zeros(n_sg))
            radar_png = os.path.join(scm_dir, 'r_diff_radar.png')
            viz.plot_radar(
                {'Real': real_means.tolist(), 'Fake': fake_means.tolist()},
                sg_names,
                title='Causal sub-graph residuals — real vs fake',
                save_path=radar_png,
            )
            # ── Data exports for the radar ──────────────────────────────
            dump_csv(
                [
                    {
                        'sub_graph': sg_names[i],
                        'mean_r_diff_real': float(real_means[i]),
                        'mean_r_diff_fake': float(fake_means[i]),
                        'std_real': float(real_stds[i]),
                        'std_fake': float(fake_stds[i]),
                    }
                    for i in range(n_sg)
                ],
                ['sub_graph', 'mean_r_diff_real', 'mean_r_diff_fake',
                 'std_real', 'std_fake'],
                sibling_path(radar_png, '.csv'),
            )
            # NPZ — full per-sample r_diff with labels and method tags.
            r_diff_arrays: Dict[str, np.ndarray] = {
                'r_diff': r_diff.astype(float),       # (N, 4)
                'labels': labels.astype(int).reshape(-1),
                'sub_graph_names': np.asarray(sg_names),
            }
            if (self._method_labels is not None
                    and len(self._method_labels) == len(labels)):
                r_diff_arrays['method_labels'] = np.asarray(
                    self._method_labels).astype(int)
            dump_npz(
                r_diff_arrays,
                with_basename(radar_png, 'r_diff_per_sample.npz'),
            )
            dump_json(
                analyzer_metadata(
                    n_samples=int(len(labels)),
                    dataset_name=self.dataset_name,
                    sub_graph_order=sg_names,
                ),
                sibling_path(radar_png, '.json'),
            )

            # Per-sub-graph histograms (four panels via plot_histogram each)
            for i, sg in enumerate(sg_names):
                viz.plot_histogram(
                    {'Real': r_diff[real_mask, i] if real_mask.any() else np.array([]),
                     'Fake': r_diff[fake_mask, i] if fake_mask.any() else np.array([])},
                    title=f'r_diff distribution — {sg}',
                    xlabel='|r_fake − r_real| (mean over features)',
                    save_path=os.path.join(scm_dir, f'r_diff_hist_{sg}.png'),
                    balance=True,
                )

            # Per-method fingerprint radar (FF-DF / FF-F2F / FF-FS / FF-NT / ...)
            if (self._method_labels is not None
                    and len(self._method_labels) == len(labels)):
                methods = self._method_labels
                uniq = np.unique(methods)
                per_method = {}
                for m in uniq:
                    m_mask = methods == m
                    if m_mask.sum() < 5:
                        continue
                    name = f'method_{int(m)}'
                    per_method[name] = r_diff[m_mask].mean(axis=0).tolist()
                if per_method:
                    pm_png = os.path.join(scm_dir, 'r_diff_radar_per_method.png')
                    viz.plot_radar(
                        per_method, sg_names,
                        title='Per-method causal fingerprint (mean r_diff)',
                        save_path=pm_png,
                    )
                    # CSV: long-format per-method × per-sub-graph means.
                    pm_rows = []
                    for method_name, vals in per_method.items():
                        for sg_idx, v in enumerate(vals):
                            pm_rows.append({
                                'method': method_name,
                                'sub_graph': sg_names[sg_idx],
                                'mean_r_diff': float(v),
                            })
                    dump_csv(
                        pm_rows, ['method', 'sub_graph', 'mean_r_diff'],
                        sibling_path(pm_png, '.csv'),
                    )
                    dump_json(
                        analyzer_metadata(
                            n_samples=int(len(labels)),
                            dataset_name=self.dataset_name,
                            n_methods=int(len(per_method)),
                        ),
                        sibling_path(pm_png, '.json'),
                    )

            # Stacked-bar: dominant sub-graph frequency per class
            dominant = np.argmax(r_diff, axis=1)
            freq_real = np.zeros(n_sg)
            freq_fake = np.zeros(n_sg)
            if real_mask.any():
                vals, cnt = np.unique(dominant[real_mask], return_counts=True)
                freq_real[vals] = cnt / real_mask.sum()
            if fake_mask.any():
                vals, cnt = np.unique(dominant[fake_mask], return_counts=True)
                freq_fake[vals] = cnt / fake_mask.sum()
            dom_png = os.path.join(scm_dir, 'dominant_subgraph_frequency.png')
            viz.plot_grouped_bar(
                {'Real': freq_real.tolist(), 'Fake': freq_fake.tolist()},
                sg_names,
                title='Dominant sub-graph per sample (argmax r_diff)',
                save_path=dom_png,
                ylabel='Fraction of samples',
                rotate_labels=20,
            )
            # CSV — long-format class × sub-graph dominance.
            dom_rows = []
            for cls_name, freq in [('Real', freq_real), ('Fake', freq_fake)]:
                for i, sg in enumerate(sg_names):
                    dom_rows.append({
                        'class': cls_name,
                        'sub_graph': sg,
                        'dominance_fraction': float(freq[i]),
                    })
            dump_csv(
                dom_rows, ['class', 'sub_graph', 'dominance_fraction'],
                sibling_path(dom_png, '.csv'),
            )
            dump_json(
                analyzer_metadata(
                    n_samples=int(len(labels)),
                    dataset_name=self.dataset_name,
                    n_real=int(real_mask.sum()),
                    n_fake=int(fake_mask.sum()),
                ),
                sibling_path(dom_png, '.json'),
            )

        for sg_name in self.SUBGRAPH_NAMES:
            A_real = self._adjacencies.get(f'A_{sg_name}_real')
            A_fake = self._adjacencies.get(f'A_{sg_name}_fake')
            names = self._node_names.get(sg_name, [])
            if A_real is None or A_fake is None:
                continue

            n = min(A_real.shape[0], A_fake.shape[0], len(names))
            # Pass the FULL name list — the heatmap auto-crops inactive
            # rows/cols and trims labels to the active block, then
            # math-prettifies them. Keeps publication-ready output even
            # when the raw graph has 80+ nodes.
            label_names = names[:n]

            # Side-by-side A_real | A_fake | |A_real - A_fake| — the
            # slide-ready figure for Section 13 / Point 5 of the talk.
            diff = np.abs(A_real[:n, :n] - A_fake[:n, :n])
            sbs_png = os.path.join(scm_dir, f'{sg_name}_side_by_side.png')
            viz.plot_side_by_side_heatmap(
                {r'$A^{\mathrm{real}}$': A_real[:n, :n],
                 r'$A^{\mathrm{fake}}$': A_fake[:n, :n],
                 r'$|A^{\mathrm{real}}-A^{\mathrm{fake}}|$': diff},
                suptitle=f'Sub-graph: {sg_name}',
                save_path=sbs_png,
                row_labels=label_names, col_labels=label_names,
            )
            # ── Per-sub-graph data exports ─────────────────────────────
            dump_npz(
                {
                    'A_real': A_real[:n, :n].astype(float),
                    'A_fake': A_fake[:n, :n].astype(float),
                    'A_diff': diff.astype(float),
                    'node_names': np.asarray(label_names),
                },
                with_basename(sbs_png, f'{sg_name}_adjacency.npz'),
            )
            # Top-50 edges by |divergence|.
            flat = diff.flatten()
            top_idx = np.argsort(-flat)[:50]
            edge_rows = []
            for fi in top_idx:
                if flat[fi] <= 1e-9:
                    break
                src_i, dst_i = int(divmod(int(fi), n)[0]), int(divmod(int(fi), n)[1])
                edge_rows.append({
                    'src': label_names[src_i],
                    'dst': label_names[dst_i],
                    'A_real': float(A_real[src_i, dst_i]),
                    'A_fake': float(A_fake[src_i, dst_i]),
                    'divergence': float(flat[fi]),
                })
            dump_csv(
                edge_rows, ['src', 'dst', 'A_real', 'A_fake', 'divergence'],
                with_basename(sbs_png, f'{sg_name}_top_edges.csv'),
            )

            # Divergence-only heatmap (kept for back-compat; hot cmap
            # is easier to eyeball than the diverging one above).
            viz.plot_heatmap(
                diff,
                title=f'Edge Divergence: {sg_name}',
                save_path=os.path.join(scm_dir, f'{sg_name}_divergence.png'),
                row_labels=label_names, col_labels=label_names,
                cmap='hot', vmin=0,
            )

            # Per-edge scatter: A_real[i,j] vs A_fake[i,j]. Diagonal mass
            # = how similar the SCMs are; off-diagonal labelled outliers
            # = the edges that drive classification. K-independent
            # (shows ALL edges) so it's rendered once.
            viz.plot_edge_weight_scatter(
                A_real[:n, :n], A_fake[:n, :n],
                names[:n],
                title=f'{sg_name}',
                save_path=os.path.join(scm_dir, f'{sg_name}_edge_scatter.png'),
            )

            # K-dependent visuals: the same plots are rendered at multiple
            # top-K levels so the paper can compare "20 most divergent",
            # "30", etc. File names get a `_k<K>` tag.
            for K in self.top_k_levels:
                # Networkx graph induced on top-K divergent edges, with
                # real (blue) and fake (red) weights drawn as parallel
                # curved arrows — shows *how* the generator rewired the
                # causal structure.
                viz.plot_divergent_graph(
                    A_real[:n, :n], A_fake[:n, :n],
                    names[:n], K,
                    title=f'{sg_name} (top-{K})',
                    save_path=os.path.join(
                        scm_dir, f'{sg_name}_graph_k{K}.png'),
                )

                # Two panels (real | fake) sharing one layout, so the
                # reader sees edge presence/absence without parallel-edge
                # clutter.
                viz.plot_separated_class_graphs(
                    A_real[:n, :n], A_fake[:n, :n],
                    names[:n], K,
                    title=f'{sg_name} (top-{K})',
                    save_path=os.path.join(
                        scm_dir, f'{sg_name}_graph_k{K}.png'),
                )

                # Real-canonical / fake-partial framing:
                # - Real panel: all top-K edges of the real-trained SCM
                #   (canonical structure, ranked by activation-weighted
                #   importance |A_real| · ⟨|x|⟩_real).
                # - Fake panel: same edges; those with A_fake/A_real
                #   below `keep_ratio` are drawn faded/dashed, i.e.
                #   "missing in fake".
                # Also writes <base>_distinctive_edges.csv and
                # <base>_distinctive_nodes.csv so the figure can be
                # rebuilt in R / Cytoscape / Gephi.
                act_real = getattr(self, '_mean_act_real', {}).get(sg_name)
                act_fake = getattr(self, '_mean_act_fake', {}).get(sg_name)
                viz.plot_class_distinctive_graphs(
                    A_real[:n, :n], A_fake[:n, :n],
                    names[:n], K,
                    title=f'{sg_name} (top-{K})',
                    save_path=os.path.join(
                        scm_dir, f'{sg_name}_graph_k{K}.png'),
                    act_real=act_real,
                    act_fake=act_fake,
                    keep_ratio=getattr(self, 'distinctive_keep_ratio', 0.5),
                )

                # Paired horizontal bars for the top-K most divergent
                # edges — magnitude AND direction side-by-side.
                viz.plot_top_edge_comparison(
                    A_real[:n, :n], A_fake[:n, :n],
                    names[:n], K,
                    title=f'{sg_name} (top-{K})',
                    save_path=os.path.join(
                        scm_dir, f'{sg_name}_top_edges_bar_k{K}.png'),
                )

                # Top divergent edges (text table — lightweight, grep-friendly)
                div_edges = self._divergent_edges(A_real, A_fake, names, K)
                if div_edges:
                    txt_path = os.path.join(
                        scm_dir, f'{sg_name}_top_edges_k{K}.txt')
                    lines = [f'Top {K} Divergent Edges: {sg_name}',
                             '=' * 50, '',
                             f'{"Rank":>4}  {"Source":<30} {"Target":<30} {"Div":>8}',
                             '-' * 76]
                    for rank, (src, tgt, w) in enumerate(div_edges, 1):
                        lines.append(f'{rank:4d}  {src:<30} {tgt:<30} {w:8.4f}')
                    with open(txt_path, 'w') as f:
                        f.write('\n'.join(lines))

        # Sub-graph divergence bar chart
        if hasattr(self, '_results') and self._results.get('subgraph_divergence'):
            viz.plot_stacked_bar(
                self._results['subgraph_divergence'],
                title='Per-Subgraph Divergence L1(A_real, A_fake)',
                save_path=os.path.join(scm_dir, 'subgraph_divergence.png'),
                ylabel='L1 Divergence',
            )

    def explain_sample(self, idx: int) -> str:
        # Prefer the per-sample residual signal (Level 4) over the global
        # adjacency divergence — it says something specific about *this*
        # sample rather than the trained model.
        if hasattr(self, '_r_diff') and self._r_diff is not None \
                and idx < len(self._r_diff):
            row = self._r_diff[idx]
            sg_names = self.SUBGRAPH_NAMES[:len(row)]
            parts = [f"{sg}={row[i]:.3f}" for i, sg in enumerate(sg_names)]
            top = sg_names[int(np.argmax(row))]
            return (f"[SCM] r_diff: {', '.join(parts)} "
                    f"(dominant sub-graph: {top})")
        if not hasattr(self, '_results') or not self._results:
            return ''
        divs = self._results.get('subgraph_divergence', {})
        if not divs:
            return '[SCM] No adjacency matrices available'
        parts = [f"{sg}={d:.2f}" for sg, d in sorted(divs.items(), key=lambda x: -x[1])]
        top_sg = max(divs, key=divs.get)
        return f"[SCM] Sub-graph divergence: {', '.join(parts)} (highest: {top_sg})"

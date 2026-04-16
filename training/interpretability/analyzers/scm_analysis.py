"""
interpretability/analyzers/scm_analysis.py
==========================================
Level 4-SCM: Structural Causal Model analysis — adjacency matrix divergence,
top divergent edges with named nodes, per-subgraph divergence profiles,
DAG sparsity metrics.
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

logger = logging.getLogger(__name__)


class SCMAnalyzer(BaseAnalyzer):
    name = 'scm_analysis'
    required_keys = ()  # adjacency matrices come from model params, not preds

    def __init__(self, top_k: int = 20):
        super().__init__()
        self.top_k = top_k
        self._adjacencies: Dict[str, np.ndarray] = {}
        self._node_names: Dict[str, List[str]] = {}
        self._load_node_names()

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
        # Also pick up adjacency matrices returned in the pred dict
        for k, v in preds.items():
            if k.startswith('A_') and isinstance(v, (torch.Tensor, np.ndarray)):
                self._adjacencies[k] = self._to_numpy(v)
        if not self._labels:
            self._labels.append(self._to_numpy(labels))

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
        if not self._adjacencies:
            return {}

        results = {
            'num_adjacencies': len(self._adjacencies),
            'subgraph_divergence': {},
            'top_divergent_edges': {},
            'dag_sparsity': {},
        }

        for sg_name in ['identity', 'forensic_structural', 'forensic_noise', 'forensic_spectral']:
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
        if not self._adjacencies:
            return

        scm_dir = os.path.join(save_dir, 'scm')
        os.makedirs(scm_dir, exist_ok=True)

        for sg_name in ['identity', 'forensic_structural', 'forensic_noise', 'forensic_spectral']:
            A_real = self._adjacencies.get(f'A_{sg_name}_real')
            A_fake = self._adjacencies.get(f'A_{sg_name}_fake')
            names = self._node_names.get(sg_name, [])
            if A_real is None or A_fake is None:
                continue

            n = min(A_real.shape[0], A_fake.shape[0], len(names))
            label_names = names[:n] if n <= 40 else None

            # Divergence heatmap (|A_real - A_fake|)
            diff = np.abs(A_real[:n, :n] - A_fake[:n, :n])
            viz.plot_heatmap(
                diff,
                title=f'Edge Divergence: {sg_name}',
                save_path=os.path.join(scm_dir, f'{sg_name}_divergence.png'),
                row_labels=label_names, col_labels=label_names,
                cmap='hot', vmin=0,
            )

            # Top divergent edges (text table only — lightweight, grep-friendly)
            div_edges = self._divergent_edges(A_real, A_fake, names, self.top_k)
            if div_edges:
                txt_path = os.path.join(scm_dir, f'{sg_name}_top_edges.txt')
                lines = [f'Top Divergent Edges: {sg_name}',
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
        if not hasattr(self, '_results') or not self._results:
            return ''
        divs = self._results.get('subgraph_divergence', {})
        if not divs:
            return '[SCM] No adjacency matrices available'
        parts = [f"{sg}={d:.2f}" for sg, d in sorted(divs.items(), key=lambda x: -x[1])]
        top_sg = max(divs, key=divs.get)
        return f"[SCM] Sub-graph divergence: {', '.join(parts)} (highest: {top_sg})"

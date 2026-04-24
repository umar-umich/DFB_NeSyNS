"""
interpretability/visualization.py
=================================
Shared stateless plotting utilities for interpretability analyzers.
All functions use matplotlib Agg backend (headless) and close figures after save.
"""

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
#  Histograms
# ═══════════════════════════════════════════════════════════════════════════

def plot_histogram(
    data_dict: Dict[str, np.ndarray],
    title: str,
    xlabel: str,
    save_path: str,
    bins: int = 50,
    alpha: float = 0.6,
):
    """Overlaid histograms for multiple groups (e.g., real vs fake)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, values in data_dict.items():
        ax.hist(values, bins=bins, alpha=alpha, label=label, density=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  Reliability diagram (ECE calibration)
# ═══════════════════════════════════════════════════════════════════════════

def plot_reliability_diagram(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    bin_counts: np.ndarray,
    ece: float,
    save_path: str,
    n_bins: int = 10,
):
    """ECE calibration plot with gap bars."""
    fig, ax = plt.subplots(figsize=(7, 6))
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = 1.0 / n_bins * 0.8

    # Accuracy bars
    ax.bar(bin_centers, accuracies, width=width, alpha=0.7,
           color='steelblue', edgecolor='black', label='Accuracy')
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    # Gap bars
    gaps = np.abs(accuracies - confidences)
    ax.bar(bin_centers, gaps, bottom=np.minimum(accuracies, confidences),
           width=width, alpha=0.3, color='red', label='Gap')

    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Reliability Diagram (ECE = {ece:.4f})')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  Stacked bar chart (evidence decomposition)
# ═══════════════════════════════════════════════════════════════════════════

def plot_stacked_bar(
    components: Dict[str, float],
    title: str,
    save_path: str,
    ylabel: str = 'Mean Evidence',
):
    """Single stacked bar showing component contributions."""
    fig, ax = plt.subplots(figsize=(6, 5))
    names = list(components.keys())
    values = [components[n] for n in names]
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

    bottom = 0.0
    for name, val, color in zip(names, values, colors):
        ax.bar('Evidence', val, bottom=bottom, color=color,
               label=f'{name}: {val:.2f}', edgecolor='white')
        bottom += val

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_evidence_by_class(
    class_components: Dict[str, Dict[str, float]],
    title: str,
    save_path: str,
):
    """Side-by-side stacked bars for real vs fake evidence decomposition."""
    fig, ax = plt.subplots(figsize=(8, 5))
    class_names = list(class_components.keys())
    branch_names = list(class_components[class_names[0]].keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(branch_names)))

    x = np.arange(len(class_names))
    width = 0.5

    for cls_idx, cls_name in enumerate(class_names):
        bottom = 0.0
        for br_idx, br_name in enumerate(branch_names):
            val = class_components[cls_name][br_name]
            label = br_name if cls_idx == 0 else None
            ax.bar(x[cls_idx], val, width, bottom=bottom,
                   color=colors[br_idx], label=label, edgecolor='white')
            bottom += val

    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylabel('Mean Evidence')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  Radar chart (forensic anomaly groups)
# ═══════════════════════════════════════════════════════════════════════════

def plot_radar(
    data_dict: Dict[str, Sequence[float]],
    category_names: List[str],
    title: str,
    save_path: str,
):
    """Radar/spider chart for multi-category comparisons."""
    n = len(category_names)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_dict)))

    for (label, values), color in zip(data_dict.items(), colors):
        vals = list(values) + [values[0]]
        ax.plot(angles, vals, 'o-', label=label, color=color, linewidth=2)
        ax.fill(angles, vals, alpha=0.15, color=color)

    ax.set_thetagrids(np.degrees(angles[:-1]), category_names)
    ax.set_title(title, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  Grouped bar chart (rule firing rates)
# ═══════════════════════════════════════════════════════════════════════════

def plot_grouped_bar(
    group_data: Dict[str, List[float]],
    category_names: List[str],
    title: str,
    save_path: str,
    ylabel: str = 'Mean Score',
    rotate_labels: int = 45,
):
    """Grouped bar chart (e.g., real vs fake per rule)."""
    n_cats = len(category_names)
    n_groups = len(group_data)
    x = np.arange(n_cats)
    width = 0.8 / n_groups
    colors = plt.cm.Set1(np.linspace(0, 1, n_groups))

    fig, ax = plt.subplots(figsize=(max(10, n_cats * 0.6), 6))
    for i, (label, values) in enumerate(group_data.items()):
        offset = (i - n_groups / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=label,
               color=colors[i], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(category_names, rotation=rotate_labels, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  Heatmap
# ═══════════════════════════════════════════════════════════════════════════

def plot_side_by_side_heatmap(
    matrices: Dict[str, np.ndarray],
    suptitle: str,
    save_path: str,
    row_labels: Optional[List[str]] = None,
    col_labels: Optional[List[str]] = None,
    cmap: str = 'RdBu_r',
    symmetric: bool = True,
):
    """Render N adjacency matrices side-by-side with a shared colour scale.

    Used for `A_real | A_fake` (and optionally `|A_real - A_fake|`)
    figures in the paper. Shared colour scale makes the two panels
    directly comparable by eye.
    """
    names = list(matrices.keys())
    arrs = [np.asarray(matrices[n]) for n in names]
    n = len(arrs)
    h, w = arrs[0].shape
    cell = 0.18 if max(h, w) > 40 else 0.30
    fig, axes = plt.subplots(
        1, n, figsize=(max(5, w * cell) * n + 1.5, max(5, h * cell)),
        squeeze=False,
    )
    axes = axes[0]

    vmax = max(np.abs(a).max() for a in arrs)
    vmin = -vmax if symmetric else 0.0

    im = None
    for ax, name, A in zip(axes, names, arrs):
        im = ax.imshow(A, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_title(name, fontsize=12)
        if row_labels is not None and len(row_labels) <= 40:
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels, fontsize=6)
        else:
            ax.set_yticks([])
        if col_labels is not None and len(col_labels) <= 40:
            ax.set_xticks(range(len(col_labels)))
            ax.set_xticklabels(col_labels, rotation=90, fontsize=6)
        else:
            ax.set_xticks([])

    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.8)
    fig.suptitle(suptitle, fontsize=13)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_divergent_graph(
    A_real: np.ndarray,
    A_fake: np.ndarray,
    node_names: List[str],
    top_k: int,
    title: str,
    save_path: str,
):
    """Draw a directed graph induced on the top-K |A_real - A_fake| edges.

    Edges are drawn twice — once with real weight (blue), once with fake
    weight (red) — so we can see *how* the generator rewired the causal
    structure. Width ∝ |weight|. Requires networkx.
    """
    try:
        import networkx as nx
    except ImportError:
        return

    n = min(A_real.shape[0], A_fake.shape[0], len(node_names))
    if n == 0:
        return
    diff = np.abs(A_real[:n, :n] - A_fake[:n, :n])
    flat_idx = np.argsort(-diff.flatten())[:top_k]
    edges = []
    nodes = set()
    for fi in flat_idx:
        if diff.flatten()[fi] <= 1e-6:
            break
        src_i, tgt_i = divmod(int(fi), n)
        src, tgt = node_names[src_i], node_names[tgt_i]
        edges.append((src, tgt, float(A_real[src_i, tgt_i]),
                      float(A_fake[src_i, tgt_i])))
        nodes.update([src, tgt])
    if not edges:
        return

    G = nx.DiGraph()
    G.add_nodes_from(sorted(nodes))
    pos = nx.spring_layout(G, seed=7, k=1.3 / max(1.0, np.sqrt(len(nodes))))

    fig, ax = plt.subplots(figsize=(11, 8))
    # Nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='#E8E8E8',
                           edgecolors='#333', node_size=850)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

    # Scale edge widths by |weight| relative to the max seen in this set
    max_w = max(max(abs(er), abs(ef)) for _, _, er, ef in edges) or 1.0

    def _curved(x, y, real_side=True):
        # Offset so real and fake arrows don't overlap
        return f"arc3,rad={'0.1' if real_side else '-0.1'}"

    for src, tgt, wr, wf in edges:
        if abs(wr) > 1e-6:
            nx.draw_networkx_edges(
                G, pos, ax=ax, edgelist=[(src, tgt)],
                edge_color='#1f77b4',   # blue = real
                width=0.8 + 3.5 * abs(wr) / max_w,
                alpha=0.85, arrows=True, arrowsize=11,
                connectionstyle=_curved(None, None, True),
            )
        if abs(wf) > 1e-6:
            nx.draw_networkx_edges(
                G, pos, ax=ax, edgelist=[(src, tgt)],
                edge_color='#d62728',   # red = fake
                width=0.8 + 3.5 * abs(wf) / max_w,
                alpha=0.85, arrows=True, arrowsize=11,
                connectionstyle=_curved(None, None, False),
            )

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color='#1f77b4', lw=2.5, label='A_real weight'),
        Line2D([0], [0], color='#d62728', lw=2.5, label='A_fake weight'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=9)
    ax.set_title(f'{title}  (top-{len(edges)} divergent edges)',
                 fontsize=12)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_heatmap(
    matrix: np.ndarray,
    title: str,
    save_path: str,
    row_labels: Optional[List[str]] = None,
    col_labels: Optional[List[str]] = None,
    cmap: str = 'RdBu_r',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Optional[Tuple[int, int]] = None,
):
    """Generic heatmap with optional labels."""
    h, w = matrix.shape
    if figsize is None:
        figsize = (max(6, w * 0.4), max(5, h * 0.4))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax)
    if row_labels is not None and len(row_labels) <= 40:
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=7)
    if col_labels is not None and len(col_labels) <= 40:
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=90, fontsize=7)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  Scatter plot
# ═══════════════════════════════════════════════════════════════════════════

def plot_scatter(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: str,
    hue: Optional[np.ndarray] = None,
    hue_labels: Optional[Dict[int, str]] = None,
    alpha: float = 0.4,
    s: float = 10,
):
    """Scatter plot with optional class coloring."""
    fig, ax = plt.subplots(figsize=(8, 6))
    if hue is not None:
        unique_vals = np.unique(hue)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_vals)))
        for val, color in zip(unique_vals, colors):
            mask = hue == val
            label = hue_labels[val] if hue_labels else str(val)
            ax.scatter(x[mask], y[mask], c=[color], alpha=alpha, s=s, label=label)
        ax.legend()
    else:
        ax.scatter(x, y, alpha=alpha, s=s)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  Edge table (for SCM adjacency analysis)
# ═══════════════════════════════════════════════════════════════════════════

def save_edge_table(
    edges: List[Tuple[str, str, float]],
    title: str,
    save_path: str,
    max_rows: int = 30,
):
    """Save a ranked edge table as PNG and text file."""
    edges = edges[:max_rows]
    txt_path = save_path.replace('.png', '.txt')

    # Text file
    lines = [title, '=' * len(title), '']
    lines.append(f'{"Rank":>4}  {"Source":<30} {"Target":<30} {"Weight":>8}')
    lines.append('-' * 76)
    for i, (src, tgt, w) in enumerate(edges, 1):
        lines.append(f'{i:4d}  {src:<30} {tgt:<30} {w:8.4f}')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines))

    # PNG table
    fig, ax = plt.subplots(figsize=(12, max(3, 0.3 * len(edges) + 1)))
    ax.axis('off')
    table_data = [[str(i + 1), src, tgt, f'{w:.4f}']
                  for i, (src, tgt, w) in enumerate(edges)]
    table = ax.table(
        cellText=table_data,
        colLabels=['Rank', 'Source', 'Target', 'Weight'],
        loc='center',
        cellLoc='left',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)
    ax.set_title(title, fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  JSON / text report
# ═══════════════════════════════════════════════════════════════════════════

def save_json_summary(results: dict, save_path: str):
    """Write results dict to JSON, converting numpy types."""
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        return obj

    with open(save_path, 'w') as f:
        json.dump(_convert(results), f, indent=2)


def save_text_report(lines: List[str], save_path: str):
    """Write a plain-text report."""
    with open(save_path, 'w') as f:
        f.write('\n'.join(lines))


# ═══════════════════════════════════════════════════════════════════════════
#  Multi-dataset mosaic (paper fig1)
# ═══════════════════════════════════════════════════════════════════════════

def build_results_mosaic(
    per_dataset_dirs: Dict[str, str],
    save_path: str,
    tile_names: Optional[List[str]] = None,
):
    """Stitch pre-rendered interpretability PNGs into a single paper
    figure.

    For each dataset and each tile in *tile_names* (relative path inside
    that dataset's interpretability folder), drop the image into a grid
    cell. Missing tiles render as empty white panels so the figure stays
    aligned.
    """
    if not per_dataset_dirs:
        return
    if tile_names is None:
        tile_names = [
            'reliability_diagram.png',
            'branch_evidence.png',
            'scm/r_diff_radar.png',
            'risk_coverage_curve.png',
        ]

    n_rows = len(per_dataset_dirs)
    n_cols = len(tile_names)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.8 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )
    for r, (ds_name, ds_dir) in enumerate(per_dataset_dirs.items()):
        for c, tile in enumerate(tile_names):
            ax = axes[r][c]
            tile_path = os.path.join(ds_dir, tile)
            ax.axis('off')
            if os.path.exists(tile_path):
                try:
                    img = plt.imread(tile_path)
                    ax.imshow(img)
                except Exception:
                    pass
            if r == 0:
                ax.set_title(tile.replace('/', ' / '), fontsize=10, pad=6)
            if c == 0:
                ax.text(-0.05, 0.5, ds_name, rotation=90, fontsize=11,
                        transform=ax.transAxes, va='center', ha='right')
    fig.suptitle('NeSy-DeFake interpretability summary', fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)

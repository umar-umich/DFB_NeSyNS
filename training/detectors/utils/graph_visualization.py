"""
Graph visualization utilities for NeSy causal discovery module.

Saves PNG visualizations of the learned causal graphs at the end of training
epochs. Handles 339-node graphs via category-level aggregation.

All functions use matplotlib Agg backend (no display), close figures after
saving, and handle all-zero adjacency matrices gracefully.
"""

import os
import logging
from typing import Dict, List, Optional

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Semantic category structure for 211 FaceBench attributes
# ---------------------------------------------------------------------------

SEMANTIC_CATEGORIES = {
    'Hair':             (0, 20),
    'Forehead':         (20, 23),
    'Eyebrows':         (23, 30),
    'Eyes':             (30, 45),
    'Eyelashes':        (45, 48),
    'Nose':             (48, 56),
    'Mouth_Lips':       (56, 66),
    'Cheeks':           (66, 70),
    'Chin_Jaw':         (70, 76),
    'Face_Shape':       (76, 82),
    'Ears':             (82, 85),
    'Skin':             (85, 100),
    'Facial_Hair':      (100, 106),
    'Neck':             (106, 108),
    'Age':              (108, 113),
    'Other_Appearance': (113, 116),
    'Accessories':      (116, 146),
    'Makeup':           (146, 159),
    'Surrounding':      (159, 171),
    'Expression':       (171, 179),
    'Action_Units':     (179, 204),
    'Identity':         (204, 211),
}


def _to_numpy(t) -> np.ndarray:
    """Convert tensor or array to numpy."""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().float().numpy()
    return np.asarray(t, dtype=np.float32)


def _aggregate_to_categories(A: np.ndarray, z_dim: int,
                             categories: Dict[str, tuple]) -> tuple:
    """
    Aggregate a (d, d) adjacency matrix into (n_cat+1, n_cat+1) category-level.

    Returns (agg_matrix, category_names) where the first entry is 'Latent'
    (aggregating z_dim latent features) followed by semantic categories.
    """
    cat_names = ['Latent']
    cat_ranges = [(0, z_dim)]
    for name, (start, end) in categories.items():
        cat_names.append(name)
        cat_ranges.append((z_dim + start, z_dim + end))

    n = len(cat_ranges)
    agg = np.zeros((n, n), dtype=np.float32)
    for i, (si, ei) in enumerate(cat_ranges):
        for j, (sj, ej) in enumerate(cat_ranges):
            block = A[si:ei, sj:ej]
            if block.size > 0:
                agg[i, j] = block.mean()
    return agg, cat_names


def _aggregate_semantic_categories(A: np.ndarray,
                                   categories: Dict[str, tuple]) -> tuple:
    """
    Aggregate a (s_dim, s_dim) semantic-only adjacency into (n_cat, n_cat).
    """
    cat_names = []
    cat_ranges = []
    for name, (start, end) in categories.items():
        cat_names.append(name)
        cat_ranges.append((start, end))

    n = len(cat_ranges)
    agg = np.zeros((n, n), dtype=np.float32)
    for i, (si, ei) in enumerate(cat_ranges):
        for j, (sj, ej) in enumerate(cat_ranges):
            block = A[si:ei, sj:ej]
            if block.size > 0:
                agg[i, j] = block.mean()
    return agg, cat_names


# ---------------------------------------------------------------------------
# Visualization functions
# ---------------------------------------------------------------------------

def save_category_heatmap(A_real: np.ndarray, A_fake: np.ndarray,
                          z_dim: int, categories: Dict[str, tuple],
                          save_path: str) -> None:
    """
    Three-panel heatmap: A_real, A_fake, divergence (A_real - A_fake).
    Aggregated from (d,d) to (n_categories+1, n_categories+1).
    """
    agg_real, cat_names = _aggregate_to_categories(A_real, z_dim, categories)
    agg_fake, _ = _aggregate_to_categories(A_fake, z_dim, categories)
    divergence = agg_real - agg_fake

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    vmax = max(agg_real.max(), agg_fake.max(), 1e-6)

    for ax, data, title, cmap in [
        (axes[0], agg_real, 'Real Graph', 'Blues'),
        (axes[1], agg_fake, 'Fake Graph', 'Reds'),
        (axes[2], divergence, 'Divergence (Real - Fake)', 'RdBu_r'),
    ]:
        if title == 'Divergence (Real - Fake)':
            vm = max(abs(divergence.min()), abs(divergence.max()), 1e-6)
            im = ax.imshow(data, cmap=cmap, vmin=-vm, vmax=vm, aspect='auto')
        else:
            im = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax, aspect='auto')

        ax.set_xticks(range(len(cat_names)))
        ax.set_yticks(range(len(cat_names)))
        ax.set_xticklabels(cat_names, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(cat_names, fontsize=7)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Effect', fontsize=9)
        ax.set_ylabel('Cause', fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_top_k_edges(A: np.ndarray, node_names: List[str], k: int,
                     save_path: str, title: str = 'Top-K Strongest Edges') -> None:
    """
    Save table of top-K strongest directed edges with named source/target.
    Saves both as PNG (matplotlib table) and .txt file.
    """
    d = A.shape[0]
    # Flatten and get top-k indices
    flat = A.flatten()
    if flat.max() < 1e-8:
        # All-zero adjacency — nothing to show
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, 'No edges learned (all-zero adjacency)',
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return

    top_k_idx = np.argsort(flat)[::-1][:k]
    rows, cols = np.unravel_index(top_k_idx, (d, d))

    # Build table data
    table_data = []
    for rank, (i, j) in enumerate(zip(rows, cols)):
        src = node_names[i] if i < len(node_names) else f'node_{i}'
        tgt = node_names[j] if j < len(node_names) else f'node_{j}'
        weight = A[i, j]
        if weight < 1e-8:
            break
        table_data.append([rank + 1, src, tgt, f'{weight:.4f}'])

    if not table_data:
        return

    # Save as text
    txt_path = save_path.rsplit('.', 1)[0] + '.txt'
    with open(txt_path, 'w') as f:
        f.write(f"{'Rank':>4}  {'Source':<30}  {'Target':<30}  {'Weight':>8}\n")
        f.write('-' * 78 + '\n')
        for row in table_data:
            f.write(f"{row[0]:>4}  {row[1]:<30}  {row[2]:<30}  {row[3]:>8}\n")

    # Save as PNG table
    fig, ax = plt.subplots(figsize=(10, max(2, 0.4 * len(table_data) + 1)))
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    table = ax.table(
        cellText=table_data,
        colLabels=['Rank', 'Source', 'Target', 'Weight'],
        loc='center',
        cellLoc='left',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_semantic_subgraph(A_sem_real: np.ndarray, A_sem_fake: np.ndarray,
                           attr_names: List[str],
                           categories: Dict[str, tuple],
                           save_path: str) -> None:
    """
    Category-level 22x22 heatmap for the 211-node intra-semantic graph.
    Three panels: real, fake, divergence.
    """
    agg_real, cat_names = _aggregate_semantic_categories(A_sem_real, categories)
    agg_fake, _ = _aggregate_semantic_categories(A_sem_fake, categories)
    divergence = agg_real - agg_fake

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    vmax = max(agg_real.max(), agg_fake.max(), 1e-6)

    for ax, data, title, cmap in [
        (axes[0], agg_real, 'Semantic Real', 'Blues'),
        (axes[1], agg_fake, 'Semantic Fake', 'Reds'),
        (axes[2], divergence, 'Semantic Divergence', 'RdBu_r'),
    ]:
        if 'Divergence' in title:
            vm = max(abs(divergence.min()), abs(divergence.max()), 1e-6)
            im = ax.imshow(data, cmap=cmap, vmin=-vm, vmax=vm, aspect='auto')
        else:
            im = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax, aspect='auto')

        ax.set_xticks(range(len(cat_names)))
        ax.set_yticks(range(len(cat_names)))
        ax.set_xticklabels(cat_names, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(cat_names, fontsize=7)
        ax.set_title(title, fontsize=11, fontweight='bold')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle('Intra-Semantic Causal Graph (211 FaceBench Attributes)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_divergence_analysis(A_real: np.ndarray, A_fake: np.ndarray,
                             z_dim: int, attr_names: List[str],
                             categories: Dict[str, tuple],
                             save_path: str) -> None:
    """
    Bar chart: which categories have most broken/created edges.
    Plus top-10 broken and top-10 created edges with named nodes.
    """
    divergence = A_real - A_fake  # positive = broken by fakes
    d = A_real.shape[0]

    # Build node names: latent + semantic
    node_names = [f'z_{i}' for i in range(z_dim)]
    if len(attr_names) > 0:
        node_names.extend(attr_names)
    else:
        node_names.extend([f's_{i}' for i in range(d - z_dim)])

    # Category-level aggregation of absolute divergence
    cat_names = ['Latent']
    cat_ranges = [(0, z_dim)]
    for name, (start, end) in categories.items():
        cat_names.append(name)
        cat_ranges.append((z_dim + start, z_dim + end))

    # Per-category: sum of broken and created edge strengths
    broken_per_cat = []
    created_per_cat = []
    for si, ei in cat_ranges:
        # Edges FROM this category
        block_out = divergence[si:ei, :]
        broken_per_cat.append(block_out.clip(min=0).sum())
        created_per_cat.append((-block_out).clip(min=0).sum())

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # Panel 1: broken vs created per category
    x = np.arange(len(cat_names))
    w = 0.35
    axes[0].bar(x - w/2, broken_per_cat, w, label='Broken by fakes', color='tab:red', alpha=0.7)
    axes[0].bar(x + w/2, created_per_cat, w, label='Created by fakes', color='tab:blue', alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cat_names, rotation=45, ha='right', fontsize=7)
    axes[0].set_ylabel('Total edge weight')
    axes[0].set_title('Edge Changes by Category', fontweight='bold')
    axes[0].legend(fontsize=8)

    # Panel 2: Top-10 broken edges
    flat_div = divergence.flatten()
    top_broken = np.argsort(flat_div)[::-1][:10]
    broken_data = []
    for idx in top_broken:
        i, j = np.unravel_index(idx, (d, d))
        if flat_div[idx] < 1e-8:
            break
        src = node_names[i] if i < len(node_names) else f'n_{i}'
        tgt = node_names[j] if j < len(node_names) else f'n_{j}'
        broken_data.append(f'{src} -> {tgt}: {flat_div[idx]:.4f}')

    axes[1].axis('off')
    axes[1].set_title('Top-10 Broken Edges', fontweight='bold')
    text = '\n'.join(broken_data) if broken_data else 'No broken edges'
    axes[1].text(0.05, 0.95, text, transform=axes[1].transAxes,
                 fontsize=8, verticalalignment='top', fontfamily='monospace')

    # Panel 3: Top-10 created edges
    top_created = np.argsort(flat_div)[:10]
    created_data = []
    for idx in top_created:
        i, j = np.unravel_index(idx, (d, d))
        if flat_div[idx] > -1e-8:
            break
        src = node_names[i] if i < len(node_names) else f'n_{i}'
        tgt = node_names[j] if j < len(node_names) else f'n_{j}'
        created_data.append(f'{src} -> {tgt}: {-flat_div[idx]:.4f}')

    axes[2].axis('off')
    axes[2].set_title('Top-10 Created Edges', fontweight='bold')
    text = '\n'.join(created_data) if created_data else 'No created edges'
    axes[2].text(0.05, 0.95, text, transform=axes[2].transAxes,
                 fontsize=8, verticalalignment='top', fontfamily='monospace')

    plt.suptitle('Causal Graph Divergence Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def save_all_causal_graphs(causal_module, log_dir: str, epoch: int,
                           top_k: int = 20) -> None:
    """
    Save all causal graph visualizations to {log_dir}/graphs/epoch_{epoch}/.

    Args:
        causal_module: CausalDiscoveryModule instance
        log_dir: base logging directory
        epoch: current epoch number
        top_k: number of top edges to show
    """
    save_dir = os.path.join(log_dir, 'graphs', f'epoch_{epoch}')
    os.makedirs(save_dir, exist_ok=True)

    categories = SEMANTIC_CATEGORIES
    z_spatial_dim = causal_module.z_spatial_dim
    z_freq_dim = causal_module.z_freq_dim

    try:
        # -- Branch graphs (spatial, frequency) ---
        for branch, z_dim in [('spatial', z_spatial_dim), ('freq', z_freq_dim)]:
            pair = causal_module.causal_spatial if branch == 'spatial' else causal_module.causal_freq
            A_real = _to_numpy(pair.causal_learner_real._A_dce_ema)
            A_fake = _to_numpy(pair.causal_learner_fake._A_dce_ema)

            node_names = causal_module.get_node_names(
                'spatial' if branch == 'spatial' else 'frequency')

            # Category heatmap
            save_category_heatmap(
                A_real, A_fake, z_dim, categories,
                os.path.join(save_dir, f'{branch}_category_heatmap.png'))

            # Top-K edges for real and fake
            save_top_k_edges(
                A_real, node_names, top_k,
                os.path.join(save_dir, f'{branch}_real_top{top_k}.png'),
                title=f'{branch.title()} Real: Top-{top_k} Edges')
            save_top_k_edges(
                A_fake, node_names, top_k,
                os.path.join(save_dir, f'{branch}_fake_top{top_k}.png'),
                title=f'{branch.title()} Fake: Top-{top_k} Edges')

            # Divergence analysis
            attr_names = causal_module.get_semantic_node_names()
            save_divergence_analysis(
                A_real, A_fake, z_dim, attr_names, categories,
                os.path.join(save_dir, f'{branch}_divergence_analysis.png'))

        # -- Semantic graph (Part A, if enabled) ---
        if causal_module.use_semantic_graph and causal_module.causal_semantic is not None:
            sem_pair = causal_module.causal_semantic
            A_sem_real = _to_numpy(sem_pair.causal_learner_real._A_dce_ema)
            A_sem_fake = _to_numpy(sem_pair.causal_learner_fake._A_dce_ema)
            attr_names = causal_module.get_semantic_node_names()

            save_semantic_subgraph(
                A_sem_real, A_sem_fake, attr_names, categories,
                os.path.join(save_dir, 'semantic_graph_heatmap.png'))

            save_top_k_edges(
                A_sem_real, attr_names, top_k,
                os.path.join(save_dir, f'semantic_real_top{top_k}.png'),
                title=f'Semantic Real: Top-{top_k} Edges')
            save_top_k_edges(
                A_sem_fake, attr_names, top_k,
                os.path.join(save_dir, f'semantic_fake_top{top_k}.png'),
                title=f'Semantic Fake: Top-{top_k} Edges')

        logger.info(f"[GraphViz] Saved causal graphs to {save_dir}")

    except Exception as e:
        logger.warning(f"[GraphViz] Failed to save graphs at epoch {epoch}: {e}")

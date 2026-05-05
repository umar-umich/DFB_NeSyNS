"""
interpretability/visualization.py
=================================
Shared stateless plotting utilities for interpretability analyzers.
All functions use matplotlib Agg backend (headless) and close figures after save.
"""

import csv
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

try:
    from .pretty_names import pretty, pretty_many, group_of, GROUP_COLORS, GROUP_LONGNAMES
except ImportError:  # allow flat-import (e.g. unit tests)
    from interpretability.pretty_names import (  # type: ignore
        pretty, pretty_many, group_of, GROUP_COLORS, GROUP_LONGNAMES)


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
    balance: bool = False,
    balance_seed: int = 0,
):
    """Overlaid histograms for multiple groups (e.g., real vs fake).

    When ``balance=True`` every group is sub-sampled (without replacement,
    fixed seed) to ``min(group sizes)`` so the distributions are directly
    comparable without one group dominating by sheer count.  The balanced
    sample size is appended to the title.
    """
    arrays = {k: np.asarray(v) for k, v in data_dict.items()}
    if balance and len(arrays) > 1:
        n_bal = min(a.size for a in arrays.values())
        rng = np.random.default_rng(balance_seed)
        arrays = {
            k: rng.choice(a, size=n_bal, replace=False)
            for k, a in arrays.items()
        }
        title = f'{title}  (n={n_bal:,} per group, balanced)'

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, values in arrays.items():
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
    """ECE calibration plot with gap bars (legacy single-panel API).

    Kept for back-compat. New callers should prefer
    ``plot_reliability_diagram_v2`` which produces the two-panel composite
    used in the paper.
    """
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


def plot_reliability_diagram_v2(
    *,
    confidences_per_sample: np.ndarray,
    accuracies_per_sample: np.ndarray,
    bin_edges_main: np.ndarray,
    bin_centers_main: np.ndarray,
    bin_confidence_main: np.ndarray,
    bin_accuracy_main: np.ndarray,
    bin_count_main: np.ndarray,
    ece_equal_width: float,
    ece_equal_mass: float,
    ace: float,
    main_scheme_label: str,
    save_path: str,
    xlim: Optional[Tuple[float, float]] = None,
):
    """Two-panel reliability composite (Goal 3, 2026-05-04).

    Top:    reliability bars over the main binning scheme; gap shading;
            faint per-bin sample-count overlay on a secondary axis.
    Bottom: predicted-class confidence histogram (same x-axis), so the
            reader can see where predictions actually concentrate.

    Title carries:  ECE (eq-width, 10 bins) | ECE (eq-mass, 15 bins) | ACE.
    """
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7.5, 7.0), sharex=True,
        gridspec_kw={'height_ratios': [3, 1.2], 'hspace': 0.10},
    )
    widths = np.diff(bin_edges_main)
    # ── Top panel: reliability bars ──────────────────────────────────
    ax_top.bar(bin_centers_main, bin_accuracy_main, width=widths * 0.9,
               alpha=0.8, color='steelblue', edgecolor='black',
               label='Bin accuracy')
    gaps = np.abs(bin_accuracy_main - bin_confidence_main)
    ax_top.bar(
        bin_centers_main, gaps,
        bottom=np.minimum(bin_accuracy_main, bin_confidence_main),
        width=widths * 0.9, alpha=0.30, color='red', label='Gap')
    ax_top.plot([0, 1], [0, 1], 'k--', lw=1.0, label='Perfect calibration')
    ax_top.set_ylabel('Accuracy')
    ax_top.set_ylim(0, 1)
    ax_top.legend(loc='upper left', fontsize=9, frameon=True)

    # Faint per-bin count overlay on secondary axis
    if bin_count_main.sum() > 0:
        ax_top_r = ax_top.twinx()
        ax_top_r.bar(bin_centers_main, bin_count_main,
                     width=widths * 0.9,
                     color='#888', alpha=0.15, edgecolor='none',
                     label='bin count')
        ax_top_r.set_ylabel('# samples in bin', fontsize=9, color='#666')
        ax_top_r.tick_params(axis='y', labelsize=8, colors='#666')

    title = (f'Reliability — {main_scheme_label} | '
             f'ECE (eq-width, 10) = {ece_equal_width:.4f}  |  '
             f'ECE (eq-mass, 15) = {ece_equal_mass:.4f}  |  '
             f'ACE = {ace:.4f}')
    ax_top.set_title(title, fontsize=10, pad=8)

    # ── Bottom panel: confidence histogram of all samples ──────────────
    if confidences_per_sample.size:
        ax_bot.hist(confidences_per_sample, bins=40,
                    color='#1f77b4', alpha=0.7, edgecolor='white',
                    linewidth=0.4)
    ax_bot.set_ylabel('# preds', fontsize=9)
    ax_bot.set_xlabel('Predicted-class confidence  $\\max(p, 1-p)$',
                      fontsize=10)

    # Axis cropping (configurable upstream via the `xlim` arg)
    if xlim is None:
        xlim = (0.0, 1.0)
    ax_top.set_xlim(*xlim)
    ax_bot.set_xlim(*xlim)

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
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
    """Single stacked bar showing component contributions.

    Compact paper-friendly layout: narrow bar, in-segment labels, and a
    total annotation above the bar.
    """
    fig, ax = plt.subplots(figsize=(3.6, 4.5))
    names = list(components.keys())
    values = [components[n] for n in names]
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

    bottom = 0.0
    total = float(sum(values))
    for name, val, color in zip(names, values, colors):
        ax.bar('Evidence', val, width=0.42, bottom=bottom, color=color,
               label=f'{name}: {val:.2f}', edgecolor='white', linewidth=0.8)
        # Label inside each segment if it's tall enough
        if total > 0 and val / total > 0.04:
            ax.text(0, bottom + val / 2, f'{val:.2f}',
                    ha='center', va='center', fontsize=8.5,
                    color='white', fontweight='bold')
        bottom += val

    # Total at the top of the stack
    ax.text(0, total + 0.02 * max(total, 1e-6),
            f'Σ = {total:.2f}', ha='center', va='bottom',
            fontsize=9, color='#222', fontweight='bold')

    ax.set_ylim(0, total + 0.12 * max(total, 1e-6))
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
              fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_evidence_by_class(
    class_components: Dict[str, Dict[str, float]],
    title: str,
    save_path: str,
    proportional: bool = True,
):
    """Side-by-side stacked bars for real-vs-fake evidence decomposition.

    By default each bar is normalised to 100 % so the *composition* of
    evidence (which branch dominates in each class) is visible — this
    avoids the absolute-magnitude trap where one class' bar dwarfs the
    other and you can't see the mix any more.

    Set ``proportional=False`` to recover the original absolute-mean view.
    """
    fig, axes = plt.subplots(
        1, 2 if proportional else 1,
        figsize=(7.5 if proportional else 4.0, 4.5),
        squeeze=False)
    axes = axes[0]
    class_names = list(class_components.keys())
    branch_names = list(class_components[class_names[0]].keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(branch_names)))
    x = np.arange(len(class_names))
    width = 0.42  # narrower bars for paper layout

    def _stacked(ax, normalize: bool):
        for cls_idx, cls_name in enumerate(class_names):
            comp = class_components[cls_name]
            raw_total = sum(comp.values())
            total = raw_total if normalize else 1.0
            total = total if total > 0 else 1.0
            bottom = 0.0
            for br_idx, br_name in enumerate(branch_names):
                val = comp[br_name] / total
                label = br_name if cls_idx == 0 else None
                ax.bar(x[cls_idx], val, width, bottom=bottom,
                       color=colors[br_idx], label=label,
                       edgecolor='white', linewidth=1.0)
                # In-segment numeric annotation if the slice is fat enough
                if (normalize and val > 0.05) or (not normalize and val > 0.5):
                    ax.text(x[cls_idx], bottom + val / 2,
                            f'{val*100:.0f}%' if normalize else f'{val:.1f}',
                            ha='center', va='center', fontsize=8.5,
                            color='white', fontweight='bold')
                bottom += val
            # Total above the stack
            top_val = bottom
            top_text = (f'{int(round(top_val * 100))}%' if normalize
                        else f'{raw_total:.2f}')
            ax.text(x[cls_idx], top_val + 0.015 * max(top_val, 1.0),
                    top_text, ha='center', va='bottom',
                    fontsize=9, color='#222', fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(class_names, fontsize=10)
        ax.set_ylabel('Share of evidence' if normalize else 'Mean evidence',
                      fontsize=10)
        if normalize:
            ax.set_ylim(0, 1.10)
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, _: f'{int(y*100)}%'))
        ax.set_xlim(-0.6, len(class_names) - 0.4)

    if proportional:
        _stacked(axes[0], normalize=True)
        axes[0].set_title('Composition (normalised)')
        _stacked(axes[1], normalize=False)
        axes[1].set_title('Magnitude (absolute)')
        axes[1].legend(loc='upper left', fontsize=9, frameon=False)
    else:
        _stacked(axes[0], normalize=False)
        axes[0].legend()

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_discriminative_gap(
    real_values: Sequence[float],
    fake_values: Sequence[float],
    category_names: List[str],
    title: str,
    save_path: str,
    top_k: int = 12,
    pretty_labels: bool = True,
):
    """Bar chart of |fake − real| per category, ranked.

    Use for consistency-rule firing rates: a flat side-by-side comparison
    hides which rules actually discriminate. Plotting the gap (signed)
    makes class-distinctive rules pop out.
    """
    real = np.asarray(real_values, dtype=float)
    fake = np.asarray(fake_values, dtype=float)
    gap = fake - real
    order = np.argsort(-np.abs(gap))[:top_k]

    cats = [category_names[i] for i in order]
    pretty_cats = pretty_many(cats) if pretty_labels else cats
    g = gap[order]

    fig_w = float(np.clip(2.5 + 0.40 * len(cats), 4.5, 9.0))
    fig, ax = plt.subplots(figsize=(fig_w, 4.0))
    colors = ['#d62728' if v > 0 else '#1f77b4' for v in g]
    bars = ax.bar(range(len(cats)), g, width=0.55, color=colors,
                  alpha=0.88, edgecolor='white', linewidth=0.5)
    ax.axhline(0, color='black', lw=0.7)
    # Numeric labels: above for positive, below for negative
    span = float(g.max() - g.min()) if len(g) else 1.0
    pad = 0.025 * (span + 1e-6)
    for b, v in zip(bars, g):
        ax.text(b.get_x() + b.get_width() / 2,
                v + (pad if v >= 0 else -pad),
                f'{v:+.3f}',
                ha='center', va='bottom' if v >= 0 else 'top',
                fontsize=7.5, color='#222')
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(pretty_cats, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel(r'Mean$_{\mathrm{fake}}$ - Mean$_{\mathrm{real}}$',
                  fontsize=10)
    ax.set_title(title, fontsize=11)
    # Headroom for labels
    ax.set_ylim(g.min() - 4 * pad, g.max() + 4 * pad)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor='#d62728', alpha=0.85, label='fires more on fakes'),
        Patch(facecolor='#1f77b4', alpha=0.85, label='fires more on reals'),
    ], fontsize=8, loc='best', frameon=False)
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
    """Grouped bar chart (e.g., real vs fake per rule).

    Sized for paper columns: narrow bars, compact figure. Numeric values
    are annotated above every bar.
    """
    n_cats = len(category_names)
    n_groups = len(group_data)
    x = np.arange(n_cats)
    # Narrower bars + total cluster width 0.6 (was 0.8) so neighbouring
    # category clusters keep visible whitespace between them.
    width = 0.6 / max(n_groups, 1)
    colors = plt.cm.Set1(np.linspace(0, 1, n_groups))

    # Compact width for paper figures; scales with #categories but
    # capped to a reasonable single/double-column footprint.
    fig_w = float(np.clip(2.0 + 0.45 * n_cats, 4.5, 9.0))
    fig, ax = plt.subplots(figsize=(fig_w, 4.0))

    all_vals = []
    for i, (label, values) in enumerate(group_data.items()):
        offset = (i - n_groups / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=label,
                      color=colors[i], alpha=0.85,
                      edgecolor='white', linewidth=0.5)
        all_vals.extend(values)
        # Numeric label above every bar
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height(),
                    f'{v:.2f}', ha='center', va='bottom',
                    fontsize=7.5, color='#222')

    ax.set_xticks(x)
    ax.set_xticklabels(category_names, rotation=rotate_labels, ha='right',
                       fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    # Headroom so the top labels don't get clipped
    if all_vals:
        vmax = max(all_vals); vmin = min(0.0, min(all_vals))
        ax.set_ylim(vmin, vmax + 0.10 * (vmax - vmin + 1e-6))
    ax.legend(fontsize=8, frameon=False)
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
    crop_inactive: bool = True,
    activity_thresh: float = 1e-4,
    max_label_count: int = 60,
    pretty_labels: bool = True,
):
    """Render N adjacency matrices side-by-side, paper-ready.

    Improvements over the original:
      • Auto-crops rows/cols whose combined activity (across all matrices)
        is below `activity_thresh`. Removes the giant white margins that
        came from sparse SCM adjacencies.
      • Uses a SHARED scale for matched matrices (e.g. ``A_real``,
        ``A_fake``) but a SEPARATE, tighter scale for any matrix whose
        name starts with ``|`` or ``Δ`` (the divergence panel) — that way
        small differences don't get flattened by a wide range.
      • Pretty-prints node labels via ``pretty_names.pretty`` so e.g.
        ``ff_srm_hedge_kurt`` becomes ``κ_SRM^Hedge`` in math mode.
      • Tints tick labels by semantic group (latent / curated / forensic
        sub-types) and adds a group-colour legend.
    """
    names = list(matrices.keys())
    arrs = [np.asarray(matrices[n]) for n in names]
    n_panels = len(arrs)

    # ── Crop inactive rows/cols (combined across panels) ───────────────
    if crop_inactive and arrs:
        combined = np.sum([np.abs(a) for a in arrs], axis=0)
        active_rows = np.where(combined.sum(axis=1) > activity_thresh)[0]
        active_cols = np.where(combined.sum(axis=0) > activity_thresh)[0]
        if active_rows.size and active_cols.size:
            r0, r1 = active_rows.min(), active_rows.max() + 1
            c0, c1 = active_cols.min(), active_cols.max() + 1
            arrs = [a[r0:r1, c0:c1] for a in arrs]
            if row_labels is not None:
                row_labels = list(row_labels[r0:r1])
            if col_labels is not None:
                col_labels = list(col_labels[c0:c1])

    h, w = arrs[0].shape
    cell = 0.30 if max(h, w) <= 40 else 0.20
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(max(5, w * cell) * n_panels + 1.5, max(5, h * cell + 2.5)),
        squeeze=False,
    )
    axes = axes[0]

    # Two scales: one for shared "raw" panels, one for diff panel.
    # A diff panel is anything whose key contains "|...|" (or starts with Δ).
    def _is_diff_key(s: str) -> bool:
        s = s.lstrip('$').lstrip()
        return s.startswith('|') or s.startswith('Δ') or s.startswith(r'\Delta')
    raw_idx = [i for i, nm in enumerate(names) if not _is_diff_key(nm)]
    diff_idx = [i for i in range(n_panels) if i not in raw_idx]
    vmax_raw = max((np.abs(arrs[i]).max() for i in raw_idx), default=1.0)
    vmax_diff = max((np.abs(arrs[i]).max() for i in diff_idx), default=1.0)
    vmax_raw = max(vmax_raw, 1e-6)
    vmax_diff = max(vmax_diff, 1e-6)

    show_labels = (row_labels is not None and len(row_labels) <= max_label_count)
    pretty_rows = pretty_many(row_labels) if (show_labels and pretty_labels) else row_labels
    pretty_cols = pretty_many(col_labels) if (show_labels and pretty_labels) else col_labels

    last_im_raw = last_im_diff = None
    for i, (ax, nm, A) in enumerate(zip(axes, names, arrs)):
        is_diff = nm.startswith('|') or nm.startswith('Δ')
        if is_diff:
            im = ax.imshow(A, cmap='magma', aspect='auto',
                           vmin=0.0, vmax=vmax_diff)
            last_im_diff = im
        else:
            vmin = -vmax_raw if symmetric else 0.0
            im = ax.imshow(A, cmap=cmap, aspect='auto',
                           vmin=vmin, vmax=vmax_raw)
            last_im_raw = im
        ax.set_title(nm, fontsize=13)

        if show_labels:
            ax.set_yticks(range(len(pretty_rows)))
            ax.set_yticklabels(pretty_rows, fontsize=7)
            ax.set_xticks(range(len(pretty_cols)))
            ax.set_xticklabels(pretty_cols, rotation=90, fontsize=7)
            # Tint tick text by semantic group (left=row labels, bottom=col labels)
            for tick, raw in zip(ax.get_yticklabels(), row_labels or []):
                tick.set_color(GROUP_COLORS.get(group_of(raw), '#333'))
            for tick, raw in zip(ax.get_xticklabels(), col_labels or []):
                tick.set_color(GROUP_COLORS.get(group_of(raw), '#333'))
        else:
            ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel('target', fontsize=10)
        if i == 0:
            ax.set_ylabel('source', fontsize=10)

    # Two colourbars so each scale is honest
    if last_im_raw is not None:
        cb1 = fig.colorbar(last_im_raw, ax=axes[raw_idx], shrink=0.75,
                           pad=0.02, fraction=0.04)
        cb1.set_label(r'edge weight $A_{ij}$', fontsize=10)
    if last_im_diff is not None:
        cb2 = fig.colorbar(last_im_diff, ax=axes[diff_idx[0]],
                           shrink=0.75, pad=0.02, fraction=0.04)
        cb2.set_label(r'$|A^{\mathrm{real}}_{ij}-A^{\mathrm{fake}}_{ij}|$',
                      fontsize=10)

    # Group-colour legend (tick-label colour key)
    if show_labels:
        from matplotlib.patches import Patch
        seen = []
        for nm in (row_labels or []) + (col_labels or []):
            g = group_of(nm)
            if g not in seen:
                seen.append(g)
        handles = [Patch(facecolor=GROUP_COLORS[g], edgecolor='none',
                         label=GROUP_LONGNAMES.get(g, g))
                   for g in seen if g in GROUP_COLORS]
        if handles:
            fig.legend(handles=handles, loc='lower center',
                       ncol=min(len(handles), 5), fontsize=9,
                       frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(suptitle, fontsize=14)
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
    """Draw the top-K most-divergent edges of the per-class SCM.

    Layout:
      • Node positions come from Graphviz `dot` if pygraphviz/pydot is
        installed (cleaner DAGs); otherwise spring_layout with a fixed
        seed.
      • Nodes are coloured by their semantic group (latent / curated /
        forensic sub-type) using the palette in ``pretty_names``.
      • Labels are placed in white-bordered tags ABOVE each node — they
        no longer overlap with the node circle or the edges.
      • Each kept (i,j) is drawn TWICE: once with the real weight (blue)
        and once with the fake weight (red); width ∝ |weight|.
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
    G.add_edges_from([(s, t) for s, t, _, _ in edges])

    # Prefer Graphviz `dot` for an acyclic layout; fall back to spring.
    pos = None
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        pos = graphviz_layout(G, prog='dot')
    except Exception:
        try:
            from networkx.drawing.nx_pydot import graphviz_layout
            pos = graphviz_layout(G, prog='dot')
        except Exception:
            pos = nx.spring_layout(
                G, seed=7, k=1.6 / max(1.0, np.sqrt(len(nodes))))

    fig, ax = plt.subplots(figsize=(13, 9))

    # ── Node circles (small, group-coloured) ──────────────────────────
    node_colors = [GROUP_COLORS.get(group_of(n), '#bbbbbb') for n in G.nodes]
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=node_colors,
        edgecolors='#222', linewidths=1.2, node_size=300, alpha=0.95,
    )

    # ── Edges: blue = real weight, red = fake weight ──────────────────
    max_w = max(max(abs(er), abs(ef)) for _, _, er, ef in edges) or 1.0
    for src, tgt, wr, wf in edges:
        if abs(wr) > 1e-6:
            nx.draw_networkx_edges(
                G, pos, ax=ax, edgelist=[(src, tgt)],
                edge_color='#1f77b4',
                width=0.8 + 4.5 * abs(wr) / max_w,
                alpha=0.85, arrows=True, arrowsize=14,
                connectionstyle='arc3,rad=0.12',
                node_size=300,
            )
        if abs(wf) > 1e-6:
            nx.draw_networkx_edges(
                G, pos, ax=ax, edgelist=[(src, tgt)],
                edge_color='#d62728',
                width=0.8 + 4.5 * abs(wf) / max_w,
                alpha=0.85, arrows=True, arrowsize=14,
                connectionstyle='arc3,rad=-0.12',
                node_size=300,
            )

    # ── Labels: pretty math, in a white-bordered bubble *above* node ──
    if pos:
        ys = [p[1] for p in pos.values()]
        dy = 0.025 * (max(ys) - min(ys) + 1.0)  # offset the tag upward
    else:
        dy = 0.0
    for nm, (x, y) in pos.items():
        ax.text(
            x, y + dy, pretty(nm),
            fontsize=9, ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.25',
                      facecolor='white', edgecolor='#888', alpha=0.92),
        )

    # ── Legends: edge colour + node group ─────────────────────────────
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    edge_handles = [
        Line2D([0], [0], color='#1f77b4', lw=2.5,
               label=r'$A^{\mathrm{real}}_{ij}$  (real-trained SCM)'),
        Line2D([0], [0], color='#d62728', lw=2.5,
               label=r'$A^{\mathrm{fake}}_{ij}$  (fake-trained SCM)'),
    ]
    seen = []
    for nm in G.nodes:
        g = group_of(nm)
        if g not in seen:
            seen.append(g)
    node_handles = [
        Patch(facecolor=GROUP_COLORS.get(g, '#bbb'),
              edgecolor='#222', label=GROUP_LONGNAMES.get(g, g))
        for g in seen
    ]
    leg1 = ax.legend(handles=edge_handles, loc='upper left', fontsize=10,
                     frameon=True, framealpha=0.9, title='Edges')
    ax.add_artist(leg1)
    ax.legend(handles=node_handles, loc='upper right', fontsize=9,
              frameon=True, framealpha=0.9, title='Node group')

    ax.set_title(f'{title} — top-{len(edges)} divergent edges',
                 fontsize=13)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(save_path, dpi=170, bbox_inches='tight')
    plt.close(fig)


def plot_separated_class_graphs(
    A_real: np.ndarray,
    A_fake: np.ndarray,
    node_names: List[str],
    top_k: int,
    title: str,
    save_path: str,
):
    """Draw the per-class SCMs as TWO separate files (one real, one fake)
    sharing one node layout, so edge presence/absence is readable without
    parallel-edge clutter.

    The ``save_path`` argument is treated as a base — this function writes
    ``<base>_real<ext>`` and ``<base>_fake<ext>``.

    Edge selection: we union the top-K edges of A_real and A_fake plus
    the top-K most-divergent edges, so each panel shows that union but
    coloured / weighted with its own class' adjacency. Edges absent in
    a class are drawn faintly (`alpha≈0.18`) — that's how you see the
    *removed* couplings at a glance.

    Readability:
      • Nodes are placed on concentric shells when there are multiple
        semantic groups (e.g. FFT vs latent), or evenly on a single
        circle when there's only one group. This avoids the dense central
        cluster that ``spring`` / ``dot`` produce on hub-shaped SCMs.
      • Reciprocal and parallel edges between the same node-pair are
        assigned distinct curvatures so they no longer overlap.
      • One panel per file at a larger figure size, with bigger nodes,
        labels and arrowheads.
    """
    try:
        import networkx as nx
    except ImportError:
        return

    n = min(A_real.shape[0], A_fake.shape[0], len(node_names))
    if n == 0:
        return

    # Edge candidates: top-K by |A_real|, |A_fake|, and |A_real - A_fake|
    def _topk(M, k):
        flat = M.flatten()
        idx = np.argsort(-np.abs(flat))[:k]
        return [(int(divmod(i, n)[0]), int(divmod(i, n)[1]))
                for i in idx if abs(flat[i]) > 1e-6]

    cand = set()
    cand.update(_topk(A_real[:n, :n], top_k))
    cand.update(_topk(A_fake[:n, :n], top_k))
    cand.update(_topk(np.abs(A_real[:n, :n] - A_fake[:n, :n]), top_k))
    if not cand:
        return

    nodes = sorted({node_names[i] for i, _ in cand}
                   | {node_names[j] for _, j in cand})

    # ── Layout: pick something that distributes nodes evenly ──────────
    # Bucket nodes by semantic group, then either:
    #   - 2-4 groups → shell layout (one concentric ring per group), or
    #   - otherwise   → kamada-kawai (still fairly even, no clusters), or
    #   - last resort → circular (perfect even spacing on one ring).
    G_union = nx.DiGraph()
    G_union.add_nodes_from(nodes)
    G_union.add_edges_from([(node_names[i], node_names[j]) for i, j in cand])

    pos = None
    groups: Dict[str, List[str]] = {}
    for nm in nodes:
        groups.setdefault(group_of(nm), []).append(nm)
    if 2 <= len(groups) <= 4:
        # Largest group on the outer ring, smallest on the inner — keeps
        # the dense category around the perimeter where there's room.
        ordered = sorted(groups.values(), key=len, reverse=True)
        try:
            pos = nx.shell_layout(G_union, nlist=ordered)
        except Exception:
            pos = None
    if pos is None:
        try:
            pos = nx.kamada_kawai_layout(G_union)
        except Exception:
            pos = nx.circular_layout(G_union)

    # ── Edge curvatures: spread parallel/reciprocal edges symmetrically
    # around 0 so they no longer stack on the same straight line.
    pair_buckets: Dict[frozenset, List[Tuple[int, int]]] = {}
    for (i, j) in cand:
        pair_buckets.setdefault(
            frozenset({node_names[i], node_names[j]}), []).append((i, j))
    rad_map: Dict[Tuple[int, int], float] = {}
    for eds in pair_buckets.values():
        m = len(eds)
        if m == 1:
            rad_map[eds[0]] = 0.12
        else:
            for ed, r in zip(eds, np.linspace(-0.32, 0.32, m)):
                rad_map[ed] = float(r)

    base, ext = os.path.splitext(save_path)
    panel_specs = [
        ('Real-trained SCM', A_real[:n, :n], '#1f77b4', f'{base}_real{ext}'),
        ('Fake-trained SCM', A_fake[:n, :n], '#d62728', f'{base}_fake{ext}'),
    ]
    max_w = max(np.abs(A_real[:n, :n]).max(),
                np.abs(A_fake[:n, :n]).max(), 1e-6)

    # Edges to numerically label: union of top-strong-in-class and top-
    # divergent. Capped so the figure doesn't drown in numbers.
    div_mat = np.abs(A_real[:n, :n] - A_fake[:n, :n])
    label_cap = min(top_k, 10)
    div_flat = div_mat.flatten()
    label_edges_div = {(int(divmod(int(i), n)[0]), int(divmod(int(i), n)[1]))
                       for i in np.argsort(-div_flat)[:label_cap]
                       if div_flat[i] > 1e-6}

    xs = [p[0] for p in pos.values()] or [0.0]
    ys = [p[1] for p in pos.values()] or [0.0]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    dy = 0.045 * span  # label offset above each node

    from matplotlib.patches import Patch
    seen_groups: List[str] = []
    for nm in nodes:
        g = group_of(nm)
        if g not in seen_groups:
            seen_groups.append(g)
    legend_handles = [
        Patch(facecolor=GROUP_COLORS.get(g, '#bbb'),
              edgecolor='#222', label=GROUP_LONGNAMES.get(g, g))
        for g in seen_groups
    ]

    for panel_title, A, edge_color, file_path in panel_specs:
        fig, ax = plt.subplots(figsize=(13, 11))

        node_colors = [GROUP_COLORS.get(group_of(nm), '#bbbbbb') for nm in nodes]
        nx.draw_networkx_nodes(
            G_union, pos, ax=ax, nodelist=nodes,
            node_color=node_colors, edgecolors='#222',
            linewidths=1.4, node_size=520, alpha=0.95,
        )

        for (i, j) in cand:
            w = float(A[i, j])
            src, tgt = node_names[i], node_names[j]
            strong = abs(w) > 1e-3
            rad = rad_map.get((i, j), 0.12)
            nx.draw_networkx_edges(
                G_union, pos, ax=ax, edgelist=[(src, tgt)],
                edge_color=edge_color if strong else '#dddddd',
                width=(0.9 + 4.5 * abs(w) / max_w) if strong else 0.6,
                alpha=0.88 if strong else 0.18,
                arrows=True, arrowsize=15,
                connectionstyle=f'arc3,rad={rad:.3f}',
                node_size=520,
            )

        # Numerical edge labels for the top-divergent edges only — these
        # are where the two classes actually disagree, so the numbers
        # carry signal. Labels for all edges would be unreadable.
        for (i, j) in label_edges_div:
            if (i, j) not in cand:
                continue
            w = float(A[i, j])
            src, tgt = node_names[i], node_names[j]
            x0, y0 = pos[src]; x1, y1 = pos[tgt]
            rad = rad_map.get((i, j), 0.12)
            # arc3 puts the midpoint perpendicular to the chord, offset
            # by ~rad * chord_len / 2. Approximate that.
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            # Perpendicular unit vector
            dx, dy_ = (x1 - x0), (y1 - y0)
            length = max((dx * dx + dy_ * dy_) ** 0.5, 1e-6)
            perp = (-dy_ / length, dx / length)
            offset = rad * length * 0.5
            lx, ly = mx + perp[0] * offset, my + perp[1] * offset
            ax.text(
                lx, ly, f'{w:.3f}',
                fontsize=7.5, ha='center', va='center',
                color='#222',
                bbox=dict(boxstyle='round,pad=0.15',
                          facecolor='white', edgecolor='none', alpha=0.85),
                zorder=10,
            )

        for nm, (x, y) in pos.items():
            ax.text(
                x, y + dy, pretty(nm),
                fontsize=10, ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.28',
                          facecolor='white', edgecolor='#666', alpha=0.95),
            )

        ax.legend(
            handles=legend_handles, loc='lower center',
            ncol=min(len(legend_handles), 3), fontsize=10,
            frameon=True, framealpha=0.92, title='Node group',
            bbox_to_anchor=(0.5, -0.04),
        )

        ax.set_title(
            f'{title} — {panel_title}\n'
            f'(numeric labels: top-{label_cap} most class-divergent edges)',
            fontsize=13, pad=12,
        )
        ax.set_axis_off()
        ax.set_xlim(min(xs) - 0.20 * span, max(xs) + 0.20 * span)
        ax.set_ylim(min(ys) - 0.15 * span, max(ys) + 0.20 * span)
        fig.tight_layout()
        fig.savefig(file_path, dpi=170, bbox_inches='tight')
        plt.close(fig)


def plot_class_distinctive_graphs(
    A_real: np.ndarray,
    A_fake: np.ndarray,
    node_names: List[str],
    top_k: int,
    title: str,
    save_path: str,
    min_div_frac: float = 0.10,    # legacy kwarg, kept for API compat
    act_real: Optional[np.ndarray] = None,
    act_fake: Optional[np.ndarray] = None,
    keep_ratio: float = 0.5,
):
    """Real-canonical / fake-partial framing for the per-class SCMs.

    Renders TWO panels with the SAME node positions and the SAME edge
    set (the "canonical" structure):

      • **Real panel** — every top-K canonical edge drawn solid blue,
        width ∝ |A_real|. This represents the complete causal structure
        learned by the real-trained SCM.

      • **Fake panel** — same edges, classified per-edge by the ratio
        ``A_fake / |A_real|``:
          - ``≥ keep_ratio`` → solid red, width ∝ |A_fake| ("preserved")
          - ``< keep_ratio`` → faded grey dashed ("missing in fake")
        The fake-trained SCM weakened or lost some couplings; this panel
        shows which.

    Canonical edge selection: top-K by ``|A_real[i,j]| · ⟨|x_j|⟩_real``
    when activations are supplied, else raw |A_real|. Phantom edges
    (high weight on dead source nodes) are excluded.

    Files written:
      - ``<base>_distinctive_real<ext>``  — real panel PNG
      - ``<base>_distinctive_fake<ext>``  — fake panel PNG
      - ``<base>_distinctive_edges.csv``  — per-edge data table
      - ``<base>_distinctive_nodes.csv``  — per-node data table

    The CSVs carry every numeric value used in the figure (A_real,
    A_fake, divergence, activations, score, fake_status, ranks, group
    tags) so the figure can be rebuilt in R / Cytoscape / Gephi
    without rerunning the model.
    """
    try:
        import networkx as nx
    except ImportError:
        return

    n = min(A_real.shape[0], A_fake.shape[0], len(node_names))
    if n == 0:
        return

    A_r = A_real[:n, :n].astype(float)
    A_f = A_fake[:n, :n].astype(float)

    # ── Activation-weighted importance for canonical-edge selection ──
    has_acts = (act_real is not None and act_fake is not None
                and len(act_real) >= n and len(act_fake) >= n)
    if has_acts:
        a_r = np.abs(np.asarray(act_real[:n], dtype=float))
        a_f = np.abs(np.asarray(act_fake[:n], dtype=float))
        importance = np.abs(A_r) * a_r[None, :]
        score_metric = 'activation-weighted |A_real|'
    else:
        a_r = np.zeros(n)
        a_f = np.zeros(n)
        importance = np.abs(A_r)
        score_metric = '|A_real|'

    if importance.max() < 1e-9:
        return

    # ── Pick canonical edges: top-K by real-side importance ──────────
    flat = importance.flatten()
    order = np.argsort(-flat)
    canonical: List[Tuple[int, int]] = []
    for fi in order:
        if flat[fi] < 1e-9:
            break
        i, j = int(divmod(int(fi), n)[0]), int(divmod(int(fi), n)[1])
        canonical.append((i, j))
        if len(canonical) >= top_k:
            break
    if not canonical:
        return

    # ── Per-edge fake status: preserved vs missing ───────────────────
    fake_status: Dict[Tuple[int, int], str] = {}
    for (i, j) in canonical:
        ar = abs(float(A_r[i, j]))
        af = abs(float(A_f[i, j]))
        if ar < 1e-9:
            fake_status[(i, j)] = 'preserved'
        else:
            fake_status[(i, j)] = ('preserved' if (af / ar) >= keep_ratio
                                                else 'missing')

    # ── Build graph / shared layout ──────────────────────────────────
    nodes = sorted({node_names[i] for i, _ in canonical}
                   | {node_names[j] for _, j in canonical})
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from([(node_names[i], node_names[j]) for i, j in canonical])

    pos = None
    groups: Dict[str, List[str]] = {}
    for nm in nodes:
        groups.setdefault(group_of(nm), []).append(nm)
    if 2 <= len(groups) <= 4:
        ordered = sorted(groups.values(), key=len, reverse=True)
        try:
            pos = nx.shell_layout(G, nlist=ordered)
        except Exception:
            pos = None
    if pos is None:
        try:
            pos = nx.kamada_kawai_layout(G)
        except Exception:
            pos = nx.circular_layout(G)

    base, ext = os.path.splitext(save_path)

    # ── Edge curvatures (same approach as plot_separated_class_graphs) ─
    pair_buckets: Dict[frozenset, List[Tuple[int, int]]] = {}
    for (i, j) in canonical:
        pair_buckets.setdefault(
            frozenset({node_names[i], node_names[j]}), []).append((i, j))
    rad_map: Dict[Tuple[int, int], float] = {}
    for eds in pair_buckets.values():
        m = len(eds)
        if m == 1:
            rad_map[eds[0]] = 0.12
        else:
            for ed, r in zip(eds, np.linspace(-0.32, 0.32, m)):
                rad_map[ed] = float(r)

    # ── Width scaling (panel-specific so each is fully visible) ──────
    max_ar = max((abs(float(A_r[i, j])) for i, j in canonical), default=1e-6)
    max_af = max((abs(float(A_f[i, j])) for i, j in canonical), default=1e-6)

    xs = [p[0] for p in pos.values()] or [0.0]
    ys = [p[1] for p in pos.values()] or [0.0]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    dy = 0.045 * span

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    seen_groups: List[str] = []
    for nm in nodes:
        g = group_of(nm)
        if g not in seen_groups:
            seen_groups.append(g)
    node_legend = [
        Patch(facecolor=GROUP_COLORS.get(g, '#bbb'),
              edgecolor='#222', label=GROUP_LONGNAMES.get(g, g))
        for g in seen_groups
    ]

    n_missing = sum(1 for v in fake_status.values() if v == 'missing')
    n_preserved = sum(1 for v in fake_status.values() if v == 'preserved')

    panel_specs = [
        ('real', '#1f77b4', f'{base}_distinctive_real{ext}'),
        ('fake', '#d62728', f'{base}_distinctive_fake{ext}'),
    ]

    for mode, base_color, file_path in panel_specs:
        fig, ax = plt.subplots(figsize=(13, 11))

        node_colors = [GROUP_COLORS.get(group_of(nm), '#bbbbbb') for nm in nodes]
        nx.draw_networkx_nodes(
            G, pos, ax=ax, nodelist=nodes,
            node_color=node_colors, edgecolors='#222',
            linewidths=1.4, node_size=520, alpha=0.95,
        )

        for (i, j) in canonical:
            src, tgt = node_names[i], node_names[j]
            ar = float(A_r[i, j])
            af = float(A_f[i, j])
            rad = rad_map.get((i, j), 0.12)

            if mode == 'real':
                color = base_color
                width = 1.0 + 5.5 * abs(ar) / max_ar
                style = 'solid'
                alpha = 0.9
                label_val = ar
            else:  # fake
                if fake_status[(i, j)] == 'preserved':
                    color = base_color
                    width = 1.0 + 5.5 * abs(af) / max_af
                    style = 'solid'
                    alpha = 0.9
                    label_val = af
                else:  # missing
                    color = '#9e9e9e'
                    width = 0.7
                    style = 'dashed'
                    alpha = 0.30
                    label_val = af

            nx.draw_networkx_edges(
                G, pos, ax=ax, edgelist=[(src, tgt)],
                edge_color=color, width=width, alpha=alpha,
                arrows=True, arrowsize=15,
                connectionstyle=f'arc3,rad={rad:.3f}',
                style=style, node_size=520,
            )
            x0, y0 = pos[src]; x1, y1 = pos[tgt]
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            dx_, dy_ = (x1 - x0), (y1 - y0)
            ln = max((dx_ * dx_ + dy_ * dy_) ** 0.5, 1e-6)
            perp = (-dy_ / ln, dx_ / ln)
            off = rad * ln * 0.5
            ax.text(
                mx + perp[0] * off, my + perp[1] * off,
                f'{label_val:.3f}',
                fontsize=8, ha='center', va='center',
                color='#666' if (mode == 'fake' and
                                 fake_status[(i, j)] == 'missing') else '#222',
                bbox=dict(boxstyle='round,pad=0.15',
                          facecolor='white', edgecolor='none', alpha=0.85),
                zorder=10,
            )

        for nm, (x, y) in pos.items():
            ax.text(
                x, y + dy, pretty(nm),
                fontsize=10, ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.28',
                          facecolor='white', edgecolor='#666', alpha=0.95),
            )

        leg1 = ax.legend(
            handles=node_legend, loc='lower center',
            ncol=min(len(node_legend), 3), fontsize=10,
            frameon=True, framealpha=0.92, title='Node group',
            bbox_to_anchor=(0.5, -0.04),
        )
        ax.add_artist(leg1)
        # Fake panel: a second legend explains the edge style code
        if mode == 'fake':
            edge_legend = [
                Line2D([0], [0], color=base_color, lw=3,
                       label='preserved in fake'),
                Line2D([0], [0], color='#9e9e9e', lw=1.5, ls='--',
                       label=f'missing in fake (A_fake/A_real < {keep_ratio:g})'),
            ]
            ax.legend(handles=edge_legend, loc='upper left',
                      fontsize=9, frameon=True, framealpha=0.9,
                      title='Edge status')

        if mode == 'real':
            sub = (f'Real-trained SCM — canonical structure '
                   f'(top-{len(canonical)} by {score_metric})')
        else:
            sub = (f'Fake-trained SCM — {n_preserved} preserved / '
                   f'{n_missing} missing of {len(canonical)} canonical edges')
        ax.set_title(f'{title} — {sub}', fontsize=12, pad=12)
        ax.set_axis_off()
        ax.set_xlim(min(xs) - 0.20 * span, max(xs) + 0.20 * span)
        ax.set_ylim(min(ys) - 0.15 * span, max(ys) + 0.20 * span)
        fig.tight_layout()
        fig.savefig(file_path, dpi=170, bbox_inches='tight')
        plt.close(fig)

    # ── CSV exports for downstream re-rendering (R / Cytoscape) ──────
    edges_csv = f'{base}_distinctive_edges.csv'
    nodes_csv = f'{base}_distinctive_nodes.csv'

    with open(edges_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'rank', 'src_idx', 'tgt_idx', 'src_name', 'tgt_name',
            'src_pretty', 'tgt_pretty', 'src_group', 'tgt_group',
            'A_real', 'A_fake', 'div_signed', 'abs_div',
            'act_real_src', 'act_fake_src',
            'score_real', 'score_fake',
            'fake_ratio', 'fake_status',
        ])
        for rank, (i, j) in enumerate(canonical, 1):
            ar = float(A_r[i, j]); af = float(A_f[i, j])
            div_signed = ar - af
            abs_div = abs(div_signed)
            ar_act = float(a_r[j]) if has_acts else 0.0
            af_act = float(a_f[j]) if has_acts else 0.0
            score_real = max(0.0, div_signed) * ar_act
            score_fake = max(0.0, -div_signed) * af_act
            ratio = (af / abs(ar)) if abs(ar) > 1e-9 else float('inf')
            w.writerow([
                rank, i, j, node_names[i], node_names[j],
                pretty(node_names[i]), pretty(node_names[j]),
                group_of(node_names[i]), group_of(node_names[j]),
                f'{ar:.6f}', f'{af:.6f}',
                f'{div_signed:.6f}', f'{abs_div:.6f}',
                f'{ar_act:.6f}', f'{af_act:.6f}',
                f'{score_real:.6f}', f'{score_fake:.6f}',
                f'{ratio:.6f}' if np.isfinite(ratio) else 'inf',
                fake_status[(i, j)],
            ])

    with open(nodes_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['node_id', 'node_pretty', 'node_group',
                    'pos_x', 'pos_y',
                    'act_real', 'act_fake',
                    'in_degree', 'out_degree'])
        # Compute degrees from the canonical edge set
        in_deg: Dict[str, int] = {nm: 0 for nm in nodes}
        out_deg: Dict[str, int] = {nm: 0 for nm in nodes}
        for (i, j) in canonical:
            out_deg[node_names[i]] = out_deg.get(node_names[i], 0) + 1
            in_deg[node_names[j]] = in_deg.get(node_names[j], 0) + 1
        for nm in nodes:
            try:
                idx = node_names.index(nm)
            except ValueError:
                idx = -1
            ar_act = (float(a_r[idx]) if has_acts and 0 <= idx < n else 0.0)
            af_act = (float(a_f[idx]) if has_acts and 0 <= idx < n else 0.0)
            x, y = pos.get(nm, (0.0, 0.0))
            w.writerow([
                nm, pretty(nm), group_of(nm),
                f'{x:.6f}', f'{y:.6f}',
                f'{ar_act:.6f}', f'{af_act:.6f}',
                in_deg.get(nm, 0), out_deg.get(nm, 0),
            ])


def plot_edge_weight_scatter(
    A_real: np.ndarray,
    A_fake: np.ndarray,
    node_names: List[str],
    title: str,
    save_path: str,
    label_top_k: int = 10,
):
    """Scatter every edge as ``(A_real[i,j], A_fake[i,j])``.

    Points on the y=x diagonal are edges identical between classes;
    off-diagonal points are class-distinctive. The top-``label_top_k``
    most divergent points get text labels with their source→target name.

    Single-figure summary that simultaneously shows:
      • how similar the two SCMs are overall (mass clustering on diagonal),
      • which specific edges drive the classification signal (outliers).
    Paper-friendly.
    """
    n = min(A_real.shape[0], A_fake.shape[0], len(node_names))
    if n == 0:
        return
    A_r = A_real[:n, :n].astype(float)
    A_f = A_fake[:n, :n].astype(float)
    iu = np.indices((n, n)).reshape(2, -1)
    xs = A_r.flatten()
    ys = A_f.flatten()

    fig, ax = plt.subplots(figsize=(8, 7.5))
    div = np.abs(xs - ys)
    sc = ax.scatter(xs, ys, c=div, cmap='magma',
                    s=12 + 60 * (div / max(div.max(), 1e-6)),
                    alpha=0.75, edgecolors='none')
    cb = fig.colorbar(sc, ax=ax, fraction=0.045)
    cb.set_label(r'$|A^{\mathrm{real}}_{ij}-A^{\mathrm{fake}}_{ij}|$',
                 fontsize=10)

    # Diagonal reference
    lo = float(min(xs.min(), ys.min(), -1e-3))
    hi = float(max(xs.max(), ys.max(), 1e-3))
    pad = 0.08 * (hi - lo + 1e-6)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            ls='--', color='#888', lw=1.0,
            label=r'$A^{\mathrm{real}}=A^{\mathrm{fake}}$')

    # Label top-K most divergent points
    order = np.argsort(-div)[:label_top_k]
    for idx in order:
        if div[idx] < 1e-6:
            break
        i, j = int(iu[0, idx]), int(iu[1, idx])
        side = 'R' if xs[idx] > ys[idx] else 'F'
        col = '#1f77b4' if side == 'R' else '#d62728'
        ax.annotate(
            f'{pretty(node_names[i])} → {pretty(node_names[j])}',
            xy=(xs[idx], ys[idx]),
            xytext=(8, 8), textcoords='offset points',
            fontsize=7.5, color=col,
            bbox=dict(boxstyle='round,pad=0.18',
                      facecolor='white', edgecolor=col, alpha=0.9),
            arrowprops=dict(arrowstyle='-', color=col, lw=0.6),
        )

    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(r'Edge weight in real-trained SCM, $A^{\mathrm{real}}_{ij}$',
                  fontsize=11)
    ax.set_ylabel(r'Edge weight in fake-trained SCM, $A^{\mathrm{fake}}_{ij}$',
                  fontsize=11)
    ax.set_title(
        f'{title} — per-edge real vs. fake weight\n'
        f'(off-diagonal points = class-distinctive edges)',
        fontsize=12, pad=10,
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper left', fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=170, bbox_inches='tight')
    plt.close(fig)


def plot_top_edge_comparison(
    A_real: np.ndarray,
    A_fake: np.ndarray,
    node_names: List[str],
    top_k: int,
    title: str,
    save_path: str,
):
    """Horizontal paired bars for the top-K most class-divergent edges:
    each row = one edge, blue bar = real weight, red bar = fake weight.

    The most direct presentation of "this is *which way* and *by how
    much* each edge differs between classes." Pairs with the scatter
    above for a complete picture.
    """
    n = min(A_real.shape[0], A_fake.shape[0], len(node_names))
    if n == 0:
        return
    A_r = A_real[:n, :n].astype(float)
    A_f = A_fake[:n, :n].astype(float)
    div = np.abs(A_r - A_f)
    if div.max() < 1e-6:
        return

    flat_idx = np.argsort(-div.flatten())[:top_k]
    rows: List[Tuple[str, float, float]] = []
    for fi in flat_idx:
        if div.flatten()[fi] < 1e-6:
            break
        i, j = int(divmod(int(fi), n)[0]), int(divmod(int(fi), n)[1])
        rows.append((
            f'{pretty(node_names[i])} → {pretty(node_names[j])}',
            float(A_r[i, j]),
            float(A_f[i, j]),
        ))
    if not rows:
        return

    labels = [r[0] for r in rows]
    rs = np.array([r[1] for r in rows])
    fs = np.array([r[2] for r in rows])
    y = np.arange(len(rows))[::-1]  # top-divergent at top of figure
    h = 0.4

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.42 * len(rows) + 1.5)))
    ax.barh(y + h / 2, rs, height=h, color='#1f77b4', alpha=0.9,
            label='real-trained $A^{\\mathrm{real}}_{ij}$')
    ax.barh(y - h / 2, fs, height=h, color='#d62728', alpha=0.9,
            label='fake-trained $A^{\\mathrm{fake}}_{ij}$')

    # Numeric annotations to the right of each bar
    for yi, ri, fi_ in zip(y, rs, fs):
        ax.text(ri + 0.002 * max(rs.max(), fs.max(), 1e-6),
                yi + h / 2, f'{ri:.3f}',
                va='center', fontsize=8, color='#1f77b4')
        ax.text(fi_ + 0.002 * max(rs.max(), fs.max(), 1e-6),
                yi - h / 2, f'{fi_:.3f}',
                va='center', fontsize=8, color='#d62728')

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(r'Edge weight $|A_{ij}|$', fontsize=11)
    ax.set_title(
        f'{title} — top-{len(rows)} most class-divergent edges',
        fontsize=12, pad=8,
    )
    ax.axvline(0, color='black', lw=0.6)
    ax.grid(True, axis='x', alpha=0.25)
    ax.legend(loc='lower right', fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=170, bbox_inches='tight')
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

#!/usr/bin/env python3
"""Figures — the ablation bar chart and the embedding panels.

    python analysis/tbiom/make_figures.py --ablation
    python analysis/tbiom/make_figures.py --panels --dataset CDFv2 --arm C

Every figure is drawn from SAVED data -- `tbiom/FULL_RESULTS.json` for the bars and the
`emb_*.npz` coordinates for the panels -- so a figure never depends on a fresh t-SNE and can be
regenerated identically.

PALETTE. Validated with the dataviz validator rather than chosen by eye; all six checks pass.
Two-class scatter #2166AC / #B2182B: CVD dE 21.1 (protan), normal 28.7. Three-group bars
#7048E8 / #E8590C / #3B5BDB: worst CVD dE 30.1.

WHAT THE PANELS CLAIM, and what they do not. Per-branch panels show how well ONE view separates
real from fake in its own feature space. The `concat` panel shows the three blocks concatenated.
That is a REPRESENTATION comparison, and it is labelled as such: the model fuses opinions, not
concatenated features, so calling that panel "our fusion" would misdescribe the architecture.
The claim the figure supports is that the three views carry complementary information -- which is
what the fusion then exploits, and what the AUROC bars measure directly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

T = Path("logs/tbiom/tsne")
FIG = Path("figures")
REAL, FAKE = "#2166AC", "#B2182B"
G_BRANCH, G_BASE, G_OURS = "#7048E8", "#E8590C", "#3B5BDB"
INK, MUTED, GRID = "#1a1d21", "#5b6572", "#dde2e8"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.alpha": 0.9,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})

# label -> (group, key in FULL_RESULTS.json)
ROWS = [
    ("FS-VFM branch alone",              "branch", "  branch: FS-VFM"),
    ("Semantic branch alone",            "branch", "  branch: semantic (CLIP)"),
    ("Artifact branch alone",            "branch", "  branch: artifact (β-VAE)"),
    ("2-view baseline (P0-DS)",          "base",   "P0-DS  (2 views, baseline)"),
    ("3 views, simple CE fusion",        "base",   "D  simple CE, 3 views"),
    ("3 views, evidential (ours)",       "ours",   "B  student Br3, 3 views"),
    ("3 views, evidential + FSFM (ours)", "ours",  "C  FSFM Br3, 3 views"),
]
OOD = ["CDFv2", "CDFv3", "DFD", "DFDC", "DFDCP", "DFEval24"]
COLOR = {"branch": G_BRANCH, "base": G_BASE, "ours": G_OURS}


def ablation(out: Path):
    """A DOT PLOT, deliberately, not bars.

    The interesting range here is 0.853-0.884 on a 0-1 metric. Bars imply a zero baseline, so
    drawing them on a truncated axis makes a 0.03 difference look like a landslide -- the single
    most common way a chart overstates its result, and the first thing a reviewer challenges. A
    dot plot carries no baseline implication, so the axis can start where the data is without
    misleading anyone, and close values are easier to compare besides.
    """
    j = json.load(open("tbiom/FULL_RESULTS.json"))
    labels, vals, groups = [], [], []
    for lab, grp, key in ROWS:
        v = [j[f"{key}|{d}"][0] for d in OOD if f"{key}|{d}" in j]
        if len(v) == len(OOD):
            labels.append(lab); vals.append(float(np.mean(v))); groups.append(grp)
    order = np.argsort(vals)
    labels = [labels[i] for i in order]; vals = [vals[i] for i in order]
    groups = [groups[i] for i in order]

    fig, ax = plt.subplots(figsize=(7.6, 0.5 * len(labels) + 1.7))
    ypos = np.arange(len(labels))
    lo, hi = min(vals) - 0.004, max(vals) + 0.006
    base = min(vals)
    # a hairline from the weakest single branch to each point: shows the GAIN, without a bar
    for i, (v, g) in enumerate(zip(vals, groups)):
        ax.plot([base, v], [i, i], color=COLOR[g], lw=1.6, alpha=0.30,
                solid_capstyle="round", zorder=2)
        ax.scatter([v], [i], s=104, color=COLOR[g], zorder=3,
                   edgecolor="white", linewidth=1.6)
        ax.text(v + 0.0011, i, f"{v:.4f}", va="center", ha="left",
                fontsize=8.6, color=INK, fontweight="bold", zorder=4)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9)
    for t, g in zip(ax.get_yticklabels(), groups):
        if g == "ours":
            t.set_color(INK); t.set_fontweight("bold")
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_xlabel("Mean video AUROC over 6 cross-dataset benchmarks", fontsize=9, color=MUTED)
    ax.grid(axis="y", visible=False)
    ax.set_title("Three collaborating views beat every single branch\nand simple-CE fusion",
                 fontsize=11.5, fontweight="bold", loc="left", pad=12)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=8, color=COLOR[g])
               for g in ("branch", "base", "ours")]
    ax.legend(handles, ["Single branch", "Baseline fusion", "Ours (evidential)"],
              loc="lower right", frameon=False, fontsize=8.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)
    print(f"wrote {out}")
    with open(out.with_suffix(".csv"), "w") as f:
        f.write("configuration,group,mean_ood_auroc\n")
        for lab, g, v in zip(labels, groups, vals):
            f.write(f'"{lab}",{g},{v:.6f}\n')


def panels(arm: str, ds: str, level: str, out: Path):
    spaces = [("semantic", "Semantic view (CLIP)"), ("artifact", "Artifact view (β-VAE)"),
              ("fsvfm", "FS-VFM view"), ("concat", "All three views together")]
    have = [(s, t) for s, t in spaces if (T / f"emb_{arm}_{ds}_{s}_{level}.npz").is_file()]
    if not have:
        print(f"  no embeddings for {arm}/{ds}/{level}"); return
    fig, axes = plt.subplots(1, len(have), figsize=(3.05 * len(have), 3.45))
    if len(have) == 1:
        axes = [axes]
    for ax, (s, title) in zip(axes, have):
        d = np.load(T / f"emb_{arm}_{ds}_{s}_{level}.npz")
        xy, y = d["xy"], d["label"]
        for cls, col, nm in ((0, REAL, "Real"), (1, FAKE, "Fake")):
            m = y == cls
            ax.scatter(xy[m, 0], xy[m, 1], s=7, c=col, alpha=0.55, linewidths=0,
                       label=nm, rasterized=True)
        ax.set_title(title, fontsize=9.5, fontweight="600" if s == "concat" else "500",
                     color=INK if s == "concat" else MUTED)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(GRID)
    axes[0].legend(loc="upper left", frameon=False, fontsize=8, markerscale=1.8)
    fig.suptitle(f"{ds} — each view alone versus all three together   ({level}-level, arm {arm})",
                 fontsize=10.5, fontweight="700", y=1.02, x=0.02, ha="left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--panels", action="store_true")
    ap.add_argument("--arm", default="C")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--level", default="video")
    args = ap.parse_args()
    if args.ablation:
        ablation(FIG / "ablation_auroc.png")
    if args.panels:
        dss = [args.dataset] if args.dataset else \
            ["CDFv2", "CDFv3", "DFD", "DFDC", "DFDCP", "DFEval24"]
        for d in dss:
            panels(args.arm, d, args.level, FIG / f"tsne_{args.arm}_{d}_{args.level}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())

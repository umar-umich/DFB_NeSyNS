#!/usr/bin/env python3
"""t-SNE and ablation figures for the three-view framework.

    python analysis/tbiom/make_tsne_figures.py --out tbiom/figures

WHAT EACH FIGURE ARGUES, and how it is kept honest.

  fig1_whole_vs_parts    Each branch's representation on its own, then all three concatenated.
                         The claim is that the joint representation separates the classes better
                         than any single view. Each panel carries a 5-fold CV logistic-probe
                         AUROC on that exact representation, so the visual impression is
                         backed by a number rather than asserted -- t-SNE geometry alone is
                         famously suggestive and proves nothing.

  fig2_arm_ablation      The same joint representation for every trained arm. Arm A is the
                         control whose two CLIP branches went vacuous (their standalone AUROC
                         fell to 0.554 and 0.391, the latter BELOW chance), so its panel is the
                         one that should look wrong. If it does not, the figure is not measuring
                         what it claims.

  fig3_configurations    Mean OOD AUROC per configuration with the seed range as an error bar.
                         The bar chart is the summary; the error bars are what stop it being
                         read as a bigger result than it is -- two configurations overlap.

VIDEO LEVEL. Features are averaged per video before anything else, using the same
`video_id` rule as every metric in this project. Plotting frames would put thousands of
near-duplicate points on the canvas and make every representation look separable.

THE PANEL NUMBERS ARE THE MODEL'S OWN CROSS-DATASET AUROC, not a probe. This matters and an
earlier version of this figure got it wrong.

A 5-fold CV logistic probe fit on the plotted dataset reports how separable the representation
is WITHIN that dataset -- it trains on 4 folds of Celeb-DF-v2 and tests on the fifth. On CDFv2
that read 0.992, against the 0.95 the model actually achieves there. The probe number is not
wrong, it answers a different question: in-domain separability, where every reported figure in
this project is zero-shot cross-dataset (trained on FF++ c23 only). Printing the two side by
side would invite the reader to believe the framework scores 0.992 cross-dataset.

So each panel now carries `p_semantic` / `p_artifact` / `p_fsvfm` / `p_fused` -- the trained
heads' video-level AUROC on that dataset, read from the same exports every table in this project
uses. Those are the numbers a reviewer can check against the results table, and they are lower.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

TSNE_DIR = Path("logs/tbiom/tsne")
# validated with the skill's validator: all-pairs CVD dE 24.7 (protan), normal dE 33.6, all PASS
C_REAL, C_FAKE = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dfe3e9"
ARMS = {"A": "A · fused-only", "B": "B · per-branch EDL", "C": "C · FSFM Branch 3",
        "D": "D · simple CE", "E": "E · trained 2-view"}
VIEWS = [("semantic", "Semantic (CLIP)"), ("artifact", "Artifact (β-VAE)"),
         ("fsvfm", "FS-VFM"), ("__all__", "All three (fused input)")]


def load_videos(arm: str, ds: str):
    """npz -> per-video mean features, using THE project video-identity rule."""
    from video_id import video_id
    import pandas as pd
    f = TSNE_DIR / f"f_{arm}_{ds}.npz"
    if not f.is_file():
        return None
    d = np.load(f, allow_pickle=True)
    vid = video_id(pd.Series([str(k) for k in d["key"]]))
    lab = np.asarray(d["label"]).astype(int)
    out = {}
    order = None
    for k in ("semantic", "artifact", "fsvfm"):
        df = pd.DataFrame(d[k]).assign(v=vid.values).groupby("v").mean()
        if order is None:
            order = df.index
        out[k] = df.loc[order].to_numpy(np.float32)
    y = pd.Series(lab).groupby(vid.values).max().loc[order].to_numpy()
    return out, y


def zs(x):
    return (x - x.mean(0)) / (x.std(0) + 1e-8)


def joint(fe):
    """The fusion's input space: each view standardised, then concatenated.

    Standardising first matters. FS-VFM is 1024-d against the others' 64-d, so on raw scales the
    concatenation would be an FS-VFM plot with decoration and the figure would be a fake.
    """
    return np.concatenate([zs(fe[k]) for k in ("semantic", "artifact", "fsvfm")], 1)


ARM_EXPORT = {"A": "run1fusedonly", "B": "run1auxedl", "C": "run1cft",
              "D": "run1dce", "E": "run1e"}
READOUT = {"semantic": "p_semantic", "artifact": "p_artifact",
           "fsvfm": "p_fsvfm", "__all__": "p_fused"}


def model_auc(arm: str, ds: str, key: str) -> float:
    """The trained head's video-level cross-dataset AUROC, from the project's own exports."""
    import pandas as pd
    from health import auroc
    from video_id import video_id
    col = READOUT[key]
    f = Path("logs/tbiom/stage23") / f"{ARM_EXPORT[arm]}_{ds}.csv"
    if not f.is_file():
        return float("nan")
    d = pd.read_csv(f)
    if col not in d:
        return float("nan")
    g = (pd.DataFrame({"v": video_id(d["key"]), "p": d[col], "y": d["label"]})
         .groupby("v", as_index=False).agg(p=("p", "mean"), y=("y", "max")))
    return auroc(g["y"].to_numpy(), g["p"].to_numpy())


def embed(X, seed=0):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    if X.shape[1] > 50:
        X = PCA(n_components=50, random_state=seed).fit_transform(X)
    per = max(5.0, min(30.0, (len(X) - 1) / 3.0))
    return TSNE(n_components=2, init="pca", perplexity=per, max_iter=1000,
                random_state=seed).fit_transform(X)


def panel(ax, E, y, title, auc, show_legend=False):
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color(GRID); spine.set_linewidth(0.8)
    for lab, col, name in ((0, C_REAL, "Real"), (1, C_FAKE, "Manipulated")):
        m = y == lab
        ax.scatter(E[m, 0], E[m, 1], s=13, c=col, alpha=0.75, linewidths=0.5,
                   edgecolors=SURFACE, label=name if show_legend else None)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10.5, color=INK, pad=7)
    ax.text(0.03, 0.03, f"AUROC {auc:.3f}", transform=ax.transAxes,
            fontsize=9, color=INK2, va="bottom",
            bbox=dict(boxstyle="round,pad=0.32", fc=SURFACE, ec=GRID, lw=0.7))


def fig_whole_vs_parts(arm, datasets, out: Path, seed=0):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [(ds, load_videos(arm, ds)) for ds in datasets]
    rows = [(ds, r) for ds, r in rows if r]
    if not rows:
        print("  no data"); return
    fig, axes = plt.subplots(len(rows), 4, figsize=(15.2, 4.0 * len(rows)),
                             facecolor=SURFACE, squeeze=False)
    for i, (ds, (fe, y)) in enumerate(rows):
        for j, (key, name) in enumerate(VIEWS):
            X = joint(fe) if key == "__all__" else zs(fe[key])
            panel(axes[i][j], embed(X, seed), y,
                  name if i == 0 else "", model_auc(arm, ds, key),
                  show_legend=(i == 0 and j == 0))
        axes[i][0].set_ylabel(ds, fontsize=11.5, color=INK, labelpad=10)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", frameon=False, fontsize=10, ncol=2,
               bbox_to_anchor=(0.995, 0.995))
    fig.suptitle(f"Each view alone, then all three fused — arm {arm}\n"
                 f"AUROC is the trained head's zero-shot cross-dataset video AUROC "
                 f"(trained on FF++ c23 only)",
                 fontsize=12.5, color=INK, y=1.0, x=0.012, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig); print(f"  wrote {out}.png / .pdf")


def fig_arm_ablation(ds, out: Path, seed=0):
    """The joint representation per arm. Arm A is the control that should look wrong."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    got = [(a, load_videos(a, ds)) for a in ARMS]
    got = [(a, r) for a, r in got if r]
    if not got:
        print("  no data"); return
    fig, axes = plt.subplots(1, len(got), figsize=(3.85 * len(got), 4.3),
                             facecolor=SURFACE, squeeze=False)
    for j, (a, (fe, y)) in enumerate(got):
        X = joint(fe)
        panel(axes[0][j], embed(X, seed), y, ARMS[a], model_auc(a, ds, "__all__"),
              show_legend=(j == 0))
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", frameon=False, fontsize=10, ncol=2,
               bbox_to_anchor=(0.995, 0.995))
    fig.suptitle(f"Fused three-view opinion across arms — {ds}\n"
                 f"AUROC is the fused head's zero-shot cross-dataset video AUROC",
                 fontsize=12.5, color=INK, y=1.0, x=0.008, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig); print(f"  wrote {out}.png / .pdf")


def fig_configurations(out: Path):
    """Mean OOD AUROC per configuration, seed range as the error bar.

    The error bars are the point: two configurations overlap, and a bare bar chart would read
    as a cleaner result than the data supports.
    """
    import json
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = Path("tbiom/SEED_SUMMARY.json")
    if not f.is_file():
        print("  no SEED_SUMMARY.json"); return
    raw = json.load(open(f))
    per = {}
    for k, v in raw.items():
        cfg = k.split("|seed")[0]
        per.setdefault(cfg, []).append(v["mean"])
    P0 = 0.8633
    order = sorted(per, key=lambda c: -np.mean(per[c]))
    means = [float(np.mean(per[c])) for c in order]
    lo = [m - min(per[c]) for m, c in zip(means, order)]
    hi = [max(per[c]) - m for m, c in zip(means, order)]

    fig, ax = plt.subplots(figsize=(8.6, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y = np.arange(len(order))
    ax.barh(y, [m - P0 for m in means], height=0.52, left=P0,
            color=C_REAL, edgecolor=SURFACE, linewidth=2)
    ax.errorbar([m for m in means], y, xerr=[lo, hi], fmt="none",
                ecolor=INK2, elinewidth=1.4, capsize=4)
    ax.axvline(P0, color=INK2, lw=1.4, ls="--")
    ax.text(P0, len(order) - 0.35, f"  two-view baseline {P0:.4f}", fontsize=9.5,
            color=INK2, va="center")
    for i, (m, c) in enumerate(zip(means, order)):
        ax.text(m + 0.0012, i, f"{m:.4f}  ({m-P0:+.4f})", va="center",
                fontsize=9.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([c.replace("arm ", "arm ") for c in order], fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlim(P0 - 0.004, max(means) + 0.010)
    ax.set_xlabel("mean video AUROC over six OOD datasets  (error bar = seed range, 3 seeds)",
                  fontsize=10, color=INK2)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="x", color=GRID, lw=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    fig.suptitle("Configuration ablation", fontsize=13.5, color=INK, y=1.0, x=0.008, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig); print(f"  wrote {out}.png / .pdf")


def fig_view_ablation(arm: str, out: Path):
    """Per-dataset AUROC for each view and the fused output, plus the mean.

    THE CLAIM THIS SUPPORTS, stated as the data supports it and no further. Fused beats every
    individual view on the MEAN over the six OOD sets (+0.0043 for arm B), and it is within
    0.008 of the best view on every single dataset. It does NOT beat every view on every
    dataset -- it wins outright on 2 of 6.

    What makes that a strong result rather than a weak one is the row of winners: the best
    single view CHANGES per dataset -- artifact on Celeb-DF-v2, semantic on DFD and DFDC,
    FS-VFM on Deepfake-Eval-2024. No view is reliable everywhere, so no view is a safe choice.
    The fused output is the only one that is never far from the top, which is what fusion is
    for: robustness across unseen generators, not a per-dataset peak.

    An earlier draft of this figure annotated an in-domain 5-fold probe (0.992 on CDFv2) and
    would have overstated the framework by four points. These are the trained heads' zero-shot
    numbers.
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OOD = ["CDFv2", "CDFv3", "DFD", "DFDC", "DFDCP", "DFEval24"]
    NICE = {"CDFv2": "Celeb-DF-v2", "CDFv3": "Celeb-DF-v3", "DFD": "DFD", "DFDC": "DFDC",
            "DFDCP": "DFDCP", "DFEval24": "Deepfake-Eval-2024"}
    series = [("semantic", "Semantic (CLIP)", "#2a78d6"),
              ("artifact", "Artifact (β-VAE)", "#eb6834"),
              ("fsvfm", "FS-VFM", "#1baf7a"),
              ("__all__", "Fused (all three)", "#4a3aa7")]
    vals = {k: [model_auc(arm, ds, k) for ds in OOD] for k, _, _ in series}
    means = {k: float(np.nanmean(v)) for k, v in vals.items()}

    fig, (ax, axm) = plt.subplots(1, 2, figsize=(14.0, 4.6), facecolor=SURFACE,
                                  gridspec_kw={"width_ratios": [3.5, 1]})
    x = np.arange(len(OOD)); w = 0.2
    for i, (k, name, col) in enumerate(series):
        off = (i - 1.5) * w
        lw = 2.2 if k == "__all__" else 0
        ax.bar(x + off, vals[k], w * 0.92, label=name, color=col,
               edgecolor=SURFACE, linewidth=2, zorder=3)
    # mark which single view wins each dataset -- the point of the figure
    for j, ds in enumerate(OOD):
        best = max(("semantic", "artifact", "fsvfm"), key=lambda k: vals[k][j])
        bi = [k for k, _, _ in series].index(best)
        ax.text(x[j] + (bi - 1.5) * w, vals[best][j] + 0.006, "★", ha="center",
                fontsize=11, color=INK2, zorder=4)
    ax.set_facecolor(SURFACE)
    ax.set_xticks(x); ax.set_xticklabels([NICE[d] for d in OOD], fontsize=9.5, rotation=18,
                                         ha="right")
    ax.set_ylim(0.62, 1.0); ax.set_ylabel("video AUROC (zero-shot)", fontsize=10, color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.7, alpha=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.legend(frameon=False, fontsize=9.5, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, 1.14))
    # above the plot, clear of the shortest bars (Deepfake-Eval-2024 sits near the floor)
    ax.text(0.0, 1.005, "★ best single view", transform=ax.transAxes, fontsize=9,
            color=INK2, ha="left", va="bottom")

    axm.set_facecolor(SURFACE)
    ks = [k for k, _, _ in series]
    axm.bar(range(4), [means[k] for k in ks],
            0.62, color=[c for _, _, c in series], edgecolor=SURFACE, linewidth=2, zorder=3)
    for i, k in enumerate(ks):
        axm.text(i, means[k] + 0.003, f"{means[k]:.4f}", ha="center", fontsize=9.5, color=INK)
    axm.set_xticks(range(4)); axm.set_xticklabels(["sem", "art", "FS-VFM", "fused"], fontsize=9.5)
    axm.set_ylim(0.62, 1.0); axm.set_yticks([])
    axm.set_title("mean over the six sets", fontsize=10.5, color=INK, pad=8)
    axm.grid(axis="y", color=GRID, lw=0.7, alpha=0.8); axm.set_axisbelow(True)
    for sp in ("top", "right", "left"): axm.spines[sp].set_visible(False)
    axm.spines["bottom"].set_color(GRID); axm.tick_params(colors=INK2, length=0)

    fig.suptitle(f"No single view is reliable everywhere — arm {arm}\n"
                 f"the best single view changes per dataset; the fused output is never far from "
                 f"the top and is best on the mean",
                 fontsize=12.5, color=INK, y=1.06, x=0.008, ha="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig); print(f"  wrote {out}.png / .pdf")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("tbiom/figures"))
    ap.add_argument("--arm", default="C")
    ap.add_argument("--datasets", nargs="+", default=["CDFv2", "DFDCP"])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print("fig1 — whole vs parts")
    fig_whole_vs_parts(args.arm, args.datasets, args.out / f"fig1_whole_vs_parts_arm{args.arm}")
    print("fig2 — arm ablation")
    for ds in args.datasets:
        fig_arm_ablation(ds, args.out / f"fig2_arm_ablation_{ds}")
    print("fig3 — configurations")
    fig_configurations(args.out / "fig3_configurations")
    print("fig4 — view ablation")
    fig_view_ablation(args.arm, args.out / f"fig4_view_ablation_arm{args.arm}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

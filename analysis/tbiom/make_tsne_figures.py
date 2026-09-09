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
# Panel titles. Plain names only -- no mechanism, no attribution. Override per run with
# --names "semantic=X,artifact=Y,fsvfm=Z,fused=W" rather than editing this.
VIEW_NAME = {
    "semantic": "Appearance",
    "artifact": "Off-manifold residual",
    "fsvfm":    "Bona-fide face prior",
    "__all__":  "Fused",
}
VIEWS = [(k, VIEW_NAME[k]) for k in ("semantic", "artifact", "fsvfm", "__all__")]


def load_frames(arm: str, ds: str):
    """npz -> raw FRAME features, no aggregation.

    Frame level shows what the branch actually sees; video level shows what it is scored on.
    Both are worth a panel and they can look different -- video averaging removes the
    within-video spread that dominates a frame plot.
    """
    f = TSNE_DIR / f"f_{arm}_{ds}.npz"
    if not f.is_file():
        return None
    d = np.load(f, allow_pickle=True)
    return ({k: d[k] for k in ("semantic", "artifact", "fsvfm")},
            np.asarray(d["label"]).astype(int))


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


def parse_ratio(spec: str):
    """'1:1' | '1:3' | '1:6' | 'natural' -> (real, fake) target proportions, or None."""
    if spec is None or str(spec).lower() in ("natural", "none", "true", ""):
        return None
    parts = str(spec).split(":")
    if len(parts) != 2:
        raise ValueError(f"ratio must look like '1:3' or be 'natural', got {spec!r}")
    r, f = (float(x) for x in parts)
    if r <= 0 or f <= 0:
        raise ValueError(f"ratio parts must be positive, got {spec!r}")
    return r, f


def subsample(y, cap: int, seed=0, ratio="1:1"):
    """Index subsample, for PLOTTING ONLY. Returns indices and a caption describing what was done.

    WHY THIS IS A KNOB AND NOT A CONSTANT. The evaluation sets are wildly fake-heavy --
    Celeb-DF-v3 runs 167,603 fake frames against 5,692 real, roughly 1:29 -- and the two honest
    choices pull in opposite directions:

        natural   shows the distribution a deployed detector actually meets, but the minority
                  class disappears under the majority and the panel reads as imbalance rather
                  than separability
        1:1       shows the geometry clearly and makes panels comparable across datasets with
                  very different ratios, but no evaluation set looks like that

    1:3 or 1:6 sit between: the imbalance is visible, the real class is still readable. Pick per
    figure and state it -- the caption this returns does that automatically, so a reader is
    never guessing which one they are looking at.

    THE METRIC IS NEVER RESAMPLED. Every AUROC annotation is computed on the full, unmodified
    set. Resampling the metric would change the number a reader checks against the results
    table, and on a 1:29 set it would change it a lot.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    idx = {c: np.flatnonzero(y == c) for c in classes}
    have = {c: len(v) for c, v in idx.items()}
    want = parse_ratio(ratio)

    if want is None or len(classes) != 2:
        if len(y) <= cap:
            return np.arange(len(y)), f"all {len(y):,}, natural {have.get(0,0)}:{have.get(1,0)}"
        take = {c: max(1, int(round(cap * n / len(y)))) for c, n in have.items()}
    else:
        r, f = want
        # largest k with k*r <= n_real, k*f <= n_fake, and k*(r+f) <= cap
        k = min(have.get(0, 0) / r, have.get(1, 0) / f, cap / (r + f))
        take = {0: max(1, int(k * r)), 1: max(1, int(k * f))}
    take = {c: min(take[c], have[c]) for c in classes}
    sel = np.sort(np.concatenate([rng.choice(idx[c], size=take[c], replace=False)
                                  for c in classes]))
    nr, nf = take.get(0, 0), take.get(1, 0)
    # the achieved ratio as 1:x -- reducing by gcd gave unreadable labels like "171:1028"
    shown = f"1:{nf / nr:.3g}" if nr else "0:1"
    return sel, f"{nr:,}R / {nf:,}F  ({shown})"


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


def fig_whole_vs_parts(arm, datasets, out: Path, seed=0, cap=1200, level="video",
                       ratio="1:1", title=""):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    LOAD = load_videos if level == "video" else load_frames
    rows = [(ds, LOAD(arm, ds)) for ds in datasets]
    rows = [(ds, r) for ds, r in rows if r]
    if not rows:
        print("  no data"); return
    fig, axes = plt.subplots(len(rows), 4, figsize=(15.2, 3.6 * len(rows)),
                             facecolor=SURFACE, squeeze=False)
    for i, (ds, (fe, y)) in enumerate(rows):
        sel, cap_txt = subsample(y, cap, seed, ratio)
        for j, (key, name) in enumerate(VIEWS):
            X = joint(fe) if key == "__all__" else zs(fe[key])
            panel(axes[i][j], embed(X[sel], seed), y[sel],
                  name if i == 0 else "", model_auc(arm, ds, key),
                  show_legend=(i == 0 and j == 0))
        unit = "videos" if level == "video" else "frames"
        n = f"{ds}\n{len(y):,} {unit}\n{cap_txt}"
        axes[i][0].set_ylabel(n, fontsize=10.5, color=INK, labelpad=10)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", frameon=False, fontsize=10, ncol=2,
               bbox_to_anchor=(0.995, 0.995))
    fig.suptitle(f"Each view alone, then all three fused — arm {arm} · {level} level\n"
                 f"AUROC is the trained head's zero-shot cross-dataset VIDEO AUROC on the full "
                 f"set (trained on FF++ c23 only); points sampled at real:fake = {ratio}",
                 fontsize=12.5, color=INK, y=1.0, x=0.012, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig); print(f"  wrote {out}.png / .pdf")


def fig_arm_ablation(ds, out: Path, seed=0, cap=1200, ratio="1:1", level="video", title=""):
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
        sel, _ = subsample(y, cap, seed, ratio)
        panel(axes[0][j], embed(X[sel], seed), y[sel], ARMS[a], model_auc(a, ds, "__all__"),
              show_legend=(j == 0))
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", frameon=False, fontsize=10, ncol=2,
               bbox_to_anchor=(0.995, 0.995))
    if title:
        fig.suptitle(title, fontsize=11.5, color=INK, y=1.0, x=0.008, ha="left")
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

    fig, ax = plt.subplots(figsize=(9.2, 3.9), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y = np.arange(len(order))
    ax.barh(y, [m - P0 for m in means], height=0.52, left=P0,
            color=C_REAL, edgecolor=SURFACE, linewidth=2)
    ax.errorbar([m for m in means], y, xerr=[lo, hi], fmt="none",
                ecolor=INK2, elinewidth=1.4, capsize=4)
    ax.axvline(P0, color=INK2, lw=1.4, ls="--")
    # above the bars, not below: the axis is inverted, so len(order)-0.35 landed on the
    # x-tick labels and the caption sat on top of them
    ax.text(P0, -0.72, f" two-view baseline {P0:.4f}", fontsize=9.5, color=INK2,
            va="center", ha="left")
    # to the right of the error bar's upper cap, not the mean -- at the mean the text runs
    # straight through the cap, which is what the first render did
    for i, (m, c, h) in enumerate(zip(means, order, hi)):
        ax.text(m + h + 0.0014, i, f"{m:.4f}   {m-P0:+.4f}", va="center",
                fontsize=9.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlim(P0 - 0.0035, max(m + h for m, h in zip(means, hi)) + 0.0075)
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


def fig_edl_vs_ce(datasets, out: Path, seed=0, cap=1200, level="video",
                  edl_arm="C", ce_arm="D", ratio="1:1", title=""):
    """EDL and CE side by side: all three branches AND the fused output, for both losses.

    Eight columns per dataset row -- appearance / off-manifold / prior / fused under EDL, then
    the same four under CE. The point is visible without reading a number: CE's two CLIP
    branches are the ones that decay (0.798 against EDL's 0.842 and 0.851 averaged over the six
    sets), while the frozen prior is unaffected because nothing about it is trained.

    ARMS. `edl_arm="C"` against `ce_arm="D"` is the ONE-VARIABLE pairing -- identical branches,
    identical FSFM Branch 3, identical schedule and seed, differing only in heads + fusion +
    loss. Passing `edl_arm="B"` gives the SELECTED configuration instead, which scores higher
    (0.8729 against C's 0.8620) but also swaps Branch 3, so any difference then mixes the loss
    with the branch and it is no longer an ablation. The default is the clean one.
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    LOAD = load_videos if level == "video" else load_frames
    keys = ["semantic", "artifact", "fsvfm", "__all__"]
    rows = []
    for ds in datasets:
        got = {a: LOAD(a, ds) for a in (edl_arm, ce_arm)}
        if all(got.values()):
            rows.append((ds, got))
    if not rows:
        print("  no data"); return
    fig, axes = plt.subplots(len(rows), 8, figsize=(26.0, 3.5 * len(rows)),
                             facecolor=SURFACE, squeeze=False)
    for i, (ds, got) in enumerate(rows):
        for blk, arm in enumerate((edl_arm, ce_arm)):
            fe, y = got[arm]
            sel, cap_txt = subsample(y, cap, seed, ratio)
            for j, k in enumerate(keys):
                X = joint(fe) if k == "__all__" else zs(fe[k])
                col = blk * 4 + j
                title = ("" if i else f"{VIEW_NAME[k]}")
                panel(axes[i][col], embed(X[sel], seed), y[sel], title,
                      model_auc(arm, ds, k), show_legend=(i == 0 and col == 0))
        unit = "videos" if level == "video" else "frames"
        axes[i][0].set_ylabel(f"{ds}\n{len(got[edl_arm][1]):,} {unit}\n{cap_txt}",
                              fontsize=10.5, color=INK, labelpad=10)
    # block headers, so nobody has to count columns to know which half is which
    for blk, (lab, x0) in enumerate((("EDL", 0.055), ("CE", 0.545))):
        fig.text(x0, 0.995, lab, fontsize=12.5, color=INK, ha="left", va="bottom",
                 fontweight="bold")
    fig.text(0.5, 0.982, "│", fontsize=13, color=GRID, ha="center", va="bottom")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", frameon=False, fontsize=10.5, ncol=2,
               bbox_to_anchor=(0.998, 1.0))
    if title:
        fig.suptitle(title, fontsize=11.5, color=INK, y=1.02, x=0.006, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=180, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig); print(f"  wrote {out}.png / .pdf")


def fig_edl_vs_ce_bars(datasets, out: Path):
    """Per-branch and fused AUROC, EDL against CE, every dataset. The quantitative companion."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    NICE = {"CDFv2": "Celeb-DF-v2", "CDFv3": "Celeb-DF-v3", "DFD": "DFD", "DFDC": "DFDC",
            "DFDCP": "DFDCP", "DFEval24": "Deepfake-Eval-2024"}
    keys = [("semantic", "Semantic"), ("artifact", "Artifact"), ("fsvfm", "FS-VFM"),
            ("__all__", "Fused")]
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.9), facecolor=SURFACE, sharey=True)
    x = np.arange(len(datasets)); w = 0.36
    for ax, (k, name) in zip(axes, keys):
        e = [model_auc("C", d, k) for d in datasets]
        c = [model_auc("D", d, k) for d in datasets]
        ax.bar(x - w / 2, e, w * 0.92, label="EDL", color="#4a3aa7",
               edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.bar(x + w / 2, c, w * 0.92, label="CE", color="#eb6834",
               edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.set_facecolor(SURFACE)
        d = np.nanmean(e) - np.nanmean(c)
        ax.set_title(f"{name}\nEDL {np.nanmean(e):.4f}  ·  CE {np.nanmean(c):.4f}"
                     f"   ({d:+.4f})", fontsize=10.5, color=INK, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([NICE.get(d, d) for d in datasets], fontsize=8.5, rotation=30,
                           ha="right")
        ax.grid(axis="y", color=GRID, lw=0.7, alpha=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, length=0)
    axes[0].set_ylim(0.60, 1.0)
    axes[0].set_ylabel("video AUROC (zero-shot)", fontsize=10, color=INK2)
    # outside the axes: inside, it landed on the Celeb-DF-v2 bars
    axes[0].legend(frameon=False, fontsize=9.5, loc="upper left",
                   bbox_to_anchor=(0.0, -0.30), ncol=2)
    fig.suptitle("Evidential supervision keeps the branches healthy; its FUSION does not convert "
                 "the advantage\n"
                 "EDL holds the two CLIP branches ~0.05 above CE, yet CE's fused output is "
                 "marginally ahead in this pairing (arm C vs arm D, same Branch 3)",
                 fontsize=12.0, color=INK, y=1.09, x=0.006, ha="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig); print(f"  wrote {out}.png / .pdf")


def fig_five_panel(datasets, out: Path, seed=0, cap=1500, level="video",
                   edl_arm="C", ce_arm="D", ratio="1:3", title=""):
    """Five panels: the three branches, then CE fusion, then EDL fusion.

    WHICH BRANCHES ARE SHOWN, because the honest answer is not obvious. The three branch panels
    come from the EDL arm -- they are the framework's own views. The CE fusion panel comes from
    a DIFFERENT model (arm D), whose branches were trained under cross-entropy and are weaker
    (0.798 against EDL's 0.842/0.851 averaged over the six sets). So the CE panel is not "the
    same branches fused differently"; it is the whole CE model's output, which is the right
    comparison for choosing an objective but must not be read as an operator swap on fixed
    inputs. The subtitle says so.

    Panel order puts the two fusions last and adjacent, so the comparison a reader is meant to
    make is the one their eye makes.
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    LOAD = load_videos if level == "video" else load_frames
    spec = [(edl_arm, "semantic"), (edl_arm, "artifact"), (edl_arm, "fsvfm"),
            (ce_arm, "__all__"), (edl_arm, "__all__")]
    titles = [VIEW_NAME["semantic"], VIEW_NAME["artifact"], VIEW_NAME["fsvfm"],
              "CE fusion", "EDL fusion"]
    rows = []
    for ds in datasets:
        got = {a: LOAD(a, ds) for a in {edl_arm, ce_arm}}
        if all(got.values()):
            rows.append((ds, got))
    if not rows:
        print("  no data"); return
    fig, axes = plt.subplots(len(rows), 5, figsize=(18.0, 3.7 * len(rows)),
                             facecolor=SURFACE, squeeze=False)
    for i, (ds, got) in enumerate(rows):
        sel_cache = {}
        for j, (arm, key) in enumerate(spec):
            fe, y = got[arm]
            if arm not in sel_cache:
                sel_cache[arm] = subsample(y, cap, seed, ratio)
            sel, cap_txt = sel_cache[arm]
            X = joint(fe) if key == "__all__" else zs(fe[key])
            panel(axes[i][j], embed(X[sel], seed), y[sel], titles[j] if i == 0 else "",
                  model_auc(arm, ds, key), show_legend=(i == 0 and j == 0))
            if j == 4:      # ring the framework's own output so the eye lands there
                for sp in axes[i][j].spines.values():
                    sp.set_color("#4a3aa7"); sp.set_linewidth(2.0)
        unit = "videos" if level == "video" else "frames"
        axes[i][0].set_ylabel(f"{ds}\n{len(got[edl_arm][1]):,} {unit}\n{sel_cache[edl_arm][1]}",
                              fontsize=10.5, color=INK, labelpad=10)
    h, l = axes[0][0].get_legend_handles_labels()
    # under the title block, not beside it -- the subtitle wraps to two lines and the legend
    # was landing on top of its right end
    fig.legend(h, l, loc="lower right", frameon=False, fontsize=10.5, ncol=2,
               bbox_to_anchor=(0.997, 1.005))
    if title:
        fig.suptitle(title, fontsize=11.5, color=INK, y=1.02, x=0.006, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=190, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig); print(f"  wrote {out}.png / .pdf")


def _edl_vs_ce_dispatch(args):
    tag = f"{args.level}_{args.ratio.replace(':', '-')}"
    if args.per_dataset:
        for ds in args.datasets:
            fig_edl_vs_ce([ds], args.out / f"fig5_edl_vs_ce_{ds}_{tag}", cap=args.cap,
                          level=args.level, edl_arm=args.edl_arm, ratio=args.ratio)
    else:
        fig_edl_vs_ce(args.datasets, args.out / f"fig5_edl_vs_ce_{tag}", cap=args.cap,
                      level=args.level, edl_arm=args.edl_arm, ratio=args.ratio)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("tbiom/figures"))
    ap.add_argument("--arm", default="C")
    ap.add_argument("--datasets", nargs="+",
                    default=["CDFv2", "CDFv3", "DFD", "DFDC", "DFDCP", "DFEval24"],
                    help="the six OOD evaluation sets by default")
    ap.add_argument("--level", default="video", choices=["video", "frame"])
    ap.add_argument("--edl-arm", default="C", help="C is the one-variable pairing with D")
    ap.add_argument("--cap", type=int, default=1200, help="max points plotted per panel")
    ap.add_argument("--names", default=None,
                    help="override panel titles, e.g. "
                         "\"semantic=View A,artifact=View B,fsvfm=View C,fused=Ours\"")
    ap.add_argument("--title", default="", help="figure title; empty means no title at all")
    ap.add_argument("--five-panel", action="store_true",
                    help="one figure of five panels: the three branches, CE fusion, EDL fusion")
    ap.add_argument("--per-dataset", action="store_true",
                    help="one file per dataset instead of one multi-row file. fig2 is always "
                         "per-dataset; this splits fig1 and fig5 the same way.")
    ap.add_argument("--ratio", default="1:1",
                    help="real:fake sampling for the PLOT only -- '1:1', '1:3', '1:6', "
                         "or 'natural' for the true distribution. Never affects the AUROC.")
    ap.add_argument("--edl-vs-ce", action="store_true",
                    help="also build the evidential-versus-cross-entropy comparison")
    ap.add_argument("--only-edl-vs-ce", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.names:
        alias = {"fused": "__all__"}
        for kv in args.names.split(","):
            k, _, v = kv.partition("=")
            k = k.strip(); VIEW_NAME[alias.get(k, k)] = v.strip()
    tag5 = f"{args.level}_{args.ratio.replace(':', '-')}"
    if args.five_panel:
        print("fig7 — five-panel: three branches + CE fusion + EDL fusion")
        if args.per_dataset:
            for ds in args.datasets:
                fig_five_panel([ds], args.out / f"fig7_five_panel_{ds}_{tag5}", cap=args.cap,
                               level=args.level, edl_arm=args.edl_arm, ratio=args.ratio,
                               title=args.title)
        else:
            fig_five_panel(args.datasets, args.out / f"fig7_five_panel_{tag5}", cap=args.cap,
                           level=args.level, edl_arm=args.edl_arm, ratio=args.ratio,
                           title=args.title)
        return 0
    if args.only_edl_vs_ce:
        print("fig5 — EDL vs CE, t-SNE")
        _edl_vs_ce_dispatch(args)
        print("fig6 — EDL vs CE, per-branch bars")
        fig_edl_vs_ce_bars(args.datasets, args.out / "fig6_edl_vs_ce_bars")
        return 0
    tag = f"{args.level}_{args.ratio.replace(':', '-')}"
    if args.per_dataset:
        print("fig1 — whole vs parts, one file per dataset")
        for ds in args.datasets:
            fig_whole_vs_parts(args.arm, [ds],
                               args.out / f"fig1_whole_vs_parts_arm{args.arm}_{ds}_{tag}",
                               cap=args.cap, level=args.level, ratio=args.ratio)
    else:
        print("fig1 — whole vs parts, all sets in one file")
        fig_whole_vs_parts(args.arm, args.datasets,
                           args.out / f"fig1_whole_vs_parts_arm{args.arm}_{tag}",
                           cap=args.cap, level=args.level, ratio=args.ratio)
    print("fig2 — arm ablation, per dataset")
    for ds in args.datasets:
        fig_arm_ablation(ds, args.out / f"fig2_arm_ablation_{ds}_{args.ratio.replace(':','-')}",
                         cap=args.cap, ratio=args.ratio, level=args.level)
    print("fig3 — configurations")
    fig_configurations(args.out / "fig3_configurations")
    print("fig4 — view ablation")
    fig_view_ablation(args.arm, args.out / f"fig4_view_ablation_arm{args.arm}")
    if args.edl_vs_ce:
        print("fig5 — EDL vs CE, t-SNE")
        _edl_vs_ce_dispatch(args)
        print("fig6 — EDL vs CE, per-branch bars")
        fig_edl_vs_ce_bars(args.datasets, args.out / "fig6_edl_vs_ce_bars")
    return 0


if __name__ == "__main__":
    sys.exit(main())

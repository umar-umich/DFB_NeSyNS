#!/usr/bin/env python3
"""The manuscript's data figures, generated from the committed reports.

    OMP_NUM_THREADS=4 python analysis/tbiom/paper/build_figures.py

Two figures, both plotting measured values only:
  fig_df40.pdf   per-generator AUROC of the proposed system against the baseline, split by family, with the
                 near-chance reenactment generators marked. This is the paper's headline evidence.
  fig_complementarity.pdf
                 per DF40 entry, the temporal view's AUROC against the better of the two spatial views. The shaded
                 band is where both spatial views are near chance and only motion evidence remains.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, OUT = Path("tbiom/reports"), Path("tbiom/paper/figures")
CFG, OURS = "42", "PCR6P-R"
# IEEE requires embedded fonts in PDF figures and suggests Times New Roman, Helvetica, Arial, Cambria or Symbol
# (Transactions-instructions, Sec. H). matplotlib's default PDF export uses Type 3 fonts, which the IEEE Graphics
# Analyzer flags; fonttype 42 embeds TrueType instead.
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "figure.dpi": 200,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42,
                     "font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"]})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    E = json.load(open(R / "EXP_PCR6P_EXPOSED.json"))[CFG]
    D = json.load(open(R / "EXP_DF40_CROSS_MANIPULATION.json"))
    meta = json.load(open("logs/tbiom/df40/meta.json"))
    g = lambda n: float(np.mean(E["df40"][OURS][n]))

    # ------------------------------------------------------------------ figure 1: per generator
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [19, 22]})
    for ax, fam, title in zip(axes, ("face swap", "reenactment"), ("Face swap", "Face reenactment")):
        ns = sorted([n for n in E["df40"][OURS] if meta[n]["family"] == fam],
                    key=lambda n: g(n))
        x = np.arange(len(ns))
        ours = [g(n) for n in ns]
        dic = [D[n]["dicome"] for n in ns]
        chance = [max(D[n]["view_APP"], D[n]["view_OM"]) < 0.62 for n in ns]
        ax.vlines(x, dic, ours, color="0.75", lw=1.0, zorder=1)
        ax.scatter(x, dic, s=14, marker="o", facecolor="white", edgecolor="0.35", lw=0.7,
                   label="baseline (DiCoME)", zorder=2)
        ax.scatter(x, ours, s=16, marker="D", color="#1f4e79", label="ours (PCR6$^+$-R)", zorder=3)
        if any(chance):
            ax.scatter(x[np.array(chance)], np.array(ours)[np.array(chance)], s=52, marker="D",
                       facecolor="none", edgecolor="#c0504d", lw=1.0, zorder=4,
                       label="both spatial views $<0.62$")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{meta[n]['method']}-{meta[n]['source']}" for n in ns],
                           rotation=90, fontsize=5.2)
        ax.set_ylim(0.40, 1.02)
        ax.set_title(title, fontsize=8)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
    axes[0].set_ylabel("video AUROC")
    axes[1].legend(frameon=False, fontsize=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_df40.pdf", bbox_inches="tight")
    print("wrote", OUT / "fig_df40.pdf")

    # ------------------------------------------------------------------ figure 2: complementarity structure
    # Per DF40 entry: how informative the temporal view is against the better of the two spatial views. Every value
    # is a measured per-view AUROC; nothing is interpolated. The lower-right region is where a spatial-only system
    # has nothing to read and the temporal view does.
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    for fam, mk, col, lab in (("face swap", "o", "0.45", "face swap"),
                              ("reenactment", "D", "#1f4e79", "reenactment")):
        ns = [n for n in E["df40"][OURS] if meta[n]["family"] == fam]
        xs = [max(D[n]["view_APP"], D[n]["view_OM"]) for n in ns]
        ys = [D[n]["view_T"] for n in ns]
        ax.scatter(xs, ys, s=20, marker=mk, facecolor="none", edgecolor=col, lw=0.9, label=lab)
        for n, x, y in zip(ns, xs, ys):
            if x < 0.62:
                ax.annotate(meta[n]["method"], (x, y), fontsize=5, xytext=(3, -1),
                            textcoords="offset points", color="#c0504d")
    ax.plot([0.4, 1.0], [0.4, 1.0], lw=0.6, ls="--", color="0.7")
    ax.axvspan(0.40, 0.62, color="#c0504d", alpha=0.07, lw=0)
    ax.set_xlabel("best spatial view (AUROC)")
    ax.set_ylabel("temporal view (AUROC)")
    ax.set_xlim(0.40, 1.02); ax.set_ylim(0.40, 1.02)
    ax.grid(lw=0.3, alpha=0.5)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_complementarity.pdf", bbox_inches="tight")
    print("wrote", OUT / "fig_complementarity.pdf")


if __name__ == "__main__":
    main()

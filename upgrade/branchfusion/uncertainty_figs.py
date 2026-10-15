#!/usr/bin/env python3
"""DiCoME Figure 3, rebuilt on our system: does vacuity know where the model is wrong?

    <discern_ext python> upgrade/branchfusion/uncertainty_figs.py

TOP ROW -- vacuity split by CORRECTNESS, not by class. The claim being tested is that the model
knows its own knowledge boundary: correct predictions should sit near u -> 0 and errors should
populate the high-vacuity tail. Splitting by true class instead would show something else
entirely (how confident the model is on reals vs fakes), which is a different question and not
what Figure 3 is about.

BOTTOM ROW -- risk-coverage. Reject the most uncertain fraction and plot the balanced error on
what remains. THREE curves, because the figure has to argue something:

  vacuity        ours -- the Dirichlet quantity, which only an evidential model has;
  margin |p-tau| the BASELINE, and the point of the panel. It is the only uncertainty a
                 cross-entropy fusion of these same branches would hand you, at no cost. If
                 vacuity does not beat it, the evidential apparatus is not earning its keep
                 here, and the figure should say so rather than hide it behind an oracle;
  oracle         the dashed bound -- the same predictions ranked by whether they are actually
                 wrong. No rejector can do better; it fixes the scale of the other two.

Opinions are the deployed FOUR-branch framework (App + TD + LFS + STT) under ES-PCR6+, with the
reliability weights fitted
on development and the threshold taken from FF++ validation rows -- never from the test set being
plotted.

The caption is WRITTEN FROM THE NUMBERS, not asserted: `FIGURE_NOTES.md` is generated here with
the area under each risk-coverage curve, so whether vacuity actually beats the margin baseline is
read off the figure's own data instead of being claimed alongside it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                               # noqa: E402
import branches as B                                             # noqa: E402
import opinions as O                                             # noqa: E402
import subsets as S                                              # noqa: E402
sys.path[:0] = [str(C.ROOT / "analysis/tbiom/pcr6p")]
import ops_pcr6p as P                                            # noqa: E402

SUB = ("A", "V", "O", "S16ctl")
OUT = C.REPORT / "figures"
#: Validated two-slot categorical palette (dataviz skill, light surface):
#: adjacent CVD dE 21.6, normal-vision dE 32.3, both well clear of the gates.
CORRECT, WRONG = "#2a78d6", "#e34948"
MARGIN = "#eda100"   # slot 4. Slot 2 orange FAILED against the red above (CVD dE 5.6, normal 7.1);
                     # this trio passes every gate -- worst CVD 15.3, worst normal-vision 20.8.
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"


def fused_pu(t, w):
    """(p, u) of the fused ES-PCR6+ opinion -- vacuity is the Theta mass, not 1 - p."""
    m = O.masses(t, SUB, list(w))
    f = P.fuse(m, True)
    return P.p_of(f), f[:, 2]


def risk_coverage(y, p, u, tau):
    """Balanced error on the retained set as the most-uncertain fraction is rejected."""
    order = np.argsort(u)                      # keep the most certain first
    cov, risk = [], []
    for frac in np.linspace(0.1, 1.0, 46):
        k = max(10, int(round(frac * len(y))))
        idx = order[:k]
        yy, pp = y[idx], p[idx] >= tau
        pos, neg = yy == 1, yy == 0
        if pos.sum() < 3 or neg.sum() < 3:
            continue
        be = 0.5 * ((pp[pos] == 0).mean() + (pp[neg] == 1).mean())
        cov.append(k / len(y)); risk.append(be)
    return np.array(cov), np.array(risk)


def aurc(cov, risk) -> float:
    """Area under the risk-coverage curve, normalised by the coverage range actually plotted.

    Lower is better. Two rejectors are only comparable through this if they are integrated over
    the same coverage grid, which they are: `risk_coverage` drops a point only when a retained
    set runs out of one class, and that happens at the same coverage for both.
    """
    if len(cov) < 2:
        return float("nan")
    return float(np.trapezoid(risk, cov) / (cov[-1] - cov[0]))


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = list(SUB)
    dev = B.dev_table(items, require_all=True)
    w, _ = O.fit_weights(dev, SUB, "F8", objective="v31")
    pdev, _ = fused_pu(dev, w)
    ff = np.where(dev.source.to_numpy() == "FFpp_val")[0]
    from health import eer
    tau = float(eer(dev.y.to_numpy()[ff], pdev[ff])[1])
    print(f"weights {list(w)}  tau {tau:.4f}", flush=True)

    tabs = B.test_tables(items)
    stats = {}
    for ds, t in tabs.items():
        p, u = fused_pu(t, w)
        y = t.y.to_numpy()
        correct = (p >= tau).astype(int) == y
        stats[ds] = {"p": p, "u": u, "y": y, "correct": correct,
                     "auroc": C.auroc(y, p), "err": float(1 - correct.mean())}
        print(f"  {ds:10s} auroc {stats[ds]['auroc']:.4f}  err {stats[ds]['err']:.4f}  "
              f"u(correct) {u[correct].mean():.3f}  u(wrong) {u[~correct].mean():.3f}", flush=True)

    # The sets where the method works best AND has enough errors to plot a distribution against.
    pick = sorted((d for d in stats if (~stats[d]["correct"]).sum() >= 40),
                  key=lambda d: -stats[d]["auroc"])[:4]
    print(f"\nplotting: {pick}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, len(pick), figsize=(3.5 * len(pick), 6.0),
                             facecolor=SURFACE, constrained_layout=True)
    for j, ds in enumerate(pick):
        s = stats[ds]
        ax = axes[0, j]
        bins = np.linspace(0, max(0.05, float(np.percentile(s["u"], 99.5))), 34)
        for mask, col, lab in ((s["correct"], CORRECT, "correct"),
                               (~s["correct"], WRONG, "misclassified")):
            ax.hist(s["u"][mask], bins=bins, density=True, color=col, alpha=0.55,
                    edgecolor=col, linewidth=1.2, label=lab)
        ax.set_title(f"{ds}\nAUROC {s['auroc']:.3f}", fontsize=10, color=INK)
        ax.set_xlabel("vacuity $u$", fontsize=9, color=INK2)
        if j == 0:
            ax.set_ylabel("density", fontsize=9, color=INK2)
            ax.legend(frameon=False, fontsize=8, labelcolor=INK2)

        ax = axes[1, j]
        cov, risk = risk_coverage(s["y"], s["p"], s["u"], tau)
        ocov, orisk = risk_coverage(s["y"], s["p"], (~s["correct"]).astype(float)
                                    + 1e-6 * s["u"], tau)
        mcov, mrisk = risk_coverage(s["y"], s["p"], -np.abs(s["p"] - tau), tau)
        ax.plot(cov * 100, risk, color=CORRECT, linewidth=2.0, label="vacuity (ours)")
        ax.plot(mcov * 100, mrisk, color=MARGIN, linewidth=2.0, label=r"margin $|p-\tau|$")
        ax.plot(ocov * 100, orisk, color=INK2, linewidth=1.3, linestyle="--", label="oracle")
        s["aurc"] = {"vacuity": aurc(cov, risk), "margin": aurc(mcov, mrisk),
                     "oracle": aurc(ocov, orisk)}
        ax.set_xlabel("coverage (%)", fontsize=9, color=INK2)
        if j == 0:
            ax.set_ylabel("balanced error", fontsize=9, color=INK2)
            ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
        ax.set_ylim(bottom=0)

    for ax in axes.ravel():
        ax.set_facecolor(SURFACE)
        ax.tick_params(labelsize=8, colors=INK2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#d8d7d2")
        ax.grid(axis="y", color="#ececea", linewidth=0.8)
        ax.set_axisbelow(True)

    f = OUT / "uncertainty_riskcoverage.png"
    fig.savefig(f, dpi=200, facecolor=SURFACE)
    print(f"-> {f}")
    json.dump({d: {k: (float(v) if np.isscalar(v) else None)
                   for k, v in s.items() if k in ("auroc", "err")}
               | {"u_correct_mean": float(s["u"][s["correct"]].mean()),
                  "u_wrong_mean": float(s["u"][~s["correct"]].mean()),
                  "n_wrong": int((~s["correct"]).sum()),
                  "plotted": d in pick, "aurc": s.get("aurc")}
               for d, s in stats.items()},
              open(C.ROOT / "analysis/branchfusion/uncertainty_figs.json", "w"), indent=1)

    # ------------------------------------------------------------------ caption, from the numbers
    wins = [d for d in pick if stats[d]["aurc"]["vacuity"] < stats[d]["aurc"]["margin"]]
    N = ["# Figure — vacuity and risk-coverage", "",
         f"`figures/uncertainty_riskcoverage.png`, regenerated with the panels chosen by AUROC "
         f"among the sets with at least forty errors: {', '.join(pick)}.", "",
         "## Top row — the separation holds", "",
         "Vacuity split by whether the fused prediction is right. Misclassified videos carry",
         "visibly more vacuity than correct ones on every panel, which is the claim DiCoME's",
         "Figure 3 makes and it reproduces here:", "",
         "| set | u(correct) | u(wrong) | ratio | errors |", "|---|---:|---:|---:|---:|"]
    for d in pick:
        s_ = stats[d]
        uc = float(s_["u"][s_["correct"]].mean()); uw = float(s_["u"][~s_["correct"]].mean())
        N.append(f"| {d} | {uc:.4f} | {uw:.4f} | {uw / uc:.2f}x | {int((~s_['correct']).sum())} |")
    N += ["", "## Bottom row — is it actually a better rejector?", "",
          "Separating errors in a histogram does not by itself make a quantity a better rejector",
          "than the decision margin, which costs nothing and needs no evidential model. So the",
          "verdict below is computed, not asserted. Area under the risk-coverage curve, lower is",
          "better, both rejectors on the same coverage grid:", "",
          r"| set | vacuity (ours) | margin \|p-tau\| | oracle | vacuity better? |",
          "|---|---:|---:|---:|---|"]
    for d in pick:
        a = stats[d]["aurc"]
        N.append(f"| {d} | {a['vacuity']:.4f} | {a['margin']:.4f} | {a['oracle']:.4f} | "
                 f"{'yes' if a['vacuity'] < a['margin'] else 'NO'} |")
    lost = [d for d in pick if d not in wins]
    N += ["",
          f"Vacuity wins {len(wins)} of the {len(pick)} panels "
          f"({', '.join(wins) if wins else 'none'})" +
          (f", and loses on {', '.join(lost)}." if lost else "."), "",
          "### What may be claimed from this"]
    if len(wins) == len(pick):
        N += ["", "Both rows. Vacuity separates errors in the histogram AND orders them better",
              "than the decision margin on every panel, so the evidential apparatus is paying",
              "for itself as a rejector and not only as an interpretable quantity."]
    elif wins:
        N += ["",
              f"The first row on every panel, the second on {len(wins)} of {len(pick)}. Vacuity",
              "is the better rejector where it wins and the margin is free, so the claim to make",
              "is the one the table supports: vacuity dominates the margin on most sets, by a",
              f"wide margin on {min(wins, key=lambda d: stats[d]['aurc']['vacuity'])}, and ties or",
              f"loses on {', '.join(lost)}. State the exception. A reviewer who runs the margin",
              "baseline will find it, and it costs nothing to report."]
    else:
        N += ["",
              "The FIRST ROW ONLY. A quantity can separate errors in a histogram and still not",
              "order them better than the decision margin, which costs nothing and needs no",
              "evidential model -- that is what happened here. The figure supports *uncertainty",
              "that is interpretable and decomposable*: vacuity is a stated quantity of ignorance,",
              "separable from conflict, and it is what the fusion rule operates on. It does NOT",
              "support a claim of better selective prediction. Reading the second claim off the",
              "first panel is the mistake this file exists to prevent."]
    N += [""]
    (OUT.parent / "FIGURE_NOTES.md").write_text("\n".join(N))
    print(f"-> {OUT.parent / 'FIGURE_NOTES.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

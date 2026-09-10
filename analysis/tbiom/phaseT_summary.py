#!/usr/bin/env python3
"""Phase T — did the symbolic coherence term buy anything?

    python analysis/tbiom/phaseT_summary.py --out tbiom/PHASE_T.md

ONE VARIABLE. phaseTc01 and phaseTc05 are arm B seed 42 with `lambda_cons` set to 0.1 and 0.5.
Everything else -- branches, fusion, aux EDL, data, seed, schedule -- is byte-identical to the
frozen run. The comparison is therefore matched, not a leaderboard entry.

TWO QUESTIONS, ASKED SEPARATELY, because they can come apart:

  1. Did detection change?   Mean OOD AUROC and real-side FPR against arm B seed 42.
  2. Did the term BITE?      The coherence quantity itself, measured on OOD. A loss that moves
                             no structure explains nothing whatever the AUROC does, and a term
                             that improves structure at no AUROC cost is a different -- and
                             still reportable -- outcome from one that does nothing at all.

Question 2 is what makes this more than a hyperparameter sweep. If lambda_cons visibly raises
cross-branch agreement and lowers confident-under-conflict mass, then the symbolic constraint
IS shaping the evidence, and the AUROC result says whether that shaping is worth anything.

FUSION CONVENTION. Probabilities are the DS replay of the three exported branch opinions, not
the exported `p_fused` column. The two differ slightly (0.8768 vs 0.8729 on arm B seed 42)
because the replay recomputes the orthogonal sum from the per-branch alphas while `p_fused` is
the model's own head output. The replay is what Phase F, the seed summary and the overlay all
use, so this table is on the same convention as every other number in the writeup. Mixing the
two across arms is exactly the CDFv3 error from the pilot table, on a smaller scale.

SIGNIFICANCE. Section 22's rule holds: |dAUROC| < 0.01 is inside the band a single seed can
produce (arm B's own three-seed range is 0.0025 on the mean, but per-dataset spreads reach
0.011). Nothing under that threshold is claimed without a confirming seed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from health import dashboard, eer  # noqa: E402
import symbolic_overlay as SO  # noqa: E402

S23 = Path("logs/tbiom/stage23")
OOD = SO.OOD
BR = SO.BRANCHES
K = 2
ARMS = [("arm B (lambda_cons=0)", "run1auxedl"),
        ("lambda_cons=0.1", "phaseTc01"),
        ("lambda_cons=0.5", "phaseTc05")]
P0 = {"CDFv2": 0.9646, "CDFv3": 0.8409, "DFD": 0.9421, "DFDC": 0.8828,
      "DFDCP": 0.8573, "DFEval24": 0.6922}


def load(pref: str, ds: str):
    f = S23 / f"{pref}_{ds}.csv"
    if not f.is_file():
        return None
    d = pd.read_csv(f)
    need = [f"{k}_{v}" for v in BR for k in ("p", "u")] + ["p_fused", "u_fused", "label"]
    if any(c not in d for c in need):
        return None
    from video_id import video_id
    d["v"] = video_id(d["key"])
    agg = {c: (c, "mean") for c in need if c != "label"}
    agg["y"] = ("label", "max")
    return d.groupby("v", as_index=False).agg(**agg)


def coherence(g: pd.DataFrame) -> dict:
    """The quantity the training term penalised, measured on held-out data.

    agreement A in [0,1] is the certainty-weighted directional consensus of the three views.
    `conflicted_confident` is the mass the penalty actually charges: mean over samples of
    (1 - A) * fused certainty. Both are read from the exported opinions, so they are the same
    functional form the loss used, evaluated where it was never applied.
    """
    ms, ws = [], []
    for b in BR:
        p, u = g[f"p_{b}"].to_numpy(), np.clip(g[f"u_{b}"].to_numpy(), 1e-9, None)
        ms.append(p - 0.5)
        ws.append(np.clip(1.0 - u, 0, None))
    m, w = np.stack(ms, 1), np.stack(ws, 1)
    num, den = np.abs((w * m).sum(1)), (w * np.abs(m)).sum(1)
    A = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 1.0)
    # fused certainty from the replayed opinion, matching the AUROC convention above
    fa = np.stack([SO.alpha(g[f"p_{b}"].to_numpy(), g[f"u_{b}"].to_numpy()) for b in BR])
    f = fa[0]
    for x in fa[1:]:
        f = SO.ds_pair(f, x)
    cert = np.clip(1.0 - K / f.sum(1), 0, None)
    return {"agreement": float(A.mean()),
            "conflicted_confident": float(((1 - A) * cert).mean()),
            "mean_vacuity": float(g[[f"u_{b}" for b in BR]].to_numpy().mean())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("tbiom/PHASE_T.md"))
    args = ap.parse_args()

    res, missing = {}, []
    for name, pref in ARMS:
        ff = load(pref, "FFpp_val")
        if ff is None:
            missing.append(f"{name}: FF++ val"); continue
        # tau frozen on this run's OWN FF++ val -- the runs are differently calibrated and a
        # shared threshold would compare operating points instead of models.
        _, tau = eer(ff["y"].to_numpy(), SO.fuse(ff))
        a, f_, co = {}, {}, {}
        for ds in OOD:
            g = load(pref, ds)
            if g is None:
                missing.append(f"{name}: {ds}"); continue
            d = dashboard(g["y"].to_numpy(), SO.fuse(g), tau)
            a[ds], f_[ds] = d["auroc"], d["fpr_real_at_tau"]
            co[ds] = coherence(g)
        if len(a) == len(OOD):
            res[name] = {"tau": float(tau), "auroc": a, "fpr": f_, "coh": co,
                         "mean": float(np.mean(list(a.values()))),
                         "mean_fpr": float(np.mean(list(f_.values()))),
                         "mean_agree": float(np.mean([c["agreement"] for c in co.values()])),
                         "mean_cc": float(np.mean([c["conflicted_confident"]
                                                   for c in co.values()]))}

    L = ["# Phase T — the symbolic coherence term", "",
         "Arm B seed 42 with `lambda_cons` added and nothing else changed. The term is the "
         "differentiable relaxation of overlay rule R4: it charges fused certainty in "
         "proportion to how far the certainty-weighted branch votes cancel. It uses no labels "
         "and no thresholds, so unlike the gate's predicates it cannot fit the development "
         "set.", ""]

    if not res:
        L += ["**No exports yet.**", ""] + [f"- missing {m}" for m in missing[:12]]
        args.out.write_text("\n".join(L) + "\n"); print("\n".join(L)); return 0

    base = res.get("arm B (lambda_cons=0)")
    L += ["## 1 — Detection", "",
          "| arm | mean OOD AUROC | vs arm B | mean FPR_real | vs arm B |",
          "|---|---:|---:|---:|---:|"]
    for name, _ in ARMS:
        r = res.get(name)
        if not r:
            L.append(f"| {name} | not exported | | | |"); continue
        if base and name != "arm B (lambda_cons=0)":
            da, df = r["mean"] - base["mean"], r["mean_fpr"] - base["mean_fpr"]
            sig = "**" if abs(da) >= 0.01 else ""
            L.append(f"| {name} | {r['mean']:.4f} | {sig}{da:+.4f}{sig} | "
                     f"{r['mean_fpr']:.3f} | {df:+.3f} |")
        else:
            L.append(f"| {name} | **{r['mean']:.4f}** | — | **{r['mean_fpr']:.3f}** | — |")

    L += ["", "### Per dataset", "",
          "| dataset | P0-DS | " + " | ".join(n for n, _ in ARMS if n in res) + " |",
          "|---" * (2 + len(res)) + "|"]
    for ds in OOD:
        L.append(f"| {ds} | {P0[ds]:.4f} | " +
                 " | ".join(f"{res[n]['auroc'][ds]:.4f}" for n, _ in ARMS if n in res) + " |")

    L += ["", "## 2 — Did the term bite?", "",
          "Measured on the six OOD sets, in the same functional form the loss used. If these "
          "do not move, the term changed nothing and the AUROC column above is noise about "
          "nothing.", "",
          "| arm | cross-branch agreement | conflicted-confident mass | mean branch vacuity |",
          "|---|---:|---:|---:|"]
    for name, _ in ARMS:
        r = res.get(name)
        if not r:
            continue
        mv = float(np.mean([c["mean_vacuity"] for c in r["coh"].values()]))
        L.append(f"| {name} | {r['mean_agree']:.4f} | {r['mean_cc']:.4f} | {mv:.4f} |")

    # ------------------------------------------------------------------ verdict
    L += ["", "## Verdict", ""]
    if base and len(res) > 1:
        best = max((n for n in res if n != "arm B (lambda_cons=0)"),
                   key=lambda n: res[n]["mean"])
        r = res[best]
        da = r["mean"] - base["mean"]
        d_cc = r["mean_cc"] - base["mean_cc"]
        bit = abs(d_cc) > 0.1 * max(base["mean_cc"], 1e-9)
        if not bit:
            L.append(f"**The term did not bite.** Conflicted-confident mass moved "
                     f"{d_cc:+.4f} from {base['mean_cc']:.4f} — the penalty was too weak at "
                     f"these lambdas to change the evidence structure, so the AUROC difference "
                     f"({da:+.4f}) is not evidence about the idea, only about these settings.")
        elif da >= 0.01:
            L.append(f"**The term helps.** {best} raises mean OOD AUROC {da:+.4f} to "
                     f"{r['mean']:.4f} and cuts conflicted-confident mass {d_cc:+.4f}. The "
                     f"structural constraint bit AND paid. This is the outcome that would earn "
                     f"the neuro-symbolic label — a symbolic constraint entering training and "
                     f"measurably improving detection — and it needs a confirming seed before "
                     f"it is claimed.")
        elif da <= -0.01:
            L.append(f"**The term bit and cost.** Conflicted-confident mass fell {d_cc:+.4f}, "
                     f"so the constraint did shape the evidence, but mean OOD AUROC dropped "
                     f"{da:+.4f}. Suppressing confidence under disagreement removes real signal "
                     f"here: some of what looks like incoherent over-confidence is a minority "
                     f"view being correct, which is exactly the R5 case the overlay found "
                     f"faithful.")
        else:
            L.append(f"**The term bit but did not pay.** Conflicted-confident mass fell "
                     f"{d_cc:+.4f} from {base['mean_cc']:.4f}, so the constraint reshaped the "
                     f"evidence as designed, while mean OOD AUROC moved {da:+.4f} — inside the "
                     f"seed band, so no detection claim either way. The honest reading: the "
                     f"symbolic constraint is free to impose, giving a better-behaved evidence "
                     f"structure at no measured accuracy cost, but it does not improve "
                     f"detection, and 'free' is not the same as 'useful'.")
    if missing:
        L += ["", "## Missing", ""] + [f"- {m}" for m in sorted(set(missing))[:16]]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(res, indent=2, default=float))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())

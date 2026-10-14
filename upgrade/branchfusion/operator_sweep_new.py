#!/usr/bin/env python3
"""Every conflict-routing rule on the CACHED opinions of our four branches.

    <discern_ext python> upgrade/branchfusion/operator_sweep_new.py

This is the cheap half of the operator question. The branches are frozen and their opinions are
already on disk, so each rule is a few seconds of arithmetic over twelve test sets -- which
means the whole comparison runs before any GPU does, and the end-to-end arms only need to try
the rules that survive here.

WHAT IT CANNOT ANSWER. A frozen branch cannot adapt to the operator, so a rule that would pay
off only when the branches can respond to it will look flat here. A loss in this table is
therefore weak evidence against a rule, while a WIN is strong evidence for it. The end-to-end
runs are what decide.

THE CAUTIOUS RULE IS THE ONE WITH A PREDICTION ATTACHED, and this is the table that tests it.
Dempster's rule requires independent sources; our branches correlate at 0.45 to 0.84 on scores,
so it double-counts shared evidence and should be over-confident. If that diagnosis is right,
`cautious` improves ECE and Brier -- possibly at a cost in AUROC, which is an ordering property
and not a calibration one. If AUROC moves and calibration does not, the diagnosis was wrong and
the rule is not the reason for any gain.
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

FINAL = ("A", "V", "O", "S16ctl")
OUT = C.REPORT / "OPERATOR_SWEEP.md"


def fpr_at_tpr(y, s, target=0.90):
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float((neg >= float(np.quantile(pos, 1.0 - target))).mean())


def ece(y, p, bins=15):
    e, n = 0.0, len(y)
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.sum():
            e += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return float(e)


def main() -> int:
    import torch
    sys.path[:0] = [str(C.ROOT / "training/joint_ibdc")]
    sys.path.insert(0, "/data/umar/Repos/DISCERN_Ext")
    from src.fusion import rules as R
    from src.fusion import pcr6p as P

    dev = B.dev_table(list(FINAL), require_all=True)
    w, _ = O.fit_weights(dev, FINAL, "F8", objective="v31")
    print(f"reliability weights fitted once on development: {[round(x, 2) for x in w]}", flush=True)

    tabs = B.test_tables(list(FINAL))
    sets = [d for d in C.TEST_SETS if d in tabs]

    # Every rule reads the SAME discounted masses, so the comparison isolates the routing of
    # conflicting mass and nothing else. `masses` applies the fitted reliability discount.
    RULES = {
        "Dempster (DiCoME eq. 7)": R.dempster,
        "PCR6+": R.pcr6_plus,
        "Yager": R.yager,
        "WBF (Josang)": R.wbf,
        "Cautious (Denoeux)": R.cautious,
    }
    rows = {k: {} for k in RULES}
    for ds in sets:
        t = tabs[ds]
        y = t.y.to_numpy()
        m = torch.tensor(O.masses(t, FINAL, list(w)), dtype=torch.float64)
        for name, fn in RULES.items():
            f = fn(m)
            p = P.p_of(f).numpy()
            rows[name][ds] = {"auroc": C.auroc(y, p), "fpr90": fpr_at_tpr(y, p),
                              "ece": ece(y, p), "brier": float(np.mean((p - y) ** 2)),
                              "vacuity": float(f[:, 2].mean())}
        print(f"  {ds}", flush=True)

    L = ["# Operator sweep -- conflict routing on the cached four-branch opinions", "",
         "App + TD + LFS + STT, reliability weights fitted ONCE on development and applied",
         "unchanged. Every rule reads the same discounted masses, so the only thing that varies",
         "is where conflicting mass goes.", "",
         "`Dempster` is DiCoME eq. (7) -- the paper's own rule, written in subjective-logic form.",
         "On a BINARY frame Yager is also Dubois-Prade, and PCR5 collapses onto PCR6, so several",
         "operators the literature separates are one row here.", "",
         "**This table cannot settle the question.** A frozen branch cannot adapt to the operator,",
         "so a rule that pays off only when branches can respond to it looks flat. A win here is",
         "strong evidence; a loss is weak evidence. The end-to-end arms decide.", ""]
    for key, title, better in [("auroc", "AUROC", "higher"), ("fpr90", "FPR at TPR 90%", "lower"),
                               ("ece", "ECE-15", "lower"), ("brier", "Brier", "lower"),
                               ("vacuity", "mean fused vacuity", "diagnostic")]:
        L += [f"## {title} ({better} is better)" if better != "diagnostic"
              else f"## {title} (diagnostic, not a score)", "",
              "| rule | " + " | ".join(sets) + " | mean |",
              "|---|" + "---:|" * (len(sets) + 1)]
        for name in RULES:
            v = [rows[name][d][key] for d in sets]
            L.append(f"| {name} | " + " | ".join(f"{x:.4f}" for x in v) +
                     f" | **{np.mean(v):.4f}** |")
        L.append("")

    base = {k: np.mean([rows["Dempster (DiCoME eq. 7)"][d][k] for d in sets])
            for k in ("auroc", "fpr90", "ece", "brier")}
    L += ["## Against DiCoME's own rule", "",
          "Difference from the Dempster row, signed so that POSITIVE IS BETTER for every column.", "",
          "| rule | AUROC | FPR@90 | ECE | Brier |", "|---|---:|---:|---:|---:|"]
    for name in RULES:
        d = {k: np.mean([rows[name][x][k] for x in sets]) for k in base}
        L.append(f"| {name} | {d['auroc'] - base['auroc']:+.4f} | "
                 f"{base['fpr90'] - d['fpr90']:+.4f} | {base['ece'] - d['ece']:+.4f} | "
                 f"{base['brier'] - d['brier']:+.4f} |")
    L.append("")
    OUT.write_text("\n".join(L) + "\n")
    json.dump(rows, open(C.ROOT / "analysis/branchfusion/operator_sweep_new.json", "w"),
              indent=1, default=str)
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

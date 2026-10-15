#!/usr/bin/env python3
"""Ours against DiCoME at the SAME operating points, per dataset.

    <discern_ext python> upgrade/branchfusion/sota_operating_points.py

WHY THIS EXISTS. The forensic table reported our FPR with nothing beside it, which is not a fair
reading: an FPR of 0.16 is bad in the abstract and may be good or bad against the field. DiCoME
publishes no FPR at all -- their tooling computes video-level AUROC and nothing else, and there
is no threshold anywhere in `eval_adaptation/aggregate_video_auc.py` -- so the comparison cannot
be taken from their paper. It can be MEASURED, because their own per-video prediction CSVs are in
the repo for the sets they ran, and a probability per video is all an operating point needs.

THE PROTOCOL IS SYMMETRIC, which is the whole point. Each system's threshold is taken from its
OWN FF++ scores, at the same target false-alarm rate, and then applied unchanged to every test
set. Neither side is tuned on the corpus it is reported on, and neither is handed the other's
threshold -- that would compare calibrations rather than detectors.

WHAT IT CANNOT DO. DiCoME's runs cover CDFv2, CDFv3, DFD, DFDC, DFDCP and FF++; they never ran
FFIW, KoDF, IDForge or DFDM, so those rows are ours alone and are marked. A missing baseline is
left blank rather than filled with the paper's AUROC, which was measured under a different
protocol and says nothing about false alarms.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                               # noqa: E402
import branches as B                                             # noqa: E402
import opinions as O                                             # noqa: E402

DICOME = Path("/data/umar/Repos/DiCoME/runs")
#: `dicome_trained` is their model trained here; `dicome_repro` is the released-weights
#: reproduction. Both are reported, because a gap between them is information about the
#: baseline's stability and not something to pick the better of.
ARMS = {"DiCoME (trained)": "dicome_trained", "DiCoME (repro)": "dicome_repro"}
SUB = ("A", "V", "O", "P16ctl", "S16ctl")
TARGETS = [0.01, 0.05]
OUT_MD = C.REPORT / "SOTA_OPERATING_POINTS.md"


def dicome_scores(arm: str, ds: str):
    f = DICOME / arm / f"{arm}_{ds}" / "csv_logs/version_0/test_video_predictions.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    return d.label.to_numpy().astype(int), d.prob_class_1.to_numpy().astype(float)


def tau_at_fpr(y, p, target):
    neg = np.sort(p[y == 0])
    if not len(neg):
        return np.inf
    return float(neg[min(int(np.floor((1 - target) * len(neg))), len(neg) - 1)])


def point(y, p, tau):
    pred = p >= tau
    pos, neg = y == 1, y == 0
    tpr = float(pred[pos].mean()) if pos.any() else float("nan")
    fpr = float(pred[neg].mean()) if neg.any() else float("nan")
    return tpr, fpr, (tpr / fpr if fpr > 0 else float("inf"))


def main() -> int:
    sys.path[:0] = [str(C.ROOT / "analysis/tbiom/pcr6p")]
    import ops_pcr6p as P
    dev = B.dev_table(list(SUB), require_all=True)
    w, _ = O.fit_weights(dev, SUB, "F8", objective="v31")

    def fused(t):
        return P.p_of(P.fuse(O.masses(t, SUB, list(w)), True))

    # Our threshold, from OUR FF++ rows only.
    ff = np.where(dev.source.to_numpy() == "FFpp_val")[0]
    pdev, ydev = fused(dev), dev.y.to_numpy()
    tabs = B.test_tables(list(SUB))
    sets = [d for d in C.TEST_SETS if d in tabs]

    # Their threshold, from THEIR FF++ run only.
    their_tau = {}
    for label, arm in ARMS.items():
        s = dicome_scores(arm, "FFpp")
        their_tau[label] = {t: (tau_at_fpr(s[0], s[1], t) if s else None) for t in TARGETS}

    L = ["# Ours against DiCoME at matched false-alarm targets", "",
         "Each system's threshold comes from its OWN FF++ scores at the same target rate, then is",
         "applied unchanged to every test set. Neither side is tuned on what it reports, and",
         "neither is given the other's threshold -- that would compare calibrations, not",
         "detectors.", "",
         "DiCoME publishes no false-alarm figures; their evaluation computes video-level AUROC and",
         "no threshold appears anywhere in it. These rows are measured from their own per-video",
         "prediction CSVs. Blank cells are sets they never ran.", ""]
    res = {}
    for t in TARGETS:
        our_tau = tau_at_fpr(ydev[ff], pdev[ff], t)
        L += [f"## Target false-alarm rate {t:.0%} (set on FF++, applied unchanged)", "",
              "| dataset | system | TPR | FPR | LR+ |", "|---|---|---:|---:|---:|"]
        for ds in sets:
            y, p = tabs[ds].y.to_numpy(), fused(tabs[ds])
            tpr, fpr, lr = point(y, p, our_tau)
            L.append(f"| **{ds}** | **ours (ES-PCR6+, 5 branches)** | {tpr:.4f} | {fpr:.4f} | {lr:.1f} |")
            res.setdefault(f"{t}", {}).setdefault(ds, {})["ours"] = [tpr, fpr, lr]
            for label, arm in ARMS.items():
                s = dicome_scores(arm, ds)
                if s is None or their_tau[label][t] is None:
                    L.append(f"| | {label} | — | — | — |"); continue
                tpr2, fpr2, lr2 = point(s[0], s[1], their_tau[label][t])
                L.append(f"| | {label} | {tpr2:.4f} | {fpr2:.4f} | {lr2:.1f} |")
                res[f"{t}"][ds][label] = [tpr2, fpr2, lr2]
        L.append("")
    OUT_MD.write_text("\n".join(L) + "\n")
    json.dump(res, open(C.ROOT / "analysis/branchfusion/sota_operating_points.json", "w"),
              indent=1, default=str)
    print(f"-> {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

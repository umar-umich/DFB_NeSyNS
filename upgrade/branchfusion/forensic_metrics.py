#!/usr/bin/env python3
"""Operating-point metrics for a courtroom or a police triage queue, per dataset.

    <discern_ext python> upgrade/branchfusion/forensic_metrics.py

WHY NOT PRECISION. Precision depends on PREVALENCE -- how many of the videos reaching the court
are fake -- which is unknown, contested, and different in every case. Reporting it embeds a prior
that opposing counsel would attack, and should. FPR and FNR are properties of the detector; a
precision figure is a property of the detector AND the caseload it was measured on.

WHY BOTH FPR AND FNR. The dangerous error flips with the role the video plays. Offered as
authentic evidence, the costly error is a FALSE NEGATIVE: a fabricated video accepted as genuine,
which convicts an innocent person. Challenged as fabricated, the costly error is a FALSE
POSITIVE: genuine footage declared fake, which discredits a truthful witness. A table reporting
only one of them has silently assumed a direction of use.

WHY NOT EER. The equal-error threshold balances the two errors, which is a research convention
and not a forensic one -- no court treats them as equal. Forensic instruments declare an
operating point in advance, so TPR is reported at FIXED low FPR (1% and 0.1%). The thresholds for
those points are taken from the FF++ validation rows inside development, never from the test set
being reported.

LIKELIHOOD RATIO. LR+ = TPR / FPR is how forensic evidence is actually presented: it is
prevalence-free and combines with the court's own prior under Bayes, which is exactly the
division of labour an expert witness is supposed to respect. It is the number to put in a report;
AUROC is not.

TRIAGE. For police screening the base rate decides everything: at FPR 0.37 -- which is what the
EER threshold gives on DFDCP -- a 100,000-video queue yields 37,000 false alarms. The
`alarms_per_100k_genuine` column states that directly rather than leaving it as an inference.
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

OUT_MD = C.REPORT / "FORENSIC_OPERATING_POINTS.md"
SUB = ("A", "V", "O", "P16ctl", "S16ctl")
TARGET_FPR = [0.01, 0.001]


def tau_at_fpr(y, p, target):
    """Smallest threshold whose FPR on GENUINE videos is at or below `target`."""
    neg = np.sort(p[y == 0])
    if not len(neg):
        return np.inf
    k = int(np.floor((1.0 - target) * len(neg)))
    return float(neg[min(k, len(neg) - 1)])


def row(y, p, tau):
    pred = p >= tau
    pos, neg = y == 1, y == 0
    tpr = float(pred[pos].mean()) if pos.any() else float("nan")
    fpr = float(pred[neg].mean()) if neg.any() else float("nan")
    return {"tpr_fake": tpr, "fnr_fake": 1 - tpr, "fpr_real": fpr,
            "lr_plus": (tpr / fpr) if fpr > 0 else float("inf"),
            "alarms_per_100k_genuine": fpr * 100_000}


def main() -> int:
    import ops_pcr6p as P
    sys.path[:0] = [str(C.ROOT / "analysis/tbiom/pcr6p")]
    dev = B.dev_table(list(SUB), require_all=True)
    w, _ = O.fit_weights(dev, SUB, "F8", objective="v31")

    def fused(t):
        f = P.fuse(O.masses(t, SUB, list(w)), True)
        return P.p_of(f)

    pdev, ydev = fused(dev), dev.y.to_numpy()
    ff = np.where(dev.source.to_numpy() == "FFpp_val")[0]
    from health import eer
    taus = {"EER (research convention)": float(eer(ydev[ff], pdev[ff])[1])}
    for t in TARGET_FPR:
        taus[f"FPR<={t:.1%} on development"] = tau_at_fpr(ydev[ff], pdev[ff], t)
    print({k: round(v, 4) for k, v in taus.items()}, flush=True)

    tabs = B.test_tables(list(SUB))
    sets = [d for d in C.TEST_SETS if d in tabs]
    res = {}
    for name, tau in taus.items():
        res[name] = {d: row(tabs[d].y.to_numpy(), fused(tabs[d]), tau) for d in sets}

    L = ["# Operating points for forensic use", "",
         "Per dataset, never a macro. Thresholds come from the FF++ validation rows inside",
         "development and are applied unchanged to every test set -- a threshold tuned per corpus",
         "is not an operating point, it is a result fitted on the thing it is reporting.", "",
         "**Precision is deliberately absent.** It depends on how many videos reaching the court",
         "are fake, which is unknown and contested; FPR and FNR are properties of the detector",
         "alone. **LR+ = TPR/FPR** is the prevalence-free quantity an expert witness can defend,",
         "and it combines with the court's own prior under Bayes.", ""]
    for name, tau in taus.items():
        L += [f"## Threshold: {name}  (tau = {tau:.4f})", "",
              "| dataset | TPR (fake caught) | FNR (fake missed) | FPR (genuine flagged) | LR+ | "
              "false alarms per 100k genuine |", "|---|---:|---:|---:|---:|---:|"]
        for d in sets:
            r = res[name][d]
            L.append(f"| {d} | {r['tpr_fake']:.4f} | {r['fnr_fake']:.4f} | {r['fpr_real']:.4f} | "
                     f"{r['lr_plus']:.1f} | {r['alarms_per_100k_genuine']:,.0f} |")
        L.append("")
    L += ["## How to read these", "",
          "A video offered as AUTHENTIC evidence is endangered by a false negative -- the FNR",
          "column. A video challenged as FABRICATED is endangered by a false positive -- the FPR",
          "column. The two are not interchangeable and the right one depends on which side is",
          "relying on the video, so both are reported at every threshold.", "",
          "For police triage the last column is the operative number: it is how many genuine",
          "videos an analyst has to clear for every hundred thousand screened, and it is what",
          "makes an EER threshold unusable at scale however good the AUROC above it looks."]
    OUT_MD.write_text("\n".join(L) + "\n")
    json.dump({"weights": list(w), "thresholds": taus, "rows": res},
              open(C.ROOT / "analysis/branchfusion/forensic_operating_points.json", "w"),
              indent=1, default=str)
    print(f"-> {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

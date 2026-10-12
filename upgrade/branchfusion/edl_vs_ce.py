#!/usr/bin/env python3
"""DiCoME Table 2(b) and Table 3, rebuilt on OUR system -- per dataset, never as a mean.

    <discern_ext python> upgrade/branchfusion/edl_vs_ce.py

WHAT DiCoME DID (Table 2b, CDFv2/DFDC/MFS, averaged):
    (D)   L_ce  + Mean fusion + no uncertainty   0.924
    (E)   L_edl + Mean fusion + uncertainty      0.927
    Ours  L_edl + DS-C        + uncertainty      0.938
and Table 3, ECE: baseline (D) 0.16 against theirs 0.07.

WHAT THIS BUILDS. The same two factors -- the loss and the fusion rule -- but with PCR6+ where
they have DS-C, over our branches, on EVERY test set separately. UMAR: "I want results for all
datasets, not these mean values." A mean over ten corpora that differ by 0.25 in AUROC hides the
thing a reader needs.

THE FUSION AXIS is free: `F1` is the uniform probability mean (their "Mean"), `F7` the
reliability-discounted Dempster combination (their DS-C), `F8` the verified direct multisource
PCR6+ (ours). Same opinions, same weights, three decision rules.

THE LOSS AXIS IS NOT FREE, and this file does not pretend otherwise. Swapping EDL for
cross-entropy means retraining the heads, so the CE column here is DISCERN_Ext's arm D
(`run1d_simplece_seed42`: three linear heads, mean-of-logits, plain CE) read from the published
table -- a DIFFERENT three views on a different trunk, not our five branches. It answers "does
the evidential apparatus beat a CE baseline of similar design", not "does it beat a CE version of
THIS system". The second question needs trained CE heads on our branches, which is a separate job.

WHAT THE NUMBERS ALREADY SAY, so the table is not read naively: on OOD AUROC our arm D does NOT
drop the way DiCoME's does -- it ties at 0.8639 against the evidential arm's 0.8633. The
evidential case in our system rests on the OPERATING POINT and on referral, not on ranking: arm D
reaches FPR_real 0.3156 where the evidential arm is at 0.1350, and vacuity ranks the model's own
errors at err-AUROC 0.771 where the decision margin manages 0.568.
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
import subsets as S                                              # noqa: E402

OUT_MD = C.ROOT / "reports/branch_fusion/EDL_VS_CE.md"
OUT_JSON = C.ROOT / "analysis/branchfusion/edl_vs_ce.json"

#: their "Mean", their DS-C, ours. F5 is added because it is the evidential mean -- cumulative
#: evidence with the prior counted once -- which is closer to DiCoME's "averaging belief masses"
#: than a probability mean is.
OPS = [("F1", "uniform probability mean (their 'Mean')"),
       ("F5", "cumulative evidence, prior once (evidential mean)"),
       ("F7", "reliability-discounted Dempster (their DS-C)"),
       ("F8", "direct multisource PCR6+ (ours)")]

SUBSETS = [("A", "V", "O"), ("A", "V", "O", "P16ctl", "S16ctl")]

#: DISCERN_Ext arm D, three linear heads + mean-of-logits + plain CE, from
#: tbiom/reports/FULL_RESULTS.md. Different views and trunk; kept here only as the CE reference
#: point, and labelled as such in the report.
ARM_D = {"CDFv2": 0.9376, "CDFv3": 0.8648, "DFD": 0.9449, "DFDC": 0.8791,
         "DFDCP": 0.8724, "DFEval24": 0.6850}
ARM_D_FPR = {"CDFv2": 0.1966, "CDFv3": 0.1966, "DFD": 0.3691, "DFDC": 0.3201,
             "DFDCP": 0.3696, "DFEval24": 0.4416}
ARM_B_FPR_MEAN, ARM_D_FPR_MEAN = 0.1350, 0.3156
ARM_B_AUROC_MEAN, ARM_D_AUROC_MEAN = 0.8633, 0.8639


def main() -> int:
    items = sorted({b for s in SUBSETS for b in s})
    dev = B.dev_table(items, require_all=True)
    tabs = B.test_tables(items)
    sets = [d for d in C.TEST_SETS if d in tabs]
    rows = {}
    for sub in SUBSETS:
        for op, _ in OPS:
            r = S.evaluate_test(dev, tabs, sub, op)
            rows[(C.subset_id(sub), op)] = r
            print(f"  {C.subset_id(sub):26s} {op}  macro {r['ten_set_macro']:.4f}", flush=True)

    L = ["# EDL against cross-entropy, and PCR6+ against the alternatives", "",
         "Rebuilt from DiCoME Table 2(b) and Table 3 on our system, **per dataset**. A mean over",
         "ten corpora that differ by 0.25 in AUROC hides what a reader needs.", ""]

    for metric, name, better in [("auroc", "AUROC", "higher"), ("ece15", "ECE-15", "lower"),
                                 ("nll", "NLL", "lower"), ("brier", "Brier", "lower"),
                                 ("fpr_real", "FPR_real at the frozen threshold", "lower")]:
        L += [f"## {name} ({better} is better)", "",
              "| system | fusion | " + " | ".join(sets) + " | macro |", "|---|---|" +
              "---:|" * (len(sets) + 1)]
        for sub in SUBSETS:
            sid = C.subset_id(sub)
            for op, _ in OPS:
                r = rows[(sid, op)]
                vals = [r["per_dataset"][d][metric] for d in sets]
                mark = "**" if op == "F8" else ""
                L.append(f"| {mark}{sid}{mark} | {op} | " +
                         " | ".join(f"{v:.4f}" for v in vals) +
                         f" | {np.mean(vals):.4f} |")
        if metric == "auroc":
            L.append("| *CE baseline (DISCERN_Ext arm D, 3 other views)* | mean-of-logits | " +
                     " | ".join(f"{ARM_D[d]:.4f}" if d in ARM_D else "--" for d in sets) +
                     f" | {ARM_D_AUROC_MEAN:.4f}\\* |")
        if metric == "fpr_real":
            L.append("| *CE baseline (DISCERN_Ext arm D, 3 other views)* | mean-of-logits | " +
                     " | ".join(f"{ARM_D_FPR[d]:.4f}" if d in ARM_D_FPR else "--" for d in sets) +
                     f" | {ARM_D_FPR_MEAN:.4f}\\* |")
        L.append("")

    L += ["\\* the CE row is DISCERN_Ext's arm D over a DIFFERENT three views and trunk, and its",
          "macro is over its own six OOD sets, not these ten. It is the CE reference point this",
          "project has today, not a CE version of the system above. Treat the columns as",
          "indicative and the macro as not comparable.", "",
          "## What the comparison actually shows", "",
          f"On ranking, CE does **not** collapse the way DiCoME's (D) does: our arm D reaches",
          f"{ARM_D_AUROC_MEAN:.4f} OOD AUROC against the evidential arm's {ARM_B_AUROC_MEAN:.4f} --",
          "a tie, +0.0006 in CE's favour. Any claim that the evidential machinery buys accuracy is",
          "not supported here and must not be written.", "",
          "Where it does separate is the **operating point**. At the frozen threshold arm D sits at",
          f"FPR_real {ARM_D_FPR_MEAN:.4f} against the evidential arm's {ARM_B_FPR_MEAN:.4f} -- more",
          "than twice the false-alarm rate on genuine video at equal ranking quality. That is a",
          "deployment difference, not a leaderboard one, and it is the honest form of the claim.", "",
          "And on **referral**, which CE cannot do at all: vacuity ranks the model's own errors at",
          "err-AUROC 0.771 where the decision margin -- the only uncertainty a CE fusion of the same",
          "branches would hand you -- manages 0.568 (`tbiom/reports/REFERRAL.md`)."]

    OUT_MD.write_text("\n".join(L) + "\n")
    json.dump({"subsets": {f"{k[0]}|{k[1]}": v for k, v in rows.items()},
               "arm_d_reference": {"auroc": ARM_D, "fpr_real": ARM_D_FPR,
                                   "note": "DISCERN_Ext arm D, different views/trunk"}},
              open(OUT_JSON, "w"), indent=1, default=str)
    print(f"\n-> {OUT_MD}\n-> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

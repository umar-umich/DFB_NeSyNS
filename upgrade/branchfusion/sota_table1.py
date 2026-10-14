#!/usr/bin/env python3
"""Table 1 -- detection performance against the field, per dataset: AUROC and FPR.

    <discern_ext python> upgrade/branchfusion/sota_table1.py

THE THRESHOLD PROBLEM, and how it is solved here. A false-alarm rate needs an operating point,
and the baselines were never scored on FF++ validation, so there is no held-out set on which to
fit each of them a threshold the way `sota_operating_points.py` does for DiCoME. Giving them OUR
threshold would compare calibrations rather than detectors, and giving each its own test-fitted
threshold would be worse.

So the operating point is FPR AT A MATCHED TPR: of the genuine videos, what fraction does each
method flag when it is tuned to catch 90% of the fakes? Every method is read at the same
sensitivity, the comparison is free of any calibration difference, and it is the number a
reviewer asking "how noisy is it in practice" actually wants.

THIS IS AN ROC PROPERTY, NOT A DEPLOYMENT CLAIM. FPR@TPR90 is read off each method's own ROC on
the set being reported, exactly as AUROC is, so it says nothing about whether a threshold chosen
in advance would hold. That separate -- and much less flattering -- question is what
`FORENSIC_OPERATING_POINTS.md` and `SOTA_OPERATING_POINTS.md` answer, and the two must not be
quoted as if they were the same measurement.
"""
from __future__ import annotations

import glob
import json
import pathlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                               # noqa: E402
import branches as B                                             # noqa: E402
import opinions as O                                             # noqa: E402

BASE = C.ROOT / "logs/tbiom/baselines"
GEND_CROPS = {"FFIW", "KoDF", "IDForge", "DFDM"}
MODELS = ["GenD_CLIP", "GenD_PE", "GenD_DINO", "FSFM", "FSFM_Adapter", "Effort", "ForAda"]
PRETTY = {"GenD_CLIP": "GenD-CLIP", "GenD_PE": "GenD-PE", "GenD_DINO": "GenD-DINO",
          "FSFM_Adapter": "FSFM-Adapter"}
SUB = ("A", "V", "O", "S16ctl")
TPR_TARGETS = [0.90, 0.95]
TODO = "*in progress*"
NOTE = """A ROW'S `mean` IS OVER ITS OWN COVERAGE, which is not a comparison. Our row spans twelve
sets, most baselines nine, DiCoME five -- and the three corpora only we cover are not the
easy ones, so our own mean is read off a harder average than anyone else's. The two
matched columns are the comparable figures:

- `mean (all)` -- every set every row in the table resolves. DiCoME's five-set coverage
  sets that width, so it is honest for all nine methods but narrow.
- `mean (wide)` -- every set that every method covering nine or more sets resolves. The
  narrow rows say in progress there rather than shrinking the column for everyone.

Rank methods on a matched column and say which one. The per-row mean belongs in the
per-dataset discussion, not in a ranking."""
OUT_MD = C.ROOT / "reports/branch_fusion/TABLE1_DETECTION.md"


def scope(df, subset):
    keep = ~subset.str.contains("IP_LAP") & (subset != "fake-audio-driven") & ~subset.str.startswith("adv-")
    return df[keep.to_numpy()].reset_index(drop=True)


def fpr_at_tpr(y, s, target):
    """Smallest FPR among thresholds reaching at least `target` TPR."""
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    tau = float(np.quantile(pos, 1.0 - target))        # catch `target` of the fakes
    return float((neg >= tau).mean())


def baseline_videos(model, ds):
    root = BASE / ("gend_runs" if ds in GEND_CROPS else "gend_runs_ours")
    hits = glob.glob(f"{root}/{model}--{ds}/**/test_predictions.csv", recursive=True)
    if not hits:
        return None
    d = pd.read_csv(hits[0])
    d = scope(d, d.files.str.split("/").str[-3])
    d["v"] = d.files.str.rsplit("/", n=1).str[0]
    g = d.groupby("v").agg(y=("labels", "max"), s=("prob_class_1", "mean"))
    return g.y.to_numpy(), g.s.to_numpy()


#: DiCoME's own runs, per video, from its repo. `dicome_trained` is the model trained on our
#: protocol; it covers five of our nine sets and never ran FFIW, KoDF, IDForge or DFDM. Those
#: cells stay blank rather than being filled from the paper, whose numbers come from a different
#: protocol and cannot be read at a matched TPR.
DICOME_RUNS = pathlib.Path("/data/umar/Repos/DiCoME/runs/dicome_trained")


def dicome_videos(ds):
    f = DICOME_RUNS / f"dicome_trained_{ds}" / "csv_logs/version_0/test_video_predictions.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    return d.label.to_numpy().astype(int), d.prob_class_1.to_numpy().astype(float)


def main() -> int:
    sys.path[:0] = [str(C.ROOT / "analysis/tbiom/pcr6p")]
    import ops_pcr6p as P
    dev = B.dev_table(list(SUB), require_all=True)
    w, _ = O.fit_weights(dev, SUB, "F8", objective="v31")
    tabs = B.test_tables(list(SUB))
    sets = [d for d in C.TEST_SETS if d in tabs]

    rows = {}
    rows["**Ours (App+TD+LFS+STT, PCR6+-R)**"] = {}
    for ds in sets:
        t = tabs[ds]
        p = P.p_of(P.fuse(O.masses(t, SUB, list(w)), True))
        y = t.y.to_numpy()
        rows["**Ours (App+TD+LFS+STT, PCR6+-R)**"][ds] = {
            "auroc": C.auroc(y, p), **{f"fpr{int(k*100)}": fpr_at_tpr(y, p, k) for k in TPR_TARGETS}}
    rows["DiCoME (trained, their repo)"] = {}
    for ds in sets:
        r = dicome_videos(ds)
        if r is None:
            continue
        y, s_ = r
        rows["DiCoME (trained, their repo)"][ds] = {
            "auroc": C.auroc(y, s_), **{f"fpr{int(k*100)}": fpr_at_tpr(y, s_, k) for k in TPR_TARGETS}}
    for m in MODELS:
        name = PRETTY.get(m, m)
        rows[name] = {}
        for ds in sets:
            r = baseline_videos(m, ds)
            if r is None:
                continue
            y, s = r
            rows[name][ds] = {"auroc": C.auroc(y, s),
                              **{f"fpr{int(k*100)}": fpr_at_tpr(y, s, k) for k in TPR_TARGETS}}

    L = ["# Table 1 — detection performance", "",
         "Per dataset. **AUROC higher is better, FPR lower is better.** FPR is read at a MATCHED",
         "true-positive rate, so every method is compared at the same sensitivity and no",
         "calibration difference enters: of the genuine videos, what fraction is flagged when the",
         "method is tuned to catch 90% (resp. 95%) of the fakes.", "",
         "This is an ROC property, like AUROC. Whether a threshold fixed in advance survives the",
         "domain shift is a different and far less flattering question, answered in",
         "`SOTA_OPERATING_POINTS.md`. The two must not be quoted as the same measurement.", "",
         f"`{TODO}`: not scored on that corpus yet, not zero and not inapplicable. Our row covers",
         "every set. The baselines were never run on the three corpora packed on 2026-10-13, and",
         "DiCoME's public checkpoints cover only the five sets its paper reports.", "",
         *NOTE.split("\n"), ""]
    for metric, title in [("auroc", "AUROC (higher is better)"),
                          ("fpr90", "FPR at TPR = 90% (lower is better)"),
                          ("fpr95", "FPR at TPR = 95% (lower is better)")]:
        all_s, wide_s = C.common_supports(rows, sets, metric)
        L += [f"## {title}", "",
              f"`mean (all)`: {len(all_s)} sets ({', '.join(all_s) or 'none'}).  ",
              f"`mean (wide)`: {len(wide_s)} sets ({', '.join(wide_s) or 'none'}).", "",
              "| method | " + " | ".join(sets) + " | mean | mean (all) | mean (wide) |",
              "|---|" + "---:|" * (len(sets) + 3)]
        for name, r in rows.items():
            vals = [r.get(d, {}).get(metric) for d in sets]
            got = [v for v in vals if v is not None and np.isfinite(v)]
            ctail = (C.matched_cells(r, all_s, metric, TODO) +
                     C.matched_cells(r, wide_s, metric, TODO))
            # IN PROGRESS, not an em dash. A dash reads as "not applicable" and these cells are
            # baselines that have not been scored on the corpus yet -- a difference a reader of
            # the draft cannot recover from the glyph.
            L.append(f"| {name} | " + " | ".join(TODO if v is None else f"{v:.4f}" for v in vals) +
                     (f" | {np.mean(got):.4f} |" if got else f" | {TODO} |") + ctail)
        L.append("")
    OUT_MD.write_text("\n".join(L) + "\n")
    json.dump(rows, open(C.ROOT / "analysis/branchfusion/table1_detection.json", "w"),
              indent=1, default=str)
    print(f"-> {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

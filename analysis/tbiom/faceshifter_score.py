#!/usr/bin/env python3
"""Score FaceShifter with the deployed system, once, after the recipe is fixed.

    OMP_NUM_THREADS=4 python analysis/tbiom/faceshifter_score.py

The fusion is not re-tuned here: DS-R's reliability weights and the logistic comparator come from the same
source-grouped fitting folds every other experiment uses, and tau is the fitted fold's FF++ val EER. FaceShifter
contributes one number per arm per split and nothing else; no threshold, weight or checkpoint is revisited after this
file is written.

What the benchmark is, from the audit of 2026-09-17 (metadata only, recorded in tbiom/docs/SEALED_BENCHMARKS.md):
the release's 140 source ids are exactly the FF++ TEST split, disjoint from our 720 training and 140 validation
source ids. It therefore tests an unseen MANIPULATION on held-out source videos. It is not called "zero-leakage":
that property belongs to this split pairing, not to the release.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent / "temporal"))
import exp1_fusion_audit as A  # noqa: E402
import dsr_system as D  # noqa: E402
import finish_line as F  # noqa: E402
from fusion_combos import attach  # noqa: E402
from health import eer  # noqa: E402

S23, BR, R = Path("logs/tbiom/stage23"), Path("logs/tbiom/temporal/branches"), Path("tbiom/reports")
VIEWS, BRANCH = ["T", "APP", "OM"], "vjepa2_lora_B_seed42"


def main():
    dev, _dproc, _tests = D.load(BRANCH)
    fs = attach(A.video_table(A.load_frames(S23 / "warmgate_sel_FaceShifter.csv"), [1.0]),
                [BR / BRANCH / "opinions" / "FaceShifter.csv"])
    y = fs.y.to_numpy()
    rows = []
    for k, Fd, Sd in D.splits(dev):
        rules, params, w = F.rules_for(Fd, VIEWS)
        r = {"split": k, "w": params["w"]}
        for rule, f in rules.items():
            s = f(fs); ref = f(Sd); ffs = np.where(Sd.source == "FFpp_val")[0]
            tau = float(eer(Sd.y.to_numpy()[ffs], ref[ffs])[1])
            pred = (s >= tau).astype(int)
            r[rule] = {"auroc": A.fast_auc(s[y == 0], s[y == 1]), "tau": tau,
                       "fpr": float((pred[y == 0] == 1).mean()), "tpr": float((pred[y == 1] == 1).mean())}
        for v in VIEWS:
            r[f"branch {v}"] = {"auroc": A.fast_auc(fs[f"p_{v}"].to_numpy()[y == 0], fs[f"p_{v}"].to_numpy()[y == 1])}
        rows.append(r)
        print(f"split {k}: DS-R {r['DS-R']['auroc']:.4f} LR {r['LR']['auroc']:.4f}", flush=True)
    json.dump(rows, open(R / "FACESHIFTER.json", "w"), indent=1, default=float)
    m = lambda arm, f="auroc": float(np.mean([x[arm][f] for x in rows]))
    L = ["# FaceShifter — the confirmation read", "", __doc__.split("\n\n", 1)[1].strip(), "",
         f"{int((y == 0).sum())} genuine and {int((y == 1).sum())} manipulated videos.", "", F.CITE, "",
         "| arm | video AUROC | FPR at the frozen τ | TPR |", "|---|---:|---:|---:|",
         f"| **DISCERN-v2 (DS-R)** | **{m('DS-R'):.4f}** | {m('DS-R', 'fpr'):.3f} | {m('DS-R', 'tpr'):.3f} |",
         f"| logistic comparator | {m('LR'):.4f} | {m('LR', 'fpr'):.3f} | {m('LR', 'tpr'):.3f} |"]
    for v in VIEWS:
        L.append(f"| {v} view alone | {m(f'branch {v}'):.4f} | — | — |")
    L += ["", "Mean over the five fitted splits; the spread across splits comes only from the fusion parameters, since "
          "the checkpoints are fixed. Read once: nothing was tuned after this file was written.", "", F.FOOT]
    (R / "FACESHIFTER.md").write_text("\n".join(L) + "\n"); print("wrote", R / "FACESHIFTER.md")


if __name__ == "__main__":
    main()

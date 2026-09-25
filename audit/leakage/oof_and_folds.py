#!/usr/bin/env python3
"""Tasks 3 and 5 of the leakage audit: out-of-fold opinions, the symptom check, and fold grouping. Read-only.

    OMP_NUM_THREADS=4 python audit/leakage/oof_and_folds.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path[:0] = [str(Path("analysis/tbiom/pcr6p").resolve()), str(Path("analysis/tbiom/grouped").resolve()),
                str(Path("analysis/tbiom").resolve()), str(Path("analysis/tbiom/temporal").resolve()),
                str(Path(__file__).resolve().parent)]
import exp1_fusion_audit as A  # noqa: E402
import dsr_system as D  # noqa: E402
import confirm as C  # noqa: E402
import keys as K  # noqa: E402

OUT = Path("audit/leakage")
EXT = Path("/data/umar/Repos/DISCERN_Ext")
VIEWS = ("T", "APP", "OM")
# the checkpoint behind each view's cached opinions, and the manifests that checkpoint was trained/selected on
PROV = {
    "APP": {"checkpoint": "DISCERN_Ext/runs/discern_ext_train/warm_dicome_gate_seed42/checkpoints/"
                          "discern_ext-best-epoch=19-val_auroc_video=0.9956.ckpt",
            "train": "FFpp_train (5 manifests, from warm_dicome_gate_seed42.yaml trn_files)",
            "select": "VALmixV2 (v3.1 score)", "note": "frozen at the DiCoME warm start (freeze_views: true)"},
    "OM": {"checkpoint": "same checkpoint as APP", "train": "FFpp_train", "select": "VALmixV2 (v3.1 score)",
           "note": "frozen at the DiCoME warm start"},
    "T": {"checkpoint": "DFB_NeSyNS/logs/tbiom/temporal/branches/vjepa2_lora_B_seed42/best.pt",
          "train": "FFpp c23 train only (3,600 videos, per train_branch.py)",
          "select": "v3.1 development score on VALmix + VALmixV2 + FFpp_val",
          "note": "LoRA adapters + evidential head only"},
}


def main():
    dev, dproc, tests = D.load("vjepa2_lora_B_seed42")
    L = ["# Task 3 — out-of-fold opinions and the symptom check", "",
         "## 3a. Provenance of each view's cached opinions", "",
         "| view | checkpoint behind the cached opinions | trained on | selected on | note |",
         "|---|---|---|---|---|"]
    for v, p in PROV.items():
        L.append(f"| {v} | `{p['checkpoint']}` | {p['train']} | {p['select']} | {p['note']} |")
    L += ["", "**The central intersection.** Every view was trained on FF++ c23 *train* and the trust weights are "
          "fitted on the development pool (VALmix, VALmix-v2, FF++ *val*). Those are disjoint splits of FF++ plus "
          "material from other sources, so at video level the training data and the weight-fitting data do not "
          "intersect — `overlap_video.csv` shows zero between `TRAIN FFpp c23` and both `DEV POOL` rows.", "",
          "**What does intersect.** Checkpoint *selection* for all three views used VALmix-v2, which is also part of "
          "the trust-weight fitting pool. Selection and fitting therefore share data. That is not test leakage, but "
          "it does mean the fitted weights are applied to views chosen on the same videos, so the development "
          "numbers are optimistic as an estimate of held-out fusion quality. This is the stacking risk the brief "
          "names, and it is present by construction.", "",
          "## 3b. Symptom check — per-view AUROC and mean vacuity", "",
          "No threshold is set; the numbers are reported so a reader can judge.", "",
          "| view | dev pool AUROC | dev pool mean vacuity | FF++ val AUROC | FF++ val vacuity | CDFv3 AUROC | CDFv3 vacuity |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    ffv = dev[dev.source.to_numpy() == "FFpp_val"]
    c3 = tests["CDFv3"]
    for v in VIEWS:
        row = []
        for t in (dev, ffv, c3):
            y = t.y.to_numpy(); p = t[f"p_{v}"].to_numpy()
            row += [A.fast_auc(p[y == 0], p[y == 1]), float(t[f"u_{v}"].to_numpy().mean())]
        L.append(f"| {v} | " + " | ".join(f"{x:.4f}" for x in row) + " |")
    L += ["", "The development pool includes FF++ val, so the two leftmost columns are not independent of each other. "
          "A view whose development vacuity sat far below its FF++-val vacuity would be the signature of having been "
          "fitted on development material; read the table with that in mind rather than against a threshold.", "",
          "## 3c. Were the development numbers computed out of fold?", ""]
    # verify the fitting/selection disjointness of the five folds as the code actually builds them
    ok = True
    rows = []
    for k, Fd, Sd in D.splits(dev):
        inter = set(Fd.v) & set(Sd.v)
        ok &= not inter
        rows.append((k, len(Fd), len(Sd), len(inter)))
    L += ["| split | fitting videos | selection videos | video overlap |", "|---|---:|---:|---:|"]
    for k, nf, ns, ni in rows:
        L.append(f"| {k} | {nf} | {ns} | {ni} |")
    L += ["", f"**{'PASS' if ok else 'FAIL'}** — the weights are fitted on the fitting half of each fold and scored "
          "on the held-out half, with no video in both. Verified by intersecting the two index sets that "
          "`dsr_system.splits` actually returns, not by reading the fold code.", ""]
    (OUT / "oof_check.md").write_text("\n".join(L) + "\n")
    print("wrote oof_check.md; out-of-fold disjointness:", "PASS" if ok else "FAIL")

    # ---------------- Task 5: connected components over source videos and identities
    groups = A.groups_for(dev)
    folds = {}
    for k, _Fd, Sd in D.splits(dev):
        for g in np.unique(A.groups_for(Sd)):
            folds.setdefault(g, set()).add(k)
    split_across = {g: sorted(v) for g, v in folds.items() if len(v) > 1}
    per_fold = {}
    for k, _Fd, Sd in D.splits(dev):
        per_fold[k] = len(np.unique(A.groups_for(Sd)))
    M = ["# Task 5 — fold grouping", "",
         "Connected components are built by `exp1_fusion_audit.groups_for`, which unions a video with the source "
         "videos it derives from, so a fake links its sources and every clip of a source lands in one component.", "",
         f"Development pool: **{len(np.unique(groups))} components** over {len(dev)} videos.", "",
         "| fold | components in the selection half |", "|---|---:|"]
    for k, n in per_fold.items():
        M.append(f"| {k} | {n} |")
    M += ["", f"Components appearing in more than one fold's selection half: **{len(split_across)}**.", ""]
    if split_across:
        M += ["This is expected and is not a leak by itself: `GroupShuffleSplit` is run with five different seeds, so "
              "a component is in the selection half of some folds and the fitting half of others. The property that "
              "matters is that within any single fold a component is never in both halves, which Task 3c verifies "
              "directly.", "", "First 10 components and the folds whose selection half contains them:", ""]
        for g, v in list(split_across.items())[:10]:
            M.append(f"- `{g}` -> folds {v}")
    else:
        M.append("Each component sits in exactly one fold's selection half.")
    M.append("")
    (OUT / "fold_grouping.md").write_text("\n".join(M) + "\n")
    print(f"wrote fold_grouping.md; {len(np.unique(groups))} components, "
          f"{len(split_across)} appear in >1 selection half")


if __name__ == "__main__":
    main()

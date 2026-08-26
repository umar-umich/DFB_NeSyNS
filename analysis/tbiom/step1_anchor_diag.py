#!/usr/bin/env python3
"""Step 1 — is the Celeb-DF-v2 anchor gap a CLIP recipe gap or DiCoME's extra machinery?

    python analysis/tbiom/step1_anchor_diag.py --out tbiom/STEP1_ANCHOR_DIAG.md

Compares five readouts on the health dashboard, per dataset:

    DiCoME-released  fused      their checkpoint, DS(semantic, artifact)
    DiCoME-released  semantic   their checkpoint, the CLIP branch ALONE
    DiCoME-P0DS      fused      our FF++ c23 retrain, retrainable
    DiCoME-P0DS      semantic   our retrain, CLIP branch alone
    CLIP port        p_sem      our reimplementation

The decision rule the brief sets:

  * If DiCoME-semantic ~= the port, there is no CLIP implementation gap — DiCoME's decomposition
    and internal DS fusion make the difference, and the "gap" is architectural by definition.
  * If DiCoME-semantic ~= fused (near 0.97) while the port stays near 0.92, there is a genuine
    recipe gap, and only then is it worth investigating LoRA/optimizer/augmentation/epoch.

A CORRECTNESS CHECK RUNS FIRST. `p_fused` from the new exporter must reproduce the numbers in
`DiCoME/eval_adaptation/RESULTS.md`, which were produced by DiCoME's own untouched eval path. If
it does not, the exporter is reading the model wrong and nothing below it means anything.

Threshold discipline: one tau per readout, the EER on that readout's own FF++ scores, frozen
across every other dataset. Per-readout because the five are differently calibrated — a shared
tau would measure calibration offset rather than operating quality, the same mistake the Stage-1
gate made before it was fixed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from health import auroc, dashboard, eer  # noqa: E402
from video_id import video_id  # noqa: E402

STEP1 = Path("logs/tbiom/step1")
CROSS = Path("logs/tbiom/crossdataset")

# dataset name in DiCoME's world -> our sweep's directory suffix
DATASETS = {"FFpp": "FaceForensics__", "CDFv2": "Celeb_DF_v2", "DFD": "DeepFakeDetection",
            "DFDC": "DFDC", "DFDCP": "DFDCP", "DFEval24": "Deepfake_Eval_2024"}

# published reproduction numbers, DiCoME's own eval path — the exporter must match these
PUBLISHED_FUSED = {"CDFv2": 0.9729, "DFD": 0.9392, "DFDC": 0.8822, "DFDCP": 0.8799,
                   "FFpp": 0.9905}


def dicome_video(ds: str, arm: str, col: str) -> pd.DataFrame | None:
    """Video-level scores, grouped on the FRAME PATH's directory.

    NOT on the exporter's `video` column. That comes from the h5 dataset's
    `_parse_sample_metadata`, which mis-parses these layouts: it yields the manipulation method
    (`Deepfakes`, 144 groups for all of FF++), the identity rather than the video (`id0` for
    Celeb-DF-v2), and the integer `1` for DFD — which collapsed every DFD video into one group and
    made its AUROC undefined. Trusting that field is what broke the first run of this comparison.

    The directory is the video, which is also DiCoME's own rule in `_save_video_level_report`
    (`f.split("/")[-2]`). The FULL directory path is used rather than its basename, so a real and
    a fake video that share a basename cannot merge — the failure mode already hit once in this
    project on DF40's borrowed authentic halves.
    """
    p = STEP1 / f"{arm}_{ds}.csv"
    if not p.is_file():
        return None
    df = pd.read_csv(p)
    if col not in df:
        return None
    vid = video_id(df["key"])
    return pd.DataFrame({"v": vid, "p": df[col], "y": df["label"]}).groupby(
        "v", as_index=False).agg(p=("p", "mean"), y=("y", "max"))


def port_video(ds: str) -> pd.DataFrame | None:
    d = CROSS / f"clip_{DATASETS[ds]}"
    parts = sorted(d.glob("*.parquet"))
    if not parts:
        return None
    df = pd.concat([pd.read_parquet(x) for x in parts], ignore_index=True)
    return pd.DataFrame({"v": video_id(df["key"]), "p": df["p_sem"], "y": df["label"]}).groupby(
        "v", as_index=False).agg(p=("p", "mean"), y=("y", "max"))


STEP2 = Path("logs/tbiom/step2")


def dicome_val(arm: str, col: str):
    """FF++ VAL scores for a DiCoME readout, exported by Step 2."""
    f = STEP2 / f"{arm}_FFpp_val.csv"
    if not f.is_file():
        return None
    df = pd.read_csv(f)
    if col not in df:
        return None
    return pd.DataFrame({"v": video_id(df["key"]), "p": df[col], "y": df["label"]}).groupby(
        "v", as_index=False).agg(p=("p", "mean"), y=("y", "max"))


def port_val():
    f = Path("logs/tbiom/score/clip_ffppval_seeded/per_sample_epoch_7.parquet")
    if not f.is_file():
        return None
    df = pd.read_parquet(f)
    return pd.DataFrame({"v": video_id(df["key"]), "p": df["p_sem"], "y": df["label"]}).groupby(
        "v", as_index=False).agg(p=("p", "mean"), y=("y", "max"))


# where each readout's FF++ VAL scores come from — the tau source, never the test split
VAL_SOURCE = {
    "DiCoME-released fused":    lambda: dicome_val("released", "p_fused"),
    "DiCoME-released semantic": lambda: dicome_val("released", "p_semantic"),
    "DiCoME-released artifact": lambda: dicome_val("released", "p_artifact"),
    "DiCoME-P0DS fused":        lambda: dicome_val("p0ds_e01", "p_fused"),
    "DiCoME-P0DS semantic":     lambda: dicome_val("p0ds_e01", "p_semantic"),
    "CLIP port":                port_val,
}

READOUTS = [
    ("DiCoME-released fused",   lambda ds: dicome_video(ds, "released", "p_fused")),
    ("DiCoME-released semantic", lambda ds: dicome_video(ds, "released", "p_semantic")),
    ("DiCoME-released artifact", lambda ds: dicome_video(ds, "released", "p_artifact")),
    ("DiCoME-P0DS fused",       lambda ds: dicome_video(ds, "p0ds", "p_fused")),
    ("DiCoME-P0DS semantic",    lambda ds: dicome_video(ds, "p0ds", "p_semantic")),
    ("CLIP port",               port_video),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("tbiom/STEP1_ANCHOR_DIAG.md"))
    args = ap.parse_args()

    # --- correctness check: does the exporter reproduce DiCoME's own eval path? --------------
    checks = []
    for ds, published in PUBLISHED_FUSED.items():
        v = dicome_video(ds, "released", "p_fused")
        if v is None:
            checks.append((ds, published, None, None))
            continue
        got = auroc(v["y"].to_numpy(), v["p"].to_numpy())
        checks.append((ds, published, got, got - published))

    # --- one tau per readout, from FF++ VAL -----------------------------------------------------
    # FF++ VAL, not FF++ test. Deriving tau from the test split was a real error in the first
    # version of this file and it inflated a finding: the released checkpoint happened to take
    # tau 0.5639 from FF++ test against P0-DS's 0.4279, and a higher threshold mechanically calls
    # fewer reals fake. That alone produced "P0-DS mean OOD FPR 0.317 vs released 0.116". On the
    # permitted source the same numbers are 0.270 vs 0.222 — a real but modest gap, not a
    # dramatic one. The threshold must come from development data for the same reason it must be
    # frozen: otherwise it is fitted to what it is being used to judge.
    taus = {}
    for name, fn in READOUTS:
        ff = VAL_SOURCE.get(name, lambda: None)()
        if ff is None:
            continue
        _, tau = eer(ff["y"].to_numpy(), ff["p"].to_numpy())
        taus[name] = tau

    cells = {}
    for name, fn in READOUTS:
        if name not in taus:
            continue
        for ds in DATASETS:
            v = fn(ds)
            if v is None:
                continue
            cells[(name, ds)] = dashboard(v["y"].to_numpy(), v["p"].to_numpy(), taus[name])

    # --- report -------------------------------------------------------------------------------
    lines = ["# Step 1 — like-for-like anchor comparison", "",
             "Does the Celeb-DF-v2 anchor gap belong to the CLIP branch, or to DiCoME's "
             "decomposition and internal DS fusion? Five readouts, one exporter, the same six "
             "datasets.", "",
             "## 0. Correctness check — does the new exporter reproduce DiCoME's own eval path?",
             "",
             "`p_fused` here must match `DiCoME/eval_adaptation/RESULTS.md`, which came from "
             "DiCoME's untouched `test_step`. If it does not, the exporter is reading the model "
             "wrong and nothing below means anything.", "",
             "| dataset | published | this exporter | delta |", "|---|---:|---:|---:|"]
    for ds, pub, got, delta in checks:
        if got is None:
            lines.append(f"| {ds} | {pub:.4f} | TODO(run) | — |")
        else:
            flag = "" if abs(delta) < 0.005 else "  ⚠️"
            lines.append(f"| {ds} | {pub:.4f} | {got:.4f} | {delta:+.4f}{flag} |")

    lines += ["", "## 1. Video AUROC", "",
              "| readout | " + " | ".join(DATASETS) + " |",
              "|---" * (len(DATASETS) + 1) + "|"]
    for name, _ in READOUTS:
        if name not in taus:
            continue
        row = [name]
        for ds in DATASETS:
            c = cells.get((name, ds))
            row.append("TODO(run)" if not c else f"{c['auroc']:.4f}")
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## 2. The operational column — FPR on REAL videos at a frozen tau", "",
              "One tau per readout, the EER on that readout's own **FF++ VAL** scores, frozen "
              "across every dataset below. Per-readout because the five are differently "
              "calibrated; a shared tau would measure calibration offset rather than operating "
              "quality. VAL and not test: an earlier version of this table took tau from FF++ "
              "test, which handed the released checkpoint a higher threshold and inflated the "
              "real-side gap from 0.048 to 0.201.", "",
              "| readout | tau | " + " | ".join(DATASETS) + " |",
              "|---|---:" + "|---:" * len(DATASETS) + "|"]
    for name, _ in READOUTS:
        if name not in taus:
            continue
        row = [name, f"{taus[name]:.4f}"]
        for ds in DATASETS:
            c = cells.get((name, ds))
            row.append("—" if not c else f"{c['fpr_real_at_tau']:.3f}")
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## 3. Probability separation `d_RF`", "",
              "| readout | " + " | ".join(DATASETS) + " |",
              "|---" * (len(DATASETS) + 1) + "|"]
    for name, _ in READOUTS:
        if name not in taus:
            continue
        row = [name]
        for ds in DATASETS:
            c = cells.get((name, ds))
            row.append("—" if not c else f"{c['delta_rf']:+.3f}")
        lines.append("| " + " | ".join(row) + " |")

    # --- the verdict the brief asks for --------------------------------------------------------
    sem = cells.get(("DiCoME-released semantic", "CDFv2"))
    fus = cells.get(("DiCoME-released fused", "CDFv2"))
    port = cells.get(("CLIP port", "CDFv2"))
    lines += ["", "## 4. Verdict", ""]
    if sem and fus and port:
        d_sem_port = sem["auroc"] - port["auroc"]
        d_fus_sem = fus["auroc"] - sem["auroc"]
        lines += [
            f"On Celeb-DF-v2: DiCoME-fused **{fus['auroc']:.4f}**, DiCoME-semantic-only "
            f"**{sem['auroc']:.4f}**, our CLIP port **{port['auroc']:.4f}**.", "",
            f"- semantic-only minus port: **{d_sem_port:+.4f}** — the CLIP *recipe* gap.",
            f"- fused minus semantic-only: **{d_fus_sem:+.4f}** — what the artifact view and "
            f"internal DS fusion add.", "",
        ]
        if abs(d_sem_port) < 0.015:
            lines.append("**No meaningful CLIP implementation gap.** The port reproduces DiCoME's "
                         "semantic branch; the difference is architectural, created by the "
                         "decomposition and DS fusion. Step 2 is therefore a chassis choice, not "
                         "a recipe search, and hunting LoRA/optimizer/augmentation settings would "
                         "be chasing a gap that is not there.")
        elif d_sem_port > 0.015:
            lines.append("**A genuine recipe gap exists.** DiCoME's own CLIP branch beats our "
                         "port by more than the noise band on identical LoRA settings, so the "
                         "difference is in the training recipe — batch size (128 vs 32), "
                         "precision (bf16-mixed vs fp32), or the VAE/alignment loss terms shaping "
                         "the shared encoder. Investigate those before choosing a chassis.")
        else:
            lines.append("**Our port beats DiCoME's own semantic branch**, which inverts the "
                         "premise of this step: the gap is not a CLIP deficit at all and lives "
                         "entirely in DiCoME's extra machinery.")
    else:
        lines.append("TODO(run) — exports incomplete.")

    # --- the operational finding, which the AUROC table hides ----------------------------------
    ood = [d for d in DATASETS if d != "FFpp"]
    def mean_fpr(name: str) -> float | None:
        vals = [cells[(name, d)]["fpr_real_at_tau"] for d in ood if (name, d) in cells]
        return float(np.mean(vals)) if vals else None

    rel, p0, prt = (mean_fpr("DiCoME-released fused"), mean_fpr("DiCoME-P0DS fused"),
                    mean_fpr("CLIP port"))
    if rel is not None and p0 is not None and prt is not None:
        lines += ["", "## 5. The operational column", "",
                  f"Mean FPR on REAL videos across the five OOD sets, tau frozen on FF++ VAL: "
                  f"our CLIP port **{prt:.3f}**, DiCoME-released **{rel:.3f}**, DiCoME-P0DS "
                  f"**{p0:.3f}**.", "",
                  "**Correction to an earlier version of this table.** It took tau from FF++ "
                  "TEST, which handed the released checkpoint a threshold of 0.5639 against "
                  "P0-DS's 0.4279 — and a higher threshold mechanically calls fewer reals fake. "
                  "That produced 'released 0.116 vs P0-DS 0.317' and a conclusion that P0-DS had "
                  "badly broken real-side health. On the permitted development source the gap is "
                  f"{p0 - rel:+.3f}, not 0.201. The threshold must come from development data for "
                  "the same reason it must be frozen: otherwise it is fitted to the thing it is "
                  "being used to judge.", "",
                  "What survives the correction: P0-DS is still the weakest of the three on "
                  "real-side health, and our CLIP port is now level with the released checkpoint "
                  "rather than far behind it. So the port's deficit is a RANKING deficit "
                  "(-0.045 AUROC on Celeb-DF-v2), not an operational one — which sharpens the "
                  "recipe question rather than answering it, and is consistent with the gap "
                  "living inside the learned representation."]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(
        {"taus": taus, "checks": [list(c) for c in checks],
         "cells": {f"{k[0]}|{k[1]}": v for k, v in cells.items()}}, indent=2, default=float))
    print("\n".join(lines[-14:]))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

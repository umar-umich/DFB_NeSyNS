#!/usr/bin/env python3
"""Stages 2-4 — the Branch-2 comparison and the decision rule.

    python analysis/tbiom/stage23_compare.py --out tbiom/STAGE234_BRANCH2.md

Four arms on one dashboard, all seven datasets, all three readouts:

    P0-DS       beta-VAE projector,  64-d artifact              the incumbent Branch 2
    P1d         MR-VAE projector,    69-d artifact (+ rate)     both changes at once
    Stage 2     MR-VAE projector,    64-d artifact              the projector ALONE
    Stage 3     beta-VAE projector,  69-d artifact (+ rate)     the rate curve ALONE

Stage 2 and Stage 3 are the two halves of P1d. Reading them against P0-DS and P1d together is
what attributes DFDCP's +0.038 and CDFv3's swing to one change or the other, which no existing
table can do.

tau is frozen per arm and per readout on that arm's OWN FF++ val -- never on a test split, and
never shared between arms, which are differently calibrated.

Stage 4's decision rule is then evaluated from these numbers rather than by eye. The standing
instruction is that the updated beta-VAE (Stage 3) is the DEFAULT Branch 2 and ships even if its
gain is not significant, subject to one guardrail: it must not degrade CDFv3 AUROC or real-side
FPR beyond the noise band. Neutral is acceptable, regression is not.
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
from video_id import video_id  # noqa: E402

MRVAE, STEP1, STEP2, S23 = (Path("logs/tbiom/mrvae_anchor"), Path("logs/tbiom/step1"),
                            Path("logs/tbiom/step2"), Path("logs/tbiom/stage23"))
OOD = ["CDFv2", "CDFv3", "DFD", "DFDC", "DFDCP", "DFEval24"]
DATASETS = OOD + ["VALmix"]
READOUTS = ["p_fused", "p_artifact", "p_semantic"]
NICE = {"p_fused": "fused (DS)", "p_artifact": "artifact view", "p_semantic": "semantic (CLIP)"}
BAND = 0.01   # the brief's significance band: |dAUC| < 0.01 needs a confirming seed

ARMS = {
    "P0-DS (β-VAE)":  (STEP2 / "p0ds_e01_FFpp_val.csv",
                       {d: STEP1 / f"p0ds_{d}.csv" for d in OOD} |
                       {"VALmix": STEP2 / "p0ds_e01_VALmix.csv"}),
    "P1d (MR-VAE+rate)": (MRVAE / "p1d_FFpp_val.csv",
                       {d: MRVAE / f"p1d_{d}.csv" for d in OOD} |
                       {"VALmix": MRVAE / "p1d_VALmix.csv"}),
    "Stage 2 (MR-VAE proj only)": (S23 / "stage2_FFpp_val.csv",
                       {d: S23 / f"stage2_{d}.csv" for d in DATASETS}),
    "Stage 3 (β-VAE + rate)": (S23 / "stage3_FFpp_val.csv",
                       {d: S23 / f"stage3_{d}.csv" for d in DATASETS}),
}
BASE, RATE = "P0-DS (β-VAE)", "Stage 3 (β-VAE + rate)"
PROJ, BOTH = "Stage 2 (MR-VAE proj only)", "P1d (MR-VAE+rate)"


def vid(path: Path, col: str) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    d = pd.read_csv(path)
    if col not in d:
        return None
    return (pd.DataFrame({"v": video_id(d["key"]), "p": d[col], "y": d["label"]})
            .groupby("v", as_index=False).agg(p=("p", "mean"), y=("y", "max")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("tbiom/STAGE234_BRANCH2.md"))
    args = ap.parse_args()

    taus, cells, missing = {}, {}, []
    for arm, (ffval, paths) in ARMS.items():
        for col in READOUTS:
            ff = vid(ffval, col)
            if ff is None:
                missing.append(f"{arm}/{col}: no FF++ val export")
                continue
            _, tau = eer(ff["y"].to_numpy(), ff["p"].to_numpy())
            taus[(arm, col)] = tau
            for ds in DATASETS:
                v = vid(paths[ds], col)
                if v is None or v["y"].nunique() < 2:
                    missing.append(f"{arm}/{col}/{ds}")
                    continue
                cells[(arm, col, ds)] = dashboard(v["y"].to_numpy(), v["p"].to_numpy(), tau)

    L = ["# Stages 2-4 — Branch 2: the projector and the rate curve, taken apart", "",
         "| arm | projector | artifact | what it isolates |", "|---|---|---|---|",
         "| **P0-DS** | β-VAE | 64-d | the incumbent Branch 2 |",
         "| **P1d** | MR-VAE | 69-d (+rate) | both changes at once |",
         "| **Stage 2** | MR-VAE | 64-d | the **projector** alone |",
         "| **Stage 3** | β-VAE | 69-d (+rate) | the **rate curve** alone |", "",
         "`τ` frozen per arm **and** per readout on that arm's own FF++ val. Arms are "
         "differently calibrated, so a shared threshold would not compare operating points.", ""]

    for col in READOUTS:
        have = [a for a in ARMS if (a, col) in taus]
        if len(have) < 2:
            continue
        L += [f"## {NICE[col]}", "",
              "| dataset | " + " | ".join(f"{a} AUROC" for a in have) + " | " +
              " | ".join(f"{a} FPR_real" for a in have) + " |",
              "|---" * (2 * len(have) + 1) + "|"]
        for ds in DATASETS:
            au = [f"{cells[(a,col,ds)]['auroc']:.4f}" if (a, col, ds) in cells else "TODO(run)"
                  for a in have]
            fp = [f"{cells[(a,col,ds)]['fpr_real_at_tau']:.3f}" if (a, col, ds) in cells else "—"
                  for a in have]
            L.append(f"| {ds} | " + " | ".join(au) + " | " + " | ".join(fp) + " |")
        L.append("")
        L += ["| arm | mean AUROC (7) | mean FPR_real (7) | Δ AUROC vs P0-DS | Δ FPR vs P0-DS |",
              "|---|---:|---:|---:|---:|"]
        base_a = [cells[(BASE, col, d)]["auroc"] for d in DATASETS if (BASE, col, d) in cells]
        base_f = [cells[(BASE, col, d)]["fpr_real_at_tau"] for d in DATASETS if (BASE, col, d) in cells]
        for a in have:
            ds_ok = [d for d in DATASETS if (a, col, d) in cells and (BASE, col, d) in cells]
            if not ds_ok:
                continue
            ma = np.mean([cells[(a, col, d)]["auroc"] for d in ds_ok])
            mf = np.mean([cells[(a, col, d)]["fpr_real_at_tau"] for d in ds_ok])
            da = ma - np.mean([cells[(BASE, col, d)]["auroc"] for d in ds_ok])
            df_ = mf - np.mean([cells[(BASE, col, d)]["fpr_real_at_tau"] for d in ds_ok])
            L.append(f"| {a} | {ma:.4f} | {mf:.3f} | {da:+.4f} | {df_:+.3f} |")
        L.append("")

    # ---- attribution: which half of P1d carries DFDCP and CDFv3 --------------------------------
    col = "p_fused"
    L += ["## Attribution — which half of P1d moved DFDCP and CDFv3", "",
          "| dataset | P0-DS | Stage 2 (projector) | Stage 3 (rate) | P1d (both) | "
          "Δ projector | Δ rate | additive? |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for ds in ("DFDCP", "CDFv3", "CDFv2", "DFD", "DFDC", "DFEval24", "VALmix"):
        got = {a: cells.get((a, col, ds)) for a in (BASE, PROJ, RATE, BOTH)}
        if not all(got.values()):
            L.append(f"| {ds} | " + " | ".join(
                f"{got[a]['auroc']:.4f}" if got[a] else "TODO(run)"
                for a in (BASE, PROJ, RATE, BOTH)) + " | | | |")
            continue
        b = got[BASE]["auroc"]
        dp, dr = got[PROJ]["auroc"] - b, got[RATE]["auroc"] - b
        actual = got[BOTH]["auroc"] - b
        gap = actual - (dp + dr)
        L.append(f"| {ds} | {b:.4f} | {got[PROJ]['auroc']:.4f} | {got[RATE]['auroc']:.4f} | "
                 f"{got[BOTH]['auroc']:.4f} | {dp:+.4f} | {dr:+.4f} | "
                 f"{'yes' if abs(gap) < BAND else f'no ({gap:+.4f})'} |")
    L += ["", "\"Additive?\" asks whether P1d's effect equals the sum of its two halves. Where it "
              "does not, the projector and the rate curve interact and neither can be read alone.",
          ""]

    # ---- Stage 4 decision rule ----------------------------------------------------------------
    L += ["## Stage 4 — the Branch-2 decision", "",
          "Standing instruction: the updated β-VAE (**Stage 3**) is the **default** Branch 2 and "
          "ships even if its gain is not significant, for differentiation and possible real-side "
          "robustness — subject to one guardrail. It must not degrade **CDFv3 AUROC** or "
          "**real-side FPR** beyond the noise band. *Neutral is acceptable, regression is not.*",
          ""]
    g = {ds: (cells.get((RATE, col, ds)), cells.get((BASE, col, ds))) for ds in DATASETS}
    if all(a and b for a, b in g.values()):
        c3a, c3b = g["CDFv3"]
        d_c3 = c3a["auroc"] - c3b["auroc"]
        d_fpr_all = [g[d][0]["fpr_real_at_tau"] - g[d][1]["fpr_real_at_tau"] for d in DATASETS]
        d_fpr_c3 = g["CDFv3"][0]["fpr_real_at_tau"] - g["CDFv3"][1]["fpr_real_at_tau"]
        worse_fpr = [d for d in DATASETS
                     if g[d][0]["fpr_real_at_tau"] - g[d][1]["fpr_real_at_tau"] > 0.02]
        L += [f"- CDFv3 AUROC: **{d_c3:+.4f}** "
              f"({'within the ±0.01 band' if abs(d_c3) < BAND else 'outside the band'})",
              f"- CDFv3 real-side FPR: **{d_fpr_c3:+.3f}**",
              f"- real-side FPR across all seven: mean **{np.mean(d_fpr_all):+.3f}**, "
              f"worse by >0.02 on **{len(worse_fpr)}** dataset(s)"
              + (f" ({', '.join(worse_fpr)})" if worse_fpr else ""), ""]
        regress = (d_c3 < -BAND) or (d_fpr_c3 > 0.02) or len(worse_fpr) > 2
        if not regress:
            L.append("**SHIP Stage 3 as Branch 2.** The rate response is neutral-or-better on the "
                     "strong axes, so the standing instruction applies and the updated β-VAE is "
                     "adopted. Report its contribution honestly — if the AUROC gain is inside the "
                     "band, say \"comparable AUROC with a real-side benefit\" and do not sell it "
                     "as a driver of gains it did not produce.")
        else:
            L.append("**DO NOT ship the regression.** Stage 3 degrades a strong axis beyond the "
                     "noise band. Apply the guardrail: enable `rate_gate: true` (a learnable "
                     "scalar on the rate block, already implemented) or demote the rate curve to "
                     "an auxiliary explanatory output not fused into `e_art`. Branch 2 then stays "
                     "β-VAE with the rate response present but non-degrading.")
        if (PROJ, col, "CDFv3") in cells:
            dp_c3 = cells[(PROJ, col, "CDFv3")]["auroc"] - c3b["auroc"]
            ds_ok = [d for d in DATASETS if (PROJ, col, d) in cells]
            mp = np.mean([cells[(PROJ, col, d)]["auroc"] for d in ds_ok])
            mr = np.mean([cells[(RATE, col, d)]["auroc"] for d in ds_ok if (RATE, col, d) in cells])
            L += ["", f"Third clause of the rule — if Stage 2's projector beats Stage 3 on the "
                      f"full framework **and** does not hurt CDFv3, it may be Branch 2 instead. "
                      f"Stage 2 mean AUROC {mp:.4f} vs Stage 3 {mr:.4f}; Stage 2 CDFv3 "
                      f"{dp_c3:+.4f} vs P0-DS. "
                      + ("**Stage 2 qualifies and should be considered.**"
                         if (mp > mr and dp_c3 > -BAND) else "**Stage 2 does not qualify.**")]
    else:
        L.append("TODO(run) — Stage 2/3 exports incomplete; decision deferred.")

    if missing:
        L += ["", "## Missing exports", ""] + [f"- {m}" for m in missing[:40]]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(
        {"taus": {f"{k[0]}|{k[1]}": v for k, v in taus.items()},
         "cells": {"|".join(k): v for k, v in cells.items()}}, indent=2, default=float))
    print(f"wrote {args.out}  ({len(cells)} cells, {len(missing)} missing)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

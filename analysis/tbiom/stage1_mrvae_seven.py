#!/usr/bin/env python3
"""Stage 1 — MR-VAE health re-score across all seven datasets.

    python analysis/tbiom/stage1_mrvae_seven.py --out tbiom/STAGE1_MRVAE_SEVEN.md

THE GAP THIS CLOSES. MR-VAE's operational advantage was established on only three of the seven
datasets (CDFv2, DFDCP, VALmix). Its AUROC losses on CDFv3 (-0.066 as first reported) and
DFEval24 (-0.027) were never re-scored on real-side metrics at all, so the claim "lower FPR
everywhere" rested on a subset that happened to exclude the two datasets where it lost most.
CDFv3 and DFEval24 are therefore the decisive rows here, and they are marked as such.

No GPU work: every export already exists. Both anchors are scored by the same code path, on the
same video grouping, with tau frozen per anchor and per readout on that anchor's OWN FF++ val.
Freezing tau on anything else would let one anchor's operating point be chosen on the data it is
being judged against -- the error that inflated Step 1's first operational table.
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

MRVAE = Path("logs/tbiom/mrvae_anchor")
STEP1, STEP2 = Path("logs/tbiom/step1"), Path("logs/tbiom/step2")

# anchor -> (ffpp_val export, {dataset: export})
ARMS = {
    "MR-VAE (P1d)": (
        MRVAE / "p1d_FFpp_val.csv",
        {d: MRVAE / f"p1d_{d}.csv" for d in
         ("CDFv2", "CDFv3", "DFD", "DFDC", "DFDCP", "DFEval24")} | {"VALmix": MRVAE / "p1d_VALmix.csv"},
    ),
    "beta-VAE (P0-DS)": (
        STEP2 / "p0ds_e01_FFpp_val.csv",
        {d: STEP1 / f"p0ds_{d}.csv" for d in
         ("CDFv2", "CDFv3", "DFD", "DFDC", "DFDCP", "DFEval24")} | {"VALmix": STEP2 / "p0ds_e01_VALmix.csv"},
    ),
}
DATASETS = ["CDFv2", "CDFv3", "DFD", "DFDC", "DFDCP", "DFEval24", "VALmix"]
DECISIVE = {"CDFv3", "DFEval24"}
READOUTS = ["p_fused", "p_artifact", "p_semantic"]
NICE = {"p_fused": "fused (DS)", "p_artifact": "artifact view", "p_semantic": "semantic (CLIP)"}

# Within-band per the brief's significance rule; a delta inside it is not a finding.
BAND = 0.01


def load(path: Path, col: str) -> pd.DataFrame | None:
    """Frame-level export -> video-level table, using THE grouping rule."""
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    if col not in df:
        return None
    return (pd.DataFrame({"v": video_id(df["key"]), "p": df[col], "y": df["label"]})
            .groupby("v", as_index=False).agg(p=("p", "mean"), y=("y", "max")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("tbiom/STAGE1_MRVAE_SEVEN.md"))
    args = ap.parse_args()

    taus: dict[tuple[str, str], float] = {}
    cells: dict[tuple[str, str, str], dict] = {}
    for arm, (ffval_path, ds_paths) in ARMS.items():
        for col in READOUTS:
            ffval = load(ffval_path, col)
            if ffval is None:
                print(f"  !! no FF++ val for {arm}/{col}")
                continue
            _, tau = eer(ffval["y"].to_numpy(), ffval["p"].to_numpy())
            taus[(arm, col)] = tau
            for ds in DATASETS:
                v = load(ds_paths[ds], col)
                if v is None or v["y"].nunique() < 2:
                    print(f"  !! missing {arm}/{col}/{ds}")
                    continue
                cells[(arm, col, ds)] = dashboard(v["y"].to_numpy(), v["p"].to_numpy(), tau)
            print(f"  {arm:18s} {col:12s} tau {tau:.4f}  "
                  f"({sum(1 for d in DATASETS if (arm, col, d) in cells)}/7 datasets)")

    A, B = "MR-VAE (P1d)", "beta-VAE (P0-DS)"
    L = ["# Stage 1 — MR-VAE health re-score, all seven datasets", "",
         "Closes the load-bearing gap: MR-VAE's operational advantage was established on only "
         "**three of seven** datasets (CDFv2, DFDCP, VALmix), and its AUROC losses on **CDFv3** "
         "and **DFEval24** were never re-scored on real-side metrics. Those two rows are marked "
         "**decisive** below.", "",
         "`tau` frozen per anchor **and per readout** on that anchor's own FF++ val, applied "
         "unchanged to all seven. No GPU work — every export already existed; both anchors go "
         "through one code path and one video-grouping rule.", ""]

    for col in READOUTS:
        if (A, col) not in taus or (B, col) not in taus:
            continue
        L += [f"## {NICE[col]}", "",
              f"τ: MR-VAE **{taus[(A,col)]:.4f}**, β-VAE **{taus[(B,col)]:.4f}**", "",
              "| dataset | | n | MR-VAE AUROC | β-VAE AUROC | Δ AUROC | **MR-VAE FPR_real** | "
              "**β-VAE FPR_real** | **Δ FPR** | Δ EER | Δ d_RF |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        d_auc, d_fpr, wins = [], [], 0
        for ds in DATASETS:
            a, b = cells.get((A, col, ds)), cells.get((B, col, ds))
            if not a or not b:
                L.append(f"| {ds} | | TODO(run) | | | | | | | | |")
                continue
            da = a["auroc"] - b["auroc"]
            df_ = a["fpr_real_at_tau"] - b["fpr_real_at_tau"]
            de = a["eer"] - b["eer"]
            dd = a["delta_rf"] - b["delta_rf"]
            d_auc.append(da); d_fpr.append(df_)
            wins += int(df_ < 0)
            mark = "**decisive**" if ds in DECISIVE else ""
            L.append(f"| {ds} | {mark} | {a['n_videos']} | {a['auroc']:.4f} | {b['auroc']:.4f} | "
                     f"{da:+.4f} | **{a['fpr_real_at_tau']:.3f}** | {b['fpr_real_at_tau']:.3f} | "
                     f"**{df_:+.3f}** | {de:+.4f} | {dd:+.3f} |")
        if d_auc:
            L += ["", f"Mean Δ AUROC **{np.mean(d_auc):+.4f}**, mean Δ FPR_real "
                      f"**{np.mean(d_fpr):+.3f}**. MR-VAE has the lower real-side FPR on "
                      f"**{wins} of {len(d_fpr)}** datasets.", ""]

    # --- the pass condition, evaluated on the fused readout ------------------------------------
    L += ["## Pass condition and branch", ""]
    col = "p_fused"
    dec = {ds: (cells.get((A, col, ds)), cells.get((B, col, ds))) for ds in DECISIVE}
    if all(a and b for a, b in dec.values()):
        L += ["The brief's rule: if MR-VAE's real-side advantage holds broadly **and** CDFv3 / "
              "DFEval24 carry no large FPR penalty, the rate response is a live Branch-2 "
              "ingredient. If either collapses on real-side FPR the way its AUROC did, the rate "
              "response must be gated per the Branch-2 rule.", "",
              "| decisive dataset | Δ AUROC | Δ FPR_real | AUROC verdict | real-side verdict |",
              "|---|---:|---:|---|---|"]
        penalty = False
        for ds, (a, b) in dec.items():
            da = a["auroc"] - b["auroc"]
            dfp = a["fpr_real_at_tau"] - b["fpr_real_at_tau"]
            av = "within band" if abs(da) < BAND else ("MR-VAE better" if da > 0 else "MR-VAE worse")
            rv = ("**penalty**" if dfp > 0.02 else
                  "no penalty" if abs(dfp) <= 0.02 else "MR-VAE better")
            penalty |= dfp > 0.02
            L.append(f"| {ds} | {da:+.4f} | {dfp:+.3f} | {av} | {rv} |")
        allf = [cells[(A, col, d)]["fpr_real_at_tau"] - cells[(B, col, d)]["fpr_real_at_tau"]
                for d in DATASETS if (A, col, d) in cells and (B, col, d) in cells]
        broad = sum(1 for x in allf if x < 0)
        L += ["", f"Broad real-side advantage: MR-VAE lower FPR on **{broad}/{len(allf)}** "
                  f"datasets, mean **{np.mean(allf):+.3f}**.", ""]
        if not penalty and broad >= len(allf) - 1:
            L.append("**PASS — the rate response is a live Branch-2 ingredient.** The real-side "
                     "advantage holds broadly, and neither decisive dataset carries an FPR "
                     "penalty: the AUROC losses on CDFv3/DFEval24 are *not* accompanied by "
                     "real-side collapse. Stage 3 (β-VAE artifact + MR-VAE rate curve) proceeds "
                     "as the preferred Branch 2.")
        elif penalty:
            L.append("**GATED — a decisive dataset carries a real-side FPR penalty.** The rate "
                     "response cannot enter as unconditional fused evidence on that axis; apply "
                     "the Stage-4 guardrail (learnable scalar gate on the rate block, or demote "
                     "the rate curve to an auxiliary explanatory output).")
        else:
            L.append("**MIXED — the real-side advantage does not hold broadly.** Record and carry "
                     "into the Stage-4 decision rule rather than treating the rate response as "
                     "established.")
    else:
        L.append("TODO(run) — decisive-row exports incomplete.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(
        {"taus": {f"{k[0]}|{k[1]}": v for k, v in taus.items()},
         "cells": {"|".join(k): v for k, v in cells.items()}}, indent=2, default=float))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

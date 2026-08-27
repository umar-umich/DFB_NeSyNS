#!/usr/bin/env python3
"""Step 3 — full health dashboard for the chosen anchor, across every domain.

    python analysis/tbiom/step3_anchor_health.py --out tbiom/STEP3_ANCHOR_HEALTH.md

Anchor: P0-DS epoch 1, the provisional retrainable chassis. All THREE readouts are scored as
genuine anchor candidates — semantic, artifact and fused — not as a ceiling and a reference.
Step 1 found artifact-only beating fused on Celeb-DF-v2, DFD and DFDC, and Step 2b found it with
the best real-side health in both ablation arms, so treating DiCoME's fused output as the
anchor by default would be inheriting a choice the evidence does not support.

THE DECISIVE QUESTION, from the brief: does the FF++-only anchor show real-side collapse on
UNFAMILIAR REALS? That is what the FF++ (+) DF40 run failed, and it is what decides whether
Step 9's corpus ablation is needed at all.

    real-side health intact  -> the corpus is fine, Step 9 is skipped
    real-side collapse       -> Step 9 is triggered later

`tau` is frozen once per readout on FF++ VAL and applied unchanged everywhere. Deriving it from
any test split would fit the operating point to the thing being judged — the error that inflated
Step 1's first operational table.

Domains are labelled by what the anchor has and has not seen, because "OOD" is not one thing:

    in-domain     FF++            trained on it
    development   VALmix          selection only, never gradient
    zero-shot     the rest        no contact of any kind
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

STEP1, STEP2 = Path("logs/tbiom/step1"), Path("logs/tbiom/step2")
ARM = "p0ds"                      # step1 exports; step2 uses the fuller `p0ds_e01` prefix
READOUTS = ["p_semantic", "p_artifact", "p_fused"]
NICE = {"p_semantic": "semantic (CLIP branch)", "p_artifact": "artifact view",
        "p_fused": "fused (DS)"}

# domain -> (file stem, provenance). Provenance is the point of the table.
DOMAINS = [
    ("FF++",               ("step1", "FFpp"),     "in-domain (trained)"),
    ("VALmix",             ("step2", "VALmix"),   "development (selection only)"),
    ("Celeb-DF-v2",        ("step1", "CDFv2"),    "zero-shot"),
    ("Celeb-DF-v3",        ("step1", "CDFv3"),    "zero-shot"),
    ("DFD",                ("step1", "DFD"),      "zero-shot"),
    ("DFDC",               ("step1", "DFDC"),     "zero-shot"),
    ("DFDCP",              ("step1", "DFDCP"),    "zero-shot"),
    ("Deepfake-Eval-2024", ("step1", "DFEval24"), "zero-shot"),
]


def load(where: str, stem: str, col: str) -> pd.DataFrame | None:
    p = (STEP1 / f"{ARM}_{stem}.csv") if where == "step1" else (STEP2 / f"{ARM}_e01_{stem}.csv")
    if not p.is_file():
        return None
    df = pd.read_csv(p)
    if col not in df:
        return None
    return pd.DataFrame({"v": video_id(df["key"]), "p": df[col], "y": df["label"]}).groupby(
        "v", as_index=False).agg(p=("p", "mean"), y=("y", "max"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("tbiom/STEP3_ANCHOR_HEALTH.md"))
    args = ap.parse_args()

    taus, cells = {}, {}
    for col in READOUTS:
        ffval = load("step2", "FFpp_val", col)
        if ffval is None:
            continue
        _, taus[col] = eer(ffval["y"].to_numpy(), ffval["p"].to_numpy())
        for name, (where, stem), _prov in DOMAINS:
            v = load(where, stem, col)
            if v is None or v["y"].nunique() < 2:
                continue
            cells[(col, name)] = dashboard(v["y"].to_numpy(), v["p"].to_numpy(), taus[col])

    zero_shot = [n for n, _, prov in DOMAINS if prov == "zero-shot"]

    def agg(col: str, key: str, names: list[str]) -> float:
        vals = [cells[(col, n)][key] for n in names if (col, n) in cells]
        return float(np.mean(vals)) if vals else float("nan")

    lines = ["# Step 3 — anchor health dashboard", "",
             "Anchor: **P0-DS epoch 1** (the provisional retrainable chassis). All three readouts "
             "scored as genuine anchor candidates.", "",
             "`tau` frozen once per readout on **FF++ VAL**, applied unchanged to every domain. "
             "Domains are labelled by what the anchor has seen, because \"OOD\" is not one thing.",
             ""]

    for col in READOUTS:
        if col not in taus:
            continue
        lines += [f"## {NICE[col]}   (tau = {taus[col]:.4f})", "",
                  "| domain | provenance | videos | AUROC | EER | **FPR_real@tau** | d_RF | mean p on real |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for name, _src, prov in DOMAINS:
            c = cells.get((col, name))
            if not c:
                lines.append(f"| {name} | {prov} | TODO(run) | | | | | |")
                continue
            lines.append(
                f"| {name} | {prov} | {c['n_videos']} | {c['auroc']:.4f} | {c['eer']:.4f} | "
                f"**{c['fpr_real_at_tau']:.3f}** | {c['delta_rf']:+.3f} | "
                f"{c['mean_p_on_real']:.3f} |")
        lines += ["", f"Zero-shot means: AUROC **{agg(col,'auroc',zero_shot):.4f}**, "
                      f"FPR_real **{agg(col,'fpr_real_at_tau',zero_shot):.3f}**, "
                      f"d_RF **{agg(col,'delta_rf',zero_shot):+.3f}**.", ""]

    # --- readout choice ------------------------------------------------------------------------
    lines += ["## Which readout should be the anchor", "",
              "| readout | zero-shot AUROC | zero-shot FPR_real | zero-shot d_RF |",
              "|---|---:|---:|---:|"]
    for col in READOUTS:
        if col not in taus:
            continue
        lines.append(f"| {NICE[col]} | {agg(col,'auroc',zero_shot):.4f} | "
                     f"{agg(col,'fpr_real_at_tau',zero_shot):.3f} | "
                     f"{agg(col,'delta_rf',zero_shot):+.3f} |")

    # --- the pass condition --------------------------------------------------------------------
    lines += ["", "## Real-side verdict — does Step 9 trigger?", ""]
    ok = {}
    for col in READOUTS:
        if col not in taus:
            continue
        f_in = cells.get((col, "FF++"), {}).get("fpr_real_at_tau", float("nan"))
        f_zs = agg(col, "fpr_real_at_tau", zero_shot)
        d_in = cells.get((col, "FF++"), {}).get("delta_rf", float("nan"))
        d_zs = agg(col, "delta_rf", zero_shot)
        ok[col] = (f_in, f_zs, d_in, d_zs)
        lines.append(
            f"- **{NICE[col]}**: FPR_real {f_in:.3f} in-domain -> {f_zs:.3f} zero-shot "
            f"(x{f_zs/f_in:.1f} if finite); d_RF {d_in:+.3f} -> {d_zs:+.3f}, "
            f"retaining {100*d_zs/d_in:.0f}% of in-domain separation.")
    lines.append("")
    best = min((c for c in ok), key=lambda c: ok[c][1]) if ok else None
    if best:
        f_zs, d_zs = ok[best][1], ok[best][3]
        # The collapse signature, stated as what it was rather than as a threshold pulled from air:
        # the FF++(+)DF40 run reached FPR_real 1.000 with d_RF 0.000 on Celeb-DF.
        collapsed = f_zs > 0.5 or d_zs < 0.10
        if collapsed:
            lines.append(
                "**Real-side collapse persists.** The anchor stops recognising unfamiliar reals, "
                "so the corpus is implicated and **Step 9 is triggered** — the FFHQ/SBI ablation "
                "becomes necessary rather than optional.")
        else:
            lines.append(
                f"**Real-side health is intact.** The best readout ({NICE[best]}) holds FPR_real "
                f"at {f_zs:.3f} on domains whose reals it has never seen, against the collapse "
                f"signature of 1.000, and retains d_RF {d_zs:+.3f} against the collapsed 0.000. "
                f"Degradation is graded, not catastrophic. **The FF++-only corpus is fine and "
                f"Step 9 is skipped**; real-support asymmetry was a property of the FF++ (+) DF40 "
                f"manifest, not of FF++ training as such.")
    else:
        lines.append("TODO(run) — exports incomplete.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(
        {"taus": taus, "cells": {f"{k[0]}|{k[1]}": v for k, v in cells.items()}},
        indent=2, default=float))
    print("\n".join(lines[-10:]))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

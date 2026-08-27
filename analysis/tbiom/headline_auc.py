#!/usr/bin/env python3
"""The headline cross-dataset AUC table — the deliverable number.

    python analysis/tbiom/headline_auc.py --out tbiom/HEADLINE_AUC.md

Every anchor candidate x every readout, on every OOD test set scored so far, with the OOD mean
computed only over columns ALL rows share. A mean over different column sets is not a
comparison, and the two checkpoints were not scored on identical sets at first — the released
one was missing Celeb-DF-v3, which is the hardest column and would have flattered it.
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

STEP1, STEP2 = Path("logs/tbiom/step1"), Path("logs/tbiom/step2")
DATASETS = [("FF++", "FFpp", True), ("Celeb-DF-v2", "CDFv2", False),
            ("Celeb-DF-v3", "CDFv3", False), ("DFD", "DFD", False), ("DFDC", "DFDC", False),
            ("DFDCP", "DFDCP", False), ("Deepfake-Eval-2024", "DFEval24", False)]
ARMS = [("DiCoME released (frozen)", "released"), ("P0-DS (retrainable)", "p0ds")]
READOUTS = [("fused", "p_fused"), ("artifact", "p_artifact"), ("semantic", "p_semantic")]
PORT = {"FF++": 0.9909, "Celeb-DF-v2": 0.9225, "DFD": 0.9222, "DFDC": 0.8477,
        "DFDCP": 0.8912, "Deepfake-Eval-2024": 0.6357}


def score(arm: str, stem: str, col: str):
    f = STEP1 / f"{arm}_{stem}.csv"
    if not f.is_file():
        return None
    d = pd.read_csv(f)
    if col not in d:
        return None
    g = pd.DataFrame({"v": video_id(d["key"]), "p": d[col], "y": d["label"]}).groupby(
        "v", as_index=False).agg(p=("p", "mean"), y=("y", "max"))
    if g["y"].nunique() < 2:
        return None
    return auroc(g["y"].to_numpy(), g["p"].to_numpy()), int(len(g))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("tbiom/HEADLINE_AUC.md"))
    args = ap.parse_args()

    cells, nvid = {}, {}
    for a_label, a in ARMS:
        for r_label, col in READOUTS:
            key = f"{a_label} · {r_label}"
            for d_label, stem, _ in DATASETS:
                got = score(a, stem, col)
                if got:
                    cells[(key, d_label)] = got[0]
                    nvid[d_label] = got[1]

    keys = sorted({k for k, _ in cells})
    ood = [d for d, _, indom in DATASETS if not indom]
    shared = [d for d in ood if all((k, d) in cells for k in keys)]

    lines = ["# Headline cross-dataset AUC", "",
             "Video-level AUROC. **The OOD mean is computed only over the columns every row "
             "shares** — averaging over different column sets is not a comparison, and the "
             "released checkpoint was initially missing Celeb-DF-v3, the hardest column, which "
             "would have flattered it.", "",
             f"Shared OOD columns: {', '.join(shared)}.", "",
             "| model · readout | " + " | ".join(d for d, _, _ in DATASETS) + " | **OOD mean** |",
             "|---" * (len(DATASETS) + 2) + "|"]
    means = {}
    for k in keys:
        vals = [f"{cells[(k, d)]:.4f}" if (k, d) in cells else "—" for d, _, _ in DATASETS]
        m = float(np.mean([cells[(k, d)] for d in shared])) if shared else float("nan")
        means[k] = m
        lines.append(f"| {k} | " + " | ".join(vals) + f" | **{m:.4f}** |")

    # The CLIP port was never scored on Celeb-DF-v3, so its mean covers FEWER columns than the
    # model rows. Comparing them directly would be the same column-set error this table exists to
    # avoid, so the port delta is computed on the port's OWN columns and both sides restricted to
    # them.
    port_cols = [d for d in shared if d in PORT]
    port_mean = float(np.mean([PORT[d] for d in port_cols])) if port_cols else float("nan")
    port_vals = [f"{PORT[d]:.4f}" if d in PORT else "—" for d, _, _ in DATASETS]
    lines.append(f"| _CLIP port (where we started)_ | " + " | ".join(port_vals) +
                 f" | _{port_mean:.4f}_ |")

    best = max(means, key=means.get)
    best_on_port_cols = float(np.mean([cells[(best, d)] for d in port_cols]))
    lines += ["", "## Headline", "",
              f"**Best OOD mean: {best} at {means[best]:.4f}** over {len(shared)} columns "
              f"({', '.join(shared)}).", "",
              f"Against the CLIP port we started from, restricted to the "
              f"{len(port_cols)} columns the port has: **{best_on_port_cols:.4f}** vs "
              f"**{port_mean:.4f}** — **{best_on_port_cols - port_mean:+.4f}**. The port was "
              f"never scored on Celeb-DF-v3, so comparing its mean against a 6-column mean would "
              f"repeat the very error this table exists to avoid.", ""]
    if "Celeb-DF-v2" in cells.get((best, "Celeb-DF-v2"), None) if False else True:
        cdf = cells.get((best, "Celeb-DF-v2"))
        if cdf:
            lines.append(f"On Celeb-DF-v2 specifically: **{cdf:.4f}** against the port's "
                         f"{PORT['Celeb-DF-v2']:.4f} — **{cdf - PORT['Celeb-DF-v2']:+.4f}**.")
    retr = {k: v for k, v in means.items() if k.startswith("P0-DS")}
    froz = {k: v for k, v in means.items() if k.startswith("DiCoME released")}
    if retr and froz:
        br, bf = max(retr, key=retr.get), max(froz, key=froz.get)
        lines += ["", "## Frozen vs retrainable — the choice that matters later", "",
                  f"| | best readout | OOD mean |", "|---|---|---:|",
                  f"| frozen (released) | {bf.split('· ')[-1]} | {froz[bf]:.4f} |",
                  f"| retrainable (P0-DS) | {br.split('· ')[-1]} | {retr[br]:.4f} |", "",
                  f"Gap **{froz[bf] - retr[br]:+.4f}** on shared columns. The frozen checkpoint "
                  f"cannot be retrained, which forecloses any later corpus or recipe work; the "
                  f"retrainable one keeps that open for that cost."]

    lines += ["", "## Caveats that belong with these numbers", "",
              "- These are TEST splits. Nothing here was used to select a checkpoint or fit a "
              "threshold — selection ran on FF++ VAL_select and VALmix throughout.",
              "- Single seed. The spec's §22 rule wants a confirming seed for |ΔAUROC| < 0.01, "
              "which covers most of the gaps between readouts in this table.",
              "- AUROC alone hid an operational collapse once in this project. The real-side "
              "health for these same models is in `tbiom/STEP3_ANCHOR_HEALTH.md`; the artifact "
              "readout has markedly better zero-shot FPR_real (0.169) than fused (0.247) at "
              "similar AUROC."]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(
        {"cells": {f"{k}|{d}": v for (k, d), v in cells.items()}, "means": means,
         "shared_columns": shared}, indent=2, default=float))
    print("\n".join(lines[6:6 + len(keys) + 4]))
    print("\n".join(lines[-14:-8]))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

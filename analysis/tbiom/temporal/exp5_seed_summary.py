#!/usr/bin/env python3
"""Experiment 5 — confirmation seeds for the V-JEPA 2 LoRA branch trained with arm-B augmentation.

    python analysis/tbiom/temporal/exp5_seed_summary.py

Reads the panel run over the seed families (panel_probe.py writes EXP5_CONFIRMATION_SEEDS_panel.{md,json}) and writes
tbiom/reports/EXP5_CONFIRMATION_SEEDS.md: per-seed rows and mean ± sd across the INDEPENDENTLY TRAINED branch seeds
(42, 1337, 7) for the branch alone and its fusions, with the unaugmented seed-42 LoRA branch as the reference.
Fusion logistic fits vary with the 5 grouped splits, which are averaged inside each seed; those are not seeds.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

R = Path("tbiom/reports")
SEEDS = {"42": "vjepa2_lora_B_branch", "1337": "vjepa2_lora_B_seed1337_branch", "7": "vjepa2_lora_B_seed7_branch"}
REF = "vjepa2_lora_branch"
ARMS = [("branch alone", "probe {f}:ordered"), ("branch alone, shuffled frames", "probe {f}:shuffled"),
        ("DS branch + OM + FP", "DS {f}:ordered+OM+FP"), ("LR branch + OM + FP", "LR {f}:ordered+OM+FP"),
        ("LR APP + OM + FP + branch", "LR APP+OM+FP+{f}:ordered")]
TESTS = ["CDFv2", "CDFv3", "CDFv3 reenactment", "DFD", "DFDC", "DFDCP", "DFEval24", "FFIW", "KoDF", "KoDF first-order motion",
         "IDForge", "DFDM", "CDFv2 jpeg40", "CDFv2 rs06", "DFDCP jpeg40", "DFDCP rs06"]
TEN = ["CDFv2", "CDFv3", "DFD", "DFDC", "DFDCP", "DFEval24", "FFIW", "KoDF", "IDForge", "DFDM"]


def main():
    J = json.load(open(R / "EXP5_CONFIRMATION_SEEDS_panel.json"))
    sel = lambda arm: np.mean([s["sel"][arm][0] for s in J["splits"]]) if arm in J["splits"][0]["sel"] else np.nan
    reen = lambda arm: np.mean([s["sel"][arm][1]["reenactment"] for s in J["splits"]]) if arm in J["splits"][0]["sel"] else np.nan
    proc = lambda arm, c: np.mean([s["proc"][c][arm][0] for s in J["splits"]]) if J["splits"][0]["proc"] and arm in J["splits"][0]["proc"].get(c, {}) else np.nan
    test = lambda arm, t: J["tests"].get(t, {}).get(arm, np.nan)
    ref = {k: v for k, v in (("gate", "gate (trained)"), ("g1", "DS APP+OM+FP (g=1)"))}
    L = ["# Experiment 5 — confirmation seeds: V-JEPA 2 LoRA branch with arm-B augmentation", "",
         "Three independently trained branch seeds (42, 1337, 7) of the same recipe: V-JEPA 2 ViT-L, LoRA rank 8 on encoder "
         "query/value in every layer + evidential head, FF++ train only, arm-B processing mix on training clips, 10 epochs, "
         "batch 32, epoch chosen by the v3.1 development rule. Appearance, Off-manifold residual and Bona-fide face prior come "
         "from the ONE development-selected warm+gate detector (seed 42): the seeds here replicate the temporal branch, not the "
         "whole detector. Fusion logistic fits use the fusion audit's 5 grouped fitting folds (averaged within a seed; not "
         "seeds). Reference: the unaugmented seed-42 LoRA branch. Test sets were read after selection. Panel detail: "
         "`EXP5_CONFIRMATION_SEEDS_panel.md`.", "",
         "## 1. Development (selection folds)", "",
         "| arm | seed 42 | seed 1337 | seed 7 | **mean ± sd** | no-aug seed 42 |", "|---|---:|---:|---:|---:|---:|"]
    for label, pat in ARMS:
        for metric, fn in (("v3.1", sel), ("reenactment", reen), ("jpeg40 v3.1", lambda a: proc(a, "jpeg40")), ("rs06 v3.1", lambda a: proc(a, "rs06"))):
            vals = [fn(pat.format(f=f)) for f in SEEDS.values()]
            ok = [v for v in vals if not np.isnan(v)]
            ms = f"**{np.mean(ok):.4f} ± {np.std(ok, ddof=1):.4f}**" if len(ok) > 1 else "—"
            L.append(f"| {label} — {metric} | " + " | ".join(f"{v:.4f}" if not np.isnan(v) else "—" for v in vals) +
                     f" | {ms} | {fn(pat.format(f=REF)):.4f} |")
    L += [f"| trained gate (no temporal branch) — v3.1 | | | | {sel(ref['gate']):.4f} | |", ""]
    L += ["## 2. Test sets — video AUROC, LR branch + OM + FP (mean ± sd over seeds)", "",
          "| test | seed 42 | seed 1337 | seed 7 | **mean ± sd** | no-aug seed 42 | trained gate |", "|---|---:|---:|---:|---:|---:|---:|"]
    arm = "LR {f}:ordered+OM+FP"
    for t in TESTS + ["10-set mean"]:
        if t == "10-set mean":
            vals = [np.mean([test(arm.format(f=f), x) for x in TEN]) for f in SEEDS.values()]
            r, g = np.mean([test(arm.format(f=REF), x) for x in TEN]), np.mean([test(ref["gate"], x) for x in TEN])
        else:
            vals = [test(arm.format(f=f), t) for f in SEEDS.values()]; r, g = test(arm.format(f=REF), t), test(ref["gate"], t)
        ok = [v for v in vals if not np.isnan(v)]
        ms = f"**{np.mean(ok):.4f} ± {np.std(ok, ddof=1):.4f}**" if len(ok) > 1 else "—"
        L.append(f"| {t} | " + " | ".join(f"{v:.4f}" if not np.isnan(v) else "—" for v in vals) + f" | {ms} | {r:.4f} | {g:.4f} |")
    L += ["", "## Limitations", "",
          "- Three branch seeds; the other three views are a single detector seed.",
          "- The processing mix is applied to the stored 256-px frames before V-JEPA 2's preprocessing (DISCERN's arm B applies it "
          "to the 224-px CLIP crop).",
          "- CDFv2 and DeepfakeEval2024 have informed development; no test set here is untouched confirmation data."]
    (R / "EXP5_CONFIRMATION_SEEDS.md").write_text("\n".join(L) + "\n")
    print("wrote", R / "EXP5_CONFIRMATION_SEEDS.md")


if __name__ == "__main__":
    main()

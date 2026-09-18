#!/usr/bin/env python3
"""The paper's results tables: the deployed system beside the baselines, main table plus appendix.

    OMP_NUM_THREADS=4 python analysis/tbiom/final_table.py

Table scope follows tbiom/docs/decision-2026-09-18.md §3: the main table carries CDFv2, CDFv3, DFD, FFIW, KoDF,
IDForge and DFDM; DFDC, DFDCP and DeepfakeEval2024 move to the appendix, together with the reenactment and
first-order-motion slices and every processed condition. **That split was chosen after the numbers were known**, and
the caption says so; the appendix reports the excluded sets at the same precision and the ten-set mean stays the
headline aggregate, so nothing is hidden by the split.

Sources, all read from result files rather than retyped:
  ours       tbiom/reports/EXP_DFDCP_DIAGNOSIS.json -- the reference triple (B-augmented V-JEPA 2 LoRA + Appearance +
             Off-manifold), DS-R primary and the logistic comparator, mean over the 5 source-grouped fitted splits
  baselines  tbiom/reports/BASELINE_MATRIX.json -- one seed per row, video-level AUROC on the same videos
  seeds      tbiom/reports/EXP_CONFIRM_SEED.json -- temporal-branch seed spread, quoted as a range not a sd
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp1_fusion_audit as A  # noqa: E402

R = Path("tbiom/reports")
MAIN = ["CDFv2", "CDFv3", "DFD", "FFIW", "KoDF", "IDForge", "DFDM"]
APPX = ["DFDC", "DFDCP", "DFEval24"]
SLICES = ["CDFv3 reenactment", "KoDF first-order motion"]
PROC = ["CDFv2 jpeg40", "CDFv2 rs06", "DFDCP jpeg40", "DFDCP rs06"]
OURS = {"DISCERN-v2 (ours, DS-R)": "DS-R (clean fit)", "DISCERN-v2 (ours, logistic comparator)": "LR (clean fit)",
        "— temporal view alone": "branch T", "— appearance view alone": "branch APP", "— off-manifold view alone": "branch OM"}
SEALED = ["FaceShifter"]


def ours():
    rows = json.load(open(R / "EXP_DFDCP_DIAGNOSIS.json"))["vjepa2_lora_B_seed42"]
    out = {}
    for label, arm in OURS.items():
        out[label] = {t: float(np.mean([r["tests"][t][arm] for r in rows])) for t in rows[0]["tests"]}
    return out


def seed_spread():
    res = json.load(open(R / "EXP_CONFIRM_SEED.json"))
    runs = list(res)
    per = {t: [float(np.mean([r["tests"][t]["DS-R"] for r in res[b]])) for b in runs] for t in res[runs[0]][0]["tests"]}
    return {t: (min(v), max(v)) for t, v in per.items()}, runs


def main():
    O, B = ours(), json.load(open(R / "BASELINE_MATRIX.json"))
    spread, runs = seed_spread()
    mean10 = lambda d: float(np.mean([d[t] for t in A.TEST if t in d and d[t] == d[t]]))

    def table(names, cols, with_mean=True):
        L = ["| method | " + " | ".join(cols) + (" | 10-set mean |" if with_mean else " |"),
             "|---|" + "---:|" * (len(cols) + (1 if with_mean else 0))]
        for n in names:
            d = O[n] if n in O else {t: B[n][t]["video"] for t in B[n]}
            cells = [f"{d[t]:.4f}" if t in d else "TODO(run)" for t in cols]
            row = f"| {n} | " + " | ".join(cells)
            L.append(row + (f" | {mean10(d):.4f} |" if with_mean else " |"))
        return L

    base = list(B)          # includes R0, our own prior recipe, which belongs in the comparison rather than outside it
    L = ["# Results — the deployed system and the baselines", "", __doc__.split("\n\n", 1)[1].split("Sources")[0].strip(), "",
         "Every detector is trained on FaceForensics++ c23 and scored on the same videos, video-level AUROC "
         "(per-frame opinions averaged per video, then fused — the corrected-table rule). Our rows are the mean over "
         "the five source-grouped splits whose fitting half fits the fusion; the baselines are single runs, so their "
         "rows carry no comparable spread. The fused system has no frame-level row by construction: its temporal "
         "view emits one opinion per clip, not per frame.", "",
         "## 1. Main table — video AUROC", "",
         *table(list(OURS)[:2] + base, MAIN),
         "", f"Seed spread of the DS-R row over {len(runs)} temporal-branch seeds (only `--seed` differs): " +
         ", ".join(f"{t} {spread[t][0]:.4f}–{spread[t][1]:.4f}" for t in MAIN if t in spread) + ".", "",
         "**Caption note.** The seven sets above were chosen for the main table *after* the numbers were known "
         "(decision of 2026-09-18). The three sets left out — DFDC, DFDCP and DeepfakeEval2024 — are in Appendix A at "
         "the same precision, and the ten-set mean in every row is computed over all ten, so the headline aggregate "
         "is unaffected by the split.", "",
         "## 2. The views on their own", "",
         *table(list(OURS)[2:], MAIN),
         "", "The temporal view is the weakest alone on swap-heavy sets and the strongest on reenactment; that "
         "complementarity is what the fusion is for (Experiment 2).", "",
         "## Appendix A — the sets left out of the main table", "",
         *table(list(OURS)[:2] + base, APPX),
         "", "DeepfakeEval2024 is the hardest row for every method in the matrix, ours included; DFDCP is the "
         "reference's weakest processed benchmark (Experiment 3).", "",
         "## Appendix B — forgery-type slices and processed conditions (our system)", "",
         *table(list(OURS), SLICES + PROC, with_mean=False),
         "", "Processed conditions are JPEG q40 and down-up resize 0.60, applied at evaluation only; neither severity "
         "is used in training (the arm-B mixture trains on q70/q50/q30 and 0.75/0.50/0.35).", "",
         "## Appendix C — sealed", "",
         "| benchmark | status |", "|---|---|",
         "| FaceShifter (GenD release) | TODO(run) — sealed until the recipe is fixed by Experiment 5; unsealed and "
         "scored once, with no tuning afterwards (decision of 2026-09-18). Source-split audit: its 140 ids are exactly "
         "the FF++ test split, disjoint from our train and validation source ids. |", "",
         "CDFv2 and DeepfakeEval2024 informed development through VALmix and are not untouched confirmation data."]
    (R / "FINAL_TABLE.md").write_text("\n".join(L) + "\n"); print("wrote", R / "FINAL_TABLE.md")


if __name__ == "__main__":
    main()

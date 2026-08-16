"""Consolidate every D-ladder rung's post-hoc metrics into the final comparison table.

test.py writes logs/test/<training_run_dir>/<dataset>/metrics.csv — one row per dataset,
columns are metric names. This walks those, keyed by which config produced the run.

    python <this> [metric]      # default video_auc, e.g. ... auc
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

REPO = pathlib.Path("/data/umar/Repos/DFB_NeSyNS")
TEST = REPO / "logs" / "test"

RUNGS = [
    ("D0 (v1 reference)",     "nesy_defake_ablation4_ccv_d0"),
    ("D1-V  visual only",     "nesy_defake_d1_v"),
    ("D1-M  manifold only",   "nesy_defake_d1_m"),
    ("D1-VM visual+manifold", "nesy_defake_d1_vm"),
    ("D2    visual+process",  "nesy_defake_d2"),
    ("D3    all three",       "nesy_defake_d3"),
    ("D3(P1d) D4 baseline",   "nesy_defake_d3_p1d"),
]
SETS = ["FaceForensics++", "DeepFakeDetection", "Celeb-DF-v1", "Celeb-DF-v2",
        "Celeb-DF-v3", "DFDC", "DFDCP", "UADFV"]


def load(base: str, metric: str) -> dict[str, float] | None:
    # exact-prefix match so d1_v does not also capture d1_vm
    cands = sorted([p for p in TEST.glob(f"{base}_2026-*") if p.is_dir()],
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for run in cands:
        out = {}
        for ds in SETS:
            f = run / ds / "metrics.csv"
            if f.exists():
                row = pd.read_csv(f).iloc[0]
                if metric in row:
                    out[ds] = float(row[metric])
        if out:
            return out
    return None


def main() -> int:
    metric = sys.argv[1] if len(sys.argv) > 1 else "auroc_video"
    table, missing = {}, []
    for label, base in RUNGS:
        m = load(base, metric)
        (table.setdefault(label, m) if m else missing.append(label))

    if not table:
        print(f"No post-hoc metrics found under {TEST}. Run the ladder queue first.")
        return 1

    df = pd.DataFrame(table).T
    df = df[[c for c in SETS if c in df.columns]]
    df["mean"] = df.mean(axis=1)
    print(f"\n### D-ladder — post-hoc {metric} on all evaluation sets\n")
    print(df.round(4).to_string())
    if missing:
        print("\nnot yet evaluated:", ", ".join(missing))

    print("\n### Gate-relevant deltas (mean)\n")

    def delta(a, b, why):
        if a in df.index and b in df.index:
            print(f"  {a:24s} - {b:24s} = {df.loc[a,'mean'] - df.loc[b,'mean']:+.4f}   {why}")
        else:
            print(f"  {a:24s} - {b:24s} = TODO(run)   {why}")

    delta("D1-VM visual+manifold", "D1-V  visual only",
          "does the manifold branch earn its place (D1 gate)")
    delta("D2    visual+process", "D1-V  visual only",
          "does the process residual survive (D2 gate)")
    delta("D3    all three", "D1-V  visual only",
          "do both complements survive together (D3 gate)")
    delta("D3    all three", "D0 (v1 reference)",
          "v2 architecture vs the v1 system")

    print("\nCAVEAT: the D1 gate is 'VM beats V ON THE INVERSION ROWS', not on this mean.\n"
          "Mean AUC is the weakest evidence available here — the inversion rows are where a\n"
          "specialist earns its place, and they are only measurable on the generator-labelled\n"
          "sources. Run a1_complementarity.py against these checkpoints for the real criterion.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

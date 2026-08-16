"""Leaked-selection vs leak-free D0, both evaluated through the identical test.py path."""
import pathlib

import pandas as pd

TEST = pathlib.Path("/data/umar/Repos/DFB_NeSyNS/logs/test")
RUNS = {
    "leaked selection": "nesy_defake_ablation4_ccv_2026-08-15-02-38-14",
    "leak-free":        "nesy_defake_ablation4_ccv_d0_2026-08-15-16-34-38",
}
SETS = ["FaceForensics++", "DeepFakeDetection", "Celeb-DF-v1", "Celeb-DF-v2",
        "Celeb-DF-v3", "DFDC", "DFDCP", "UADFV"]

for metric in ("auroc_video", "auroc_frame"):
    rows = {}
    for label, d in RUNS.items():
        vals = {}
        for s in SETS:
            f = TEST / d / s / "metrics.csv"
            if f.exists():
                r = pd.read_csv(f).iloc[0]
                if metric in r:
                    vals[s] = float(r[metric])
        if vals:
            rows[label] = vals
    if len(rows) < 2:
        print(f"{metric}: only have {list(rows)} — waiting on the other")
        continue
    df = pd.DataFrame(rows).T[SETS]
    df["mean"] = df.mean(axis=1)
    df.loc["gap (leak-free - leaked)"] = df.loc["leak-free"] - df.loc["leaked selection"]
    print(f"\n### {metric}\n")
    print(df.round(4).to_string())

#!/usr/bin/env python3
"""Frame- and video-level AUROC for baseline runs from run_gend_baselines.py.
video = parent directory of the frame; video score = mean prob_class_1 over its frames (GenD's own reduction).
KoDF main score excludes fake-audio-driven (protocol v3.1 scope); both are printed."""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd

RUNS = Path("/data/umar/Repos/DFB_NeSyNS/logs/tbiom/baselines/gend_runs")


def auroc(y, s):
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    r = pd.Series(s).rank().to_numpy(); n1 = (y == 1).sum(); n0 = (y == 0).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def score(csv, drop_sources=()):
    d = pd.read_csv(csv)
    d["source"] = d.files.str.split("/").str[-3]
    d = d[~d.source.isin(drop_sources)]
    d["video"] = d.files.str.rsplit("/", n=1).str[0]
    v = d.groupby("video").agg(y=("labels", "max"), s=("prob_class_1", "mean"), source=("source", "first"))
    return {"frame": auroc(d.labels, d.prob_class_1), "video": auroc(v.y, v.s),
            "n_frames": int(len(d)), "n_videos": int(len(v)), "n_real_videos": int((v.y == 0).sum())}


out = {}
for csv in sorted(glob.glob(f"{RUNS}/*/**/test_predictions.csv", recursive=True)):
    name = Path(csv).relative_to(RUNS).parts[0]; model, ds = name.split("--")
    out[name] = score(csv, ("fake-audio-driven",) if ds == "KoDF" else ())
    if ds == "KoDF":
        out[name + " (incl. audio-driven)"] = score(csv)
for k, r in out.items():
    print(f"{k:42} frame {r['frame']:.4f}  video {r['video']:.4f}  videos {r['n_videos']:6d} (real {r['n_real_videos']})")
json.dump(out, open(RUNS / "scores.json", "w"), indent=1)

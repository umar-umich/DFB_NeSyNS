#!/usr/bin/env python3
"""Pool DF40's two source halves into one test set per generator, from VIDEOS not from AUROCs.

A generator ships twice in DF40, built from Celeb-DF sources and from FaceForensics++ sources.
Reporting both halves costs 19 + 22 columns; pooling costs 10 + 11, which is what fits a page.

POOLED, NOT AVERAGED, and the distinction is not cosmetic. Averaging two halves' AUROCs weights a
140-video half equally with an 818-video one and corresponds to no test set anyone could build.
On `danet` the halves are 0.9030 and 0.9711. So every row here is recomputed from the videos.

THE REAL POOL IS COUNTED ONCE. Both halves of a generator are scored against their own source's
genuine videos, and concatenating the two tables would list each real video of each pool once --
but a generator's two halves share no real videos, so the union is simply both pools. Where a
source IS shared (one real pool serving several generators) the duplicate is dropped on video id.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                               # noqa: E402

HALVES = ("cdf", "ff")
BASE = C.ROOT / "logs/tbiom/baselines/gend_runs_ours"
S23 = C.ROOT / "logs/tbiom/stage23"
REAL = {"cdf": "DF40-cdf-real", "ff": "DF40-ff-real"}


def generator_of(ds: str) -> str:
    """`DF40_blendface_cdf` -> `DF40_blendface`; a half-less name is already a generator."""
    n = ds[len("DF40_"):]
    g = n.rsplit("_", 1)[0] if n.rsplit("_", 1)[-1] in HALVES else n
    return f"DF40_{g}"


def group(sets):
    """{generator name: [its per-half set names]}, in a stable order."""
    out = {}
    for ds in sets:
        out.setdefault(generator_of(ds), []).append(ds)
    return {k: sorted(v) for k, v in sorted(out.items())}


def pool_tables(tabs: dict, sets) -> dict:
    """Branch opinion tables, concatenated per generator with the genuine pool counted once."""
    return {g: pd.concat([tabs[m] for m in members], ignore_index=True)
            .drop_duplicates(subset="v").reset_index(drop=True)
            for g, members in group(sets).items()}


def _video_auroc(d):
    g = d.groupby("v").agg(y=("labels", "max"), s=("prob_class_1", "mean"))
    y, s = g.y.to_numpy().astype(int), g.s.to_numpy()
    if len(np.unique(y)) < 2:
        return None
    return C.auroc(y, s)


def baseline_pooled(model, jobs):
    """One AUROC per generator for a GenD-repo baseline, from the pooled DF40 run's frames."""
    hits = glob.glob(f"{BASE}/{model}--DF40/**/test_predictions.csv", recursive=True)
    if not hits:
        return {}
    d = pd.read_csv(hits[0])
    parts = d.files.str.split("/")
    d = d.assign(subset=parts.str[-3], v=d.files.str.rsplit("/", n=1).str[0])
    subs = set(d.subset.unique())
    out = {}
    for gen, pairs in jobs.items():
        want = set()
        for fk, rl in pairs:
            if fk in subs and rl in subs:
                want |= {fk, rl}
        if not want:
            continue
        a = _video_auroc(d[d.subset.isin(want)])
        if a is not None:
            out[gen] = a
    return out


def dicome_ours_pooled(jobs):
    """Our trained DiCoME, pooled the same way, from the stage-23 per-set exports."""
    sys.path.insert(0, str(C.ROOT / "analysis/tbiom"))
    from video_id import video_id
    out = {}
    for gen, pairs in jobs.items():
        frames = []
        for ds in group([f"{gen}_{h}" for h in HALVES] + [gen]).get(gen, []):
            f = S23 / f"dicome_{ds}.csv"
            if f.exists():
                frames.append(pd.read_csv(f))
        if not frames:
            continue
        d = pd.concat(frames, ignore_index=True)
        d = d.assign(v=video_id(d["key"]), labels=d["label"],
                     prob_class_1=d["p_artifact"]).drop_duplicates(subset=["v", "key"])
        a = _video_auroc(d)
        if a is not None:
            out[gen] = a
    return out


def jobs_for(meta):
    """{generator: [(fake subset, real subset), ...]} across both halves."""
    out = {}
    for name, m in meta.items():
        g, gen = m["method"], generator_of(name)
        if name.endswith("_cdf") or name.endswith("_ff"):
            h = name.rsplit("_", 1)[1]
            out.setdefault(gen, []).append((f"{g}-{h}", REAL[h]))
        else:
            out.setdefault(gen, []).append((g, f"{g}-real"))
    return out

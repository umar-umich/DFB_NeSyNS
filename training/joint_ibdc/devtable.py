#!/usr/bin/env python3
"""The development table, with the video view's opinions taken from the feature cache.

`confirm.dev_table` attaches the temporal branch's historical opinion CSVs. This builds the same
table but attaches the CACHE-DERIVED ones written by `reexport_temporal.py`, so every view in this
study derives from a single cache. The spatial half is unchanged and still read from the deployed
stage-23 exports.

`compare()` reports what the substitution does, because a change that is never measured is a change
nobody can defend.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(ROOT / "analysis/tbiom/pcr6p"), str(ROOT / "analysis/tbiom/grouped"),
                str(ROOT / "analysis/tbiom"), str(ROOT / "analysis/tbiom/temporal")]

import confirm as C            # noqa: E402
import exp1_fusion_audit as A  # noqa: E402
import df40_analysis as X      # noqa: E402
from fusion_combos import attach  # noqa: E402

OPI = ROOT / "logs/joint_ibdc/opinions"
SPATIAL, BRANCH = "warmgate_e19", "vjepa2_lora_B_seed42"


def dev_table_cached(spatial_tag: str = SPATIAL):
    missing = [n for n in C.DEV_SETS if not (OPI / f"{n}.csv").exists()]
    if missing:
        raise FileNotFoundError(f"cache-derived temporal opinions missing for {missing}; "
                                f"run training/joint_ibdc/reexport_temporal.py")
    return attach(A.dev_tables({n: X.S23 / f"{spatial_tag}_{n}.csv" for n in C.DEV_SETS}, [1.0]),
                  [OPI / f"{n}.csv" for n in C.DEV_SETS])


def compare():
    hist = C.dev_table(SPATIAL, BRANCH)
    cach = dev_table_cached()
    assert list(hist.v) == list(cach.v), "row order must match for a paired comparison"
    out = {}
    for col in ("p_T", "u_T"):
        d = np.abs(hist[col].to_numpy() - cach[col].to_numpy())
        out[col] = {"max": float(d.max()), "median": float(np.median(d))}
    # the spatial columns must be untouched
    for col in ("p_APP", "u_APP", "p_OM", "u_OM"):
        assert np.abs(hist[col].to_numpy() - cach[col].to_numpy()).max() == 0.0, col
    return hist, cach, out


if __name__ == "__main__":
    h, c, o = compare()
    print(f"development table {len(h)} videos")
    for k, v in o.items():
        print(f"  {k}: max |diff| {v['max']:.3e}, median {v['median']:.3e}")
    print("  spatial columns identical: True")

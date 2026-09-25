#!/usr/bin/env python3
"""Recover cache shards left behind by the .partial naming bug, but only when they are provably whole.

    python branches/ijepa/recover_partials.py [--apply]

np.savez appends `.npz` when the given name does not already end in it, so the extractor's temporary
`X.npz.partial` was actually written as `X.npz.partial.npz` and the rename that followed could not
find it. The save itself had already completed, so the data is whole -- but "probably whole" is not
good enough to promote into a cache that later gates read, so each candidate is verified against the
timing record the same run wrote before saving: the frame count, and that every array is present and
agrees in length. Anything that does not match is left alone and reported.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

CACHE = Path("/data/umar/Repos/DFB_NeSyNS/analysis/ijepa/cache")
TIMING = Path("/data/umar/Repos/DFB_NeSyNS/analysis/ijepa")
NEED = ("eps", "rbar", "pooled", "sbar", "q", "keys")


def candidates():
    return sorted(list(CACHE.glob("*.npz.partial.npz")) + list(CACHE.glob("*.partial.npz")))


def target_of(p: Path) -> Path:
    return CACHE / (re.sub(r"\.(npz\.)?partial\.npz$", "", p.name) + ".npz")


def check(p: Path):
    dest = target_of(p)
    stem = dest.name[:-4]
    tj = TIMING / f"timing_{stem}.json"
    if dest.exists():
        return False, f"{dest.name} already exists"
    if not tj.exists():
        return False, f"no timing record {tj.name}"
    want = json.load(open(tj))["frames"]
    try:
        with np.load(p, allow_pickle=True) as z:
            missing = [a for a in NEED if a not in z.files]
            if missing:
                return False, f"missing arrays {missing}"
            n = len(z["keys"])
            bad = [a for a in NEED if len(z[a]) != n]
            if bad:
                return False, f"length mismatch in {bad}"
    except Exception as e:                                  # a truncated npz fails to open
        return False, f"unreadable: {type(e).__name__}"
    if n != want:
        return False, f"{n} frames against {want} in the timing record"
    return True, f"{n} frames, all arrays present and agreeing"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="perform the renames")
    a = ap.parse_args()
    cs = candidates()
    if not cs:
        print("no partial shards to recover")
        return
    ok = 0
    for p in cs:
        good, why = check(p)
        print(f"  {'RECOVER' if good else 'LEAVE  '} {p.name}: {why}")
        if good and a.apply:
            p.rename(target_of(p))
            ok += 1
    if a.apply:
        print(f"recovered {ok} of {len(cs)}")
    else:
        print(f"{sum(1 for p in cs if check(p)[0])} of {len(cs)} would be recovered; pass --apply")


if __name__ == "__main__":
    main()

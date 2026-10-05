#!/usr/bin/env python3
"""Write each layer readout as a standard feature cache, so `screen_eval.py` can measure its fused
incremental contribution without a second code path.

    <discern_ext python> upgrade/materialize_readouts.py

Keep-rule condition 1 is the incremental contribution over LFS and TD, which only `screen_eval`
computes (with the source-grouped bootstrap). That script reads `logs/upgrade/feats/<name>_<ds>.npz`
and a head at `analysis/upgrade/heads/view_<name>_s0.pt`, so the readouts are published under that
naming rather than teaching it a second layout -- the fused numbers then come from exactly the code
that produced the Part 3b bar.
"""
from __future__ import annotations

import json, shutil, sys
from pathlib import Path

import numpy as np

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(ROOT / "upgrade")]
from train_layer_head import load_readout, LF, OUT as LOUT  # noqa: E402

FEATS = ROOT / "logs/upgrade/feats"
HEADS = ROOT / "analysis/upgrade/heads"
SETS = ("FFpp_val", "VALmixV2", "VALmix", "FFpp_train")
READOUTS = ("last", "mid", "half", "last+mid", "last+mid+half")


def main() -> int:
    HEADS.mkdir(parents=True, exist_ok=True)
    n = 0
    for bb in ("pe_core_l", "clip"):
        for ro in READOUTS:
            name = f"{bb}-{ro.replace('+', '')}"
            src = LOUT / f"{bb}_{ro.replace('+', '-')}_s0.json"
            if not src.exists():
                print(f"  skip {bb} {ro}: no trained head"); continue
            for ds in SETS:
                out = FEATS / f"{name}_{ds}.npz"
                if out.exists():
                    continue
                if not (LF / f"{bb}_{ds}.npz").exists():
                    print(f"  skip {name} {ds}: no layer cache"); continue
                z = np.load(LF / f"{bb}_{ds}.npz", allow_pickle=True)
                D = load_readout(bb, ds, ro)
                tmp = out.with_suffix(".partial.npz")
                np.savez(tmp, feature=D["x"].astype(np.float16), key=z["key"],
                         label=z["label"], video=z["video"])
                tmp.replace(out); n += 1
            shutil.copy(src, HEADS / f"view_{name}_s0.json")
            shutil.copy(LOUT / f"{bb}_{ro.replace('+', '-')}_s0.pt", HEADS / f"view_{name}_s0.pt")
            print(f"  {name}: features + head published")
    print(f"wrote {n} feature files")
    return 0


if __name__ == "__main__":
    sys.exit(main())

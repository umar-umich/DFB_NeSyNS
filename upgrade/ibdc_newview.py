#!/usr/bin/env python3
"""Rerun the joint-IBDC J-runs with the NEW appearance view, fusion fixed.

    <discern_ext python> upgrade/ibdc_newview.py --run J1 --seed 0
    <discern_ext python> upgrade/ibdc_newview.py --run J2b --seed 0 --lam 0.1

The brief requires J1, J2a and J2b rerun "with fusion fixed, if any view changed". The spatial view
changed, so this swaps ONLY that view and leaves `training/joint_ibdc/runs.py` untouched: its loss,
its fusion, its optimiser, its class-order conversion and its selection are imported and used, not
copied. Monkey-patching the two loaders and the appearance head is what keeps the rerun comparable
with the original; editing runs.py in place would make the two sets of numbers products of different
code.

WHAT IS SWAPPED
  training features  logs/upgrade/ibdc_cache/FFpp_train_B_d{d}_<view>.npz, built by replaying the
                     SAME recorded arm-B draws, so the augmentation realisation is identical.
  dev features       logs/upgrade/feats/<view>_{VALmix,VALmixV2,FFpp_val}.npz
  appearance head    analysis/upgrade/heads/view_<view>_s0.pt -- the 1024 -> 256 -> 2 evidential head
                     trained in Part 3b, replacing the deployed 64-dim head.
The off-manifold and temporal views, and every cached temporal opinion, are the deployed ones.
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(ROOT / "training/joint_ibdc"), str(ROOT / "upgrade"),
                str(ROOT / "analysis/tbiom"), str(ROOT / "analysis/tbiom/temporal")]
import runs as R                       # noqa: E402  the study's code, used not copied
import draws as DR                     # noqa: E402
from video_id import video_id          # noqa: E402

NEW_CACHE = ROOT / "logs/upgrade/ibdc_cache"
FEATS = ROOT / "logs/upgrade/feats"
HEADS = ROOT / "analysis/upgrade/heads"


class NewAppHead(nn.Module):
    """The Part 3b head for the new view: Linear(d,256) -> GELU -> Linear(256,2) -> softplus."""

    def __init__(self, dim):
        super().__init__()
        self.fc1, self.drop, self.fc2 = nn.Linear(dim, 256), nn.Dropout(0.1), nn.Linear(256, 2)

    def forward(self, x):
        return F.softplus(self.fc2(self.drop(F.gelu(self.fc1(x)))))


def install(view: str):
    """Replace the two loaders and the appearance head; leave everything else alone."""
    st = torch.load(HEADS / f"view_{view}_s0.pt", map_location="cpu")
    dim = int(st["dim"])

    def load_train():
        out = []
        for d in range(R.N_DRAWS):
            fa = NEW_CACHE / f"FFpp_train_B_d{d}_{view}.npz"
            fs, ft = R.CACHE / f"FFpp_train_B_d{d}.npz", R.CACHE / f"FFpp_train_T_B_d{d}.npz"
            za, zs, zt = (np.load(f, allow_pickle=True) for f in (fa, fs, ft))
            assert len(za["key"]) == len(zs["key"]), "new-view cache is not frame-aligned with the deployed one"
            tv = {str(v): i for i, v in enumerate(zt["video"])}
            vid = [DR.video_of(str(k)) for k in zs["key"]]
            keep = np.array([i for i, v in enumerate(vid) if v in tv])
            idx = np.array([tv[vid[i]] for i in keep])
            out.append({"app": za["feature"][keep].astype(np.float32),     # NEW view
                        "om": zs["artifact"][keep].astype(np.float32),     # deployed
                        "t": zt["feature"][idx].astype(np.float32),        # deployed
                        "y": zs["label"][keep].astype(np.int64)})
        n = {len(o["y"]) for o in out}
        assert len(n) == 1, f"draws disagree on sample count: {n}"
        return out

    def load_dev():
        out = {}
        for name, sf in R.DEV_SPATIAL.items():
            af = FEATS / f"{view}_{name}.npz"
            tf = R.DEV_TEMPORAL[name]
            za, zs, zt = (np.load(f, allow_pickle=True) for f in (af, sf, tf))
            # The two extractors write keys in different forms -- the deployed cache keeps the
            # `<h5 path>::<key>` prefix and the upgrade cache does not -- so alignment strips the
            # prefix from both. Aligning by ROW ORDER instead would look fine and silently pair a
            # frame's new-view feature with a different frame's off-manifold feature.
            strip = lambda s: str(s).split("::", 1)[-1]
            ka = np.array([strip(k) for k in za["key"]])
            ks = np.array([strip(k) for k in zs["key"]])
            pos = {k: i for i, k in enumerate(ka)}
            sel = np.array([pos[k] for k in ks if k in pos])
            ks_keep = np.array([i for i, k in enumerate(ks) if k in pos])
            vid = video_id(pd.Series(ks[ks_keep])).to_numpy()
            tv = {str(v): i for i, v in enumerate(zt["video"])}
            keep2 = np.array([i for i, v in enumerate(vid) if v in tv])
            order, inv = np.unique(vid[keep2], return_inverse=True)
            lab = zs["label"][ks_keep][keep2]
            out[name] = {"app": za["feature"][sel][keep2].astype(np.float32),
                         "om": zs["artifact"][ks_keep][keep2].astype(np.float32),
                         "t": zt["feature"][[tv[v] for v in order]].astype(np.float32),
                         "inv": inv, "order": order,
                         "y": np.array([int(lab[inv == i].max()) for i in range(len(order))]),
                         "key": np.array([ks[ks_keep][keep2][inv == i][0] for i in range(len(order))])}
        return out

    orig_heads = R.load_deployed_heads

    def load_heads(dev="cpu", freeze_norms=True):
        _app, om, t = orig_heads(dev, freeze_norms)
        app = NewAppHead(dim)
        app.load_state_dict(st["state"])
        return app.to(dev), om, t

    R.load_train, R.load_dev, R.load_deployed_heads = load_train, load_dev, load_heads
    print(f"installed view '{view}' (dim {dim}) into the IBDC runs", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="pe_core_lN")
    ap.add_argument("--run", default="J1", choices=["J0", "J1", "J2a", "J2b"])
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--lam", type=float, default=0.3)
    a = ap.parse_args()
    install(a.view)
    R.OUT = ROOT / "analysis/upgrade/ibdc"
    R.OUT.mkdir(parents=True, exist_ok=True)
    sys.argv = ["runs.py", "--run", a.run, "--seed", str(a.seed), "--lam", str(a.lam)]
    return R.main()


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Task 2 gate, run on the cache that already exists: do the deployed heads on cached features
reproduce the deployed per-view opinions?

    OMP_NUM_THREADS=4 <discern_ext python> training/joint_ibdc/verify_cache.py

This runs BEFORE any new extraction, deliberately. `eval_adaptation/export_views.py --features`
already caches the per-frame blocks the spatial heads consume, and `logs/tsne/ours_*.npz` holds them
for the deployed epoch-19 checkpoint. If applying the frozen LayerNorm and the deployed head to those
blocks does not reproduce the deployed opinions, the whole cache design for Task 2 is wrong, and it is
much cheaper to discover that on CPU than after a GPU sweep.

WHAT IS CACHED, AND WHY PRE-NORM IS THE RIGHT THING TO CACHE. The blocks are `semantic_feature` and
the rebuilt `artifact_feature`, both BEFORE `semantic_norm` / `artifact_norm`. Those norms are
nn.LayerNorm (`src/model/core_model.py:116-117`), which is deterministic and has no running state, and
they are frozen in this study because only the heads train. Caching pre-norm and applying the frozen
norm at training time is therefore identical to caching post-norm, and leaves the option open.

AGGREGATION, confirmed from deployed code rather than assumed:
`analysis/tbiom/exp1_fusion_audit.py:104` -- "Per-video means of every frame quantity (the
corrected-table rule: mean opinion, then fuse)". The video opinion is the MEAN OF THE FRAME OPINIONS,
p and u averaged separately. It is not an average of evidence, and fusion happens after averaging.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
EXT = Path("/data/umar/Repos/DISCERN_Ext")
sys.path[:0] = [str(ROOT / "analysis/tbiom/pcr6p"), str(ROOT / "analysis/tbiom/grouped"),
                str(ROOT / "analysis/tbiom"), str(ROOT / "analysis/tbiom/temporal")]

import confirm as C          # noqa: E402
from video_id import video_id  # noqa: E402  THE video rule, never a local shortcut

CKPT = EXT / "runs/discern_ext_train/warm_dicome_gate_seed42/checkpoints/discern_ext-best-epoch=19-val_auroc_video=0.9956.ckpt"
# every cached split that the development table also covers, so the gate is not a one-off
SOURCES = {
    "FFpp_val": ROOT / "logs/tsne/ours_FFpp_val.npz",
    "VALmix": ROOT / "logs/joint_ibdc/cache/VALmix.npz",
    "VALmixV2": ROOT / "logs/joint_ibdc/cache/VALmixV2.npz",
}
SPATIAL, BRANCH = "warmgate_e19", "vjepa2_lora_B_seed42"
TOL = 1e-5
OUT = ROOT / "analysis/joint_ibdc"
# cached block -> (norm prefix, head prefix, the dev table's column suffix)
VIEWMAP = {"semantic": ("model.semantic_norm", "model.semantic_head", "APP"),
           "artifact": ("model.artifact_norm", "model.artifact_head", "OM")}


def head_opinions(x, sd, norm_p, head_p):
    """LayerNorm -> Linear -> ReLU -> Linear -> softplus -> (p_fake, vacuity), as deployed."""
    x = torch.as_tensor(x, dtype=torch.float32)
    x = Fn.layer_norm(x, (x.shape[-1],), sd[f"{norm_p}.weight"].float(), sd[f"{norm_p}.bias"].float())
    h = Fn.relu(Fn.linear(x, sd[f"{head_p}.fc1.weight"].float(), sd[f"{head_p}.fc1.bias"].float()))
    ev = Fn.softplus(Fn.linear(h, sd[f"{head_p}.fc2.weight"].float(), sd[f"{head_p}.fc2.bias"].float()))
    alpha = ev + 1
    S = alpha.sum(1, keepdim=True).clamp_min(1e-6)
    return (alpha / S)[:, 1].numpy(), (alpha.shape[1] / S.squeeze(1)).numpy()


def main():
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    dev = C.dev_table(SPATIAL, BRANCH)
    print(f"deployed development table: {len(dev)} videos")
    import pandas as pd

    allres, worst_all, checked = {}, 0.0, 0
    for name, feat in SOURCES.items():
        if not feat.exists():
            print(f"  {name:9s} not cached yet, skipped")
            continue
        z = np.load(feat, allow_pickle=True)
        keys = np.array([str(k) for k in z["key"]])
        vids = video_id(pd.Series(keys)).to_numpy()
        order = np.unique(vids)
        res, worst = {}, 0.0
        for block, (norm_p, head_p, col) in VIEWMAP.items():
            p, u = head_opinions(z[block], sd, norm_p, head_p)
            inv = {v: i for i, v in enumerate(order)}
            idx = np.array([inv[v] for v in vids])
            cnt = np.bincount(idx, minlength=len(order)).astype(float)
            pv = np.bincount(idx, weights=p, minlength=len(order)) / cnt
            uv = np.bincount(idx, weights=u, minlength=len(order)) / cnt
            m = {v: i for i, v in enumerate(order)}
            keep = [i for i, v in enumerate(dev.v) if v in m]
            if not keep:
                print(f"  {name}/{block}: no overlap with the development table"); continue
            j = np.array([m[dev.v[i]] for i in keep])
            dp = float(np.abs(pv[j] - dev[f"p_{col}"].to_numpy()[keep]).max())
            du = float(np.abs(uv[j] - dev[f"u_{col}"].to_numpy()[keep]).max())
            worst = max(worst, dp, du)
            res[block] = {"view": col, "n_videos_matched": len(keep),
                          "max_abs_diff_p": dp, "max_abs_diff_u": du}
            print(f"  {name:9s} {block:9s} -> {col:3s}  {len(keep):5d} videos   "
                  f"p {dp:.3e}   u {du:.3e}")
        allres[name] = {"features": str(feat), "worst": worst, "per_block": res}
        worst_all = max(worst_all, worst)
        checked += 1

    ok = checked > 0 and worst_all <= TOL
    out = {"gate": "Task 2 cache equivalence", "verdict": "PASS" if ok else "FAIL",
           "tolerance": TOL, "worst": worst_all, "splits_checked": checked,
           "checkpoint": str(CKPT),
           "aggregation": "mean of frame opinions (p and u separately), then fuse "
                          "[analysis/tbiom/exp1_fusion_audit.py:104]",
           "per_split": allres}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "task2_cache_equivalence.json", "w"), indent=1)
    print(f"\n{checked} split(s) checked, worst {worst_all:.3e} against tolerance {TOL}")
    print(f"Task 2 cache equivalence: {out['verdict']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

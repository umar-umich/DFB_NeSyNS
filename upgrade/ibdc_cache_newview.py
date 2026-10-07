#!/usr/bin/env python3
"""Rebuild the joint-IBDC FF++ train cache for a NEW appearance view, on the SAME arm-B draws.

    CUDA_VISIBLE_DEVICES=1 <discern_ext_timm python> upgrade/ibdc_cache_newview.py \
        --view pe_core_l --recipe N --draws 3

The draws are not re-sampled. Each existing cache file records the arm-B setting that was drawn for
every frame, and those settings are REPLAYED here, so the new view sees exactly the frames the
deployed views saw in exactly the condition they saw them. Re-drawing would make the IBDC comparison
confound the view with the augmentation realisation.

Arm B is applied to the NATIVE 256 crop BEFORE the backbone's own preprocessing, which is where
training applies it -- the same reason `extract_ffpp_train.py` exists instead of `--process`.
"""
from __future__ import annotations

import argparse, io, json, sys, time
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
EXT = Path("/data/umar/Repos/DISCERN_Ext/eval_adaptation/data")
CACHE = ROOT / "logs/joint_ibdc/cache"
OUT = ROOT / "logs/upgrade/ibdc_cache"
sys.path[:0] = [str(ROOT / "upgrade"), "/data/umar/Repos/DISCERN_Ext"]
import adapt_backbone as AB  # noqa: E402
from src.dataset.training_arms import SharedProcessing  # noqa: E402

torch.set_num_threads(4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="pe_core_l"); ap.add_argument("--recipe", default="N")
    ap.add_argument("--draws", type=int, default=3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    m, mean, std, spec, _p, _f = AB.build(a.view, dev, a.recipe)
    ck = torch.load(AB.OUT / f"{a.view}_{a.recipe}_s{a.seed}.pt", map_location=dev)
    sd = m.state_dict()
    adapted = {k: v.to(dev) for k, v in ck["adapt"].items() if k in sd}
    m.load_state_dict({**sd, **adapted}, strict=False); m.eval()
    proc = SharedProcessing(enabled=True)
    tag = f"{a.view}{a.recipe}"
    print(f"{tag}: epoch {ck['epoch']}, input {spec['size']}", flush=True)

    for d in range(a.draws):
        f = OUT / f"FFpp_train_B_d{d}_{tag}.npz"
        if f.exists():
            print(f"  draw {d} cached"); continue
        src = CACHE / f"FFpp_train_B_d{d}.npz"
        z = np.load(src, allow_pickle=True)
        keys = [str(k) for k in z["key"]]
        settings = [str(s) for s in z["setting"]]
        print(f"  draw {d}: replaying {len(keys)} recorded settings", flush=True)
        Z, t0 = [], time.time()
        buf = []
        with h5py.File(EXT / "h5" / "FFpp_train.h5", "r") as h5:
            for i, (k, st) in enumerate(zip(keys, settings)):
                kk = k.split("::", 1)[-1]
                arr = h5[kk][()]
                im = (Image.open(io.BytesIO(arr.tobytes())) if arr.dtype == np.uint8 and arr.ndim == 1
                      else Image.fromarray(np.asarray(arr).astype(np.uint8))).convert("RGB")
                x = np.asarray(im, dtype=np.uint8)
                if st != "none":                       # replay the RECORDED draw, never a new one
                    kind, val = st.split(":")
                    x = proc.apply(x, (kind, float(val) if kind == "resize" else int(val)))
                im = Image.fromarray(x)
                if im.size != (spec["size"],) * 2:
                    im = im.resize((spec["size"],) * 2, Image.BICUBIC)
                buf.append(np.asarray(im, dtype=np.uint8))
                if len(buf) == a.batch_size or i == len(keys) - 1:
                    t = torch.from_numpy(np.stack(buf)).permute(0, 3, 1, 2).float().div_(255).to(dev)
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        Z.append(AB.embed(m, spec, (t - mean) / std, a.recipe).float().cpu().numpy())
                    buf = []
                    if i % 20000 < a.batch_size:
                        print(f"    {i}/{len(keys)} {i/max(1,time.time()-t0):.0f} f/s", flush=True)
        F = np.concatenate(Z).astype(np.float16)
        tmp = f.with_suffix(".partial.npz")
        np.savez(tmp, feature=F, label=z["label"], key=z["key"], setting=z["setting"])
        tmp.replace(f)
        print(f"  -> {f.name} {F.shape} in {(time.time()-t0)/60:.1f} min", flush=True)
    json.dump({"view": a.view, "recipe": a.recipe, "draws": a.draws,
               "settings_replayed_from": str(CACHE),
               "note": "arm-B settings are replayed from the deployed cache, never re-drawn"},
              open(OUT / f"manifest_{tag}.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

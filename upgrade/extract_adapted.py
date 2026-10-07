#!/usr/bin/env python3
"""Part 3c winner rule: cache features from an ADAPTED backbone so the fused metric can be measured.

    CUDA_VISIBLE_DEVICES=1 <discern_ext_timm python> upgrade/extract_adapted.py \
        --backbone pe_core_l --recipe L --dataset VALmix

Standalone dev v3.1 is not the winner rule. The rule is the incremental contribution over LFS and
TD, which needs the adapted view's opinions inside the three-view fusion -- and the independent-views
study found a view that improved standalone while costing the fusion, so the two must not be
conflated. This writes the same npz shape as `extract_spatial.py`, so `screen_eval.py` reads an
adapted candidate exactly as it reads a frozen one.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
OUT = ROOT / "logs/upgrade/feats"
sys.path[:0] = [str(ROOT / "upgrade"), str(ROOT / "branches/ijepa")]
import adapt_backbone as AB   # noqa: E402
from data import video_of     # noqa: E402

torch.set_num_threads(4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True); ap.add_argument("--recipe", required=True, choices=["L", "N", "LN"])
    ap.add_argument("--dataset", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    a = ap.parse_args()
    # The SEED belongs in the filename. Without it every seed wrote to one path, the second seed
    # found the first seed's file and reported "exists", and a caller that renamed the result
    # afterwards silently relabelled seed 0's features as another seed's. Seed 0 keeps the bare name
    # so the existing caches and every path that reads them stay valid.
    tag = f"{a.backbone}{a.recipe}" + ("" if a.seed == 0 else f"_s{a.seed}")
    f = OUT / f"{tag}_{a.dataset}.npz"
    if f.exists():
        print(f"{f.name} exists"); return 0

    dev = "cuda"
    m, mean, std, spec, _params, _fused = AB.build(a.backbone, dev, a.recipe)
    ck = torch.load(AB.OUT / f"{a.backbone}_{a.recipe}_s{a.seed}.pt", map_location=dev)
    # The run saved only the ADAPTED tensors (adapters, or LayerNorm); everything else stays at the
    # pretrained value, so a strict load would fail by design. Assert the adapted keys all landed.
    sd = m.state_dict()
    adapted = {k: v.to(dev) for k, v in ck["adapt"].items() if k in sd}
    assert len(adapted) == len(ck["adapt"]), \
        f"{len(ck['adapt']) - len(adapted)} adapted tensors do not match this backbone"
    m.load_state_dict({**sd, **adapted}, strict=False)
    m.eval()
    print(f"{tag}: epoch {ck['epoch']}, {len(adapted)} adapted tensors restored, input {spec['size']}", flush=True)

    X, K = AB.frames_of(a.dataset, spec["size"])
    Z, t0 = [], time.time()
    with torch.no_grad():
        for i in range(0, len(X), a.batch_size):
            x = torch.from_numpy(X[i:i + a.batch_size]).permute(0, 3, 1, 2).float().div_(255).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                Z.append(AB.embed(m, spec, (x - mean) / std, a.recipe).float().cpu().numpy())
    F = np.concatenate(Z).astype(np.float16)
    y = np.array([0 if "real" in str(k).split("/")[-3].lower() else 1 for k in K], dtype=np.int64)
    tmp = f.with_suffix(".partial.npz")
    np.savez(tmp, feature=F, key=K, label=y, video=np.array([video_of(str(k)) for k in K], dtype=object))
    tmp.replace(f)
    json.dump({"backbone": a.backbone, "recipe": a.recipe, "adapted_epoch": int(ck["epoch"]),
               "dataset": a.dataset, "frames": int(len(K)), "dim": int(F.shape[1]),
               "input_size": spec["size"], "minutes": round((time.time() - t0) / 60, 1)},
              open(f.with_suffix(".json"), "w"), indent=1)
    print(f"  -> {f.name} {F.shape} in {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

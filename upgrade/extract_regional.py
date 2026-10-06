#!/usr/bin/env python3
"""Part 6a (regional half): V-JEPA 2 dense tokens pooled into the five facial regions, per frame group.

    CUDA_VISIBLE_DEVICES=1 <temporal_vjepa2 python> upgrade/extract_regional.py \
        --model vjepa2_h --dataset FFpp_val

Part 6a cached GLOBAL pooled features only, which is why Part 6c could not run. This adds the dense
pass: the encoder's patch tokens are reshaped back to (time group, row, column) and mean-pooled over
each region's patch box, giving one vector per (region, time group) per video.

REGIONS ARE THE I-JEPA BRIEF'S, UNCHANGED -- the same five boxes on the same 16x16 patch grid, so a
regional result here is comparable with that study instead of being a new partition that happens to
be called regional. The boxes OVERLAP by construction (centre shares patches with every other box,
and the cheeks share a column band); that is the brief's definition and the region-ablation
diagnostic in Part 6c is read with it in mind.

TOKEN LAYOUT IS ASSERTED, NOT ASSUMED. V-JEPA 2 emits (T/tubelet) * (H/patch) * (W/patch) tokens and
the tubelet size is a property of the checkpoint; the reshape is checked against the token count so a
layout change fails here rather than silently pooling the wrong patches into a region.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
OUT = ROOT / "logs/upgrade/vfeats"
sys.path[:0] = [str(ROOT / "analysis/tbiom"), str(ROOT / "analysis/tbiom/xclip"),
                str(ROOT / "analysis/tbiom/temporal"), str(ROOT / "upgrade"),
                "/data/umar/Repos/DISCERN_Ext"]
import train_branch as TB            # noqa: E402

# Defined here rather than imported: `analysis/tbiom/temporal/extract_video.py` is on sys.path ahead
# of this package and shadows `upgrade/extract_video.py`, so the import silently picked up a
# different MODELS table. Kept in step with upgrade/extract_video.py by hand.
MODELS = {"vjepa2_l": ("facebook/vjepa2-vitl-fpc64-256", 16),
          "vjepa2_h": ("facebook/vjepa2-vith-fpc64-256", 16)}

torch.set_num_threads(4)

# (row_start, row_end, col_start, col_end) on the 16x16 patch grid -- branches/ijepa/features.py
BLOCKS = {"upper": (2, 9, 4, 12), "lower": (9, 16, 4, 12), "left_cheek": (5, 12, 0, 8),
          "right_cheek": (5, 12, 8, 16), "centre": (4, 11, 4, 12)}
REGION_NAMES = list(BLOCKS)


class DenseEncoder(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        from transformers import AutoVideoProcessor, VJEPA2Model
        mid, self.T = MODELS[name]
        self.backbone = VJEPA2Model.from_pretrained(mid).eval()
        self.processor = AutoVideoProcessor.from_pretrained(mid)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.name = name

    def preprocess(self, frames):
        return self.processor(np.stack(frames), return_tensors="pt")["pixel_values_videos"][0]

    @torch.no_grad()
    def forward(self, px):
        h = self.backbone(pixel_values_videos=px, skip_predictor=True).last_hidden_state
        B, N, D = h.shape
        side = 16                                   # 256 px / 16 px patches
        assert N % (side * side) == 0, (
            f"token count {N} is not a multiple of {side*side}; the spatial grid is not {side}x{side} "
            f"for this checkpoint and the regional reshape would pool the wrong patches")
        Tg = N // (side * side)                     # temporal groups after tubelet folding
        g = h.view(B, Tg, side, side, D)
        out = []
        for r0, r1, c0, c1 in BLOCKS.values():
            out.append(g[:, :, r0:r1, c0:c1, :].mean(dim=(2, 3)))    # (B, Tg, D)
        return torch.stack(out, dim=2), Tg          # (B, Tg, R, D)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vjepa2_h", choices=sorted(MODELS))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{a.model}_regional_{a.dataset}.npz"
    if f.exists():
        print(f"{f.name} exists"); return 0
    dev = "cuda"
    enc = DenseEncoder(a.model).to(dev)
    ds = TB.Clips(a.dataset, enc.preprocess, enc.T)
    if a.limit:
        ds.items = ds.items[:a.limit]
    loader = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, num_workers=a.workers,
                                         worker_init_fn=lambda _: torch.set_num_threads(1))
    Z, Y, V, items, t0, n, Tg = [], [], [], ds.items, time.time(), 0, None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for px, y, i in loader:
            z, Tg = enc(px.to(dev, non_blocking=True))
            Z.append(z.float().cpu().numpy().astype(np.float16)); Y.append(y.numpy())
            V += [items[int(k)][2] for k in i.tolist()]
            n += len(y)
            if n % (a.batch_size * 50) < a.batch_size:
                print(f"  {n}/{len(items)} videos, {n/(time.time()-t0):.2f} vid/s", flush=True)
    Z = np.concatenate(Z); Y = np.concatenate(Y)
    secs = time.time() - t0
    if a.limit:
        print(f"PROBE: {n} videos in {secs/60:.1f} min -> {n/secs:.2f} vid/s, shape {Z.shape} "
              f"(videos, time groups, regions, dim)"); return 0
    tmp = f.with_suffix(".partial.npz")
    np.savez(tmp, feature=Z, label=Y, video=np.array(V, dtype=object),
             regions=np.array(REGION_NAMES, dtype=object))
    tmp.replace(f)
    json.dump({"model": a.model, "hf_id": MODELS[a.model][0], "dataset": a.dataset,
               "videos": int(len(Y)), "time_groups": int(Tg), "regions": REGION_NAMES,
               "region_boxes": {k: list(v) for k, v in BLOCKS.items()},
               "dim": int(Z.shape[-1]), "shape": list(Z.shape),
               "note": "regions are the I-JEPA brief's five boxes on the 16x16 grid and OVERLAP by design",
               "minutes": round(secs / 60, 1)}, open(f.with_suffix(".json"), "w"), indent=1)
    print(f"  -> {f.name} {Z.shape} in {secs/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

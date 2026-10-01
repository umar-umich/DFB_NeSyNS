#!/usr/bin/env python3
"""Part 6a: cache per-video features from a video backbone, for the TD (video view) upgrade screen.

    CUDA_VISIBLE_DEVICES=3 <temporal_vjepa2 python> upgrade/extract_video.py \
        --model vjepa2_h --dataset FFpp_val

WHICH CHECKPOINT, AND WHY NOT "V-JEPA 2.1". The brief asks for V-JEPA 2.1 ViT-L/16 and says to
confirm the released checkpoint. Meta has NOT released a 2.1: the hub carries only `facebook/vjepa2-*`
from August 2025. The single "2.1" is `apiantonio/vjepa2.1-vit-large-384`, third party, two likes,
`custom_code`, 384 input -- unverified provenance for what would become a headline view, and 384 from
a 256 crop is upsampling as well. Substituted instead: `facebook/vjepa2-vith-fpc64-256`, which is
official, is a genuine capacity step over the deployed ViT-L, and runs at the SAME 256 the crops are
stored at, so the comparison against TD changes capacity and nothing else.

CLIPS. Identical to the deployed TD view: 16 frames sampled uniformly IN ORDER over a video's stored
crops, via `train_branch.Clips`, so TD and the candidate see the same clips and the comparison is the
backbone alone. Reusing that class rather than reimplementing it also keeps the video-identity rule
and the short-video repeat behaviour the same.

CACHED. Global pooled features (mean over tokens), one row per video. Dense regional pooling for
Part 6c is a separate pass and is not done here.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
OUT = ROOT / "logs/upgrade/vfeats"
sys.path[:0] = [str(ROOT / "analysis/tbiom"), str(ROOT / "analysis/tbiom/xclip"),
                str(ROOT / "analysis/tbiom/temporal"), "/data/umar/Repos/DISCERN_Ext"]
import train_branch as TB  # noqa: E402

torch.set_num_threads(4)

MODELS = {  # name -> (hf id, frames per clip)
    "vjepa2_l": ("facebook/vjepa2-vitl-fpc64-256", 16),   # the DEPLOYED TD backbone, for the bar
    "vjepa2_h": ("facebook/vjepa2-vith-fpc64-256", 16),   # the official capacity upgrade
}


class Encoder(torch.nn.Module):
    """Frozen V-JEPA 2 encoder with the model's own processor, token-mean pooled."""

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
        # skip_predictor keeps the encoder output; the JEPA predictor is a pretraining head.
        return self.backbone(pixel_values_videos=px, skip_predictor=True).last_hidden_state.mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="probe on the first N videos only")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{a.model}_{a.dataset}.npz"
    if f.exists():
        print(f"{f.name} exists"); return 0
    dev = "cuda"
    enc = Encoder(a.model).to(dev)
    n_par = sum(p.numel() for p in enc.backbone.parameters()) / 1e6
    print(f"{a.model}: {n_par:.0f} M params, {enc.T} frames/clip", flush=True)

    ds = TB.Clips(a.dataset, enc.preprocess, enc.T)
    if a.limit:
        ds.items = ds.items[:a.limit]
    loader = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, num_workers=a.workers,
                                         worker_init_fn=lambda _: torch.set_num_threads(1))
    Z, Y, V, t0, n = [], [], [], time.time(), 0
    items = ds.items
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for px, y, i in loader:
            z = enc(px.to(dev, non_blocking=True))
            Z.append(z.float().cpu().numpy()); Y.append(y.numpy())
            V += [items[int(k)][2] for k in i.tolist()]
            n += len(y)
            if n % (a.batch_size * 25) < a.batch_size:
                print(f"  {n}/{len(items)} videos, {n/(time.time()-t0):.1f} vid/s", flush=True)
    Z = np.concatenate(Z).astype(np.float16); Y = np.concatenate(Y)
    secs = time.time() - t0
    if a.limit:
        print(f"PROBE: {n} videos in {secs/60:.1f} min -> {n/secs:.2f} vid/s", flush=True)
        return 0
    tmp = f.with_suffix(".partial.npz")
    np.savez(tmp, feature=Z, label=Y, video=np.array(V, dtype=object))
    tmp.replace(f)
    json.dump({"model": a.model, "hf_id": MODELS[a.model][0], "params_M": round(n_par, 1),
               "frames_per_clip": enc.T, "input": 256, "upsampled_fraction": 0.0,
               "dataset": a.dataset, "videos": int(len(Y)), "dim": int(Z.shape[1]),
               "minutes": round(secs / 60, 1)},
              open(f.with_suffix(".json"), "w"), indent=1)
    print(f"  -> {f.name}  {Z.shape}  in {secs/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

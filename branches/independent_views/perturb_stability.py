#!/usr/bin/env python3
"""Part 6: training-free perturbation stability (RIGID, He et al. 2024; Tsai et al. 2024 on faces).

    CUDA_VISIBLE_DEVICES=2 <discern_ext_timm python> branches/independent_views/perturb_stability.py \
        --backbone dinov3 --dataset FFpp_val

For each face crop, the cosine DISTANCE between its frozen embedding and the embedding of a perturbed
copy. A real face is expected to be more stable under perturbation than a manipulated one, so the
distance is used directly as a detection score with no training at all.

Perturbations, as the brief fixes them:
    gauss noise sd 0.01, gauss noise sd 0.05, gauss blur sigma 0.5, gauss blur sigma 1.0,
    and the MINIMUM distance over the four, which is the follow-up study's minimum-distance detector.

The perturbation is chosen on FF++ c23 validation and then applied unchanged elsewhere.

Compression check, mirroring G1: on FF++ validation REAL videos, the shift in distance from clean to
JPEG q50 must be smaller than the real-versus-fake gap on clean videos. The point is the same one G1
makes -- a score that moves more with compression than with manipulation is measuring compression.
"""
from __future__ import annotations

import argparse, io, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path.insert(0, str(ROOT / "branches/independent_views"))
sys.path.insert(0, str(ROOT / "branches/ijepa"))
from extract_backbone import SPECS, build, stream  # noqa: E402
from data import video_of                          # noqa: E402

torch.set_num_threads(4)
OUT = ROOT / "logs/independent_views/perturb"
PERTS = {"noise0.01": ("noise", 0.01), "noise0.05": ("noise", 0.05),
         "blur0.5": ("blur", 0.5), "blur1.0": ("blur", 1.0)}


def perturb(x, kind, val, gen):
    """x is a float tensor in [0,1], (B,3,H,W)."""
    if kind == "noise":
        return (x + torch.randn(x.shape, generator=gen, device=x.device) * val).clamp(0, 1)
    k = max(3, int(2 * round(3 * val) + 1))
    coords = torch.arange(k, dtype=torch.float32, device=x.device) - k // 2
    g = torch.exp(-(coords ** 2) / (2 * val ** 2)); g = (g / g.sum()).view(1, 1, 1, k)
    pad = k // 2
    y = F.conv2d(F.pad(x, (pad, pad, 0, 0), mode="reflect"), g.expand(3, 1, 1, k), groups=3)
    return F.conv2d(F.pad(y, (0, 0, pad, pad), mode="reflect"),
                    g.transpose(2, 3).expand(3, 1, k, 1), groups=3).clamp(0, 1)


def jpeg(x, q=50):
    u8 = (x * 255).round().clamp(0, 255).to(torch.uint8).cpu()
    out = []
    for img in u8:
        b = io.BytesIO()
        Image.fromarray(img.permute(1, 2, 0).numpy()).save(b, format="JPEG", quality=q); b.seek(0)
        out.append(torch.from_numpy(np.asarray(Image.open(b).convert("RGB"))).permute(2, 0, 1))
    return (torch.stack(out).float() / 255.0).to(x.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=list(SPECS))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--jpeg-check", action="store_true",
                    help="also score a JPEG q50 copy, for the compression check")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{a.backbone}_{a.dataset}{'_jpeg50' if a.jpeg_check else ''}.npz"
    if f.exists():
        print(f"{f.name} exists"); return
    m, mean, std, spec = build(a.backbone, dev)
    from extract_backbone import embed
    gen = torch.Generator(device=dev).manual_seed(0)      # the same perturbation draw for every model
    D = {k: [] for k in PERTS}; K = []
    t0, n = time.time(), 0
    for arrs, keys in stream(a.dataset, 8192, spec["size"]):
        for i in range(0, len(arrs), a.batch_size):
            x = torch.from_numpy(arrs[i:i + a.batch_size]).permute(0, 3, 1, 2).float().div_(255).to(dev)
            if a.jpeg_check:
                x = jpeg(x, 50)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z0 = F.normalize(embed(m, spec, (x - mean) / std, dev).float(), dim=1)
                for name, (kind, val) in PERTS.items():
                    xp = perturb(x, kind, val, gen)
                    zp = F.normalize(embed(m, spec, (xp - mean) / std, dev).float(), dim=1)
                    D[name].append((1 - (z0 * zp).sum(1)).cpu().numpy())
        K += keys; n += len(arrs)
        print(f"  {n} frames, {n/(time.time()-t0):.0f} f/s", flush=True)
    K = np.array(K, dtype=object)
    arrs = {k: np.concatenate(v).astype(np.float32) for k, v in D.items()}
    arrs["min"] = np.minimum.reduce(list(arrs.values()))
    y = np.array([0 if "real" in str(k).split("/")[-3].lower() else 1 for k in K], dtype=np.int64)
    np.savez(f, key=K, label=y, video=np.array([video_of(str(k)) for k in K], dtype=object), **arrs)
    print(f"  -> {f.name}", flush=True)


if __name__ == "__main__":
    main()

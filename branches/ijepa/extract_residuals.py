#!/usr/bin/env python3
"""I-JEPA regional prediction residuals: per frame, per block, hide a face region and measure the prediction error.

    CUDA_VISIBLE_DEVICES=3 python branches/ijepa/extract_residuals.py --dataset FFpp_train --limit-videos 100 --time-only

Everything in I-JEPA stays frozen; nothing is trained here. Context uses the `encoder` weights, targets use the
`target_encoder` weights followed by layer normalisation over the feature dimension, as in I-JEPA training. The
hidden block never enters the context encoder.

Per frame and per block k it stores the mean squared residual eps_k and the mean residual vector rbar_k (1280-d),
plus the mean-pooled target feature of the whole face (1280-d) for the R2 control, and label-free quality
descriptors. Everything is cached fp16.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")   # shared box: fill the GPU, not the CPU
import argparse, io, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(4)

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
IJ = ROOT / "branches/ijepa"
sys.path.insert(0, str(IJ / "ijepa"))
from src.helper import init_model  # noqa: E402
from src.masks.utils import apply_masks  # noqa: E402

CKPT = IJ / "IN1K-vit.h.14-300e.pth.tar"
EXT = Path("/data/umar/Repos/DISCERN_Ext/eval_adaptation/data")
G, P = 16, 14
BLOCKS = {"upper": (2, 9, 4, 12), "lower": (9, 16, 4, 12), "left_cheek": (5, 12, 0, 8),
          "right_cheek": (5, 12, 8, 16), "centre": (4, 11, 4, 12)}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_models(dev):
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    strip = lambda sd: {k.replace("module.", ""): v for k, v in sd.items()}
    cfg = dict(patch_size=14, model_name="vit_huge", crop_size=224, pred_depth=12, pred_emb_dim=384)
    enc, pred = init_model(device=dev, **cfg)
    tgt, _ = init_model(device=dev, **cfg)
    enc.load_state_dict(strip(ck["encoder"]), strict=True)
    pred.load_state_dict(strip(ck["predictor"]), strict=True)
    tgt.load_state_dict(strip(ck["target_encoder"]), strict=True)
    for m in (enc, pred, tgt):
        m.eval().requires_grad_(False)
    return enc, pred, tgt


def masks(dev, B):
    out = {}
    for b, (r0, r1, c0, c1) in BLOCKS.items():
        hid = [r * G + c for r in range(r0, r1) for c in range(c0, c1)]
        vis = [i for i in range(G * G) if i not in set(hid)]
        out[b] = (torch.tensor(vis, device=dev).repeat(B, 1), torch.tensor(hid, device=dev).repeat(B, 1))
    return out


def quality(arr):
    """Label-free per-frame descriptors: blur, blockiness, and crop size."""
    g = arr.astype(np.float32).mean(2)
    lap = (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1])
    blur = float(lap.var())
    d = np.abs(np.diff(g, axis=1))
    blockiness = float(d[:, 7::8].mean() / (d.mean() + 1e-6))     # 8-pixel JPEG grid vs all edges
    return blur, blockiness, float(g.shape[0])


@torch.no_grad()
def run(frames, enc, pred, tgt, dev, bs):
    """frames: uint8 (N,224,224,3). Returns eps (N,5), rbar (N,5,1280), pooled (N,1280)."""
    E, R, PL = [], [], []
    for i in range(0, len(frames), bs):
        x = torch.from_numpy(frames[i:i + bs]).permute(0, 3, 1, 2).float().div_(255)
        x = ((x - MEAN) / STD).to(dev, non_blocking=True)
        B = x.shape[0]
        h = tgt(x)
        h = F.layer_norm(h, (h.size(-1),))
        PL.append(h.mean(1).half().cpu().numpy())
        mk = masks(dev, B)
        e_b, r_b = [], []
        for b in BLOCKS:
            ctx, hid = mk[b]
            z = enc(x, [ctx])                       # hidden patches never enter the context encoder
            yh = pred(z, [ctx], [hid])
            y = apply_masks(h, [hid])
            r = y - yh
            e_b.append((r ** 2).mean((1, 2)).half().cpu().numpy())
            r_b.append(r.mean(1).half().cpu().numpy())
        E.append(np.stack(e_b, 1)); R.append(np.stack(r_b, 1))
    return np.concatenate(E), np.concatenate(R), np.concatenate(PL)


def stream_frames(dataset, limit_videos, chunk, shard, nshards):
    """Yield (uint8 (n,224,224,3), keys) in chunks, so a 200k-frame set never sits in memory at once."""
    import h5py
    from PIL import Image
    man_dir = EXT / "splits" / dataset
    lines = []
    for m in sorted(man_dir.glob("*.txt")):
        lines += [l.strip() for l in m.read_text().splitlines() if l.strip()]
    byvid = {}
    for l in lines:
        byvid.setdefault(l.split("/")[-2], []).append(l)
    vids = sorted(byvid)
    if limit_videos:
        vids = vids[:limit_videos]
    if nshards > 1:
        vids = vids[shard::nshards]          # shard by video, so a video is never split across shards
    keys = [k for v in vids for k in byvid[v]]
    arrs, kept = [], []
    with h5py.File(h5p_of(dataset), "r") as f:
        for k in keys:
            if k not in f:
                continue
            a = f[k][()]
            im = Image.open(io.BytesIO(a.tobytes())) if a.dtype == np.uint8 and a.ndim == 1 \
                else Image.fromarray(np.asarray(a).astype(np.uint8))
            arrs.append(np.asarray(im.convert("RGB").resize((224, 224), Image.BICUBIC)))
            kept.append(k)
            if len(arrs) >= chunk:
                yield np.stack(arrs), kept, len(vids)
                arrs, kept = [], []
    if arrs:
        yield np.stack(arrs), kept, len(vids)


def h5p_of(dataset):
    return EXT / "h5" / f"{dataset}.h5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="FFpp_train")
    ap.add_argument("--limit-videos", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--chunk-frames", type=int, default=8192,
                    help="frames decoded into memory at a time; the GPU pass runs per chunk")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out", default="", help="cache stem; defaults to the dataset (plus the shard suffix)")
    ap.add_argument("--time-only", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    stem = a.out or (a.dataset if a.num_shards == 1 else f"{a.dataset}.s{a.shard}of{a.num_shards}")
    cache = Path("analysis/ijepa/cache") / f"{stem}.npz"
    if cache.exists() and not a.overwrite and not a.time_only:
        print(f"{cache} exists, nothing to do")           # resume: a finished shard is never redone
        return

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    enc, pred, tgt = load_models(dev)
    t_load = time.time() - t0

    E, R, PL, Q, K = [], [], [], [], []
    t_read = t_gpu = 0.0
    nvid = 0
    tc = time.time()
    for arrs, keys, nvid in stream_frames(a.dataset, a.limit_videos, a.chunk_frames, a.shard, a.num_shards):
        t_read += time.time() - tc
        t1 = time.time()
        eps, rbar, pooled = run(arrs, enc, pred, tgt, dev, a.batch_size)
        t_gpu += time.time() - t1
        E.append(eps); R.append(rbar); PL.append(pooled)
        Q.append(np.stack([quality(x) for x in arrs]).astype(np.float16))
        K += keys
        print(f"  {stem}: {len(K)} frames, {t_gpu:.0f}s gpu, {len(K) / max(t_gpu, 1e-9):.1f} f/s", flush=True)
        if a.time_only:
            break
        tc = time.time()

    eps = np.concatenate(E); rbar = np.concatenate(R); pooled = np.concatenate(PL); q = np.concatenate(Q)
    out = {"dataset": a.dataset, "stem": stem, "videos": nvid, "frames": len(K),
           "frames_per_video": round(len(K) / max(nvid, 1), 2),
           "seconds": {"model_load": round(t_load, 1), "h5_read": round(t_read, 1), "gpu": round(t_gpu, 1)},
           "frames_per_second_gpu": round(len(K) / max(t_gpu, 1e-9), 2),
           "eps_mean_per_block": {b: float(eps[:, i].astype(np.float32).mean())
                                  for i, b in enumerate(BLOCKS)},
           "bytes_per_frame_cached": int(eps[0].nbytes + rbar[0].nbytes + pooled[0].nbytes + q[0].nbytes)}
    print(json.dumps(out, indent=1))
    Path("analysis/ijepa").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(f"analysis/ijepa/timing_{stem}.json", "w"), indent=1)
    if not a.time_only:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".npz.partial")          # a killed job never leaves a half cache behind
        np.savez(tmp, eps=eps, rbar=rbar, pooled=pooled, q=q, keys=np.array(K, object))
        tmp.replace(cache)
        print("cached ->", cache)


if __name__ == "__main__":
    main()

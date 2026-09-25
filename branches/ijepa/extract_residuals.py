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


# Arm B's processing grid, copied from DISCERN_Ext/src/dataset/training_arms.py:SharedProcessing.
# Same severities and same weights; the draw here is per VIDEO, as the brief asks, not per sample.
ARM_B_GRID = [("none", None), ("jpeg", 70), ("jpeg", 50), ("jpeg", 30),
              ("resize", 0.75), ("resize", 0.50), ("resize", 0.35)]
ARM_B_W = np.array([0.40, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10])
ARM_B_W = ARM_B_W / ARM_B_W.sum()


def apply_uint8(img, setting):
    """Arm B processing on a native-size crop, before the 224 resize, as in training."""
    from PIL import Image
    kind, val = setting
    if kind == "none":
        return img
    im = Image.fromarray(img)
    if kind == "jpeg":
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=int(val))
        buf.seek(0)
        return np.array(Image.open(buf).convert("RGB"))
    w, h = im.size
    small = im.resize((max(1, int(w * val)), max(1, int(h * val))), Image.BILINEAR)
    return np.array(small.resize((w, h), Image.BILINEAR).convert("RGB"))


def draw_arm_b(video, seed=42):
    """One setting per clip, reproducible from the video name alone."""
    import zlib
    rng = np.random.default_rng([seed, zlib.crc32(video.encode())])
    return ARM_B_GRID[int(rng.choice(len(ARM_B_GRID), p=ARM_B_W))]


def apply_eval_process(x, spec):
    """Evaluation-side processing on the [0,1] 224 batch, mirroring export_views.apply_processing,
    which denormalises, processes and renormalises -- so applying here, before normalisation, is
    the same operation at the same point in the pipeline."""
    import torchvision.transforms.functional as TF
    from PIL import Image
    kind, _, val = spec.partition(":")
    if kind == "resize":
        sc = float(val)
        h, w = x.shape[-2:]
        return TF.resize(TF.resize(x, [max(1, int(h * sc)), max(1, int(w * sc))], antialias=True),
                         [h, w], antialias=True)
    if kind == "jpeg":
        q = int(val)
        u8 = (x * 255).round().clamp(0, 255).to(torch.uint8).cpu()
        out = []
        for img in u8:
            buf = io.BytesIO()
            Image.fromarray(img.permute(1, 2, 0).numpy()).save(buf, format="JPEG", quality=q)
            buf.seek(0)
            out.append(TF.pil_to_tensor(Image.open(buf).convert("RGB")))
        return (torch.stack(out).float() / 255.0).to(x.device)
    raise ValueError(f"unknown processing spec {spec!r}")


def quality(arr):
    """Label-free per-frame descriptors: blur, blockiness, and crop size."""
    g = arr.astype(np.float32).mean(2)
    lap = (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1])
    blur = float(lap.var())
    d = np.abs(np.diff(g, axis=1))
    blockiness = float(d[:, 7::8].mean() / (d.mean() + 1e-6))     # 8-pixel JPEG grid vs all edges
    return blur, blockiness, float(g.shape[0])


@torch.no_grad()
def run(frames, enc, pred, tgt, dev, bs, amp="bf16", process=""):
    """frames: uint8 (N,224,224,3). Returns eps (N,5), rbar (N,5,1280), pooled (N,1280),
    sbar (N,5,1280) and q (N,3) -- the quality descriptors of the PROCESSED pixels.

    The forward runs under bf16 autocast by default. Everything is cached fp16 regardless, so
    fp16 storage, not bf16 compute, sets the precision of what is kept.
    """
    E, R, PL, S, Q = [], [], [], [], []
    ctxm = (torch.autocast("cuda", dtype=torch.bfloat16) if amp == "bf16" and dev == "cuda"
            else torch.autocast("cuda", enabled=False))
    for i in range(0, len(frames), bs):
        x = torch.from_numpy(frames[i:i + bs]).permute(0, 3, 1, 2).float().div_(255)
        if process:
            x = apply_eval_process(x, process)
        # The quality descriptors MUST come from the pixels the residual is computed from. They used
        # to be taken from the pre-processing array, so eps described a processed frame while q
        # described a clean one, and the standardiser was asked to predict a shift from descriptors
        # that could not see it. Nothing failed; mu_k(q) simply returned the same value at every
        # processing level and absorbed none of the confound.
        Q.append(np.stack([quality(f) for f in
                           (x * 255).clamp(0, 255).permute(0, 2, 3, 1).numpy()]).astype(np.float16))
        x = ((x - MEAN) / STD).to(dev, non_blocking=True)
        B = x.shape[0]
        with ctxm:
            h = tgt(x)
            h = F.layer_norm(h.float(), (h.size(-1),))
            PL.append(h.mean(1).half().cpu().numpy())
            mk = masks(dev, B)
            e_b, r_b, s_b = [], [], []
            for b in BLOCKS:
                ctx, hid = mk[b]
                z = enc(x, [ctx])                   # hidden patches never enter the context encoder
                yh = pred(z, [ctx], [hid]).float()
                y = apply_masks(h, [hid])
                r = y - yh
                e_b.append((r ** 2).mean((1, 2)).half().cpu().numpy())
                r_b.append(r.mean(1).half().cpu().numpy())
                # s_bar_k for the R2b regional feature control (amendment 1, section A). y is the
                # layer-normalised target feature at this block's 56 patches and is already computed,
                # so this costs nothing and is guaranteed to share the frame's processing draw.
                s_b.append(y.mean(1).half().cpu().numpy())
        E.append(np.stack(e_b, 1)); R.append(np.stack(r_b, 1)); S.append(np.stack(s_b, 1))
    return (np.concatenate(E), np.concatenate(R), np.concatenate(PL), np.concatenate(S),
            np.concatenate(Q))


def stream_frames(dataset, limit_videos, chunk, shard, nshards, arm_b=False):
    """Yield (uint8 (n,224,224,3), keys) in chunks, so a 200k-frame set never sits in memory at once."""
    import h5py
    from PIL import Image
    man_dir = EXT / "splits" / dataset
    mans = sorted(man_dir.glob("*.txt"))
    # DF40 ships 80 manifests but the study reports 41 generator sets, whose configs name only 44 of
    # them. The other 36 -- entire-face-synthesis (DiT, SiT, StyleGAN*, VQGAN, ddim, sd2.1, rddm,
    # e4e), wav2lip lip-sync and heygen-real -- feed no reported number and are the families the v3.1
    # protocol excludes anyway. Extracting them would cost 347,947 frames for nothing. The keep list
    # is DERIVED from the configs (analysis/ijepa/df40_manifests.txt) rather than hardcoded here.
    keep_file = ROOT / "analysis/ijepa" / f"{dataset.lower()}_manifests.txt"
    if keep_file.exists():
        keep = {x.strip() for x in keep_file.read_text().split() if x.strip()}
        mans = [m for m in mans if m.stem in keep]
        if not mans:
            raise ValueError(f"{keep_file.name} matched none of {dataset}'s manifests")
    lines = []
    for m in mans:
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
    setting_of = {v: draw_arm_b(v) for v in vids} if arm_b else {}
    with h5py.File(h5p_of(dataset), "r") as f:
        for k in keys:
            if k not in f:
                continue
            a = f[k][()]
            im = Image.open(io.BytesIO(a.tobytes())) if a.dtype == np.uint8 and a.ndim == 1 \
                else Image.fromarray(np.asarray(a).astype(np.uint8))
            u8 = np.asarray(im.convert("RGB"))
            if arm_b:
                u8 = apply_uint8(u8, setting_of[k.split("/")[-2]])
            arrs.append(np.asarray(Image.fromarray(u8).resize((224, 224), Image.BICUBIC)))
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
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--arm-b", action="store_true", help="B mixture, one draw per clip (FF++ train)")
    ap.add_argument("--arm-b-at-224", action="store_true",
                    help="draw the B setting per clip but apply it AFTER the 224 resize, where the "
                         "evaluation severities are applied. Used only to fit a standardiser whose "
                         "descriptor range covers the evaluation cells: arm B applied at native 256 "
                         "and then resized smears the 8x8 JPEG grid, so its blockiness never reaches "
                         "the values the evaluation cells produce and the fit extrapolates there.")
    ap.add_argument("--process", default="", help="eval-side processing, e.g. jpeg:40 or resize:0.60")
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

    E, R, PL, SB, Q, K = [], [], [], [], [], []
    t_read = t_gpu = 0.0
    nvid = 0
    tc = time.time()
    for arrs, keys, nvid in stream_frames(a.dataset, a.limit_videos, a.chunk_frames, a.shard,
                                          a.num_shards, a.arm_b and not a.arm_b_at_224):
        t_read += time.time() - tc
        t1 = time.time()
        proc = a.process
        if a.arm_b_at_224:
            # one setting per clip, same draw rule, applied where evaluation applies it
            vids = [k.split("/")[-2] for k in keys]
            eps_l, rb_l, pl_l, sb_l, q_l = [], [], [], [], []
            import itertools as _it
            for v, grp in _it.groupby(range(len(keys)), key=lambda i: vids[i]):
                idx = list(grp)
                st_ = draw_arm_b(v)
                spec = "" if st_[0] == "none" else (f"jpeg:{st_[1]}" if st_[0] == "jpeg"
                                                    else f"resize:{st_[1]}")
                e_, r_, p_, s_, q_ = run(arrs[idx], enc, pred, tgt, dev, a.batch_size, a.amp, spec)
                eps_l.append(e_); rb_l.append(r_); pl_l.append(p_); sb_l.append(s_); q_l.append(q_)
            eps, rbar, pooled = np.concatenate(eps_l), np.concatenate(rb_l), np.concatenate(pl_l)
            sbar, qc = np.concatenate(sb_l), np.concatenate(q_l)
        else:
            eps, rbar, pooled, sbar, qc = run(arrs, enc, pred, tgt, dev, a.batch_size, a.amp, proc)
        t_gpu += time.time() - t1
        E.append(eps); R.append(rbar); PL.append(pooled); SB.append(sbar)
        Q.append(qc)
        K += keys
        print(f"  {stem}: {len(K)} frames, {t_gpu:.0f}s gpu, {len(K) / max(t_gpu, 1e-9):.1f} f/s", flush=True)
        if a.time_only:
            break
        tc = time.time()

    eps = np.concatenate(E); rbar = np.concatenate(R); pooled = np.concatenate(PL)
    sbar = np.concatenate(SB); q = np.concatenate(Q)
    out = {"dataset": a.dataset, "stem": stem, "amp": a.amp,
           "arm_b": bool(a.arm_b), "process": a.process or "clean", "videos": nvid, "frames": len(K),
           "frames_per_video": round(len(K) / max(nvid, 1), 2),
           "seconds": {"model_load": round(t_load, 1), "h5_read": round(t_read, 1), "gpu": round(t_gpu, 1)},
           "frames_per_second_gpu": round(len(K) / max(t_gpu, 1e-9), 2),
           "eps_mean_per_block": {b: float(eps[:, i].astype(np.float32).mean())
                                  for i, b in enumerate(BLOCKS)},
           "bytes_per_frame_cached": int(eps[0].nbytes + rbar[0].nbytes + pooled[0].nbytes
                                        + sbar[0].nbytes + q[0].nbytes)}
    print(json.dumps(out, indent=1))
    Path("analysis/ijepa").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(f"analysis/ijepa/timing_{stem}.json", "w"), indent=1)
    if not a.time_only:
        cache.parent.mkdir(parents=True, exist_ok=True)
        # NOTE: np.savez APPENDS .npz when the name does not already end in it, so the temp
        # name must itself end in .npz or the file lands somewhere the rename cannot find.
        tmp = cache.with_suffix(".partial.npz")          # a killed job never leaves a half cache behind
        np.savez(tmp, eps=eps, rbar=rbar, pooled=pooled, sbar=sbar, q=q, keys=np.array(K, object))
        tmp.replace(cache)
        print("cached ->", cache)


if __name__ == "__main__":
    main()

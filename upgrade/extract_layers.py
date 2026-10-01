#!/usr/bin/env python3
"""Part 3d: cache the readout of SEVERAL depths in one pass, for the layer-readout study.

    CUDA_VISIBLE_DEVICES=1 <discern_ext_timm python> upgrade/extract_layers.py \
        --backbone pe_core_l --dataset FFpp_val

The premise (UMAR, 2026-10-01): the final block is the most task-abstract and tends to be the most
robust to processing, while an intermediate block keeps detail a forgery detector may need on clean
frames. The deployed appearance view reads the last block only and that has never been tested.

One forward pass yields every depth, so this costs no more than the Part 3a extraction it mirrors.
Hooks capture the patch-token output of the chosen blocks; `last` keeps the model's own global
readout so the `last` row is byte-comparable with the Part 3b cache rather than a near-copy of it.

DEPTH INDICES ARE RECORDED, not described. "Two-thirds of 24 blocks" could mean 15 or 16, and the
index is what another lab needs to reproduce this, so the sidecar JSON carries the integer.

PROCESSING. `--process jpeg:30` or `resize:0.35` degrades the decoded crop BEFORE the model's
normalisation, which is how the robustness half of the keep rule is measured.
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
OUT = ROOT / "logs/upgrade/lfeats"
sys.path[:0] = [str(ROOT / "upgrade"), str(ROOT / "branches/ijepa")]
from extract_spatial import SPECS  # noqa: E402
from data import video_of          # noqa: E402

torch.set_num_threads(4)
DEPTHS = {"mid": 2 / 3, "half": 1 / 2}


def blocks_of(m, kind):
    if kind == "hf":
        return list(m.encoder.layers)
    for attr in ("blocks", "layers"):
        if hasattr(m, attr):
            return list(getattr(m, attr))
    raise RuntimeError("cannot locate transformer blocks")


def degrade(im: Image.Image, spec: str | None) -> Image.Image:
    if not spec:
        return im
    kind, val = spec.split(":")
    if kind == "jpeg":
        b = io.BytesIO(); im.save(b, format="JPEG", quality=int(val)); b.seek(0)
        return Image.open(b).convert("RGB")
    w, h = im.size; s = float(val)
    sm = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
    return sm.resize((w, h), Image.BILINEAR).convert("RGB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=list(SPECS))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--process", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{a.backbone}_{a.dataset}" + (f"_{a.process.replace(':','')}" if a.process else "")
    f = OUT / f"{tag}.npz"
    if f.exists():
        print(f"{f.name} exists"); return 0

    spec = SPECS[a.backbone]; dev = "cuda"
    if spec["kind"] == "hf":
        from transformers import CLIPVisionModel
        m = CLIPVisionModel.from_pretrained(spec["id"]).vision_model.to(dev).eval()
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(dev)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(dev)
    else:
        import timm
        m = timm.create_model(spec["id"], pretrained=True, num_classes=0).to(dev).eval()
        cfg = m.pretrained_cfg
        mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).to(dev)
        std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).to(dev)
    for p in m.parameters():
        p.requires_grad_(False)

    blocks = blocks_of(m, spec["kind"])
    n_blk = len(blocks)
    idx = {k: max(0, min(n_blk - 1, int(round(n_blk * r)) - 1)) for k, r in DEPTHS.items()}
    grabbed: dict[str, torch.Tensor] = {}

    def hook(name):
        def fn(_mod, _inp, out):
            o = out[0] if isinstance(out, tuple) else out
            grabbed[name] = o.detach()
        return fn

    for name, i in idx.items():
        blocks[i].register_forward_hook(hook(name))
    print(f"{a.backbone}: {n_blk} blocks, depths {idx}, input {spec['size']}"
          + (f", process {a.process}" if a.process else ""), flush=True)

    man = sorted((EXT / "splits" / a.dataset).glob("*.txt"))
    lines = [l.strip() for mm in man for l in mm.read_text().splitlines() if l.strip()]
    feats = {k: [] for k in ["last", *idx]}
    keys, t0 = [], time.time()
    with h5py.File(EXT / "h5" / f"{a.dataset}.h5", "r") as h5:
        buf, bk = [], []
        for k in lines:
            if k not in h5:
                continue
            arr = h5[k][()]
            im = (Image.open(io.BytesIO(arr.tobytes())) if arr.dtype == np.uint8 and arr.ndim == 1
                  else Image.fromarray(np.asarray(arr).astype(np.uint8))).convert("RGB")
            im = degrade(im, a.process)
            if im.size != (spec["size"],) * 2:
                im = im.resize((spec["size"],) * 2, Image.BICUBIC)
            buf.append(np.asarray(im, dtype=np.uint8)); bk.append(k)
            if len(buf) == a.batch_size:
                x = torch.from_numpy(np.stack(buf)).permute(0, 3, 1, 2).float().div_(255).to(dev)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    z = m((x - mean) / std)
                    last = z.last_hidden_state[:, 0] if spec["kind"] == "hf" else z
                feats["last"].append(last.float().cpu().numpy())
                for nm in idx:                       # mean over PATCH tokens, class token dropped
                    g = grabbed[nm]
                    feats[nm].append(g[:, 1:].mean(1).float().cpu().numpy())
                keys += bk; buf, bk = [], []
                if len(keys) % 8192 < a.batch_size:
                    print(f"  {len(keys)} frames, {len(keys)/(time.time()-t0):.0f} f/s", flush=True)
        if buf:
            x = torch.from_numpy(np.stack(buf)).permute(0, 3, 1, 2).float().div_(255).to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                z = m((x - mean) / std)
                last = z.last_hidden_state[:, 0] if spec["kind"] == "hf" else z
            feats["last"].append(last.float().cpu().numpy())
            for nm in idx:
                feats[nm].append(grabbed[nm][:, 1:].mean(1).float().cpu().numpy())
            keys += bk

    K = np.array(keys, dtype=object)
    arrs = {f"feat_{k}": np.concatenate(v).astype(np.float16) for k, v in feats.items()}
    y = np.array([0 if "real" in str(k).split("/")[-3].lower() else 1 for k in K], dtype=np.int64)
    tmp = f.with_suffix(".partial.npz")
    np.savez(tmp, key=K, label=y, video=np.array([video_of(str(k)) for k in K], dtype=object), **arrs)
    tmp.replace(f)
    json.dump({"backbone": a.backbone, "id": spec["id"], "dataset": a.dataset,
               "process": a.process, "n_blocks": n_blk, "block_index": idx,
               "input_size": spec["size"], "frames": int(len(K)),
               "dims": {k: int(v.shape[1]) for k, v in arrs.items()},
               "minutes": round((time.time() - t0) / 60, 1)},
              open(f.with_suffix(".json"), "w"), indent=1)
    print(f"  -> {f.name}  " + ", ".join(f"{k} {v.shape}" for k, v in arrs.items()), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

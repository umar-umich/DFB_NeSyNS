#!/usr/bin/env python3
"""PE-Core frame features from the DENSE stride-1 crops, for the S-dense / P-dense pilot.

    CUDA_VISIBLE_DEVICES=1 <discern_ext_timm python> upgrade/branchfusion/dense_features.py \
        --part ffpp_train --backbone pe_core_l
    ... --part dev2

WHY THIS EXISTS. Every temporal branch in this project -- V, M and P alike -- has been trained on
frames sampled UNIFORMLY ACROSS THE WHOLE VIDEO. Measured from the caches, consecutive stored frames
are 10 to 22 source frames apart, so a "clip" spans several seconds and holds almost no motion
continuity. That is the most likely reason P's temporal stage is order-invariant to three decimal
places, and it is a property of the sampling rather than of the architecture.

`/data/umar/Datasets/tbiom_dense` already holds the fix: 32 CONSECUTIVE frames per video at stride 1,
median span 1.2 s, for FF++ train (3,596 videos, 115,072 frames) and the development pool. It was
extracted on 2026-09-20 and never used. So the dense pilot section 2.2 pre-registers costs one
feature extraction, not a re-decode.

The output lands in the same `{backbone}_{dataset}.npz` layout every other cache uses, with the
dataset name suffixed `-dense`, so `train_seq.py --dense` and `common.load_frames` read it with no
special case. Frame indices come from the key exactly as they do for the sparse caches, so the
timestamp encoding is unchanged -- it simply now sees consecutive integers.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                      # noqa: E402

sys.path[:0] = [str(C.ROOT / "upgrade"), str(C.ROOT / "branches/ijepa")]
from extract_spatial import SPECS                       # noqa: E402
from extract_layers import blocks_of, degrade, DEPTHS    # noqa: E402
from data import video_of                               # noqa: E402

C.thread_caps()

DENSE = Path("/data/umar/Datasets/tbiom_dense")
#: Which h5 holds which partition, and which dataset names it carries.
PARTS = {
    "ffpp_train": {"h5": DENSE / "ffpp_train_dense.h5", "keys": DENSE / "ffpp_train_dense_keys.txt",
                   "manifest": DENSE / "ffpp_train_dense_manifest.json",
                   "datasets": ("FFpp_train",)},
    "dev2": {"h5": DENSE / "dev2_dense.h5", "keys": DENSE / "dev2_dense_keys.txt",
             "manifest": DENSE / "dev2_dense_manifest.json",
             "datasets": ("FFpp_val", "VALmix", "VALmixV2")},
}


def label_of(key: str) -> int:
    """The established rule: the class directory is the third component from the end.

    Identical expression to `upgrade/extract_spatial.py:141`, so dense labels cannot disagree with
    sparse ones. `FFpp-real`, `CDFv2val-Celeb-real` and `DFEval24val-real` all contain "real".
    """
    return 0 if "real" in str(key).split("/")[-3].lower() else 1


def dataset_of(key: str) -> str:
    """`VALmix/CDFv2val-Celeb-real/<vid>/000214.png` -> `VALmix`."""
    return str(key).split("/", 1)[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=sorted(PARTS))
    ap.add_argument("--backbone", default="pe_core_l", choices=list(SPECS))
    ap.add_argument("--depth", default=None, choices=[None, *sorted(DEPTHS)],
                    help="omit for the model's own global readout; 'mid' for the block-16 patch mean")
    ap.add_argument("--process", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    part = PARTS[a.part]
    for p in (part["h5"], part["keys"]):
        if not p.exists():
            raise FileNotFoundError(p)
    tag_bb = a.backbone + (f"-{a.depth}" if a.depth else "")
    want = {ds: C.FEATS / f"{tag_bb}_{ds}-dense.npz" for ds in part["datasets"]}
    if all(f.exists() for f in want.values()) and not a.force:
        print("all dense caches exist: " + ", ".join(f.name for f in want.values()))
        return 0

    spec = SPECS[a.backbone]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    import timm
    m = timm.create_model(spec["id"], pretrained=True, num_classes=0).to(dev).eval()
    cfg = m.pretrained_cfg
    mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).to(dev)
    std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).to(dev)
    for p in m.parameters():
        p.requires_grad_(False)
    n_prefix = int(getattr(m, "num_prefix_tokens", 1))

    grabbed: dict = {}
    if a.depth:
        blocks = blocks_of(m, spec["kind"])
        di = max(0, min(len(blocks) - 1, int(round(len(blocks) * DEPTHS[a.depth])) - 1))

        def hook(_mod, _inp, out):
            grabbed["z"] = (out[0] if isinstance(out, tuple) else out).detach()
        blocks[di].register_forward_hook(hook)
        print(f"{a.backbone}: depth '{a.depth}' -> block index {di} (1-based {di + 1}), "
              f"prefix {n_prefix}", flush=True)

    keys = [l.strip() for l in part["keys"].read_text().splitlines() if l.strip()]
    span = None
    if part["manifest"].exists():
        try:
            span = json.load(open(part["manifest"]))["summary"]
        except Exception:
            span = None
    print(f"{a.part}: {len(keys)} dense frames, input {spec['size']}, "
          f"manifest summary {span}", flush=True)

    acc = {ds: {"f": [], "k": []} for ds in part["datasets"]}
    t0 = time.time()
    with h5py.File(part["h5"], "r") as h5:
        buf, bk = [], []

        def flush():
            if not buf:
                return
            x = torch.from_numpy(np.stack(buf)).permute(0, 3, 1, 2).float().div_(255).to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=(dev == "cuda")):
                z = m((x - mean) / std)
                f = (grabbed["z"][:, n_prefix:].float().mean(1) if a.depth else z.float())
            fn = f.cpu().numpy().astype(np.float16)
            for kk, row in zip(bk, fn):
                ds = dataset_of(kk)
                if ds in acc:
                    acc[ds]["f"].append(row)
                    acc[ds]["k"].append(kk)
            buf.clear()
            bk.clear()

        for k in keys:
            if k not in h5:
                continue
            arr = h5[k][()]
            im = (Image.open(io.BytesIO(arr.tobytes())) if arr.dtype == np.uint8 and arr.ndim == 1
                  else Image.fromarray(np.asarray(arr).astype(np.uint8))).convert("RGB")
            im = degrade(im, a.process)
            if im.size != (spec["size"],) * 2:
                im = im.resize((spec["size"],) * 2, Image.BICUBIC)
            buf.append(np.asarray(im, dtype=np.uint8))
            bk.append(k)
            if len(buf) == a.batch_size:
                n_done = sum(len(v["k"]) for v in acc.values())
                flush()
                if n_done % 16384 < a.batch_size:
                    print(f"  {n_done} frames, {n_done / max(time.time() - t0, 1):.0f} f/s",
                          flush=True)
        flush()

    out = {}
    for ds, v in acc.items():
        if not v["k"]:
            print(f"  {ds}: no frames")
            continue
        K = np.array(v["k"], dtype=object)
        F = np.stack(v["f"])
        vid = np.array([video_of(str(k)) for k in K], dtype=object)
        y = np.array([label_of(k) for k in K], dtype=np.int64)
        f = want[ds]
        np.savez_compressed(f, feature=F, key=K, video=vid, label=y)
        meta = {"backbone": a.backbone, "timm_id": spec["id"], "dataset": f"{ds}-dense",
                "depth": a.depth, "prefix_tokens": n_prefix, "input_size": spec["size"],
                "crop_resolution": 256, "process": a.process, "frames": int(len(K)),
                "videos": int(len(np.unique(vid))), "dim": int(F.shape[1]),
                "sampling": "stride 1, consecutive, from tbiom_dense",
                "source_h5": str(part["h5"]), "manifest_summary": span,
                "adapted": False, "minutes": round((time.time() - t0) / 60, 1)}
        meta["config_hash"] = C.config_hash({k: meta[k] for k in
                                             ("backbone", "dataset", "depth", "input_size",
                                              "sampling")})
        json.dump(meta, open(f.with_suffix(".json"), "w"), indent=1)
        out[ds] = f"{len(K)} frames, {meta['videos']} videos -> {f.name}"
        print(f"  {ds}: {out[ds]}", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

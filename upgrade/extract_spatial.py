#!/usr/bin/env python3
"""Part 3: frozen global face embeddings from candidate backbones, for the Appearance-view screen.

    CUDA_VISIBLE_DEVICES=2 <discern_ext_timm python> branches/independent_views/extract_backbone.py \
        --backbone dinov3 --dataset FFpp_train

RESOLUTION. The brief forbids upsampling the stored crops. Every crop in this corpus is 256 px, so a
backbone is only run at an input size <= 256:

    CLIP ViT-L/14      224   downscale from 256
    DINOv3 ViT-L/16    256   NATIVE, no resize at all
    PE-Core B16        224   downscale from 256
    PE-Core L14/336    336   NOT RUN -- would be upsampling

PE therefore enters the screen at 92.9 M against 303 M for CLIP and DINOv3. That size gap is a
confound and is recorded rather than hidden; the brief's own PE-Core-L14-336 control is the way to
separate backbone from scale, and it can only be run by accepting the upsampling.

EMBEDDINGS, one per backbone, as the brief specifies:
    CLIP     pre-projection class token (`vision_model` last_hidden_state[:, 0], pre-`post_layernorm`)
    DINOv3   class token
    PE-Core  pooled output
    I-JEPA   mean of patch tokens -- already cached as `pooled` by the residual extractor, not redone

AUGMENTATION. FF++ c23 train is extracted CLEAN, because the deployed Appearance view trains with no
augmentation: `warm_dicome_gate_seed42.yaml` carries no `training_arm` key, and arm B lives in a
separate config. This is the same asymmetry recorded in the joint-IBDC report.
"""
from __future__ import annotations

import argparse, io, json, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
EXT = Path("/data/umar/Repos/DISCERN_Ext/eval_adaptation/data")
OUT = ROOT / "logs/upgrade/feats"
sys.path.insert(0, str(ROOT / "branches/ijepa"))
from data import video_of  # noqa: E402  THE video rule, shared

torch.set_num_threads(4)

SPECS = {
    "clip":   {"kind": "hf",   "id": "openai/clip-vit-large-patch14",   "size": 224, "readout": "cls_pre_proj"},
    "dinov3": {"kind": "timm", "id": "vit_large_patch16_dinov3",        "size": 256, "readout": "cls"},
    "pe":     {"kind": "timm", "id": "vit_pe_core_base_patch16_224",    "size": 224, "readout": "pooled"},
    # CROSS-EXAM brief Part 3a. PE-SPATIAL, unlike PE-Core, has no checkpoint at or below 256: Meta
    # released only L14-448 and G14-448, and the smallest loadable one anywhere is timm's base/512.
    # Every stored crop is 256, so this candidate is 100% UPSAMPLED, by 2.0x. The earlier brief
    # forbade that and PE was dropped; this brief instead asks for the fraction to be reported, so it
    # runs and the number is in the report. Scale is also not matched to CLIP/DINOv3 -- 86M against
    # 303M -- so a loss here is not evidence against the PE family.
    "pe_spatial": {"kind": "timm", "id": "vit_pe_spatial_base_patch16_512", "size": 512, "readout": "pooled"},
    # PE-Core LARGE, added 2026-10-01 on Dr Kutub's suggestion that PE-Core is strong spatially.
    # `pe` above is PE-Core BASE at 224: 86M against CLIP ViT-L/14's 303M, so when it lost the
    # earlier screen the result confounded backbone with capacity. This is the size-MATCHED control
    # the independent-views report said was needed and could not run, because 256 -> 336 is an
    # upsample and that brief forbade it. This brief asks for the fraction instead, so it runs:
    # 100% upsampled, 1.31x -- far milder than pe_spatial's 2.0x.
    "pe_core_l": {"kind": "timm", "id": "vit_pe_core_large_patch14_336", "size": 336, "readout": "pooled"},
}


def build(name, dev):
    spec = SPECS[name]
    if spec["kind"] == "hf":
        from transformers import CLIPVisionModel
        m = CLIPVisionModel.from_pretrained(spec["id"]).vision_model.to(dev).eval()
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    else:
        import timm
        m = timm.create_model(spec["id"], pretrained=True, num_classes=0).to(dev).eval()
        cfg = m.pretrained_cfg
        mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1)
        std = torch.tensor(cfg["std"]).view(1, 3, 1, 1)
    for p in m.parameters():
        p.requires_grad_(False)
    return m, mean.to(dev), std.to(dev), spec


@torch.no_grad()
def embed(m, spec, x, dev):
    if spec["kind"] == "hf":
        return m(x).last_hidden_state[:, 0]            # pre-projection class token
    return m(x)                                        # timm num_classes=0: cls or pooled per model


def stream(dataset, chunk, size):
    import h5py
    man = sorted((EXT / "splits" / dataset).glob("*.txt"))
    lines = [l.strip() for m in man for l in m.read_text().splitlines() if l.strip()]
    arrs, keys = [], []
    with h5py.File(EXT / "h5" / f"{dataset}.h5", "r") as f:
        for k in lines:
            if k not in f:
                continue
            a = f[k][()]
            im = (Image.open(io.BytesIO(a.tobytes())) if a.dtype == np.uint8 and a.ndim == 1
                  else Image.fromarray(np.asarray(a).astype(np.uint8))).convert("RGB")
            if im.size != (size, size):
                # Downscale for every backbone except pe_spatial, where 256 -> 512 is an upsample.
                # `upsampled` in the sidecar JSON records which it was, per the brief.
                im = im.resize((size, size), Image.BICUBIC)
            arrs.append(np.asarray(im)); keys.append(k)
            if len(arrs) >= chunk:
                yield np.stack(arrs), keys; arrs, keys = [], []
    if arrs:
        yield np.stack(arrs), keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=list(SPECS))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=8192)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{a.backbone}_{a.dataset}.npz"
    if f.exists():
        print(f"{f.name} exists"); return
    m, mean, std, spec = build(a.backbone, dev)
    print(f"{a.backbone}: {sum(p.numel() for p in m.parameters())/1e6:.1f} M, input {spec['size']}, "
          f"readout {spec['readout']}", flush=True)
    F, K, t0, n = [], [], time.time(), 0
    for arrs, keys in stream(a.dataset, a.chunk, spec["size"]):
        for i in range(0, len(arrs), a.batch_size):
            x = torch.from_numpy(arrs[i:i + a.batch_size]).permute(0, 3, 1, 2).float().div_(255).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = embed(m, spec, (x - mean) / std, dev)
            F.append(z.float().cpu().numpy())
        K += keys; n += len(arrs)
        print(f"  {n} frames, {n/(time.time()-t0):.0f} f/s", flush=True)
    F = np.concatenate(F).astype(np.float16)
    K = np.array(K, dtype=object)
    y = np.array([0 if "real" in str(k).split("/")[-3].lower() else 1 for k in K], dtype=np.int64)
    tmp = f.with_suffix(".partial.npz")
    np.savez(tmp, feature=F, key=K, label=y, video=np.array([video_of(str(k)) for k in K], dtype=object))
    tmp.replace(f)
    json.dump({"backbone": a.backbone, "hf_or_timm_id": spec["id"], "input_size": spec["size"],
               "stored_crop": 256, "upsampled_fraction": 1.0 if spec["size"] > 256 else 0.0,
               "upsample_factor": round(spec["size"] / 256, 3),
               "readout": spec["readout"], "dataset": a.dataset, "frames": int(len(K)),
               "dim": int(F.shape[1]), "seconds": round(time.time() - t0, 1)},
              open(ROOT / f"analysis/independent_views/feat_{a.backbone}_{a.dataset}.json", "w"), indent=1)
    print(f"  -> {f.name}  {F.shape}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Frozen X-CLIP video features for the temporal-branch probe (brief Experiment 3, step 1), plus controls.

    CUDA_VISIBLE_DEVICES=1 python analysis/tbiom/xclip/extract_xclip.py --dataset CDFv2 [--process jpeg:40]

Per video (the same face crops the detector sees, from our h5 files), 16 frames sampled uniformly over the stored
frames, in temporal order:
  xclip_temporal   X-CLIP (microsoft/xclip-base-patch16-16-frames) video embedding, frames in order      (512-d)
  xclip_shuffled   same model and frames, frame ORDER shuffled (fixed per video)  -> order sensitivity    (512-d)
  xclip_framemean  mean of X-CLIP's per-frame projected CLS embeddings (its encoder still exchanges
                   cross-frame message tokens, so this is not purely spatial)                             (512-d)
  clip_framemean   plain CLIP ViT-B/16 image embeddings averaged over the same frames -> spatial control   (512-d)

Caveat recorded with the features: our h5 stores up to 32 frames per video sampled across the whole clip, so
16 of them are ~0.5-1 s apart. Temporal dynamics are therefore coarse (no dense tracked clips).

Processing (--process) reproduces export_views.apply_processing exactly: applied to the CLIP-normalised 224x224 batch
by de-normalising, JPEG/resize on 8-bit pixels, re-normalising. Video ids use the shared video_id rule on
"<tst_h5_path>::<key>", so features join the detector exports row for row.
"""
from __future__ import annotations
import argparse, io, sys, time
from pathlib import Path
import h5py, numpy as np, pandas as pd, torch, yaml
import torchvision.transforms.functional as TF
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from video_id import video_id  # noqa: E402
from transformers import AutoProcessor, CLIPModel, XCLIPModel  # noqa: E402

torch.set_num_threads(4)  # shared box (load > 1000): torch would otherwise spawn one thread per core
EXT = Path("/data/umar/Repos/DISCERN_Ext")
OUT = Path("/data/umar/Repos/DFB_NeSyNS/logs/tbiom/xclip/feats")
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
# X-CLIP's own HF processor (VideoMAEImageProcessor) normalises with ImageNet statistics, not OpenAI CLIP's.
# Out of the study's scope (protocol v3.1): never extracted, to save time.
OUT_OF_SCOPE = {"IP_LAP", "fake-audio-driven", "adv-real", "adv-fake",
                "retalking", "wav2lip", "diff2lip", "latentsync", "memo",
                "ddim", "DiT", "pixart", "RDDM", "sd2.1", "SiT", "StyleGAN2", "StyleGAN3", "StyleGANXL", "VQGAN"}
XCLIP_ID, CLIP_ID, T = "microsoft/xclip-base-patch16-16-frames", "openai/clip-vit-base-patch16", 16


def apply_processing(images: torch.Tensor, spec: str, mean_t, std_t) -> torch.Tensor:
    """Logic of DISCERN_Ext/eval_adaptation/export_views.py::apply_processing, with the normalisation of the model
    whose input this is (the pixel operation on the 224x224 8-bit crop is identical)."""
    mean = torch.tensor(mean_t, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(std_t, device=images.device).view(1, 3, 1, 1)
    x = (images * std + mean).clamp(0, 1)
    kind, _, val = spec.partition(":")
    if kind == "resize":
        scale = float(val); h, w = x.shape[-2:]
        small = TF.resize(x, [max(1, int(h * scale)), max(1, int(w * scale))], antialias=True)
        x = TF.resize(small, [h, w], antialias=True)
    elif kind == "jpeg":
        q = int(val); u8 = (x * 255).round().clamp(0, 255).to(torch.uint8).cpu(); out = []
        for img in u8:
            buf = io.BytesIO(); Image.fromarray(img.permute(1, 2, 0).numpy()).save(buf, format="JPEG", quality=q)
            buf.seek(0); out.append(TF.pil_to_tensor(Image.open(buf).convert("RGB")))
        x = (torch.stack(out).float() / 255.0).to(images.device)
    else:
        raise ValueError(spec)
    return (x - mean) / std


def dataset_spec(ds: str):
    cfg = EXT / "eval_adaptation/configs" / ("run1_auxedl.yaml" if ds == "FFpp_train" else f"run1auxedl_{ds}.yaml")
    c = yaml.safe_load(open(cfg))
    if ds == "FFpp_train":
        return c["trn_h5_path"], list(c["trn_files"])
    return c["tst_h5_path"], [f for fs in c["tst_files"].values() for f in fs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--process", default=None)
    ap.add_argument("--videos-per-batch", type=int, default=8); ap.add_argument("--limit-videos", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    h5rel, lists = dataset_spec(a.dataset)
    keys = [k for f in lists for k in (EXT / f).read_text().split("\n") if k]
    keys = [k for k in keys if not (k.split("/")[-3] in OUT_OF_SCOPE or k.split("/")[-3].split("-")[-1] in OUT_OF_SCOPE)]
    vids: dict[str, list[str]] = {}
    for k in keys:
        vids.setdefault(k.rsplit("/", 1)[0], []).append(k)
    order = sorted(vids)[a.shard::a.nshards]
    if a.limit_videos:
        order = order[:a.limit_videos]
    tag = a.dataset + (f"_{a.process.replace(':', '').replace('.', '')}" if a.process else "") + \
        (f".s{a.shard}of{a.nshards}" if a.nshards > 1 else "")
    out = a.out or OUT / f"{tag}.npz"; out.parent.mkdir(parents=True, exist_ok=True)

    dev = "cuda"
    xproc = AutoProcessor.from_pretrained(XCLIP_ID).image_processor      # each model gets its OWN preprocessing
    cproc = AutoProcessor.from_pretrained(CLIP_ID).image_processor
    assert tuple(round(x, 6) for x in cproc.image_mean) == tuple(round(x, 6) for x in CLIP_MEAN)
    xclip = XCLIPModel.from_pretrained(XCLIP_ID).to(dev).eval().half()
    clip = CLIPModel.from_pretrained(CLIP_ID).to(dev).eval().half()
    rng = np.random.default_rng(0)
    rows = {"vid": [], "label": [], "subset": [], "n_stored": []}
    feats = {"xclip_temporal": [], "xclip_shuffled": [], "xclip_framemean": [], "clip_framemean": []}
    t0 = time.time()
    with h5py.File(EXT / h5rel, "r") as h, torch.no_grad():
        for b in range(0, len(order), a.videos_per_batch):
            batch = order[b:b + a.videos_per_batch]
            frames, perms = [], []
            for v in batch:
                ks = sorted(vids[v])
                idx = np.linspace(0, len(ks) - 1, T).round().astype(int)
                frames += [np.asarray(h[ks[i]]) for i in idx]
                perms.append(rng.permutation(T))
                rows["vid"].append(video_id(pd.Series([f"{h5rel}::{ks[0]}"])).iloc[0])
                rows["label"].append(0 if "real" in ks[0].split("/")[-3].lower() else 1)
                rows["subset"].append(ks[0].split("/")[-3]); rows["n_stored"].append(len(ks))
            pil = [Image.fromarray(f) for f in frames]
            # VideoMAEImageProcessor treats a frame list as ONE video: (1, N, 3, 224, 224) -> flatten to frames
            px = xproc(images=pil, return_tensors="pt")["pixel_values"].reshape(-1, 3, 224, 224).to(dev)
            pc = cproc(images=pil, return_tensors="pt")["pixel_values"].to(dev)
            if a.process:
                px = apply_processing(px.float(), a.process, xproc.image_mean, xproc.image_std)
                pc = apply_processing(pc.float(), a.process, CLIP_MEAN, CLIP_STD)
            px, pc = px.half(), pc.half()
            B = len(batch); pv = px.view(B, T, *px.shape[1:])
            cls = xclip.visual_projection(xclip.vision_model(pixel_values=px)[1]).view(B, T, -1)
            feats["xclip_temporal"].append(xclip.mit(cls)[1].float().cpu())
            feats["xclip_framemean"].append(cls.mean(1).float().cpu())
            shuf = torch.stack([pv[i, torch.as_tensor(perms[i], device=dev)] for i in range(B)]).view_as(px)
            cls_s = xclip.visual_projection(xclip.vision_model(pixel_values=shuf)[1]).view(B, T, -1)
            feats["xclip_shuffled"].append(xclip.mit(cls_s)[1].float().cpu())
            feats["clip_framemean"].append(clip.get_image_features(pixel_values=pc).view(B, T, -1).mean(1).float().cpu())
            if (b // a.videos_per_batch) % 50 == 0:
                print(f"  {b + B}/{len(order)} videos  {time.time() - t0:.0f}s", flush=True)
    np.savez(out.with_suffix(".tmp.npz"), **{k: np.asarray(v) for k, v in rows.items()},
             **{k: torch.cat(v).numpy() for k, v in feats.items()},
             meta=np.asarray([XCLIP_ID, CLIP_ID, str(T), a.process or "none", h5rel]))
    out.with_suffix(".tmp.npz").rename(out)
    print(f"wrote {out}: {len(rows['vid'])} videos in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

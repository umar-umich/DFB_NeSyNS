#!/usr/bin/env python3
"""Frozen video-model features on our stored face-crop frames: TimeSformer, VideoMAE, V-JEPA 2 (temporal panel).

    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 python analysis/tbiom/temporal/extract_video.py --model timesformer --dataset CDFv2
    (V-JEPA 2 runs in the temporal_vjepa2 env: a clone of discern_ext_timm with transformers >= 4.53)

For every video, T frames sampled uniformly over the stored crops (<= 32, ~0.25-0.6 s apart), passed twice:
  ordered   frames in temporal order
  shuffled  the same frames in a fixed random order (seed 0)  -> the order-sensitivity control
Features: TimeSformer CLS (768-d); VideoMAE mean of the last hidden state (768-d); V-JEPA 2 mean of the encoder's last
hidden state (1024-d). Each model uses its own HF preprocessing and normalisation. --process reproduces
export_views.apply_processing on the model-sized crop (8-bit JPEG / down-up resize, re-normalised with the model's
statistics). Same scope filter, shards and video-id rule as the X-CLIP extractor, so all features join the detector
exports row for row.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import h5py, numpy as np, pandas as pd, torch
from PIL import Image
torch.set_num_threads(4)  # shared box
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "xclip"))
from video_id import video_id  # noqa: E402
from extract_xclip import apply_processing, dataset_spec, OUT_OF_SCOPE, EXT  # noqa: E402

MODELS = {"timesformer": ("facebook/timesformer-base-finetuned-k400", 8),
          "videomae": ("MCG-NJU/videomae-base-finetuned-kinetics", 16),
          "vjepa2": ("facebook/vjepa2-vitl-fpc64-256", 16)}
OUT = Path("/data/umar/Repos/DFB_NeSyNS/logs/tbiom/temporal/feats")


class Encoder:
    def __init__(self, name, dev):
        mid, self.T = MODELS[name]; self.name = name; self.dev = dev
        if name == "vjepa2":
            from transformers import AutoModel, AutoVideoProcessor
            self.proc = AutoVideoProcessor.from_pretrained(mid)
            self.model = AutoModel.from_pretrained(mid).to(dev).eval().half()
        else:
            from transformers import AutoImageProcessor, TimesformerModel, VideoMAEModel
            self.proc = AutoImageProcessor.from_pretrained(mid)
            self.model = (TimesformerModel if name == "timesformer" else VideoMAEModel).from_pretrained(mid).to(dev).eval().half()
        self.mean, self.std = self.proc.image_mean, self.proc.image_std

    def pixels(self, videos: list[list[np.ndarray]]) -> torch.Tensor:
        """(B, T, 3, H, W), normalised by the model's own processor."""
        if self.name == "vjepa2":
            # The video processor stacks frames before resizing, so mixed crop sizes (CDFv3 validation, DFDM) must be
            # brought to one size first; 256x256 is the model crop and leaves the usual 256-px crops unchanged.
            fix = lambda f: f if f.shape[:2] == (256, 256) else np.asarray(Image.fromarray(f).resize((256, 256), Image.BILINEAR))
            out = [self.proc(np.stack([fix(f) for f in v]), return_tensors="pt")["pixel_values_videos"][0] for v in videos]
            return torch.stack(out)
        return self.proc([[Image.fromarray(f) for f in v] for v in videos], return_tensors="pt")["pixel_values"]

    @torch.no_grad()
    def embed(self, px: torch.Tensor) -> torch.Tensor:
        px = px.to(self.dev).half()
        if self.name == "timesformer":
            return self.model(pixel_values=px).last_hidden_state[:, 0].float().cpu()
        if self.name == "videomae":
            return self.model(pixel_values=px).last_hidden_state.mean(1).float().cpu()
        return self.model(pixel_values_videos=px, skip_predictor=True).last_hidden_state.mean(1).float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS)); ap.add_argument("--dataset", required=True)
    ap.add_argument("--process", default=None); ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1); ap.add_argument("--videos-per-batch", type=int, default=8)
    ap.add_argument("--limit-videos", type=int, default=0); ap.add_argument("--out", type=Path, default=None)
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
    out = a.out or OUT / a.model / f"{tag}.npz"; out.parent.mkdir(parents=True, exist_ok=True)
    enc = Encoder(a.model, "cuda"); T = enc.T
    rng = np.random.default_rng(0)
    rows = {"vid": [], "label": [], "subset": [], "n_stored": []}; feats = {"ordered": [], "shuffled": []}
    t0 = time.time()
    with h5py.File(EXT / h5rel, "r") as h:
        for b in range(0, len(order), a.videos_per_batch):
            batch, videos = order[b:b + a.videos_per_batch], []
            for v in batch:
                ks = sorted(vids[v]); idx = np.linspace(0, len(ks) - 1, T).round().astype(int)
                videos.append([np.asarray(h[ks[i]]) for i in idx])
                rows["vid"].append(video_id(pd.Series([f"{h5rel}::{ks[0]}"])).iloc[0])
                rows["label"].append(0 if "real" in ks[0].split("/")[-3].lower() else 1)
                rows["subset"].append(ks[0].split("/")[-3]); rows["n_stored"].append(len(ks))
            px = enc.pixels(videos).float()                           # (B, T, 3, H, W)
            if a.process:
                B_, T_ = px.shape[:2]
                px = apply_processing(px.flatten(0, 1).to(enc.dev), a.process, enc.mean, enc.std).view(B_, T_, *px.shape[2:])
            perm = torch.stack([torch.as_tensor(rng.permutation(T)) for _ in batch])
            shuf = torch.stack([px[i, perm[i].to(px.device)] for i in range(len(batch))])
            feats["ordered"].append(enc.embed(px)); feats["shuffled"].append(enc.embed(shuf))
            if (b // a.videos_per_batch) % 50 == 0:
                print(f"  {b + len(batch)}/{len(order)} videos  {time.time() - t0:.0f}s", flush=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, **{k: np.asarray(v) for k, v in rows.items()}, **{k: torch.cat(v).numpy() for k, v in feats.items()},
             meta=np.asarray([a.model, MODELS[a.model][0], str(T), a.process or "none", h5rel]))
    tmp.rename(out)
    print(f"wrote {out}: {len(rows['vid'])} videos in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

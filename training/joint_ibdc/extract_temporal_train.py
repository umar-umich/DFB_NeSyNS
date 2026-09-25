#!/usr/bin/env python3
"""FF++ c23 TRAIN per-clip features for the V-JEPA 2 video view, with MATCHED arm-B draws (Task 2).

    <temporal_vjepa2 python> training/joint_ibdc/extract_temporal_train.py --verify
    CUDA_VISIBLE_DEVICES=3 <temporal_vjepa2 python> training/joint_ibdc/extract_temporal_train.py --draws 3

Caches the exact input of the video view's evidential head: z = mean over tokens of the backbone's
last hidden state, 1024-d, BEFORE the head's LayerNorm. The head itself
(LayerNorm -> 128 GELU -> 2 -> softplus) stays frozen and is applied at training time, exactly as the
spatial cache stores pre-norm blocks.

OPTION B, MATCHED DRAWS. The arm-B setting comes from `draws.setting_for(video, draw)`, the same
function the spatial extractor uses, so a given video gets the SAME processing in every view for a
given draw. The deployed temporal loader instead draws from a per-worker RNG, which is reproducible
only as a distribution and cannot be matched across views; that is replaced here deliberately and the
deviation is recorded. Everything else about the clip -- 16 frames sampled uniformly in order, the
256-px resize, the model's own processor -- follows `train_branch.Clips` unchanged.

The two FF++ train sources were checked to be the same files before relying on this: the temporal
side reads `run1_auxedl.yaml`'s trn_files and the spatial side my `warmgate_FFpp_train.yaml`, and
both name the identical five manifests and the identical h5, so video identities line up.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
EXT = Path("/data/umar/Repos/DISCERN_Ext")
sys.path[:0] = [str(ROOT / "analysis/tbiom/temporal"), str(ROOT / "analysis/tbiom/xclip"),
                str(ROOT / "analysis/tbiom"), str(EXT), str(Path(__file__).resolve().parent)]

import h5py                              # noqa: E402
import train_branch as TB                # noqa: E402
from draws import setting_for, apply_to, SharedProcessing  # noqa: E402

BR = ROOT / "logs/tbiom/temporal/branches/vjepa2_lora_B_seed42"
OUT = ROOT / "logs/joint_ibdc/cache"
DATASET, T = "FFpp_train", 16


class MatchedClips(torch.utils.data.Dataset):
    """train_branch.Clips with the per-worker RNG replaced by the shared per-video draw."""

    def __init__(self, ds, proc, draw, arm="B"):
        self.h5rel, self.items = TB.videos_of(ds)
        self.proc, self.draw, self.arm = proc, draw, arm
        self.processing = SharedProcessing(enabled=(arm == "B"))
        self.h = None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        if self.h is None:
            self.h = h5py.File(EXT / self.h5rel, "r")
        ks, y, v = self.items[i]
        idx = np.linspace(0, len(ks) - 1, T).round().astype(int)
        fr = [np.asarray(self.h[ks[j]]) for j in idx]
        fr = [f if f.shape[:2] == (256, 256)
              else np.asarray(Image.fromarray(f).resize((256, 256), Image.BILINEAR)) for f in fr]
        st = setting_for(v, self.draw, self.arm, proc=self.processing)     # matched to the spatial cache
        if st[0] != "none":
            fr = [apply_to(f, st, self.processing) for f in fr]
        return self.proc(fr), y, i, f"{st[0]}:{st[1]}" if st[1] is not None else "none"


def load_model(dev):
    m = TB.Branch("vjepa2", lora=True).to(dev)
    ck = torch.load(BR / "best.pt", map_location=dev)
    m.trainable_blocks.load_state_dict(ck["trainable_blocks"])
    m.head.load_state_dict(ck["head"])
    m.eval()
    print(f"  development-selected epoch {ck['epoch']}", flush=True)
    return m


@torch.no_grad()
def extract(model, draw, dev, bs=8, workers=6):
    ds = MatchedClips(DATASET, model.preprocess, draw)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=workers, shuffle=False,
                                     worker_init_fn=lambda _: torch.set_num_threads(1))
    Z, Y, V, S = [], [], [], []
    items = ds.items
    t0 = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for bi, (px, y, i, st) in enumerate(dl):
            z = model.backbone(pixel_values_videos=px.to(dev),
                               skip_predictor=True).last_hidden_state.mean(1).float()
            Z.append(z.cpu().numpy()); Y.append(np.asarray(y))
            V += [items[int(k)][2] for k in i]; S += list(st)
            if bi % 50 == 0:
                n = (bi + 1) * bs
                print(f"    draw {draw} batch {bi} ~{n} clips, "
                      f"{n / max(time.time() - t0, 1e-9):.1f} clips/s", flush=True)
    return (np.concatenate(Z), np.concatenate(Y), np.array(V, dtype=object),
            np.array(S, dtype=object), time.time() - t0)


def verify():
    """No GPU: the temporal draws must equal the spatial ones for the same video."""
    _h5, items = TB.videos_of(DATASET)
    print(f"  {len(items)} training videos")
    proc = SharedProcessing(enabled=True)
    for d in range(3):
        from collections import Counter
        c = Counter(f"{a}:{b}" for a, b in (setting_for(v, d, proc=proc) for _k, _y, v in items))
        tot = sum(c.values())
        print(f"  draw {d} mix: " + ", ".join(f"{k} {v / tot:.2f}" for k, v in sorted(c.items())))
    # a video id and one of its frame keys must give the same setting
    bad = [v for ks, _y, v in items[:400] if setting_for(v, 0, proc=proc) != setting_for(ks[0], 0, proc=proc)]
    print(f"  video id and frame key agree on {400 - len(bad)}/400 videos")
    assert not bad, bad[:3]
    print("verification OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--arm", default="B", choices=["B", "none"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(4)
    if a.verify:
        return verify()
    dev = "cuda"
    OUT.mkdir(parents=True, exist_ok=True)
    model = load_model(dev)
    meta = {"view": "T (V-JEPA 2)", "branch": str(BR), "arm": a.arm, "draws": a.draws,
            "feature": "backbone last_hidden_state.mean(1), 1024-d, before the head LayerNorm",
            "draw_granularity": "per video, matched to the spatial cache (option B)",
            "deviation": "the deployed temporal loader draws arm B from a per-worker RNG; that is "
                         "replaced by the shared per-video draw so the views match",
            "per_draw": []}
    for d in range(a.draws):
        f = OUT / f"FFpp_train_T_{a.arm}_d{d}.npz"
        if f.exists():
            print(f"  skip draw {d} (exists)"); continue
        Z, Y, V, S, secs = extract(model, d, dev, a.batch_size, a.workers)
        tmp = f.with_suffix(".partial.npz")
        np.savez_compressed(tmp, feature=Z, label=Y, video=V, setting=S)
        tmp.replace(f)
        meta["per_draw"].append({"draw": d, "clips": int(len(Y)), "seconds": round(secs, 1),
                                 "file": str(f), "size_bytes": f.stat().st_size})
        print(f"  draw {d}: {len(Y)} clips in {secs / 60:.1f} min -> {f.name}", flush=True)
    json.dump(meta, open(ROOT / "analysis/joint_ibdc/ffpp_train_temporal_cache.json", "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()

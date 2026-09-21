#!/usr/bin/env python3
"""Stage 1A inputs (executable brief): segment-level temporal opinions and quality descriptors, clean AND processed.

    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 <temporal_vjepa2 python> analysis/tbiom/acquisition/cache_segments.py \
        --datasets VALmix VALmixV2 FFpp_val --process none jpeg:40 resize:0.6

Why a v2 of the acquisition cache (see REFERENCE_MANIFEST.md §4, the Stage 0 gate):
  * the controllers train on clean and processed examples, and processed segment opinions did not exist;
  * processing is applied exactly as every earlier processed export applied it -- `apply_processing` on the model's
    input tensor -- so the `full` segment of each condition must reproduce the existing full-span processed opinion
    (`--check` measures it);
  * quality descriptors are computed from that same model input, after processing, on the DISTINCT crops of the
    segment only (a clip padded to 16 frames by repeating crops would otherwise report zero motion);
  * an empty segment (a video with fewer crops than segments) gets NO row. v1 filled it with the full-span opinion,
    which is the per-segment simulation the brief forbids.
Segments: the video's sorted stored crops split into 4 contiguous parts (`np.array_split`), each sampled to 16 frames
by `linspace` in order -- so a segment of 4-8 distinct crops repeats them (recorded as `n_src`). `t0`/`t1` are the
source-frame indices of the segment's first and last crop, parsed from the crop file names.
"""
from __future__ import annotations
import argparse, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, h5py
from PIL import Image
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "temporal"), str(HERE.parent), str(HERE.parent / "xclip")]
import train_branch as TB  # noqa: E402
from extract_xclip import apply_processing  # noqa: E402

torch.set_num_threads(4)
OUT = Path("logs/tbiom/acquisition/v2")
N_SEG, T = 4, 16


def frame_no(key):
    m = re.findall(r"\d+", key.rsplit("/", 1)[-1])
    return int(m[-1]) if m else -1


class Segments(torch.utils.data.Dataset):
    def __init__(self, ds, proc):
        self.h5rel, self.items = TB.videos_of(ds); self.proc, self.h = proc, None
        self.pairs = []
        for i, (ks, _y, _v) in enumerate(self.items):
            parts = np.array_split(np.arange(len(ks)), N_SEG)
            for s, part in enumerate(parts):
                if len(part):                                   # empty segment -> no row, never a filled one
                    self.pairs.append((i, s, part))
            self.pairs.append((i, N_SEG, np.arange(len(ks))))   # the full-span observation

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, n):
        if self.h is None:
            self.h = h5py.File(TB.EXT / self.h5rel, "r")
        i, s, part = self.pairs[n]
        ks = [self.items[i][0][j] for j in part]
        idx = np.linspace(0, len(ks) - 1, T).round().astype(int)
        fr = [np.asarray(self.h[ks[j]]) for j in idx]
        fr = [f if f.shape[:2] == (256, 256) else np.asarray(Image.fromarray(f).resize((256, 256), Image.BILINEAR)) for f in fr]
        first = np.r_[True, idx[1:] != idx[:-1]].astype(np.float32)          # positions of distinct crops
        return self.proc(fr), n, len(ks), frame_no(ks[0]), frame_no(ks[-1]), torch.from_numpy(first)


def descriptors(x, first):
    """x: (B, T, 3, H, W) in [0, 1] -- the processed model input. Computed on distinct crops only."""
    g = x.mean(2) * 255.0                                                   # (B, T, H, W)
    lap = (g[..., :-2, 1:-1] + g[..., 2:, 1:-1] + g[..., 1:-1, :-2] + g[..., 1:-1, 2:] - 4 * g[..., 1:-1, 1:-1])
    lapv = lap.flatten(2).var(2)                                            # (B, T)
    wts = first / first.sum(1, keepdim=True)
    blur = (lapv * wts).sum(1)
    out = {"blur": blur, "luma": (g.flatten(2).mean(2) * wts).sum(1), "contrast": (g.flatten(2).std(2) * wts).sum(1)}
    mot = []
    for b in range(g.shape[0]):
        d = g[b][first[b].bool()]
        mot.append((d[1:] - d[:-1]).abs().mean() if len(d) > 1 else torch.tensor(0.0, device=g.device))
    out["motion"] = torch.stack(mot)
    mid = g[:, g.shape[1] // 2]
    cols = (mid[..., 1:] - mid[..., :-1]).abs()
    out["blockiness"] = cols[..., 7::8].mean((1, 2)) / (cols.mean((1, 2)) + 1e-6)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", default="vjepa2_lora_B_seed42")
    ap.add_argument("--datasets", nargs="+", default=["VALmix", "VALmixV2", "FFpp_val"])
    ap.add_argument("--process", nargs="+", default=["none", "jpeg:40", "resize:0.6"])
    ap.add_argument("--batch", type=int, default=48); ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    out = OUT / a.branch; out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    model = TB.Branch("vjepa2", lora=True).to(dev)
    ck = torch.load(Path("logs/tbiom/temporal/branches") / a.branch / "best.pt", map_location=dev)
    model.trainable_blocks.load_state_dict(ck["trainable_blocks"]); model.head.load_state_dict(ck["head"]); model.eval()
    mean = torch.tensor(model.mean, device=dev).view(1, 3, 1, 1); std = torch.tensor(model.std, device=dev).view(1, 3, 1, 1)
    print(f"frozen reference temporal branch {a.branch} (epoch {ck['epoch']})", flush=True)
    for ds in a.datasets:
        seg = Segments(ds, model.preprocess)
        for spec in a.process:
            tag = {"none": "", "jpeg:40": "_jpeg40", "resize:0.6": "_rs06"}[spec]
            f = out / f"{ds}{tag}.csv"
            if f.exists():
                print("skip", f); continue
            loader = torch.utils.data.DataLoader(seg, batch_size=a.batch, num_workers=a.workers,
                                                 worker_init_fn=lambda _: torch.set_num_threads(1))
            rows, t0 = [], time.time()
            with torch.no_grad():
                for px, n, n_src, f0, f1, first in loader:
                    px = px.to(dev); B, Tn = px.shape[:2]
                    if spec != "none":
                        px = apply_processing(px.flatten(0, 1).float(), spec, model.mean, model.std).view(B, Tn, *px.shape[2:])
                    q = descriptors((px.flatten(0, 1).float() * std + mean).clamp(0, 1).view(B, Tn, *px.shape[2:]), first.to(dev))
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        a_ = model(px).double().cpu().numpy() + 1
                    for k, nn_ in enumerate(n.tolist()):
                        i, s, _ = seg.pairs[nn_]
                        rows.append((seg.items[i][2], int(seg.items[i][1]), "full" if s == N_SEG else f"s{s}",
                                     a_[k, 1] / a_[k].sum(), 2 / a_[k].sum(), int(n_src[k]), int(f0[k]), int(f1[k]),
                                     *[float(q[c][k]) for c in ("blur", "blockiness", "contrast", "motion", "luma")]))
            pd.DataFrame(rows, columns=["v", "y", "segment", "p", "u", "n_src", "t0", "t1", "blur", "blockiness",
                                        "contrast", "motion", "luma"]).to_csv(f, index=False)
            print(f"wrote {f} ({len(rows)} rows, {(time.time() - t0) / 60:.1f} min)", flush=True)
    (out / "DONE").write_text("ok\n")


if __name__ == "__main__":
    main()

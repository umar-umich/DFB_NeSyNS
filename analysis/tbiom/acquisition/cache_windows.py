#!/usr/bin/env python3
"""Experiment 7, Step 1a: cache the frozen reference detector's outputs on candidate clip windows.

    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 <timm python> analysis/tbiom/acquisition/cache_windows.py \
        --datasets VALmix VALmixV2 FFpp_val [--batch 48] [--workers 12]

The acquisition pilot asks which ADDITIONAL clip window to inspect under a budget. Step 1 of the brief is caching and
controls, before any PPO: every candidate window's detector output is precomputed so the controls, the supervised
one-step selector and any later policy all draw from the same pool at the same cost.

WHAT IS COMPUTED HERE. Only the TEMPORAL branch needs new inference: the Appearance and Off-manifold views are
frame-level and already cached per frame, so a window's spatial opinion is the mean of its frames' cached opinions
(done on CPU in the analysis step). The temporal branch is the FROZEN reference checkpoint
(`vjepa2_lora_B_seed42`, the B-augmented V-JEPA 2 LoRA branch); no backbone is trained here and no parameter moves.

WINDOWS. A video's stored crops are split into W contiguous segments; each segment is sampled to the branch's 16
frames exactly as training does (`np.linspace`, rounded, in order). Window `full` is the reference observation the
deployed system already uses -- the whole span, uniformly sampled -- and is recomputed here so every row in the pool
comes from one code path. Videos with fewer frames than a segment needs repeat frames through the same rule, which is
recorded per row (`n_src`) rather than hidden.

QUALITY MEASUREMENTS, computed from the same crops in the loader (they are part of the pilot's state, and must be
available at inference without labels): blur (variance of the Laplacian), motion (mean absolute difference between
consecutive sampled frames), blockiness (an 8x8 JPEG grid-edge ratio) and the crop's mean luminance and contrast.

Output: one CSV per dataset at logs/tbiom/acquisition/<branch>/<dataset>.csv with columns
v, y, window, p, u, n_src, blur, motion, blockiness, luma, contrast.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "temporal")); sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "xclip"))
import train_branch as TB  # noqa: E402
import h5py  # noqa: E402
from PIL import Image  # noqa: E402

torch.set_num_threads(4)
OUT = Path("logs/tbiom/acquisition")
W_DEFAULT = 4


def quality(frames: list[np.ndarray]) -> dict:
    """Cheap, label-free descriptors of what an inspector would see in this window."""
    g = [f.astype(np.float32).mean(2) for f in frames]
    lap = []
    for x in g:
        d = (x[:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, :-2] + x[1:-1, 2:] - 4 * x[1:-1, 1:-1])
        lap.append(float(d.var()))
    mot = float(np.mean([np.abs(g[i + 1] - g[i]).mean() for i in range(len(g) - 1)])) if len(g) > 1 else 0.0
    x = g[len(g) // 2]
    # blockiness: mean |step| across the 8-px JPEG grid lines against all other columns
    cols = np.abs(np.diff(x, axis=1)); on = cols[:, 7::8].mean(); off = cols.mean() + 1e-6
    return {"blur": float(np.mean(lap)), "motion": mot, "blockiness": float(on / off),
            "luma": float(np.mean([y.mean() for y in g])), "contrast": float(np.mean([y.std() for y in g]))}


class Windows(torch.utils.data.Dataset):
    """(preprocessed clip, window id, item index) for every (video, window) pair."""

    def __init__(self, ds, proc, T, n_win):
        self.h5rel, self.items = TB.videos_of(ds)
        self.proc, self.T, self.n_win, self.h = proc, T, n_win, None
        self.pairs = [(i, w) for i in range(len(self.items)) for w in range(n_win + 1)]   # +1 = the 'full' window

    def __len__(self):
        return len(self.pairs)

    def keys_for(self, ks, w):
        """Window w's 16 frames, plus how many stored crops the window actually had.

        A video with fewer stored crops than windows leaves some segments empty; those degenerate to the full span and
        are marked n_src = 0, so the analysis can exclude or label them instead of silently scoring a duplicate.
        """
        if w == self.n_win:
            seg = ks
        else:
            seg = [ks[j] for j in np.array_split(np.arange(len(ks)), self.n_win)[w]]
        n_src = len(seg)
        if n_src == 0:
            seg = ks
        idx = np.linspace(0, len(seg) - 1, self.T).round().astype(int)
        return [seg[j] for j in idx], n_src

    def __getitem__(self, n):
        if self.h is None:
            self.h = h5py.File(TB.EXT / self.h5rel, "r")
        i, w = self.pairs[n]
        ks, y, _v = self.items[i]
        sel, n_src = self.keys_for(ks, w)
        fr = [np.asarray(self.h[k]) for k in sel]
        fr = [f if f.shape[:2] == (256, 256) else np.asarray(Image.fromarray(f).resize((256, 256), Image.BILINEAR))
              for f in fr]
        q = quality(fr)
        return self.proc(fr), y, n, n_src, np.array([q["blur"], q["motion"], q["blockiness"], q["luma"], q["contrast"]],
                                                    dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", default="vjepa2_lora_B_seed42")
    ap.add_argument("--datasets", nargs="+", default=["VALmix", "VALmixV2", "FFpp_val"])
    ap.add_argument("--windows", type=int, default=W_DEFAULT)
    ap.add_argument("--batch", type=int, default=48); ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    out = OUT / a.branch; out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    model = TB.Branch("vjepa2", lora=True).to(dev)
    ck = torch.load(TB.Path("logs/tbiom/temporal/branches") / a.branch / "best.pt", map_location=dev)
    model.trainable_blocks.load_state_dict(ck["trainable_blocks"]); model.head.load_state_dict(ck["head"])
    model.eval()
    print(f"frozen reference temporal branch {a.branch} (epoch {ck['epoch']}), {a.windows} candidate windows + full",
          flush=True)
    for ds in a.datasets:
        f = out / f"{ds}.csv"
        if f.exists():
            print(f"skip {ds} (exists)"); continue
        w_ds = Windows(ds, model.preprocess, model.T, a.windows)
        loader = torch.utils.data.DataLoader(w_ds, batch_size=a.batch, num_workers=a.workers,
                                             worker_init_fn=lambda _: torch.set_num_threads(1))
        rows, t0 = [], time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for px, y, n, n_src, q in loader:
                alpha = model(px.to(dev)).double().cpu().numpy() + 1
                for k, idx in enumerate(n.tolist()):
                    i, w = w_ds.pairs[idx]
                    rows.append((w_ds.items[i][2], int(y[k]), "full" if w == a.windows else f"w{w}",
                                 alpha[k, 1] / alpha[k].sum(), 2 / alpha[k].sum(), int(n_src[k]), *q[k].tolist()))
                if len(rows) % (a.batch * 40) < a.batch:
                    print(f"  {ds}: {len(rows)}/{len(w_ds)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        pd.DataFrame(rows, columns=["v", "y", "window", "p", "u", "n_src", "blur", "motion", "blockiness",
                                    "luma", "contrast"]).to_csv(f, index=False)
        print(f"wrote {f} ({len(rows)} rows, {(time.time() - t0) / 60:.1f} min)", flush=True)
    (out / "DONE").write_text("\n".join(a.datasets) + "\n")


if __name__ == "__main__":
    main()

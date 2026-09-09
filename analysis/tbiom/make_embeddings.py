#!/usr/bin/env python3
"""t-SNE embeddings for every arm and dataset — coordinates SAVED, then plotted.

    python analysis/tbiom/make_embeddings.py --arm C --dataset CDFv2
    python analysis/tbiom/make_embeddings.py --all

WHY COORDINATES ARE SAVED SEPARATELY. t-SNE is stochastic and slow. Writing the fitted
coordinates to `logs/tbiom/tsne/emb_<arm>_<dataset>_<space>_<level>.npz` (and a .csv beside it)
means every later figure -- recolouring, re-cropping, a different panel arrangement, a reviewer
asking for one dataset at a different size -- costs no recomputation and lands on the SAME
embedding. A figure regenerated from a fresh t-SNE would not be the same picture.

SPACES EMBEDDED, and what each one honestly shows:

    semantic / artifact / fsvfm   each branch's own feature space. Shows how well a single view
                                  separates real from fake ON ITS OWN.
    concat                        the three feature blocks concatenated. This is the
                                  representation-quality view. It is NOT the architecture: the
                                  model fuses OPINIONS, not concatenated features, so this panel
                                  is labelled as a representation comparison and never as "our
                                  fusion".

The opinion space is handled by make_opinion_plot.py, because it is 3-d and readable directly --
running t-SNE on three numbers would destroy the very structure worth seeing.

LEVELS. `frame` uses every exported frame; `video` averages frames within a video using the
project's single video-identity rule, matching how every AUROC in the report is computed. Both
come from one GPU pass.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_id import video_id  # noqa: E402

T = Path("logs/tbiom/tsne")
ARMS = ["A", "B", "C", "D", "E"]
DATASETS = ["CDFv2", "CDFv3", "DFD", "DFDC", "DFDCP", "DFEval24", "VALmix", "FFpp_val"]
SPACES = ["semantic", "artifact", "fsvfm", "concat"]
MAX_POINTS = 4000        # t-SNE is O(n log n) but plots stop being readable long before that


def load(arm: str, ds: str, level: str):
    f = T / f"f_{arm}_{ds}.npz"
    if not f.is_file():
        return None
    d = np.load(f, allow_pickle=True)
    if "key" not in d.files:
        return None
    blocks = {k: d[k] for k in ("semantic", "artifact", "fsvfm") if k in d.files}
    y = d["label"].astype(int)
    if level == "video":
        vid = video_id(pd.Series([str(k) for k in d["key"]]))
        g = pd.DataFrame({"v": vid.to_numpy(), "y": y})
        idx = g.groupby("v").indices
        keys = sorted(idx)
        blocks = {k: np.stack([v[idx[q]].mean(0) for q in keys]) for k, v in blocks.items()}
        y = np.array([int(g["y"].to_numpy()[idx[q]].max()) for q in keys])
    return blocks, y


def embed(X: np.ndarray, seed: int = 42) -> np.ndarray:
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0))
    # PCA down first when the block is wide: t-SNE on 1152 raw dims is dominated by noise
    # directions and is far slower for no gain in structure.
    if X.shape[1] > 50:
        from sklearn.decomposition import PCA
        X = PCA(n_components=50, random_state=seed).fit_transform(X)
    perp = float(min(30, max(5, (len(X) - 1) / 3)))
    return TSNE(n_components=2, perplexity=perp, init="pca", learning_rate="auto",
                random_state=seed, max_iter=1000).fit_transform(X)


def run_one(arm: str, ds: str, level: str, force: bool = False) -> int:
    got = load(arm, ds, level)
    if got is None:
        return 0
    blocks, y = got
    if len(y) < 20 or len(np.unique(y)) < 2:
        return 0
    rng = np.random.default_rng(42)
    sel = (rng.choice(len(y), MAX_POINTS, replace=False) if len(y) > MAX_POINTS
           else np.arange(len(y)))
    sel.sort()
    n = 0
    for space in SPACES:
        out = T / f"emb_{arm}_{ds}_{space}_{level}.npz"
        if out.is_file() and not force:
            continue
        if space == "concat":
            if not all(k in blocks for k in ("semantic", "artifact", "fsvfm")):
                continue
            X = np.concatenate([blocks[k] for k in ("semantic", "artifact", "fsvfm")], 1)
        else:
            if space not in blocks:
                continue
            X = blocks[space]
        Z = embed(X[sel])
        np.savez_compressed(out, xy=Z.astype(np.float32), label=y[sel].astype(np.int8))
        pd.DataFrame({"x": Z[:, 0], "y": Z[:, 1], "label": y[sel]}).to_csv(
            out.with_suffix(".csv"), index=False)
        n += 1
        print(f"  {arm}/{ds}/{space}/{level}: {len(sel)} points -> {out.name}")
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--level", default=None, choices=["frame", "video"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    arms = [args.arm] if args.arm else ARMS
    dss = [args.dataset] if args.dataset else DATASETS
    levels = [args.level] if args.level else ["video", "frame"]
    total = 0
    for a in arms:
        for d in dss:
            for lv in levels:
                total += run_one(a, d, lv, args.force)
    print(f"\nwrote {total} embeddings to {T}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

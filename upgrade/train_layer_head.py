#!/usr/bin/env python3
"""Part 3d: train the screen head on one layer READOUT of a backbone.

    <discern_ext python> upgrade/train_layer_head.py --backbone pe_core_l --readout last+mid --seed 0

Readouts: `last`, `mid` (two-thirds depth), `half`, and concatenations. Each component is
L2-NORMALISED before concatenation, so a combined readout cannot be won by whichever block happens
to have the larger activation scale -- without that the comparison would be about norm, not content.

Head, loss, schedule and selection are identical to `train_head.py`, so a row here differs from the
Part 3b bar only by which depth it reads.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(ROOT / "upgrade"), str(ROOT / "branches/ijepa")]
import train_head as TH  # noqa: E402

LF = ROOT / "logs/upgrade/lfeats"
OUT = ROOT / "analysis/upgrade/layers"


def load_readout(backbone, dataset, readout, process=None):
    tag = f"{backbone}_{dataset}" + (f"_{process.replace(':','')}" if process else "")
    z = np.load(LF / f"{tag}.npz", allow_pickle=True)
    parts = []
    for comp in readout.split("+"):
        x = z[f"feat_{comp}"].astype(np.float32)
        x = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-6, None)   # L2 per component
        parts.append(x)
    x = np.concatenate(parts, 1) if len(parts) > 1 else parts[0]
    vid = np.array([str(v) for v in z["video"]])
    order, inv = np.unique(vid, return_inverse=True)
    first = {}
    for k, v in zip(z["key"], vid):
        first.setdefault(v, str(k))
    return {"x": x, "y": z["label"].astype(np.int64), "inv": inv, "order": order,
            "key": np.array([first[v] for v in order], dtype=object),
            "yv": np.array([int(z["label"][inv == i].max()) for i in range(len(order))])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True); ap.add_argument("--readout", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{a.backbone}_{a.readout.replace('+','-')}_s{a.seed}"
    if (OUT / f"{tag}.json").exists():
        print(f"{tag} exists"); return 0

    # reuse the Part 3b trainer wholesale by swapping its loader
    TH.load = lambda bb, ds: load_readout(bb, ds, a.readout)
    TH.OUT = OUT
    sys.argv = ["x", "--backbone", a.backbone, "--seed", str(a.seed)]
    rc = TH.main()

    src = OUT / f"view_{a.backbone}_s{a.seed}.json"
    if src.exists():
        d = json.load(open(src)); d["readout"] = a.readout
        # robustness half of the keep rule: score the SELECTED head on the processed caches
        import torch as T
        head = TH.Head(d["dim"]); head.load_state_dict(T.load(OUT / f"view_{a.backbone}_s{a.seed}.pt")["state"])
        head.eval()
        for proc in ("jpeg:30", "resize:0.35"):
            f = LF / f"{a.backbone}_FFpp_val_{proc.replace(':','')}.npz"
            if not f.exists():
                d[f"ffpp_val_{proc}"] = None; continue
            D = load_readout(a.backbone, "FFpp_val", a.readout, proc)
            p = TH.video_p(head, D, "cpu")
            import importlib; V31 = importlib.import_module("v31")
            d[f"ffpp_val_{proc}"] = float(V31.fast_auc(p[D["yv"] == 0], p[D["yv"] == 1]))
        json.dump(d, open(OUT / f"{tag}.json", "w"), indent=1)
        src.unlink(); (OUT / f"view_{a.backbone}_s{a.seed}.pt").rename(OUT / f"{tag}.pt")
        print(f"{tag}: dev v3.1 {d['dev_v31']:.4f}  FF++val {d['ffpp_val_auroc']:.4f}  "
              f"jpeg30 {d.get('ffpp_val_jpeg:30')}  rs035 {d.get('ffpp_val_resize:0.35')}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())

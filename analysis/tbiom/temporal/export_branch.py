#!/usr/bin/env python3
"""Export a trained temporal branch's opinions on named datasets (inference only, no training).

    CUDA_VISIBLE_DEVICES=1 <temporal_vjepa2 python> analysis/tbiom/temporal/export_branch.py \
        --branch vjepa2_lora_B_seed42 --datasets FaceShifter

Loads the run's development-selected `best.pt` and writes `opinions/<dataset>.csv` (v, y, p, u) with exactly the
procedure every other export used: one clip per video, 16 frames sampled uniformly in order over the stored crops,
the model's own processor. Existing files are left alone.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_branch as TB  # noqa: E402

BR = Path("/data/umar/Repos/DFB_NeSyNS/logs/tbiom/temporal/branches")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", default="vjepa2_lora_B_seed42"); ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--model", default="vjepa2"); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(4)
    run = BR / a.branch; dev = "cuda"
    model = TB.Branch(a.model, lora=True).to(dev)
    ck = torch.load(run / "best.pt", map_location=dev)
    model.trainable_blocks.load_state_dict(ck["trainable_blocks"]); model.head.load_state_dict(ck["head"])
    print(f"{a.branch}: development-selected epoch {ck['epoch']}", flush=True)
    for n in a.datasets:
        out = run / "opinions" / f"{n}.csv"
        if out.exists():
            print(f"skip {n} (exists)"); continue
        TB.opinions(model, n, dev, workers=a.workers).to_csv(out, index=False)
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()

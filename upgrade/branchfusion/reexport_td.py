#!/usr/bin/env python3
"""Re-export TD's opinions for named sets from a FINISHED run's best.pt.

    BF_SCOPE=nolipadv CUDA_VISIBLE_DEVICES=3 <temporal_vjepa2 python> \
        upgrade/branchfusion/reexport_td.py --sets CDFv3,KoDF

`train_branch.py` writes the per-set opinions inside its training flow, and its only re-export
flag, `--export-shuffled-only`, writes the SHUFFLED control instead -- so widening the scope had
no entry point that did not also retrain from scratch. This loads `best.pt` and calls the same
`opinions()` the trainer calls, so the numbers come from the identical code path; nothing about
the branch is reimplemented here.

The branch is unchanged. What changes is the key list `opinions()` iterates, which is filtered
by `extract_xclip.OUT_OF_SCOPE` and therefore now follows `BF_SCOPE`.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(ROOT / "analysis/tbiom/temporal"), str(ROOT / "analysis/tbiom/xclip"),
                str(ROOT / "analysis/tbiom")]
import train_branch as TB                                          # noqa: E402
import extract_xclip as X                                          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="vjepa2_lora_B_seed42")
    ap.add_argument("--model", default="vjepa2")
    ap.add_argument("--sets", required=True, help="comma list, e.g. CDFv3,KoDF")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    out_dir = ROOT / "logs/tbiom/temporal/branches" / a.run
    ck_path = out_dir / "best.pt"
    if not ck_path.exists():
        raise FileNotFoundError(ck_path)
    print(f"scope BF_SCOPE={os.environ.get('BF_SCOPE', 'v31')} "
          f"-> {len(X.OUT_OF_SCOPE)} families excluded", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = TB.Branch(a.model, lora=True).to(dev)
    ck = torch.load(ck_path, map_location=dev)
    # `best.pt` holds only {epoch, trainable_blocks, head} -- the frozen backbone is NOT in it,
    # by design, so `load_state_dict` on the whole model reports every backbone tensor missing.
    # `TB.load_trainable` is what the trainer's own re-export path uses; reusing it means this
    # script cannot load the checkpoint differently than training did.
    print(f"loaded epoch {ck['epoch']} ({TB.load_trainable(model, ck)})", flush=True)
    model.eval()

    for ds in [s for s in a.sets.split(",") if s]:
        o = TB.opinions(model, ds, dev, workers=a.workers)
        f = out_dir / "opinions" / f"{ds}.csv"
        # The previous file is the v3.1 export and is KEPT: a scope change has to be reversible
        # without re-encoding, and the published number must stay reproducible.
        if f.exists() and not (f.parent / f"{ds}.v31.csv").exists():
            f.rename(f.parent / f"{ds}.v31.csv")
            print(f"  kept the v3.1 export as {ds}.v31.csv", flush=True)
        o.to_csv(f, index=False)
        print(f"  {ds}: {len(o)} videos -> {f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Opinions for an encoder-in-the-loop OM run, so it can enter the ledger like any other branch.

    <discern_ext python> upgrade/branchfusion/om_eil_export.py \
        --ckpt logs/upgrade/om/om_pe_core_l_LN_bvae_s0_b16.pt --cache omEIL_b16 --label Y16

`adapt_om.py` trains the adapters and the residual head together and saves both. `extract_adapted.py
--ckpt` then writes the ADAPTED features to an ordinary feature cache, which means the head can be
replayed over that cache instead of the encoder being re-run per dataset: the encoder is fixed once
training ends, so the two are numerically the same and the cache costs one pass instead of one per
read. Everything downstream -- the per-video reduction, the video sets, the ledger format -- is
`om_branch`'s, so a difference between an in-loop branch and a frozen one is the training and not
the bookkeeping.

The exported arm is the RESIDUAL, which is what "off-manifold" names. The fused arm is available
with --arm but carries the plain appearance opinion inside it, so a branch exported that way would
duplicate A rather than stand beside it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                               # noqa: E402

sys.path[:0] = [str(C.ROOT / "analysis/tbiom/temporal")]
import residual_head as RH                                       # noqa: E402
import om_branch as OB                                           # noqa: E402
import branches as B                                             # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint written by adapt_om.py")
    ap.add_argument("--cache", required=True, help="feature-cache prefix written by extract_adapted --cache-tag")
    ap.add_argument("--label", required=True, help="branch label to register, e.g. Y16")
    ap.add_argument("--arm", default="r", choices=["r", "t", "fused"])
    ap.add_argument("--register", action="store_true", help="add the branch to the dynamic registry")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(C.ROOT / a.ckpt if not Path(a.ckpt).is_absolute() else a.ckpt, map_location=dev)
    model = RH.TemporalResidualHead(ck["in_dim"], ck["operator"]).to(dev)
    model.load_state_dict(ck["head"]); model.eval()
    print(f"{a.label}: head from epoch {ck['epoch']}, operator {ck['operator']}, "
          f"readout {ck['readout']}, dim {ck['in_dim']}, arm '{a.arm}'", flush=True)

    out = C.BRANCH_OUT / f"om_eil_{a.label}"
    out.mkdir(parents=True, exist_ok=True)
    done, missing = {}, []
    with torch.no_grad():
        for ds in list(C.DEV_SETS) + list(C.TEST_SETS):
            try:
                D = C.load_frames(a.cache, ds)
            except FileNotFoundError:
                missing.append(ds); print(f"  {ds}: no {a.cache} cache -- SKIPPED", flush=True)
                continue
            p, u = OB.video_pu(model, D, dev, a.arm)
            pd.DataFrame({"v": [str(x) for x in D["order"]], "y": D["yv"], "p": p, "u": u}) \
                .to_csv(out / f"{ds}.csv", index=False)
            done[ds] = int(len(D["order"]))
            print(f"  {ds}: {len(D['order'])} videos", flush=True)

    rec = {"label": a.label, "ckpt": str(a.ckpt), "cache": a.cache, "arm": a.arm,
           "epoch": int(ck["epoch"]), "operator": ck["operator"], "readout": ck["readout"],
           "encoder_in_loop": True, "datasets": done, "missing": missing, "dir": str(out)}
    (C.ROOT / "analysis/branchfusion").mkdir(parents=True, exist_ok=True)
    (C.ROOT / f"analysis/branchfusion/om_eil_{a.label}_export.json").write_text(json.dumps(rec, indent=1))
    if a.register and not missing:
        B.register(a.label, out, note=f"OM encoder-in-loop, {ck['readout']}, arm {a.arm}")
        print(f"  registered as {a.label}", flush=True)
    elif a.register:
        print(f"  NOT registered: {len(missing)} dataset(s) missing -- a branch with partial "
              f"coverage would silently change every macro mean it enters", flush=True)
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

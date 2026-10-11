#!/usr/bin/env python3
"""Register a finished PE-Core off-manifold run as a branch, so the fusion screen can see it.

    <discern_ext python> upgrade/branchfusion/om_pecore_export.py \
        --run pecore_l14_lora_B_ombvae_b16_cat_seed42 --label P16om --register

`train_om_pecore` already writes one opinion CSV per dataset, in the (v, y, p, u) format every
other branch uses, so there is nothing to recompute here: this copies the thirteen development and
test files into the ledger, records where they came from, and registers the label. The equivalent
for the spatial trunk is `om_eil_export.py`, which has to replay a head over a feature cache
because `adapt_om` saves a head rather than per-dataset opinions.

WHY THIS EXISTS AT ALL, given both constructions FAIL gate 1. Gate 1 asks whether the residual
helped the network it sits in; the deployment question is whether the branch helps the ENSEMBLE,
and those come apart. The deployed off-manifold view O scores 0.8252 on development -- BELOW the
0.8409 and 0.8497 these two reach -- and earns its row by being decorrelated, not by being strong.
These are the first PE-trunk branches to leave A's cluster (phi +0.415 and +0.500 against A, where
Y16 sat at +0.678), so the screen is the measurement that settles it, and it is CSV-level.

Coverage is checked before anything is registered: a branch present for some datasets and absent
for others would silently change every macro mean it enters.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                               # noqa: E402
import branches as B                                             # noqa: E402

RUNS = C.ROOT / "logs/branchfusion/om_pecore"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="directory name under logs/branchfusion/om_pecore")
    ap.add_argument("--label", required=True, help="branch label to register, e.g. P16om")
    ap.add_argument("--register", action="store_true")
    a = ap.parse_args()

    src = RUNS / a.run / "opinions"
    rec_f = RUNS / a.run / "train_record.json"
    if not src.is_dir() or not rec_f.exists():
        print(f"no finished run at {RUNS / a.run}")
        return 1
    rec = json.load(open(rec_f))
    print(f"{a.label}: {a.run}, best epoch {rec['best_epoch']}, dev v3.1 {rec['best_dev_v31']:.4f}",
          flush=True)

    out = C.BRANCH_OUT / a.label
    out.mkdir(parents=True, exist_ok=True)
    done, missing = {}, []
    for ds in list(C.DEV_SETS) + list(C.TEST_SETS):
        f = src / f"{ds}.csv"
        if not f.exists():
            missing.append(ds); print(f"  {ds}: MISSING", flush=True); continue
        d = pd.read_csv(f)
        bad = int((~d.p.between(0, 1)).sum())
        if bad:                                   # the pre-softplus exports put p outside [0, 1]
            print(f"  {ds}: {bad} opinions outside [0,1] -- refusing", flush=True)
            return 1
        shutil.copyfile(f, out / f"{ds}.csv")
        done[ds] = len(d)
        print(f"  {ds}: {len(d)} videos", flush=True)

    prov = {"label": a.label, "run": a.run, "best_epoch": rec["best_epoch"],
            "dev_v31": rec["best_dev_v31"], "budget": rec.get("budget"),
            "encoder_in_loop": True, "datasets": done, "missing": missing, "dir": str(out)}
    (C.ROOT / "analysis/branchfusion").mkdir(parents=True, exist_ok=True)
    (C.ROOT / f"analysis/branchfusion/{a.label}_export.json").write_text(json.dumps(prov, indent=1))
    if a.register and not missing:
        B.register(a.label, out, note=f"PE-Core b16 in-loop, {a.run}")
        print(f"  registered as {a.label}", flush=True)
    elif a.register:
        print(f"  NOT registered: {len(missing)} dataset(s) missing", flush=True)
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Task 1 gate: the differentiable fusion must reproduce the deployed one to within 1e-6.

    OMP_NUM_THREADS=4 <discern_ext python> training/joint_ibdc/verify_fusion.py

The deployed path is `analysis/tbiom/pcr6p/ops_pcr6p.py` on the deployed configuration's own
development exports. The torch path is `fusion_torch.py`, written independently. Agreement is checked
on the fused mass vector, the projected probability and the fused vacuity, over 1,000 development
videos, at every fitted per-fold trust triple.

The deployed configuration is `42`: warm+gate epoch 19 spatial views with vjepa2_lora_B_seed42
(`tbiom/reports/PCR6P_REFERENCE_MANIFEST.md` section 3). The armB tags are a different TRAINING
RECIPE, not a seed of the deployed one, so they are not the reference here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(ROOT / "analysis/tbiom/pcr6p"), str(ROOT / "analysis/tbiom/grouped"),
                str(ROOT / "analysis/tbiom"), str(ROOT / "analysis/tbiom/temporal"),
                str(Path(__file__).resolve().parent)]

import ops_pcr6p as P            # noqa: E402  deployed numpy fusion
import confirm as C              # noqa: E402  dev_table
import fusion_torch as T         # noqa: E402  the port under test

SPATIAL, BRANCH, TAG = "warmgate_e19", "vjepa2_lora_B_seed42", "42"
N_CHECK, TOL = 1000, 1e-6
OUT = ROOT / "analysis/joint_ibdc"
VIEWS = ("T", "APP", "OM")


def main():
    dev = C.dev_table(SPATIAL, BRANCH)
    print(f"development table: {len(dev)} videos, columns present for {VIEWS}")
    t = dev.iloc[:N_CHECK]

    w_folds = json.load(open(ROOT / "tbiom/reports/EXP_PCR6P_DEVELOPMENT.json"))[TAG]
    triples = [s["w"]["PCR6P-R"] for s in w_folds]
    triples.append([1.0, 1.0, 1.0])                       # undiscounted, as an extra case
    print(f"checking {len(triples)} trust triples on {len(t)} videos, tolerance {TOL}")

    worst = {"mass": 0.0, "p": 0.0, "u": 0.0}
    rows = []
    for w in triples:
        mn = P.masses_of(t, VIEWS, w)                     # (n, S, 3) deployed, numpy
        fn = P.fuse(mn, plus=True)
        pn, un = P.p_of(fn), fn[:, 2]

        mt = torch.tensor(mn, dtype=torch.float64)        # same discounted inputs, torch fusion
        ft = T.fuse(mt, plus=True)
        pt, ut = T.p_of(ft), T.vacuity_of(ft)

        dm = float(np.abs(ft.numpy() - fn).max())
        dp = float(np.abs(pt.numpy() - pn).max())
        du = float(np.abs(ut.numpy() - un).max())
        worst = {"mass": max(worst["mass"], dm), "p": max(worst["p"], dp), "u": max(worst["u"], du)}
        rows.append({"w": list(w), "max_abs_diff_mass": dm, "max_abs_diff_p": dp,
                     "max_abs_diff_u": du})
        print(f"  w={str(w):22s} mass {dm:.3e}  p {dp:.3e}  u {du:.3e}")

    # the discount itself, computed independently in torch from (p, u) rather than handed the masses
    w = triples[0]
    o = np.stack([P.G.opinion_from_pu(t[f"p_{v}"].to_numpy(), t[f"u_{v}"].to_numpy())[0]
                  for v in VIEWS], 1)
    ot = torch.tensor(o, dtype=torch.float64)
    mt2 = torch.stack([T.discount(ot[:, i], float(w[i])) for i in range(len(VIEWS))], dim=1)
    dd = float(np.abs(mt2.numpy() - P.masses_of(t, VIEWS, w)).max())
    print(f"  discount computed independently in torch: max abs diff {dd:.3e}")
    worst["discount"] = dd

    ok = all(v <= TOL for v in worst.values())
    res = {"gate": "Task 1 fusion equivalence", "verdict": "PASS" if ok else "FAIL",
           "tolerance": TOL, "n_videos": int(len(t)), "system": TAG,
           "spatial": SPATIAL, "branch": BRANCH, "worst": worst, "per_triple": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "task1_fusion_equivalence.json", "w"), indent=1)
    print(f"\nworst over all triples: {worst}")
    print(f"Task 1 fusion equivalence: {res['verdict']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

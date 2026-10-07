#!/usr/bin/env python3
"""Part 2a: is a weak test set a RANKING problem or a THRESHOLD-TRANSFER problem?

    <discern_ext python> upgrade/part2a_threshold.py

For every test set and the deployed system, report AUROC, FPR and TPR at the FF++ validation
threshold, FPR at the set's OWN EER threshold, and ECE. A set where AUROC is healthy but FPR at the
FF++ threshold is high is mis-CALIBRATED for that domain, not mis-ranked: the ordering is fine and
the operating point travelled badly. The brief is explicit that this is analysis only and decides
nothing, so no rule reads it.

The deployed opinions are replayed from the cached per-view exports through the deployed PCR6+-R
fusion, exactly as the headline table scores them.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(ROOT / "analysis/tbiom/pcr6p"), str(ROOT / "analysis/tbiom"),
                str(ROOT / "analysis/tbiom/temporal"), str(ROOT / "analysis/tbiom/grouped")]
import arms as AR               # noqa: E402
import ops_pcr6p as P           # noqa: E402
import dsr_system as DS         # noqa: E402
import exp1_fusion_audit as A   # noqa: E402
from health import auroc, eer   # noqa: E402

OUT = ROOT / "analysis/upgrade"
ARM = "PCR6P-R"


def ece(y, p, bins=15):
    """Expected calibration error, equal-width bins on the fused probability."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e, n = 0.0, len(y)
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        e += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return float(e)


def main() -> int:
    # DS.load returns (dev, processed-dev, tests). Taking [1] gave the PROCESSED DEVELOPMENT
    # dict, so the table listed rs06 and jpeg40 -- development cells -- as if they were test sets.
    dev, _dproc, tests = DS.load("vjepa2_lora_B_seed42")
    folds = [Fd for _k, Fd, _S in DS.splits(dev)]
    ws = [AR.fit_w(Fd, ARM) for Fd in folds]

    # The deployed threshold is the FF++ validation EER of the fused score, averaged over folds.
    ff = tests.get("FFpp_val", dev[dev.source == "FFpp_val"] if "source" in dev else None)
    taus = []
    for w in ws:
        s = P.p_of(AR.fuse(dev, ARM, w))
        m = (dev.source == "FFpp_val").to_numpy() if "source" in dev else np.ones(len(dev), bool)
        if m.any():
            taus.append(eer(dev.y.to_numpy()[m], s[m])[1])
    tau = float(np.mean(taus)) if taus else 0.5

    rows = {}
    for name, t in tests.items():
        y = t.y.to_numpy()
        if len(np.unique(y)) < 2:
            continue
        ps = np.mean([P.p_of(AR.fuse(t, ARM, w)) for w in ws], axis=0)
        own = eer(y, ps)[1]
        rows[name] = {
            "videos": int(len(y)), "auroc": auroc(y, ps),
            "fpr_at_ffpp_tau": float((ps[y == 0] >= tau).mean()),
            "tpr_at_ffpp_tau": float((ps[y == 1] >= tau).mean()),
            "fpr_at_own_eer": float((ps[y == 0] >= own).mean()),
            "own_eer_tau": float(own), "ece": ece(y, ps),
        }
        rows[name]["threshold_transfer_gap"] = rows[name]["fpr_at_ffpp_tau"] - rows[name]["fpr_at_own_eer"]

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"arm": ARM, "ffpp_tau": tau, "sets": rows,
               "note": ("Analysis only; no rule reads this. A high `fpr_at_ffpp_tau` beside a healthy "
                        "`auroc` and a small `fpr_at_own_eer` is a threshold-transfer problem, not a "
                        "ranking one.")},
              open(OUT / "part2a_threshold.json", "w"), indent=1)

    print(f"deployed FF++ tau = {tau:.4f}\n")
    print(f"{'set':12s} {'n':>6s} {'AUROC':>7s} {'FPR@tau':>8s} {'TPR@tau':>8s} {'FPR@own':>8s} {'gap':>7s} {'ECE':>6s}")
    for n, r in sorted(rows.items(), key=lambda kv: -kv[1]["threshold_transfer_gap"]):
        print(f"{n:12s} {r['videos']:6d} {r['auroc']:7.4f} {r['fpr_at_ffpp_tau']:8.4f} "
              f"{r['tpr_at_ffpp_tau']:8.4f} {r['fpr_at_own_eer']:8.4f} {r['threshold_transfer_gap']:+7.4f} {r['ece']:6.4f}")
    print(f"\nwrote {OUT/'part2a_threshold.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

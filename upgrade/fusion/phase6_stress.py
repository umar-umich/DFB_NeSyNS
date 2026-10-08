#!/usr/bin/env python3
"""Final evaluation: DeepfakeEval24 first, DFDC second, then the fixed panel.

    <discern_ext python> upgrade/fusion/phase6_stress.py

Runs after the Phase 2 gate decision and refuses to start without it. Every configuration is frozen
before this executes: no fitting, no threshold tuning, no retraining in response to what it shows.

Rows: B0-matched, B1-refit, L, S, F. The gate FAILED, so no conditional variant is included --
the brief allows at most one chosen on development data and there is none.

Both stress sets are PREVIOUSLY EXPOSED. These are targeted evaluations on known-difficult datasets,
not untouched confirmation, and the report says so.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(ROOT / "upgrade"), str(ROOT / "upgrade/fusion"),
                str(ROOT / "analysis/tbiom/pcr6p"), str(ROOT / "analysis/tbiom/grouped"),
                str(ROOT / "analysis/tbiom"), str(ROOT / "analysis/tbiom/temporal")]
import arms as AR               # noqa: E402
import ops_pcr6p as P           # noqa: E402
import ops as G                 # noqa: E402
import dsr_system as DS         # noqa: E402
from health import auroc, eer   # noqa: E402
import final_eval as FE         # noqa: E402
import phase2 as P2             # noqa: E402

ARM = "PCR6P-R"
VIEWS = ("T", "APP", "OM")
PANEL = ["DFEval24", "DFDC", "CDFv2", "CDFv3", "DFD", "DFDCP", "FFIW", "KoDF", "IDForge", "DFDM"]


def main() -> int:
    run = (ROOT / "outputs/fusion_experiments/CURRENT_RUN").read_text().strip()
    out = ROOT / "outputs/fusion_experiments" / run
    gate = json.load(open(out / "gate_decision.json"))
    b1 = json.load(open(out / "phase1_b0_b1.json"))
    man = json.load(open(out / "data_access_manifest.json"))

    lock = {"run_id": run, "gate_passed": gate["gate_passed"],
            "rows": ["B0-matched", "B1-refit", "L", "S", "F"],
            "conditional_variant": None,
            "reason": "the Phase 2 gate failed, so no conditional variant was selected",
            "gamma": b1["B1_best_gamma"], "panel": PANEL,
            "fitting_excluded": man["reserved_from_fitting"],
            "frozen_before_evaluation": True}
    json.dump(lock, open(out / "stress_test_lock.json", "w"), indent=1)

    dev_t, _dp, tests = DS.load("vjepa2_lora_B_seed42")
    op_dev = FE.opinions("pe_core_lN", 0, "cuda", ("VALmix", "VALmixV2", "FFpp_val"))
    dsw = FE.swap_app(dev_t, op_dev, "development")
    keep = ~dsw["source"].astype(str).isin(man["reserved_from_fitting"]).to_numpy()
    fitpool = dsw[keep].reset_index(drop=True)
    ws = [AR.fit_w(Fd, ARM) for _k, Fd, _S in DS.splits(fitpool)]
    gamma = b1["B1_best_gamma"]

    # threshold from development out-of-fold predictions, frozen before any test set is touched
    pdev = np.mean([P.p_of(AR.fuse(fitpool, ARM, w)) for w in ws], 0)
    thr = float(eer(fitpool.y.to_numpy(), pdev)[1])
    print(f"frozen development threshold {thr:.4f}; gamma {gamma}", flush=True)

    op_test = FE.opinions("pe_core_lN", 0, "cuda", PANEL)
    rows = {}
    for n in PANEL:
        if n not in tests:
            continue
        try:
            t = FE.swap_app(tests[n], op_test, n)
        except KeyError as e:
            print(f"  {n}: skipped -- {e}"); continue
        y = t.y.to_numpy()
        s_b0 = np.mean([P.p_of(AR.fuse(t, ARM, w)) for w in ws], 0)
        s_b1 = np.mean([P.p_of(P2.__dict__.get("_dg", lambda *a: None)) if False else
                        P.p_of(_discount(t, w, gamma)) for w in ws], 0)
        cell = {}
        for name, s in (("B0-matched", s_b0), ("B1-refit", s_b1)):
            cell[name] = {"auroc": auroc(y, s),
                          "fpr_genuine_at_dev_threshold": float((s[y == 0] >= thr).mean()),
                          "nll": float(-np.mean(np.log(np.clip(np.where(y == 1, s, 1 - s), 1e-9, 1))))}
        rows[n] = {"videos": int(len(y)), **cell}
        print(f"  {n:10s} B0 {cell['B0-matched']['auroc']:.4f}  B1 {cell['B1-refit']['auroc']:.4f}  "
              f"FPR {cell['B1-refit']['fpr_genuine_at_dev_threshold']:.4f}", flush=True)

    res = {"run_id": run, "threshold": thr, "gamma": gamma, "panel": rows,
           "note": ("DeepfakeEval24 and DFDC are PREVIOUSLY EXPOSED. This is targeted evaluation on "
                    "known-difficult datasets, not untouched confirmation. L, S and F are reported "
                    "on development out-of-fold only: their heads require per-view FEATURES, and the "
                    "off-manifold cache does not cover DFDC, CDFv2, CDFv3 or KoDF, so a panel row "
                    "for them would be computed on a different set of datasets than B0/B1.")}
    json.dump(res, open(out / "stress_results.json", "w"), indent=1)
    print(f"\nwrote {out}/stress_results.json")
    return 0


def _discount(t, w, gamma):
    ms = []
    for i, v in enumerate(VIEWS):
        o, _ = G.opinion_from_pu(t[f"p_{v}"].to_numpy(), t[f"u_{v}"].to_numpy())
        ms.append(G.discount(o, float(gamma * w[i])))
    return P.fuse(np.stack(ms, 1), True)


if __name__ == "__main__":
    sys.exit(main())

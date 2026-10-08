#!/usr/bin/env python3
"""Do cumulative-R or cautious-R beat PCR6+-R on the CURRENT configuration?

    <discern_ext python> upgrade/fusion/operator_sweep.py

The 41-generator DF40 study ranked operators on the DEPLOYED CLIP appearance view. The appearance
view has since changed to PE-Core-L with LayerNorm adaptation, so that ranking is not automatically
inherited -- a rule that handled a quiet appearance view well need not handle a near-saturated one
the same way. This re-runs the comparison on the current views, under the same nested source-grouped
protocol, with the reliability triple refitted inside each training partition for EVERY operator so
no rule is judged on another's weights.

averaging-R is EXCLUDED at UMAR's instruction. Its DF40 number (0.9604, the best in that table) is
already on record and is reported here beside the others so the exclusion is visible rather than
silently removing the strongest competitor.
"""
from __future__ import annotations

import itertools, json, sys
from pathlib import Path

import numpy as np

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(ROOT / "upgrade"), str(ROOT / "upgrade/fusion"),
                str(ROOT / "analysis/tbiom/pcr6p"), str(ROOT / "analysis/tbiom/grouped"),
                str(ROOT / "analysis/tbiom"), str(ROOT / "analysis/tbiom/temporal")]
import fusion_operators_v2 as FO   # noqa: E402
import ops as G                    # noqa: E402
import ops_pcr6p as P              # noqa: E402
import arms as AR                  # noqa: E402
import dsr_system as DS            # noqa: E402
from health import auroc           # noqa: E402
import final_eval as FE            # noqa: E402
import phase2 as P2                # noqa: E402

VIEWS = ("T", "APP", "OM")
GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
WGRID = [w for w in itertools.product(GRID, repeat=3) if max(w) == 1.0]
OPS = ["cumulative", "cautious", "Dempster", "Yager", "PCR6", "Murphy"]


def masses(t, w):
    return np.stack([G.discount(G.opinion_from_pu(t[f"p_{v}"].to_numpy(),
                                                  t[f"u_{v}"].to_numpy())[0], float(w[i]))
                     for i, v in enumerate(VIEWS)], 1)


def score_op(t, w, op):
    if op == "PCR6P":
        return P.p_of(P.fuse(masses(t, w), True))
    return G.p_of(FO.combine(masses(t, w), op))


def fit_w(Fd, op):
    best, bw = -1.0, (1.0, 1.0, 1.0)
    y = Fd.y.to_numpy()
    for w in WGRID:
        s = score_op(Fd, w, op)
        a = P2.macro_auroc(y, s, Fd["source"].astype(str).to_numpy())
        if np.isfinite(a) and a > best:
            best, bw = a, w
    return bw


def main() -> int:
    run = (ROOT / "outputs/fusion_experiments/CURRENT_RUN").read_text().strip()
    out = ROOT / "outputs/fusion_experiments" / run
    man = json.load(open(out / "data_access_manifest.json"))
    dev_t, _dp, _tests = DS.load("vjepa2_lora_B_seed42")
    op_ = FE.opinions("pe_core_lN", 0, "cuda", ("VALmix", "VALmixV2", "FFpp_val"))
    t = FE.swap_app(dev_t, op_, "development")
    t = t[~t["source"].astype(str).isin(man["reserved_from_fitting"]).to_numpy()].reset_index(drop=True)
    folds = list(DS.splits(t))
    print(f"pool {len(t)} videos, {len(folds)} outer folds\n", flush=True)

    res = {}
    for op in ["PCR6P"] + OPS:
        sc = []
        for _k, Fd, Sd in folds:
            w = fit_w(Fd, op)                      # every operator gets its OWN refitted weights
            s = score_op(Sd, w, op)
            sc.append(P2.macro_auroc(Sd.y.to_numpy(), s, Sd["source"].astype(str).to_numpy()))
        res[op + "-R"] = {"macro": float(np.mean(sc)), "per_fold": [float(x) for x in sc]}
        print(f"  {op+'-R':14s} {np.mean(sc):.4f}", flush=True)

    base = res["PCR6P-R"]["macro"]
    for k in res:
        res[k]["delta_vs_PCR6P_R"] = res[k]["macro"] - base
    res["_excluded"] = {"averaging-R": "excluded at UMAR's instruction; its DF40 figure is 0.9604, "
                                       "the best in the prior 41-generator table"}
    res["_note"] = ("The prior ranking was measured on the deployed CLIP appearance view. This is the "
                    "same comparison on the current PE-Core-L LayerNorm view, with weights refitted "
                    "per operator inside each training partition.")
    json.dump(res, open(out / "operator_sweep.json", "w"), indent=1)
    print(f"\nbaseline PCR6+-R {base:.4f}")
    for k, v in sorted(res.items(), key=lambda kv: -kv[1]["macro"] if isinstance(kv[1], dict) and "macro" in kv[1] else 0):
        if isinstance(v, dict) and "macro" in v:
            print(f"  {k:14s} {v['macro']:.4f}  {v['delta_vs_PCR6P_R']:+.4f}")
    print(f"wrote {out}/operator_sweep.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

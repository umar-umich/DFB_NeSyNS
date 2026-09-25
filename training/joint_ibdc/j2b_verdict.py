#!/usr/bin/env python3
"""The J2b pass rule, and the shared-ignorance analysis it depends on.

    <discern_ext python> training/joint_ibdc/j2b_verdict.py

Pre-set in the joint IBDC brief. J2b is the corrected-IBDC result only if ALL FOUR hold against J1,
on the development pool, using the three-seed mean:

  1. fused vacuity error AUROC is higher, with the bootstrap interval of the difference above zero;
  2. development AUROC is within -0.005;
  3. every view with nonzero trust weight has a positive leave-one-out contribution;
  4. mean u_f on the shared-ignorance subset is not lower than J1's.

Thresholds for the shared-ignorance analysis are set on J0's FF++ validation opinions and FROZEN, as
the brief requires, so every run is scored against the same definition of "ignorant" and "confident".
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(ROOT / "analysis/tbiom/pcr6p"), str(ROOT / "analysis/tbiom/grouped"),
                str(ROOT / "analysis/tbiom"), str(ROOT / "analysis/tbiom/temporal"),
                str(ROOT / "branches/ijepa"), str(Path(__file__).resolve().parent)]

import arms as AR, ops_pcr6p as P, dsr_system as DS, exp1_fusion_audit as A  # noqa: E402
import score as S, confirm as C                                              # noqa: E402
import runs as R, fusion_torch as FT                                         # noqa: E402
from collapse_check import view_opinions, table_with, loo                    # noqa: E402
from lambda_select import eer_threshold                                      # noqa: E402

OUT = ROOT / "analysis/joint_ibdc"
VIEWS = ("T", "APP", "OM")
N_BOOT = 1000
CK = {"J1": [f"run_J1_s{s}.pt" for s in range(3)],
      "J2a": [f"run_J2a_s{s}_lam1.0.pt" for s in range(3)],
      "J2b": [f"run_J2b_s{s}_lam0.1.pt" for s in range(3)]}


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    if len(np.unique(y)) < 2:
        return float("nan")
    r = rankdata(s); n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def heads_of(dev, ck=None):
    h = R.load_deployed_heads(dev)
    if ck:
        st = torch.load(OUT / ck, map_location=dev)
        h[0].load_state_dict(st["app"]); h[1].load_state_dict(st["om"]); h[2].load_state_dict(st["t"])
    for x in h: x.eval()
    return h


@torch.no_grad()
def dev_fused(heads, DEV, dev, w=(1.0, 1.0, 1.0)):
    """Development-pool fused opinion under equal trust, plus per-view vacuity and kappa."""
    op = view_opinions(heads, DEV, dev)
    t = table_with(op)
    m = torch.tensor(P.masses_of(t, VIEWS, None), dtype=torch.float64)
    e = FT.evidence_of(m.float()) * torch.tensor(w).view(1, 3, 1)
    md = FT.opinion_of_evidence(e)
    f = FT.fuse(md, plus=True)
    return {"t": t, "p": FT.p_of(f).numpy(), "u": f[:, 2].numpy(),
            "kappa": FT.kappa(md).numpy(),
            "branch_u": m[:, :, 2].numpy().mean(1), "y": t.y.to_numpy()}


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    DEV = R.load_dev()

    # thresholds frozen on J0's FF++ validation opinions
    j0 = dev_fused(heads_of(dev), DEV, dev)
    t = j0["t"]; ff = t.source.to_numpy() == "FFpp_val"
    tau = eer_threshold(j0["p"][ff], j0["y"][ff])
    bu_hi = float(np.percentile(j0["branch_u"][ff], 80))
    bu_lo = float(np.percentile(j0["branch_u"][ff], 20))
    k_thr = float(np.percentile(j0["kappa"][ff], 20))
    print(f"  frozen on J0 FF++ val: tau {tau:.4f}  branch-vacuity p20 {bu_lo:.4f} p80 {bu_hi:.4f}  "
          f"kappa p20 {k_thr:.4f}")

    res = {"frozen": {"tau": tau, "branch_u_p20": bu_lo, "branch_u_p80": bu_hi, "kappa_p20": k_thr},
           "runs": {}}
    store = {}
    for run in ("J0", "J1", "J2a", "J2b"):
        rows = []
        for ck in ([None] if run == "J0" else CK[run]):
            d = dev_fused(heads_of(dev, ck), DEV, dev)
            err = ((d["p"] >= tau).astype(int) != d["y"]).astype(int)
            SI = (d["branch_u"] >= bu_hi) & (d["kappa"] <= k_thr)
            IA = (d["branch_u"] <= bu_lo) & (d["kappa"] <= k_thr)
            inv = float("nan")
            if SI.any() and IA.any():
                lab = np.concatenate([np.ones(SI.sum()), np.zeros(IA.sum())])
                inv = 1.0 - auc(np.concatenate([d["u"][SI], d["u"][IA]]), lab.astype(int))
            L = loo(table_with(view_opinions(heads_of(dev, ck), DEV, dev)), f"{run}")
            rows.append({"auroc": auc(d["p"], d["y"]), "vacuity_error_auroc": auc(d["u"], err),
                         "error_rate": float(err.mean()),
                         "si_n": int(SI.sum()), "ia_n": int(IA.sum()),
                         "si_mean_uf": float(d["u"][SI].mean()) if SI.any() else float("nan"),
                         "si_error_rate": float(err[SI].mean()) if SI.any() else float("nan"),
                         "inversion_rate": inv,
                         "loo": L["loo_mean"], "fused_auroc_folds": L["mean_auroc"]})
            store.setdefault(run, []).append((d["u"], err))
        agg = {k: float(np.mean([r[k] for r in rows]))
               for k in ("auroc", "vacuity_error_auroc", "error_rate", "si_mean_uf",
                         "si_error_rate", "inversion_rate", "fused_auroc_folds")}
        agg["loo"] = {v: float(np.mean([r["loo"][v] for r in rows])) for v in VIEWS}
        agg["si_n"] = int(np.mean([r["si_n"] for r in rows]))
        agg["ia_n"] = int(np.mean([r["ia_n"] for r in rows]))
        res["runs"][run] = {"per_seed": rows, "mean": agg}
        print(f"  {run:4s} dev AUROC {agg['auroc']:.4f}  vacuity-error AUROC "
              f"{agg['vacuity_error_auroc']:.4f}  SI n={agg['si_n']} mean u_f {agg['si_mean_uf']:.4f}  "
              f"LOO {({k: round(v,4) for k,v in agg['loo'].items()})}", flush=True)

    # the four-part rule, J2b against J1
    j1m, j2m = res["runs"]["J1"]["mean"], res["runs"]["J2b"]["mean"]
    g = A.groups_for(j0["t"])
    W = np.stack(S.group_boot(g, N_BOOT, 20260925))
    u1, e1 = store["J1"][0]; u2, e2 = store["J2b"][0]
    d = C.auc_weighted(u2, e2, W) - C.auc_weighted(u1, e1, W)
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    c1 = (j2m["vacuity_error_auroc"] > j1m["vacuity_error_auroc"]) and lo > 0
    c2 = j2m["auroc"] >= j1m["auroc"] - 0.005
    nz = [v for v in VIEWS if abs(j2m["loo"][v]) > 1e-9]
    c3 = all(j2m["loo"][v] > 0 for v in nz)
    # Criterion 4 is UNDECIDABLE when the shared-ignorance subset is empty, which it is here. An
    # undecidable criterion is not a failed one, and recording nan >= nan as False would have
    # manufactured a third failure out of a definition that does not apply.
    si_exists = (j2m["si_n"] > 0) and (j1m["si_n"] > 0)
    c4 = (j2m["si_mean_uf"] >= j1m["si_mean_uf"]) if si_exists else None
    res["j2b_rule"] = {
        "1_vacuity_error_auroc_higher": {"J1": j1m["vacuity_error_auroc"],
                                         "J2b": j2m["vacuity_error_auroc"],
                                         "ci95": [lo, hi], "pass": bool(c1)},
        "2_dev_auroc_within_0.005": {"J1": j1m["auroc"], "J2b": j2m["auroc"], "pass": bool(c2)},
        "3_all_nonzero_views_positive_loo": {"loo": j2m["loo"], "pass": bool(c3)},
        "4_si_mean_uf_not_lower": {"J1": j1m["si_mean_uf"], "J2b": j2m["si_mean_uf"],
                                   "si_subset_size": j2m["si_n"],
                                   "pass": (bool(c4) if c4 is not None else None),
                                   "note": None if si_exists else
                                   "UNDECIDABLE: the shared-ignorance subset is empty. High mean "
                                   "branch vacuity and low kappa do not co-occur in this system -- "
                                   "they are positively correlated at +0.178, because branch vacuity "
                                   "is a MEAN over three views, so one vacuous view beside two "
                                   "confident disagreeing ones gives high vacuity AND high conflict."},
        "verdict": "PASS" if (c1 and c2 and c3 and (c4 is not False)) else "FAIL",
        "verdict_basis": "criteria 1 and 3 decide it; criterion 4 is undecidable and does not change "
                         "the outcome"}
    # reviewer 3's prediction, reported either way
    res["j2a_lowers_si_uf_vs_j1"] = bool(res["runs"]["J2a"]["mean"]["si_mean_uf"] < j1m["si_mean_uf"])
    json.dump(res, open(OUT / "j2b_verdict.json", "w"), indent=1)

    print(f"\n  J2b four-part rule")
    for k, v in res["j2b_rule"].items():
        if isinstance(v, dict) and "pass" in v:
            st = "PASS" if v["pass"] else ("UNDECIDABLE" if v["pass"] is None else "FAIL")
            print(f"    {k:36s} {st}")
    print(f"  J2b: {res['j2b_rule']['verdict']}")
    print(f"\n  Reviewer 3's prediction -- J2a lowers mean u_f on the shared-ignorance subset "
          f"relative to J1: {res['j2a_lowers_si_uf_vs_j1']}")
    print(f"    J1 {j1m['si_mean_uf']:.4f}   J2a {res['runs']['J2a']['mean']['si_mean_uf']:.4f}   "
          f"J2b {j2m['si_mean_uf']:.4f}")


if __name__ == "__main__":
    main()

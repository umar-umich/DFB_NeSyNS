#!/usr/bin/env python3
"""Choose lambda for J2a and J2b, by the brief's pre-set rule.

    <discern_ext python> training/joint_ibdc/lambda_select.py --run J2b

The rule: "run lambda in {0.1, 0.3, 1.0} with seed 0. Choose the lambda with the highest fused
vacuity error AUROC on VALmix-v2, among those whose VALmix-v2 score is within -0.005 of J1's. The
development pool is not used to choose lambda."

Two metrics are needed that the training loop does not record, so they are computed here from each
checkpoint.

**VALmix-v2 score.** The v3.1 objective restricted to the cells whose source lives in VALmix-v2 --
CDFv3valG, DF40train, DS20val, DS11val and FFIWval. DF40train is scored against FF++ validation
genuine, as everywhere else, because it ships no genuine videos of its own.

**Fused vacuity error AUROC.** How well u_f ranks the fused system's OWN errors, where an error is a
wrong decision at the threshold. The threshold is taken at FF++ c23 validation EER, frozen, as
deployed. This is the quantity the corrected IBDC loss is supposed to improve, so it is the right
thing to select lambda on.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(ROOT / "analysis/tbiom/pcr6p"), str(ROOT / "analysis/tbiom/grouped"),
                str(ROOT / "analysis/tbiom"), str(ROOT / "analysis/tbiom/temporal"),
                str(ROOT / "branches/ijepa"), str(Path(__file__).resolve().parent)]

import runs as R          # noqa: E402
import fusion_torch as FT # noqa: E402
import v31 as V31         # noqa: E402

OUT = ROOT / "analysis/joint_ibdc"
LAMBDAS = (0.1, 0.3, 1.0)
V2_SOURCES = {"CDFv3valG", "DF40train", "DS20val", "DS11val", "FFIWval"}


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    if len(np.unique(y)) < 2:
        return float("nan")
    r = rankdata(s); n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def eer_threshold(p, y):
    """FF++ validation EER threshold, the deployed convention."""
    order = np.argsort(p)
    ps, ys = np.asarray(p)[order], np.asarray(y)[order]
    best, tau = 1e9, 0.5
    for t in np.unique(ps):
        fpr = float(((ps >= t) & (ys == 0)).sum()) / max((ys == 0).sum(), 1)
        fnr = float(((ps < t) & (ys == 1)).sum()) / max((ys == 1).sum(), 1)
        if abs(fpr - fnr) < best:
            best, tau = abs(fpr - fnr), float(t)
    return tau


@torch.no_grad()
def measure(heads, DEV, dev):
    """VALmix-v2 score, and the vacuity error AUROC on VALmix-v2 at the FF++ val EER threshold."""
    app, om, t = heads
    per = {}
    for name, D in DEV.items():
        ea, eo = app(torch.from_numpy(D["app"]).to(dev)), om(torch.from_numpy(D["om"]).to(dev))
        oa = FT.opinion_of_evidence(R.to_fusion(ea)); oo = FT.opinion_of_evidence(R.to_fusion(eo))
        inv = torch.from_numpy(D["inv"]).to(dev); n = len(D["order"])
        cnt = torch.bincount(inv, minlength=n).clamp_min(1).unsqueeze(1).float()
        va = torch.zeros(n, 3, device=dev).index_add_(0, inv, oa) / cnt
        vo = torch.zeros(n, 3, device=dev).index_add_(0, inv, oo) / cnt
        vt = FT.opinion_of_evidence(R.to_fusion(t(torch.from_numpy(D["t"]).to(dev))))
        f = FT.fuse(torch.stack([vt, va, vo], dim=1), plus=True)
        per[name] = {"p": FT.p_of(f).float().cpu().numpy(), "u": f[:, 2].float().cpu().numpy(),
                     "y": D["y"], "key": D["key"]}

    ff = per["FFpp_val"]
    tau = eer_threshold(ff["p"], ff["y"])

    # VALmix-v2 score: v3.1 restricted to VALmix-v2 sources
    d = per["VALmixV2"]
    sc, types, cells = V31.score(d["key"], d["p"], d["y"], ff["key"], ff["p"], ff["y"])
    v2 = float(np.nanmean([v for (typ, src), v in cells.items()
                           if src in V2_SOURCES and not np.isnan(v)]))

    err = ((d["p"] >= tau).astype(int) != d["y"]).astype(int)
    vac_auc = auc(d["u"], err)
    return {"valmixv2_score": v2, "vacuity_error_auroc_v2": vac_auc, "tau_ffpp_val_eer": tau,
            "error_rate_v2": float(err.mean()), "full_v31": sc, "types": types}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default="J2b"); a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    DEV = R.load_dev()

    # J1 reference: the three-seed mean of its VALmix-v2 score
    j1 = []
    for s in range(3):
        ck = OUT / f"run_J1_s{s}.pt"
        if not ck.exists():
            continue
        h = R.load_deployed_heads(dev); st = torch.load(ck, map_location=dev)
        h[0].load_state_dict(st["app"]); h[1].load_state_dict(st["om"]); h[2].load_state_dict(st["t"])
        for x in h: x.eval()
        j1.append(measure(h, DEV, dev))
    j1_v2 = float(np.mean([m["valmixv2_score"] for m in j1]))
    print(f"  J1 VALmix-v2 score (mean of {len(j1)} seeds): {j1_v2:.4f}")

    rows = {}
    for lam in LAMBDAS:
        ck = OUT / f"run_{a.run}_s0_lam{lam}.pt"
        if not ck.exists():
            print(f"  lambda {lam}: checkpoint missing, skipped"); continue
        h = R.load_deployed_heads(dev); st = torch.load(ck, map_location=dev)
        h[0].load_state_dict(st["app"]); h[1].load_state_dict(st["om"]); h[2].load_state_dict(st["t"])
        for x in h: x.eval()
        rows[lam] = measure(h, DEV, dev)
        r = rows[lam]
        print(f"  lambda {lam}: VALmix-v2 {r['valmixv2_score']:.4f} "
              f"(vs J1 {r['valmixv2_score']-j1_v2:+.4f})   vacuity error AUROC {r['vacuity_error_auroc_v2']:.4f}")

    eligible = {l: r for l, r in rows.items() if r["valmixv2_score"] >= j1_v2 - 0.005}
    print(f"\n  eligible (VALmix-v2 within -0.005 of J1): {sorted(eligible) or 'none'}")
    chosen = max(eligible, key=lambda l: eligible[l]["vacuity_error_auroc_v2"]) if eligible else None
    print(f"  chosen lambda for {a.run}: {chosen}"
          + (f"  (vacuity error AUROC {eligible[chosen]['vacuity_error_auroc_v2']:.4f})" if chosen else ""))
    json.dump({"run": a.run, "j1_valmixv2_mean": j1_v2, "per_lambda": {str(k): v for k, v in rows.items()},
               "eligible": [str(k) for k in sorted(eligible)], "chosen_lambda": chosen,
               "rule": "highest fused vacuity error AUROC on VALmix-v2 among lambdas whose VALmix-v2 "
                       "score is within -0.005 of J1's"},
              open(OUT / f"lambda_{a.run}.json", "w"), indent=1)


if __name__ == "__main__":
    main()

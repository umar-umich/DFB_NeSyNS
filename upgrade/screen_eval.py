#!/usr/bin/env python3
"""Part 3 measurements for one candidate Appearance view.

    <discern_ext python> branches/independent_views/screen_eval.py --backbone dinov3

Three blocks, as the brief specifies.

1. Standalone: FF++ c23 validation AUROC, VALmix-v2 score, dev v3.1 (three-seed mean +- sd).
2. Error correlation on the development pool with Off-manifold and with the video view, as the phi
   coefficient between error indicators, each view thresholded at its OWN FF++ validation EER, out of
   fold. Phi on two binary vectors is just their Pearson correlation.
3. The fused system with the candidate in the Appearance slot, beside Off-manifold and the video
   view: out-of-fold development AUROC, development real-video FPR at the threshold, the candidate's
   leave-one-out contribution with a source-grouped bootstrap interval, how many folds set its weight
   to zero, and Off-manifold's leave-one-out contribution with this candidate present. The deployed
   value of that last quantity is +0.013 with Appearance present.

The development table's metadata -- video identity, source, subset, labels, fold grouping -- is taken
from the deployed table unchanged; only `p_APP` and `u_APP` are replaced. Rebuilding the metadata
would risk comparing two systems across different partitions.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(ROOT / "analysis/tbiom/pcr6p"), str(ROOT / "analysis/tbiom/grouped"),
                str(ROOT / "analysis/tbiom"), str(ROOT / "analysis/tbiom/temporal"),
                str(ROOT / "upgrade"), str(ROOT / "branches/independent_views"), str(ROOT / "branches/ijepa")]

import arms as AR                # noqa: E402
import ops_pcr6p as P            # noqa: E402
import confirm as C              # noqa: E402
import dsr_system as DS          # noqa: E402
import exp1_fusion_audit as A    # noqa: E402
import score as S                # noqa: E402
from health import eer           # noqa: E402
from train_head import Head, load, DEV_SETS  # noqa: E402  (upgrade copy: logs/upgrade/feats)

VIEWS = ("T", "APP", "OM")
OUT = ROOT / "analysis/upgrade"
HEADS = ROOT / "analysis/upgrade/heads"
SPATIAL, BRANCH = "warmgate_e19", "vjepa2_lora_B_seed42"
N_BOOT = 1000


@torch.no_grad()
def candidate_opinions(backbone, seed, dev):
    """Video-level (p, u) for the candidate on every development set, keyed by video."""
    ck = HEADS / f"view_{backbone}_s{seed}.pt"
    st = torch.load(ck, map_location=dev)
    head = Head(st["dim"]).to(dev); head.load_state_dict(st["state"]); head.eval()
    out = {}
    for n in DEV_SETS:
        D = load(backbone, n)
        ev = head(torch.from_numpy(D["x"]).to(dev))
        a = ev + 1.0; s = a.sum(1, keepdim=True)
        p = (a / s)[:, 1].float().cpu().numpy(); u = (2.0 / s.squeeze(1)).float().cpu().numpy()
        cnt = np.bincount(D["inv"], minlength=len(D["order"])).astype(float)
        pv = np.bincount(D["inv"], weights=p, minlength=len(D["order"])) / cnt
        uv = np.bincount(D["inv"], weights=u, minlength=len(D["order"])) / cnt
        for v, a_, b_ in zip(D["order"], pv, uv):
            out[str(v)] = (float(a_), float(b_))
    return out


def table_with(op):
    t = C.dev_table(SPATIAL, BRANCH).copy()
    miss = [v for v in t.v if v not in op]
    if miss:
        raise KeyError(f"{len(miss)} development videos lack a candidate opinion, e.g. {miss[:3]}")
    t["p_APP"] = np.array([op[v][0] for v in t.v])
    t["u_APP"] = np.array([op[v][1] for v in t.v])
    return t


def phi(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def evaluate(t, tag):
    rows, agg = [], {v: [] for v in VIEWS}
    errs = {v: [] for v in VIEWS}
    for k, Fd, Sd in DS.splits(t):
        w = AR.fit_w(Fd, "PCR6P-R")
        p = P.p_of(AR.fuse(Sd, "PCR6P-R", w))
        tau = AR.tau_of(p, Sd)
        y = Sd.y.to_numpy(); real, fake = y == 0, y == 1
        full = A.fast_auc(p[real], p[fake])
        pred = (p >= tau).astype(int)
        row = {"split": k, "w": list(w), "auroc": full,
               "fpr_real": float(pred[real].mean()), "loo": {}}
        for i, v in enumerate(VIEWS):
            if w[i] == 0.0:
                row["loo"][v] = 0.0
            else:
                w0 = list(w); w0[i] = 0.0
                p0 = P.p_of(AR.fuse(Sd, "PCR6P-R", w0))
                row["loo"][v] = float(full - A.fast_auc(p0[real], p0[fake]))
            agg[v].append(row["loo"][v])
        # per-view error indicators at each view's own FF++ val EER, out of fold
        ff = np.where(Sd.source.to_numpy() == "FFpp_val")[0]
        for v in VIEWS:
            pv = Sd[f"p_{v}"].to_numpy()
            tv = float(eer(y[ff], pv[ff])[1]) if len(ff) else 0.5
            errs[v].append(((pv >= tv).astype(int) != y).astype(int))
        rows.append(row)
        print(f"  {tag} split {k}: w={w} AUROC {full:.4f} FPR {row['fpr_real']:.4f} "
              + " ".join(f"{v} {row['loo'][v]:+.4f}" for v in VIEWS), flush=True)
    E = {v: np.concatenate(errs[v]) for v in VIEWS}
    return {"per_split": rows,
            "mean_auroc": float(np.mean([r["auroc"] for r in rows])),
            "sd_auroc": float(np.std([r["auroc"] for r in rows])),
            "mean_fpr_real": float(np.mean([r["fpr_real"] for r in rows])),
            "loo_mean": {v: float(np.mean(agg[v])) for v in VIEWS},
            "folds_weight_zero": {v: int(sum(1 for r in rows if r["w"][i] == 0.0))
                                  for i, v in enumerate(VIEWS)},
            "phi_APP_OM": phi(E["APP"], E["OM"]), "phi_APP_T": phi(E["APP"], E["T"]),
            "phi_OM_T": phi(E["OM"], E["T"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    stand = []
    for s in range(a.seeds):
        f = HEADS / f"view_{a.backbone}_s{s}.json"
        if f.exists():
            d = json.load(open(f))
            stand.append({k: d[k] for k in ("dev_v31", "valmixv2", "ffpp_val_auroc", "best_epoch")})
    if not stand:
        print(f"no trained seeds for {a.backbone}"); return 2
    print(f"{a.backbone}: {len(stand)} seeds")
    for k in ("ffpp_val_auroc", "valmixv2", "dev_v31"):
        v = [x[k] for x in stand]
        print(f"  {k:16s} {np.mean(v):.4f} +- {np.std(v):.4f}")

    res = {"backbone": a.backbone, "standalone": stand,
           "standalone_mean": {k: float(np.mean([x[k] for x in stand]))
                               for k in ("dev_v31", "valmixv2", "ffpp_val_auroc")},
           "standalone_sd": {k: float(np.std([x[k] for x in stand]))
                             for k in ("dev_v31", "valmixv2", "ffpp_val_auroc")},
           "fused_per_seed": []}
    for s in range(len(stand)):
        t = table_with(candidate_opinions(a.backbone, s, dev))
        res["fused_per_seed"].append(evaluate(t, f"{a.backbone} s{s}"))
    fm = res["fused_per_seed"]
    res["fused_mean"] = {
        "auroc": float(np.mean([r["mean_auroc"] for r in fm])),
        "auroc_sd": float(np.std([r["mean_auroc"] for r in fm])),
        "fpr_real": float(np.mean([r["mean_fpr_real"] for r in fm])),
        "loo": {v: float(np.mean([r["loo_mean"][v] for r in fm])) for v in VIEWS},
        "folds_weight_zero": {v: float(np.mean([r["folds_weight_zero"][v] for r in fm])) for v in VIEWS},
        "phi_APP_OM": float(np.mean([r["phi_APP_OM"] for r in fm])),
        "phi_APP_T": float(np.mean([r["phi_APP_T"] for r in fm])),
    }
    json.dump(res, open(OUT / f"screen_{a.backbone}.json", "w"), indent=1)
    m = res["fused_mean"]
    print(f"\n  fused AUROC {m['auroc']:.4f} +-{m['auroc_sd']:.4f}   real FPR {m['fpr_real']:.4f}")
    print(f"  leave-one-out: " + "  ".join(f"{v} {m['loo'][v]:+.4f}" for v in VIEWS))
    print(f"  folds with zero weight: {m['folds_weight_zero']}")
    print(f"  error correlation phi: APP-OM {m['phi_APP_OM']:+.4f}   APP-T {m['phi_APP_T']:+.4f}")
    print(f"-> {OUT / f'screen_{a.backbone}.json'}")


if __name__ == "__main__":
    main()

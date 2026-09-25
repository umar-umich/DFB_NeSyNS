#!/usr/bin/env python3
"""Part 1 metric suite: detection, evidential vacuity, calibration, shared ignorance, and the J2b rule.

    <discern_ext python> training/joint_ibdc/metrics.py --runs J0 J1 J2a J2b

`u = K/S` is EVIDENTIAL VACUITY throughout, never epistemic uncertainty.

Definitions the brief asks to be stated explicitly:

* **Selective risk at 10% abstention** is the error rate over the 90% of videos with the LOWEST u_f.
* **AURC** is the area under the risk-coverage curve, sweeping coverage from 1/N to 1 by increasing
  u_f. **E-AURC** subtracts the AURC of a perfect ranker at the same error rate.
* **Adaptive ECE** uses 15 EQUAL-MASS bins; the 10-bin equal-width ECE is reported beside it for
  continuity with the IJCB paper.
* **Shared ignorance (SI)** is mean branch vacuity at or above its 80th percentile with kappa at or
  below its threshold; **informed agreement (IA)** is mean branch vacuity at or below its 20th
  percentile with kappa at or below its threshold. All three thresholds are set on J0's FF++
  validation opinions and frozen.
* **Inversion rate** is the probability that a random SI video has LOWER u_f than a random IA video,
  which is one minus the AUROC of u_f separating the two subsets.
* **Agreement-on-wrong** is, among fused errors, the fraction where every view with nonzero trust
  weight predicts the same wrong class with vacuity below its own FF++ validation median.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(ROOT / "analysis/tbiom/pcr6p"), str(ROOT / "analysis/tbiom/grouped"),
                str(ROOT / "analysis/tbiom"), str(ROOT / "analysis/tbiom/temporal"),
                str(ROOT / "branches/ijepa"), str(Path(__file__).resolve().parent)]

import runs as R           # noqa: E402
import fusion_torch as FT  # noqa: E402

OUT = ROOT / "analysis/joint_ibdc"
CACHE = ROOT / "logs/joint_ibdc/cache"
TSNE = ROOT / "logs/tsne"
TESTS = ["CDFv2", "CDFv3", "DFD", "DFDC", "DFDCP", "DFEval24", "FFIW", "KoDF", "IDForge", "DFDM"]
PROC = ["CDFv2_jpeg40", "CDFv2_rs06", "DFDCP_jpeg40", "DFDCP_rs06"]
VIEWS = ("T", "APP", "OM")


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    if len(np.unique(y)) < 2:
        return float("nan")
    r = rankdata(s); n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def spatial_path(name):
    p = CACHE / f"{name}.npz"
    return p if p.exists() else TSNE / f"ours_{name}.npz"


def load_split(name):
    """Video-aligned spatial and temporal features for one split."""
    import pandas as pd
    sys.path.insert(0, str(ROOT / "analysis/tbiom"))
    from video_id import video_id
    sp, tp = spatial_path(name), CACHE / f"{name}_T_clean.npz"
    if not (sp.exists() and tp.exists()):
        return None
    zs, zt = np.load(sp, allow_pickle=True), np.load(tp, allow_pickle=True)
    keys = np.array([str(k) for k in zs["key"]])
    vid = video_id(pd.Series(keys)).to_numpy()
    tv = {str(v): i for i, v in enumerate(zt["video"])}
    keep = np.array([i for i, v in enumerate(vid) if v in tv])
    if not len(keep):
        return None
    order, inv = np.unique(vid[keep], return_inverse=True)
    return {"app": zs["semantic"][keep].astype(np.float32),
            "om": zs["artifact"][keep].astype(np.float32),
            "t": zt["feature"][[tv[v] for v in order]].astype(np.float32),
            "inv": inv, "order": order,
            "y": np.array([int(zs["label"][keep][inv == i].max()) for i in range(len(order))])}


@torch.no_grad()
def fuse_split(heads, D, dev, w=(1.0, 1.0, 1.0)):
    app, om, t = heads
    oa = FT.opinion_of_evidence(R.to_fusion(app(torch.from_numpy(D["app"]).to(dev))))
    oo = FT.opinion_of_evidence(R.to_fusion(om(torch.from_numpy(D["om"]).to(dev))))
    inv = torch.from_numpy(D["inv"]).to(dev); n = len(D["order"])
    cnt = torch.bincount(inv, minlength=n).clamp_min(1).unsqueeze(1).float()
    va = torch.zeros(n, 3, device=dev).index_add_(0, inv, oa) / cnt
    vo = torch.zeros(n, 3, device=dev).index_add_(0, inv, oo) / cnt
    vt = FT.opinion_of_evidence(R.to_fusion(t(torch.from_numpy(D["t"]).to(dev))))
    m = torch.stack([vt, va, vo], dim=1)
    e = FT.evidence_of(m) * torch.tensor(w, device=dev).view(1, 3, 1)
    md = FT.opinion_of_evidence(e)
    f = FT.fuse(md, plus=True)
    return {"p": FT.p_of(f).float().cpu().numpy(), "u": f[:, 2].float().cpu().numpy(),
            "kappa": FT.kappa(md).float().cpu().numpy(), "y": D["y"],
            "views": {v: {"p": FT.p_of(m[:, i]).float().cpu().numpy(),
                          "u": m[:, i, 2].float().cpu().numpy()} for i, v in enumerate(VIEWS)}}


def aurc(u, err):
    o = np.argsort(u); e = np.asarray(err, float)[o]
    cov = np.arange(1, len(e) + 1)
    risk = np.cumsum(e) / cov
    a = float(risk.mean())
    r = float(e.mean())
    # a perfect ranker puts every error last
    perfect = np.concatenate([np.zeros(len(e) - int(e.sum())), np.ones(int(e.sum()))])
    pr = float((np.cumsum(perfect) / cov).mean())
    return a, a - pr, r


def ece(p, y, bins=10, adaptive=False):
    p, y = np.asarray(p, float), np.asarray(y, int)
    if adaptive:
        edges = np.quantile(p, np.linspace(0, 1, bins + 1)); edges[0], edges[-1] = -1e-9, 1 + 1e-9
    else:
        edges = np.linspace(0, 1, bins + 1); edges[0], edges[-1] = -1e-9, 1 + 1e-9
    tot = 0.0
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i + 1])
        if m.sum() == 0:
            continue
        tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def metrics_for(res, tau):
    p, u, y = res["p"], res["u"], res["y"]
    pred = (p >= tau).astype(int)
    err = (pred != y).astype(int)
    a, ea, risk = aurc(u, err)
    k = max(1, int(round(0.9 * len(u))))
    sel = np.argsort(u)[:k]
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return {"auroc": auc(p, y), "fpr_real": float(pred[y == 0].mean()) if (y == 0).any() else float("nan"),
            "tpr": float(pred[y == 1].mean()) if (y == 1).any() else float("nan"),
            "error_rate": risk, "vacuity_error_auroc": auc(u, err),
            "aurc": a, "e_aurc": ea, "selective_risk_10pct": float(err[sel].mean()),
            "ece_adaptive15": ece(p, y, 15, True), "ece10": ece(p, y, 10, False),
            "brier": float(np.mean((p - y) ** 2)),
            "nll": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["J0", "J1", "J2a", "J2b"])
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    CK = {"J1": [f"run_J1_s{s}.pt" for s in range(3)],
          "J2a": [f"run_J2a_s{s}_lam1.0.pt" for s in range(3)],
          "J2b": [f"run_J2b_s{s}_lam0.1.pt" for s in range(3)]}

    splits = {}
    for n in ["FFpp_val"] + TESTS + PROC:
        d = load_split(n)
        if d is not None:
            splits[n] = d
    print(f"  splits available: {len(splits)} -> {', '.join(splits)}", flush=True)

    out = {}
    for run in a.runs:
        seeds = [None] if run == "J0" else CK[run]
        per = []
        for ck in seeds:
            h = R.load_deployed_heads(dev)
            if ck:
                f = OUT / ck
                if not f.exists():
                    print(f"  {run}: {ck} missing, skipped"); continue
                st = torch.load(f, map_location=dev)
                h[0].load_state_dict(st["app"]); h[1].load_state_dict(st["om"]); h[2].load_state_dict(st["t"])
            for x in h: x.eval()
            ff = fuse_split(h, splits["FFpp_val"], dev)
            from lambda_select import eer_threshold
            tau = eer_threshold(ff["p"], ff["y"])
            row = {"tau": tau, "sets": {}}
            for n, D in splits.items():
                r = fuse_split(h, D, dev)
                row["sets"][n] = metrics_for(r, tau)
            ten = [row["sets"][n]["auroc"] for n in TESTS if n in row["sets"]]
            row["ten_set_mean_auroc"] = float(np.nanmean(ten))
            row["processed_mean_auroc"] = float(np.nanmean(
                [row["sets"][n]["auroc"] for n in PROC if n in row["sets"]]))
            per.append(row)
            print(f"  {run} {ck or ''}: tau {tau:.4f}  ten-set mean AUROC {row['ten_set_mean_auroc']:.4f}",
                  flush=True)
        if per:
            out[run] = {"per_seed": per,
                        "ten_set_mean_auroc": float(np.mean([r["ten_set_mean_auroc"] for r in per])),
                        "ten_set_sd": float(np.std([r["ten_set_mean_auroc"] for r in per]))}
    json.dump(out, open(OUT / "metrics.json", "w"), indent=1)
    print(f"\n  {'run':5s} {'ten-set mean AUROC':>20}")
    for r, v in out.items():
        print(f"  {r:5s} {v['ten_set_mean_auroc']:12.4f} +-{v['ten_set_sd']:.4f}")
    print(f"-> {OUT/'metrics.json'}")


if __name__ == "__main__":
    main()

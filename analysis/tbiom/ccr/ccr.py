#!/usr/bin/env python3
"""Conflict-conditioned reliability: make the discount a function of inter-view conflict.

    OMP_NUM_THREADS=4 <discern_ext python> analysis/tbiom/ccr/ccr.py

Pre-registered in `tbiom/docs/ccr-preregistration.md`. Arms, features, folds, controls and the
acceptance rule were fixed before any number here existed.

The deployed system fits ONE reliability triple per fold. CCR predicts w per VIDEO from the conflict
between the views, trained by back-propagating binary cross-entropy on the fused probability THROUGH
the PCR6+ fusion -- which is possible because the Task 1 implementation is differentiable and was
verified to reproduce the deployed fusion exactly.

The control that makes this interpretable is `ccr-shuffled`: the same network on conflict features
permuted across videos. If that also passes, the gain is capacity rather than conflict.
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(ROOT / "analysis/tbiom/pcr6p"), str(ROOT / "analysis/tbiom/grouped"),
                str(ROOT / "analysis/tbiom"), str(ROOT / "analysis/tbiom/temporal"),
                str(ROOT / "training/joint_ibdc")]

import arms as AR                 # noqa: E402
import ops_pcr6p as P             # noqa: E402
import confirm as C               # noqa: E402
import dsr_system as DS           # noqa: E402
import exp1_fusion_audit as A     # noqa: E402
import score as S                 # noqa: E402
import fusion_torch as FT         # noqa: E402

torch.set_num_threads(4)
VIEWS = ("T", "APP", "OM")
SPATIAL, BRANCH = "warmgate_e19", "vjepa2_lora_B_seed42"
OUT = ROOT / "analysis/ccr"
EPOCHS, LR, WD, HID, N_BOOT = 60, 1e-3, 1e-4, 32, 1000
QC = ["blur", "blockiness"]                       # the known-negative reference, where available


def opinions_of(t):
    """(n, 3, 3) undiscounted opinions in VIEWS order."""
    return torch.tensor(P.masses_of(t, VIEWS, None), dtype=torch.float64)


def conflict_features(m):
    """Nine inputs: three pairwise kappa, three vacuities, three margins. All from the opinions."""
    kij = FT.pairwise_kappa(m)                                   # (n, 3)
    u = m[:, :, 2]                                               # (n, 3)
    p = m[:, :, 0] + m[:, :, 2] / 2
    marg = -(p - 0.5).abs()
    return torch.cat([kij, u, marg], dim=1).float()


class WNet(nn.Module):
    """Features -> a reliability in [0,1] per view. Initialised near the constant triple."""

    def __init__(self, d_in, w0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, HID), nn.GELU(), nn.Linear(HID, 3))
        nn.init.zeros_(self.net[-1].weight)
        self.net[-1].bias.data = torch.logit(torch.tensor(w0, dtype=torch.float32).clamp(1e-3, 1 - 1e-3))

    def forward(self, x):
        return torch.sigmoid(self.net(x))


def fuse_with(m, w):
    """m (n,3,3) undiscounted, w (n,3) per-video reliabilities -> fused p. Differentiable in w."""
    e = FT.evidence_of(m.float())                                # (n,3,2)
    o = FT.opinion_of_evidence(e * w.unsqueeze(-1))              # evidence scaling, the deployed discount
    return FT.p_of(FT.fuse(o, plus=True))


def fit(Xf, mf, yf, w0, seed=0):
    torch.manual_seed(seed)
    net = WNet(Xf.shape[1], w0)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    y = torch.tensor(yf, dtype=torch.float32)
    for _ in range(EPOCHS):
        p = fuse_with(mf, net(Xf)).clamp(1e-6, 1 - 1e-6)
        loss = nn.functional.binary_cross_entropy(p.float(), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return net.eval()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = C.dev_table(SPATIAL, BRANCH)
    print(f"development pool {len(dev)} videos, operator PCR6+-R, discount e' = r*e")
    rows = []
    for k, Fd, Sd in DS.splits(dev):
        w0 = AR.fit_w(Fd, "PCR6P-R")                             # the deployed constant baseline
        mf, ms = opinions_of(Fd), opinions_of(Sd)
        Xf, Xs = conflict_features(mf), conflict_features(ms)
        cl = A.cells(Sd)
        out = {"split": k, "w_const": list(w0), "arms": {}}

        p_const = P.p_of(AR.fuse(Sd, "PCR6P-R", w0))
        out["arms"]["const"] = {"v31": A.v31(p_const, cl)[0]}

        rng = np.random.default_rng(1000 + k)
        variants = {
            "ccr": (Xf, Xs),
            "ccr-shuffled": (Xf[torch.from_numpy(rng.permutation(len(Xf)))],
                             Xs[torch.from_numpy(rng.permutation(len(Xs)))]),
        }
        for name, (xf, xs) in variants.items():
            net = fit(xf, mf, Fd.y.to_numpy(), w0, seed=k)
            with torch.no_grad():
                w = net(xs)
                p = fuse_with(ms, w).numpy()
            out["arms"][name] = {"v31": A.v31(p, cl)[0],
                                 "w_mean": w.mean(0).tolist(), "w_sd": w.std(0).tolist()}
            out.setdefault("_p", {})[name] = p
        out["_p"]["const"] = p_const
        out["_y"] = Sd.y.to_numpy()
        out["_g"] = A.groups_for(Sd)
        rows.append(out)
        print(f"  split {k}: w_const={w0}  " +
              "  ".join(f"{n} {out['arms'][n]['v31']:.4f}" for n in out["arms"]), flush=True)

    res = {"study": "conflict-conditioned reliability", "prereg": "tbiom/docs/ccr-preregistration.md",
           "n_dev": int(len(dev)), "per_split": [], "summary": {}}
    arms = list(rows[0]["arms"])
    for a in arms:
        v = [r["arms"][a]["v31"] for r in rows]
        res["summary"][a] = {"v31_mean": float(np.mean(v)), "v31_sd": float(np.std(v)),
                             "per_fold": v}
    # paired source-grouped bootstrap of each arm against const, per fold
    for a in arms:
        if a == "const":
            continue
        above = 0
        for r in rows:
            W = np.stack(S.group_boot(r["_g"], N_BOOT, 20260925 + r["split"]))
            d = (C.auc_weighted(r["_p"][a], r["_y"], W)
                 - C.auc_weighted(r["_p"]["const"], r["_y"], W))
            lo = float(np.percentile(d, 2.5))
            above += lo > 0
        res["summary"][a]["folds_interval_above_zero"] = above
    for r in rows:
        res["per_split"].append({k: v for k, v in r.items() if not k.startswith("_")})

    d_mean = res["summary"]["ccr"]["v31_mean"] - res["summary"]["const"]["v31_mean"]
    sh_mean = res["summary"]["ccr-shuffled"]["v31_mean"] - res["summary"]["const"]["v31_mean"]
    passes = (d_mean >= 0.005 and res["summary"]["ccr"]["folds_interval_above_zero"] >= 3)
    ctrl_ok = not (sh_mean >= 0.005 and res["summary"]["ccr-shuffled"]["folds_interval_above_zero"] >= 3)
    res["verdict"] = "ACCEPT" if (passes and ctrl_ok) else "REJECT"
    res["delta_v31"] = {"ccr": d_mean, "ccr-shuffled": sh_mean}
    res["control_clean"] = ctrl_ok
    json.dump(res, open(OUT / "ccr.json", "w"), indent=1)

    print(f"\n  {'arm':14s} {'v3.1 mean':>10} {'sd':>7} {'delta':>9} {'folds CI>0':>11}")
    base = res["summary"]["const"]["v31_mean"]
    for a in arms:
        sm = res["summary"][a]
        delta = "" if a == "const" else f"{sm['v31_mean'] - base:+.4f}"
        folds = "" if a == "const" else f"{sm['folds_interval_above_zero']}/5"
        print(f"  {a:14s} {sm['v31_mean']:10.4f} {sm['v31_sd']:7.4f} {delta:>9} {folds:>11}")
    print(f"\n  bar: +0.005 mean AND interval above zero in >= 3 of 5 folds; "
          f"shuffled control must NOT pass")
    print(f"CCR {res['verdict']} -> {OUT / 'ccr.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

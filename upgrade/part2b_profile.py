#!/usr/bin/env python3
"""Part 2b: feature-space factor profile, adapted from evidence subspace projection (Xiao et al., 2026).

    <discern_ext python> upgrade/part2b_profile.py

ADAPTED, NOT REPRODUCED. ESP is a neuron-level method; this works on whole-representation contrasts
and reports how much of the class-separation direction lies inside each factor's subspace. Neuron
level is deferred to Part 7 on the selected system only, and the report must not call this ESP.

METHOD. Standardise with FF++ TRAIN statistics. For each factor group build contrast vectors r_g
between its populations, take an SVD basis Q of those contrasts, and report
    E_delta = ||Q^T d||^2    for d the class-separation direction (fake mean minus real mean),
plus E_delta divided by the effective rank, so a factor with many directions cannot win merely by
spanning more of the space.

NULL. 1,000 permutations, and the permutation respects the structure: processing labels are permuted
WITHIN each video, every other group across videos. The 95th percentile of the null is reported
beside each statistic -- a raw E_delta means nothing without it, because a high-dimensional basis
captures a sizeable share of any direction by chance.

TERMINOLOGY, as the brief fixes it: `d` is the CLASS-SEPARATION DIRECTION, the fake-minus-real mean
contrast. It is NOT the decision direction of the nonlinear evidential head, and nothing here shows
that a nuisance CAUSES the detector's errors -- these are properties of the representation.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(ROOT / "upgrade"), str(ROOT / "branches/ijepa")]
FE = ROOT / "logs/upgrade/feats"
LF = ROOT / "logs/upgrade/lfeats"
OUT = ROOT / "analysis/upgrade"
NPERM = 1000
PROCS = ["jpeg:70", "jpeg:50", "jpeg:30", "resize:0.75", "resize:0.50", "resize:0.35"]


def standardise(X, mu, sd):
    return (X - mu) / sd


def contrasts_to_basis(R):
    R = np.asarray(R, dtype=np.float64)
    U, s, _ = np.linalg.svd(R.T, full_matrices=False)
    k = int((s > 1e-8 * max(1.0, s[0])).sum())
    return U[:, :k], k


def energy(Q, d):
    d = d / max(1e-12, np.linalg.norm(d))
    return float(np.sum((Q.T @ d) ** 2))


def eff_rank(Q):
    return max(1, Q.shape[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="pe_core_l", help="feature cache prefix to profile")
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    tr = np.load(FE / f"{a.view}_FFpp_train.npz", allow_pickle=True)
    Xtr = tr["feature"].astype(np.float64)
    mu, sd = Xtr.mean(0), np.clip(Xtr.std(0), 1e-6, None)      # FF++ TRAIN statistics, as the brief fixes

    val = np.load(FE / f"{a.view}_FFpp_val.npz", allow_pickle=True)
    X = standardise(val["feature"].astype(np.float64), mu, sd)
    y = val["label"].astype(int)
    vid = np.array([str(v) for v in val["video"]])

    d_class = X[y == 1].mean(0) - X[y == 0].mean(0)            # the CLASS-SEPARATION DIRECTION
    res = {"view": a.view, "dim": int(X.shape[1]), "frames": int(len(X)), "groups": {}}

    # ---- processing: paired renders of FF++ val, permuted WITHIN video
    rows = []
    for p in PROCS:
        f = LF / f"{a.view}_FFpp_val_{p.replace(':','')}.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        xp = standardise(z["feat_last"].astype(np.float64), mu[:z["feat_last"].shape[1]],
                         sd[:z["feat_last"].shape[1]]) if z["feat_last"].shape[1] == X.shape[1] else None
        if xp is None:
            continue
        n = min(len(xp), len(X))
        rows.append(xp[:n].mean(0) - X[:n].mean(0))
    if rows:
        Q, k = contrasts_to_basis(rows)
        e = energy(Q, d_class)
        null = []
        for _ in range(NPERM):
            dd = X[rng.permutation(len(X))]
            null.append(energy(Q, dd[y == 1].mean(0) - dd[y == 0].mean(0)))
        res["groups"]["processing"] = {"rank": k, "E_delta": e, "E_per_rank": e / eff_rank(Q),
                                       "null_p95": float(np.percentile(null, 95)),
                                       "above_null": bool(e > np.percentile(null, 95))}

    # ---- source: development real videos only, grouped by source dataset
    for name, sets in (("source", ("VALmix", "VALmixV2")),):
        rows, labs = [], []
        for s in sets:
            f = FE / f"{a.view}_{s}.npz"
            if not f.exists():
                continue
            z = np.load(f, allow_pickle=True)
            xs = standardise(z["feature"].astype(np.float64), mu, sd)
            ys = z["label"].astype(int)
            ks = np.array([str(k) for k in z["key"]])
            src = np.array([k.split("/")[1] if "/" in k else s for k in ks])
            real = ys == 0
            for u in np.unique(src[real]):
                m = real & (src == u)
                if m.sum() >= 30:
                    rows.append(xs[m].mean(0) - xs[real].mean(0)); labs.append(u)
        if len(rows) >= 2:
            Q, k = contrasts_to_basis(rows)
            e = energy(Q, d_class)
            null = [energy(Q, (lambda d: d[y == 1].mean(0) - d[y == 0].mean(0))(X[rng.permutation(len(X))]))
                    for _ in range(NPERM)]
            res["groups"][name] = {"rank": k, "levels": labs, "E_delta": e, "E_per_rank": e / eff_rank(Q),
                                   "null_p95": float(np.percentile(null, 95)),
                                   "above_null": bool(e > np.percentile(null, 95))}

    # ---- manipulation family: FF++ val fakes only, swap vs reenactment
    SWAP, REEN = {"Deepfakes", "FaceSwap"}, {"Face2Face", "NeuralTextures"}
    ks = np.array([str(k) for k in val["key"]])
    fam = np.array([k.split("/")[1] if "/" in k else "" for k in ks])
    msw = (y == 1) & np.isin(fam, list(SWAP)); mre = (y == 1) & np.isin(fam, list(REEN))
    if msw.sum() > 30 and mre.sum() > 30:
        Q, k = contrasts_to_basis([X[msw].mean(0) - X[mre].mean(0)])
        e = energy(Q, d_class)
        null = [energy(Q, (lambda d: d[y == 1].mean(0) - d[y == 0].mean(0))(X[rng.permutation(len(X))]))
                for _ in range(NPERM)]
        res["groups"]["manipulation_family"] = {"rank": k, "E_delta": e, "E_per_rank": e / eff_rank(Q),
                                                "null_p95": float(np.percentile(null, 95)),
                                                "above_null": bool(e > np.percentile(null, 95))}

    res["note"] = ("`d` is the class-separation direction (fake mean minus real mean), NOT the "
                   "decision direction of the evidential head. These describe the representation; "
                   "they do not show that a nuisance causes the detector's errors.")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / f"part2b_profile_{a.view}.json", "w"), indent=1)
    print(f"{'group':22s} {'rank':>5s} {'E_delta':>9s} {'per rank':>9s} {'null p95':>9s}  above null")
    for g, r in res["groups"].items():
        print(f"{g:22s} {r['rank']:5d} {r['E_delta']:9.4f} {r['E_per_rank']:9.5f} "
              f"{r['null_p95']:9.4f}  {'YES' if r['above_null'] else 'no'}")
    print(f"\nwrote {OUT}/part2b_profile_{a.view}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

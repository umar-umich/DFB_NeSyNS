#!/usr/bin/env python3
"""Shafer-discounted fusion — keep all three views, give them proportional influence.

    python analysis/tbiom/discounted_fusion.py --arm C

THE PROBLEM THIS TARGETS. On arm C's 3-view configuration the best operator is `Learned`, and
inspecting its fitted weights shows why: semantic = 0.000, artifact = 0.536, fsvfm = 0.464. It
wins by ZEROING the semantic view -- silently reproducing the 2-view configuration. Every
evidential operator (DS, Murphy, Yager, PCR6, ...) treats its sources as equally reliable, so
none of them can express "this view is worth less here", and all of them lose to a three-
parameter linear weighting.

THE FIX. Shafer discounting is the standard subjective-logic operation for exactly this:

    b' = q * b            u' = (1 - q) + q * u

A source with reliability q < 1 has its belief shrunk toward vacuity, so it contributes
proportionally rather than being deleted. Discount first, then fuse with any operator. The view
stays in the framework -- which is the requirement -- while the fusion stops over-counting a
view that is redundant with another.

WHERE q COMES FROM. Two options, both fitted on the DEVELOPMENT set only and applied unchanged
to every OOD set:

    accuracy   q_v = 2 * (video AUROC of view v on VALmix) - 1, clipped to [0, 1]. Parameter-free
               given the dev set: a chance view gets q = 0, a perfect one q = 1.
    fitted     q optimised directly to minimise BCE of the fused probability on VALmix. One
               parameter per view, the same budget `Learned` uses -- so if this still loses to
               `Learned`, the deficit is the operator, not the parameter count.

FIREWALL. No OOD export is opened while fitting. VALmix is the development split throughout, and
its own column is reported but excluded from the OOD mean.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from health import auroc, dashboard, eer  # noqa: E402
from video_id import video_id  # noqa: E402

S23 = Path("logs/tbiom/stage23")
K, EPS = 2, 1e-12
OOD = ["CDFv2", "CDFv3", "DFD", "DFDC", "DFDCP", "DFEval24"]
P0 = {"CDFv2": 0.9646, "CDFv3": 0.8409, "DFD": 0.9421, "DFDC": 0.8828,
      "DFDCP": 0.8573, "DFEval24": 0.6922}
V3 = ("semantic", "artifact", "fsvfm")
ARMS = {"B": ["run1auxedl", "seed1337", "armBs7"], "C": ["run1cft", "armCs1337", "armCs7"]}


def load(pref, ds, views=V3):
    f = S23 / f"{pref}_{ds}.csv"
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    if any(f"{k}_{v}" not in d for v in views for k in ("p", "u")):
        return None
    d["v"] = video_id(d["key"])
    agg = {f"{k}_{v}": (f"{k}_{v}", "mean") for v in views for k in ("p", "u")}
    agg["y"] = ("label", "max")
    return d.groupby("v", as_index=False).agg(**agg)


def masses(p, u, q=1.0):
    """(p, u) -> (m_real, m_fake, m_theta), Shafer-discounted by q."""
    u = np.clip(u, 1e-6, 1.0)
    mr, mf = (1 - p) * (1 - u), p * (1 - u)
    return q * mr, q * mf, (1 - q) + q * u


def ds_pair(a, b):
    ar, af, at = a; br, bf, bt = b
    mr = ar * br + ar * bt + at * br
    mf = af * bf + af * bt + at * bf
    mt = at * bt
    d = np.clip(1 - (ar * bf + af * br), EPS, None)
    return mr / d, mf / d, mt / d


def fold_ds(ms):
    f = ms[0]
    for m in ms[1:]:
        f = ds_pair(f, m)
    return f


def fold_murphy(ms):
    avg = tuple(np.mean([m[j] for m in ms], 0) for j in range(3))
    return fold_ds([avg] * len(ms))


def prob(m):
    mr, mf, mt = m
    return (mf + 0.5 * mt) / np.clip(mr + mf + mt, EPS, None)


def build(g, qs, rule):
    ms = [masses(g[f"p_{v}"].to_numpy(), g[f"u_{v}"].to_numpy(), qs[i])
          for i, v in enumerate(V3)]
    return prob(rule(ms))


def q_from_accuracy(dev):
    return np.array([float(np.clip(2 * auroc(dev["y"].to_numpy(),
                                             dev[f"p_{v}"].to_numpy()) - 1, 0, 1)) for v in V3])


def q_fitted(dev, rule):
    from scipy.optimize import minimize
    y = dev["y"].to_numpy().astype(float)

    def nll(t):
        qs = 1.0 / (1.0 + np.exp(-t))                 # sigmoid keeps q in (0, 1)
        p = np.clip(build(dev, qs, rule), 1e-6, 1 - 1e-6)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    r = minimize(nll, np.zeros(len(V3)), method="Nelder-Mead",
                 options={"maxiter": 3000, "fatol": 1e-9})
    return 1.0 / (1.0 + np.exp(-r.x))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", default="C", choices=["B", "C"])
    args = ap.parse_args()
    prefs = ARMS[args.arm]
    mp0 = float(np.mean([P0[d] for d in OOD]))

    print(f"arm {args.arm} — 3 views kept, discounted.  P0-DS reference {mp0:.4f}\n")
    schemes = []
    for rname, rule in (("DS", fold_ds), ("Murphy", fold_murphy)):
        schemes += [(f"{rname}  (no discount)", rule, "none"),
                    (f"{rname} + q=accuracy", rule, "acc"),
                    (f"{rname} + q fitted", rule, "fit")]

    print(f"{'scheme':26s} " + " ".join(f"{d:>8s}" for d in OOD) +
          f" {'mean':>8s} {'vsP0':>8s} {'FPR':>6s}   q (sem/art/fsvfm)")
    for label, rule, qmode in schemes:
        per_seed_a, per_seed_f, qshow = [], [], None
        for pref in prefs:
            dev, ff = load(pref, "VALmix"), load(pref, "FFpp_val")
            if dev is None or ff is None:
                continue
            qs = (np.ones(3) if qmode == "none" else
                  q_from_accuracy(dev) if qmode == "acc" else q_fitted(dev, rule))
            if qshow is None:
                qshow = qs
            _, tau = eer(ff["y"].to_numpy(), build(ff, qs, rule))
            a, f_ = [], []
            for ds in OOD:
                g = load(pref, ds)
                if g is None:
                    continue
                d = dashboard(g["y"].to_numpy(), build(g, qs, rule), tau)
                a.append(d["auroc"]); f_.append(d["fpr_real_at_tau"])
            if len(a) == len(OOD):
                per_seed_a.append(a); per_seed_f.append(np.mean(f_))
        if not per_seed_a:
            continue
        A = np.mean(per_seed_a, 0)
        print(f"{label:26s} " + " ".join(f"{x:>8.4f}" for x in A) +
              f" {A.mean():>8.4f} {A.mean()-mp0:>+8.4f} {np.mean(per_seed_f):>6.3f}   "
              + "/".join(f"{q:.2f}" for q in qshow))
    print("\nReference from the operator grid: Learned (linear weights) "
          f"{'0.8809' if args.arm=='C' else '0.8771'}, and its weights zero the semantic view.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Stage 7b — the conflict-aware operators Stage 7 did not cover.

    python analysis/tbiom/stage7b_operators.py --arm run1cft

Stage 7 tested DS, Yager, Murphy and CCF and adopted none. Three operators were named
afterwards and are covered here.

FRAME. Two classes, so every opinion has exactly three focal elements: {real}, {fake}, and
Theta = {real, fake} carrying the vacuity. That fact does real work below.

  PCR5/PCR6 (DSmT)  Proportional Conflict Redistribution. Instead of renormalising conflict
                    away (DS) or dumping it into vacuity (Yager), each partial conflict is
                    given BACK to the masses that produced it, in proportion to them. For two
                    sources PCR5 and PCR6 coincide; for three they do not, and PCR is
                    NON-ASSOCIATIVE, so the pairwise fold used here is PCR6-style and is
                    reported as such rather than claimed to be the n-ary operator.

  Dubois-Prade      Conflict is assigned to the UNION of the conflicting focal elements.
                    ON A BINARY FRAME THIS IS EXACTLY YAGER, and not approximately:
                    {real} and {fake} are the only conflicting pair and their union is Theta,
                    which is where Yager already puts the conflict. Verified numerically below
                    rather than asserted. It therefore adds no new operator here -- worth
                    knowing before anyone implements it a second time.

  Deng-entropy      Weight each source by its information content, average the weighted BBAs,
                    then combine the average with itself N-1 times (Murphy's schema with
                    entropy weights instead of uniform ones). Deng entropy on this frame is
                    E = -[b_r log2 b_r + b_f log2 b_f + u log2(u/3)], the /3 being 2^|Theta| - 1.
                    Weight is taken inversely proportional to E: a lower-entropy source is more
                    informative and counts for more. That choice is recorded because other
                    papers normalise the exponential of E instead, and the two differ.
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
OOD = ["CDFv2", "CDFv3", "DFD", "DFDC", "DFDCP", "DFEval24"]
P0 = {"CDFv2": 0.9646, "CDFv3": 0.8409, "DFD": 0.9421, "DFDC": 0.8828,
      "DFDCP": 0.8573, "DFEval24": 0.6922}
EPS = 1e-12


def masses(p, u):
    """(p, u) -> (m_real, m_fake, m_theta) with the three summing to 1."""
    u = np.clip(u, 1e-6, 1.0)
    return (1.0 - p) * (1.0 - u), p * (1.0 - u), u


def conj(a, b):
    """Unnormalised conjunctive core plus the conflict scalar."""
    ar, af, at = a
    br, bf, bt = b
    mr = ar * br + ar * bt + at * br
    mf = af * bf + af * bt + at * bf
    mt = at * bt
    k = ar * bf + af * br
    return mr, mf, mt, k


def ds(a, b):
    mr, mf, mt, k = conj(a, b)
    d = np.clip(1.0 - k, EPS, None)
    return mr / d, mf / d, mt / d


def yager(a, b):
    mr, mf, mt, k = conj(a, b)
    return mr, mf, mt + k


def dubois_prade(a, b):
    """Conflict -> union of the conflicting elements. On a binary frame that union is Theta."""
    mr, mf, mt, k = conj(a, b)
    return mr, mf, mt + k          # identical to Yager here, by construction


def pcr(a, b):
    """PCR5/PCR6 for two sources (they coincide at N=2)."""
    ar, af, at = a
    br, bf, bt = b
    mr, mf, mt, _ = conj(a, b)
    # partial conflict ar*bf goes back to {real} and {fake} proportionally to ar and bf
    d1 = np.clip(ar + bf, EPS, None)
    d2 = np.clip(af + br, EPS, None)
    mr = mr + ar * ar * bf / d1 + br * br * af / d2
    mf = mf + bf * bf * ar / d1 + af * af * br / d2
    return mr, mf, mt


def deng_entropy(m):
    mr, mf, mt = m
    def t(x, card):
        x = np.clip(x, EPS, None)
        return -x * np.log2(x / card)
    return t(mr, 1.0) + t(mf, 1.0) + t(mt, 3.0)      # 2^|A| - 1: singleton 1, Theta 3


def fold(ms, rule):
    f = ms[0]
    for m in ms[1:]:
        f = rule(f, m)
    return f


def deng_combine(ms):
    E = np.stack([deng_entropy(m) for m in ms], 0)             # (N, B)
    w = (1.0 / np.clip(E, EPS, None))
    w = w / w.sum(0, keepdims=True)
    avg = tuple(sum(w[i] * ms[i][j] for i in range(len(ms))) for j in range(3))
    return fold([avg] * len(ms), ds)


def prob(m):
    mr, mf, mt = m
    tot = np.clip(mr + mf + mt, EPS, None)
    return (mf + 0.5 * mt) / tot                                # uniform base rate on Theta


OPS = {
    "DS": lambda ms: fold(ms, ds),
    "Yager": lambda ms: fold(ms, yager),
    "Dubois-Prade": lambda ms: fold(ms, dubois_prade),
    "PCR6": lambda ms: fold(ms, pcr),
    "Deng-entropy": deng_combine,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", default="run1cft")
    ap.add_argument("--views", nargs="+", default=["semantic", "artifact", "fsvfm"])
    args = ap.parse_args()
    V = args.views

    def load(ds_):
        f = S23 / f"{args.arm}_{ds_}.csv"
        if not os.path.exists(f):
            return None
        d = pd.read_csv(f)
        d["v"] = video_id(d["key"])
        agg = {f"{k}_{v}": (f"{k}_{v}", "mean") for v in V for k in ("p", "u")}
        agg["y"] = ("label", "max")
        return d.groupby("v", as_index=False).agg(**agg)

    def ms_of(g):
        return [masses(g[f"p_{v}"].to_numpy(), g[f"u_{v}"].to_numpy()) for v in V]

    ff = load("FFpp_val")
    taus = {o: eer(ff["y"].to_numpy(), prob(OPS[o](ms_of(ff))))[1] for o in OPS}

    print(f"arm {args.arm} | views: {', '.join(V)}\n")
    print(f"{'operator':14s} " + " ".join(f"{d:>8s}" for d in OOD) +
          f" {'mean':>8s} {'vsP0':>8s} {'FPR':>7s}")
    mp0 = np.mean([P0[d] for d in OOD])
    res = {}
    for o in OPS:
        a, f_ = [], []
        for d_ in OOD:
            g = load(d_)
            dd = dashboard(g["y"].to_numpy(), prob(OPS[o](ms_of(g))), taus[o])
            a.append(dd["auroc"]); f_.append(dd["fpr_real_at_tau"])
        res[o] = (np.mean(a), np.mean(f_))
        print(f"{o:14s} " + " ".join(f"{x:>8.4f}" for x in a) +
              f" {np.mean(a):>8.4f} {np.mean(a)-mp0:>+8.4f} {np.mean(f_):>7.3f}")

    # the analytic claim, checked numerically rather than asserted
    g = load("CDFv2")
    py = prob(OPS["Yager"](ms_of(g)))
    pd_ = prob(OPS["Dubois-Prade"](ms_of(g)))
    print(f"\nDubois-Prade vs Yager, max |diff| on CDFv2: {np.abs(py-pd_).max():.2e}"
          f"  -> {'IDENTICAL on a binary frame, as predicted' if np.abs(py-pd_).max() < 1e-12 else 'DIFFER'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

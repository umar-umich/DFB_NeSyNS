#!/usr/bin/env python3
"""Vector-valued reliability: a per-view, PER-CHANNEL discount instead of one scalar.

    <discern_ext python> upgrade/channel_reliability.py --view pe_core_lN

THE LIMITATION BEING TESTED. PCR6+-R discounts a view by `e' = r e`, one scalar scaling BOTH
evidence channels. So a view is either trusted or not; it cannot be trusted when it says "fake" and
doubted when it says "real". Measured on development, the views are exactly that asymmetric -- the
temporal view separates reenactment at 0.609 and swaps at 0.357, off-manifold is the reverse at
0.296 and 0.481 -- and because the real videos are shared, that asymmetry lives entirely in the
FAKE channel.

THE PARAMETERISATION. Each view keeps its fitted scalar `w_v` and gains one asymmetry `rho_v`:

    e'_fake = w_v * rho_v * e_fake        e'_real = (w_v / rho_v) * e_real

`rho = 1` is exactly the deployed discount, so the deployed system is inside the search space and the
comparison cannot be won by the parameterisation alone. Six numbers instead of three.

WHY THIS IS NOT THE FOUR THINGS THAT ALREADY FAILED. Reliability controllers and CCR both made `w` a
FUNCTION OF THE INPUT, and CCR's shuffled control scored within 0.0003 of it -- whatever it gained
was capacity, not conditioning. This is a fixed parameter, not conditioned on anything, so there is
no association for a shuffled control to destroy. The matching control here is different and is run:
a RANDOM asymmetry of the same magnitude, which tests whether any off-centre split helps as much as
the fitted one.
"""
from __future__ import annotations

import argparse, itertools, json, sys
from pathlib import Path

import numpy as np

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(ROOT / "upgrade"), str(ROOT / "analysis/tbiom/pcr6p"),
                str(ROOT / "analysis/tbiom/grouped"), str(ROOT / "analysis/tbiom"),
                str(ROOT / "analysis/tbiom/temporal")]
import arms as AR               # noqa: E402
import ops_pcr6p as P           # noqa: E402
import ops as G                 # noqa: E402
import dsr_system as DS         # noqa: E402
from health import auroc        # noqa: E402
import final_eval as FE         # noqa: E402

OUT = ROOT / "analysis/upgrade/final"
VIEWS = ("T", "APP", "OM")
RHO = (0.5, 0.71, 1.0, 1.41, 2.0)
GUARD = 0.005


def fuse_channel(t, w, rho):
    """PCR6+ over per-channel-discounted opinions. Evidence order is (fake, genuine)."""
    ms = []
    for i, v in enumerate(VIEWS):
        o, _ = G.opinion_from_pu(t[f"p_{v}"].to_numpy(), t[f"u_{v}"].to_numpy())
        e = G.evidence_of(o)                       # (n, 2) = (fake, genuine)
        scale = np.array([w[i] * rho[i], w[i] / rho[i]], dtype=float)
        ms.append(G.opinion_of_evidence(e * scale))
    return P.fuse(np.stack(ms, 1), True)


def score(t, w, rho):
    p = P.p_of(fuse_channel(t, w, rho))
    y = t.y.to_numpy()
    m = FE.reen_mask(t)
    m = m.to_numpy() if hasattr(m, "to_numpy") else m
    re = auroc(y[m], p[m]) if m.sum() > 10 and len(np.unique(y[m])) > 1 else float("nan")
    return auroc(y, p), re


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="pe_core_lN"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev_t, _dp, _tests = DS.load("vjepa2_lora_B_seed42")
    op = FE.opinions(a.view, a.seed, "cuda", ("VALmix", "VALmixV2", "FFpp_val"))
    t = FE.swap_app(dev_t, op, "development")

    folds = list(DS.splits(t))
    ws = {k: AR.fit_w(Fd, "PCR6P-R") for k, Fd, _S in folds}     # the deployed scalar triple per fold

    rows = []
    for rho in itertools.product(RHO, repeat=3):
        ov, re = [], []
        for k, _Fd, Sd in folds:
            o, r = score(Sd, ws[k], rho)
            ov.append(o)
            if not np.isnan(r):
                re.append(r)
        rows.append({"rho": list(rho), "overall": float(np.mean(ov)),
                     "reenactment": float(np.mean(re)) if re else None})
    base = next(r for r in rows if r["rho"] == [1.0, 1.0, 1.0])

    # CONTROL: random asymmetries of the same magnitude, to test whether any off-centre split helps
    rng = np.random.default_rng(0)
    ctrl = []
    for _ in range(20):
        rho = tuple(rng.choice(RHO) for _ in range(3))
        ov, re = [], []
        for k, _Fd, Sd in folds:
            o, r = score(Sd, ws[k], rho)
            ov.append(o)
            if not np.isnan(r):
                re.append(r)
        ctrl.append({"rho": list(rho), "overall": float(np.mean(ov)),
                     "reenactment": float(np.mean(re)) if re else None})

    ok = [r for r in rows if r["overall"] >= base["overall"] - GUARD]
    best = max(ok, key=lambda r: r["reenactment"]) if ok else base
    cbest = max(ctrl, key=lambda r: r["reenactment"])

    rows.sort(key=lambda r: -r["reenactment"])
    print(f"deployed (rho = 1,1,1):  overall {base['overall']:.4f}  reenactment {base['reenactment']:.4f}\n")
    print(f"{'rhoT':>6s} {'rhoAPP':>7s} {'rhoOM':>6s} {'overall':>9s} {'reenact':>9s}")
    for r in rows[:8]:
        flag = "" if r["overall"] >= base["overall"] - GUARD else "  (guard)"
        print(f"{r['rho'][0]:6.2f} {r['rho'][1]:7.2f} {r['rho'][2]:6.2f} "
              f"{r['overall']:9.4f} {r['reenactment']:9.4f}{flag}")
    print(f"\nselected (guard-respecting): rho = {best['rho']}  "
          f"overall {best['overall']:.4f} ({best['overall']-base['overall']:+.4f})  "
          f"reenactment {best['reenactment']:.4f} ({best['reenactment']-base['reenactment']:+.4f})")
    print(f"random-asymmetry control, best of 20: reenactment {cbest['reenactment']:.4f} "
          f"({cbest['reenactment']-base['reenactment']:+.4f}) at rho = {cbest['rho']}")
    print("\nIf the control matches the fitted asymmetry, the gain is the extra parameters and not "
          "the direction -- the same lesson CCR's shuffled control taught.")
    json.dump({"view": a.view, "rho_grid": list(RHO), "guard": GUARD, "baseline": base,
               "selected": best, "control_best": cbest, "all": rows, "control": ctrl},
              open(OUT / f"channel_reliability_{a.view}.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

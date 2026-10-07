#!/usr/bin/env python3
"""Phase 0 algebra audit: are our opinions valid, and does discounting do what the brief says?

    <discern_ext python> upgrade/fusion/opinion_algebra.py

THE CHECK THAT MATTERS MOST. A binary opinion is (b_G, b_F, u) with b_G + b_F + u = 1 and every
mass non-negative. If a calibration holds p fixed and multiplies u, then

    b_F = p - u/2,   b_G = 1 - p - u/2,   so validity needs   0 < u <= 2 min(p, 1-p).

An adapted view sitting at p = 0.99 therefore admits u <= 0.02. The vacuity scaling used earlier in
this project multiplied u by up to 4 with only a clip to 1.0 and no validity check, so it could
produce NEGATIVE belief mass on confident videos. This script counts how often that happened,
measures what repair costs, and reports it rather than quietly rewriting the earlier numbers.

Evidence discounting is audited against the brief's closed form:
    D = r(1-u) + u,  u' = u/D,  b'_k = r b_k / D,  p'_F = (r e_F + 1) / (r (e_F + e_G) + 2)
with r = 1 the identity and r = 0 the vacuous opinion (0, 0, 1), p_F = 0.5.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(ROOT / "upgrade"), str(ROOT / "analysis/tbiom/grouped"),
                str(ROOT / "analysis/tbiom/pcr6p"), str(ROOT / "analysis/tbiom"),
                str(ROOT / "analysis/tbiom/temporal")]
import ops as G  # noqa: E402

VIEWS = ("T", "APP", "OM")


def masses_from_p_u(p, u):
    """The decomposition the brief gives: b_F = p - u/2, b_G = 1 - p - u/2."""
    return 1.0 - p - u / 2.0, p - u / 2.0


def validity(p, u):
    bG, bF = masses_from_p_u(p, u)
    return bG, bF, (bG >= -1e-12) & (bF >= -1e-12)


def audit_vacuity_scaling(p, u, mults=(1.25, 1.5, 2.0, 3.0, 4.0)):
    """What the project's u-scaling actually produced, per multiplier."""
    rows = []
    for m in mults:
        u2 = np.clip(u * m, 1e-6, 1.0)
        bG, bF, ok = validity(p, u2)
        umax = 2.0 * np.minimum(p, 1.0 - p)
        rows.append({"multiplier": m,
                     "invalid_fraction": float((~ok).mean()),
                     "n_invalid": int((~ok).sum()),
                     "worst_negative_mass": float(min(bG.min(), bF.min())),
                     "mean_u_requested": float(u2.mean()),
                     "mean_u_max_allowed": float(umax.mean()),
                     "fraction_exceeding_cap": float((u2 > umax + 1e-12).mean())})
    return rows


def audit_discount(rng, n=20000):
    """Closed form against the repository's own implementation, plus the two fixed points."""
    p = rng.uniform(0.01, 0.99, n); u = rng.uniform(1e-3, 1.0, n)
    u = np.minimum(u, 2.0 * np.minimum(p, 1 - p) * 0.999)        # start from VALID opinions only
    o, _ = G.opinion_from_pu(p, u)
    worst_id, worst_vac, worst_form = 0.0, 0.0, 0.0
    for r in (1.0, 0.75, 0.5, 0.25, 0.1):
        got = G.discount(o, r)
        D = r * (1 - u) + u
        exp_u = u / D
        e = G.evidence_of(o)
        exp_p = (r * e[..., 0] + 1.0) / (r * (e[..., 0] + e[..., 1]) + 2.0)
        worst_form = max(worst_form, float(np.abs(got[..., 2] - exp_u).max()),
                         float(np.abs(G.p_of(got) - exp_p).max()))
        if r == 1.0:
            worst_id = float(np.abs(got - o).max())
    vac = G.discount(o, 0.0)
    worst_vac = max(float(np.abs(vac[..., 2] - 1.0).max()), float(np.abs(G.p_of(vac) - 0.5).max()))
    return {"r1_identity_max_abs_err": worst_id,
            "r0_vacuous_max_abs_err": worst_vac,
            "closed_form_max_abs_err": worst_form}


def main() -> int:
    run = (ROOT / "outputs/fusion_experiments/CURRENT_RUN").read_text().strip()
    out = ROOT / "outputs/fusion_experiments" / run
    rng = np.random.default_rng(0)

    # --- real opinions from the locked B0 row
    import dsr_system as DS, final_eval as FE
    dev_t, _dp, _tests = DS.load("vjepa2_lora_B_seed42")
    op = FE.opinions("pe_core_lN", 0, "cuda", ("VALmix", "VALmixV2", "FFpp_val"))
    t = FE.swap_app(dev_t, op, "development")

    res = {"run_id": run, "views": {}, "discount": audit_discount(rng)}
    print("=== evidence discounting against the brief's closed form ===")
    for k, v in res["discount"].items():
        print(f"  {k:32s} {v:.3e}")

    print("\n=== are the BASE opinions valid? ===")
    for v in VIEWS:
        p, u = t[f"p_{v}"].to_numpy(), t[f"u_{v}"].to_numpy()
        bG, bF, ok = validity(p, u)
        res["views"][v] = {"n": int(len(p)), "base_invalid": int((~ok).sum()),
                           "mean_p": float(p.mean()), "mean_u": float(u.mean()),
                           "mean_u_cap": float((2 * np.minimum(p, 1 - p)).mean()),
                           "vacuity_scaling": audit_vacuity_scaling(p, u)}
        print(f"  {v:4s} n={len(p):6d}  invalid={int((~ok).sum()):5d}  "
              f"mean u={u.mean():.4f}  mean cap={2*np.minimum(p,1-p).mean():.4f}")

    print("\n=== what the project's VACUITY SCALING did to validity ===")
    print(f"{'view':5s} {'mult':>5s} {'invalid':>9s} {'frac':>8s} {'worst mass':>11s} {'over cap':>9s}")
    for v in VIEWS:
        for r in res["views"][v]["vacuity_scaling"]:
            print(f"{v:5s} {r['multiplier']:5.2f} {r['n_invalid']:9d} {r['invalid_fraction']:8.4f} "
                  f"{r['worst_negative_mass']:11.4f} {r['fraction_exceeding_cap']:9.4f}")

    json.dump(res, open(out / "phase0_algebra.json", "w"), indent=1)
    print(f"\nwrote {out}/phase0_algebra.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

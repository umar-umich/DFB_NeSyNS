#!/usr/bin/env python3
"""Does it matter HOW a view is quietened, holding the amount fixed?

    <discern_ext python> upgrade/matched_magnitude.py --view pe_core_lN

Specified in analysis/upgrade/FUSION-PROBLEM-REPORT.md section 6c before this ran.

Six interventions on reliability have failed and all six act on EVIDENCE; the one that passed a
pre-registered test acts on VACUITY with the probability preserved. That could be about the FORM of
the discount or merely about the AMOUNT of quietening each happened to apply. This separates them.

  EVIDENCE arm   e' = r e        -> u' = 2 / (r (S - 2) + 2), and p moves toward 0.5
  VACUITY  arm   u' = m u                                      p is untouched

The two are matched on MEAN VACUITY, not on parameter: for each target `m` the vacuity arm uses it
directly and `r` is solved numerically so the evidence arm reaches the same mean u on that fold.
Matching on the parameter instead would compare two different amounts of quietening and could say
nothing about mechanism.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

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
MULTS = (1.25, 1.5, 2.0, 3.0, 4.0)


def u_after_evidence(u, r):
    """u = 2/S, so S = 2/u and e_total = S - 2. Scaling evidence by r gives this new vacuity."""
    S = 2.0 / np.clip(u, 1e-9, 1.0)
    return 2.0 / (r * (S - 2.0) + 2.0)


def solve_r(u, target_mean):
    """The evidence scale whose MEAN vacuity equals the vacuity arm's."""
    f = lambda r: u_after_evidence(u, r).mean() - target_mean
    lo, hi = 1e-4, 1.0
    if f(lo) * f(hi) > 0:
        return None
    return float(brentq(f, lo, hi, xtol=1e-6))


def apply_vacuity(t, views, m):
    tt = t.copy()
    for v in views:
        tt[f"u_{v}"] = np.clip(tt[f"u_{v}"].to_numpy() * m, 1e-6, 1.0)
    return tt


def apply_evidence(t, views, rs):
    """e' = r e per view, recomputed back into (p, u) -- p moves, which is the point."""
    tt = t.copy()
    for v, r in zip(views, rs):
        o, _ = G.opinion_from_pu(tt[f"p_{v}"].to_numpy(), tt[f"u_{v}"].to_numpy())
        o2 = G.opinion_of_evidence(G.evidence_of(o) * r)
        tt[f"p_{v}"] = G.p_of(o2)
        tt[f"u_{v}"] = o2[..., 2]
    return tt


def evaluate(t, folds_src):
    ov, re = [], []
    for _k, Fd, Sd in DS.splits(t):
        w = AR.fit_w(Fd, "PCR6P-R")
        p = P.p_of(AR.fuse(Sd, "PCR6P-R", w)); y = Sd.y.to_numpy()
        ov.append(auroc(y, p))
        m = FE.reen_mask(Sd)
        m = m.to_numpy() if hasattr(m, "to_numpy") else m
        if m.sum() > 10 and len(np.unique(y[m])) > 1:
            re.append(auroc(y[m], p[m]))
    return float(np.mean(ov)), (float(np.mean(re)) if re else float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="pe_core_lN"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev_t, _dp, _tests = DS.load("vjepa2_lora_B_seed42")
    op = FE.opinions(a.view, a.seed, "cuda", ("VALmix", "VALmixV2", "FFpp_val"))
    base = FE.swap_app(dev_t, op, "development")
    b_ov, b_re = evaluate(base, None)
    print(f"baseline: overall {b_ov:.4f}  reenactment {b_re:.4f}\n")

    rows = []
    targets = [("APP",), ("T",), ("OM",), VIEWS]
    for views in targets:
        for m in MULTS:
            tv = apply_vacuity(base, views, m)
            # match on MEAN vacuity, per view, on the full development table
            rs, ok = [], True
            for v in views:
                u = base[f"u_{v}"].to_numpy()
                tgt = np.clip(u * m, 1e-6, 1.0).mean()
                r = solve_r(u, tgt)
                if r is None:
                    ok = False; break
                rs.append(r)
            if not ok:
                continue
            te = apply_evidence(base, views, rs)
            v_ov, v_re = evaluate(tv, None)
            e_ov, e_re = evaluate(te, None)
            mu_v = np.mean([np.clip(base[f"u_{v}"].to_numpy()*m,1e-6,1).mean() for v in views])
            mu_e = np.mean([u_after_evidence(base[f"u_{v}"].to_numpy(), r).mean() for v, r in zip(views, rs)])
            rows.append({"views": list(views), "m": m, "r": rs,
                         "mean_u_vacuity_arm": float(mu_v), "mean_u_evidence_arm": float(mu_e),
                         "vac_overall": v_ov, "vac_reen": v_re,
                         "ev_overall": e_ov, "ev_reen": e_re})
            tag = "+".join(views)
            print(f"{tag:14s} m={m:4.2f}  mean u matched {mu_v:.4f}/{mu_e:.4f}   "
                  f"overall vac {v_ov:.4f} ev {e_ov:.4f} ({v_ov-e_ov:+.4f})   "
                  f"reen vac {v_re:.4f} ev {e_re:.4f} ({v_re-e_re:+.4f})", flush=True)

    d_ov = [r["vac_overall"] - r["ev_overall"] for r in rows]
    d_re = [r["vac_reen"] - r["ev_reen"] for r in rows]
    wins_ov = sum(d > 0 for d in d_ov); wins_re = sum(d > 0 for d in d_re)
    print(f"\nvacuity arm beats evidence arm on OVERALL in {wins_ov}/{len(rows)} cells, "
          f"mean delta {np.mean(d_ov):+.4f}")
    print(f"vacuity arm beats evidence arm on REENACTMENT in {wins_re}/{len(rows)} cells, "
          f"mean delta {np.mean(d_re):+.4f}")
    json.dump({"view": a.view, "baseline": {"overall": b_ov, "reenactment": b_re},
               "multipliers": list(MULTS), "rows": rows,
               "vacuity_wins_overall": f"{wins_ov}/{len(rows)}",
               "vacuity_wins_reenactment": f"{wins_re}/{len(rows)}",
               "mean_delta_overall": float(np.mean(d_ov)),
               "mean_delta_reenactment": float(np.mean(d_re))},
              open(OUT / f"matched_magnitude_{a.view}.json", "w"), indent=1)
    print(f"wrote {OUT}/matched_magnitude_{a.view}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

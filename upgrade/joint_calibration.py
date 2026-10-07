#!/usr/bin/env python3
"""Per-view vacuity calibration, fitted JOINTLY for all three views.

    <discern_ext python> upgrade/joint_calibration.py --view pe_core_lN --objective reenactment

WHY THIS IS NOT THE TRUST WEIGHT AGAIN. PCR6+-R's reliability discount is already evidence scaling,
`e' = r e`, so it is a per-view magnitude knob. Three things make this a different operator rather
than a second copy of it:

  1. It is p-PRESERVING. Evidence scaling moves a view's probability toward 0.5 as well as raising
     its vacuity; scaling vacuity alone leaves p bit-for-bit unchanged, so a view's RANKING and its
     standalone AUROC cannot move. Only its loudness changes.
  2. The weight grid is coarse and CONSTRAINED -- five points with `max(w) == 1`, so at least one
     view is always pinned at full volume and only relative scales are reachable. A calibration can
     quieten every view at once, which no weight triple can express.
  3. It is fitted here against a stated objective, including the reenactment one, rather than only
     the overall development score.

The weights are refitted inside every calibration candidate, so the comparison is never calibration
against weights -- it is weights alone against weights plus calibration.

HONEST ABOUT OVERLAP: because both act on magnitude, part of any gain here is reachable by a finer
weight grid. The report says so, and the single-view result (scale 2.0 on appearance) is the one
that already passed a pre-registered rule.
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
import dsr_system as DS         # noqa: E402
from health import auroc        # noqa: E402
import final_eval as FE         # noqa: E402

OUT = ROOT / "analysis/upgrade/final"
ARM = "PCR6P-R"
VIEWS = ("T", "APP", "OM")
GRID = (0.5, 1.0, 2.0, 4.0)
GUARD = 0.005


def apply_scales(t, s):
    tt = t.copy()
    for v, k in zip(VIEWS, s):
        if k != 1.0:
            tt[f"u_{v}"] = np.clip(tt[f"u_{v}"].to_numpy() * k, 1e-6, 1.0)
    return tt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="pe_core_lN")
    ap.add_argument("--objective", default="reenactment", choices=["overall", "reenactment"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    dev_t, _dp, tests = DS.load("vjepa2_lora_B_seed42")
    op = FE.opinions(a.view, a.seed, dev, ("VALmix", "VALmixV2", "FFpp_val"))
    base_t = FE.swap_app(dev_t, op, "development")

    rows = []
    for s in itertools.product(GRID, repeat=3):
        tt = apply_scales(base_t, s)
        ov, re = [], []
        for _k, Fd, Sd in DS.splits(tt):
            w = AR.fit_w(Fd, ARM)                      # weights REFITTED inside every candidate
            p = P.p_of(AR.fuse(Sd, ARM, w)); y = Sd.y.to_numpy()
            ov.append(auroc(y, p))
            m = FE.reen_mask(Sd)
            if m.sum() > 10 and len(np.unique(y[m])) > 1:
                re.append(auroc(y[m], p[m]))
        rows.append({"scales": list(s), "dev_overall": float(np.mean(ov)),
                     "dev_reenactment": float(np.mean(re)) if re else None})
    base = next(r for r in rows if r["scales"] == [1.0, 1.0, 1.0])
    ok = [r for r in rows if r["dev_overall"] >= base["dev_overall"] - GUARD]
    key = "dev_reenactment" if a.objective == "reenactment" else "dev_overall"
    best = max(ok, key=lambda r: r[key]) if ok else base

    rows.sort(key=lambda r: -r[key])
    print(f"baseline (1,1,1): overall {base['dev_overall']:.4f}  reenactment {base['dev_reenactment']:.4f}")
    print(f"guard: overall >= {base['dev_overall'] - GUARD:.4f}; {len(ok)} of {len(rows)} candidates qualify\n")
    print(f"{'T':>5s} {'APP':>5s} {'OM':>5s} {'overall':>9s} {'reenact':>9s}")
    for r in rows[:10]:
        g = "" if r["dev_overall"] >= base["dev_overall"] - GUARD else "  (guard)"
        print(f"{r['scales'][0]:5.1f} {r['scales'][1]:5.1f} {r['scales'][2]:5.1f} "
              f"{r['dev_overall']:9.4f} {r['dev_reenactment']:9.4f}{g}")
    print(f"\nselected: T={best['scales'][0]} APP={best['scales'][1]} OM={best['scales'][2]}  "
          f"overall {best['dev_overall']:.4f} ({best['dev_overall']-base['dev_overall']:+.4f})  "
          f"reenactment {best['dev_reenactment']:.4f} ({best['dev_reenactment']-base['dev_reenactment']:+.4f})")
    json.dump({"view": a.view, "objective": a.objective, "grid": list(GRID), "guard": GUARD,
               "baseline": base, "selected": best, "all": rows},
              open(OUT / f"joint_calibration_{a.view}_{a.objective}.json", "w"), indent=1)
    print(f"wrote {OUT}/joint_calibration_{a.view}_{a.objective}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

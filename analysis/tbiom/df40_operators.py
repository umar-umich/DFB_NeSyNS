#!/usr/bin/env python3
"""The combination-rule study extended to every DF40 generator.

    OMP_NUM_THREADS=8 python analysis/tbiom/df40_operators.py

`EXP_FUSION_OPERATORS.md` compares the seven rules on development and the ten test sets. This applies the same
fourteen arms to the 41 DF40 generator sets, so an alternative to Dempster's rule can be chosen on cross-manipulation
evidence too, and not only on the sets the deployed rule was already selected against.

Nothing is trained here: an operator is a closed-form combination of three cached view opinions. The only fitted
quantity is the three-view reliability weight, a grid search on the fitting fold of each split, exactly as in the
report this extends. Every arm sees identical opinions, so the comparison is decided by the operator alone.
"""
from __future__ import annotations
import itertools, json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "temporal")]
import exp1_fusion_audit as A  # noqa: E402
import dsr_system as D  # noqa: E402
import finish_line as F  # noqa: E402
import df40_analysis as X  # noqa: E402
from fusion_operators_v2 import OPS, GRID, masses, combine, p_of  # noqa: E402

R, LOG = Path("tbiom/reports"), Path("logs/tbiom/df40")


def main():
    t0 = time.time()
    dev = D.load("vjepa2_lora_B_seed42")[0]
    wgrid = [w for w in itertools.product(GRID, repeat=3) if max(w) == 1.0]
    meta = json.load(open(LOG / "meta.json"))
    names = [n for n in meta if (X.S23 / f"ours_{n}.csv").exists()]
    tabs = {n: X.ours_table(n) for n in names}
    print(f"{len(tabs)} generator sets loaded in {time.time() - t0:.0f}s", flush=True)

    fits = []                                                  # per split: {arm: weight vector or None}
    for k, Fd, _Sd in D.splits(dev):
        clF = A.cells(Fd); f = {}
        for op in OPS:
            f[op] = None
            f[f"{op}-R"] = max(wgrid, key=lambda w: A.v31(p_of(combine(masses(Fd, w), op)), clF)[0])
        fits.append(f)
        print(f"  split {k} weights fitted ({time.time() - t0:.0f}s)", flush=True)

    res = {}
    for n, t in tabs.items():
        y = t.y.to_numpy(); r = {}
        for arm in fits[0]:
            op = arm[:-2] if arm.endswith("-R") else arm
            a = [A.fast_auc(s[y == 0], s[y == 1]) for s in
                 (p_of(combine(masses(t, f[arm]), op)) for f in fits)]
            r[arm] = float(np.mean(a))
        res[n] = r
    json.dump(res, open(R / "EXP_DF40_OPERATORS.json", "w"), indent=1, default=float)
    write(res, meta, time.time() - t0)


def write(res, meta, secs):
    arms = list(next(iter(res.values())))
    CELLS = (("face swap", "cdf"), ("face swap", "ff"), ("reenactment", "cdf"), ("reenactment", "ff"))
    sel = lambda fam, src: [res[n] for n in res if meta[n]["family"] == fam and meta[n]["source"] == src]
    m = lambda rs, a: float(np.mean([r[a] for r in rs]))
    L = ["# Combination rules on every DF40 generator", "", __doc__.split("\n\n", 2)[2].strip(), "",
         f"Wall clock for the whole study: **{secs / 60:.1f} minutes** on CPU, cached opinions, no GPU.", "",
         "| operator | swap, CDF | swap, FF++ | reenactment, CDF | reenactment, FF++ | **all 41** | Δ vs Dempster-R |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    allr = list(res.values())
    base = m(allr, "Dempster-R")
    for a in arms:
        L.append(f"| {a} | " + " | ".join(f"{m(sel(*c), a):.4f}" for c in CELLS)
                 + f" | **{m(allr, a):.4f}** | {m(allr, a) - base:+.4f} |")
    alt = [a for a in arms if not a.startswith("Dempster")]
    best = max(alt, key=lambda a: m(allr, a))
    ree = sel("reenactment", "cdf") + sel("reenactment", "ff")
    bestr = max(alt, key=lambda a: m(ree, a))
    L += ["", f"Best alternative to the deployed rule over all 41 generator sets: **{best}** ({m(allr, best):.4f}, "
          f"{m(allr, best) - base:+.4f}). On the 22 reenactment sets alone it is **{bestr}** ({m(ree, bestr):.4f} "
          f"against Dempster-R's {m(ree, 'Dempster-R'):.4f}).", "",
          "DF40 played no part in fitting: the reliability weights come from the development folds, and every arm is "
          "applied to these sets unchanged. The ten-set and development view of the same fourteen arms is in "
          "`EXP_FUSION_OPERATORS.md`; sections 6 and 6a of `DETAILED_RESULTS.md` carry it per test set."]
    (R / "EXP_DF40_OPERATORS.md").write_text("\n".join(L) + "\n")
    print("wrote", R / "EXP_DF40_OPERATORS.md", f"in {secs / 60:.1f} min")


if __name__ == "__main__":
    main()

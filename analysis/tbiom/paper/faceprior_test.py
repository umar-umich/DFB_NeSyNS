#!/usr/bin/env python3
"""Can the face-prior view replace (or join) the appearance view?

    OMP_NUM_THREADS=8 python analysis/tbiom/paper/faceprior_test.py

The spatial checkpoint carries a fourth evidential head -- a face-prior branch distilled from FS-VFM (our
`studentA_preserve_seed42` student) -- whose opinion is cached in every export as `p_fsvfm_raw`/`u_fsvfm_raw`
(ungated) and `p_fsvfm`/`u_fsvfm` (after the learned gate). It was excluded from the deployed three-view fusion.
This scores candidate view sets that include it, under the proposed PCR6+ operator, with reliability weights fitted
per fold by the same grid and objective as every other arm. Cached opinions only; no inference.
"""
from __future__ import annotations
import itertools, json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "pcr6p"), str(HERE.parent / "grouped"),
                str(HERE.parent), str(HERE.parent / "temporal")]
import ops_pcr6p as P  # noqa: E402
import ops as G  # noqa: E402
import exp1_fusion_audit as A  # noqa: E402
import dsr_system as D  # noqa: E402
import confirm as C  # noqa: E402

R, LOG = Path("tbiom/reports"), Path("logs/tbiom/df40")
GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
TESTS = ["CDFv2", "CDFv3", "DFD", "DFDC", "DFDCP", "DFEval24", "FFIW", "KoDF", "IDForge", "DFDM"]
CELLS = (("face swap", "cdf"), ("face swap", "ff"), ("reenactment", "cdf"), ("reenactment", "ff"))
SETS = {"T+A+O (deployed)": ("T", "APP", "OM"),
        "T+O+F": ("T", "OM", "FP"),
        "T+A+F": ("T", "APP", "FP"),
        "T+A+O+F": ("T", "APP", "OM", "FP"),
        "T+F": ("T", "FP"),
        "F only": ("FP",),
        "T+O": ("T", "OM")}


def masses(t, views, w):
    out = []
    for i, v in enumerate(views):
        o, _ = G.opinion_from_pu(t[f"p_{v}"].to_numpy(), t[f"u_{v}"].to_numpy())
        out.append(G.discount(o, w[i]))
    return np.stack(out, 1)


def fuse(t, views, w):
    m = masses(t, views, w)
    return m[:, 0, :] if len(views) == 1 else P.fuse(m, True)


def grid(n):
    return [w for w in itertools.product(GRID, repeat=n) if max(w) == 1.0]


def main():
    t0 = time.time()
    dev = C.dev_table("warmgate_e19", "vjepa2_lora_B_seed42")
    tests = D.load("vjepa2_lora_B_seed42")[2]
    meta = json.load(open(LOG / "meta.json"))
    d40 = {n: C.table(n, "ours", "vjepa2_lora_B_seed42") for n in meta
           if (C.X.S23 / f"ours_{n}.csv").exists()}
    assert "p_FP" in dev.columns, "face-prior opinion not present in the development table"
    res = {}
    for name, views in SETS.items():
        gr = grid(len(views))
        acc = {"dev": [], "reen": [], "tests": {tn: [] for tn in TESTS}, "df40": {n: [] for n in d40}, "w": []}
        for _k, Fd, Sd in D.splits(dev):
            clF, clS = A.cells(Fd), A.cells(Sd)
            w = max(gr, key=lambda ww: A.v31(P.p_of(fuse(Fd, views, ww)), clF)[0])
            acc["w"].append(list(w))
            tot, ty = A.v31(P.p_of(fuse(Sd, views, w)), clS)
            acc["dev"].append(tot); acc["reen"].append(ty["reenactment"])
            for tn in TESTS:
                T_ = tests[tn]; y = T_.y.to_numpy(); p = P.p_of(fuse(T_, views, w))
                acc["tests"][tn].append(A.fast_auc(p[y == 0], p[y == 1]))
            for n, tt in d40.items():
                y = tt.y.to_numpy(); p = P.p_of(fuse(tt, views, w))
                acc["df40"][n].append(A.fast_auc(p[y == 0], p[y == 1]))
        g = lambda n: float(np.mean(acc["df40"][n]))
        bal = float(np.mean([np.mean([g(n) for n in d40 if meta[n]["family"] == f and meta[n]["source"] == s])
                             for f, s in CELLS]))
        res[name] = {"views": list(views), "dev": float(np.mean(acc["dev"])), "reen": float(np.mean(acc["reen"])),
                     "tenset": float(np.mean([np.mean(acc["tests"][tn]) for tn in TESTS])), "df40_bal": bal,
                     "per_test": {tn: float(np.mean(acc["tests"][tn])) for tn in TESTS},
                     "per_df40": {n: g(n) for n in d40},
                     "w": acc["w"], "zero_frac": {v: float(np.mean([x[i] == 0.0 for x in acc["w"]]))
                                                  for i, v in enumerate(views)}}
        print(f"{name:20s} dev {res[name]['dev']:.4f}  reen {res[name]['reen']:.4f}  "
              f"10-set {res[name]['tenset']:.4f}  DF40 {bal:.4f}  zero-w {res[name]['zero_frac']} "
              f"({time.time() - t0:.0f}s)", flush=True)
    json.dump(res, open(R / "EXP_FACEPRIOR_VIEW.json", "w"), indent=1, default=float)
    print("wrote", R / "EXP_FACEPRIOR_VIEW.json")


if __name__ == "__main__":
    main()

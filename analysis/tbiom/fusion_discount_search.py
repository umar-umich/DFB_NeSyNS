#!/usr/bin/env python3
"""Dev-selected view reliability for DS fusion (evidence discounting), no retraining.

Each view's Dirichlet evidence is scaled by a weight w in [0, 1] before the DS replay (alpha' = 1 + w (alpha - 1)).
Weights are chosen ONLY on the development data with the v3.1 checkpoint score (VALmix + VALmix-v2 cells, FF++ val
reals for DF40 cells); test sets are then read once with the chosen weights.
    python analysis/tbiom/fusion_discount_search.py --tag run1auxedl_e06 --test_tag run1auxedl_v2e06
"""
import argparse, itertools, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import symbolic_overlay as SO  # noqa: E402
import select_checkpoint_v3 as V3  # noqa: E402
from corrected_table_v3 import with_subset  # noqa: E402
from health import auroc  # noqa: E402

K = 2.0
TEST = ["CDFv2", "CDFv3", "DFD", "DFDC", "DFDCP", "DFEval24", "FFIW", "KoDF", "IDForge", "DFDM"]


def discount(g, w):
    g = g.copy()
    for b, wb in w.items():
        S = K / np.clip(g[f"u_{b}"].to_numpy(), 1e-9, None); p = g[f"p_{b}"].to_numpy()
        a = np.stack([(1 - p) * S, p * S], 1); a = 1 + wb * (a - 1); S2 = a.sum(1)
        g[f"p_{b}"], g[f"u_{b}"] = a[:, 1] / S2, K / S2
    g["s"] = SO.fuse(g); return g


def dev_score(tabs, w):
    vm, v2, ff = (discount(t, w) for t in tabs)
    allv = pd.concat([vm, v2], ignore_index=True); ffreal = ff[ff.y == 0]; types = {}
    for t, srcs in V3.TYPES.items():
        vals = []
        for src, subs in srcs.items():
            s = allv[allv.source == src]
            fake = s[s.y == 1] if subs == "*" else s[(s.y == 1) & s.subset.isin(subs)]
            real = ffreal if src == "DF40train" else s[s.y == 0]
            x = pd.concat([real, fake]); vals.append(auroc(x.y.to_numpy(), x.s.to_numpy()))
        types[t] = float(np.mean(vals))
    return float(np.mean(list(types.values()))), types


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--test_tag", required=True)
    ap.add_argument("--grid", type=float, nargs="+", default=[0, .25, .5, .75, 1])
    a = ap.parse_args(); O = V3.OUT
    tabs = [V3.table(O / f"{a.tag}_{n}.csv") for n in ("VALmix", "VALmixV2", "FFpp_val")]
    res = []
    for ws in itertools.product(a.grid, repeat=3):
        if max(ws) < 1: continue                      # scale-equivalent to a grid point with a 1
        w = dict(zip(SO.BRANCHES, ws)); sc, ty = dev_score(tabs, w); res.append((sc, ws, ty))
    res.sort(key=lambda r: -r[0])
    base = next(r for r in res if r[1] == (1, 1, 1))
    print("dev (v3.1 score) top 5 and equal weights:")
    for sc, ws, ty in res[:5] + [base]:
        print(f"  w(sem,art,face)={ws}  score {sc:.4f}  " + " ".join(f"{k} {v:.4f}" for k, v in ty.items()))
    best = dict(zip(SO.BRANCHES, res[0][1]))
    print(f"\ntest AUROC (video), equal vs dev-selected {res[0][1]}:")
    for ds in TEST:
        g = with_subset(a.test_tag, ds)
        if g is None: continue
        e, s = discount(g, dict(zip(SO.BRANCHES, (1, 1, 1)))), discount(g, best)
        print(f"  {ds:9} {auroc(e.y.to_numpy(), e.s.to_numpy()):.4f} -> {auroc(s.y.to_numpy(), s.s.to_numpy()):.4f}")


if __name__ == "__main__":
    main()

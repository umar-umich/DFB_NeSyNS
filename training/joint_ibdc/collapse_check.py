#!/usr/bin/env python3
"""The collapse check: did joint training remove a view's contribution?

    <discern_ext python> training/joint_ibdc/collapse_check.py --run J1 --seed 0

The brief: "If any view with nonzero trust weight has a leave-one-out contribution on the development
pool that is at most zero in J1 but positive in J0, stop after J1 and report."

Contributions are measured the way the deployed system is evaluated, not the way it is trained: trust
weights are REFIT out of fold on the five source-grouped development folds, and a view's contribution
is the fused AUROC minus the fused AUROC with that view's weight set to zero, with no refitting in
between. PCR6+'s neutrality property is what makes that subtraction mean "remove the view": a source
at r = 0 is vacuous, and a vacuous source is exactly an absent one.

The development table's metadata -- video identity, source, subset, labels, fold grouping -- is taken
from the deployed table unchanged, and ONLY the six per-view opinion columns are replaced by the run's
own. Rebuilding the metadata would risk regrouping folds or cells and comparing two systems across
different partitions.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(ROOT / "analysis/tbiom/pcr6p"), str(ROOT / "analysis/tbiom/grouped"),
                str(ROOT / "analysis/tbiom"), str(ROOT / "analysis/tbiom/temporal"),
                str(Path(__file__).resolve().parent)]

import arms as AR                 # noqa: E402
import ops_pcr6p as P             # noqa: E402
import dsr_system as DS           # noqa: E402
import exp1_fusion_audit as A     # noqa: E402
import devtable as DT             # noqa: E402
import runs as R                  # noqa: E402
import fusion_torch as FT         # noqa: E402

VIEWS = ("T", "APP", "OM")
OUT = ROOT / "analysis/joint_ibdc"


@torch.no_grad()
def view_opinions(heads, DEV, dev):
    """Per-view VIDEO-level (p, u) for every development set, keyed by video."""
    app, om, t = heads
    out = {}
    for name, D in DEV.items():
        ea, eo = app(torch.from_numpy(D["app"]).to(dev)), om(torch.from_numpy(D["om"]).to(dev))
        oa = FT.opinion_of_evidence(R.to_fusion(ea))
        oo = FT.opinion_of_evidence(R.to_fusion(eo))
        inv = torch.from_numpy(D["inv"]).to(dev)
        n = len(D["order"])
        cnt = torch.bincount(inv, minlength=n).clamp_min(1).unsqueeze(1).float()
        va = torch.zeros(n, 3, device=dev).index_add_(0, inv, oa) / cnt     # mean opinion per video
        vo = torch.zeros(n, 3, device=dev).index_add_(0, inv, oo) / cnt
        vt = FT.opinion_of_evidence(R.to_fusion(t(torch.from_numpy(D["t"]).to(dev))))
        for v, a_, o_, t_ in zip(D["order"], va.cpu().numpy(), vo.cpu().numpy(), vt.cpu().numpy()):
            out[str(v)] = {"APP": a_, "OM": o_, "T": t_}
    return out


def table_with(op):
    """The deployed development table with only its six opinion columns replaced."""
    t = DT.C.dev_table(DT.SPATIAL, DT.BRANCH).copy()
    miss = [v for v in t.v if v not in op]
    if miss:
        raise KeyError(f"{len(miss)} development videos have no opinion, e.g. {miss[:3]}")
    for view in VIEWS:
        o = np.stack([op[v][view] for v in t.v])
        t[f"p_{view}"] = np.clip(o[:, 0] + o[:, 2] / 2, 0.0, 1.0)
        t[f"u_{view}"] = o[:, 2]
    return t


def loo(t, tag):
    rows, agg = [], {v: [] for v in VIEWS}
    for k, Fd, Sd in DS.splits(t):
        w = AR.fit_w(Fd, "PCR6P-R")
        p = P.p_of(AR.fuse(Sd, "PCR6P-R", w))
        y = Sd.y.to_numpy(); real, fake = y == 0, y == 1
        full = A.fast_auc(p[real], p[fake])
        row = {"split": k, "w": list(w), "auroc": full, "loo": {}}
        for i, v in enumerate(VIEWS):
            if w[i] == 0.0:
                row["loo"][v] = 0.0
            else:
                w0 = list(w); w0[i] = 0.0
                p0 = P.p_of(AR.fuse(Sd, "PCR6P-R", w0))
                row["loo"][v] = float(full - A.fast_auc(p0[real], p0[fake]))
            agg[v].append(row["loo"][v])
        rows.append(row)
        print(f"  {tag} split {k}: w={w} AUROC {full:.4f} "
              + " ".join(f"{v} {row['loo'][v]:+.4f}" for v in VIEWS), flush=True)
    return {"per_split": rows, "mean_auroc": float(np.mean([r["auroc"] for r in rows])),
            "loo_mean": {v: float(np.mean(agg[v])) for v in VIEWS},
            "nonzero_folds": {v: int(sum(1 for r in rows if r["w"][i] > 0))
                              for i, v in enumerate(VIEWS)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="J1")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    DEV = R.load_dev()

    j0 = R.load_deployed_heads(dev)
    res0 = loo(table_with(view_opinions(j0, DEV, dev)), "J0")

    ck = OUT / f"run_{a.run}_s{a.seed}.pt"
    if not ck.exists():
        print(f"{ck} not found; run {a.run} first"); return 2
    st = torch.load(ck, map_location=dev)
    j1 = R.load_deployed_heads(dev)
    j1[0].load_state_dict(st["app"]); j1[1].load_state_dict(st["om"]); j1[2].load_state_dict(st["t"])
    for h in j1:
        h.eval()
    print(f"\n{a.run} seed {a.seed}, development-selected epoch {st['epoch']} "
          f"(dev v3.1 {st['dev_v31']:.4f})")
    res1 = loo(table_with(view_opinions(j1, DEV, dev)), a.run)

    fired, detail = [], {}
    for v in VIEWS:
        p0, p1 = res0["loo_mean"][v], res1["loo_mean"][v]
        nz = res1["nonzero_folds"][v] > 0
        trip = bool(nz and p0 > 0 and p1 <= 0)
        detail[v] = {"J0": p0, a.run: p1,
                     "nonzero_folds_in_run": res1["nonzero_folds"][v], "collapsed": trip}
        if trip:
            fired.append(v)
    out = {"check": "collapse after J1", "run": a.run, "seed": a.seed,
           "verdict": "COLLAPSE" if fired else "no collapse",
           "views_collapsed": fired, "per_view": detail,
           "J0": {k: v for k, v in res0.items() if k != "per_split"},
           a.run: {k: v for k, v in res1.items() if k != "per_split"},
           "J0_splits": res0["per_split"], f"{a.run}_splits": res1["per_split"]}
    json.dump(out, open(OUT / f"collapse_{a.run}_s{a.seed}.json", "w"), indent=1)
    print(f"\n  {'view':5s} {'J0 LOO':>9} {a.run + ' LOO':>9}  folds  collapsed")
    for v in VIEWS:
        d = detail[v]
        print(f"  {v:5s} {d['J0']:+9.4f} {d[a.run]:+9.4f} {d['nonzero_folds_in_run']:6d}  "
              f"{'YES' if d['collapsed'] else 'no'}")
    print(f"\n  fused development AUROC: J0 {res0['mean_auroc']:.4f}  "
          f"{a.run} {res1['mean_auroc']:.4f}")
    print(f"collapse check: {out['verdict']}"
          + (f" ({', '.join(fired)})" if fired else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

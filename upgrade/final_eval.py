#!/usr/bin/env python3
"""Part 8 tests: the ten-set table, threshold metrics and view complementarity for a final row.

    <discern_ext python> upgrade/final_eval.py --view pe_core_lN --label "PE-Core-L + LayerNorm"

ONE ROW PER CALL, and the test sets are touched once. Trust weights and the threshold are fitted on
the DEVELOPMENT folds and then applied unchanged -- nothing here reads a test set to choose
anything, which is the brief's rule and the reason these numbers mean what they claim.

The deployed table's metadata -- video identity, source, subset, labels, folds -- is reused unchanged
and only `p_APP` / `u_APP` are replaced, so the comparison is the spatial view and not a different
partition. A video missing a candidate opinion is an error, never a silent drop.

COMPLEMENTARITY. Every subset of the three views is scored on the same videos: each single view,
each pair, and the triple. A pair is scored by making the absent view VACUOUS (zero evidence, the
identity of the fusion) rather than by deleting its term, so the operator is unchanged and the
comparison isolates what the view contributes.
"""
from __future__ import annotations

import argparse, itertools, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(ROOT / "upgrade"), str(ROOT / "analysis/tbiom/pcr6p"),
                str(ROOT / "analysis/tbiom/grouped"), str(ROOT / "analysis/tbiom"),
                str(ROOT / "analysis/tbiom/temporal"), str(ROOT / "branches/ijepa")]
import arms as AR               # noqa: E402
import ops_pcr6p as P           # noqa: E402
import dsr_system as DS         # noqa: E402
import exp1_fusion_audit as A   # noqa: E402
from health import auroc, eer   # noqa: E402
from train_head import Head, load  # noqa: E402

HEADS = ROOT / "analysis/upgrade/heads"
OUT = ROOT / "analysis/upgrade/final"
VIEWS = ("T", "APP", "OM")
ARM = "PCR6P-R"
TEST_SETS = ["CDFv2", "CDFv3", "DFD", "DFDC", "DFDCP", "DFEval24", "FFIW", "KoDF", "IDForge", "DFDM"]
# Reenactment-specific slices that dsr_system already builds. They share videos with CDFv3 and KoDF,
# so they are kept OUT of the ten-set mean and reported separately -- counting a slice and its parent
# in one average would weight those corpora twice.
SLICES = ["CDFv3 reenactment", "KoDF first-order motion"]


@torch.no_grad()
def opinions(view, seed, dev, datasets):
    st = torch.load(HEADS / f"view_{view}_s{seed}.pt", map_location=dev)
    head = Head(st["dim"]).to(dev); head.load_state_dict(st["state"]); head.eval()
    out = {}
    for n in datasets:
        f = ROOT / f"logs/upgrade/feats/{view}_{n}.npz"
        if not f.exists():
            print(f"  no features for {view} on {n}; skipped"); continue
        D = load(view, n)
        ev = head(torch.from_numpy(D["x"]).to(dev))
        a = ev + 1.0; s = a.sum(1, keepdim=True)
        p = (a / s)[:, 1].float().cpu().numpy(); u = (2.0 / s.squeeze(1)).float().cpu().numpy()
        cnt = np.bincount(D["inv"], minlength=len(D["order"])).astype(float)
        pv = np.bincount(D["inv"], weights=p, minlength=len(D["order"])) / cnt
        uv = np.bincount(D["inv"], weights=u, minlength=len(D["order"])) / cnt
        for v, a_, b_ in zip(D["order"], pv, uv):
            out[str(v)] = (float(a_), float(b_))
    return out


def swap_app(t, op, name):
    t = t.copy()
    miss = [v for v in t.v if str(v) not in op]
    if miss:
        raise KeyError(f"{name}: {len(miss)} videos lack a candidate opinion, e.g. {miss[:3]}")
    t["p_APP"] = np.array([op[str(v)][0] for v in t.v])
    t["u_APP"] = np.array([op[str(v)][1] for v in t.v])
    return t


def subset_score(t, views, w):
    """Views outside `views` are made VACUOUS, which is the identity of the fusion -- the operator
    is unchanged, so the difference is what the view said rather than how many terms were combined."""
    tt = t.copy()
    for v in VIEWS:
        if v not in views:
            tt[f"p_{v}"] = 0.5
            tt[f"u_{v}"] = 1.0
    return P.p_of(AR.fuse(tt, ARM, w))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", required=True, help="feature/head prefix, e.g. pe_core_lN")
    ap.add_argument("--label", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    label = a.label or a.view
    dev_t, _dproc, tests = DS.load("vjepa2_lora_B_seed42")   # (dev, processed-dev, tests)

    devc = "cuda" if torch.cuda.is_available() else "cpu"
    dev_op = opinions(a.view, a.seed, devc, ("VALmix", "VALmixV2", "FFpp_val"))
    dev_sw = swap_app(dev_t, dev_op, "development")

    # weights and threshold FITTED ON DEVELOPMENT, then frozen
    folds = [Fd for _k, Fd, _S in DS.splits(dev_sw)]
    ws = [AR.fit_w(Fd, ARM) for Fd in folds]
    taus = []
    for w in ws:
        p = P.p_of(AR.fuse(dev_sw, ARM, w))
        m = (dev_sw.source == "FFpp_val").to_numpy() if "source" in dev_sw else np.ones(len(dev_sw), bool)
        if m.any():
            taus.append(eer(dev_sw.y.to_numpy()[m], p[m])[1])
    tau = float(np.mean(taus))
    print(f"{label}: weights fitted on development, tau={tau:.4f} (FF++ val EER)", flush=True)

    test_op = opinions(a.view, a.seed, devc, TEST_SETS)
    rows, comp, slices = {}, {}, {}
    for n in TEST_SETS + SLICES:
        if n not in tests:
            continue
        try:
            t = swap_app(tests[n], test_op, n)
        except KeyError as e:
            print(f"  {n}: {e}"); continue
        y = t.y.to_numpy()
        ps = np.mean([P.p_of(AR.fuse(t, ARM, w)) for w in ws], axis=0)
        own = eer(y, ps)[1]
        tgt = slices if n in SLICES else rows
        tgt[n] = {"videos": int(len(y)), "auroc": auroc(y, ps),
                   "fpr_at_dev_tau": float((ps[y == 0] >= tau).mean()),
                   "tpr_at_dev_tau": float((ps[y == 1] >= tau).mean()),
                   "fpr_at_own_eer": float((ps[y == 0] >= own).mean())}
        c = {}
        for r in (1, 2, 3):
            for sub in itertools.combinations(VIEWS, r):
                s = np.mean([subset_score(t, sub, w) for w in ws], axis=0)
                c["+".join(sub)] = auroc(y, s)
        comp[n] = c
        print(f"  {n:24s} AUROC {tgt[n]['auroc']:.4f}  FPR@tau {tgt[n]['fpr_at_dev_tau']:.4f}", flush=True)

    if rows:
        mean = float(np.mean([r["auroc"] for r in rows.values()]))
        print(f"\n  ten-set mean AUROC {mean:.4f}")
    json.dump({"view": a.view, "label": label, "seed": a.seed, "arm": ARM, "dev_tau": tau,
               "weights_per_fold": [list(map(float, w)) for w in ws],
               "tests": rows, "slices": slices, "complementarity": comp,
               "mean_auroc": float(np.mean([r["auroc"] for r in rows.values()])) if rows else None,
               "slice_note": ("the reenactment slices share videos with CDFv3 and KoDF and are "
                              "therefore excluded from the ten-set mean"),
               "note": ("Weights and threshold are fitted on the development folds and applied "
                        "unchanged; no test set is read to choose anything. Pairs are scored by "
                        "making the absent view vacuous, not by deleting its term.")},
              open(OUT / f"{a.view}_s{a.seed}.json", "w"), indent=1)
    print(f"wrote {OUT}/{a.view}_s{a.seed}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

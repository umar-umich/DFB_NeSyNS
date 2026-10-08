#!/usr/bin/env python3
"""Appendix backup, step 5: the six comparison arms on the frozen subspaces.

    <discern_ext python> upgrade/fusion/backup_arms.py

Every fixed-subspace arm shares the SAME downstream head, the same six native evidence inputs, the
same loss, optimiser and budget, and the same nested source-grouped protocol as Phase 2. Only the
8-dimensional feature mapping differs, so a difference between arms is the mapping.

  existing-F           F's own learned projector (extra parameters, reported separately)
  PCA                  ordinary PCA to 8 -- is dimensionality reduction enough?
  random               fixed random orthonormal 8-D inside PCA-32, three seeds, mean not best
  shuffled-pairs       guided fit with genuine partners from DIFFERENT source groups
  manipulation-only    leading directions of C_man with no nuisance denominator
  intervention-guided  the full generalized-eigenvector projection

S and B1-refit are carried as reference comparators from the main run.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(ROOT / "upgrade"), str(ROOT / "upgrade/fusion"),
                str(ROOT / "analysis/tbiom/pcr6p"), str(ROOT / "analysis/tbiom/grouped"),
                str(ROOT / "analysis/tbiom"), str(ROOT / "analysis/tbiom/temporal")]
import phase2 as P2   # noqa: E402

SEEDS = P2.SEEDS


def main() -> int:
    brun = (ROOT / "outputs/fusion_experiments/CURRENT_BACKUP_RUN").read_text().strip()
    out = ROOT / "outputs/fusion_experiments" / brun
    main_run = (ROOT / "outputs/fusion_experiments/CURRENT_RUN").read_text().strip()
    mout = ROOT / "outputs/fusion_experiments" / main_run
    sub = np.load(out / "backup_subspaces.npz")
    sel = json.load(open(mout / "phase2_search.json"))

    t, ev, feats, y, grp = P2.build()
    dom = t["source"].astype(str).to_numpy()
    folds = P2.outer_folds(grp, y, 5)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    W, mu = sub["pca_basis"], sub["mu"]
    app = feats["APP"]
    pca32 = (app.astype(np.float64) - mu) @ W                       # the shared PCA-32 space

    arms = {"PCA": [sub["pca"]], "shuffled-pairs": [sub["shuffled"]],
            "manipulation-only": [sub["man_only"]], "intervention-guided": [sub["guided"]],
            "random": [sub[f"random_s{s}"] for s in SEEDS]}

    res, preds = {}, {}
    for name, bases in arms.items():
        per_seed = []
        for si, seed in enumerate(SEEDS):
            Q = bases[si % len(bases)]                               # random uses its own per seed
            z8 = (pca32 @ Q).astype(np.float32)
            p_all = np.zeros(len(y))
            for tri, tei in folds:
                TR = {"s": ev[tri], "f": [z8[tri]], "y": y[tri], "dom": dom[tri], "g": grp[tri]}
                TE = {"s": ev[tei], "f": [z8[tei]], "y": y[tei], "dom": dom[tei], "g": grp[tei]}
                # standardise the projected coordinates on TRAINING rows only, then tanh in the head
                m_, s_ = TR["f"][0].mean(0), np.clip(TR["f"][0].std(0), 1e-6, None)
                TR["f"][0] = (TR["f"][0] - m_) / s_; TE["f"][0] = (TE["f"][0] - m_) / s_
                TR["s"], TE["s"] = P2.standardise(np.log1p(TR["s"]), np.log1p(TE["s"]))
                cfg = sel["F"]
                mdl, _e, _v = P2.fit_one("F", TR, None, (z8.shape[1],), sel["widths"]["F"],
                                         cfg["lr"], seed, max_ep=max(1, cfg["epochs"] + 1),
                                         fixed_epochs=True)
                p_all[tei] = P2.predict(mdl, "F", TE, dev)
            sc = P2.macro_auroc(y, p_all, dom)
            per_seed.append(sc); preds[f"{name}_s{seed}"] = p_all
            print(f"  {name:20s} seed {seed}: macro {sc:.4f}", flush=True)
        res[name] = {"per_seed": per_seed, "mean": float(np.mean(per_seed)),
                     "sd": float(np.std(per_seed))}

    # reference comparators from the main run
    z = np.load(mout / "phase2_preds.npz", allow_pickle=True)
    gate = json.load(open(mout / "gate_decision.json"))
    res["S (reference)"] = {"mean": gate["macro"]["S"]}
    res["existing-F (reference)"] = {"mean": gate["macro"]["F"]}
    res["B1-refit (reference)"] = {"mean": gate["macro"]["B1_refit"]}

    np.savez(out / "backup_preds.npz", y=y, dom=dom, grp=grp, **preds)
    # the two primary utility contrasts, with the main brief's one-sided bound
    g = np.mean([preds[f"intervention-guided_s{s}"] for s in SEEDS], 0)
    p_ = np.mean([preds[f"PCA_s{s}"] for s in SEEDS], 0)
    contrasts = {}
    for nm, other in (("guided - PCA", p_), ("guided - shuffled",
                       np.mean([preds[f"shuffled-pairs_s{s}"] for s in SEEDS], 0)),
                      ("guided - manipulation-only",
                       np.mean([preds[f"manipulation-only_s{s}"] for s in SEEDS], 0)),
                      ("guided - random", np.mean([preds[f"random_s{s}"] for s in SEEDS], 0))):
        m_, lo, hi = P2.paired_bootstrap(y, dom, grp, g, other)
        contrasts[nm] = {"point": P2.macro_auroc(y, g, dom) - P2.macro_auroc(y, other, dom),
                         "boot_mean": m_, "lo97.5": lo, "hi": hi}
    contrasts["guided - B1refit"] = {"point": P2.macro_auroc(y, g, dom) - gate["macro"]["B1_refit"],
                                     "interval": "not paired: B1 uses the Phase 1 partition"}
    json.dump({"backup_run_id": brun, "arms": res, "contrasts": contrasts},
              open(out / "backup_arms.json", "w"), indent=1)
    print("\n=== contrasts ===")
    for k, v in contrasts.items():
        lo = v.get("lo97.5")
        print(f"  {k:28s} {v['point']:+.4f}" + (f"   97.5% lower {lo:+.4f}" if lo is not None else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

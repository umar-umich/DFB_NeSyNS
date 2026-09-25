#!/usr/bin/env python3
"""Gate G0a: does conflict add anything to vacuity when predicting the fused system's own errors?

    OMP_NUM_THREADS=4 <discern_ext python> training/joint_ibdc/gate_g0a.py

On J0, the deployed staged system, using out-of-fold development opinions. The error label is whether
the fused decision at the frozen threshold is wrong. Four logistic models are fitted with 5-fold
source-grouped cross-validation inside the development pool:

    1. u_f alone          2. kappa alone          3. u_f and kappa          4. u_f and the three kappa_ij

PASS if the source-grouped bootstrap 95% interval for (set 3 - set 1) lies above zero.

WHY THIS IS NOT REUSED FROM THE PCR6+ STUDY. The brief allows reuse "with the same definitions". The
PCR6+ conflict-by-source model is a logistic on [vacuity, K_TS, K_AO, K_all, margin] -- it carries a
margin term this gate does not have, and its conflict inputs are the tuple-level attributions K_TS,
K_AO and K_all rather than the pairwise kappa_ij the IBDC loss actually uses. Different inputs, so it
is recomputed here.

OUT OF FOLD. The deployed protocol fits reliability on five random group-disjoint 50/50 splits, so
every number is produced on the selection half by weights fitted on the other half. Results are
reported per split and as the mean, which is how the rest of this codebase reports development
metrics; the bootstrap pools the splits and resamples SOURCE GROUPS, so a video appearing in more
than one selection half moves with its group rather than being treated as independent evidence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(ROOT / "analysis/tbiom/pcr6p"), str(ROOT / "analysis/tbiom/grouped"),
                str(ROOT / "analysis/tbiom"), str(ROOT / "analysis/tbiom/temporal"),
                str(Path(__file__).resolve().parent)]

import arms as AR                # noqa: E402
import ops_pcr6p as P            # noqa: E402
import confirm as C              # noqa: E402
import dsr_system as D           # noqa: E402
import exp1_fusion_audit as A    # noqa: E402
import score as S                # noqa: E402
import fusion_torch as FT        # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold       # noqa: E402
from sklearn.preprocessing import StandardScaler     # noqa: E402

SPATIAL, BRANCH, TAG = "warmgate_e19", "vjepa2_lora_B_seed42", "42"
VIEWS = ("T", "APP", "OM")
PAIR_NAMES = ["k_T_APP", "k_T_OM", "k_APP_OM"]
N_BOOT, N_CV = 1000, 5
OUT = ROOT / "analysis/joint_ibdc"
SETS = {"1_uf": ["u_f"], "2_kappa": ["kappa"], "3_uf_kappa": ["u_f", "kappa"],
        "4_uf_pairs": ["u_f"] + PAIR_NAMES}


def cv_predict(X, y, groups):
    """Out-of-sample predicted error probability under source-grouped K-fold."""
    out = np.full(len(y), np.nan)
    ng = len(np.unique(groups))
    gkf = GroupKFold(n_splits=min(N_CV, ng))
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            out[te] = y[tr].mean() if len(tr) else 0.5
            continue
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        out[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return out


def main():
    dev = C.dev_table(SPATIAL, BRANCH)
    print(f"development pool: {len(dev)} videos (system {TAG}: {SPATIAL} + {BRANCH})")

    rows, pooled = [], {k: [] for k in SETS}
    pooled_err, pooled_grp = [], []
    for k, Fd, Sd in D.splits(dev):
        w = AR.fit_w(Fd, "PCR6P-R")
        p, f, _sc, _att = AR.scores(Sd, "PCR6P-R", w)
        tau = AR.tau_of(p, Sd)
        err = ((p >= tau).astype(int) != Sd.y.to_numpy()).astype(int)

        m = P.masses_of(Sd, VIEWS, w)                       # discounted opinions (n, 3, 3)
        kij = FT.pairwise_kappa(torch.tensor(m, dtype=torch.float64)).numpy()
        feats = {"u_f": f[:, 2], "kappa": kij.max(1)}
        for i, nm in enumerate(PAIR_NAMES):
            feats[nm] = kij[:, i]

        groups = A.groups_for(Sd)
        row = {"split": k, "w": list(w), "tau": tau, "n": int(len(Sd)),
               "error_rate": float(err.mean()), "aurocs": {}}
        if err.min() == err.max():
            print(f"  split {k}: degenerate error label, skipped")
            continue
        for name, cols in SETS.items():
            X = np.stack([feats[c] for c in cols], 1)
            pr = cv_predict(X, err, groups)
            row["aurocs"][name] = float(C.auc_weighted(pr, err, np.ones((1, len(err)), dtype=int))[0])
            pooled[name].append(pr)
        rows.append(row)
        pooled_err.append(err)
        pooled_grp.append(np.array([f"s{k}|{g}" for g in groups]))
        print(f"  split {k}: w={w} tau={tau:.4f} err={err.mean():.4f}  " +
              "  ".join(f"{n}={row['aurocs'][n]:.4f}" for n in SETS), flush=True)

    if not rows:
        print("G0a: no usable split"); return 2

    P_ = {n: np.concatenate(v) for n, v in pooled.items()}
    E = np.concatenate(pooled_err)
    # the bootstrap resamples the SOURCE GROUP, carrying all of its rows, across splits at once
    G_ = np.concatenate([np.array([g.split("|", 1)[1] for g in gg]) for gg in pooled_grp])
    W = np.stack(S.group_boot(G_, N_BOOT, 20260925))          # (draws, n), all draws at once
    diffs = C.auc_weighted(P_["3_uf_kappa"], E, W) - C.auc_weighted(P_["1_uf"], E, W)
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    one = np.ones((1, len(E)), dtype=int)
    point = float(C.auc_weighted(P_["3_uf_kappa"], E, one)[0] - C.auc_weighted(P_["1_uf"], E, one)[0])
    ok = lo > 0.0

    pooled_auc = {n: float(C.auc_weighted(P_[n], E, one)[0]) for n in SETS}
    means = {n: float(np.mean([r["aurocs"][n] for r in rows])) for n in SETS}
    sds = {n: float(np.std([r["aurocs"][n] for r in rows])) for n in SETS}
    res = {"gate": "G0a", "verdict": "PASS" if ok else "FAIL",
           "rule": "source-grouped bootstrap 95% interval for (set 3 - set 1) above zero",
           "system": TAG, "n_boot": N_BOOT, "mean_auroc": means, "sd_auroc": sds,
           "pooled_auroc": pooled_auc, "set3_minus_set1": {"point": point, "ci95": [lo, hi]}, "per_split": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "gate_g0a.json", "w"), indent=1)

    print()
    for n in SETS:
        print(f"  {n:12s} error AUROC {means[n]:.4f} +-{sds[n]:.4f}  (pooled {pooled_auc[n]:.4f})")
    print(f"\n  set 3 - set 1 = {point:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"G0a {res['verdict']} -> {OUT / 'gate_g0a.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

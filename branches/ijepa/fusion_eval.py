#!/usr/bin/env python3
"""Fused-system evaluation for any candidate artifact slot: trust weights, threshold, leave-one-out.

    <discern_ext python> branches/ijepa/fusion_eval.py --self-check

Every keep rule in this study needs the same thing: put a candidate row into the ARTIFACT SLOT beside
the deployed Appearance and V-JEPA 2 video views, refit the trust weights on the five source-grouped
development folds and the threshold at FF++ c23 validation EER exactly as deployed, then report
out-of-fold development AUROC, real-video FPR at that threshold, and each view's leave-one-out
contribution. The brief's G3, Amendment 1's keep rule and Amendment 2's keep rule all reduce to it,
so it lives in one place.

THE SLOT. The question the brief asks is whether a residual view can REPLACE the DiCoME-style
Off-manifold view, so a candidate is substituted into that slot and the other two views are untouched.
Views stay ordered (T, APP, OM) because that is the order `analysis/tbiom/pcr6p/arms.py` fits and
fuses in, and a silent reordering would put the weights on the wrong views.

LEAVE-ONE-OUT USES THE NEUTRALITY PROPERTY. Under PCR6+ a view discounted to r = 0 is vacuous and a
vacuous view is exactly a removed one, which is the property the operator was chosen for. So a view's
contribution is the fused AUROC with the fitted weights minus the fused AUROC with that view's weight
set to zero, with no refitting in between -- refitting would measure a different system, not that
view's contribution to this one.

SELF-CHECK. Substituting the deployed Off-manifold view back into its own slot must reproduce the
deployed PCR6+-R development numbers. If it does not, the harness is wrong and no candidate measured
through it means anything.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(ROOT / "analysis/tbiom/pcr6p"), str(ROOT / "analysis/tbiom/grouped"),
                str(ROOT / "analysis/tbiom"), str(ROOT / "analysis/tbiom/temporal")]

import arms as AR                # noqa: E402
import ops_pcr6p as P            # noqa: E402
import confirm as C              # noqa: E402
import dsr_system as DS          # noqa: E402
import exp1_fusion_audit as A    # noqa: E402
import score as S                # noqa: E402

SPATIAL, BRANCH = "warmgate_e19", "vjepa2_lora_B_seed42"
VIEWS = ("T", "APP", "OM")
N_BOOT = 1000
OUT = ROOT / "analysis/ijepa"


def dev_table():
    return C.dev_table(SPATIAL, BRANCH)


def substitute(dev, p=None, u=None, videos=None):
    """Return a copy of the development table with the artifact slot replaced by (p, u).

    `videos` names the row order of p and u; it is aligned onto the table rather than assumed, and a
    video the candidate does not cover is an error rather than a silently dropped row.
    """
    t = dev.copy()
    if p is None:
        return t                                        # self-check: the deployed slot, untouched
    idx = {v: i for i, v in enumerate(videos)}
    missing = [v for v in t.v if v not in idx]
    if missing:
        raise KeyError(f"candidate does not cover {len(missing)} development videos, e.g. {missing[:3]}")
    j = np.array([idx[v] for v in t.v])
    t["p_OM"], t["u_OM"] = np.asarray(p)[j], np.asarray(u)[j]
    return t


def evaluate(t, tag="candidate"):
    """Fit per fold, score out of fold, and report leave-one-out contributions."""
    rows, oof_p, oof_y, oof_g = [], [], [], []
    loo = {v: [] for v in VIEWS}
    for k, Fd, Sd in DS.splits(t):
        w = AR.fit_w(Fd, "PCR6P-R")
        p, f, _sc, _att = AR.scores(Sd, "PCR6P-R", w)
        tau = AR.tau_of(p, Sd)
        y = Sd.y.to_numpy()
        pred = (p >= tau).astype(int)
        real, fake = y == 0, y == 1
        full = A.fast_auc(p[real], p[fake]) if real.any() and fake.any() else float("nan")
        row = {"split": k, "w": list(w), "tau": float(tau), "auroc": full,
               "fpr_real": float(pred[real].mean()) if real.any() else float("nan"),
               "tpr": float(pred[fake].mean()) if fake.any() else float("nan"), "loo": {}}
        for i, v in enumerate(VIEWS):
            if w[i] == 0.0:
                row["loo"][v] = 0.0                     # already absent; removing it changes nothing
            else:
                w0 = list(w); w0[i] = 0.0
                p0 = P.p_of(AR.fuse(Sd, "PCR6P-R", w0))
                row["loo"][v] = float(full - A.fast_auc(p0[real], p0[fake]))
            loo[v].append(row["loo"][v])
        rows.append(row)
        oof_p.append(p); oof_y.append(y); oof_g.append(A.groups_for(Sd))
        print(f"  {tag} split {k}: w={w} tau={tau:.4f} AUROC {full:.4f} "
              f"FPR {row['fpr_real']:.4f} TPR {row['tpr']:.4f}", flush=True)

    P_, Y_, G_ = np.concatenate(oof_p), np.concatenate(oof_y), np.concatenate(oof_g)
    one = np.ones((1, len(Y_)), dtype=int)
    pooled = float(C.auc_weighted(P_, Y_, one)[0])
    return {"tag": tag, "per_split": rows,
            "mean_auroc": float(np.mean([r["auroc"] for r in rows])),
            "sd_auroc": float(np.std([r["auroc"] for r in rows])),
            "pooled_oof_auroc": pooled,
            "mean_fpr_real": float(np.mean([r["fpr_real"] for r in rows])),
            "mean_tpr": float(np.mean([r["tpr"] for r in rows])),
            "loo_mean": {v: float(np.mean(loo[v])) for v in VIEWS},
            "loo_folds_nonzero_weight": {v: int(sum(1 for r in rows if r["w"][i] > 0))
                                         for i, v in enumerate(VIEWS)},
            "_oof": (P_, Y_, G_)}


def keep_rule(cand, comp):
    """Amendment 1's three-part rule, restated in Amendment 2. A tie is NOT kept."""
    c1 = cand["pooled_oof_auroc"] > comp["pooled_oof_auroc"]     # single-view parts are supplied separately
    c2 = cand["mean_auroc"] >= comp["mean_auroc"]
    c3 = cand["mean_fpr_real"] <= comp["mean_fpr_real"] + 0.01
    return {"fused_dev_auroc_at_least": bool(c2), "fpr_within_0.01": bool(c3),
            "pooled_oof_higher": bool(c1), "kept_on_fusion_parts": bool(c2 and c3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    dev = dev_table()
    print(f"development pool {len(dev)} videos, views {VIEWS}")
    res = evaluate(substitute(dev), tag="deployed OM")

    out = {"system": "42", "views": list(VIEWS),
           "deployed_slot": {k: v for k, v in res.items() if k != "_oof"}}
    if a.self_check:
        ref = json.load(open(ROOT / "tbiom/reports/EXP_PCR6P_DEVELOPMENT.json"))["42"]
        want = float(np.mean([s["arms"]["PCR6P-R"]["auroc"] for s in ref]))
        got = res["mean_auroc"]
        wref = [s["w"]["PCR6P-R"] for s in ref]
        wgot = [r["w"] for r in res["per_split"]]
        ok = abs(got - want) < 1e-9 and all(list(x) == list(y) for x, y in zip(wref, wgot))
        out["self_check"] = {"deployed_mean_auroc": want, "harness_mean_auroc": got,
                             "abs_diff": abs(got - want), "weights_match": wgot == [list(x) for x in wref],
                             "verdict": "PASS" if ok else "FAIL"}
        print(f"\nself-check: deployed {want:.6f} vs harness {got:.6f} "
              f"(diff {abs(got-want):.2e}), weights match {out['self_check']['weights_match']}")
        print(f"self-check {out['self_check']['verdict']}")
    print(f"\nleave-one-out (mean over folds): {res['loo_mean']}")
    print(f"folds with nonzero weight: {res['loo_folds_nonzero_weight']}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "fusion_eval_selfcheck.json", "w"), indent=1)
    print(f"-> {OUT / 'fusion_eval_selfcheck.json'}")
    return 0 if (not a.self_check or out["self_check"]["verdict"] == "PASS") else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Section 5: the F0-F8 fusion matrix on the shortlisted subsets, and "with versus without uncertainty".

    <discern_ext_timm python> upgrade/branchfusion/fusion_matrix.py \
        --subsets A+V+O,A+S+V,A+V+O+M+P --ops all

THE THREE QUESTIONS SECTION 5 SEPARATES, and which row answers which:

  (a) CE versus evidential TRAINING of the heads   -> F3 against F4 (same inputs, same architecture)
  (b) fusion of PROBABILITIES versus of EVIDENCE   -> F1 on EDL probabilities against F5-F8
  (c) uncertainty-based ABSTENTION at deployment   -> `uncertainty.py`, not here

Hiding an uncertainty output while leaving the probabilities unchanged cannot change AUROC, and is
never reported here as a new detector. Every row states its track, and the F3/F4 rows carry
"fusion/heads trained on development data" because they are Track D.

ALGEBRA IS CHECKED BEFORE ANY ROW IS BELIEVED. `--ops all` runs `opinions.algebra_report` first:
endpoint behaviour at rho = 0 and rho = 1, normalisation and nonnegative evidence, vacuous-opinion
behaviour, high-conflict stability, and input-order behaviour over all permutations. Section 11 makes
invalid operator normalisation a hard stop, so a failure here aborts rather than producing a table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                      # noqa: E402
import branches as B                                    # noqa: E402
import opinions as O                                    # noqa: E402
import subsets as SS                                    # noqa: E402

# Fail loudly if a sibling import resolved outside this package: four of these module
# names also exist on the legacy sys.path entries, doing different jobs.
for _m in (B, O, SS):
    C.assert_own_module(_m)


C.thread_caps()

#: F0-F8. F3/F4 are Track D and delegated to `heads.py`; the rest act on opinions.
ALL_OPS = ("F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8")


def run_opinion_op(dev, subset, op, objective="v31") -> dict:
    """One opinion-level operator, out of fold on development, with its conflict diagnostics."""
    y = dev.y.to_numpy()
    pos = {str(v): i for i, v in enumerate(dev.v)}
    oof = np.full(len(dev), np.nan)
    ws, stats_acc = [], {}
    dev_single = {b: C.v31_macro(dev, dev[f"p_{b}"].to_numpy())[0] for b in subset}
    for k, fitpart, scorepart in C.dev_folds(dev):
        si = np.array([pos[str(v)] for v in scorepart.v])
        if op == "F0":
            b, p = O.best_single(scorepart, subset, dev_single)
            ws.append({"branch": b})
        elif op == "F1":
            p = O.fuse(scorepart, subset, "F1")
            ws.append(None)
        else:
            w, _val = O.fit_weights(fitpart, subset, op, objective=objective)
            st: dict = {}
            p = O.fuse(scorepart, subset, op, list(w), stats=st)
            ws.append(list(w))
            for kk, vv in st.items():
                stats_acc.setdefault(kk, []).append(vv)
        oof[si] = p
    m = np.isfinite(oof)
    sub_t = dev[m].reset_index(drop=True)
    macro, per_type = C.v31_macro(sub_t, oof[m])
    out = {"operator": op, "subset": C.subset_id(subset), "track": "P",
           "label": ("source-trained representations; only the low-dimensional trust weights "
                     "use development"),
           "oof_auroc": C.auroc(y[m], oof[m]), "v31_macro": macro, "v31_per_type": per_type,
           "nll": C.nll(y[m], oof[m]), "brier": C.brier(y[m], oof[m]),
           "ece15": C.ece(y[m], oof[m], 15), "ece10": C.ece(y[m], oof[m], 10),
           "weights_per_fold": ws, "n_scored": int(m.sum()),
           "conflict": {k: float(np.mean(v)) for k, v in stats_acc.items()},
           "oof_p": oof}
    if op == "F0":
        out["selected_branch"] = max(dev_single, key=lambda b: dev_single[b])
        out["note"] = "best single branch chosen on DEVELOPMENT, not per test dataset"
    return out


def run_track_d(dev, subset, loss, arch="mlp", seed=0, param_match_to=None) -> dict:
    import trackd_heads as H
    avail = [b for b in subset if H.rep_available(b, "CDFv2")]
    if len(avail) != len(subset):
        return {"operator": "F4" if loss == "edl" else "F3", "subset": C.subset_id(subset),
                "status": "skipped",
                "reason": f"no representation for {[b for b in subset if b not in avail]}; "
                          f"run the extractions first"}
    r = H.track_d_concat(dev, subset, arch=arch, loss=loss, seed=seed,
                         param_match_to=param_match_to)
    r["ece15"] = C.ece(np.array(r["y"]), np.array(r["p"]), 15)
    r["ece10"] = C.ece(np.array(r["y"]), np.array(r["p"]), 10)
    r["brier"] = C.brier(np.array(r["y"]), np.array(r["p"]))
    return r


def matrix(dev, subset, ops=ALL_OPS, seed=0) -> dict:
    """The full F0-F8 row for one subset."""
    rows = {}
    for op in ops:
        print(f"  {C.subset_id(subset)} / {op}", flush=True)
        if op in ("F3", "F4"):
            rows[op] = run_track_d(dev, subset, "edl" if op == "F4" else "ce", seed=seed)
        else:
            r = run_opinion_op(dev, subset, op)
            rows[op] = {k: v for k, v in r.items() if k != "oof_p"}
        got = rows[op]
        if "v31_macro" in got:
            print(f"    macro {got['v31_macro']:.4f}  auroc {got.get('oof_auroc', float('nan')):.4f}"
                  f"  nll {got.get('nll', float('nan')):.4f}", flush=True)
        elif "oof_v31_macro" in got:
            print(f"    macro {got['oof_v31_macro']:.4f}  auroc {got['oof_auroc']:.4f}", flush=True)
        else:
            print(f"    {got.get('status', '?')}: {got.get('reason', '')[:70]}", flush=True)
    return rows


def contrasts(rows: dict) -> dict:
    """The comparisons section 5 says each pair isolates."""
    def macro(op):
        r = rows.get(op, {})
        return r.get("v31_macro", r.get("oof_v31_macro"))

    out = {}
    if macro("F1") is not None:
        for op in ("F5", "F6", "F7", "F8"):
            if macro(op) is not None:
                out[f"{op} - F1"] = {
                    "delta": macro(op) - macro("F1"),
                    "isolates": ("use of evidence STRENGTH over the same EDL probabilities "
                                 "(section 5)")}
    if macro("F3") is not None and macro("F4") is not None:
        out["F4 - F3"] = {
            "delta": macro("F4") - macro("F3"),
            "isolates": "the head LOSS under a fixed representation",
            "label": ("frozen-representation head comparison, NOT a full training-objective "
                      "comparison (section 5)")}
    if macro("F0") is not None:
        for op in ("F1", "F6", "F7", "F8"):
            if macro(op) is not None:
                out[f"{op} - F0"] = {
                    "delta": macro(op) - macro("F0"),
                    "isolates": "whether combining beats the best single branch at all"}
    if macro("F2") is not None and macro("F1") is not None:
        out["F2 - F1"] = {"delta": macro("F2") - macro("F1"),
                          "isolates": "fitted simplex weights over uniform averaging"}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", required=True)
    ap.add_argument("--ops", default="all")
    ap.add_argument("--branches", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-algebra", action="store_true")
    a = ap.parse_args()

    subs = [tuple(s.strip().split("+")) for s in a.subsets.split(",") if s.strip()]
    need = sorted({b for s in subs for b in s})
    items = [x.strip() for x in a.branches.split(",") if x.strip()] or need
    dev = B.dev_table(items, require_all=True)
    print(f"development pool: {len(dev)} videos, branches {items}")

    res = {"subsets": {}, "algebra": None}
    if not a.skip_algebra:
        print("\nalgebra checks (section 5)")
        alg = O.algebra_report(dev, tuple(need[:3]) if len(need) >= 3 else tuple(need))
        res["algebra"] = alg
        bad = []
        for op in ("F5", "F6", "F7", "F8"):
            e = alg.get(op, {}).get("endpoints", {})
            for k, v in e.items():
                if isinstance(v, float) and v > 1e-6:
                    bad.append(f"{op}.{k}={v:.2e}")
            for nm, cell in alg.get(op, {}).get("conflict", {}).items():
                if isinstance(cell, dict) and not cell.get("in_range", True):
                    bad.append(f"{op}.{nm} out of range")
        inv = {k: v for k, v in alg["invalid_stored_opinions"].items() if v}
        if inv:
            bad.append(f"invalid stored opinions {inv}")
        pm = alg.get("permutation", {})
        for op in ("F5", "F7", "F8"):
            if op in pm and not pm[op]["invariant"]:
                bad.append(f"{op} not permutation-invariant "
                           f"({pm[op]['max_abs_change']:.2e})")
        if bad:
            print("ALGEBRA FAILURES -- section 11 hard stop:")
            for s in bad:
                print(f"  {s}")
            json.dump(res, open(C.OUT / "fusion_matrix.json", "w"), indent=1, default=str)
            return 1
        print(f"  endpoints, normalisation, conflict and permutation all pass; "
              f"pairwise PCR6 spread {pm.get('pairwise', {}).get('max_abs_change', float('nan')):.4f} "
              f"(expected nonzero: it is a different operator)")

    ops = ALL_OPS if a.ops == "all" else tuple(x.strip() for x in a.ops.split(","))
    for s in subs:
        print(f"\n=== {C.subset_id(s)}")
        rows = matrix(dev, s, ops, a.seed)
        res["subsets"][C.subset_id(s)] = {"rows": rows, "contrasts": contrasts(rows)}
        for k, v in res["subsets"][C.subset_id(s)]["contrasts"].items():
            print(f"  {k}: {v['delta']:+.4f}  [{v['isolates']}]")
    json.dump(res, open(C.OUT / "fusion_matrix.json", "w"), indent=1, default=str)
    print(f"\n-> {C.OUT / 'fusion_matrix.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

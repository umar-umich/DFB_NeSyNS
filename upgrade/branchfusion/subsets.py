#!/usr/bin/env python3
"""Section 4: individual branches, every subset, nested selection and the branch-contribution table.

    <discern_ext_timm python> upgrade/branchfusion/subsets.py --screen
    <discern_ext_timm python> upgrade/branchfusion/subsets.py --nested
    <discern_ext_timm python> upgrade/branchfusion/subsets.py --test --subsets A+V+O,A+S+V

THE SCREEN RULE IS FIXED BEFORE ANY SUBSET IS SCORED. B1 -- the global reliability discount under
PCR6+ -- is the evidential screen rule, with uniform probability averaging (F1) as its inexpensive
control. Using one rule for every subset is what separates branch selection from operator search;
section 5 does the operator search afterwards, on a shortlist.

TWO ESTIMATES, NEVER CONFLATED.

  `screen`  Every subset scored out of fold on development with the fixed screen rule. This is the
            comparison table, and the best cell in it is an EXPLORATORY statistic -- section 3 says
            so in as many words, because the maximum over 31 subsets is biased upward.
  `nested`  The subset AND the operator are selected inside each outer fold, on that fold's inner
            folds, and the selected system is scored on the held-out outer part. This is the
            unbiased estimate, and it is the one a decision rests on.

THE ORACLE ROW IS NOT DEPLOYABLE AND IS LABELLED THAT WAY. It picks the best single branch per
dataset, which requires the test labels. It exists to answer one question section 4 poses directly:
does fusion ever beat view selection? If a subset never beats the oracle, the fusion is selecting a
view rather than combining evidence -- which is exactly what the previous campaign found for the
deployed triple.

THE DIAGNOSTICS EXPLAIN, THEY DO NOT DECIDE. Error correlation, rescue rate, harmful-overrule rate
and zero trust weights are reported for every shortlisted subset. Section 4 forbids promoting a
branch because its errors decorrelate, or rejecting one only because a fitted weight reaches zero in
some fold.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                      # noqa: E402
import branches as B                                    # noqa: E402
import opinions as O                                    # noqa: E402

# Fail loudly if a sibling import resolved outside this package: four of these module
# names also exist on the legacy sys.path entries, doing different jobs.
for _m in (B, O):
    C.assert_own_module(_m)


import exp1_fusion_audit as A                           # noqa: E402
from health import eer                                  # noqa: E402

C.thread_caps()

#: Section 4's screen rule and its control. Fixed here so no caller can vary it per subset.
SCREEN_OP = "F8"                     # PCR6+ over the subset
SCREEN_NAME = "B1 (PCR6+ with fitted reliability discount)"
CONTROL_OP = "F1"                    # uniform probability mean
#: Section 5's operator shortlist for the nested selection. F3/F4 are Track D and run separately.
NESTED_OPS = ("F1", "F2", "F5", "F6", "F7", "F8")
#: Section 4's tie-break: within this margin, prefer fewer encoder evaluations, then fewer parameters.
TIE_MARGIN = 0.002
#: Encoder evaluations per branch -- the first tie-break key. A and S share one PE pass when their
#: weights, preprocessing and frames are identical, which section 2.2 requires them to be.
ENCODER_COST = {"A": 1, "S": 1, "V": 1, "M": 1, "O": 1, "R": 1, "P": 1}
SHARED_ENCODER = ({"A", "S"}, {"A", "R"}, {"A", "S", "R"})
#: Trainable parameters per branch, the second tie-break key. Filled from the run records when present.
PARAM_COST = {"A": 890_880, "S": 1_842_946, "V": 919_938, "M": 394_138, "O": 0, "R": 136_000,
              # P unfreezes two blocks on top of its adapters: 7,624,192 + 66,946 from its own
              # train_record.json. It is by far the largest trainable budget in the panel and the
              # tie-break must see that.
              "P": 7_691_138}


def encoder_evals(subset) -> int:
    """How many distinct encoder passes a subset needs.

    A, S and R all read the same PE-Core forward pass (A pools it, S attends over it, R summarises its
    patch tokens at block 16), so a subset holding several of them pays for one PE evaluation, not
    three. V and M each need their own video encoder.
    """
    s = set(subset)
    pe = s & {"A", "S", "R"}
    # P runs its OWN PE-Core forward pass: it is a 16-frame clip encoder with its own adapted
    # weights, so it shares nothing with A/S/R despite also being PE-Core.
    return (1 if pe else 0) + len(s & {"V", "M", "P"}) + (0 if "O" not in s else 1)


def trainable_params(subset) -> int:
    return sum(PARAM_COST.get(b, 0) for b in subset)


# ----------------------------------------------------------------------------- the screen
def fit_and_score(fit_t, score_t, subset, op, objective="v31"):
    """Fit the operator's weights on `fit_t`, score `score_t`. Returns (p, weights, fit value)."""
    if op in ("F1",):
        return O.fuse(score_t, subset, op), None, None
    w, val = O.fit_weights(fit_t, subset, op, objective=objective)
    return O.fuse(score_t, subset, op, list(w)), list(w), val


def screen_subset(dev, subset, op=SCREEN_OP, objective="v31") -> dict:
    """One subset, out of fold on development under a fixed rule."""
    y = dev.y.to_numpy()
    oof = np.full(len(dev), np.nan)
    ws, per_fold = [], []
    pos = {str(v): i for i, v in enumerate(dev.v)}
    for k, fitpart, scorepart in C.dev_folds(dev):
        si = np.array([pos[str(v)] for v in scorepart.v])
        p, w, val = fit_and_score(fitpart, scorepart, subset, op, objective)
        oof[si] = p
        ws.append(w)
        per_fold.append({"fold": k, "w": w, "fit_objective": val,
                         "auroc": C.auroc(scorepart.y.to_numpy(), p),
                         "n": int(len(si))})
    m = np.isfinite(oof)
    sub_t = dev[m].reset_index(drop=True)
    macro, per_type = C.v31_macro(sub_t, oof[m])
    zero = {b: int(sum(1 for w in ws if w is not None and w[i] == 0.0))
            for i, b in enumerate(subset)}
    return {"subset": C.subset_id(subset), "branches": list(subset), "operator": op,
            "n_branches": len(subset), "encoder_evals": encoder_evals(subset),
            "trainable_params": trainable_params(subset),
            "oof_auroc": C.auroc(y[m], oof[m]), "v31_macro": macro, "v31_per_type": per_type,
            "oof_nll": C.nll(y[m], oof[m]), "n_scored": int(m.sum()),
            "weights_per_fold": ws, "folds_weight_zero": zero, "per_fold": per_fold,
            "oof_p": oof}


def screen_all(dev, items=None, op=SCREEN_OP, control=CONTROL_OP, sizes=(1, 2, 3, 4, 5)) -> dict:
    """Every subset of the given sizes, under the screen rule and its control."""
    items = tuple(items or [b for b in C.CORE if f"p_{b}" in dev.columns])
    subs = C.subsets_of(items, sizes)
    rows, ctrl = [], []
    print(f"screening {len(subs)} subsets of {items} under {SCREEN_NAME}", flush=True)
    for s in subs:
        r = screen_subset(dev, s, op)
        rows.append({k: v for k, v in r.items() if k != "oof_p"})
        print(f"  {C.subset_id(s):16s} macro {r['v31_macro']:.4f}  pooled {r['oof_auroc']:.4f}  "
              f"enc {r['encoder_evals']}  zeros {r['folds_weight_zero']}", flush=True)
        if len(s) > 1:
            c = screen_subset(dev, s, control)
            ctrl.append({k: v for k, v in c.items() if k != "oof_p"})
    return {"items": list(items), "screen_rule": SCREEN_NAME, "screen_op": op,
            "control_op": control, "rows": rows, "control_rows": ctrl,
            "note": ("the best cell here is EXPLORATORY: it is the maximum over "
                     f"{len(subs)} subsets on the same folds. The unbiased estimate is `--nested`.")}


# ----------------------------------------------------------------------------- selection rule
def select(rows, margin=TIE_MARGIN) -> dict:
    """Section 4's selection rule, applied to screen rows.

    By fused development macro AUROC, then within `margin` prefer fewer encoder evaluations, then
    fewer trainable parameters. Explicitly NOT the three highest standalone AUROCs.
    """
    best = max(rows, key=lambda r: r["v31_macro"])
    near = [r for r in rows if r["v31_macro"] >= best["v31_macro"] - margin]
    pick = sorted(near, key=lambda r: (r["encoder_evals"], r["trainable_params"],
                                       -r["v31_macro"]))[0]
    triples = [r for r in rows if r["n_branches"] == 3]
    pairs = [r for r in rows if r["n_branches"] == 2]
    bt = max(triples, key=lambda r: r["v31_macro"]) if triples else None
    bp = max(pairs, key=lambda r: r["v31_macro"]) if pairs else None
    return {"best_raw": best["subset"], "best_raw_macro": best["v31_macro"],
            "selected": pick["subset"], "selected_macro": pick["v31_macro"],
            "selected_encoder_evals": pick["encoder_evals"],
            "selected_params": pick["trainable_params"],
            "within_margin": [r["subset"] for r in near], "margin": margin,
            "best_triple": bt["subset"] if bt else None,
            "best_triple_macro": bt["v31_macro"] if bt else None,
            "best_pair": bp["subset"] if bp else None,
            "best_pair_macro": bp["v31_macro"] if bp else None,
            "rule": ("fused development macro AUROC, averaged equally across the fixed development "
                     "datasets; reenactment secondary; ties within 0.002 broken by fewer encoder "
                     "evaluations then fewer trainable parameters")}


# ----------------------------------------------------------------------------- nested selection
def nested(dev, items=None, ops=NESTED_OPS, sizes=(1, 2, 3)) -> dict:
    """Select the subset AND the operator inside each outer fold; score on the held-out part.

    This is the unbiased estimate section 3 asks for. The selection sees only the outer fold's
    training part, split into three inner folds; the outer part is touched once, to score.
    """
    items = tuple(items or [b for b in C.CORE if f"p_{b}" in dev.columns])
    subs = C.subsets_of(items, sizes)
    picks, scores = [], []
    for k, fitpart, scorepart in C.dev_folds(dev):
        inner = list(C.inner_folds(fitpart))
        tally = {}
        for s in subs:
            for op in ops:
                vals = []
                for j, itr, iva in inner:
                    p, _w, _v = fit_and_score(itr, iva, s, op)
                    vals.append(C.v31_macro(iva, p)[0])
                tally[(C.subset_id(s), op)] = float(np.nanmean(vals))
        key = max(tally, key=lambda kk: tally[kk])
        sub = tuple(key[0].split("+"))
        p, w, _v = fit_and_score(fitpart, scorepart, sub, key[1])
        got = C.v31_macro(scorepart, p)[0]
        picks.append({"fold": k, "subset": key[0], "operator": key[1],
                      "inner_macro": tally[key], "w": w, "outer_macro": got,
                      "outer_auroc": C.auroc(scorepart.y.to_numpy(), p)})
        scores.append(got)
        print(f"  fold {k}: selected {key[0]} with {key[1]} "
              f"(inner {tally[key]:.4f}) -> outer {got:.4f}", flush=True)
    sel = [p["subset"] for p in picks]
    return {"per_fold": picks, "nested_macro_mean": float(np.mean(scores)),
            "nested_macro_sd": float(np.std(scores)),
            "selection_stable": len(set(sel)) == 1,
            "selected_per_fold": sel,
            "modal_subset": max(set(sel), key=sel.count),
            "note": ("unbiased: the subset and the operator were chosen on inner folds of each "
                     "outer training part, and each outer part was scored once")}


# ----------------------------------------------------------------------------- diagnostics
def diagnostics(dev, subset, op=SCREEN_OP) -> dict:
    """Error correlation, rescue rate, harmful-overrule rate and zero trust weights.

    Thresholds are fixed on FF++ validation rows, which are inside the development pool but are the
    allowed calibration source (section 6). Correlations are reported CLASS-CONDITIONALLY as section 4
    requires, because a pooled correlation mixes prevalence into the number.
    """
    r = screen_subset(dev, subset, op)
    fused = r["oof_p"]
    m = np.isfinite(fused)
    t = dev[m].reset_index(drop=True)
    fp = fused[m]
    y = t.y.to_numpy()
    ff = np.where(t.source.to_numpy() == "FFpp_val")[0] if "source" in t.columns else np.array([])

    def thr(p):
        return float(eer(y[ff], p[ff])[1]) if len(ff) else 0.5

    errs, taus = {}, {}
    for b in subset:
        pb = t[f"p_{b}"].to_numpy()
        taus[b] = thr(pb)
        errs[b] = ((pb >= taus[b]).astype(int) != y).astype(int)
    tau_f = thr(fp)
    err_f = ((fp >= tau_f).astype(int) != y).astype(int)

    def corr(a, b, sel=None):
        a2, b2 = (a[sel], b[sel]) if sel is not None else (a, b)
        if len(a2) < 3 or a2.std() < 1e-12 or b2.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(a2, b2)[0, 1])

    real, fake = y == 0, y == 1
    phi = {}
    for a, b in itertools.combinations(subset, 2):
        phi[f"{a}-{b}"] = {"all": corr(errs[a], errs[b]),
                           "real": corr(errs[a], errs[b], real),
                           "fake": corr(errs[a], errs[b], fake)}
    # rescue: every branch wrong, fusion right. harmful overrule: some branch right, fusion wrong.
    allwrong = np.ones(len(y), bool)
    anyright = np.zeros(len(y), bool)
    for b in subset:
        allwrong &= errs[b] == 1
        anyright |= errs[b] == 0
    rescue = int(((allwrong) & (err_f == 0)).sum())
    harmful = int(((anyright) & (err_f == 1)).sum())
    return {"subset": C.subset_id(subset), "operator": op, "tau_fused": tau_f,
            "tau_per_branch": taus, "error_correlation_phi": phi,
            "rescue_count": rescue,
            "rescue_rate": float(rescue / max(int(allwrong.sum()), 1)),
            "rescue_denominator": int(allwrong.sum()),
            "harmful_overrule_count": harmful,
            "harmful_overrule_rate": float(harmful / max(int(anyright.sum()), 1)),
            "harmful_denominator": int(anyright.sum()),
            "folds_weight_zero": r["folds_weight_zero"],
            "fused_error_rate": float(err_f.mean()),
            "branch_error_rate": {b: float(errs[b].mean()) for b in subset},
            "caveat": ("explanatory only. Section 4 forbids promoting a branch because its errors "
                       "decorrelate, or rejecting one only because a fitted weight reaches zero.")}


def contribution(dev, subset, op=SCREEN_OP, n_boot=C.N_BOOT_FINAL) -> dict:
    """Each branch's incremental contribution, with the fusion REFIT for the reduced subset.

    Section 4: removing a branch while keeping weights fitted for the full subset is a different
    quantity -- a sensitivity analysis -- and is reported separately as `frozen_weight_delta`.
    """
    full = screen_subset(dev, subset, op)
    m = np.isfinite(full["oof_p"])
    t = dev[m].reset_index(drop=True)
    y = t.y.to_numpy()
    g = C.groups_of(t)
    pf = full["oof_p"][m]
    out = {"subset": C.subset_id(subset), "operator": op,
           "full_auroc": C.auroc(y, pf), "full_macro": full["v31_macro"], "per_branch": {}}
    for b in subset:
        red = tuple(x for x in subset if x != b)
        if not red:
            continue
        r = screen_subset(dev, red, op)
        pr = r["oof_p"][m]
        ok = np.isfinite(pr)
        d = C.paired_delta(y[ok], pf[ok], pr[ok], g[ok], n_rep=n_boot)
        # the frozen-weight variant: drop the branch by zeroing its weight, keep the rest as fitted
        froz = []
        pos = {str(v): i for i, v in enumerate(dev.v)}
        oof2 = np.full(len(dev), np.nan)
        for (k, fitpart, scorepart), w in zip(C.dev_folds(dev), full["weights_per_fold"]):
            if w is None:
                continue
            w2 = list(w)
            w2[list(subset).index(b)] = 0.0
            si = np.array([pos[str(v)] for v in scorepart.v])
            oof2[si] = O.fuse(scorepart, subset, op, w2)
        o2 = oof2[m]
        ok2 = np.isfinite(o2)
        out["per_branch"][b] = {
            "refit_delta": d, "refit_reduced_auroc": C.auroc(y[ok], pr[ok]),
            "refit_reduced_macro": r["v31_macro"],
            "frozen_weight_delta": float(C.auroc(y[ok2], pf[ok2]) - C.auroc(y[ok2], o2[ok2])),
            "frozen_weight_note": ("sensitivity analysis: weights fitted for the full subset, one "
                                   "branch zeroed. Not the same quantity as the refit delta."),
            "folds_weight_zero": full["folds_weight_zero"].get(b)}
        print(f"  {b}: refit delta {d['point']:+.4f} "
              f"[{d['lo2.5']:+.4f},{d['hi97.5']:+.4f}] "
              f"{'EXCLUDES 0' if d['excludes_zero'] else ''}", flush=True)
    return out


# ----------------------------------------------------------------------------- test evaluation
def oracle_row(test_tabs, branches) -> dict:
    """The non-deployable reference: best single branch chosen PER DATASET."""
    out = {}
    for ds, t in test_tabs.items():
        b, au, allb = O.oracle_single(t, branches)
        out[ds] = {"branch": b, "auroc": au, "per_branch": allb}
    aus = [v["auroc"] for v in out.values()]
    return {"per_dataset": out, "macro": float(np.mean(aus)) if aus else float("nan"),
            "label": "ORACLE, not deployable: uses test labels to pick the branch per dataset",
            "purpose": ("section 4's question -- does fusion beat view selection? A subset that "
                        "never beats this row is selecting a view, not combining evidence.")}


def evaluate_test(dev, test_tabs, subset, op=SCREEN_OP, tau_source="FFpp_val",
                  grid=O.WEIGHT_GRID) -> dict:
    """Score a LOCKED subset+operator on the test partitions, once.

    Weights are fitted on the whole development pool (the locked setting), and the decision threshold
    comes from FF++ validation rows inside development -- never from a test set.

    `grid` is the weight search, and it is the whole cost of this function: `fit_weights` is
    EXHAUSTIVE over `len(grid)**k - (len(grid)-1)**k` vectors, so a size-8 subset costs 325,089
    fusions and a size-12 subset 227 million. The locked grid is the default and nothing that
    reports a locked number may pass anything else; a coarser grid exists only so a wide
    exploratory map can cover subset sizes the locked fit cannot reach in any amount of time, and
    every artifact built that way has to say which grid produced it.
    """
    w, fitval = O.fit_weights(dev, subset, op, objective="v31", grid=grid)
    pdev = O.fuse(dev, subset, op, list(w))
    if "source" in dev.columns:
        ff = np.where(dev.source.to_numpy() == tau_source)[0]
        tau = float(eer(dev.y.to_numpy()[ff], pdev[ff])[1]) if len(ff) else 0.5
    else:
        tau = C.fit_threshold(dev.y.to_numpy(), pdev)
    rows = {}
    for ds, t in test_tabs.items():
        p = O.fuse(t, subset, op, list(w))
        y = t.y.to_numpy()
        rows[ds] = {"auroc": C.auroc(y, p), **C.threshold_metrics(y, p, tau),
                    "nll": C.nll(y, p), "brier": C.brier(y, p),
                    "ece15": C.ece(y, p, 15), "ece10": C.ece(y, p, 10)}
    aus = [rows[d]["auroc"] for d in C.TEST_SETS if d in rows]
    return {"subset": C.subset_id(subset), "operator": op, "weights": list(w),
            "fit_objective": fitval, "tau": tau, "tau_source": tau_source,
            "weight_grid": list(grid), "locked_grid": tuple(grid) == tuple(O.WEIGHT_GRID),
            "per_dataset": rows, "ten_set_macro": float(np.mean(aus)) if aus else float("nan"),
            "n_sets": len(aus),
            "locked": True}


def beats_oracle(test_eval: dict, oracle: dict) -> dict:
    """On how many of the ten test sets does the fused subset beat the oracle single branch?"""
    wins, detail = 0, {}
    for ds, r in test_eval["per_dataset"].items():
        o = oracle["per_dataset"].get(ds, {}).get("auroc")
        if o is None:
            continue
        d = r["auroc"] - o
        detail[ds] = {"fused": r["auroc"], "oracle": o, "delta": d,
                      "oracle_branch": oracle["per_dataset"][ds]["branch"]}
        wins += int(d > 0)
    return {"wins": wins, "n": len(detail), "threshold": 4,
            "passes": wins > 4, "per_dataset": detail,
            "reading": ("fusion that never beats the best single branch is selecting a view, not "
                        "combining evidence (section 4)")}


# ----------------------------------------------------------------------------- CLI
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branches", default="", help="default: every branch with full coverage")
    ap.add_argument("--screen", action="store_true")
    ap.add_argument("--nested", action="store_true")
    ap.add_argument("--diagnostics", default="", help="comma list of subsets, e.g. A+V+O")
    ap.add_argument("--contribution", default="")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--subsets", default="", help="locked subsets to score on test")
    ap.add_argument("--sizes", default="1,2,3,4,5", help="subset sizes for --screen")
    ap.add_argument("--nested-sizes", default="1,2,3",
                    help="subset sizes for --nested. SEPARATE from --sizes on purpose: the nested "
                         "estimate costs 6 operators x 3 inner folds x 5 outer folds per subset, so "
                         "its space is chosen deliberately rather than inherited. Until 2026-10-09 "
                         "--sizes was read ONLY by --screen, so passing it with --nested silently "
                         "did nothing and the nested result was a triples-only search presented "
                         "beside a full-space screen.")
    a = ap.parse_args()

    items = ([x.strip() for x in a.branches.split(",") if x.strip()]
             or [b for b in C.CORE
                 if all(B.available(b, n) or (C.BRANCH_OUT / b / f"{n}.csv").exists()
                        for n in list(C.DEV_SETS) + list(C.TEST_SETS))])
    print(f"branches: {items}")
    dev = B.dev_table(items, require_all=True)
    print(f"development pool: {len(dev)} videos")
    res = {}

    if a.screen:
        sizes = tuple(int(x) for x in a.sizes.split(","))
        res["screen"] = screen_all(dev, items, sizes=sizes)
        res["screen"]["sizes"] = list(sizes)
        res["selection"] = select(res["screen"]["rows"])
        print(f"\nselected: {res['selection']['selected']} "
              f"(macro {res['selection']['selected_macro']:.4f}); "
              f"best raw {res['selection']['best_raw']} "
              f"{res['selection']['best_raw_macro']:.4f}")
        json.dump(res["screen"], open(C.OUT / "screen.json", "w"), indent=1, default=str)
        json.dump(res["selection"], open(C.OUT / "selection.json", "w"), indent=1, default=str)

    if a.nested:
        nsizes = tuple(int(x) for x in a.nested_sizes.split(","))
        print(f"nested search space: sizes {nsizes}")
        res["nested"] = nested(dev, items, sizes=nsizes)
        res["nested"]["sizes"] = list(nsizes)
        res["nested"]["items"] = list(items)
        print(f"\nnested macro {res['nested']['nested_macro_mean']:.4f} "
              f"+-{res['nested']['nested_macro_sd']:.4f}; stable "
              f"{res['nested']['selection_stable']}")
        json.dump(res["nested"], open(C.OUT / "nested.json", "w"), indent=1, default=str)

    for spec, fn, name in ((a.diagnostics, diagnostics, "diagnostics"),
                           (a.contribution, contribution, "contribution")):
        if spec:
            out = {}
            for s in spec.split(","):
                sub = tuple(s.strip().split("+"))
                print(f"\n{name} for {s}")
                out[s] = fn(dev, sub)
            json.dump(out, open(C.OUT / f"{name}.json", "w"), indent=1, default=str)
            res[name] = out

    if a.test:
        tabs = B.test_tables(items, require_all=True)
        orc = oracle_row(tabs, items)
        print(f"\noracle macro {orc['macro']:.4f}")
        subs = ([tuple(s.strip().split("+")) for s in a.subsets.split(",") if s.strip()]
                or [tuple(items[:3])])
        out = {"oracle": orc, "subsets": {}}
        for sub in subs:
            e = evaluate_test(dev, tabs, sub)
            e["vs_oracle"] = beats_oracle(e, orc)
            out["subsets"][C.subset_id(sub)] = e
            print(f"  {C.subset_id(sub)}: ten-set macro {e['ten_set_macro']:.4f}, "
                  f"beats oracle on {e['vs_oracle']['wins']}/{e['vs_oracle']['n']}")
        json.dump(out, open(C.OUT / "test_eval.json", "w"), indent=1, default=str)
        res["test"] = out

    if not any((a.screen, a.nested, a.diagnostics, a.contribution, a.test)):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

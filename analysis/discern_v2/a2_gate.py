#!/usr/bin/env python3
"""A2 — oracle headroom and the realizable gate (feasibility vs deployment).

Two different questions, deliberately kept apart:

  A2 oracle   How much is there to win if routing were perfect?  (an upper bound)
  A2a gate    How much of it can an *observable* gate actually recover?  (the go/no-go)

The Phase-1 go/no-go for building D4 is A2a. A large oracle headroom with near-zero
gate recovery is a legitimate negative result -- "applicability is not inferable from
observable evidence here" -- and D4 becomes a paragraph rather than a module.

Oracle metrics
--------------
Accuracy, balanced accuracy, union-of-correct coverage `P(exists b : y_b = y)`, and the
remaining shared-error rate. Oracle-routed AUROC is deliberately NOT a headline: the oracle
consults the label to pick a branch, so the routed score is a function of the label and the
resulting ranking is not a detector's ranking. It is written to a clearly-marked
supplementary file instead.

The realizable gate
-------------------
The gate learns **specialist utility**, not an expert-ID label:

    dloss_b = loss_vis - loss_b     positive -> specialist b improves on the visual baseline

`dloss_b` is a TRAINING TARGET ONLY -- it is computed from the loss, which uses the label.
The gate's INPUTS are strictly label-free: evidence, vacuity, and cross-branch conflict,
which are all available at inference. `assert_label_free()` enforces this rather than
trusting a comment, because a leak here would silently invalidate the whole go/no-go.

P0's probability is explicitly not the primary feature: P0 is *confidently wrong* on the
inverted rows, so a gate keyed on P0 confidence would trust it exactly where it fails.

A2a vs A2b
----------
* **A2a (this phase)** leave-one-generator-out CV. The gate trains on sibling generators,
  so it is an optimistic *feasibility ceiling*: it answers "is applicability inferable at
  all?" Per-fold scalers are fit on training generators only -- fitting globally before
  splitting leaks the held-out generator and inflates every number.
* **A2b (spec now, run later)** the deployment-valid protocol: gate trained only on data the
  generalization protocol permits, with no generator/dataset/family ID as input. Written to
  `A2b_PROTOCOL.md` by this script; not run here.

    python analysis/discern_v2/a2_gate.py
    python analysis/discern_v2/a2_gate.py --source DF40 --candidates P0-DS,P2a
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

# Candidate specialist sets from the instructions. The first is the intended default; the
# second is the strongest-rescue P1 variant, kept because best-standalone != best-specialist.
DEFAULT_CANDIDATE_SETS = [
    ["P0-DS", "P1d", "P2a"],
    ["P0-DS", "P1b", "P2a"],
]

# Feature names that would leak the target if they ever reached the gate's input matrix.
FORBIDDEN_FEATURE_TOKENS = ("label", "dloss", "family", "method", "generator", "source",
                            "y_true", "correct", "oracle")


def assert_label_free(names: list[str]) -> None:
    """Fail loudly if a label-derived or identity feature reaches the gate input.

    The instructions are explicit that dloss_b, the label, and generator/dataset/family ID
    must never be inference features. This is checked rather than commented because the
    failure mode is silent: the gate would simply look brilliant.
    """
    bad = [n for n in names if any(t in n.lower() for t in FORBIDDEN_FEATURE_TOKENS)]
    if bad:
        raise AssertionError(
            f"label-leaking features in the gate input: {bad}. dloss is a TRAINING TARGET "
            f"only; generator/dataset/family IDs are never inference features.")


def bce(p: np.ndarray, y: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    out = []
    for c in (0, 1):
        m = y == c
        if m.any():
            out.append((pred[m] == c).mean())
    return float(np.mean(out)) if out else float("nan")


def build_state(source: str, pilots: list[str], thr: float) -> pd.DataFrame:
    """Video-level evidence state for every branch, aligned on video_id."""
    base = None
    for p in pilots:
        arrays, meta = C.load(p, source)
        vals = {f"p_{p}": arrays["p_fused"], f"u_{p}": arrays["u_fused"]}
        # per-branch sub-head evidence: extra observable signal, still label-free
        for sub in ("semantic", "artifact"):
            if f"p_{sub}" in arrays:
                vals[f"p{sub[0]}_{p}"] = arrays[f"p_{sub}"]
            if f"u_{sub}" in arrays:
                vals[f"u{sub[0]}_{p}"] = arrays[f"u_{sub}"]
        v = C.video_level(meta, vals)
        cols = ["video_id", "label", "family", "method"] + list(vals)
        v = v[cols]
        base = v if base is None else base.merge(
            v.drop(columns=["label", "family", "method"]), on="video_id", how="inner")
    assert base is not None
    for p in pilots:
        base[f"pred_{p}"] = (base[f"p_{p}"] >= thr).astype(int)
        base[f"ok_{p}"] = (base[f"pred_{p}"] == base["label"]).astype(int)
        base[f"loss_{p}"] = bce(base[f"p_{p}"].values, base["label"].values)
    return base


def oracle_metrics(df: pd.DataFrame, pilots: list[str]) -> dict:
    """Upper bound from perfect per-sample routing. Label-aware by construction."""
    y = df["label"].values
    ok = np.column_stack([df[f"ok_{p}"].values for p in pilots]).astype(bool)
    coverage = ok.any(1)
    # oracle picks a correct branch when one exists, else falls back to the visual baseline
    idx = np.where(coverage, ok.argmax(1), pilots.index(C.P0))
    pred = np.column_stack([df[f"pred_{p}"].values for p in pilots])[np.arange(len(df)), idx]
    base_pred = df[f"pred_{C.P0}"].values
    return {
        "n": int(len(df)),
        "branches": pilots,
        "base_accuracy": float((base_pred == y).mean()),
        "base_balanced_accuracy": balanced_accuracy(y, base_pred),
        "oracle_accuracy": float((pred == y).mean()),
        "oracle_balanced_accuracy": balanced_accuracy(y, pred),
        "union_of_correct_coverage": float(coverage.mean()),
        "remaining_shared_error_rate": float((~coverage).mean()),
        "headroom_balanced_accuracy": float(
            balanced_accuracy(y, pred) - balanced_accuracy(y, base_pred)),
    }


def gate_features(df: pd.DataFrame, specialist: str) -> tuple[np.ndarray, list[str]]:
    """Label-free evidence state q for one specialist, relative to the visual baseline.

    Includes cross-branch conflict and vacuity explicitly: the gate's job is to notice
    "these two branches disagree and the visual one is unusually uncertain / unusually
    confident in a way that has been wrong before", which is not visible from either
    branch's score alone.
    """
    v, b = C.P0, specialist
    cols: dict[str, np.ndarray] = {}
    for tag, p in (("vis", v), ("spec", b)):
        cols[f"p_{tag}"] = df[f"p_{p}"].values
        cols[f"u_{tag}"] = df[f"u_{p}"].values
        # distance from the decision boundary: confidence magnitude, direction-free
        cols[f"margin_{tag}"] = np.abs(df[f"p_{p}"].values - 0.5)
        for sub in ("ps", "pa", "us", "ua"):
            if f"{sub}_{p}" in df:
                cols[f"{sub}_{tag}"] = df[f"{sub}_{p}"].values
    cols["conflict"] = np.abs(df[f"p_{v}"].values - df[f"p_{b}"].values)
    cols["disagree"] = (df[f"pred_{v}"].values != df[f"pred_{b}"].values).astype(float)
    cols["u_ratio"] = df[f"u_{b}"].values / (df[f"u_{v}"].values + 1e-8)
    names = list(cols)
    assert_label_free(names)
    return np.column_stack([cols[n] for n in names]), names


def run_gate(df: pd.DataFrame, pilots: list[str], min_per_group: int,
             hidden: int) -> tuple[pd.DataFrame, dict]:
    """A2a: leave-one-generator-out gate, reporting Gate-Recovery per held-out generator."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    specialists = [p for p in pilots if p != C.P0]
    groups = df["method"].values
    y_all = df["label"].values
    rows = []

    # two-class folds: a generator group is fakes-only, so the shared real pool is split
    # into disjoint train/test halves (see common.logo_folds_two_class). Without this every
    # fold is single-class and the whole diagnostic silently returns nothing.
    for held, tr, te in C.logo_folds_two_class(groups, y_all, min_per_group):

        # predicted utility per specialist, on the held-out generator
        util = np.zeros((int(te.sum()), len(specialists)))
        for j, b in enumerate(specialists):
            X, _ = gate_features(df, b)
            # target: does this specialist beat the visual baseline on this sample?
            dloss = df[f"loss_{C.P0}"].values - df[f"loss_{b}"].values
            t = (dloss > 0).astype(int)
            if len(np.unique(t[tr])) < 2:
                continue
            Xtr, Xte = C.fold_scaler(X[tr], X[te])
            clf = (LogisticRegression(max_iter=2000) if hidden <= 0 else
                   MLPClassifier(hidden_layer_sizes=(hidden,), max_iter=1500,
                                 random_state=C.SEED))
            clf.fit(Xtr, t[tr])
            util[:, j] = clf.predict_proba(Xte)[:, 1]

        # route: take the specialist whose predicted utility clears 0.5, else keep P0.
        # Keeping the baseline when no specialist is confidently better is the whole point
        # of an applicability gate -- it must not harm what P0 already gets right.
        best = util.argmax(1)
        use_spec = util.max(1) > 0.5
        pred_stack = np.column_stack([df[f"pred_{p}"].values[te] for p in specialists])
        gate_pred = np.where(use_spec, pred_stack[np.arange(len(best)), best],
                             df[f"pred_{C.P0}"].values[te])

        yte = y_all[te]
        ba_base = balanced_accuracy(yte, df[f"pred_{C.P0}"].values[te])
        ba_gate = balanced_accuracy(yte, gate_pred)
        om = oracle_metrics(df[te], pilots)
        ba_oracle = om["oracle_balanced_accuracy"]
        denom = ba_oracle - ba_base
        rows.append({
            "held_out": held, "n_test": int(te.sum()),
            "ba_base": ba_base, "ba_gate": ba_gate, "ba_oracle": ba_oracle,
            "oracle_headroom": denom,
            "gate_recovery": float((ba_gate - ba_base) / denom) if abs(denom) > 1e-9 else np.nan,
            "routed_fraction": float(use_spec.mean()),
        })

    folds = pd.DataFrame(rows)
    summary = {}
    if not folds.empty:
        valid = folds["gate_recovery"].replace([np.inf, -np.inf], np.nan).dropna()
        summary = {
            "n_folds": int(len(folds)),
            "mean_gate_recovery": float(valid.mean()) if len(valid) else None,
            "median_gate_recovery": float(valid.median()) if len(valid) else None,
            "folds_with_positive_recovery": int((valid > 0).sum()),
            "mean_ba_base": float(folds["ba_base"].mean()),
            "mean_ba_gate": float(folds["ba_gate"].mean()),
            "mean_ba_oracle": float(folds["ba_oracle"].mean()),
            "mean_routed_fraction": float(folds["routed_fraction"].mean()),
        }
    return folds, summary


A2B_PROTOCOL = """# A2b — deployment-valid gate protocol (spec; not run in Phase 1)

A2a answers *can applicability be inferred at all*. It trains the gate on sibling
generators from the same OOD pool, so it is an optimistic ceiling, not a deployable
result. A2b is the protocol the shipped DISCERN gate must satisfy. Any D4 built on A2a
numbers alone carries that caveat in the paper.

## Training data
Only what the generalization protocol permits:
* FF++ train split, and internal manipulation holdouts.
* Approved augmentation of the above.
* **No DF40, CDFv3, or Deepfake-Eval-2024 data at any stage of gate fitting**, including
  scaler and calibrator fitting.

## Inputs (strictly label-free, identical at train and inference)
Permitted: per-branch evidence, alpha, p, vacuity, cross-branch conflict and disagreement,
specialist residual response statistics.
Forbidden as inputs, at both train and inference: the label, `dloss_b`, generator ID,
dataset ID, and forgery-family ID. `assert_label_free()` in `a2_gate.py` enforces the
naming-level check; the protocol-level check is that no OOD-derived column can exist.

## Target
`dloss_b = loss_vis - loss_b` on permitted data only. It is a training target computed from
the label; it is never an input.

## Selection
The threshold and any gate hyperparameters are chosen on the permitted validation slice and
frozen before any OOD source is scored -- one threshold for every OOD dataset and
generator. Note the Table-4 finding: in-domain FF++ validation cannot rank cross-domain
detectors, so a leak-free OOD selection protocol is the honest alternative and is itself a
candidate contribution.

## Reporting
Report A2a and A2b side by side. If A2b recovery collapses relative to A2a, that gap *is*
the finding: applicability is inferable from sibling generators but does not transfer from
the training protocol alone.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", nargs="+", default=["DF40", "CDFv3"])
    ap.add_argument("--candidates", default=None,
                    help="comma-separated branch set; repeatable via multiple runs")
    ap.add_argument("--min-per-group", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=0,
                    help="0 = logistic gate; >0 = one hidden layer of this width")
    args = ap.parse_args()

    out = C.out_dir("A2_gate")
    (out / "A2b_PROTOCOL.md").write_text(A2B_PROTOCOL)

    sets = ([args.candidates.split(",")] if args.candidates else DEFAULT_CANDIDATE_SETS)
    thr, prov = C.frozen_threshold(C.P0)
    results: dict = {"threshold": thr, "threshold_provenance": prov, "sources": {}}
    print(f"frozen threshold = {thr:.4f}  ({prov})\n")

    for source in args.source:
        per_source = {}
        for cand in sets:
            name = "+".join(cand)
            missing = [p for p in cand if not C.has_export(p, source)]
            if missing:
                print(f"[{source}] candidate set {name}: SKIPPED, missing {missing} "
                      f"(UMAR-RUNS)")
                for p in missing:
                    print(f"    cd {C.DICOME} && {C.export_command(p, [source])}")
                per_source[name] = {"status": C.todo(f"missing exports {missing}")}
                continue

            df = build_state(source, cand, thr)
            om = oracle_metrics(df, cand)
            folds, summary = run_gate(df, cand, args.min_per_group, args.hidden)

            sdir = C.out_dir(f"A2_gate/{source}/{name}")
            pd.DataFrame([om]).to_csv(sdir / "oracle_metrics.csv", index=False)
            if not folds.empty:
                folds.to_csv(sdir / "A2a_gate_recovery_folds.csv", index=False)

            # supplementary only, explicitly labelled non-deployable
            y = df["label"].values
            ok = np.column_stack([df[f"ok_{p}"].values for p in cand]).astype(bool)
            idx = np.where(ok.any(1), ok.argmax(1), cand.index(C.P0))
            routed_p = np.column_stack([df[f"p_{p}"].values for p in cand])[
                np.arange(len(df)), idx]
            from sklearn.metrics import roc_auc_score
            (sdir / "SUPPLEMENTARY_oracle_routed_auroc.txt").write_text(
                "LABEL-PEEKING / NON-DEPLOYABLE -- not a headline metric.\n"
                "The oracle consults the label to choose a branch, so this ranking is a\n"
                "function of the label and is not a detector's ranking.\n\n"
                f"oracle_routed_auroc = {roc_auc_score(y, routed_p):.6f}\n")

            per_source[name] = {"oracle": om, "A2a": summary}
            print(f"[{source}] {name}: oracle BA {om['oracle_balanced_accuracy']:.4f} vs "
                  f"base {om['base_balanced_accuracy']:.4f} "
                  f"(coverage {om['union_of_correct_coverage']:.4f}) | "
                  f"gate recovery {summary.get('mean_gate_recovery')}")
        results["sources"][source] = per_source

    (out / "A2_results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {out}/A2_results.json and A2b_PROTOCOL.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

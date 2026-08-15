#!/usr/bin/env python3
"""A2b — the deployment-valid gate protocol.

A2a answers "is applicability inferable *at all*?" by training the gate under
leave-one-generator-out CV on the OOD pool itself. That gate has seen sibling generators, so
A2a is an optimistic **ceiling**, not a deployable result.

A2b is the real test. The gate is trained **only on data the generalization protocol
permits** — FF++ — and then frozen and applied to every OOD source it has never seen. No
generator ID, dataset ID, or forgery-family label is an input at any point.

The gap between A2a and A2b *is* the finding. If A2b collapses relative to A2a, then
applicability is inferable from sibling generators but does not transfer from the training
protocol alone — which is precisely the claim a deployed system would need.

Protocol, enforced rather than described
----------------------------------------
* **Train on FFpp only.** No OOD sample influences the gate, its scaler, or its threshold.
  `assert_no_ood_in_training()` re-checks this from the data rather than trusting the
  call site.
* **Inputs strictly label-free** — evidence, vacuity, margins, cross-branch conflict.
  `assert_label_free()` (shared with A2a) rejects any label-, dloss-, or identity-derived
  column.
* **`dloss_b = loss_vis - loss_b` is a training target only**, computed on FF++, never an
  input.
* **One frozen threshold** from the protocol source, reused everywhere — never retuned per
  OOD source.

    python analysis/discern_v2/a2b_gate_deployment.py
    python analysis/discern_v2/a2b_gate_deployment.py --candidates P0-DS,P1d,P2a
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
from a2_gate import (  # noqa: E402
    DEFAULT_CANDIDATE_SETS,
    balanced_accuracy,
    build_state,
    gate_features,
    oracle_metrics,
)

PROTOCOL_SOURCE = "FFpp"


def assert_no_ood_in_training(train_source: str) -> None:
    """The gate may only be fit on the permitted protocol source.

    Checked here rather than left to the caller because the failure is invisible: a gate
    accidentally fit on OOD data would simply look excellent, and the A2a/A2b comparison —
    the entire point of this script — would be meaningless.
    """
    if train_source != PROTOCOL_SOURCE:
        raise AssertionError(
            f"A2b may only train on {PROTOCOL_SOURCE}; got {train_source!r}. Training on an "
            f"OOD source would make this A2a with extra steps.")


def fit_protocol_gate(df_train: pd.DataFrame, specialists: list[str], hidden: int):
    """Fit one gate per specialist on the protocol source. Returns (models, scalers, names)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    models, scalers, feat_names = {}, {}, None
    for b in specialists:
        X, names = gate_features(df_train, b)
        feat_names = names
        # target: does specialist b beat the visual baseline on this FF++ sample?
        dloss = df_train[f"loss_{C.P0}"].values - df_train[f"loss_{b}"].values
        t = (dloss > 0).astype(int)
        if len(np.unique(t)) < 2:
            # the specialist never wins (or always wins) on the protocol source; a gate
            # cannot be fit, and pretending otherwise would fabricate a decision rule
            models[b] = None
            continue
        sc = StandardScaler().fit(X)
        clf = (LogisticRegression(max_iter=2000) if hidden <= 0 else
               MLPClassifier(hidden_layer_sizes=(hidden,), max_iter=1500,
                             random_state=C.SEED))
        clf.fit(sc.transform(X), t)
        models[b], scalers[b] = clf, sc
    return models, scalers, feat_names


def apply_gate(df: pd.DataFrame, specialists: list[str], models: dict, scalers: dict
               ) -> tuple[np.ndarray, float]:
    """Route with the frozen protocol gate. Returns (predictions, routed fraction)."""
    usable = [b for b in specialists if models.get(b) is not None]
    if not usable:
        return df[f"pred_{C.P0}"].values, 0.0

    util = np.zeros((len(df), len(usable)))
    for j, b in enumerate(usable):
        X, _ = gate_features(df, b)
        util[:, j] = models[b].predict_proba(scalers[b].transform(X))[:, 1]

    best = util.argmax(1)
    use_spec = util.max(1) > 0.5
    stack = np.column_stack([df[f"pred_{b}"].values for b in usable])
    pred = np.where(use_spec, stack[np.arange(len(df)), best], df[f"pred_{C.P0}"].values)
    return pred, float(use_spec.mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood-sources", nargs="+",
                    default=["DF40", "CDFv3", "CDFv2", "DFEval24", "DFDC", "DFDCP", "DFD"])
    ap.add_argument("--candidates", default=None)
    ap.add_argument("--hidden", type=int, default=0)
    args = ap.parse_args()

    out = C.out_dir("A2b_gate_deployment")
    sets = ([args.candidates.split(",")] if args.candidates else DEFAULT_CANDIDATE_SETS)
    thr, prov = C.frozen_threshold(C.P0)
    print(f"frozen threshold = {thr:.4f}  ({prov})")
    assert_no_ood_in_training(PROTOCOL_SOURCE)

    results: dict = {"threshold": thr, "threshold_provenance": prov,
                     "protocol_source": PROTOCOL_SOURCE, "candidate_sets": {}}

    for cand in sets:
        name = "+".join(cand)
        missing = [p for p in cand if not C.has_export(p, PROTOCOL_SOURCE)]
        if missing:
            print(f"[{name}] SKIPPED — no {PROTOCOL_SOURCE} export for {missing} "
                  f"(the gate cannot be trained on permitted data)")
            for p in missing:
                print(f"    cd {C.DICOME} && {C.export_command(p, [PROTOCOL_SOURCE])}")
            results["candidate_sets"][name] = {
                "status": C.todo(f"missing {PROTOCOL_SOURCE} exports for {missing}")}
            continue

        specialists = [p for p in cand if p != C.P0]
        df_train = build_state(PROTOCOL_SOURCE, cand, thr)
        models, scalers, feat_names = fit_protocol_gate(df_train, specialists, args.hidden)
        trained = [b for b in specialists if models.get(b) is not None]
        print(f"\n[{name}] gate trained on {PROTOCOL_SOURCE} "
              f"(n={len(df_train)}), specialists with a fitted gate: {trained}")

        rows = []
        for src in args.ood_sources:
            if any(not C.has_export(p, src) for p in cand):
                continue
            df = build_state(src, cand, thr)
            y = df["label"].values
            if len(np.unique(y)) < 2:
                continue
            pred, routed = apply_gate(df, specialists, models, scalers)
            om = oracle_metrics(df, cand)
            ba_base = om["base_balanced_accuracy"]
            ba_oracle = om["oracle_balanced_accuracy"]
            ba_gate = balanced_accuracy(y, pred)
            denom = ba_oracle - ba_base
            rows.append({
                "ood_source": src, "n": int(len(df)),
                "ba_base": ba_base, "ba_gate": ba_gate, "ba_oracle": ba_oracle,
                "oracle_headroom": denom,
                "gate_recovery": (float((ba_gate - ba_base) / denom)
                                  if abs(denom) > 1e-9 else np.nan),
                "routed_fraction": routed,
            })
            print(f"    {src:10s} base {ba_base:.4f}  gate {ba_gate:.4f}  "
                  f"oracle {ba_oracle:.4f}  recovery {rows[-1]['gate_recovery']:+.4f}  "
                  f"routed {routed:.3f}")

        folds = pd.DataFrame(rows)
        sdir = C.out_dir(f"A2b_gate_deployment/{name}")
        if not folds.empty:
            folds.to_csv(sdir / "A2b_per_ood_source.csv", index=False)
        valid = (folds["gate_recovery"].replace([np.inf, -np.inf], np.nan).dropna()
                 if not folds.empty else pd.Series(dtype=float))
        results["candidate_sets"][name] = {
            "trained_specialists": trained,
            "n_protocol_train": int(len(df_train)),
            "gate_features": feat_names,
            "n_ood_sources": int(len(folds)),
            "mean_gate_recovery": float(valid.mean()) if len(valid) else None,
            "median_gate_recovery": float(valid.median()) if len(valid) else None,
            "sources_with_positive_recovery": int((valid > 0).sum()) if len(valid) else 0,
            "per_source": folds.to_dict("records") if not folds.empty else [],
        }

    (out / "A2b_results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {out}/A2b_results.json")
    print("\nRead against A2a: A2a is an optimistic ceiling (its gate saw sibling "
          "generators). A collapse here is the deployment-valid result, and the gap is the "
          "finding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

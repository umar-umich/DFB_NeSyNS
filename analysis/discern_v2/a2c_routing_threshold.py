#!/usr/bin/env python3
"""A2c — how selective should the gate be?

A2b showed a deployment-valid gate recovering +0.106 of the oracle headroom on average with
P1d, but two things about it are uncomfortable:

  * it routes to a specialist on **80-90%** of samples, so it is barely being selective; and
  * on **DFEval24**, the in-the-wild source with the weakest baseline, it does the most
    harm (-0.290).

The design goal is "exploit specialists without harming what the primary detector already
gets right". A gate that hands off almost everything is not doing that, it is just replacing
the detector. This sweeps the routing threshold to ask whether a stricter gate trades a
smaller mean gain for not hurting the sources that matter most.

    route if   max_b P(specialist b beats the visual baseline)  >  tau

tau = 0.5 is the A2b default (route whenever the specialist is more likely than not to help).
Raising tau demands more confidence before abandoning the baseline.

Everything else is A2b's protocol unchanged: gate trained on FFpp only, frozen, applied to
OOD sources it has never seen, label-free inputs, one frozen decision threshold.

The headline is deliberately NOT the mean across sources. A gate that helps on curated
benchmarks and hurts in the wild is a bad gate for a reliability paper, however good its
average looks, so the worst-source column is reported beside the mean.

    python analysis/discern_v2/a2c_routing_threshold.py
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
from a2_gate import DEFAULT_CANDIDATE_SETS, balanced_accuracy, build_state, oracle_metrics  # noqa: E402
from a2b_gate_deployment import PROTOCOL_SOURCE, fit_protocol_gate, gate_features  # noqa: E402


def route_at_tau(df: pd.DataFrame, specialists: list[str], models: dict, scalers: dict,
                 tau: float) -> tuple[np.ndarray, float]:
    """Route only where the gate clears tau; otherwise keep the visual baseline."""
    usable = [b for b in specialists if models.get(b) is not None]
    if not usable:
        return df[f"pred_{C.P0}"].values, 0.0
    util = np.zeros((len(df), len(usable)))
    for j, b in enumerate(usable):
        X, _ = gate_features(df, b)
        util[:, j] = models[b].predict_proba(scalers[b].transform(X))[:, 1]
    best = util.argmax(1)
    use_spec = util.max(1) > tau
    stack = np.column_stack([df[f"pred_{b}"].values for b in usable])
    pred = np.where(use_spec, stack[np.arange(len(df)), best], df[f"pred_{C.P0}"].values)
    return pred, float(use_spec.mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood-sources", nargs="+",
                    default=["DF40", "CDFv3", "CDFv2", "DFEval24", "DFDC", "DFDCP", "DFD"])
    ap.add_argument("--taus", nargs="+", type=float,
                    default=[0.50, 0.60, 0.70, 0.80, 0.90, 0.95])
    ap.add_argument("--candidates", default=None)
    args = ap.parse_args()

    out = C.out_dir("A2c_routing")
    sets = ([args.candidates.split(",")] if args.candidates else DEFAULT_CANDIDATE_SETS)
    thr, prov = C.frozen_threshold(C.P0)
    print(f"frozen threshold = {thr:.4f}  ({prov})")

    results: dict = {"threshold": thr, "protocol_source": PROTOCOL_SOURCE,
                     "taus": args.taus, "candidate_sets": {}}

    for cand in sets:
        name = "+".join(cand)
        if any(not C.has_export(p, PROTOCOL_SOURCE) for p in cand):
            print(f"[{name}] skipped — missing {PROTOCOL_SOURCE} export")
            continue
        specialists = [p for p in cand if p != C.P0]
        df_train = build_state(PROTOCOL_SOURCE, cand, thr)
        models, scalers, _ = fit_protocol_gate(df_train, specialists, hidden=0)

        # cache per-source state once; the tau sweep only changes the routing rule
        states = {}
        for src in args.ood_sources:
            if any(not C.has_export(p, src) for p in cand):
                continue
            d = build_state(src, cand, thr)
            if len(np.unique(d["label"].values)) < 2:
                continue
            om = oracle_metrics(d, cand)
            states[src] = (d, om)

        rows = []
        for tau in args.taus:
            per_src = {}
            for src, (d, om) in states.items():
                pred, routed = route_at_tau(d, specialists, models, scalers, tau)
                ba_base, ba_oracle = om["base_balanced_accuracy"], om["oracle_balanced_accuracy"]
                ba_gate = balanced_accuracy(d["label"].values, pred)
                denom = ba_oracle - ba_base
                per_src[src] = {
                    "recovery": (float((ba_gate - ba_base) / denom)
                                 if abs(denom) > 1e-9 else np.nan),
                    "delta_ba": float(ba_gate - ba_base),
                    "routed": routed,
                }
            recs = [v["recovery"] for v in per_src.values() if np.isfinite(v["recovery"])]
            deltas = {s: v["delta_ba"] for s, v in per_src.items()}
            worst_src = min(deltas, key=deltas.get) if deltas else None
            rows.append({
                "tau": tau,
                "mean_recovery": float(np.mean(recs)) if recs else np.nan,
                "n_sources_positive": int(sum(r > 0 for r in recs)),
                "n_sources": len(recs),
                # the number that should govern the decision: how badly does the gate hurt
                # the source it hurts most?
                "worst_source": worst_src,
                "worst_delta_ba": float(min(deltas.values())) if deltas else np.nan,
                "mean_routed_fraction": float(np.mean([v["routed"] for v in per_src.values()])),
                "dfeval24_delta_ba": deltas.get("DFEval24", np.nan),
                **{f"rec_{s}": per_src[s]["recovery"] for s in per_src},
            })

        df = pd.DataFrame(rows)
        sdir = C.out_dir(f"A2c_routing/{name}")
        df.to_csv(sdir / "tau_sweep.csv", index=False)
        results["candidate_sets"][name] = df.to_dict("records")

        print(f"\n[{name}]")
        print(f"{'tau':>5} {'mean rec':>9} {'pos':>6} {'routed':>7} "
              f"{'worst delta BA':>15} {'worst source':>14} {'DFEval24 dBA':>13}")
        for r in rows:
            print(f"{r['tau']:>5.2f} {r['mean_recovery']:>9.4f} "
                  f"{r['n_sources_positive']:>2}/{r['n_sources']:<3} "
                  f"{r['mean_routed_fraction']:>7.3f} {r['worst_delta_ba']:>15.4f} "
                  f"{str(r['worst_source']):>14} {r['dfeval24_delta_ba']:>13.4f}")

    (out / "A2c_results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {out}/A2c_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

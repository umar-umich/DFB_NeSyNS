#!/usr/bin/env python3
"""Replay saved branch opinions through three fusion operators — no retraining.

    python analysis/tbiom/fusion_replay.py --arm run1auxedl --out tbiom/RUN1_FUSION.md

THE QUESTION. Run 1 asks two things in order. First, does per-branch supervision keep all three
experts alive? Second, GIVEN live experts, which simple fusion gives the strongest AUC without
worsening real-side FPR? This script answers the second, and it needs no GPU: every view's
Dirichlet opinion is recoverable from the exported (p, u) pair, so the fusion can be swapped
after the fact.

    S = K / u        alpha_fake = p * S        alpha_real = (1 - p) * S

    F1  simple averaging          mean of the per-view probabilities. The plainest baseline a
                                  non-evidential system would use.
    F2  global learned weights    w = softmax(theta), one non-negative weight per view, summing
                                  to 1. Fitted by minimising binary cross-entropy of sum_b w_b p_b.
    F3  Dempster-Shafer           the incumbent operator, replayed from the same opinions.

FIREWALL. F2's weights are fitted on the DEVELOPMENT set only (VALmix by default) and applied
unchanged to every OOD test set. Fitting them on test would be selecting the operator on the
data it is judged against, which is the error that inflated an earlier operational table. The
fit set is named in the report, and OOD exports are never opened during fitting.

If F1 or F2 beats F3, that isolates a SECOND defect: DS combines complementary experts poorly
even once branch withdrawal is fixed. That is the evidence that would motivate a conflict-aware
operator later -- measured here rather than assumed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from health import auroc, dashboard, eer  # noqa: E402
from video_id import video_id  # noqa: E402

S23 = Path("logs/tbiom/stage23")
K = 2
VIEWS = ["semantic", "artifact", "fsvfm"]
DATASETS = ["CDFv2", "CDFv3", "DFD", "DFDC", "DFDCP", "DFEval24", "VALmix"]
P0DS = {"CDFv2": 0.9646, "CDFv3": 0.8409, "DFD": 0.9421, "DFDC": 0.8828,
        "DFDCP": 0.8573, "DFEval24": 0.6922, "VALmix": 0.8852}


def opinion(p: np.ndarray, u: np.ndarray) -> np.ndarray:
    S = K / np.clip(u, 1e-12, None)
    return np.stack([(1.0 - p) * S, p * S], axis=1)


def ds_two(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    out = []
    for a in (a1, a2):
        S = a.sum(axis=1, keepdims=True)
        out.append(((a - 1.0) / S, K / S))
    (b1, u1), (b2, u2) = out
    conflict = (b1.sum(axis=1) * b2.sum(axis=1)) - (b1 * b2).sum(axis=1)
    den = (1.0 - conflict)[:, None]
    b = (b1 * b2 + b1 * u2 + b2 * u1) / den
    u = (u1 * u2) / den
    return b * (K / u) + 1.0


def load(arm: str, ds: str) -> pd.DataFrame | None:
    f = S23 / f"{arm}_{ds}.csv"
    if not f.is_file():
        return None
    d = pd.read_csv(f)
    need = [f"{k}_{v}" for v in VIEWS for k in ("p", "u")]
    if any(c not in d for c in need):
        return None
    d["v"] = video_id(d["key"])
    agg = {f"{k}_{v}": (f"{k}_{v}", "mean") for v in VIEWS for k in ("p", "u")}
    agg["y"] = ("label", "max")
    return d.groupby("v", as_index=False).agg(**agg)


def probs(g: pd.DataFrame) -> dict[str, np.ndarray]:
    return {v: g[f"p_{v}"].to_numpy() for v in VIEWS}


def fuse(g: pd.DataFrame, op: str, w: np.ndarray | None = None) -> np.ndarray:
    P = probs(g)
    if op == "F1":
        return np.mean([P[v] for v in VIEWS], axis=0)
    if op == "F2":
        return np.sum([w[i] * P[v] for i, v in enumerate(VIEWS)], axis=0)
    alphas = [opinion(g[f"p_{v}"].to_numpy(), g[f"u_{v}"].to_numpy()) for v in VIEWS]
    f = alphas[0]
    for a in alphas[1:]:
        f = ds_two(f, a)
    return f[:, 1] / f.sum(axis=1)


def fit_weights(g: pd.DataFrame) -> np.ndarray:
    """softmax-parameterised non-negative weights, minimising BCE on the development set."""
    from scipy.optimize import minimize
    P = np.stack([g[f"p_{v}"].to_numpy() for v in VIEWS], axis=1)
    y = g["y"].to_numpy().astype(float)

    def nll(theta):
        w = np.exp(theta - theta.max()); w /= w.sum()
        p = np.clip(P @ w, 1e-6, 1 - 1e-6)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    r = minimize(nll, np.zeros(len(VIEWS)), method="Nelder-Mead",
                 options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-9})
    w = np.exp(r.x - r.x.max()); return w / w.sum()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", default="run1auxedl")
    ap.add_argument("--fit-on", default="VALmix", help="development set for F2's weights")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or Path(f"tbiom/RUN1_FUSION_{args.arm}.md")

    ffval = load(args.arm, "FFpp_val")
    dev = load(args.arm, args.fit_on)
    if dev is None:
        print(f"no {args.fit_on} export for {args.arm} — F2 cannot be fitted"); return 1
    w = fit_weights(dev)
    print(f"F2 weights fitted on {args.fit_on}: " +
          ", ".join(f"{v}={w[i]:.3f}" for i, v in enumerate(VIEWS)))

    # tau per operator, frozen on FF++ val
    taus = {}
    if ffval is not None:
        for op in ("F1", "F2", "F3"):
            p = fuse(ffval, op, w)
            _, taus[op] = eer(ffval["y"].to_numpy(), p)

    rows = {}
    for ds in DATASETS:
        g = load(args.arm, ds)
        if g is None or g["y"].nunique() < 2:
            continue
        for op in ("F1", "F2", "F3"):
            p = fuse(g, op, w)
            rows[(op, ds)] = dashboard(g["y"].to_numpy(), p, taus.get(op, 0.5))
        for v in VIEWS:
            rows[(v, ds)] = dashboard(g["y"].to_numpy(), g[f"p_{v}"].to_numpy(), 0.5)
            rows[(v, ds)]["u"] = float(g[f"u_{v}"].mean())

    L = [f"# Run 1 — fusion replay ({args.arm})", "",
         "Three operators over the SAME saved branch opinions. No retraining: each view's "
         "Dirichlet opinion is recovered from its exported `(p, u)` via `S = K/u`, so the fusion "
         "can be swapped after the fact.", "",
         f"**F2's weights are fitted on {args.fit_on} only** and applied unchanged to every OOD "
         f"set. No OOD export is opened during fitting.", "",
         "| view | F2 weight |", "|---|---:|"]
    for i, v in enumerate(VIEWS):
        L.append(f"| {v} | {w[i]:.3f} |")
    L += ["", "τ frozen per operator on FF++ val: " +
          ", ".join(f"{o} = {t:.4f}" for o, t in taus.items()), "",
          "## Complete-framework video AUROC", "",
          "| dataset | F1 averaging | F2 learned | F3 DS | best branch | P0-DS |",
          "|---|---:|---:|---:|---:|---:|"]
    for ds in DATASETS:
        if ("F3", ds) not in rows:
            L.append(f"| {ds} | TODO(run) | | | | |")
            continue
        best = max(rows[(v, ds)]["auroc"] for v in VIEWS if (v, ds) in rows)
        cells = [f"{rows[(o, ds)]['auroc']:.4f}" for o in ("F1", "F2", "F3")]
        top = max(range(3), key=lambda i: float(cells[i]))
        cells[top] = f"**{cells[top]}**"
        L.append(f"| {ds} | " + " | ".join(cells) +
                 f" | {best:.4f} | {P0DS.get(ds, float('nan')):.4f} |")
    done = [d for d in DATASETS if ("F3", d) in rows]
    if done:
        L += ["", "| operator | mean AUROC | mean FPR_real | vs P0-DS |", "|---|---:|---:|---:|"]
        mp0 = np.mean([P0DS[d] for d in done])
        for o in ("F1", "F2", "F3"):
            ma = np.mean([rows[(o, d)]["auroc"] for d in done])
            mf = np.mean([rows[(o, d)]["fpr_real_at_tau"] for d in done])
            L.append(f"| {o} | {ma:.4f} | {mf:.3f} | {ma-mp0:+.4f} |")

    L += ["", "## Real-side FPR at frozen τ", "",
          "| dataset | F1 | F2 | F3 DS |", "|---|---:|---:|---:|"]
    for ds in done:
        L.append(f"| {ds} | " + " | ".join(
            f"{rows[(o, ds)]['fpr_real_at_tau']:.3f}" for o in ("F1", "F2", "F3")) + " |")

    L += ["", "## Per-branch vacuity and standalone AUROC", "",
          "| dataset | " + " | ".join(f"{v} AUROC / u" for v in VIEWS) + " |",
          "|---" * (len(VIEWS) + 1) + "|"]
    for ds in done:
        L.append(f"| {ds} | " + " | ".join(
            f"{rows[(v, ds)]['auroc']:.4f} / {rows[(v, ds)]['u']:.3f}"
            if (v, ds) in rows else "—" for v in VIEWS) + " |")

    if done:
        m = {o: np.mean([rows[(o, d)]["auroc"] for d in done]) for o in ("F1", "F2", "F3")}
        win = max(m, key=m.get)
        L += ["", "## Verdict", ""]
        gap = m[win] - m["F3"]
        if win == "F3" or gap < 0.005:
            L.append(f"**No second defect isolated.** Best is {win} at {m[win]:.4f} against DS's "
                     f"{m['F3']:.4f} — a gap of {gap:+.4f}, which is inside the noise band and "
                     f"NOT evidence that DS combines these experts poorly. F1 {m['F1']:.4f}, "
                     f"F2 {m['F2']:.4f}, F3 {m['F3']:.4f}. A difference has to clear ~0.005 "
                     f"before it says anything; anything smaller is operator-choice noise.")
        else:
            L.append(f"**{win} beats DS** ({m[win]:.4f} vs {m['F3']:.4f}, "
                     f"**{m[win]-m['F3']:+.4f}**). That isolates a second problem: DS combines "
                     f"these experts poorly even with branch withdrawal fixed. It is the measured "
                     f"motivation for a conflict-aware operator — subject to the real-side FPR "
                     f"column above not regressing.")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    out.with_suffix(".json").write_text(json.dumps(
        {"weights": dict(zip(VIEWS, w.tolist())), "fit_on": args.fit_on, "taus": taus,
         "rows": {"|".join(k): v for k, v in rows.items()}}, indent=2, default=float))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())

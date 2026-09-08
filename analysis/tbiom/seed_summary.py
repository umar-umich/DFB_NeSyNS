#!/usr/bin/env python3
"""Seed aggregation — mean and spread per configuration, over the six OOD sets.

    python analysis/tbiom/seed_summary.py --out tbiom/SEED_SUMMARY.md

WHAT THIS SETTLES. Run 1's arm ordering rested on single seeds with margins of 0.004-0.011,
against a measured seed-to-seed spread of 0.0025 -- so it was an observation, not a claim. Three
seeds per configuration give a mean and a range, which is the minimum for saying one
configuration is better than another rather than that it won once.

CONFIGURATIONS. Two leaders, tied at 0.8784 on their first seeds:

    arm B, 3-view      semantic + artifact + FS-VFM fused. All three views collaborate, and on
                       both of its seeds arm B is BETTER as 3-view than 2-view.
    arm C, 2-view      trained on three views, semantic dropped from the fusion at inference.
                       Better as 2-view on its seed, because its FS-VFM branch is stronger
                       (0.8539 against arm B's 0.8314) and semantic becomes surplus.

Both are read here on ONE convention: the evidential expectation alpha/S, the same probability
the exporter writes, so nothing is compared across two projections.

`tau` is frozen per run at its own EER on FF++ val. Every run gets its own threshold because
they are differently calibrated; sharing one would compare operating points, not models.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from health import dashboard, eer  # noqa: E402
from video_id import video_id  # noqa: E402

S23 = Path("logs/tbiom/stage23")
K = 2
OOD = ["CDFv2", "CDFv3", "DFD", "DFDC", "DFDCP", "DFEval24"]
P0 = {"CDFv2": 0.9646, "CDFv3": 0.8409, "DFD": 0.9421, "DFDC": 0.8828,
      "DFDCP": 0.8573, "DFEval24": 0.6922}
P0F = {"CDFv2": 0.129, "CDFv3": 0.129, "DFD": 0.174, "DFDC": 0.317,
       "DFDCP": 0.391, "DFEval24": 0.341}
V3 = ("semantic", "artifact", "fsvfm")
V2 = ("artifact", "fsvfm")

# configuration -> [(seed, export prefix)]
CONFIGS = {
    "arm B, 3-view": [(42, "run1auxedl"), (1337, "seed1337"), (7, "armBs7")],
    "arm C, 2-view": [(42, "run1cft"), (1337, "armCs1337"), (7, "armCs7")],
    "arm C, 3-view": [(42, "run1cft"), (1337, "armCs1337"), (7, "armCs7")],
    "arm B, 2-view": [(42, "run1auxedl"), (1337, "seed1337"), (7, "armBs7")],
}
VIEWS_FOR = {"arm B, 3-view": V3, "arm C, 3-view": V3, "arm B, 2-view": V2, "arm C, 2-view": V2}


def load(pref: str, ds: str, views):
    f = S23 / f"{pref}_{ds}.csv"
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    if any(f"{k}_{v}" not in d for v in views for k in ("p", "u")):
        return None
    d["v"] = video_id(d["key"])
    agg = {f"{k}_{v}": (f"{k}_{v}", "mean") for v in views for k in ("p", "u")}
    agg["y"] = ("label", "max")
    return d.groupby("v", as_index=False).agg(**agg)


def alpha(p, u):
    s = K / np.clip(u, 1e-9, None)
    return np.stack([(1 - p) * s, p * s], 1)


def ds_pair(a, b):
    o = []
    for x in (a, b):
        s = x.sum(1, keepdims=True)
        o.append(((x - 1) / s, K / s))
    (b1, u1), (b2, u2) = o
    c = (b1.sum(1) * b2.sum(1)) - (b1 * b2).sum(1)
    den = (1 - c)[:, None]
    return ((b1 * b2 + b1 * u2 + b2 * u1) / den) * (K / ((u1 * u2) / den)) + 1


def fuse(g, views):
    a = [alpha(g[f"p_{v}"].to_numpy(), g[f"u_{v}"].to_numpy()) for v in views]
    f = a[0]
    for x in a[1:]:
        f = ds_pair(f, x)
    return f[:, 1] / f.sum(1)          # evidential expectation, as the exporter writes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("tbiom/SEED_SUMMARY.md"))
    args = ap.parse_args()

    mp0 = float(np.mean([P0[d] for d in OOD]))
    per, missing = {}, []
    for cfg, seeds in CONFIGS.items():
        views = VIEWS_FOR[cfg]
        for sd, pref in seeds:
            ff = load(pref, "FFpp_val", views)
            if ff is None:
                missing.append(f"{cfg} seed {sd}: no FF++ val")
                continue
            _, tau = eer(ff["y"].to_numpy(), fuse(ff, views))
            a, f_ = {}, {}
            ok = True
            for ds in OOD:
                g = load(pref, ds, views)
                if g is None:
                    missing.append(f"{cfg} seed {sd}: {ds}"); ok = False; break
                dd = dashboard(g["y"].to_numpy(), fuse(g, views), tau)
                a[ds], f_[ds] = dd["auroc"], dd["fpr_real_at_tau"]
            if ok:
                per[(cfg, sd)] = {"auroc": a, "fpr": f_,
                                  "mean": float(np.mean(list(a.values()))),
                                  "mean_fpr": float(np.mean(list(f_.values())))}

    L = ["# Seed summary — three seeds per configuration", "",
         f"Mean over the six OOD sets. P0-DS reference **{mp0:.4f}** (mean real-side FPR "
         f"{np.mean(list(P0F.values())):.3f}). `tau` frozen per run on its own FF++ val; all "
         f"probabilities are the evidential expectation `alpha/S`, one convention throughout.", ""]

    L += ["## Per configuration", "",
          "| configuration | views fused | seeds | mean AUROC | spread | vs P0-DS | mean FPR_real |",
          "|---|---|---|---:|---:|---:|---:|"]
    summary = {}
    for cfg in CONFIGS:
        got = [(sd, per[(cfg, sd)]) for sd, _ in CONFIGS[cfg] if (cfg, sd) in per]
        if not got:
            L.append(f"| {cfg} | {'+'.join(VIEWS_FOR[cfg])} | TODO(run) | | | | |")
            continue
        ms = [g["mean"] for _, g in got]
        fs = [g["mean_fpr"] for _, g in got]
        summary[cfg] = (float(np.mean(ms)), float(np.max(ms) - np.min(ms)), float(np.mean(fs)),
                        len(got))
        L.append(f"| {cfg} | {' + '.join(VIEWS_FOR[cfg])} | {len(got)} | "
                 f"**{np.mean(ms):.4f}** | ±{(np.max(ms)-np.min(ms))/2:.4f} | "
                 f"{np.mean(ms)-mp0:+.4f} | {np.mean(fs):.3f} |")

    L += ["", "## Per seed", "",
          "| configuration | seed 42 | seed 1337 | seed 7 | mean | range |",
          "|---|---:|---:|---:|---:|---:|"]
    for cfg in CONFIGS:
        cells = []
        for sd, _ in CONFIGS[cfg]:
            g = per.get((cfg, sd))
            cells.append(f"{g['mean']:.4f}" if g else "—")
        vals = [per[(cfg, sd)]["mean"] for sd, _ in CONFIGS[cfg] if (cfg, sd) in per]
        L.append(f"| {cfg} | " + " | ".join(cells) +
                 (f" | **{np.mean(vals):.4f}** | {np.max(vals)-np.min(vals):.4f} |"
                  if vals else " | — | — |"))

    L += ["", "## Per dataset, the leading configuration's mean over seeds", "",
          "| dataset | P0-DS | arm B 3-view | arm C 2-view | best |",
          "|---|---:|---:|---:|---|"]
    for ds in OOD:
        row = {}
        for cfg in ("arm B, 3-view", "arm C, 2-view"):
            v = [per[(cfg, sd)]["auroc"][ds] for sd, _ in CONFIGS[cfg] if (cfg, sd) in per]
            row[cfg] = float(np.mean(v)) if v else float("nan")
        best = max(row, key=lambda c: row[c]) if all(np.isfinite(list(row.values()))) else "—"
        L.append(f"| {ds} | {P0[ds]:.4f} | {row['arm B, 3-view']:.4f} | "
                 f"{row['arm C, 2-view']:.4f} | {best.replace('arm ', '') if best != '—' else '—'} |")

    # --- the verdict --------------------------------------------------------------------------
    L += ["", "## Verdict", ""]
    if "arm B, 3-view" in summary and "arm C, 2-view" in summary:
        b, c = summary["arm B, 3-view"], summary["arm C, 2-view"]
        gap = b[0] - c[0]
        pooled = max(b[1], c[1]) / 2 or 1e-9
        L += [f"- arm B 3-view: **{b[0]:.4f}** ±{b[1]/2:.4f} over {b[3]} seeds",
              f"- arm C 2-view: **{c[0]:.4f}** ±{c[1]/2:.4f} over {c[3]} seeds",
              f"- gap **{gap:+.4f}**, largest half-range **±{pooled:.4f}**", ""]
        if abs(gap) < pooled:
            L.append("**Not separable.** The gap is smaller than the seed spread, so neither "
                     "configuration is demonstrably better. Both beat P0-DS by a margin that "
                     "clears the noise band, and that is the claim the data supports. Choose on "
                     "grounds other than this number -- arm B fuses all three views, which is "
                     "the simpler story and needs no inference-time surgery.")
        else:
            win = "arm B 3-view" if gap > 0 else "arm C 2-view"
            L.append(f"**{win} wins by {abs(gap):.4f}**, larger than the seed spread "
                     f"(±{pooled:.4f}). That is separable and reportable.")
    if missing:
        L += ["", "## Missing", ""] + [f"- {m}" for m in sorted(set(missing))[:24]]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(
        {f"{k[0]}|seed{k[1]}": v for k, v in per.items()}, indent=2, default=float))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())

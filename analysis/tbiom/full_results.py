#!/usr/bin/env python3
"""The complete grid — every configuration on every test dataset.

    python analysis/tbiom/full_results.py --out tbiom/FULL_RESULTS.md

WHAT THE NUMBERS ARE. Two metrics, both at VIDEO level:

  AUROC       Frames are averaged within a video (the video-identity rule in video_id.py), then
              ROC-AUC is taken over videos. Threshold-free: it measures ranking only.
  FPR_real    Of the REAL videos, the fraction scored at or above tau -- i.e. FALSE ALARMS.
              Lower is better. tau is frozen ONCE per model at its own EER on FF++ val and
              applied unchanged to every test set, so this is a fixed operating point and not
              re-tuned per dataset.

Every probability is the evidential expectation alpha/S, the same quantity the exporter writes.
Multi-seed configurations report the MEAN over seeds and the min-max range.

TEST SETS. Six OOD (never seen in training or selection) plus two references: FF++ val, which is
in-domain and where tau is frozen, and VALmix, the development split used for checkpoint
selection. Neither reference enters an OOD mean.
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
REF = ["VALmix", "FFpp_val"]
ALL = OOD + REF
NICE = {"CDFv2": "Celeb-DF-v2", "CDFv3": "Celeb-DF-v3", "DFD": "DFD", "DFDC": "DFDC",
        "DFDCP": "DFDCP", "DFEval24": "Deepfake-Eval-2024", "VALmix": "VALmix *(dev)*",
        "FFpp_val": "FF++ val *(in-domain)*"}
V3 = ("semantic", "artifact", "fsvfm")
V2 = ("artifact", "fsvfm")

# label -> (list of export prefixes = seeds, views fused or a single branch name)
CONFIGS = [
    ("P0-DS  (2 views, baseline)",            ["p0ds"],                         "P0DS"),
    ("A  fused-only, 3 views",                ["run1fusedonly"],                V3),
    ("B  student Br3, 3 views",               ["run1auxedl", "seed1337", "armBs7"], V3),
    ("B  student Br3, 2 views",               ["run1auxedl", "seed1337", "armBs7"], V2),
    ("C  FSFM Br3, 3 views",                  ["run1cft", "armCs1337", "armCs7"],   V3),
    ("C  FSFM Br3, 2 views",                  ["run1cft", "armCs1337", "armCs7"],   V2),
    ("D  simple CE, 3 views",                 ["run1dce"],                      V3),
    ("E  trained 2-view",                     ["run1e"],                        V2),
    ("  branch: semantic (CLIP)",             ["run1cft", "armCs1337", "armCs7"],   "semantic"),
    ("  branch: artifact (β-VAE)",            ["run1cft", "armCs1337", "armCs7"],   "artifact"),
    ("  branch: FS-VFM",                      ["run1cft", "armCs1337", "armCs7"],   "fsvfm"),
]

# P0-DS has its own export layout
P0_PATH = {d: (Path("logs/tbiom/step1") / f"p0ds_{d}.csv") for d in OOD}
P0_PATH["VALmix"] = Path("logs/tbiom/step2/p0ds_e01_VALmix.csv")
P0_PATH["FFpp_val"] = Path("logs/tbiom/step2/p0ds_e01_FFpp_val.csv")


def read(path: Path, cols):
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    if any(c not in d for c in cols):
        return None
    d["v"] = video_id(d["key"])
    agg = {c: (c, "mean") for c in cols}
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


def score(pref, ds, spec):
    """-> (auroc, fpr) for one seed of one configuration on one dataset."""
    if spec == "P0DS":
        g = read(P0_PATH[ds], ["p_fused"])
        gf = read(P0_PATH["FFpp_val"], ["p_fused"])
        if g is None or gf is None:
            return None
        _, tau = eer(gf["y"].to_numpy(), gf["p_fused"].to_numpy())
        d = dashboard(g["y"].to_numpy(), g["p_fused"].to_numpy(), tau)
        return d["auroc"], d["fpr_real_at_tau"]
    if isinstance(spec, str):                       # a single branch
        cols = [f"p_{spec}"]
        g, gf = read(S23 / f"{pref}_{ds}.csv", cols), read(S23 / f"{pref}_FFpp_val.csv", cols)
        if g is None or gf is None:
            return None
        _, tau = eer(gf["y"].to_numpy(), gf[cols[0]].to_numpy())
        d = dashboard(g["y"].to_numpy(), g[cols[0]].to_numpy(), tau)
        return d["auroc"], d["fpr_real_at_tau"]
    cols = [f"{k}_{v}" for v in spec for k in ("p", "u")]
    g, gf = read(S23 / f"{pref}_{ds}.csv", cols), read(S23 / f"{pref}_FFpp_val.csv", cols)
    if g is None or gf is None:
        return None

    def fuse(t):
        a = [alpha(t[f"p_{v}"].to_numpy(), t[f"u_{v}"].to_numpy()) for v in spec]
        f = a[0]
        for x in a[1:]:
            f = ds_pair(f, x)
        return f[:, 1] / f.sum(1)
    _, tau = eer(gf["y"].to_numpy(), fuse(gf))
    d = dashboard(g["y"].to_numpy(), fuse(g), tau)
    return d["auroc"], d["fpr_real_at_tau"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("tbiom/FULL_RESULTS.md"))
    args = ap.parse_args()

    grid = {}
    for label, prefs, spec in CONFIGS:
        for ds in ALL:
            vals = [r for p in prefs if (r := score(p, ds, spec))]
            if vals:
                grid[(label, ds)] = (np.mean([v[0] for v in vals]),
                                     np.mean([v[1] for v in vals]),
                                     len(vals),
                                     max(v[0] for v in vals) - min(v[0] for v in vals))

    L = ["# Full results — every configuration on every test dataset", "",
         "## What the numbers are", "",
         "**AUROC** — frames averaged within a video, then ROC-AUC over videos. Threshold-free: "
         "it measures ranking only. Higher is better.", "",
         "**FPR_real** — of the REAL videos, the fraction scored at or above `τ`; i.e. **false "
         "alarms**. Lower is better. `τ` is frozen **once per model** at its own EER on FF++ val "
         "and applied unchanged to every test set, so this is a fixed operating point, never "
         "re-tuned per dataset.", "",
         "Six OOD sets, never seen in training or selection. Two references reported but excluded "
         "from every mean: **FF++ val** (in-domain, where `τ` is frozen) and **VALmix** (the "
         "development split used for checkpoint selection).", "",
         "Multi-seed rows are the **mean over 3 seeds** (42 / 1337 / 7); the range is in the "
         "seed-spread table at the end.", ""]

    for metric, idx, better in (("AUROC", 0, "higher"), ("FPR_real @ frozen τ", 1, "lower")):
        L += [f"## {metric}  ({better} is better)", "",
              "| configuration | seeds | " + " | ".join(NICE[d] for d in OOD) +
              " | **OOD mean** | " + " | ".join(NICE[d] for d in REF) + " |",
              "|---|---:|" + "---:|" * (len(OOD) + 1) + "---:|" * len(REF)]
        for label, _p, _s in CONFIGS:
            cells, oodv = [], []
            for d in OOD:
                g = grid.get((label, d))
                cells.append(f"{g[idx]:.4f}" if g else "—")
                if g:
                    oodv.append(g[idx])
            m = f"**{np.mean(oodv):.4f}**" if len(oodv) == len(OOD) else "—"
            refs = []
            for d in REF:
                g = grid.get((label, d))
                refs.append(f"{g[idx]:.4f}" if g else "—")
            n = grid.get((label, OOD[0]), (0, 0, 0, 0))[2]
            L.append(f"| {label} | {n} | " + " | ".join(cells) + f" | {m} | " +
                     " | ".join(refs) + " |")
        L.append("")

    L += ["## Seed spread (max − min of the OOD mean, 3 seeds)", "",
          "| configuration | range |", "|---|---:|"]
    for label, prefs, spec in CONFIGS:
        if len(prefs) < 2:
            continue
        per_seed = []
        for p in prefs:
            v = [r[0] for d in OOD if (r := score(p, d, spec))]
            if len(v) == len(OOD):
                per_seed.append(np.mean(v))
        if len(per_seed) > 1:
            L.append(f"| {label} | {max(per_seed)-min(per_seed):.4f} |")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(
        {f"{k[0]}|{k[1]}": v for k, v in grid.items()}, indent=2, default=float))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())

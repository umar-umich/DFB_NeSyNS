#!/usr/bin/env python3
"""Arms A/B/C — the controlled training comparison, read out.

    python analysis/tbiom/sbi_arms_summary.py --out tbiom/reports/SBI_ARMS.md

READ THE ROWS IN THIS ORDER, because three of the four differences answer different questions
and only one of them is about SBI.

    A - R0    LOADER, NOT TREATMENT. Arm A runs the existing recipe through the new wrapper
              while R0 read the h5 directly. Any gap here is preprocessing and must never be
              attributed to processing exposure or synthesis. If it is large, every downstream
              comparison is measured against A, not R0.
    B - A     processing exposure alone
    C - B     spatial synthesis on top of processing
    C - A     the headline an augmentation paper would quote, and the least informative of the
              three, because it confounds processing with synthesis

Convention is the DS replay of the three branch opinions with tau frozen per run at its own
FF++ val EER -- the same convention as Phase F, the seed summary and the overlay, so these rows
sit beside the existing tables without mixing bases.

A stable AUROC does not imply a stable operating point. Stage 2 measured the two moving in
opposite directions from one frozen threshold (jpeg q30 drove FPR_real to 0.089 while resize
0.35 drove it to 0.584), so FPR_real is reported beside every AUROC and neither is quoted alone.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from health import dashboard, eer  # noqa: E402
import symbolic_overlay as SO  # noqa: E402
from video_id import video_id  # noqa: E402

S23 = Path("logs/tbiom/stage23")
BR = SO.BRANCHES
OOD = SO.OOD
ARMS = [("R0  (historical arm B)", "run1auxedl"),
        ("A   common pipeline", "sbiarmA"),
        ("B   + processing", "sbiarmB"),
        ("C   + SBI 50%", "sbiarmC")]


def load(pref: str, ds: str):
    f = S23 / f"{pref}_{ds}.csv"
    if not f.is_file():
        return None
    d = pd.read_csv(f)
    need = [f"{k}_{v}" for v in BR for k in ("p", "u")] + ["p_fused", "u_fused", "label"]
    if any(c not in d for c in need):
        return None
    d["v"] = video_id(d["key"])
    agg = {c: (c, "mean") for c in need if c != "label"}
    agg["y"] = ("label", "max")
    return d.groupby("v", as_index=False).agg(**agg)


def score(pref: str):
    ff = load(pref, "FFpp_val")
    if ff is None:
        return None
    _, tau = eer(ff["y"].to_numpy(), SO.fuse(ff))
    a, f_ = {}, {}
    for ds in OOD:
        g = load(pref, ds)
        if g is None:
            return None
        d = dashboard(g["y"].to_numpy(), SO.fuse(g), tau)
        a[ds], f_[ds] = d["auroc"], d["fpr_real_at_tau"]
    return {"tau": float(tau), "auroc": a, "fpr": f_,
            "mean": float(np.mean(list(a.values()))),
            "mean_fpr": float(np.mean(list(f_.values())))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("tbiom/reports/SBI_ARMS.md"))
    args = ap.parse_args()

    res = {n: s for n, p in ARMS if (s := score(p))}
    missing = [n for n, p in ARMS if n not in res]

    L = ["# SBI training comparison — arms A / B / C", "",
         "Matched arms, one variable each, all reading the same eligible source pool. `tau` "
         "frozen per run at its own FF++ val EER; probabilities are the DS replay, matching "
         "every other table in the writeup.", ""]
    if len(res) < 2:
        L += ["**Not enough arms exported yet.**", ""] + [f"- missing `{m}`" for m in missing]
        args.out.write_text("\n".join(L) + "\n"); print("\n".join(L)); return 0

    L += ["| arm | mean OOD AUROC | mean FPR_real |", "|---|---:|---:|"]
    for n, _ in ARMS:
        if n in res:
            L.append(f"| {n} | **{res[n]['mean']:.4f}** | {res[n]['mean_fpr']:.3f} |")

    L += ["", "## The differences, in the order they should be read", "",
          "| comparison | ΔAUROC | ΔFPR_real | what it measures |", "|---|---:|---:|---|"]
    pairs = [("A   common pipeline", "R0  (historical arm B)", "**loader, not treatment**"),
             ("B   + processing", "A   common pipeline", "processing exposure alone"),
             ("C   + SBI 50%", "B   + processing", "spatial synthesis alone"),
             ("C   + SBI 50%", "A   common pipeline", "both together (confounded)")]
    for hi, lo, what in pairs:
        if hi in res and lo in res:
            da = res[hi]["mean"] - res[lo]["mean"]
            df = res[hi]["mean_fpr"] - res[lo]["mean_fpr"]
            sig = "**" if abs(da) >= 0.01 else ""
            L.append(f"| {hi.split()[0]} − {lo.split()[0]} | {sig}{da:+.4f}{sig} | "
                     f"{df:+.3f} | {what} |")

    L += ["", "Bold marks |ΔAUROC| ≥ 0.01, the band a single seed can produce in this project. "
          "Smaller differences are reported and not claimed; the seed protocol (42/1337/7) "
          "applies to anything that survives.", "",
          "## Per dataset", "",
          "| dataset | " + " | ".join(n.split()[0] for n, _ in ARMS if n in res) + " |",
          "|---" * (1 + len(res)) + "|"]
    for ds in OOD:
        L.append(f"| {ds} | " + " | ".join(f"{res[n]['auroc'][ds]:.4f}"
                                           for n, _ in ARMS if n in res) + " |")

    # verdict
    if "A   common pipeline" in res and "R0  (historical arm B)" in res:
        d = res["A   common pipeline"]["mean"] - res["R0  (historical arm B)"]["mean"]
        L += ["", "## Verdict", "",
              f"**A − R0 = {d:+.4f}.** " +
              (f"Inside the seed band, so the wrapper is not itself moving the result and "
               f"B and C can be read against either baseline."
               if abs(d) < 0.01 else
               f"**Outside the seed band. The wrapper changes the result on its own**, so "
               f"every later comparison is against A and never against R0, and the gap is a "
               f"preprocessing difference rather than evidence about any treatment.")]
        if "C   + SBI 50%" in res and "B   + processing" in res:
            c = res["C   + SBI 50%"]["mean"] - res["B   + processing"]["mean"]
            L.append("")
            L.append(f"**C − B = {c:+.4f}** is the SBI result proper. " +
                     ("It clears the seed band, so spatial synthesis is doing something "
                      "beyond processing exposure and deserves the three-seed confirmation."
                      if abs(c) >= 0.01 else
                      "It is inside the seed band on one seed, which is not yet a result "
                      "either way."))
    if missing:
        L += ["", "## Not yet exported", ""] + [f"- `{m}`" for m in missing]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(res, indent=2, default=float))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())

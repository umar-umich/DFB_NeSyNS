#!/usr/bin/env python3
"""Frozen detector, per-OOD-dataset evaluation, joined to the overlay's per-set behaviour.

    python analysis/tbiom/ood_evaluation.py

Everything here is a read of results already produced: seed_summary.json for the detector,
SYMBOLIC_OVERLAY.md's S3 table for the overlay. Nothing is re-fitted and no threshold moves,
so this cannot leak an OOD set into a selection decision -- tau was frozen on FF++ val before
any of these numbers existed.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np

OOD = ["CDFv2", "CDFv3", "DFD", "DFDC", "DFDCP", "DFEval24"]
P0 = {"CDFv2": 0.9646, "CDFv3": 0.8409, "DFD": 0.9421, "DFDC": 0.8828,
      "DFDCP": 0.8573, "DFEval24": 0.6922}
P0F = {"CDFv2": 0.129, "CDFv3": 0.129, "DFD": 0.174, "DFDC": 0.317,
       "DFDCP": 0.391, "DFEval24": 0.341}
FROZEN = "arm B, 3-view"
SEEDS = [42, 1337, 7]


def overlay_rows():
    """Pull the S3 per-dataset table rather than recomputing it, so the two reports cannot drift."""
    txt = Path("tbiom/SYMBOLIC_OVERLAY.md").read_text()
    rows = {}
    for m in re.finditer(r"^\| (\w+) \| (\d+) \| ([\d.]+) \| ([\d.]+) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \|$",
                         txt, re.M):
        ds, n, ag, cov, dr, fl, wk, un = m.groups()
        if ds in OOD:
            rows[ds] = dict(n=int(n), agree=float(ag), cov=float(cov), driver=int(dr),
                            flip=int(fl), weak=int(wk), unch=int(un))
    return rows


def main() -> int:
    d = json.load(open("tbiom/SEED_SUMMARY.json"))
    ov = overlay_rows()

    L = ["# OOD evaluation — frozen detector, per test set", "",
         "Frozen configuration: **arm B, three-view, DS fusion, per-branch EDL**, over seeds "
         "42 / 1337 / 7. `tau` is frozen per seed at its own EER on FF++ val; no OOD set was "
         "consulted to set anything here. P0-DS is the published baseline read on the same "
         "convention.", "",
         "## Detector", "",
         "| test set | P0-DS AUROC | frozen AUROC | spread | ΔAUROC | P0-DS FPR_real | frozen FPR_real | ΔFPR |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    aur, fpr = {}, {}
    for ds in OOD:
        a = [d[f"{FROZEN}|seed{s}"]["auroc"][ds] for s in SEEDS]
        f = [d[f"{FROZEN}|seed{s}"]["fpr"][ds] for s in SEEDS]
        aur[ds], fpr[ds] = float(np.mean(a)), float(np.mean(f))
        half = (max(a) - min(a)) / 2
        # |dAUROC| < 0.01 is inside the band a single seed can produce, so mark it rather than
        # letting a 0.003 read as a result (section 22 rule).
        da = aur[ds] - P0[ds]
        mark = "**" if abs(da) >= 0.01 else ""
        L.append(f"| {ds} | {P0[ds]:.4f} | {aur[ds]:.4f} | ±{half:.4f} | {mark}{da:+.4f}{mark} | "
                 f"{P0F[ds]:.3f} | {fpr[ds]:.3f} | {fpr[ds]-P0F[ds]:+.3f} |")
    ma, mf = np.mean(list(aur.values())), np.mean(list(fpr.values()))
    L.append(f"| **mean** | {np.mean(list(P0.values())):.4f} | **{ma:.4f}** | | "
             f"**{ma-np.mean(list(P0.values())):+.4f}** | {np.mean(list(P0F.values())):.3f} | "
             f"**{mf:.3f}** | {mf-np.mean(list(P0F.values())):+.3f} |")
    L += ["", "Bold ΔAUROC marks |Δ| ≥ 0.01, the band a single seed can produce. Smaller "
          "differences are reported but not claimed.", ""]

    L += ["## Overlay, on the same six sets", "",
          "| test set | detector AUROC | rule coverage | driver claims | intervention survival |",
          "|---|---:|---:|---:|---:|"]
    for ds in OOD:
        r = ov.get(ds)
        if not r:
            L.append(f"| {ds} | {aur[ds]:.4f} | — | — | — |"); continue
        surv = (r["flip"] + r["weak"]) / r["driver"] if r["driver"] else float("nan")
        L.append(f"| {ds} | {aur[ds]:.4f} | {r['cov']:.3f} | {r['driver']} | {surv:.3f} |")
    tot_d = sum(r["driver"] for r in ov.values())
    tot_s = sum(r["flip"] + r["weak"] for r in ov.values())
    L.append(f"| **all** | **{ma:.4f}** | {np.mean([r['cov'] for r in ov.values()]):.3f} | "
             f"{tot_d} | **{tot_s/tot_d:.3f}** |")

    up = [ds for ds in OOD if aur[ds] - P0[ds] >= 0.01]
    dn = [ds for ds in OOD if aur[ds] - P0[ds] <= -0.01]
    flat = [ds for ds in OOD if abs(aur[ds] - P0[ds]) < 0.01]
    L += ["", "## What the two tables say together", "",
          f"- **Where the gain comes from.** The +0.0149 mean is not spread evenly. It is "
          f"carried by {' and '.join(up)} (+{aur['CDFv3']-P0['CDFv3']:.4f} and "
          f"+{aur['DFDCP']-P0['DFDCP']:.4f}), paid for partly on "
          f"{' and '.join(dn)} ({aur['CDFv2']-P0['CDFv2']:+.4f}), and flat within the seed band "
          f"on {', '.join(flat)}. Two clear wins, one clear loss, three ties.",
          "- **The real-side improvement is the uniform one.** FPR_real drops on all six sets, "
          f"mean {np.mean(list(P0F.values())):.3f} → {mf:.3f}, with no set worse. That is the "
          "more robust claim than the AUROC mean.",
          "- **CDFv2 and CDFv3 report the same FPR_real by construction**, not by coincidence: "
          "both draw their real videos from the identical Celeb-real pool (5,692 frames, "
          "byte-identical keys apart from the dataset prefix), so any threshold scores them "
          "identically on the real side. Only their fake pools differ.",
          "- **Deepfake-Eval-2024 remains the stated limitation.** At 0.6875 it is 0.19 below "
          "the next-worst set, and its shortfall against P0-DS (−0.0047) is inside the seed "
          "band, so the honest statement is that the frozen model does not improve on "
          "in-the-wild 2024 content rather than that it loses there. The overlay agrees: 3 "
          "driver claims on 814 videos, because the branches almost never hold decisive "
          "evidence on this set.",
          "- **Overlay coverage does not track detector accuracy.** DFDC has the highest "
          "coverage (0.672) at a mid-pack AUROC, CDFv2 the highest AUROC at 0.598 coverage. The "
          "overlay is not simply narrating the samples the detector finds easy.",
          "", "Traces: `tbiom/SYMBOLIC_TRACES.md`. Overlay construction and validation: "
          "`tbiom/SYMBOLIC_OVERLAY.md`. Freeze: `tbiom/BRANCH3_REPORT.md` Phase F."]

    out = Path("tbiom/OOD_EVALUATION.md")
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())

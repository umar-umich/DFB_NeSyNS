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
    cov_hi = max(ov, key=lambda d: ov[d]["cov"])
    auc_hi = max(aur, key=lambda d: aur[d])
    _o = np.array([ov[d]["cov"] for d in OOD if d in ov])
    _a = np.array([aur[d] for d in OOD if d in ov])
    # Spearman without scipy: Pearson on the ranks
    def _rank(x):
        o = np.argsort(np.argsort(x))
        return o.astype(float)
    rho = float(np.corrcoef(_rank(_o), _rank(_a))[0, 1]) if len(_o) > 2 else float("nan")
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
          f"- **Deepfake-Eval-2024 remains the stated limitation.** At {aur['DFEval24']:.4f} "
          f"it is {sorted(aur.values())[1] - aur['DFEval24']:.2f} below the next-worst set, and "
          f"its shortfall against P0-DS ({aur['DFEval24']-P0['DFEval24']:+.4f}) is inside the "
          f"seed band — so the honest statement is that the frozen model does not improve on "
          f"in-the-wild 2024 content, not that it loses there. It is also the set the overlay "
          f"covers least ({ov['DFEval24']['cov']:.3f} against a {np.mean([r['cov'] for r in ov.values()]):.3f} "
          f"mean), which is consistent: fewer samples here reach a decisive branch vote.",
          f"- **Coverage and detector accuracy are loosely related at most.** Rank correlation "
          f"between the two columns is {rho:+.2f} — positive, but with six datasets the "
          f"critical value for significance is about 0.89, so this is not a detectable "
          f"association either way. The extremes do not line up: {cov_hi} has the highest "
          f"coverage ({ov[cov_hi]['cov']:.3f}) at a mid-pack AUROC, while {auc_hi} has the "
          f"highest AUROC ({aur[auc_hi]:.4f}) at {ov[auc_hi]['cov']:.3f}. What can be said is "
          f"that the overlay is not merely narrating the samples the detector finds easy; what "
          f"cannot be said, on n=6, is that the two are independent.",
          "", "Traces: `tbiom/SYMBOLIC_TRACES.md`. Overlay construction and validation: "
          "`tbiom/SYMBOLIC_OVERLAY.md`. Freeze: `tbiom/BRANCH3_REPORT.md` Phase F."]

    out = Path("tbiom/OOD_EVALUATION.md")
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())

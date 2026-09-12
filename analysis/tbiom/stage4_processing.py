#!/usr/bin/env python3
"""Stage 4, third test — processing discrimination. The central empirical claim.

    python analysis/tbiom/stage4_processing.py --out tbiom/reports/EVIDENCE_STAGE4B.md

THE QUESTION, AS THE BRIEF PUTS IT. "Under held-out processing conditions, does the system
reduce genuine-video false positives while retaining sensitivity to manipulation. This is the
property that would justify the whole direction, and it is measured, not assumed."

WHICH DETECTOR THIS DESCRIBES. Arm B, the frozen configuration. It is deliberately run before
the SBI training comparison returns, so there is a baseline reading of the property; if a
retrained arm wins on dev, this must be repeated against that detector and this table
superseded. The writeup has to name the detector, which is why it is named here.

WHAT "HELD-OUT" MEANS HERE. Stage 2 fitted reliability curves on dev at jpeg 70/50/30 and
resize 0.75/0.50/0.35. This test uses jpeg:40 and resize:0.60 on CDFv2 and DFDCP -- severities
that appear in NO fitted curve and datasets that were never used to fit anything. Fitting and
testing share no severity, so the conditions are genuinely unseen.

TWO NUMBERS, KEPT APART, because the brief requires it.

  deployable      FPR_real and TPR at a threshold frozen on BASELINE DEV and applied unchanged.
                  This is what a deployed system actually does when the input degrades.
  retrospective   FPR at matched TPR. Useful for reading discrimination separately from
                  calibration, and NOT a deployable result, because matching TPR after the fact
                  requires knowing the labels.

Lowering false positives by missing more fakes is not an improvement. So FPR_real is never
reported without TPR beside it, and a drop in both is named as what it is.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from health import eer  # noqa: E402
import symbolic_overlay as SO  # noqa: E402
from video_id import video_id  # noqa: E402

S2 = Path("logs/tbiom/stage2")
BR = SO.BRANCHES
HELD_OUT = [("jpeg40", "jpeg q40"), ("rs06", "resize 0.60")]
SETS = ["CDFv2", "DFDCP"]


PREFIX = ""


def load(name: str):
    f = S2 / f"p_{PREFIX}{name}.csv"
    if not f.is_file():
        return None
    d = pd.read_csv(f)
    d["v"] = video_id(d["key"])
    cols = [f"{k}_{v}" for v in BR for k in ("p", "u")] + ["p_fused", "u_fused"]
    agg = {c: (c, "mean") for c in cols if c in d}
    agg["y"] = ("label", "max")
    return d.groupby("v", as_index=False).agg(**agg)


def rates(y, p, tau):
    r, f = y == 0, y == 1
    return (float((p[r] >= tau).mean()) if r.any() else float("nan"),
            float((p[f] >= tau).mean()) if f.any() else float("nan"))


def fpr_at_tpr(y, p, target):
    """Smallest FPR achievable while holding TPR >= target. Retrospective by construction."""
    r, f = y == 0, y == 1
    if not r.any() or not f.any():
        return float("nan")
    thr = np.quantile(p[f], max(0.0, 1.0 - target))     # threshold giving that TPR on fakes
    return float((p[r] >= thr).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("tbiom/reports/EVIDENCE_STAGE4B.md"))
    ap.add_argument("--prefix", default="",
                    help="export prefix, e.g. armA_ ; empty reads the frozen arm B files")
    ap.add_argument("--detector", default="arm B, the frozen configuration")
    args = ap.parse_args()
    global PREFIX
    PREFIX = args.prefix

    dev = load("FFpp_val_base")
    if dev is None:
        print("no baseline dev export"); return 1
    _, tau = eer(dev["y"].to_numpy(), SO.fuse(dev))

    L = ["# Stage 4 — processing discrimination (the central test)", "",
         f"**Detector: {args.detector}.** The detector is named because this property is a "
         f"property OF a detector, and a table that does not say which one invites being "
         f"quoted about the wrong model.", "",
         f"`tau` = {tau:.4f}, the EER on **baseline** FF++ val, applied unchanged everywhere "
         "below. Held-out severities (jpeg q40, resize 0.60) appear in no fitted Stage 2 curve.",
         "", "## Deployable: frozen threshold, never refitted", "",
         "| dataset | condition | FPR_real | TPR | ΔFPR | ΔTPR |",
         "|---|---|---:|---:|---:|---:|"]

    rows, verdict_bits = [], []
    for ds in SETS:
        base = load(f"{ds}_base")
        if base is None:
            continue
        y0 = base["y"].to_numpy(); p0 = SO.fuse(base)
        f0, t0 = rates(y0, p0, tau)
        L.append(f"| {ds} | baseline | {f0:.3f} | {t0:.3f} | — | — |")
        for tag, nice in HELD_OUT:
            g = load(f"{ds}_{tag}")
            if g is None:
                continue
            y = g["y"].to_numpy(); p = SO.fuse(g)
            fp, tp = rates(y, p, tau)
            L.append(f"| {ds} | {nice} | {fp:.3f} | {tp:.3f} | {fp-f0:+.3f} | {tp-t0:+.3f} |")
            rows.append({"dataset": ds, "cond": nice, "fpr": fp, "tpr": tp,
                         "dfpr": fp - f0, "dtpr": tp - t0})

    L += ["", "## Retrospective: FPR at matched TPR (not deployable)", "",
          "Matching TPR after the fact needs the labels, so this reads discrimination "
          "separately from calibration and is not an operating point anyone could set.", "",
          "| dataset | condition | FPR @ TPR=0.80 | FPR @ TPR=0.90 |", "|---|---|---:|---:|"]
    for ds in SETS:
        for tag, nice in [("base", "baseline")] + HELD_OUT:
            g = load(f"{ds}_{tag}")
            if g is None:
                continue
            y = g["y"].to_numpy(); p = SO.fuse(g)
            L.append(f"| {ds} | {nice} | {fpr_at_tpr(y,p,0.80):.3f} | "
                     f"{fpr_at_tpr(y,p,0.90):.3f} |")

    # ---- does fusion resist processing better than its branches?
    L += ["", "## Does fusion resist processing better than any single view?", "",
          "The claim that would justify a multi-view design is that combining views is more "
          "robust than the best view alone, not merely more accurate on clean data.", "",
          "| dataset | condition | " + " | ".join(SO.LABEL[b] for b in BR) + " | Fused |",
          "|---|---|" + "---:|" * (len(BR) + 1)]
    for ds in SETS:
        base = load(f"{ds}_base")
        if base is None:
            continue
        b0 = {b: rates(base["y"].to_numpy(), base[f"p_{b}"].to_numpy(), tau)[0] for b in BR}
        f0 = rates(base["y"].to_numpy(), SO.fuse(base), tau)[0]
        for tag, nice in HELD_OUT:
            g = load(f"{ds}_{tag}")
            if g is None:
                continue
            y = g["y"].to_numpy()
            cells = [f"{rates(y, g[f'p_{b}'].to_numpy(), tau)[0] - b0[b]:+.3f}" for b in BR]
            cells.append(f"**{rates(y, SO.fuse(g), tau)[0] - f0:+.3f}**")
            L.append(f"| {ds} | {nice} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("Cells are the CHANGE in FPR_real from that dataset's own baseline, so a smaller "
             "number is a view that resists the processing better.")

    # ---- verdict
    up = [r for r in rows if r["dfpr"] > 0.02]
    down_tpr = [r for r in rows if r["dtpr"] < -0.05]
    L += ["", "## Verdict", ""]
    if not rows:
        L.append("No held-out exports found.")
    else:
        mean_dfpr = float(np.mean([r["dfpr"] for r in rows]))
        mean_dtpr = float(np.mean([r["dtpr"] for r in rows]))
        L.append(
            f"**Mean ΔFPR_real {mean_dfpr:+.3f}, mean ΔTPR {mean_dtpr:+.3f}** across "
            f"{len(rows)} held-out conditions.")
        L.append("")
        if mean_dfpr > 0.02:
            L.append(
                f"**The system does not resist held-out processing.** False alarms on genuine "
                f"video rise {mean_dfpr:+.3f} at a threshold that was never refitted, on "
                f"{len(up)} of {len(rows)} conditions. That is the nuisance-induced false "
                f"positive the whole direction exists to fix, and the frozen detector does not "
                f"fix it — which is the finding, and it says improve the measurements before "
                f"building a reasoning layer on top of them.")
        elif mean_dtpr < -0.05:
            L.append(
                f"**False positives are controlled, but by missing fakes.** FPR moves "
                f"{mean_dfpr:+.3f} while TPR falls {mean_dtpr:+.3f}. Lowering false positives "
                f"by losing sensitivity is not an improvement, and reporting the FPR alone "
                f"would have looked like one.")
        else:
            L.append(
                f"**The system holds up under held-out processing**: FPR {mean_dfpr:+.3f} with "
                f"TPR {mean_dtpr:+.3f}, so false alarms are contained without trading away "
                f"sensitivity. This is the property that would justify the direction, measured "
                f"on unseen severities and unseen datasets.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(
        {"tau": float(tau), "rows": rows}, indent=2, default=float))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())

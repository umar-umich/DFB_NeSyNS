"""T13c — the separation plot (the acceptance artifact, not the number).

Overlaid distributions of true-region NM vs pooled control NM, per method, with
the chosen region threshold marked. This is what goes in the paper — visible
separation is the GATE T13 pass criterion, not the threshold value itself.

Reads results/region_gate_calib/region_gate_calib.json (from calibrate_region_gate.py).

Run:
    /data/umar/miniconda3/envs/GenD/bin/python cec/scripts/plot_region_separation.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CALIB_DIR = REPO / "results" / "region_gate_calib"
IOU_TRUE, IOU_CTRL = 0.30, 0.05


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument", default="fsfm")
    args = ap.parse_args()
    CALIB = CALIB_DIR / f"region_gate_calib_{args.instrument}.json"
    if not CALIB.exists():
        print(f"no calibration file at {CALIB}; run calibrate_region_gate.py --instrument {args.instrument}")
        return 1
    d = json.loads(CALIB.read_text())
    rows = d["rows"]
    methods = sorted({r["method"] for r in rows})
    gap = d["derived"]["gap"]
    margin = d["derived"]["margin"]

    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 4.2), squeeze=False)
    for ax, method in zip(axes[0], methods):
        mrows = [r for r in rows if r["method"] == method]
        true_nm = [r["nm"] for r in mrows if r["iou_gt"] >= IOU_TRUE]
        ctrl_nm = [max(r["nm_blur"], r["nm_shift"]) for r in mrows if r["iou_gt"] <= IOU_CTRL]
        bins = np.linspace(-0.2, 0.8, 40)
        ax.hist(ctrl_nm, bins=bins, alpha=0.55, label=f"control NM (n={len(ctrl_nm)})", color="#888")
        ax.hist(true_nm, bins=bins, alpha=0.55, label=f"true-region NM (n={len(true_nm)})", color="#1a7f37")
        ax.axvline(margin, ls="--", c="k", lw=1, label=f"margin {margin:.2f}")
        ax.axvline(gap, ls=":", c="crimson", lw=1, label=f"gap {gap:.2f}")
        ax.set_title(f"{method}"); ax.set_xlabel("necessity margin"); ax.legend(fontsize=8)
    fig.suptitle(f"Region-gate separation ({d['instrument']}, {d['split']} split) — "
                 f"true-region pass {d['true_region_pass_rate']:.0%} / "
                 f"control pass {d['control_region_pass_rate']:.0%}", y=1.02)
    fig.tight_layout()
    out = CALIB.parent / f"region_separation_{args.instrument}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"[written] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

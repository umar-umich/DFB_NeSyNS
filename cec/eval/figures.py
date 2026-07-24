"""T22 — paper figures. Everything regenerated from records/results; nothing hand-entered.

1. NM localization vs GT mask (NeuralTextures) — per-region necessity profile
   (mouth spike) against the diffuse GT mask overlap. THE headline figure: the
   causal signal localizes the manipulation better than the dataset's own
   annotation.
2. Separation plot — see cec/scripts/plot_region_separation.py (the region-gate
   justification artifact).
3. FP-claim vs coverage, base -> tuned (from Pilot D).
4. Qualitative triptych — certified region · composite · abstention (real).

Run:
    /data/umar/miniconda3/envs/GenD/bin/python cec/eval/figures.py --which 1 3
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

OUT = REPO / "results" / "figures"
CALIB = REPO / "results" / "region_gate_calib" / "region_gate_calib.json"
PILOT_D = REPO / "results" / "pilot_d" / "pilot_d.json"


def fig1_nm_localization(method="NeuralTextures", instrument="fsfm"):
    """Per-region NM vs per-region GT-mask overlap — the localization contrast.

    The caption is DERIVED from the data (top region by median NM), never asserted:
    if the causal signal does not concentrate, the figure must say so.
    """
    calib = CALIB.parent / f"region_gate_calib_{instrument}.json"
    if not calib.exists():
        print(f"[fig1] missing {calib}; run calibrate_region_gate.py --instrument {instrument}")
        return
    rows = [r for r in json.loads(calib.read_text())["rows"] if r["method"] == method]
    if not rows:
        print(f"[fig1] no rows for {method}")
        return
    nm, iou = defaultdict(list), defaultdict(list)
    for r in rows:
        nm[r["region"]].append(r["nm"])
        iou[r["region"]].append(r["iou_gt"])
    regions = sorted(nm, key=lambda r: -np.median(nm[r]))
    nm_med = [np.median(nm[r]) for r in regions]
    iou_med = [np.median(iou[r]) for r in regions]

    fig, ax = plt.subplots(figsize=(11, 4.4))
    x = np.arange(len(regions))
    ax.bar(x - 0.2, nm_med, 0.4, label="necessity margin (causal signal)", color="#1a7f37")
    ax.bar(x + 0.2, iou_med, 0.4, label="IoU with GT mask (dataset annotation)", color="#999")
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("_", "\n") for r in regions], fontsize=8)
    ax.set_ylabel("median value")
    ax.axhline(0, c="k", lw=0.6)
    # Caption derived from the data — no asserted claim about which region wins.
    top_region, top_nm = regions[0], nm_med[0]
    second_nm = nm_med[1] if len(nm_med) > 1 else 0.0
    concentrated = second_nm <= 0 or top_nm >= 2 * max(second_nm, 1e-9)
    verdict = (f"NM concentrates at '{top_region}' ({top_nm:.3f}, "
               f"{top_nm / max(second_nm, 1e-9):.1f}x the next region)" if concentrated
               else f"NM does NOT concentrate (top '{top_region}' {top_nm:.3f} vs next {second_nm:.3f})")
    ax.set_title(f"{method} · {instrument}: causal signal vs dataset annotation\n"
                 f"{verdict}; GT-mask IoU is diffuse, max {max(iou_med):.2f}", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = OUT / "fig1_nm_localization.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig1] {out}")


def fig3_fp_vs_coverage():
    """FP-claim rate vs coverage, base -> tuned (the pair that must move together)."""
    if not PILOT_D.exists():
        print(f"[fig3] missing {PILOT_D}; run Pilot D (T21) first")
        return
    d = json.loads(PILOT_D.read_text())
    b, t = d["base"], d["tuned"]
    fig, ax = plt.subplots(figsize=(5.4, 5))
    ax.scatter([b["fp_claim_rate"]], [b["coverage"]], s=110, c="#999", label="base", zorder=3)
    ax.scatter([t["fp_claim_rate"]], [t["coverage"]], s=110, c="#1a7f37", label="tuned", zorder=3)
    ax.annotate("", xy=(t["fp_claim_rate"], t["coverage"]),
                xytext=(b["fp_claim_rate"], b["coverage"]),
                arrowprops=dict(arrowstyle="->", lw=1.6, color="#444"))
    ax.set_xlabel("FP-claim rate on reals (lower is better)")
    ax.set_ylabel("coverage on fakes (higher is better)")
    ax.set_title(f"DPO effect: {d.get('verdict','')}", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = OUT / "fig3_fp_vs_coverage.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig3] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", default=["1", "3"])
    ap.add_argument("--method", default="NeuralTextures")
    ap.add_argument("--instrument", default="fsfm")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if "1" in args.which:
        fig1_nm_localization(args.method, args.instrument)
    if "3" in args.which:
        fig3_fp_vs_coverage()
    print("(fig2 = cec/scripts/plot_region_separation.py · fig4 qualitative = "
          "render via cec.assembly.DisclosurePolicy on chosen records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

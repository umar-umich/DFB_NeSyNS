"""T13b — calibrate the REGION-scope gate thresholds.

The frozen `gate:` block was calibrated on FULL-MASK repair (drops 0.65-0.84).
The gate's REGION scope repairs a single landmark region (drops ~0.15), so those
thresholds are unreachable there. This re-derives the SAME rules at REGION scale,
on the TRAIN split (disjoint from the rev3 test split — codebook rev-3 rule).

Per image, per non-empty landmark region, it reuses `CertificationGate.measure_region`
(the EXACT ops the gate uses — no drift) and records NM + control NMs + the region's
IoU with the GT manipulation mask. Regions are labeled:
    true    iou_gt >= 0.30   (region overlaps the manipulation)
    control iou_gt <= 0.05   (region off the manipulation)

Thresholds (mirroring the frozen derivation):
    margin_region = max(0.10, ceil_0.05(p99(pooled control drops {blur,shift,wrong})))
    gap_region    = max(0.05, ceil_0.05(p99(control-region contrast nm-max(blur,shift))))
    wrong / offset re-measured; kept at 0.05 / 0.15 unless the region p99 demands more.

Prints a ready-to-paste `gate_region:` block. Does NOT edit params.yaml (freeze
discipline — Umar pastes it and flips calibrated:true). Also writes the raw
measurements for the T13c separation plot.

Run (🔴 UMAR-RUNS):
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    /data/umar/miniconda3/envs/GenD/bin/python cec/scripts/calibrate_region_gate.py \
        --split train --n 60 --instrument fsfm
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from cec.data.pairing import build_pair, fake_dir, split_video_ids  # noqa: E402
from cec.gate.certify import CertificationGate  # noqa: E402
from cec.masks.regions import region_mask  # noqa: E402
from cec.registration.vocab import SPATIAL_REGIONS  # noqa: E402
from cec.repair import Lpips  # noqa: E402

SEED = 0
OUT = REPO / "results" / "region_gate_calib"
# Region regime = localized methods; full-face methods go to composite (gate:).
LOCALIZED = ["NeuralTextures", "Face2Face"]
FULLFACE = ["Deepfakes", "FaceSwap", "DeepFakeDetection"]
IOU_TRUE, IOU_CTRL = 0.30, 0.05
MARGIN_FLOOR = 0.05   # T19-T22 decision: region-scale floor (control-p99 rule)


def ceil_05(x):
    return math.ceil(max(x, 0.0) / 0.05) * 0.05


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def gather(methods, split_vids, n, rng):
    pairs = []
    per = max(1, n // len(methods))
    for method in methods:
        fdir = fake_dir(method) / "frames"
        vids = [v.name for v in fdir.iterdir() if v.name in split_vids]
        rng.shuffle(vids)
        got = 0
        for vid in vids:
            if got >= per:
                break
            frames = sorted(p.stem for p in (fdir / vid).glob("*.png"))
            if not frames:
                continue
            pair = build_pair(method, vid, frames[len(frames) // 2])
            if pair is not None:
                pairs.append(pair)
                got += 1
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=60, help="images per method")
    ap.add_argument("--instrument", default="fsfm")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    split_vids = split_video_ids(args.split)

    gate = CertificationGate(args.instrument, device=args.device, lpips=Lpips(args.device))
    pairs = gather(LOCALIZED, split_vids, args.n, rng)
    print(f"[calib] {args.instrument} · split={args.split} · {len(pairs)} localized images "
          f"({LOCALIZED})")

    rows = []
    for i, pair in enumerate(pairs):
        for r in sorted(SPATIAL_REGIONS):
            rm = region_mask(r, pair.landmarks)
            if rm is None or rm.sum() == 0:
                continue
            m = gate.measure_region(pair, rm)
            if m is None:
                continue
            m.update(region=r, method=pair.method, iou_gt=iou(rm, pair.mask))
            rows.append(m)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(pairs)}")

    true_rows = [r for r in rows if r["iou_gt"] >= IOU_TRUE]
    ctrl_rows = [r for r in rows if r["iou_gt"] <= IOU_CTRL]

    # pooled control drops = {blur, shift, wrong} over control regions
    pooled_ctrl = np.array(
        [v for r in ctrl_rows for v in (r["nm_blur"], r["nm_shift"], r["nm_wrong"])]
    )
    ctrl_contrast = np.array([r["nm"] - max(r["nm_blur"], r["nm_shift"]) for r in ctrl_rows])
    true_contrast = np.array([r["nm"] - max(r["nm_blur"], r["nm_shift"]) for r in true_rows])
    wrong_ctrl = np.abs([r["nm_wrong"] for r in ctrl_rows])
    offset_all = np.array([r["real_offset"] for r in rows])

    def p99(a):
        return float(np.percentile(a, 99)) if len(a) else 0.0

    margin_region = max(MARGIN_FLOOR, ceil_05(p99(pooled_ctrl)))
    gap_region = max(0.05, ceil_05(p99(ctrl_contrast)))
    wrong_region = max(0.05, ceil_05(p99(wrong_ctrl)))
    offset_region = max(0.15, ceil_05(p99(offset_all)))

    # true-region pass-rate at the derived bar (the "is this bar meaningful" check)
    def passes(r):
        return (r["nm"] >= margin_region
                and (r["nm"] - max(r["nm_blur"], r["nm_shift"])) >= gap_region
                and abs(r["nm_wrong"]) <= wrong_region and r["real_offset"] <= offset_region)
    true_pass = float(np.mean([passes(r) for r in true_rows])) if true_rows else 0.0
    ctrl_pass = float(np.mean([passes(r) for r in ctrl_rows])) if ctrl_rows else 0.0

    result = {
        "instrument": args.instrument, "split": args.split, "seed": SEED,
        "n_images": len(pairs), "n_region_samples": len(rows),
        "n_true": len(true_rows), "n_control": len(ctrl_rows),
        "p99": {"pooled_control": p99(pooled_ctrl), "ctrl_contrast": p99(ctrl_contrast),
                "wrong_control": p99(wrong_ctrl), "offset": p99(offset_all)},
        "derived": {"margin": margin_region, "gap": gap_region,
                    "wrong_region_inert_max": wrong_region, "real_offset_max": offset_region},
        "true_region_pass_rate": round(true_pass, 3),
        "control_region_pass_rate": round(ctrl_pass, 3),
        "true_contrast_median": round(float(np.median(true_contrast)), 4) if len(true_contrast) else None,
        "ctrl_contrast_median": round(float(np.median(ctrl_contrast)), 4) if len(ctrl_contrast) else None,
        "rows": rows,  # for the T13c separation plot
    }
    (OUT / f"region_gate_calib_{args.instrument}.json").write_text(json.dumps(result, indent=2))

    print(f"\n[calib] region samples: {len(rows)} (true {len(true_rows)} / control {len(ctrl_rows)})")
    print(f"  p99 pooled-control={result['p99']['pooled_control']:.4f}  "
          f"ctrl-contrast={result['p99']['ctrl_contrast']:.4f}")
    print(f"  true-region pass-rate {true_pass:.1%} · control-region pass-rate {ctrl_pass:.1%}")
    print(f"  true contrast median {result['true_contrast_median']} vs "
          f"control {result['ctrl_contrast_median']}")

    print("\n--- ready-to-paste gate_region: block (verify, then flip calibrated:true) ---")
    print(f"""gate_region:
  calibrated: true
  calibration_split: {args.split}
  source: results/region_gate_calib/region_gate_calib.json
  certify_margin:
    {args.instrument}: {margin_region:.2f}
  wrong_region_inert_max: {wrong_region:.2f}
  real_offset_max: {offset_region:.2f}
  matched_corruption_gap: {gap_region:.2f}""")
    print(f"\n[written] {OUT / f'region_gate_calib_{args.instrument}.json'}  (feeds T13c separation plot)")
    print("GATE T13: PASS if true vs control contrast visibly separated at the bar; "
          "else report that method region-certification unreliable (do NOT lower the bar).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

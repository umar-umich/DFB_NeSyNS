"""T14b — DF40 composite-pairing feasibility probe (gated; do NOT assume it works).

Composite scope needs no GT mask (inner-face is landmark-derived) but DOES need a
geometry-consistent paired real. DF40 `ff` swaps use FF++-compatible `<id1_id2>`
naming, so a paired real might be recoverable from the FF++ original.

FINDING that shapes this probe: DF40 `ff` landmarks are 81-point and in CROP space
(values within the 256 crop), NOT raw-frame space — so the raw re-crop trick from
`pairing.py` cannot be applied directly. The viable variant is a CROP-TO-CROP
affine: extract the 5 canonical points from DF40's 81-pt set (dlib-68 layout
assumed for indices 0-67), extract the 5-pt from the FF++ preprocessed REAL crop
of the same target identity, fit an affine (FF++ real -> DF40 crop geometry),
warp, and measure bg-MSE outside the inner-face against the DF40 fake crop.

If the 81-pt layout assumption is wrong or the crops don't register, bg-MSE will
be high and the gate FAILS — a valid, documented outcome (DF40 stays out; no
re-preprocessing of the 159 GB tree).

Run (🔴 UMAR-RUNS):
    /data/umar/miniconda3/envs/GenD/bin/python cec/pilots/pilot_df40/probe_pairing.py \
        --families simswap inswap faceswap --n 50
"""
import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from cec.masks.regions import composite_mask  # noqa: E402

DF40 = Path("/data/umar/Datasets/df40/test")
FFPP = Path("/data/umar/Datasets/preprocessed/FaceForensics++")
QC_MSE = 60.0
SEED = 0

# dlib-68 canonical indices (assumed to hold for the first 68 of DF40's 81 points).
DLIB5 = {"le": list(range(36, 42)), "re": list(range(42, 48)),
         "nose": [30], "lm": [48], "rm": [54]}
TEMPLATE = np.array([[0.34, 0.46], [0.66, 0.46], [0.50, 0.64],
                     [0.37, 0.82], [0.63, 0.82]], dtype=np.float32) * 256


def five_from_68(lm81):
    return np.array([lm81[idx].mean(axis=0) for idx in
                     (DLIB5["le"], DLIB5["re"], DLIB5["nose"], DLIB5["lm"], DLIB5["rm"])],
                    dtype=np.float32)


def ffpp_real_crop_and_5pt(target_id, frame):
    """The FF++ preprocessed youtube crop + its 5-pt landmarks for (id, frame)."""
    base = FFPP / "original_sequences" / "youtube" / "c23"
    fp = base / "frames" / target_id / f"{frame}.png"
    lp = base / "landmarks" / target_id / f"{frame}.npy"
    if not fp.exists() or not lp.exists():
        return None, None
    return cv2.imread(str(fp)), np.load(lp).astype(np.float32)


def probe_family(family, n, rng):
    fdir = DF40 / family / "ff" / "frames"
    ldir = DF40 / family / "ff" / "landmarks"
    if not fdir.is_dir():
        return {"family": family, "error": "no ff/frames"}
    vids = sorted(p.name for p in fdir.iterdir())
    rng.shuffle(vids)
    bg_mses, attempted, passing = [], 0, 0
    for vid in vids:
        if attempted >= n:
            break
        target_id = vid.split("_")[0]
        frames = sorted(p.stem for p in (fdir / vid).glob("*.png"))
        if not frames:
            continue
        frame = frames[len(frames) // 2]
        lp = ldir / vid / f"{frame}.npy"
        if not lp.exists():
            continue
        fake = cv2.imread(str(fdir / vid / f"{frame}.png"))
        lm81 = np.load(lp).astype(np.float32)
        real_crop, real5 = ffpp_real_crop_and_5pt(target_id, frame)
        if fake is None or real_crop is None or lm81.shape[0] < 68:
            continue
        attempted += 1
        df40_5 = five_from_68(lm81)
        M = cv2.estimateAffinePartial2D(real5, df40_5, method=cv2.LMEDS)[0]
        if M is None:
            continue
        warped = cv2.warpAffine(real_crop, M, (256, 256), flags=cv2.INTER_LINEAR)
        mask = composite_mask(df40_5)  # inner-face from the DF40 5-pt
        if mask is None:
            continue
        outside = ~mask
        bg = float(((fake.astype(float) - warped.astype(float))[outside] ** 2).mean())
        bg_mses.append(bg)
        passing += bg <= QC_MSE
    return {"family": family, "n_attempted": attempted, "n_passing_QC": passing,
            "median_bg_mse": round(float(np.median(bg_mses)), 2) if bg_mses else None,
            "pass_rate": round(passing / attempted, 3) if attempted else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=["simswap", "inswap", "faceswap"])
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()
    rng = random.Random(SEED)
    out_dir = REPO / "results" / "pilot_df40"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [probe_family(f, args.n, rng) for f in args.families]
    (out_dir / "probe_pairing.json").write_text(json.dumps(rows, indent=2))
    print(f"{'family':<12}{'attempted':>10}{'pass_QC':>9}{'median_bg_mse':>15}{'pass_rate':>11}")
    for r in rows:
        print(f"{r['family']:<12}{r.get('n_attempted','-'):>10}{r.get('n_passing_QC','-'):>9}"
              f"{str(r.get('median_bg_mse')):>15}{str(r.get('pass_rate')):>11}")
    overall = [r["pass_rate"] for r in rows if r.get("pass_rate") is not None]
    mean_pass = float(np.mean(overall)) if overall else 0.0
    print(f"\nGATE T14b: {'PASS' if mean_pass >= 0.60 else 'FAIL'} "
          f"(mean pass-rate {mean_pass:.0%}; >=60% -> DF40 ff joins composite arm).")
    print(f"[written] {out_dir / 'probe_pairing.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

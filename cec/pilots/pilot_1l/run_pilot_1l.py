"""TASK 10 — PILOT 1L: does region repair recover the pixel-mask necessity, and
how many claims can certify (the DPO positive pool)?

Two measurements per fake, on the primary instrument (fsfm):
  ceiling   NM from repairing the FULL GT manipulation mask (the pilot's ~0.71
            for fsfm) — proves the pipeline CAN certify and sets the upper bound.
  region    NM from repairing each landmark region the codebook offers.
            recovery = region_NM / ceiling_NM. Task 10 asks: >= 70%?
  pool      count of (image, region) pairs that CERTIFY under the frozen gate
            thresholds — this IS the DPO positive pool that gates Task 11.

Run over Deepfakes/FaceSwap (full-face swaps — expect low single-region recovery,
per CURRENT_STATE "14/376") AND NeuralTextures (the localized method on disk).
The contrast is the finding.

Run:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    /data/umar/miniconda3/envs/GenD/bin/python cec/pilots/pilot_1l/run_pilot_1l.py --n 40
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from cec.data.pairing import build_pair, fake_dir  # noqa: E402
from cec.instruments.spatial import SpatialInstrument  # noqa: E402
from cec.masks.regions import region_mask  # noqa: E402
from cec.registration import load_params  # noqa: E402
from cec.registration.vocab import SPATIAL_REGIONS  # noqa: E402
from cec.repair import poisson_paste  # noqa: E402

SEED = 0
OUT = REPO / "results" / "pilot_1l"
METHODS = ["Deepfakes", "FaceSwap", "NeuralTextures"]


def gather(method, n, rng):
    fdir = fake_dir(method) / "frames"
    vids = sorted(p.name for p in fdir.iterdir())
    rng.shuffle(vids)
    pairs = []
    for vid in vids:
        if len(pairs) >= n:
            break
        frames = sorted(p.stem for p in (fdir / vid).glob("*.png"))
        if not frames:
            continue
        pair = build_pair(method, vid, frames[len(frames) // 2])
        if pair is not None:
            pairs.append(pair)
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="fakes per method")
    ap.add_argument("--instrument", default="fsfm")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    params = load_params()
    margin = params.certify_margin(args.instrument)
    inst = SpatialInstrument(args.instrument, device=args.device)

    results = {"instrument": args.instrument, "margin": margin, "n_per_method": args.n, "methods": {}}
    print(f"instrument={args.instrument} margin={margin}\n")

    for method in METHODS:
        pairs = gather(method, args.n, rng)
        ceil_nms, best_region_nms, recoveries = [], [], []
        pool = 0  # (image,region) pairs that certify
        region_certs = {r: 0 for r in sorted(SPATIAL_REGIONS)}
        for pair in pairs:
            # ceiling: full GT mask repair
            gt_rep = poisson_paste(pair.fake, pair.real, pair.mask)
            # all landmark regions in one batch
            region_imgs, region_names = [], []
            for r in sorted(SPATIAL_REGIONS):
                m = region_mask(r, pair.landmarks)
                if m is not None and m.sum() > 0:
                    region_imgs.append(poisson_paste(pair.fake, pair.real, m))
                    region_names.append(r)
            batch = [pair.fake, gt_rep] + region_imgs
            p = inst.p_fake_batch(batch)
            p_fake, p_gt = p[0], p[1]
            ceil_nm = float(p_fake - p_gt)
            region_nm = {n: float(p_fake - q) for n, q in zip(region_names, p[2:])}
            best = max(region_nm.values()) if region_nm else 0.0
            ceil_nms.append(ceil_nm)
            best_region_nms.append(best)
            if ceil_nm > 0.05:
                recoveries.append(best / ceil_nm)
            for n, nm in region_nm.items():
                if nm >= margin:
                    pool += 1
                    region_certs[n] += 1

        res = {
            "n": len(pairs),
            "ceiling_nm_median": round(float(np.median(ceil_nms)), 4) if ceil_nms else None,
            "best_region_nm_median": round(float(np.median(best_region_nms)), 4) if best_region_nms else None,
            "recovery_median": round(float(np.median(recoveries)), 3) if recoveries else None,
            "recovery_ge_70pct": round(float(np.mean([r >= 0.7 for r in recoveries])), 3) if recoveries else None,
            "certified_region_claims": pool,
            "certs_by_region": {k: v for k, v in region_certs.items() if v > 0},
        }
        results["methods"][method] = res
        print(f"[{method}] n={res['n']}  ceiling NM={res['ceiling_nm_median']}  "
              f"best-region NM={res['best_region_nm_median']}  recovery={res['recovery_median']}  "
              f"(≥70%: {res['recovery_ge_70pct']})")
        print(f"    certified region-claims (DPO pool): {pool}  by region: {res['certs_by_region']}")

    (OUT / "pilot_1l.json").write_text(json.dumps(results, indent=2))
    print(f"\n[written] {OUT / 'pilot_1l.json'}")
    print("DPO pool = total certified region-claims. If scarce -> 🟡 (Task 11 gate).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

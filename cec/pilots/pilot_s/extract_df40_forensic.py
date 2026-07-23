"""Extract the DISCERN forensic/spectral bank on res-matched DF40 families.

Runs the exact DISCERN ForensicPrecomputer (SegFormer parse + DCT/forensic
features) used to build the FF++ bank, on DF40 crops, so the spectral probe can
be tested where it matters: unseen generative families.

Only RESOLUTION-MATCHED families (fake and real at the same native size) are
extracted — otherwise a uniform resize injects a per-class interpolation
fingerprint and the probe separates on that, not the generator. stargan and
starganv2 are 256↔256 (GAN); the diffusion families ship mismatched reals
(CollabDiff 512↔178, MidJourney 1024↔256) and are deferred.

MUST run in the dfb_nesy env (has mediapipe/insightface; isolated from GenD, so
frozen anchors are untouched):
    /data/umar/miniconda3/envs/dfb_nesy/bin/python \
        cec/pilots/pilot_s/extract_df40_forensic.py --n 150
"""
import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "preprocessing"))

from precompute_forensic_features import ForensicPrecomputer  # noqa: E402

DF40 = Path("/data/umar/Datasets/df40/test")
OUT = REPO / "results" / "pilot_s" / "df40_forensic"
RES_MATCHED = ["stargan", "starganv2"]  # fake+real both 256 (verified)
SEED = 0


def gather_paths(family, n):
    """(fake_paths, real_paths) for a family, capped at n each, sorted+deterministic."""
    import random
    rng = random.Random(SEED)

    def glob_imgs(root):
        out = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            out.extend(root.rglob(ext))
        return sorted(str(p) for p in out if "__MACOSX" not in p.parts)

    fakes = glob_imgs(DF40 / family / "fake")
    reals = glob_imgs(DF40 / family / "real")
    rng.shuffle(fakes)
    rng.shuffle(reals)
    return fakes[:n], reals[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150, help="crops per class per family")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print("[init] loading DISCERN ForensicPrecomputer (SegFormer + insightface) ...")
    pre = ForensicPrecomputer(device=args.device)

    for family in RES_MATCHED:
        fakes, reals = gather_paths(family, args.n)
        print(f"[{family}] fake={len(fakes)} real={len(reals)}")
        for label_name, paths in (("fake", fakes), ("real", reals)):
            out = pre.process_video(paths, batch_size=32)
            dest = OUT / f"{family}_{label_name}.pt"
            torch.save(
                {"features": out["features"], "frame_paths": out["frame_paths"],
                 "feature_names": out["feature_names"], "family": family,
                 "label": 1 if label_name == "fake" else 0},
                dest,
            )
            print(f"  wrote {dest.name}: features {tuple(out['features'].shape)}")

    print(f"\n[done] DF40 forensic banks in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

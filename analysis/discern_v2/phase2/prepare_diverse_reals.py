#!/usr/bin/env python3
"""Stage 1.1 — re-crop a diverse-real corpus through DISCERN's OWN crop pipeline.

    🔴 UMAR-RUNS (GPU for RetinaFace, ~10 min for FFHQ-8750):

    python analysis/discern_v2/phase2/prepare_diverse_reals.py \
        --images /data/umar/Datasets/ffhq256_subset \
        --out /data/umar/Datasets/preprocessed/FFHQ-recrop/frames

Why this step exists at all
---------------------------
Stage 1 refits `P_R` so its residual measures "unlike authentic" instead of "unlike FF++
authentic". That only works if the crop convention is held fixed. FFHQ ships in **its own**
alignment (eye-line centred, generous margin, 1024 originally, 256 here); every DISCERN crop is
RetinaFace 5-point landmarks warped onto the GenD template at `scale=1.3`, 256x256
(`preprocessing/preprocess.py:129`). Fitting `P_R` on FFHQ-as-shipped and evaluating it on
DISCERN crops would make `r_ref` partly a *crop-convention* residual — which is precisely the
"detects the corpus, not the forgery" failure Stage 1 exists to remove, reintroduced by the fix.

So the corpus is re-cropped with the identical detector, identical landmark template and identical
scale. `align_face` is imported from the preprocessing module rather than reimplemented, so the
two cannot drift.

What is reported, not assumed
-----------------------------
Detection is not guaranteed. Images where RetinaFace finds no face, or where the affine solve
fails, are DROPPED and counted, and the rate is written to the manifest. A silently shortened
corpus would change what `P_R` was fit on without changing anything a reader could see.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "preprocessing"))

from preprocessing.preprocess import align_face  # noqa: E402
from preprocessing.retinaface import prepare_model  # noqa: E402

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def largest_face(model, image: np.ndarray):
    """Landmarks of the largest detected face, or None.

    Largest rather than first: a diverse-real corpus contains group photographs, and taking an
    arbitrary detection would put a bystander's face in the reference population.
    """
    boxes, landmarks = model.detect(image, input_size=(640, 640))
    if boxes is None or len(boxes) == 0:
        return None
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return landmarks[int(np.argmax(areas))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--images", type=Path, required=True, help="directory of real face images")
    ap.add_argument("--out", type=Path, required=True, help="directory for 256x256 crops")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--scale", type=float, default=1.3,
                    help="context margin; MUST match preprocessing/config.yaml's crop scale")
    ap.add_argument("--det-thres", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0, help="0 = all; >0 for a smoke run")
    args = ap.parse_args()

    paths = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"no images under {args.images}")
    print(f"{len(paths)} images from {args.images}")

    args.out.mkdir(parents=True, exist_ok=True)
    model = prepare_model(det_thres=args.det_thres)

    kept, no_face, no_affine, unreadable = 0, 0, 0, 0
    for i, path in enumerate(paths):
        image = cv2.imread(str(path))
        if image is None:
            unreadable += 1
            continue
        landmarks = largest_face(model, image)
        if landmarks is None:
            no_face += 1
            continue
        aligned, _ = align_face(image, np.asarray(landmarks, dtype=np.float32),
                                target_size=(args.size, args.size), scale=args.scale)
        if aligned is None:
            no_affine += 1
            continue
        cv2.imwrite(str(args.out / f"{path.stem}.png"), aligned)
        kept += 1
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(paths)}  kept {kept}")

    rate = kept / len(paths)
    manifest = {
        "source_images": str(args.images), "output": str(args.out),
        "n_input": len(paths), "n_kept": kept, "detection_rate": rate,
        "dropped": {"no_face": no_face, "affine_failed": no_affine, "unreadable": unreadable},
        "crop": {"detector": "RetinaFace buffalo_l det_10g", "template": "GenD 5-point",
                 "size": args.size, "scale": args.scale,
                 "source": "preprocessing/preprocess.py::align_face"},
        "why": "identical crop convention to every other DISCERN split, so that a refit P_R's "
               "residual cannot be a crop-convention residual",
    }
    (args.out.parent / "recrop_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nkept {kept}/{len(paths)} ({rate:.3f}) -> {args.out}")
    print(f"  dropped: no face {no_face}, affine failed {no_affine}, unreadable {unreadable}")
    if rate < 0.9:
        print("\n  ⚠️  detection rate below 0.90. A corpus that only yields faces on the easy "
              "images is a BIASED reference population — the reals P_R never sees are exactly "
              "the hard poses and lightings it most needs. Report this rate in "
              "STAGE1_REFERENCE.md rather than proceeding quietly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

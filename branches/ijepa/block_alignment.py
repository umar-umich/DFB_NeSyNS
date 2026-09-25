#!/usr/bin/env python3
"""Does each I-JEPA block land on the face? Measured with 81-point landmarks, not eyeballed.

    /data/umar/miniconda3/envs/dfb_nesy/bin/python branches/ijepa/block_alignment.py

For each of the 40 overlay crops (20 FF++ c23 real, 20 CDFv2 real) the face convex hull is built from the 81
landmarks and each block's coverage is the fraction of its own area inside that hull. A block that is mostly
background is not measuring a face region, whatever it is called.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import dlib
from PIL import Image, ImageDraw

PRED = "/data/umar/Repos/model_weights_backup/model_weights/video/shape_predictor_81_face_landmarks.dat"
OVL = Path("analysis/ijepa/block_overlays")
BLOCKS = {"upper face": (2, 9, 4, 12), "lower face": (9, 16, 4, 12), "left cheek": (5, 12, 0, 8),
          "right cheek": (5, 12, 8, 16), "centre": (4, 11, 4, 12)}
P = 14


def hull_mask(img, det, sp):
    """These are already-aligned face crops, so the detector often refuses them. The shape predictor is therefore
    given the whole crop as the face box, which is the standard handling for pre-cropped faces and lets every crop
    contribute instead of the minority the detector happens to accept."""
    a = np.asarray(img)
    faces = det(a, 1)
    box = faces[0] if faces else dlib.rectangle(0, 0, img.size[0] - 1, img.size[1] - 1)
    s = sp(a, box)
    pts = np.array([[s.part(i).x, s.part(i).y] for i in range(s.num_parts)], np.int32)
    m = Image.new("1", img.size, 0)
    from scipy.spatial import ConvexHull
    h = ConvexHull(pts)
    ImageDraw.Draw(m).polygon([tuple(pts[v]) for v in h.vertices], fill=1)
    return np.asarray(m, bool)


def main():
    det, sp = dlib.get_frontal_face_detector(), dlib.shape_predictor(PRED)
    res = {}
    for tag in ("FFpp", "CDFv2"):
        cov = {b: [] for b in BLOCKS}
        found = 0
        for p in sorted(OVL.glob(f"{tag}_*.png")):
            img = Image.open(p).convert("RGB")
            m = hull_mask(img, det, sp)
            if m is None:
                continue
            found += 1
            for b, (r0, r1, c0, c1) in BLOCKS.items():
                sub = m[r0 * P:r1 * P, c0 * P:c1 * P]
                cov[b].append(float(sub.mean()))
        res[tag] = {"crops_with_a_detected_face": found, "n_crops": len(list(OVL.glob(f"{tag}_*.png"))),
                    "coverage": {b: {"mean": float(np.mean(v)), "min": float(np.min(v)),
                                     "max": float(np.max(v))} for b, v in cov.items() if v}}
        print(f"\n{tag}: landmarks found on {found} of {res[tag]['n_crops']} crops")
        for b, s in res[tag]["coverage"].items():
            print(f"  {b:12s} inside the face hull: mean {s['mean']:.2f}  (min {s['min']:.2f}, max {s['max']:.2f})")
    json.dump(res, open("analysis/ijepa/block_alignment.json", "w"), indent=1)
    print("\nwrote analysis/ijepa/block_alignment.json")


if __name__ == "__main__":
    main()

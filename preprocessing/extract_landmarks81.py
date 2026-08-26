#!/usr/bin/env python
"""Extract dlib 81-point landmarks for SBI, on the cropped frames we already have.

    python preprocessing/extract_landmarks81.py \
        --root /data/umar/Datasets/preprocessed/FaceForensics++/original_sequences/youtube/c23 \
        --workers 24

WHY THIS EXISTS. SBI (Self-Blended Images) builds its blend mask from a convex hull over the
81-point dlib landmark set — `sbi_api.py` slices `landmark[:68]` and indexes points 68..80 in
`reorder_landmark`. Our preprocessing is the v2 RetinaFace path, which stores FIVE points per
frame in `landmarks/`. Those five cannot produce a face hull, so SBI is unrunnable against the
tree as it stands; this closes that gap.

WHY IT RUNS ON THE CROPS RATHER THAN RE-CROPPING. The frames on disk were aligned by RetinaFace.
Re-running DeepfakeBench's dlib crop path would produce DIFFERENT pixels from the ones every
model in this project was trained and scored on, so a self-blend built there would be blending a
face the detector never sees. Running the predictor over the existing 256x256 crops keeps the
landmarks in the coordinate frame of the actual training pixels. Detection rate on those crops
was measured at 60/60 with no upsampling before this was written, which is why `--upsample`
defaults to 0.

WHY IT WRITES landmarks81/ AND NOT landmarks/. The 5-point files are live input elsewhere (the
CEC region masks read them). This is additive; nothing existing is touched.

SBI only ever blends REAL frames — `SBIDataset` filters to `label == 0` and synthesises its fakes
— so pointing `--root` at the authentic source is sufficient and the manipulated trees are not
needed.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pathlib
import sys

import cv2
import numpy as np

PREDICTOR = "preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat"
_det = _pred = None


def _init(predictor_path: str) -> None:
    """One detector per worker. dlib objects do not survive pickling, so they are built here."""
    global _det, _pred
    import dlib
    _det = dlib.get_frontal_face_detector()
    _pred = dlib.shape_predictor(predictor_path)


def _largest(faces):
    """The crop holds one subject, but the detector can still return a spurious extra box.

    Taking the largest is what DeepfakeBench's own preprocessing does, and on a tight crop the
    true face is the dominant box by a wide margin.
    """
    return max(faces, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()))


def process_video(job: tuple[str, str, int]) -> dict:
    src_dir, dst_dir, upsample = job
    src, dst = pathlib.Path(src_dir), pathlib.Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    frames = sorted(src.glob("*.png"))
    written = skipped = missed = failed = 0
    for f in frames:
        out = dst / (f.stem + ".npy")
        if out.exists():
            skipped += 1
            continue
        img = cv2.imread(str(f))
        if img is None:
            failed += 1
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = _det(rgb, upsample)
        if not faces:
            # Recorded, not silently dropped: a frame with no landmarks is a frame SBI cannot
            # blend, and the coverage number is what tells us whether the corpus is usable.
            missed += 1
            continue
        shape = _pred(rgb, _largest(faces))
        pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(shape.num_parts)],
                       dtype=np.int32)
        if pts.shape != (81, 2):
            failed += 1
            continue
        np.save(out, pts)
        written += 1
    return {"video": src.name, "frames": len(frames), "written": written,
            "skipped": skipped, "missed": missed, "failed": failed}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=pathlib.Path, required=True,
                    help="a compression dir holding frames/ (e.g. .../youtube/c23)")
    ap.add_argument("--frames-dir", default="frames")
    ap.add_argument("--out-dir", default="landmarks81",
                    help="written beside frames/; deliberately NOT `landmarks`, which holds the "
                         "5-point RetinaFace files other code still reads")
    ap.add_argument("--predictor", default=PREDICTOR)
    ap.add_argument("--upsample", type=int, default=0,
                    help="dlib pyramid levels; 0 measured sufficient on 256x256 crops")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="first N videos only, for a smoke run")
    args = ap.parse_args()

    src_root = args.root / args.frames_dir
    dst_root = args.root / args.out_dir
    if not src_root.is_dir():
        raise SystemExit(f"no frames under {src_root}")
    if not pathlib.Path(args.predictor).exists():
        raise SystemExit(f"missing 81-point predictor at {args.predictor}")

    videos = sorted(p for p in src_root.iterdir() if p.is_dir())
    if args.limit:
        videos = videos[:args.limit]
    jobs = [(str(v), str(dst_root / v.name), args.upsample) for v in videos]
    print(f"{len(jobs)} videos under {src_root} -> {dst_root} ({args.workers} workers)")

    reports = []
    with mp.Pool(args.workers, initializer=_init, initargs=(args.predictor,)) as pool:
        for i, rep in enumerate(pool.imap_unordered(process_video, jobs, chunksize=4), 1):
            reports.append(rep)
            if i % 100 == 0 or i == len(jobs):
                done = sum(r["written"] + r["skipped"] for r in reports)
                miss = sum(r["missed"] for r in reports)
                print(f"  {i}/{len(jobs)} videos · {done} landmark files · {miss} undetected",
                      flush=True)

    tot_f = sum(r["frames"] for r in reports)
    tot_w = sum(r["written"] for r in reports)
    tot_s = sum(r["skipped"] for r in reports)
    tot_m = sum(r["missed"] for r in reports)
    tot_x = sum(r["failed"] for r in reports)
    have = tot_w + tot_s
    summary = {
        "root": str(args.root), "videos": len(reports), "frames": tot_f,
        "written": tot_w, "already_present": tot_s, "undetected": tot_m, "failed": tot_x,
        "coverage": (have / tot_f) if tot_f else 0.0,
        "videos_fully_covered": sum(1 for r in reports
                                    if r["written"] + r["skipped"] == r["frames"]),
        "upsample": args.upsample, "predictor": args.predictor,
    }
    (dst_root.parent / f"{args.out_dir}_report.json").write_text(json.dumps(
        {"summary": summary, "per_video": reports}, indent=2))
    print(f"\ncoverage {have}/{tot_f} = {summary['coverage']:.3%} "
          f"({tot_m} undetected, {tot_x} unreadable)")
    print(f"fully covered videos: {summary['videos_fully_covered']}/{len(reports)}")
    print(f"report -> {dst_root.parent / (args.out_dir + '_report.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

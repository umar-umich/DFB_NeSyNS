#!/usr/bin/env python3
"""Stage 2 inputs: dense face-crop windows from raw video, cut with the pipeline that made the stored crops.

    OMP_NUM_THREADS=1 <dfb_nesy python> analysis/tbiom/temporal/extract_dense.py --set ffpp_train --workers 12
    OMP_NUM_THREADS=1 <dfb_nesy python> analysis/tbiom/temporal/extract_dense.py --set dev2 --workers 12

Frozen in tbiom/docs/stage2-temporal-preregistration.md. For every video: the 32 consecutive source frames centred on
its midpoint, each cropped by `preprocessing/preprocess_current.py::extract_aligned_face_dlib` (dlib HOG detector,
81 landmarks, aligned 256-px crop at scale 1.3) -- the function that produced every stored crop, imported, not
copied -- so sparse and dense inputs share their crop geometry. The window yields the dense configuration (stride 2)
and the dense-subsampled one (stride 8) from the same frames.

A frame whose face is not found is RECORDED (per video, per frame index) and filled with the nearest successful crop
in the window so a clip keeps 32 slots; the manifest carries the count, and videos above 25% failures are flagged for
exclusion. Output: one h5 per set (key = <set>/<source>/<video>/<frame>.png, RGB uint8 256x256), a split list, and a
JSON manifest with fps, frame indices, physical span and failures per video.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

REPO = Path("/data/umar/Repos/DFB_NeSyNS")
EXT = Path("/data/umar/Repos/DISCERN_Ext")
DS = Path("/data/umar/Datasets")
OUT = DS / "tbiom_dense"
WIN = 32
FF = DS / "FaceForensics++"
FF_METHODS = {"Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"}


def ff_raw(vdir: str) -> Path | None:
    """FFpp_train/Deepfakes/Deepfakes__001_870 -> manipulated_sequences/Deepfakes/c23/videos/001_870.mp4."""
    parts = vdir.split("/")
    sub, name = parts[-2], parts[-1]
    if sub in FF_METHODS:
        return FF / "manipulated_sequences" / sub / "c23/videos" / f"{name.split('__')[-1]}.mp4"
    if "real" in sub.lower():
        return FF / "original_sequences/youtube/c23/videos" / f"{name}.mp4"
    return None


def cdf3_index():
    idx = {}
    root = DS / "Celeb-DF-v3"
    for p in root.rglob("*.mp4"):
        rel = p.relative_to(root).parts
        method = rel[-2] if rel[0] == "Celeb-synthesis" else rel[0]
        idx[(method, p.stem)] = p
    return idx


def dev_raw(vdir: str, cdf3) -> Path | None:
    parts = vdir.split("/")
    if parts[0] == "FFpp_val":
        return ff_raw(vdir)
    if "CDFv2val" in parts[1]:
        sub = parts[1].split("-", 1)[1]                                   # Celeb-real / Celeb-synthesis / YouTube-real
        return DS / "Celeb-DF-v2" / sub / f"{parts[2].removeprefix('cdf2-')}.mp4"
    if "DFEval24val" in parts[1]:
        return DS / "video_eval24" / f"{parts[2].removeprefix('dfe24-')}.mp4"
    if parts[1] == "CDFv3valG":
        method = {"R3DPortrait": "Real3DPortrait"}.get(parts[2], parts[2])   # the raw release spells it out
        return cdf3.get((method, parts[3]))
    return None


def video_list(which: str):
    import pandas as pd
    if which == "ffpp_train":
        vids = set()
        for f in (EXT / "eval_adaptation/data/splits/FFpp_train").glob("*.txt"):
            vids |= {l.strip().rsplit("/", 1)[0] for l in open(f) if l.strip()}
        return [(v, ff_raw(v)) for v in sorted(vids)]
    cdf3 = cdf3_index()
    rows = []
    for name in ("VALmix", "VALmixV2", "FFpp_val"):
        d = pd.read_csv(REPO / f"logs/tbiom/stage23/warmgate_e19_{name}.csv", usecols=["key"])
        rows += sorted(set(d.key.str.split("::").str[-1].str.rsplit("/", n=1).str[0]))
    keep, cdf3_rows = [], []
    for v in rows:
        p = v.split("/")
        if p[0] == "FFpp_val" or (len(p) > 1 and ("CDFv2val" in p[1] or "DFEval24val" in p[1])):
            keep.append(v)
        elif len(p) > 2 and p[1] == "CDFv3valG" and p[2] != "IP_LAP":
            cdf3_rows.append(v)
    rng = np.random.default_rng(2026)
    by_m = {}
    for v in cdf3_rows:
        by_m.setdefault(v.split("/")[2], []).append(v)
    quota = 600 / len(cdf3_rows)
    for m, vs in sorted(by_m.items()):                                   # stratified by method, seed 2026
        k = max(1, int(round(len(vs) * quota)))
        keep += sorted(rng.choice(vs, size=min(k, len(vs)), replace=False).tolist())
    return [(v, dev_raw(v, cdf3)) for v in keep]


_det = _pred = None


def work(item):
    global _det, _pred
    import cv2
    cv2.setNumThreads(1)
    vdir, raw = item
    if raw is None or not Path(raw).exists():
        return vdir, None, {"error": "no raw video", "raw": str(raw)}
    if _det is None:
        import dlib
        sys.path.insert(0, str(REPO / "preprocessing"))
        _det = dlib.get_frontal_face_detector()
        _pred = dlib.shape_predictor(str(REPO / "preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat"))
    from preprocess_current import extract_aligned_face_dlib
    cap = cv2.VideoCapture(str(raw)); fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    start = max(0, fc // 2 - WIN // 2)
    crops, fails, idxs, i = [], [], [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i >= start:
            face, lm, _ = extract_aligned_face_dlib(_det, _pred, fr)
            crops.append(None if face is None else cv2.cvtColor(face, cv2.COLOR_BGR2RGB)); idxs.append(i)
            if face is None:
                fails.append(i)
            if len(crops) == WIN:
                break
        i += 1
    cap.release()
    good = [j for j, c in enumerate(crops) if c is not None]
    if not good:
        return vdir, None, {"error": "no face in window", "fc": fc, "fps": fps}
    filled = [crops[j] if crops[j] is not None else crops[min(good, key=lambda g: abs(g - j))] for j in range(len(crops))]
    meta = {"raw": str(raw), "fc": fc, "fps": fps, "frames": idxs, "failed": fails, "n": len(filled),
            "span_s": (len(filled) / fps) if fps else None, "fail_rate": len(fails) / max(1, len(crops))}
    return vdir, np.stack(filled), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["ffpp_train", "dev2"]); ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import h5py
    OUT.mkdir(parents=True, exist_ok=True)
    items = video_list(a.set)
    if a.limit:
        items = items[: a.limit]
    unresolved = [v for v, p in items if p is None or not Path(p).exists()]
    print(f"{a.set}: {len(items)} videos, raw resolved for {len(items) - len(unresolved)}", flush=True)
    h5p = OUT / f"{a.set}_dense{'_smoke' if a.limit else ''}.h5"
    man, keys, t0 = {}, [], time.time()
    with h5py.File(h5p.with_suffix(".h5.partial"), "w") as h, Pool(a.workers) as pool:
        for n, (vdir, arr, meta) in enumerate(pool.imap_unordered(work, items, chunksize=2)):
            man[vdir] = meta
            if arr is not None:
                for j, (fi, img) in enumerate(zip(meta["frames"], arr)):
                    k = f"{vdir}/{fi:06d}.png"
                    h.create_dataset(k, data=img, compression="gzip", compression_opts=4); keys.append(k)
            if n % 200 == 0:
                print(f"  {n}/{len(items)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    h5p.with_suffix(".h5.partial").rename(h5p)
    (OUT / f"{a.set}_dense{'_smoke' if a.limit else ''}_keys.txt").write_text("\n".join(sorted(keys)) + "\n")
    fr = [m.get("fail_rate", 1.0) for m in man.values()]
    summary = {"videos": len(items), "extracted": sum(1 for m in man.values() if "error" not in m),
               "errors": {e: sum(1 for m in man.values() if m.get("error") == e) for e in ("no raw video", "no face in window")},
               "frames": len(keys), "mean_fail_rate": float(np.mean(fr)), "videos_over_25pct_fail": int(sum(f > 0.25 for f in fr)),
               "median_span_s": float(np.median([m["span_s"] for m in man.values() if m.get("span_s")])),
               "minutes": (time.time() - t0) / 60}
    json.dump({"summary": summary, "videos": man}, open(OUT / f"{a.set}_dense{'_smoke' if a.limit else ''}_manifest.json", "w"))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()

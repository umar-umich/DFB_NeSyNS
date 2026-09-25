#!/usr/bin/env python3
"""Amendment 2 precondition: is the cache per-frame, and are the cached frames consecutive?

    python branches/ijepa/cache_check.py

Two questions, and the second decides how every T2/T3 result may be WORDED.

1. The cache must hold per-frame eps_k and r_bar_k with a video id and a frame index for every
   split. a_k is not cached because it is derived: it is log eps_k standardised by the frozen
   mu_k(q), sigma_k fitted on real FF++ training frames, so caching it would freeze a fit that the
   gates are allowed to refit.

2. Frames per video and the spacing between cached frames. The amendment is explicit: if the cached
   frames are not consecutive at the native frame rate, every result here is aggregation over
   sampled frames and must never be called motion or temporal dynamics.

ON SECONDS. The stored crop filenames carry the ORIGINAL video frame index, so the stride in source
frames is exact and assumption-free. Converting it to seconds needs the native frame rate, which is
NOT recorded anywhere in the preprocessing metadata -- `/data/umar/Datasets/preprocessed/*/*.json`
holds resolutions only. Seconds are therefore reported as a range under an explicitly stated fps
assumption and labelled as such, rather than quoted as a measured quantity.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as D  # noqa: E402

SPLITS = Path("/data/umar/Repos/DISCERN_Ext/eval_adaptation/data/splits")
OUT = Path("/data/umar/Repos/DFB_NeSyNS/analysis/ijepa")
FPS_ASSUMED = (25.0, 30.0)          # stated, not measured; no fps is recorded in the metadata
NEED = ("eps", "rbar", "pooled", "sbar", "q", "keys")


def frame_index(key: str):
    """The original video frame index, from the stored crop's filename."""
    stem = str(key).split("/")[-1].rsplit(".", 1)[0]
    m = re.findall(r"\d+", stem)
    return int(m[-1]) if m else None


def per_dataset_spacing(ds):
    d = SPLITS / ds
    if not d.is_dir():
        return None
    byvid = {}
    for m in sorted(d.glob("*.txt")):
        for line in m.read_text().splitlines():
            line = line.strip()
            if line:
                byvid.setdefault(D.video_of(line), []).append(line)
    fpv, strides = [], []
    for v, ks in byvid.items():
        idx = sorted(i for i in (frame_index(k) for k in ks) if i is not None)
        fpv.append(len(ks))
        if len(idx) > 1:
            strides += list(np.diff(idx))
    if not fpv:
        return None
    s = np.array(strides) if strides else np.array([0])
    return {"videos": len(byvid), "frames": int(sum(fpv)),
            "frames_per_video_median": float(np.median(fpv)),
            "frames_per_video_min": int(np.min(fpv)), "frames_per_video_max": int(np.max(fpv)),
            "stride_source_frames_median": float(np.median(s)),
            "stride_source_frames_p05": float(np.percentile(s, 5)),
            "stride_source_frames_p95": float(np.percentile(s, 95)),
            "consecutive": bool(np.median(s) <= 1.0)}


def main():
    datasets = ["FFpp_train", "FFpp_val", "VALmix", "VALmixV2", "CDFv2", "CDFv3", "DFD", "DFDC",
                "DFDCP", "DFDM", "DFEval24", "FaceShifter", "FFIW", "IDForge", "KoDF", "DF40"]
    rows = {}
    for ds in datasets:
        r = per_dataset_spacing(ds)
        if r:
            lo, hi = (r["stride_source_frames_median"] / f for f in reversed(FPS_ASSUMED))
            r["spacing_seconds_assumed_range"] = [round(lo, 3), round(hi, 3)]
            rows[ds] = r

    # the per-frame structure of whatever is cached so far
    have = D.available()
    cached = {}
    for stem in sorted(have):
        try:
            z = D.load(stem, arrays=("eps", "rbar"))
        except Exception as e:
            cached[stem] = {"error": f"{type(e).__name__}: {e}"}
            continue
        n = len(z["keys"])
        idx = [frame_index(k) for k in z["keys"][:2000]]
        cached[stem] = {"frames": n, "videos": int(len(set(z["video"]))),
                        "eps_shape": list(z["eps"].shape), "rbar_shape": list(z["rbar"].shape),
                        "per_frame": bool(z["eps"].shape[0] == n and z["rbar"].shape[0] == n),
                        "frame_index_recoverable": bool(all(i is not None for i in idx))}

    any_consec = [d for d, r in rows.items() if r["consecutive"]]
    verdict = {
        "per_frame_cache": all(c.get("per_frame") for c in cached.values()) if cached else None,
        "frame_index_recoverable": all(c.get("frame_index_recoverable") for c in cached.values())
                                   if cached else None,
        "a_k": "derived, not cached: log eps_k standardised by the frozen training-fitted mu_k, sigma_k",
        "consecutive_at_native_rate": bool(any_consec),
        "wording_constraint": ("Cached frames are NOT consecutive at the native frame rate. Every "
                               "T0-T3 result is aggregation over SAMPLED frames and must never be "
                               "described as motion or temporal dynamics.")
                              if not any_consec else "frames are consecutive; no constraint",
        "fps_assumed_for_seconds": list(FPS_ASSUMED),
        "fps_source": "ASSUMED. No native frame rate is recorded in the preprocessing metadata.",
    }
    res = {"check": "Amendment 2 precondition", "verdict": verdict,
           "per_dataset": rows, "cached_so_far": cached}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "amendment2_cache_check.json", "w"), indent=1)

    print(f"{'dataset':13s} {'videos':>7} {'f/vid':>6} {'stride (src frames)':>21} {'~seconds':>14}")
    for ds, r in rows.items():
        sp = r["spacing_seconds_assumed_range"]
        print(f"{ds:13s} {r['videos']:7d} {r['frames_per_video_median']:6.0f} "
              f"{r['stride_source_frames_median']:8.1f} "
              f"[{r['stride_source_frames_p05']:.0f}-{r['stride_source_frames_p95']:.0f}]"
              f"{'':>4} {sp[0]:.2f}-{sp[1]:.2f}s")
    print(f"\ncached so far: {len(cached)} stems")
    for s, c in cached.items():
        if "error" in c:
            print(f"  {s:12s} {c['error']}")
        else:
            print(f"  {s:12s} {c['frames']:7d} frames, {c['videos']:5d} videos, "
                  f"per-frame={c['per_frame']}, frame index recoverable={c['frame_index_recoverable']}")
    print(f"\nconsecutive at native rate: {verdict['consecutive_at_native_rate']}")
    print(f"WORDING: {verdict['wording_constraint']}")
    print(f"-> {OUT / 'amendment2_cache_check.json'}")


if __name__ == "__main__":
    main()

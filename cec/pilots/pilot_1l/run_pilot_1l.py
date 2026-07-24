"""TASK 10 / T14 — PILOT 1L: the certifiable-region CEILING under the REAL gate.

Fixed (T14): this calls the same `CertificationGate` the audit uses — no
margin-only shortcut (that was Error 1: the old "47" overcounted vs the audit's
0). Any count here is gate-certified by construction, directly comparable to the
audit.

Two regimes:
  region     NeuralTextures, Face2Face — exhaustive over landmark regions, each
             judged at REGION scope (`gate_region:`). "How many regions certify?"
  composite  Deepfakes, FaceSwap, DeepFakeDetection — the inner-face composite
             claim, judged at COMPOSITE scope (frozen `gate:`). "Does the
             whole-face claim certify?" (single regions are expected to fail.)

Video-level sampling (T14): `qc.frames_per_video` frames per video, aggregated
per video; certification reported per-frame AND per-video. Train/test split
honored via `--split` (default test — disjoint from the region calibration).

Reports per method: certification rate (frame + video), scope, fail-reason
breakdown. This is the certifiable CEILING — the DPO-viability upper bound (the
proposer->oracle hit-rate, T14d, lives in the audit where the proposer runs).

Run (🔴 UMAR-RUNS):
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    /data/umar/miniconda3/envs/GenD/bin/python cec/pilots/pilot_1l/run_pilot_1l.py \
        --split test --instrument fsfm
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from cec.data.pairing import build_pair, fake_dir, split_video_ids  # noqa: E402
from cec.gate.certify import CertificationGate  # noqa: E402
from cec.proposer.validate import Claim  # noqa: E402
from cec.registration import load_params  # noqa: E402
from cec.registration.vocab import SPATIAL_REGIONS, WHOLE_FACE  # noqa: E402
from cec.repair import Lpips  # noqa: E402

SEED = 0
OUT = REPO / "results" / "pilot_1l"
REGION_METHODS = ["NeuralTextures", "Face2Face"]
COMPOSITE_METHODS = ["Deepfakes", "FaceSwap"]  # DeepFakeDetection dropped: ids outside FF++ split json


def gather(method, n_videos, frames_per_video, split_vids, rng):
    """Up to n_videos videos, frames_per_video frames each -> list of PairSamples tagged by vid."""
    fdir = fake_dir(method) / "frames"
    vids = [v.name for v in fdir.iterdir() if v.name in split_vids]
    rng.shuffle(vids)
    out = []
    for vid in vids:
        if len({p.vid for p in out}) >= n_videos:
            break
        frames = sorted(p.stem for p in (fdir / vid).glob("*.png"))
        if not frames:
            continue
        idx = np.linspace(0, len(frames) - 1, min(frames_per_video, len(frames))).astype(int)
        for j in sorted(set(idx)):
            pair = build_pair(method, vid, frames[j])
            if pair is not None:
                out.append(pair)
    return out


def region_claims():
    return [Claim(f"c_{r}", "boundary_artifact", r, "", "region") for r in sorted(SPATIAL_REGIONS)]


def composite_claim():
    return Claim("c_composite", "texture_anomaly", WHOLE_FACE, "", "composite")


def run_method(gate, method, regime, pairs):
    """Per-frame + per-video certification under the real gate."""
    per_video = defaultdict(lambda: {"frames": 0, "cert_frames": 0, "certifiable": False})
    fail = Counter()
    frame_cert = 0
    n_frames = 0
    claims = region_claims() if regime == "region" else [composite_claim()]
    for pair in pairs:
        n_frames += 1
        pv = per_video[pair.vid]
        pv["frames"] += 1
        any_cert = False
        for cl in claims:
            res = gate.certify_claim(pair, cl)
            if res["label"] == "CERTIFIED":
                any_cert = True
            elif res.get("fail_reasons"):
                fail.update(res["fail_reasons"])
        if any_cert:
            frame_cert += 1
            pv["cert_frames"] += 1
            pv["certifiable"] = True
    n_videos = len(per_video)
    video_cert = sum(1 for v in per_video.values() if v["certifiable"])
    return {
        "regime": regime, "n_frames": n_frames, "n_videos": n_videos,
        "frame_cert_rate": round(frame_cert / max(1, n_frames), 3),
        "video_cert_rate": round(video_cert / max(1, n_videos), 3),
        "certified_frames": frame_cert, "certified_videos": video_cert,
        "fail_reason_breakdown": dict(fail),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=int, default=20, help="videos per method")
    ap.add_argument("--split", default="test")
    ap.add_argument("--instrument", default="fsfm")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    params = load_params()
    fpv = params.qc["frames_per_video"]
    split_vids = split_video_ids(args.split)
    gate = CertificationGate(args.instrument, device=args.device, lpips=Lpips(args.device))

    results = {"instrument": args.instrument, "split": args.split, "frames_per_video": fpv,
               "region_gate_calibrated": params.raw["gate_region"].get("calibrated"),
               "methods": {}}
    print(f"instrument={args.instrument} split={args.split} frames/video={fpv} "
          f"(region_gate calibrated={results['region_gate_calibrated']})\n")

    for regime, methods in (("region", REGION_METHODS), ("composite", COMPOSITE_METHODS)):
        for method in methods:
            pairs = gather(method, args.videos, fpv, split_vids, rng)
            res = run_method(gate, method, regime, pairs)
            results["methods"][method] = res
            print(f"[{method:18} {regime:9}] frames={res['n_frames']} videos={res['n_videos']}  "
                  f"cert frame={res['frame_cert_rate']} video={res['video_cert_rate']}  "
                  f"fails={res['fail_reason_breakdown']}")

    (OUT / "pilot_1l.json").write_text(json.dumps(results, indent=2))
    print(f"\n[written] {OUT / 'pilot_1l.json'}")
    print("Region regime = certifiable-region ceiling under the fixed gate; composite regime "
          "= whole-face claim certification. Both via the SAME gate as the audit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""TASK 9 / T16 — audit run + FREEZE (scope routing, video sampling, reals, hit-rate).

Per proposer, over FF++ fakes AND reals (1:1):
  1. Control re-validation (offset gate) before trusting the model.
  2. Full run: proposer -> validate -> gate.certify_image -> per-image Certified
     Evidence Record (resumable JSONL). Scope routing is automatic (each claim
     carries region|composite|spectral|untestable). Reals get split_label='real'
     and their raw manipulation-claim count (for the FP-claim metric).
  3. Optional proposer->oracle hit-rate (--oracle): the gate finds certifiable
     regions independently; hit = the proposer cited one of them.
  4. FREEZE: hash records/, print the new hash. Keep the old hash in the LOG.

FF++ only (masks + paired reals). Single spatial instrument (fsfm). Deterministic.
Video-level sampling: qc.frames_per_video frames/video, per-video aggregation.

Run (🔴 UMAR-RUNS), once per proposer:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    /data/umar/miniconda3/envs/GenD/bin/python cec/scripts/run_audit.py \
        --proposer internvl3-8b --group all --split test --videos 40 --oracle
"""
import argparse
import hashlib
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from cec.data.pairing import (  # noqa: E402
    build_pair, build_real_pair, fake_dir, split_video_ids,
)
from cec.gate.certify import CertificationGate  # noqa: E402
from cec.proposer import Proposer, validate  # noqa: E402
from cec.proposer.validate import Claim  # noqa: E402
from cec.records import RECORDS_DIR, RecordStore, record_key  # noqa: E402
from cec.registration import load_params  # noqa: E402
from cec.registration.vocab import SPATIAL_REGIONS  # noqa: E402
from cec.repair import Lpips  # noqa: E402

SEED = 0
GROUPS = {
    "localized": ["NeuralTextures", "Face2Face"],
    "fullface": ["Deepfakes", "FaceSwap"],  # DeepFakeDetection dropped (ids outside FF++ split json)
}
GROUPS["all"] = GROUPS["localized"] + GROUPS["fullface"]


def _video_frames(fdir, vid, fpv):
    frames = sorted(p.stem for p in (fdir / vid).glob("*.png"))
    if not frames:
        return []
    idx = np.linspace(0, len(frames) - 1, min(fpv, len(frames))).astype(int)
    return [frames[j] for j in sorted(set(idx))]


def gather_fakes(methods, n_videos, fpv, split_vids, rng):
    pairs = []
    per = max(1, n_videos // len(methods))
    for method in methods:
        fdir = fake_dir(method) / "frames"
        vids = [v.name for v in fdir.iterdir() if v.name in split_vids]
        rng.shuffle(vids)
        used = 0
        for vid in vids:
            if used >= per:
                break
            got = False
            for frame in _video_frames(fdir, vid, fpv):
                p = build_pair(method, vid, frame)
                if p is not None:
                    pairs.append(p)
                    got = True
            used += 1 if got else 0
    return pairs


def gather_reals(n_target, fpv, split_vids, rng):
    """~n_target real images (1:1 with fakes) from youtube originals in-split."""
    rdir = Path("/data/umar/Datasets/preprocessed/FaceForensics++/original_sequences/youtube/c23/frames")
    split_ids = {p.split("_")[0] for p in split_vids}  # source identities in-split
    ids = [d.name for d in rdir.iterdir() if d.is_dir() and d.name in split_ids]
    rng.shuffle(ids)
    pairs = []
    for sid in ids:
        if len(pairs) >= n_target:
            break
        for frame in _video_frames(rdir, sid, fpv):
            p = build_real_pair(sid, frame)
            if p is not None:
                pairs.append(p)
    return pairs[:n_target]


def control_revalidation(gate, proposer, pairs, prov):
    offsets, cert = [], 0
    for pair in pairs:
        vr = validate(proposer.propose(pair.image_path))
        if not vr.schema_valid:
            continue
        for c in gate.certify_image(pair, vr, prov)["claims"]:
            if "controls" in c:
                offsets.append(c["controls"]["offset"])
                cert += c["label"] == "CERTIFIED"
    mean_off = float(np.mean(offsets)) if offsets else 0.0
    return mean_off < 0.15, {"mean_offset": round(mean_off, 4), "certified": cert}


def oracle_certifiable(gate, pair):
    """Regions the gate certifies independent of what was cited (T14d)."""
    hits = []
    for r in sorted(SPATIAL_REGIONS):
        res = gate.certify_claim(pair, Claim(f"o_{r}", "boundary_artifact", r, "", "region"))
        if res["label"] == "CERTIFIED":
            hits.append(r)
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposer", default="internvl3-8b")
    ap.add_argument("--instrument", default="fsfm")
    ap.add_argument("--group", default="all", choices=list(GROUPS))
    ap.add_argument("--split", default="test")
    ap.add_argument("--videos", type=int, default=40, help="videos per method")
    ap.add_argument("--reval", type=int, default=30)
    ap.add_argument("--oracle", action="store_true", help="compute proposer->oracle hit-rate (costly)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    rng = random.Random(SEED)
    params = load_params()
    fpv = params.qc["frames_per_video"]
    split_vids = split_video_ids(args.split)
    methods = GROUPS[args.group]

    print(f"[gather] group={args.group} split={args.split} fpv={fpv} ...")
    fakes = gather_fakes(methods, args.videos, fpv, split_vids, rng)
    reals = gather_reals(len(fakes), fpv, split_vids, rng)  # 1:1
    print(f"  {len(fakes)} fake frames / {len(reals)} real frames")

    proposer = Proposer(args.proposer, device=args.device)
    gate = CertificationGate(args.instrument, device=args.device, lpips=Lpips(args.device))
    prov = {"proposer": args.proposer, "prompt_hash": proposer.prompt_hash, "seed": params.seed}

    print(f"[reval] control re-validation on {args.reval} fakes ...")
    ok, stats = control_revalidation(gate, proposer, fakes[: args.reval], prov)
    print(f"  {stats} -> {'PASS' if ok else 'DROP model'}")
    if not ok:
        print(f"[audit] {args.proposer} DROPPED (offset gate). Reported, not run.")
        return 1

    store = RecordStore(f"audit_{args.proposer}_{args.instrument}_{args.group}")
    done = store.done_keys()
    print(f"[audit] {len(fakes)} fakes + {len(reals)} reals ({len(done)} done) ...")

    for pair in fakes:
        image = f"{pair.method}/{pair.vid}/{pair.frame}"
        if record_key(image, args.instrument, args.proposer) in done:
            continue
        vr = validate(proposer.propose(pair.image_path))
        if not vr.schema_valid:
            continue
        rec = gate.certify_image(pair, vr, prov)
        rec["split_label"] = "fake"
        rec["family"] = pair.method
        rec["n_manipulation_claims"] = len(vr.manipulation_claims)
        if args.oracle:
            oc = oracle_certifiable(gate, pair)
            cited = {c.location for c in vr.manipulation_claims}
            rec["oracle"] = {"certifiable": oc, "proposer_cited": sorted(x for x in cited if x),
                             "hit": bool(set(oc) & cited)}
        store.append(rec)

    for pair in reals:
        image = f"{pair.method}/{pair.vid}/{pair.frame}"
        if record_key(image, args.instrument, args.proposer) in done:
            continue
        vr = validate(proposer.propose(pair.image_path))
        if not vr.schema_valid:
            continue
        rec = gate.certify_image(pair, vr, prov)
        rec["split_label"] = "real"
        rec["family"] = "youtube-real"
        rec["n_manipulation_claims"] = len(vr.manipulation_claims)
        store.append(rec)

    h = hashlib.sha256()
    for f in sorted(RECORDS_DIR.glob("*.jsonl")):
        h.update(f.read_bytes())
    print(f"\n[FREEZE] records hash: {h.hexdigest()[:16]}  ({RECORDS_DIR})")
    print("  commit records/ + log this hash beside the old 49ffffc4f486bd8b. T18 gated on it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

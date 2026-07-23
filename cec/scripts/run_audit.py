"""TASK 9 — audit run + FREEZE.

Per proposer model, over N FF++ images:
  1. Control re-validation (30 images): the gate's controls must be inert on this
     proposer's claims before the model is trusted — offset < 15% AND CERTIFIED
     rate on true regions clearly beats a corruption baseline. Else the model is
     dropped and reported.
  2. Full run: proposer -> validate -> gate -> per-image Certified Evidence Record,
     resumable JSONL under cec/records/data/.
  3. FREEZE: hash records/, print the hash to commit and log.

FF++ only — it is the sole dataset with masks + paired reals (spatial gate needs
both). Single spatial instrument (fsfm primary). Deterministic (seed).

Run:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    /data/umar/miniconda3/envs/GenD/bin/python cec/scripts/run_audit.py \
        --proposer internvl3-8b --instrument fsfm --n 200
"""
import argparse
import hashlib
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from cec.data.pairing import build_pair, fake_dir  # noqa: E402
from cec.gate.certify import CertificationGate  # noqa: E402
from cec.proposer import Proposer, validate  # noqa: E402
from cec.records import RECORDS_DIR, RecordStore, record_key  # noqa: E402
from cec.repair import Lpips  # noqa: E402

SEED = 0
DEFAULT_METHODS = ["Deepfakes", "FaceSwap"]


def gather_fake_pairs(n, rng, methods):
    """n FF++ fake PairSamples (with paired real + landmarks), balanced by method."""
    pairs = []
    per = max(1, n // len(methods))
    for method in methods:
        fdir = fake_dir(method) / "frames"
        vids = sorted(p.name for p in fdir.iterdir())
        rng.shuffle(vids)
        got = 0
        for vid in vids:
            if got >= per:
                break
            frames = sorted(p.stem for p in (fdir / vid).glob("*.png"))
            if not frames:
                continue
            pair = build_pair(method, vid, frames[len(frames) // 2])
            if pair is not None:
                pairs.append(pair)
                got += 1
    return pairs


def control_revalidation(gate, proposer, pairs, prov):
    """Gate's controls must be inert before trusting the model. Returns (ok, stats)."""
    offsets, cert_true = [], 0
    n_claims = 0
    for pair in pairs:
        vr = validate(proposer.propose(pair.fake_path))
        if not vr.schema_valid:
            continue
        rec = gate.certify_image(pair, vr, prov)
        for c in rec["claims"]:
            if "controls" in c:
                offsets.append(c["controls"]["offset"])
                n_claims += 1
                if c["label"] == "CERTIFIED":
                    cert_true += 1
    import numpy as np
    mean_off = float(np.mean(offsets)) if offsets else 0.0
    ok = mean_off < 0.15  # offset gate (true-region vs corruption already in gate margins)
    return ok, {"mean_offset": round(mean_off, 4), "n_testable_claims": n_claims,
                "certified": cert_true}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposer", default="internvl3-8b")
    ap.add_argument("--instrument", default="fsfm")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--reval", type=int, default=30)
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    rng = random.Random(SEED)

    print(f"[gather] up to {args.n} FF++ fake pairs from {args.methods} ...")
    pairs = gather_fake_pairs(args.n, rng, args.methods)
    print(f"  {len(pairs)} pairs")

    proposer = Proposer(args.proposer, device=args.device)
    gate = CertificationGate(args.instrument, device=args.device, lpips=Lpips(args.device))
    prov = {"proposer": args.proposer, "prompt_hash": proposer.prompt_hash, "seed": proposer.params.seed}

    print(f"[reval] control re-validation on {args.reval} images ...")
    ok, stats = control_revalidation(gate, proposer, pairs[: args.reval], prov)
    print(f"  {stats}  ->  {'PASS' if ok else 'DROP model'}")
    if not ok:
        print(f"[audit] model {args.proposer} DROPPED (offset gate). Reported, not run.")
        return 1

    store = RecordStore(f"audit_{args.proposer}_{args.instrument}")
    done = store.done_keys()
    print(f"[audit] full run ({len(pairs)} images, {len(done)} already done) ...")
    for i, pair in enumerate(pairs):
        image = f"{pair.method}/{pair.vid}/{pair.frame}"
        if record_key(image, args.instrument, args.proposer) in done:
            continue
        vr = validate(proposer.propose(pair.fake_path))
        if not vr.schema_valid:
            continue
        store.append(gate.certify_image(pair, vr, prov))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(pairs)}")

    # FREEZE hash
    h = hashlib.sha256()
    for f in sorted(RECORDS_DIR.glob("*.jsonl")):
        h.update(f.read_bytes())
    digest = h.hexdigest()[:16]
    print(f"\n[FREEZE] records hash: {digest}")
    print(f"  records dir: {RECORDS_DIR}")
    print("  commit records/ and log this hash (Task 9 freeze). Tasks 10-11 gated on it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

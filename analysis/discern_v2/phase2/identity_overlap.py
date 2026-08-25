#!/usr/bin/env python3
"""Stage 1.1 — measure identity overlap between a candidate reference corpus and the test reals.

    🔴 UMAR-RUNS (GPU, ~5 min):

    python analysis/discern_v2/phase2/identity_overlap.py \
        --corpus /data/umar/Datasets/preprocessed/FFHQ-recrop/frames \
        --corpus-name FFHQ-recrop \
        --out phase2/stage1

The brief's first constraint on the diverse-real corpus is identity disjointness from Celeb-DF and
DFDC subjects, "or document the overlap and quantify its likely effect rather than proceeding
silently". A frozen `P_R` fit on familiar identities turns partly into an identity-familiarity
detector on those sources, which would recreate the V1 failure in a new coordinate system.

FFHQ has no identity metadata, so this cannot be a name lookup. It is measured with ArcFace
(`w600k_r50`, InsightFace buffalo_l) embeddings and the answer is calibrated **from the data**
rather than against a threshold quoted from memory:

  same-identity reference     Celeb-DF real videos whose names share an `idN` prefix are the
                              same subject. Their pairwise similarity is what "same person"
                              looks like for this embedding on these crops.
  different-identity          Celeb-DF pairs with different `idN` prefixes.
  the question                for each corpus image, its maximum similarity to any test real.

If the corpus→test distribution sits on top of the different-identity distribution, the corpora
are disjoint at this embedding's resolution. If a tail reaches into the same-identity range, that
tail is the overlap, and its size is the number to report.

Two limits stated up front, because they bound what this can prove
------------------------------------------------------------------
* **Crop mismatch.** ArcFace expects its own 112x112 alignment; these are DISCERN's 256x256
  GenD-template crops resized. That depresses absolute similarity. It does NOT invalidate the
  comparison, because all three distributions are computed through the same path — but it means
  the numbers are not comparable to published ArcFace verification rates.
* **Absence of evidence.** A clean result says no overlap is *detectable at this resolution*, not
  that none exists.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

ARCFACE = Path.home() / ".insightface/models/buffalo_l/w600k_r50.onnx"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
TEST_REAL_DIRS = {
    "Celeb-DF-v2": ["Celeb-real/frames", "YouTube-real/frames"],
    "Celeb-DF-v3": ["Celeb-real/frames", "YouTube-real/frames"],
}
PREPROCESSED = Path("/data/umar/Datasets/preprocessed")
ID_PATTERN = re.compile(r"^(id\d+)")


def embed(paths: list[Path], batch_size: int = 64) -> np.ndarray:
    """L2-normalised ArcFace embeddings, (N, 512)."""
    import cv2
    import onnxruntime as ort

    if not ARCFACE.is_file():
        raise SystemExit(
            f"{ARCFACE} not found. Identity overlap cannot be measured without a recognition "
            f"model, and asserting disjointness without measuring it is exactly what the brief "
            f"forbids.")
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if "CUDAExecutionProvider" in ort.get_available_providers()
                 else ["CPUExecutionProvider"])
    sess = ort.InferenceSession(str(ARCFACE), providers=providers)
    name = sess.get_inputs()[0].name

    out = []
    for start in range(0, len(paths), batch_size):
        images = []
        for p in paths[start:start + batch_size]:
            img = cv2.imread(str(p))
            if img is None:
                continue
            img = cv2.resize(img, (112, 112))
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32))
        if not images:
            continue
        blob = (np.stack(images).transpose(0, 3, 1, 2) - 127.5) / 127.5
        out.append(sess.run(None, {name: blob.astype(np.float32)})[0])
    emb = np.concatenate(out, axis=0)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-9)


def sample_frames(root: Path, per_video: int, rng: np.random.Generator) -> list[Path]:
    """One-ish frame per video: sampling many frames of one video measures nothing but itself."""
    if not root.is_dir():
        return []
    picked = []
    for video in sorted(p for p in root.iterdir() if p.is_dir()):
        frames = sorted(f for f in video.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
        if frames:
            idx = rng.choice(len(frames), size=min(per_video, len(frames)), replace=False)
            picked.extend(frames[i] for i in idx)
    return picked


def identity_of(path: Path) -> str | None:
    m = ID_PATTERN.match(path.parent.name)
    return m.group(1) if m else None


def percentiles(x: np.ndarray) -> dict:
    if len(x) == 0:
        return {}
    qs = [1, 5, 25, 50, 75, 95, 99, 100]
    return {f"p{q}": float(np.percentile(x, q)) for q in qs} | {"mean": float(x.mean()),
                                                                "n": int(len(x))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--corpus-name", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--corpus-sample", type=int, default=3000)
    ap.add_argument("--per-video", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    corpus_paths = sorted(p for p in args.corpus.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not corpus_paths:
        raise SystemExit(f"no images under {args.corpus}")
    if len(corpus_paths) > args.corpus_sample:
        idx = rng.choice(len(corpus_paths), size=args.corpus_sample, replace=False)
        corpus_paths = [corpus_paths[i] for i in sorted(idx)]

    test_paths, test_ids = [], []
    for source, subdirs in TEST_REAL_DIRS.items():
        for sub in subdirs:
            frames = sample_frames(PREPROCESSED / source / sub, args.per_video, rng)
            test_paths.extend(frames)
            test_ids.extend(f"{source}:{identity_of(f) or f.parent.name}" for f in frames)
    if not test_paths:
        raise SystemExit(f"no test-real crops found under {PREPROCESSED}; check the layout")

    print(f"corpus {args.corpus_name}: {len(corpus_paths)} images")
    print(f"test reals: {len(test_paths)} frames, {len(set(test_ids))} distinct identities")

    corpus_emb = embed(corpus_paths)
    test_emb = embed(test_paths)
    test_ids = np.asarray(test_ids[: len(test_emb)])

    # calibration: what "same person" and "different person" look like for THIS embedding on
    # THESE crops, rather than a verification threshold quoted from elsewhere
    sim_tt = test_emb @ test_emb.T
    same = test_ids[:, None] == test_ids[None, :]
    np.fill_diagonal(same, False)
    triu = np.triu(np.ones_like(sim_tt, dtype=bool), k=1)
    same_identity = sim_tt[same & triu]
    diff_identity = sim_tt[(~same) & triu]

    # the question
    sim_ct = corpus_emb @ test_emb.T
    max_sim = sim_ct.max(axis=1)
    nearest = test_ids[sim_ct.argmax(axis=1)]

    # the overlap threshold, read off the calibration instead of assumed: the similarity below
    # which 99% of genuinely different identities fall
    threshold = float(np.percentile(diff_identity, 99)) if len(diff_identity) else float("nan")
    flagged = int((max_sim > threshold).sum())

    report = {
        "corpus": {"name": args.corpus_name, "path": str(args.corpus),
                   "n_sampled": len(corpus_paths), "n_total": len(list(args.corpus.rglob("*")))},
        "test_reals": {"sources": list(TEST_REAL_DIRS), "n_frames": len(test_paths),
                       "n_identities": int(len(set(test_ids.tolist())))},
        "embedding": {"model": str(ARCFACE), "note":
                      "DISCERN 256x256 GenD crops resized to 112x112; absolute similarities are "
                      "not comparable to published ArcFace verification numbers"},
        "calibration": {"same_identity": percentiles(same_identity),
                        "different_identity": percentiles(diff_identity)},
        "corpus_to_test_max_similarity": percentiles(max_sim),
        "overlap_threshold": {"value": threshold,
                              "derived_as": "99th percentile of the different-identity "
                                            "distribution measured on the test reals themselves"},
        "flagged": {"n": flagged, "fraction": flagged / len(max_sim),
                    "nearest_identities": sorted(set(nearest[max_sim > threshold].tolist()))[:20]},
    }
    (args.out / f"identity_overlap_{args.corpus_name}.json").write_text(
        json.dumps(report, indent=2))

    print(f"\ncalibration on the test reals themselves:")
    print(f"  same identity      median {np.median(same_identity):.3f}  "
          f"(n={len(same_identity)})")
    print(f"  different identity median {np.median(diff_identity):.3f}  p99 {threshold:.3f}  "
          f"(n={len(diff_identity)})")
    print(f"\n{args.corpus_name} -> test reals, max similarity per image:")
    print(f"  median {np.median(max_sim):.3f}  p95 {np.percentile(max_sim, 95):.3f}  "
          f"max {max_sim.max():.3f}")
    print(f"\n{flagged}/{len(max_sim)} ({flagged / len(max_sim):.4f}) exceed the "
          f"different-identity p99 threshold {threshold:.3f}")
    if flagged / len(max_sim) > 0.01:
        print("  ⚠️  a non-trivial tail reaches into same-identity territory. Document it in "
              "STAGE1_REFERENCE.md and quantify the effect; do not proceed silently.")
    else:
        print("  no overlap detectable AT THIS EMBEDDING'S RESOLUTION — which is not the same "
              "as proving none exists. Word the paper accordingly.")
    print(f"\nwrote {args.out}/identity_overlap_{args.corpus_name}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

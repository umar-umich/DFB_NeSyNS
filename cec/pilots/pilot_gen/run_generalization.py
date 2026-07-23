"""Cross-dataset generalization pilot — which detector is the certification instrument?

The whole framework anchors certification to ONE frozen detector. That choice
must be made on cross-dataset generalization, not FF++ alone. This scores all
four spatial detectors (effort, fsfm, gend, forada) as AUC(fake vs real) on the
field-standard generalization benchmarks — Celeb-DF-v2 and DFDC — with FF++ as
the in-domain reference.

Certification-gate behaviour (GT-region repair) is NOT tested here: Celeb-DF and
DFDC ship no masks (same as DF40), so region repair can't run. AUC is the
relevant number — it decides the instrument; the gate itself was already
validated byte-exact on FF++ (Task 4).

Random balanced samples, one frame per clip, fixed seed.

Run:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    /data/umar/miniconda3/envs/GenD/bin/python cec/pilots/pilot_gen/run_generalization.py --n 200
"""
import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from cec.instruments.spatial import INSTRUMENTS, SpatialInstrument  # noqa: E402

PRE = Path("/data/umar/Datasets/preprocessed")
OUT = REPO / "results" / "pilot_gen"
SEED = 0


def _first_frame(frames_dir: Path):
    if not frames_dir.is_dir():
        return None
    pngs = sorted(frames_dir.glob("*.png"))
    return pngs[len(pngs) // 2] if pngs else None  # middle frame


def gather_celebdf(n, rng):
    """(paths, labels) from the official Celeb-DF-v2 test list. label: 1=fake, 0=real."""
    root = PRE / "Celeb-DF-v2"
    lines = (root / "List_of_testing_videos.txt").read_text().strip().splitlines()
    fakes, reals = [], []
    for ln in lines:
        lab, rel = ln.split()
        sub, clip = rel.split("/")
        clip = clip[:-4] if clip.endswith(".mp4") else clip
        fp = _first_frame(root / sub / "frames" / clip)
        if fp is None:
            continue
        (reals if lab == "1" else fakes).append(fp)  # list: 1=real, 0=fake
    rng.shuffle(fakes)
    rng.shuffle(reals)
    k = min(n // 2, len(fakes), len(reals))
    paths = fakes[:k] + reals[:k]
    labels = [1] * k + [0] * k
    return paths, labels


def gather_dfdc(n, rng):
    """(paths, labels) from DFDC metadata (is_fake). label: 1=fake, 0=real."""
    root = PRE / "DFDC" / "test"
    meta = json.loads((root / "metadata.json").read_text())
    fakes, reals = [], []
    for clip, info in meta.items():
        fp = _first_frame(root / "frames" / clip[:-4])
        if fp is None:
            continue
        (fakes if info["is_fake"] == 1 else reals).append(fp)
    rng.shuffle(fakes)
    rng.shuffle(reals)
    k = min(n // 2, len(fakes), len(reals))
    paths = fakes[:k] + reals[:k]
    labels = [1] * k + [0] * k
    return paths, labels


def gather_ffpp(n, rng):
    """FF++ in-domain reference (Deepfakes+FaceSwap fakes vs youtube reals)."""
    root = PRE / "FaceForensics++"
    fakes, reals = [], []
    for method in ("Deepfakes", "FaceSwap"):
        fdir = root / "manipulated_sequences" / method / "c23" / "frames"
        for vid in sorted(p.name for p in fdir.iterdir()):
            fp = _first_frame(fdir / vid)
            if fp is not None:
                fakes.append(fp)
    rdir = root / "original_sequences" / "youtube" / "c23" / "frames"
    for vid in sorted(p.name for p in rdir.iterdir()):
        fp = _first_frame(rdir / vid)
        if fp is not None:
            reals.append(fp)
    rng.shuffle(fakes)
    rng.shuffle(reals)
    k = min(n // 2, len(fakes), len(reals))
    return fakes[:k] + reals[:k], [1] * k + [0] * k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="samples per dataset (half fake, half real)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    datasets = {}
    for name, fn in (("ffpp", gather_ffpp), ("celebdf_v2", gather_celebdf), ("dfdc", gather_dfdc)):
        paths, labels = fn(args.n, rng)
        datasets[name] = (paths, labels)
        print(f"[gather] {name}: {sum(labels)} fake / {len(labels) - sum(labels)} real")

    results = {"n": args.n, "seed": SEED, "auc": {}}
    print(f"\n{'dataset':<14}" + "".join(f"{d:>10}" for d in INSTRUMENTS))
    for ds, (paths, labels) in datasets.items():
        imgs = [cv2.imread(str(p)) for p in paths]
        keep = [(im, y) for im, y in zip(imgs, labels) if im is not None]
        imgs = [im for im, _ in keep]
        y = np.array([yy for _, yy in keep])
        row = {}
        for det in INSTRUMENTS:
            inst = SpatialInstrument(det, device=args.device)
            scores = inst.p_fake_batch(imgs)
            row[det] = float(roc_auc_score(y, scores)) if len(set(y.tolist())) == 2 else float("nan")
        results["auc"][ds] = row
        print(f"{ds:<14}" + "".join(f"{row[d]:>10.3f}" for d in INSTRUMENTS))

    (OUT / "generalization.json").write_text(json.dumps(results, indent=2))
    print(f"\n[written] {OUT / 'generalization.json'}")
    print("Instrument choice: the detector with the best CROSS-DATASET AUC "
          "(celebdf_v2, dfdc), not just FF++. Umar's call.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

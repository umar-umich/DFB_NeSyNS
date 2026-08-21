#!/usr/bin/env python3
"""Stage 1 external baseline — build the ImageFolder layout the official FS-VFM protocol expects.

    🔴 UMAR-RUNS (CPU, minutes; creates symlinks only):

    python analysis/fpad/build_fsvfm_imagefolder.py --out /data/umar/Datasets/fsvfm_lp

The brief requires the external baseline because "`B0` as our frozen head is our own use of
FS-VFM and a reviewer will call it weak". So we run the AUTHORS' downstream protocol
(`FSFM-CVPR25/fsvfm/linearprobe/cross_dataset_DFD_and_DiFF`) on the SAME FF++ c23 train split and
the same OOD test sets, and report it as its own row.

Their loader is a plain `torchvision.ImageFolder`, and their test loader derives video identity
from the FILENAME:

    video_name = path.split('/')[-1].split('_frame_')[0]        util/datasets.py:379

So video-level AUROC — which is the metric our main table uses — is only available if the files are
named `<video>_frame_<n>.png`. That naming is the reason this builder exists rather than pointing
their loader at our tree directly.

Symlinks, never copies
----------------------
Our preprocessed FF++ alone is 45 GB and `/data` sits at 99% full. Every file here is a symlink to
the frame we already have, so the layout costs kilobytes. It also guarantees the baseline reads
*byte-identical pixels* to the ones our own rungs read, which a re-extraction would not.

Splits come from OUR dataset JSONs
----------------------------------
The video lists are read from the same JSON manifests `NeSyDeFakeDataset` uses, so the baseline
trains on exactly the FF++ videos our students train on and tests on exactly the OOD videos our
rungs are evaluated on. A baseline run on a different split is not a baseline, it is a different
experiment.

Class directories are `0_real` and `1_fake`: ImageFolder assigns labels by sorted class name, so
this fixes real=0 and fake=1 to match our convention. Getting it backwards silently inverts every
AUROC.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(REPO / "analysis" / "discern_v2"))

JSON_DIR = Path("/data/umar/Datasets/preprocessed/dataset_json")
CLASSES = {0: "0_real", 1: "1_fake"}
# The OOD suite the main table reports. `DF40` is deliberately absent: the external baseline is a
# row in the conventional table, and DF40 is handled by the sealed Dev/Holdout split.
OOD = ("Celeb-DF-v1", "Celeb-DF-v2", "Celeb-DF-v3", "DFDC", "DFDCP", "DeepFakeDetection",
       "UADFV", "Deepfake-Eval-2024")


def read_manifest(source: str) -> dict:
    path = JSON_DIR / f"{source}.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def iter_frames(blob: dict, split: str, compression: str, label_dict: dict[str, int],
                source: str) -> list[tuple[str, str, int]]:
    """(video, frame_path, label) for one split, walking the manifest the loader walks."""
    out = []
    root = blob[next(iter(blob))]
    for label_key, splits in root.items():
        label = label_from_key(label_key, label_dict, source)
        section = splits.get(split) or {}
        # some manifests nest a compression level under the split
        if section and all(k in ("c23", "c40", "raw") for k in list(section)[:2]):
            section = section.get(compression, {})
        for video, info in section.items():
            for frame in (info or {}).get("frames") or []:
                # The label key (the manipulation) is part of the video identity. FF++ names the
                # same source video identically under all four manipulations, so `757_573` alone
                # is NOT unique — see the collision note in `link_split`.
                out.append((f"{label_key}-{video}", str(frame), label))
    return out


def load_label_dict() -> dict[str, int]:
    """The repo's authoritative real/fake mapping, read from config — never inferred.

    FF++'s manifest keys are `FF-real`, `FF-DF`, `FF-F2F`, `FF-NT`, `FF-FS`. A substring matcher
    over those returns None for every FAKE key, which produced a train split of 23,039 reals and
    ZERO fakes — a baseline that would have trained, converged and reported numbers on one class.
    So the mapping comes from `test_config.yaml`'s `label_dict`, which is what the real loader
    uses, and an unmapped key RAISES rather than being silently skipped.
    """
    import yaml

    merged: dict[str, int] = {}
    for name in ("test_config.yaml", "train_config.yaml"):
        path = REPO / "training" / "config" / name
        if path.is_file():
            merged.update((yaml.safe_load(path.read_text()) or {}).get("label_dict") or {})
    if not merged:
        raise SystemExit("no label_dict found in training/config — refusing to guess real/fake")
    return merged


def label_from_key(key: str, label_dict: dict[str, int], source: str) -> int:
    if key in label_dict:
        return int(label_dict[key])
    raise SystemExit(
        f"{source}: manifest label key {key!r} is not in the repo's label_dict, so whether it is "
        f"real or fake is unknown. Refusing to guess — a flipped or dropped label silently "
        f"inverts or empties this baseline's numbers. Add it to training/config/test_config.yaml.")


def link_split(rows: list[tuple[str, str, int]], dest: Path, dry_run: bool) -> dict:
    """Symlink one split into ImageFolder layout, refusing to lose frames to name collisions.

    The filename must be `<video>_frame_<n>.<ext>`, because their test loader derives video
    identity as `filename.split('_frame_')[0]` (`util/datasets.py:379`). That makes the name
    load-bearing twice over, and the first version of this function got it wrong in a way that
    still produced a plausible tree:

    FF++ names the same source video identically under all four manipulations, and frame numbers
    repeat too, so `757_573_frame_354.png` was claimed by Deepfakes, Face2Face, FaceSwap AND
    NeuralTextures. Skipping an existing link silently dropped 48,202 of 92,159 fakes — and worse,
    it would have pooled four different manipulations of one video into one "video" for
    video-level AUROC. The manipulation is now part of the video id, and a collision between two
    DIFFERENT targets is counted and raised rather than skipped.
    """
    counts = {"0_real": 0, "1_fake": 0}
    missing = 0
    claimed: dict[Path, str] = {}
    collisions = []
    for cls in CLASSES.values():
        (dest / cls).mkdir(parents=True, exist_ok=True)
    for video, frame, label in rows:
        src = Path(frame)
        if not src.is_file():
            missing += 1
            continue
        cls = CLASSES[label]
        name = f"{video.replace('/', '-')}_frame_{src.stem}{src.suffix}"
        link = dest / cls / name
        target = str(src.resolve())
        if link in claimed:
            if claimed[link] != target:
                collisions.append((name, claimed[link], target))
            continue
        claimed[link] = target
        counts[cls] += 1
        if dry_run:
            continue
        if link.is_symlink() or link.exists():
            if os.path.realpath(link) != target:
                link.unlink()
                os.symlink(target, link)
            continue
        os.symlink(target, link)
    if collisions:
        raise SystemExit(
            f"{len(collisions)} filename collisions between DIFFERENT frames, e.g. "
            f"{collisions[0][0]} claimed by both {collisions[0][1]} and {collisions[0][2]}. "
            f"Silently skipping these would drop frames AND merge distinct videos under one id "
            f"for video-level AUROC.")
    return {"counts": counts, "missing_frames": missing,
            "n_videos": len({v for v, _, _ in rows})}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--compression", default="c23")
    ap.add_argument("--sources", nargs="+", default=list(OOD))
    ap.add_argument("--dry-run", action="store_true",
                    help="count what would be linked without creating anything")
    args = ap.parse_args()

    label_dict = load_label_dict()
    print(f"label_dict: {len(label_dict)} entries from training/config (authoritative)")
    report: dict = {"out": str(args.out), "compression": args.compression,
                    "classes": CLASSES, "symlinks_only": True, "splits": {}}

    ffpp = read_manifest("FaceForensics++")
    if not ffpp:
        raise SystemExit(f"no FaceForensics++.json under {JSON_DIR}")
    for split in ("train", "val"):
        rows = iter_frames(ffpp, split, args.compression, label_dict, "FaceForensics++")
        if not rows:
            raise SystemExit(f"FF++ {split}: no frames found — check the manifest's label keys")
        info = link_split(rows, args.out / "FFpp_c23" / split, args.dry_run)
        if min(info["counts"].values()) == 0:
            raise SystemExit(
                f"FF++ {split} resolved to a SINGLE class {info['counts']}. A linear probe trained "
                f"on one class is not a baseline; it is a bug that still produces numbers.")
        report["splits"][f"FFpp_c23/{split}"] = info
        print(f"  FF++ {split:5s}: {info['counts']} over {info['n_videos']} videos"
              + (f"  ({info['missing_frames']} frames missing)" if info["missing_frames"] else ""))

    for source in args.sources:
        blob = read_manifest(source)
        if not blob:
            report["splits"][source] = {"status": "no manifest"}
            print(f"  {source:22s} no manifest — row stays TODO(run)")
            continue
        rows = iter_frames(blob, "test", args.compression, label_dict, source)
        if not rows:
            report["splits"][source] = {"status": "no test frames"}
            print(f"  {source:22s} manifest has no usable test split")
            continue
        info = link_split(rows, args.out / source / "test", args.dry_run)
        report["splits"][source] = info
        print(f"  {source:22s} {info['counts']} over {info['n_videos']} videos"
              + (f"  ({info['missing_frames']} missing)" if info["missing_frames"] else ""))

    if not args.dry_run:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "build_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\n{'DRY RUN — nothing created' if args.dry_run else f'wrote {args.out}'}")
    print("\nNext (🔴, GPU): the authors' protocol, unchanged, on this tree —")
    print("  cd /data/umar/Repos/FSFM-CVPR25/fsvfm/linearprobe/cross_dataset_DFD_and_DiFF")
    print("  python main_linearprobe_DfD.py --model vit_large_patch16 --nb_classes 2 \\")
    print("      --apply_simple_augment --batch_size 128 --epochs 50 --blr 1e-2 \\")
    print("      --layer_decay 0 --weight_decay 0 --drop_path 0.1 --reprob 0.25 \\")
    print("      --mixup 0.8 --cutmix 1.0 \\")
    print(f"      --finetune {REPO}/weights/FS-VFM/checkpoint-599.pth \\")
    print(f"      --finetune_data_path {args.out}/FFpp_c23 --output_dir <run dir>")
    print("  (their ViT-L hyperparameters, from scripts_DFD/run_LP_DfD-ViT-L.sh, unchanged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

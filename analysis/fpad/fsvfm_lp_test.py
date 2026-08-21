#!/usr/bin/env python3
"""Evaluate an official FS-VFM linear-probe checkpoint on OUR sources, with THEIR test code.

    🔴 UMAR-RUNS (GPU):

    python analysis/fpad/fsvfm_lp_test.py \
        --resume logs/fpad/fsvfm_lp/checkpoint-min_val_loss.pth \
        --data-root /data/umar/Datasets/fsvfm_lp \
        --out docs/DiCoME_eval/fsvfm_linearprobe.json

Why a driver rather than their script
-------------------------------------
`main_test_DfD.py` loops a HARDCODED `cross_dataset_test_path` dict pointing at
`../../../datasets/finetune_datasets/...`, which does not exist on this machine. Editing that dict
would mean modifying the read-only baseline repo, and the honest alternative is to *call* their
code rather than change it. So this imports their `models_vit`, their `build_transform`, their
`TestImageFolder` and their `test_binary_video_frames` — the same model, transform, video grouping
and metric — and loops over our sources instead of theirs.

`test_binary_video_frames` is the function that produces `video_auc`: it reads `batch[-1]` as the
video name, which `TestImageFolder` appends by splitting the filename on `_frame_`. That is why the
symlink tree names files `<manipulation>-<video>_frame_<n>.png`.

Normalization is read from `pretrain_ds_mean_std.txt` beside `--resume`, exactly as their code
does. If that file is missing their loader silently falls back to ImageNet statistics — which for
this checkpoint is wrong and would depress every number — so this refuses instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

FSFM = Path("/data/umar/Repos/FSFM-CVPR25/fsvfm")
FINETUNE_DIR = FSFM / "finetune" / "cross_dataset_DFD_and_DiFF"
SKIP = {"FFpp_c23"}          # the training split; not an OOD row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--resume", type=Path, required=True, help="the LP checkpoint")
    ap.add_argument("--data-root", type=Path, required=True,
                    help="the ImageFolder tree from build_fsvfm_imagefolder.py")
    ap.add_argument("--model", default="vit_large_patch16")
    ap.add_argument("--nb-classes", type=int, default=2)
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sources", nargs="+", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    stats = args.resume.parent / "pretrain_ds_mean_std.txt"
    if not stats.is_file():
        raise SystemExit(
            f"{stats} is missing. Their `build_transform` reads the normalization from beside the "
            f"resumed checkpoint and SILENTLY falls back to ImageNet statistics when it cannot "
            f"find it — which is wrong for this checkpoint and would depress every number "
            f"without erroring. Copy weights/FS-VFM/pretrain_ds_mean_std.txt next to the "
            f"checkpoint before evaluating.")

    # their code, imported not copied
    sys.path.insert(0, str(FINETUNE_DIR))
    sys.path.insert(0, str(FSFM))
    import models_vit                                    # noqa: E402
    from engine_finetune import test_binary_video_frames  # noqa: E402
    from util.datasets import TestImageFolder, build_transform  # noqa: E402

    device = torch.device(args.device)
    model = models_vit.__dict__[args.model](num_classes=args.nb_classes, global_pool=True)
    blob = torch.load(str(args.resume), map_location="cpu", weights_only=False)
    state = blob.get("model", blob)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if any("blocks" in k or "head" in k for k in missing):
        raise SystemExit(f"checkpoint is missing backbone/head tensors: {list(missing)[:5]}")
    model.to(device).eval()
    print(f"loaded {args.resume} (epoch {blob.get('epoch', '?')})")
    if unexpected:
        print(f"  ignored {len(unexpected)} unexpected keys")

    # the args their transform expects
    targs = SimpleNamespace(input_size=args.input_size, eval=True, resume=str(args.resume),
                            normalize_from_IMN=False, output_dir=str(args.resume.parent),
                            finetune="", color_jitter=None, aa="rand-m9-mstd0.5-inc1",
                            reprob=0.0, remode="pixel", recount=1, apply_simple_augment=True)
    transform = build_transform(False, targs)
    print(f"  transform: {transform}")

    sources = args.sources or sorted(
        p.name for p in args.data_root.iterdir()
        if p.is_dir() and p.name not in SKIP and (p / "test").is_dir())
    results, detail = {}, {}
    for source in sources:
        root = args.data_root / source / "test"
        dataset = TestImageFolder(str(root), transform=transform)
        loader = torch.utils.data.DataLoader(
            dataset, sampler=torch.utils.data.SequentialSampler(dataset),
            batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
            drop_last=False)
        stats_out = test_binary_video_frames(loader, model, device)
        results[source] = float(stats_out["video_auc"])
        detail[source] = {k: float(v) for k, v in stats_out.items()
                          if isinstance(v, (int, float))}
        print(f"  {source:22s} frames={len(dataset):7d}  "
              f"frame_auc={stats_out.get('frame_auc', float('nan')):.4f}  "
              f"video_auc={stats_out['video_auc']:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    args.out.with_name(args.out.stem + "_detail.json").write_text(json.dumps({
        "checkpoint": str(args.resume), "model": args.model,
        "epochs_trained": blob.get("epoch"),
        "protocol": "official FS-VFM linear probe (FSFM-CVPR25 "
                    "fsvfm/linearprobe/cross_dataset_DFD_and_DiFF), hyperparameters from "
                    "scripts_DFD/run_LP_DfD-ViT-L.sh",
        "deviations": [
            "crops are RetinaFace + GenD template at scale 1.3, not the authors' DLIB + 30% — "
            "chosen so the baseline reads byte-identical pixels to our own rungs",
            "epoch count may differ from the authors' 50; see epochs_trained",
        ],
        "per_source": detail}, indent=2, default=str))
    print(f"\nwrote {args.out} — pass it to main_table.py --external")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

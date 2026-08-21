#!/usr/bin/env python3
"""Stage 1.2 — cache frozen FS-VFM features for a directory of face crops.

    🔴 UMAR-RUNS (GPU, minutes):

    python analysis/discern_v2/phase2/cache_image_features.py \
        --images /data/umar/Datasets/preprocessed/FFHQ-recrop/frames \
        --name FFHQ-recrop \
        --out cache/discern_v2/fsvfm_frozen

`analysis/discern_v2/cache_encoder_features.py` reaches only corpora that have a dataset JSON,
because it builds a `NeSyDeFakeDataset`. A diverse-real corpus is a bare image directory, so this
covers that case — and only that case. The encoder, its normalization and the output format are
imported from the existing cacher rather than reimplemented, so a feature cached here and a
feature cached there are the same feature.

Everything written here is labelled real (`label = 0`). That is not an assumption about the
directory; it is what the caller asserts by pointing this at a *reference* corpus, and
`fit_reference.py` filters on those labels, so a mislabelled directory would put fakes into the
bona-fide manifold. `--confirm-all-real` exists so that assertion is explicit at the call site.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(REPO / "analysis" / "discern_v2"))

from cache_encoder_features import build_fsvfm, git_commit, merge_manifest  # noqa: E402

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class ImageDirDataset(torch.utils.data.Dataset):
    """Crops as `raw_frames` in [0, 1] — FS-VFM applies its own statistics downstream (§6)."""

    def __init__(self, paths: list[Path], size: int = 224):
        self.paths = paths
        self.size = size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> dict:
        img = Image.open(self.paths[i]).convert("RGB").resize(
            (self.size, self.size), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return {"raw_frames": x, "key": str(self.paths[i])}

    @staticmethod
    def collate(batch: list[dict]) -> dict:
        return {"raw_frames": torch.stack([b["raw_frames"] for b in batch]),
                "key": [b["key"] for b in batch]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--name", required=True, help="cache slice name, e.g. FFHQ-recrop")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path,
                    default=REPO / "training/config/detector/nesy_defake_d1_v.yaml")
    ap.add_argument("--pooling", choices=("global_pool", "cls"), default="global_pool")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--confirm-all-real", action="store_true", required=True,
                    help="assert that every image here is authentic; they are cached as label 0 "
                         "and P_R is fit on exactly these")
    args = ap.parse_args()

    paths = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise SystemExit(f"no images under {args.images}")
    print(f"{len(paths)} images from {args.images}")

    config = yaml.safe_load(args.config.read_text())
    encoder, provenance, featurize = build_fsvfm(config, args.device, args.pooling,
                                                 args.checkpoint)
    print(f"  FS-VFM epoch={provenance['epoch']} pooling={args.pooling}")

    loader = torch.utils.data.DataLoader(
        ImageDirDataset(paths), batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=ImageDirDataset.collate)

    chunks, keys = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            chunks.append(featurize(batch).cpu().numpy())
            keys.extend(batch["key"])
            if (i + 1) % 20 == 0:
                print(f"  batch {i + 1}/{len(loader)}")
    features = np.concatenate(chunks, axis=0).astype(np.float32)
    labels = np.zeros(len(features), dtype=np.int64)

    dest = args.out / f"{args.name}_real"
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / "features.npz", f0=features, labels=labels)
    # standalone too: fit_reference.py's --labels loader takes no key and would otherwise pick
    # the first 2-D array in the npz, i.e. the features
    np.save(dest / "labels.npy", labels)
    pd.DataFrame({"key": keys, "label": labels, "method": "real", "video_id": keys,
                  "source": args.name, "split": "reference"}).to_csv(
        dest / "metadata.csv", index=False)

    # The manifest keys are the ones `fit_reference.assert_frozen_feature_space` actually reads
    # — `encoder_frozen` and `fingerprint`, not just a human-readable mode string. A cache
    # written without them is refused downstream, which is the correct behaviour but a confusing
    # place to discover a missing field.
    manifest = merge_manifest(args.out, {
        "encoder_mode": "frozen",
        "encoder_frozen": True,
        "checkpoint": provenance["checkpoint"],
        "config": str(args.config),
        "backbone_family": "fsvfm",
        "backbone": {"name": "fsvfm_vit_large_patch16", "pooling": args.pooling,
                     "checkpoint": provenance["checkpoint"], "epoch": provenance["epoch"],
                     "mean": provenance["mean"], "std": provenance["std"], "feature": "z_ref"},
        "fsvfm_provenance": provenance,
        "fingerprint": {"all": provenance["fingerprint"], "layernorm": None,
                        "n_params": sum(p.numel() for p in encoder.parameters()),
                        "n_layernorm_params": None},
        "feature": "z_ref",
        "input": "image directory (already cropped) -> FS-VFM normalization",
        "git_commit": git_commit(), "datasets": {}})
    manifest["datasets"][f"{args.name}/reference"] = {
        "n": int(features.shape[0]), "dim": int(features.shape[1]),
        "n_real": int(len(labels)), "n_fake": 0, "split": "reference", "partial": False,
        "path": str(dest), "source_images": str(args.images),
        "all_real_confirmed_by_caller": True}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    print(f"\nwrote {dest}/features.npz  {features.shape}")
    print("\nNext (Stage 1.2) — fit P_R on the COMBINED diverse-real population. Fitting on this "
          "corpus alone would trade one narrow reference for another:")
    print(f"  python analysis/discern_v2/phase2/fit_diverse_reference.py \\\n"
          f"      --features cache/discern_v2/fsvfm_frozen/FaceForensics++_train/features.npz \\\n"
          f"                 {dest}/features.npz \\\n"
          f"      --out configs/discern_v2/reference_diverse")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

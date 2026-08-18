#!/usr/bin/env python3
"""Cache encoder features — the artifact Stage I, Task 0 and E1 all read.

`fit_reference.py` asks for "a cached CLIP feature matrix for FF++ authentic training frames,
from a FROZEN encoder" and nothing in the repo produced one, so Stage I could not run. This
script is that producer. It is also what Task 0 (LN-tuned vs frozen encoder) and the E1
domain-detector audit consume, because all three need the *same* feature space to be
comparable — a reference fit on one cache and audited on another measures encoder drift, not
forensics.

What it caches
--------------
`spatial_raw` — the (N, 1024) raw CLIP ViT-L/14 CLS feature, taken from
`SpatialFeatureExtractor`, which is the tensor the detector hands to the v2 stack as
`visual_feature` when `manifold.input: spatial_raw`. Deliberately NOT `projected`: the
post-`spatial_proj` feature is trained, so it drifts every epoch, and a reference fit in it is
measured against a stale manifold the moment training continues. That is the Phase-1 bug in a
different costume.

Only the spatial extractor is built, not the whole detector. Building
`nesydefake_hybrid` would pull in the SDXL-VAE process operator and the semantic modules,
none of which contribute to `spatial_raw` — that is minutes of load time and GB of VRAM for a
tensor they do not touch.

The two encoder modes (Task 0)
------------------------------
    --encoder frozen           pretrained CLIP as published; no checkpoint may be passed
    --encoder tuned  --checkpoint <run.pth>    LN weights as a training run left them

Task 0 compares the two on the conventional axis and Umar locks the decision. Whatever it
decides, `assert_reference_config` still requires the *reference's* input space to be frozen,
so the `frozen` cache is the one Stage I fits on in either case.

The fingerprint guard
---------------------
Every cache records a sha256 over the backbone's parameters (and separately over just its
LayerNorm parameters, which are the only ones a GenD-style run moves). A reference artifact
carries the fingerprint of the cache it was fit on, so a later stage can *prove* it is reading
the same encoder state rather than assuming it. Without this the stale-manifold failure is
silent: the residuals stay finite, plausible, and meaningless.

Splits
------
`--split {train,val,test}`. The dataset JSON has all three, but `abstract_dataset` only
accepts `train`/`test` as a *mode* — and mode also switches on augmentation and shuffling,
which must be off for a feature cache. So the dataset is always constructed in `test` mode
(test-time transforms) and the split key is swapped for collection only. A cache built with
train-mode augmentation would make the reference's authentic manifold a manifold of *augmented*
reals, and `mu_R, sigma_R` would be calibrated to a distribution that never occurs at
inference.

Usage (🔴 UMAR-RUNS — no training here, forward passes only):

    # Stage I input: FF++ authentic train frames, frozen encoder
    python analysis/discern_v2/cache_encoder_features.py \
        --config training/config/detector/nesy_defake_d1_v.yaml \
        --encoder frozen --datasets FaceForensics++ --split train \
        --out cache/discern_v2/clip_frozen

    # E1 audit input: OOD reals and fakes, same frozen encoder
    python analysis/discern_v2/cache_encoder_features.py \
        --config training/config/detector/nesy_defake_d1_v.yaml \
        --encoder frozen --datasets Celeb-DF-v3 DFDC DFDCP --split test \
        --out cache/discern_v2/clip_frozen

    # Task 0 second arm: the same sources through the LN-tuned encoder
    python analysis/discern_v2/cache_encoder_features.py \
        --config training/config/detector/nesy_defake_d1_v.yaml \
        --encoder tuned --checkpoint logs/<run>/ckpt_best_avg.pth \
        --datasets FaceForensics++ Celeb-DF-v3 --split test \
        --out cache/discern_v2/clip_tuned
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "training"))

from dataset.nesy_defake_dataset import NeSyDeFakeDataset  # noqa: E402
from networks.nesy_defake.foundation_models import SpatialFeatureExtractor  # noqa: E402

SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def load_config(detector_path: Path) -> dict:
    """Merge the detector yaml with test_config.yaml, exactly as training/test.py does.

    Ported rather than re-derived: the dataset needs keys (`dataset_json_folder`,
    `rgb_dir`, …) that live only in the shared config, and a cache built from a
    differently-merged config would silently read different frames.
    """
    with open(detector_path) as f:
        config = yaml.safe_load(f)
    shared = REPO / "training" / "config" / "test_config.yaml"
    if shared.exists():
        with open(shared) as f:
            config.update(yaml.safe_load(f))
    return config


def strip_unused_loading(config: dict) -> dict:
    """Turn off the per-sample side loads the encoder never reads.

    `spatial_frames` is the only tensor that reaches CLIP. The Face-LLaVA attributes and the
    precomputed forensic features are one `.pt`/`.npy` read per frame each, which dominates
    wall-clock over a 700-video split and cannot change a single cached value.
    """
    config["load_semantic_features"] = False
    for key in ("fast_semantic", "forensic_features"):
        if isinstance(config.get(key), dict):
            config[key] = {**config[key], "enabled": False}
    # augmentation is mode-gated, but make the intent explicit and testable
    config["use_data_augmentation"] = False
    config["balance_classes"] = False
    config["paired_training"] = False
    return config


# ---------------------------------------------------------------------------
# encoder
# ---------------------------------------------------------------------------


def _sha256_of_params(named: list[tuple[str, torch.Tensor]]) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(named, key=lambda kv: kv[0]):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().float().numpy().tobytes())
    return h.hexdigest()


def encoder_fingerprint(extractor: SpatialFeatureExtractor) -> dict:
    """Fingerprint the backbone, and its LayerNorms separately.

    Two hashes rather than one because they answer different questions. `all` proves the cache
    came from this exact encoder state. `layernorm` isolates the parameters a GenD-style run
    actually moves, so a frozen-vs-tuned comparison can state *how much* moved instead of just
    "the hashes differ" — which they would for any unrelated reason too.
    """
    backbone = extractor.backbone
    ln = [(n, p) for n, p in backbone.named_parameters()
          if any(tok in n.lower() for tok in ("layernorm", "layer_norm", "ln_", ".norm"))]
    return {
        "all": _sha256_of_params(list(backbone.named_parameters())),
        "layernorm": _sha256_of_params(ln),
        "n_params": sum(p.numel() for p in backbone.parameters()),
        "n_layernorm_params": sum(p.numel() for _, p in ln),
    }


def build_encoder(config: dict, mode: str, checkpoint: Path | None,
                  device: str) -> SpatialFeatureExtractor:
    """Build the spatial extractor in one of the two Task-0 encoder states.

    The mode/checkpoint coupling is enforced, not documented: `frozen` with a checkpoint
    would produce a cache labelled frozen that carries tuned LN weights, and every downstream
    audit would inherit the mislabel with nothing to reveal it.
    """
    if mode == "frozen" and checkpoint is not None:
        raise ValueError(
            "--encoder frozen takes no --checkpoint. Loading tuned LN weights into a cache "
            "labelled 'frozen' is the exact provenance error the fingerprint exists to catch; "
            "use --encoder tuned if you want the checkpoint's encoder.")
    if mode == "tuned" and checkpoint is None:
        raise ValueError("--encoder tuned requires --checkpoint <run weights>")

    if mode == "frozen":
        # LN training is a *training* switch; for a frozen snapshot it must be off so the
        # extractor reports (and the guard can assert) a fully frozen backbone.
        config = {**config, "foundation_models": {
            **config["foundation_models"],
            "spatial": {**config["foundation_models"]["spatial"], "train_layernorms": False},
        }}

    extractor = SpatialFeatureExtractor(config).to(device)

    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        prefix = "spatial_extractor."
        sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if not sub:
            raise KeyError(
                f"{checkpoint} has no '{prefix}*' keys — this does not look like a "
                f"nesydefake_hybrid checkpoint, and silently caching the pretrained encoder "
                f"under --encoder tuned would invalidate Task 0.")
        missing, unexpected = extractor.load_state_dict(sub, strict=False)
        # strict=False so a checkpoint from a differently-configured run still loads its
        # encoder, but report what did not line up — a large `missing` set means the tuned
        # arm is mostly pretrained weights and the Task-0 delta would be diluted.
        print(f"  loaded {len(sub)} encoder tensors from {checkpoint}")
        if missing:
            print(f"  WARNING: {len(missing)} extractor params NOT in checkpoint "
                  f"(first: {list(missing)[:3]})")
        if unexpected:
            print(f"  WARNING: {len(unexpected)} checkpoint params unused "
                  f"(first: {list(unexpected)[:3]})")

    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad_(False)
    return extractor


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def load_split(config: dict, dataset_name: str, split: str) -> NeSyDeFakeDataset:
    """Build a dataset over `split` with test-time transforms.

    `abstract_dataset` keys the JSON on `self.mode` and also gates augmentation/shuffling on
    it, so mode cannot express "the train split, evaluated". The dataset is constructed in
    `test` mode and the split key is swapped only for the collection call, then restored.
    """
    if split not in SPLITS:
        raise ValueError(f"--split must be one of {SPLITS}, got {split!r}")

    cfg = {**config, "test_dataset": dataset_name}
    dataset = NeSyDeFakeDataset(cfg, mode="test")
    if split == "test":
        return dataset

    frames_for_split = config["frame_num"].get(split, config["frame_num"]["test"])
    saved_mode, saved_frames = dataset.mode, dataset.frame_num
    try:
        dataset.mode, dataset.frame_num = split, frames_for_split
        images, labels, _names = dataset.collect_img_and_label_for_one_dataset(dataset_name)
    finally:
        # restored BEFORE any __getitem__ runs: with mode left at 'train' the loader would
        # apply augmentation and the cache would describe augmented reals.
        dataset.mode, dataset.frame_num = saved_mode, saved_frames

    dataset.image_list, dataset.label_list = images, labels
    dataset.data_dict = {"image": images, "label": labels}
    dataset._build_source_video_maps()
    print(f"  {dataset_name}/{split}: {len(images)} frames "
          f"({sum(1 for x in labels if x == 0)} real / {sum(1 for x in labels if x != 0)} fake)")
    return dataset


def path_metadata(path: str) -> tuple[str, str]:
    """(method, video_id) from a frame path.

    FF++ paths carry the manipulation in a fixed position
    (`.../manipulated_sequences/Deepfakes/c23/frames/<video>/<frame>.png`); other datasets do
    not, so this falls back to the directory above the frame. Read as provenance for grouping
    (per-generator rows, leave-one-manipulation-out folds), never as a model input — the E4
    protocol forbids generator identity at inference.
    """
    parts = Path(path).parts
    video_id = parts[-2] if len(parts) >= 2 else "unknown"
    method = "unknown"
    for anchor in ("manipulated_sequences", "original_sequences"):
        if anchor in parts:
            idx = parts.index(anchor)
            if idx + 1 < len(parts):
                method = parts[idx + 1]
            break
    else:
        if len(parts) >= 3:
            method = parts[-3]
    return method, video_id


@torch.no_grad()
def extract(extractor: SpatialFeatureExtractor, loader, device: str,
            max_batches: int = 0) -> tuple[np.ndarray, np.ndarray, list[str]]:
    from tqdm import tqdm

    feats: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    names: list[str] = []
    for i, batch in enumerate(tqdm(loader, desc="  encode")):
        if max_batches and i >= max_batches:
            break
        # no resize here: SpatialFeatureExtractor.forward() already interpolates to
        # required_size when needs_resize, which is the same call the detector makes.
        out = extractor(batch["spatial_frames"].to(device))
        feats.append(out.float().cpu().numpy())
        # binarised here, matching test.py: the cache's `label` is the authenticity label the
        # reference and every audit use; the fine-grained method stays in the metadata column.
        labels.append(torch.where(batch["label"] != 0, 1, 0).cpu().numpy())
        names.extend(batch["name"])
    if not feats:
        raise RuntimeError("no batches produced features")
    return np.concatenate(feats), np.concatenate(labels), names


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                       text=True).strip()
    except Exception:
        return "unknown"


def merge_manifest(out: Path, fresh: dict) -> dict:
    """Accumulate into an existing manifest instead of overwriting it.

    One cache directory is normally filled by several invocations — the FF++ train split for
    Stage I, then the OOD test splits for the audits. Overwriting would drop the record of the
    slices already on disk, so an audit could read features whose encoder provenance the
    manifest no longer describes.

    A fingerprint mismatch is refused rather than merged: two encoder states inside one cache
    directory is precisely what `cache_io.assert_same_encoder` exists to catch downstream, and
    catching it at write time keeps the bad artifact from being created at all.
    """
    path = out / "manifest.json"
    if not path.is_file():
        return fresh
    existing = json.loads(path.read_text())
    if existing.get("fingerprint", {}).get("all") != fresh["fingerprint"]["all"]:
        raise SystemExit(
            f"{path} already holds a cache from a different encoder state "
            f"(existing {existing.get('encoder_mode')} "
            f"{existing.get('fingerprint', {}).get('all', '?')[:12]}… vs new "
            f"{fresh['encoder_mode']} {fresh['fingerprint']['all'][:12]}…). Residuals across "
            f"mixed encoder states are not comparable — use a separate --out per encoder.")
    merged = {**existing, **fresh}
    merged["datasets"] = {**existing.get("datasets", {}), **fresh["datasets"]}
    return merged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, required=True,
                    help="detector yaml; only foundation_models.spatial and the data keys are used")
    ap.add_argument("--encoder", choices=("frozen", "tuned"), required=True)
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="required for --encoder tuned, refused for --encoder frozen")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--split", choices=SPLITS, default="test")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-batches", type=int, default=0,
                    help="0 = whole split; >0 caps for a smoke test (marked in the manifest)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    config = strip_unused_loading(load_config(args.config))
    config["test_batchSize"] = args.batch_size
    if args.workers is not None:
        config["workers"] = args.workers

    print(f"building {args.encoder} encoder on {args.device}")
    extractor = build_encoder(config, args.encoder, args.checkpoint, args.device)
    fingerprint = encoder_fingerprint(extractor)
    print(f"  fingerprint all={fingerprint['all'][:16]}… "
          f"layernorm={fingerprint['layernorm'][:16]}…")

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "encoder_mode": args.encoder,
        # the flag Stage II's assert_reference_config reads: only a 'frozen' cache may be a
        # reference input space.
        "encoder_frozen": args.encoder == "frozen",
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "config": str(args.config),
        # the EFFECTIVE encoder state, read off the built extractor rather than copied from
        # the yaml: --encoder frozen overrides train_layernorms, and a manifest that reported
        # the config's value would document the opposite of what produced the features.
        "backbone": {**config["foundation_models"]["spatial"],
                     "train_layernorms": bool(extractor.train_layernorms),
                     "freeze_backbone": bool(extractor.freeze_backbone)},
        "fingerprint": fingerprint,
        "feature": "spatial_raw",
        "git_commit": git_commit(),
        "datasets": {},
    }
    manifest = merge_manifest(args.out, manifest)

    for name in args.datasets:
        print(f"\n=== {name} / {args.split} ===")
        dataset = load_split(config, name, args.split)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=int(config["workers"]), collate_fn=dataset.collate_fn,
            drop_last=False)
        features, labels, names = extract(extractor, loader, args.device, args.max_batches)

        methods, videos = zip(*(path_metadata(p) for p in names)) if names else ((), ())
        meta = pd.DataFrame({"key": names, "label": labels, "method": methods,
                             "video_id": videos, "source": name, "split": args.split})

        dest = args.out / f"{name}_{args.split}"
        dest.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest / "features.npz", f0=features.astype(np.float32),
                            labels=labels.astype(np.int64))
        # labels also standalone: fit_reference.py's --labels loader takes no key and would
        # pick the first 2-D array in an .npz, i.e. the features, and then "enforce" a
        # reals-only filter against the wrong vector.
        np.save(dest / "labels.npy", labels.astype(np.int64))
        meta.to_csv(dest / "metadata.csv", index=False)
        manifest["datasets"][f"{name}/{args.split}"] = {
            "n": int(features.shape[0]), "dim": int(features.shape[1]),
            "n_real": int((labels == 0).sum()), "n_fake": int((labels == 1).sum()),
            # split and partial live per slice, not once at the top: one cache directory is
            # normally filled by several invocations (train for Stage I, test for the audits),
            # and a single top-level value would describe only the last one.
            "split": args.split, "partial": bool(args.max_batches),
            "path": str(dest),
        }
        print(f"  wrote {dest}/features.npz  {features.shape}")

    manifest["splits"] = sorted({d["split"] for d in manifest["datasets"].values()})
    manifest["partial"] = any(d["partial"] for d in manifest["datasets"].values())
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {args.out}/manifest.json")
    if args.encoder == "frozen" and args.split == "train":
        ffpp = [k for k in manifest["datasets"] if k.startswith("FaceForensics++")]
        if ffpp:
            d = manifest["datasets"][ffpp[0]]
            print("\nStage I is now runnable:\n"
                  f"  python analysis/discern_v2/fit_reference.py \\\n"
                  f"      --features {d['path']}/features.npz --feature-key f0 \\\n"
                  f"      --labels {d['path']}/labels.npy \\\n"
                  f"      --arms C1_random C2_linear C3_ae --objectives cosine mse")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

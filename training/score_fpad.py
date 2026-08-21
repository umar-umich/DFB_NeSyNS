#!/usr/bin/env python3
"""Score a frozen Stage-A student and dump its depth-resolved adaptation profile.

    🔴 UMAR-RUNS (GPU):

    # FF++ val — the in-domain side of the Stage-2 audit
    python training/score_fpad.py --student logs/fpad/studentA_ordinary_seed42 \
        --datasets FaceForensics++ --split val \
        --output logs/fpad/score/ordinary_ffppval --device cuda:1

    # DF40-Dev — the OOD side. `--df40` treats --datasets as per-method names.
    python training/score_fpad.py --student logs/fpad/studentA_ordinary_seed42 \
        --datasets danet_cdf mcnet_cdf tpsm_cdf facevid2vid_cdf --df40 \
        --output logs/fpad/score/ordinary_df40dev --device cuda:1

Writes one parquet row per frame with the full profile `D(x)` (one column per configured layer),
the CLS diagnostic, patch-map summaries and the direct head's probability. Stage 2 reads it; no
stage recomputes a forward pass it could have read.

Provenance of the OOD real half is recorded per frame
-----------------------------------------------------
DF40 borrows its authentic halves from other corpora: a `*_cdf` method's reals are Celeb-DF-v2
frames, a `*_ff` method's reals are FF++ frames. The Stage-2 gate needs that distinction, because
FF++-sourced reals overlap FF++-only training and would understate domain separability. So
`real_source` is derived from each frame's resolved path and written out, rather than inferred
later from the method name — the same method can supply both halves and the manifest is the only
thing that knows which frame came from where.
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(REPO / "analysis" / "discern_v2"))

from networks.fpad import DirectEvidenceHead, FPADTeacherStudent  # noqa: E402
from train_fpad import FpadViews  # noqa: E402
from train_v1 import load_split, prepare_dataset_config, video_of  # noqa: E402


def real_source_of(path: str) -> str:
    """Which corpus this frame's pixels actually came from, read off the resolved path."""
    p = str(path)
    for marker, name in (("/Celeb-DF-v2/", "Celeb-DF-v2"), ("/Celeb-DF-v3/", "Celeb-DF-v3"),
                         ("/FaceForensics++/", "FaceForensics++"), ("/DFDC", "DFDC"),
                         ("/UADFV/", "UADFV"), ("/Deepfake-Eval-2024/", "Deepfake-Eval-2024")):
        if marker in p:
            return name
    if "/df40/test/" in p or "/df40/real/" in p:
        return "DF40-generated"
    return "unknown"


@torch.no_grad()
def score_dataset(model, direct, loader, views, device: str, name: str,
                  max_batches: int = 0) -> pd.DataFrame:
    from tqdm import tqdm

    model.eval()
    direct.eval()
    rows = []
    layers_1x = [i + 1 for i in model.layers]
    for i, batch in enumerate(tqdm(loader, desc=f"  {name}", leave=False)):
        if max_batches and i >= max_batches:
            break
        pixels = views(batch, device)
        out = model(pixels)
        head = direct(out["h_student"])
        record = {
            "dataset": name,
            "key": [str(k) for k in batch["name"]],
            "video_id": [video_of(p) for p in batch["name"]],
            "label": torch.where(batch["label"] != 0, 1, 0).numpy(),
            "real_source": [real_source_of(p) for p in batch["name"]],
            "p_direct": head["prob"].cpu().numpy(),
            "u_direct": head["u"].cpu().numpy(),
        }
        D = out["D"].cpu().numpy()
        D_cls = out["D_cls"].cpu().numpy()
        for j, layer in enumerate(layers_1x):
            record[f"d_l{layer}"] = D[:, j]
            record[f"d_cls_l{layer}"] = D_cls[:, j]
        record["d_mean"] = D.mean(axis=1)
        record["d_early"] = D[:, : max(1, len(layers_1x) // 2)].mean(axis=1)
        record["d_late"] = D[:, len(layers_1x) // 2:].mean(axis=1)
        pm = out["patch_map"].flatten(1).cpu().numpy()
        record["patch_max"] = pm.max(axis=1)
        record["patch_std"] = pm.std(axis=1)
        # concentration: how much of the total adaptation sits in the top 10% of patches. A
        # localized manipulation should concentrate; a global domain shift should not.
        k = max(1, pm.shape[1] // 10)
        top = np.sort(pm, axis=1)[:, -k:].sum(axis=1)
        record["patch_top10_share"] = top / np.clip(pm.sum(axis=1), 1e-12, None)
        rows.append(pd.DataFrame(record))
    if not rows:
        raise RuntimeError(f"{name}: no batches produced predictions")
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--student", type=Path, required=True, help="a Stage-A run directory")
    ap.add_argument("--epoch", type=int, default=None,
                    help="which Stage-A epoch to score (default: the latest on disk)")
    ap.add_argument("--config", type=Path, default=REPO / "training/config/fpad/FPAD_CONFIG.yaml")
    ap.add_argument("--detector-config", type=Path,
                    default=REPO / "training/config/detector/nesy_defake_d1_v.yaml")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--df40", action="store_true",
                    help="treat --datasets as DF40 per-method names")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    checkpoints = sorted(args.student.glob("epoch_*.pth"))
    if not checkpoints:
        raise SystemExit(f"no Stage-A checkpoints in {args.student}")
    chosen = (args.student / f"epoch_{args.epoch:03d}.pth") if args.epoch is not None \
        else checkpoints[-1]
    if not chosen.is_file():
        raise SystemExit(f"{chosen} not found; available: {[c.name for c in checkpoints]}")
    blob = torch.load(str(chosen), map_location="cpu", weights_only=False)
    student_meta = blob["run_meta"]
    print(f"student {chosen} — arm `{student_meta.get('arm')}`, "
          f"lambda_preserve {student_meta.get('lambda_preserve')}, epoch {blob['epoch']}")

    cfg = yaml.safe_load(args.config.read_text())
    model = FPADTeacherStudent(
        checkpoint=cfg["encoder"].get("checkpoint") or (REPO / "weights/FS-VFM/checkpoint-599.pth"),
        layers=tuple(student_meta["layers"]), lora=student_meta["lora"],
        img_size=int(cfg["encoder"].get("img_size", 224))).to(args.device)
    model.load_state_dict(blob["lora"], strict=False)
    direct = DirectEvidenceHead(feature_dim=model.embed_dim,
                                hidden_dim=int(cfg["heads"]["hidden_dim"])).to(args.device)
    direct.load_state_dict(blob["direct_head"])
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    data_cfg = prepare_dataset_config(args.detector_config, args.batch_size, args.workers)
    views = FpadViews(augment=False, matched=False)

    df40 = None
    resolution = {}
    if args.df40:
        from dataset import df40_paths as df40
        json_dir = df40.DF40_JSON_DIR
        data_cfg = {**data_cfg, "dataset_json_folder": str(json_dir),
                    "label_dict": {**data_cfg.get("label_dict", {}),
                                   **df40.label_dict_for(args.datasets, json_dir)}}
        print(f"DF40 mode: {json_dir}")

    args.output.mkdir(parents=True, exist_ok=True)
    frames = []
    for name in args.datasets:
        dataset = load_split(data_cfg, name, args.split)
        if df40 is not None:
            resolution[name] = df40.remap_dataset(dataset)
            print(f"  {name}: resolved {resolution[name]['resolution_rate']:.3f}")
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=int(data_cfg["workers"]), collate_fn=dataset.collate_fn)
        frames.append(score_dataset(model, direct, loader, views, args.device, name,
                                    args.max_batches))
        got = frames[-1]
        print(f"  {name}: {len(got)} frames · real_source "
              f"{got.groupby('real_source').size().to_dict()} · "
              f"mean d {got['d_mean'].mean():.5f}")

    table = pd.concat(frames, ignore_index=True)
    dest = args.output / f"profile_epoch_{blob['epoch']:03d}.parquet"
    table.to_parquet(dest, index=False)
    (args.output / "score_meta.json").write_text(json.dumps({
        "student_run": str(args.student), "checkpoint": str(chosen),
        "student_meta": student_meta, "epoch": int(blob["epoch"]),
        "datasets": args.datasets, "split": args.split, "df40": bool(args.df40),
        "df40_resolution": resolution or None, "n_frames": int(len(table)),
        "layers_one_indexed": [i + 1 for i in model.layers],
    }, indent=2, default=str))
    print(f"\nwrote {dest} ({len(table)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

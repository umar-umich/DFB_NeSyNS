#!/usr/bin/env python3
"""Evaluate a Stage-B checkpoint on FF++ test and the OOD suite (§20, §21).

    🔴 UMAR-RUNS.

    python training/eval_v1.py \
        --checkpoint logs/v1/stage_b_seed42/epoch_001.pth \
        --datasets FaceForensics++ Celeb-DF-v2 Celeb-DF-v3 DFDC DFDCP \
        --output logs/v1/eval/epoch_001 --device cuda:N

**This is not checkpoint selection.** §11 selects on VAL_select only and says epoch-wise OOD
scores are saved "for later analysis only". Nothing this script writes may pick an epoch, a
hyperparameter, or a threshold; the output filename carries the epoch so the distinction survives
into the results directory.

What "fused" means here, precisely
---------------------------------
At Stage B there is no applicability gate (Stage D) and no defer policy (Stage E), so fusion runs
**ungated, q = 1** — the same configuration Stage B trains under. Every table this writes is
therefore *plain DS over three ungated experts*, which is the baseline the applicability layer is
later supposed to beat, not the V1 system. Reported explicitly rather than left for the reader to
infer, because "DISCERN-v2 V1" in a paper table means the gated, calibrated system.

Per-sample instrumentation (§20)
--------------------------------
Writes one row per frame with everything §20 lists: per-branch evidence / p / u, `e_direct` and
its prediction, `branch_valid_*`, reference and process diagnostics, the DS conflict and its
degenerate flag, fused p / u, and V / C / A. That file is the input to the §20 domain-detector
distribution audit and to the post-hoc contribution diagnostics, so those never need a second
forward pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(REPO / "analysis" / "discern_v2"))

from networks.discern_v2.discern_v1_model import build_v1_model  # noqa: E402
from networks.discern_v2.dirichlet import to_dirichlet  # noqa: E402
from train_v1 import ViewMaker, prepare_dataset_config, video_of  # noqa: E402

BRANCHES = ("sem", "ref", "proc")


def load_model(checkpoint: Path, device: str):
    """Rebuild the model from the checkpoint's own config snapshot, then load its weights.

    The config travels inside the checkpoint, so an evaluation cannot silently use a different
    reference artifact or process statistics than the run was trained with.
    """
    blob = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    cfg = blob["config"]
    model = build_v1_model({
        "backbone": cfg["semantic"]["backbone"],
        "semantic_feature_dim": cfg["semantic"]["feature_dim"],
        "enable_lora": cfg["semantic"]["enable_lora"],
        "reference": {**cfg["reference"],
                      "artifact_path": str(REPO / cfg["reference"]["artifact_path"]),
                      "fsvfm_checkpoint": str(REPO / cfg["reference"]["fsvfm_checkpoint"])},
        "process": {**cfg["process"],
                    "stats_artifact": str(REPO / cfg["process"]["stats_artifact"])},
    }, stage="B")
    missing, unexpected = model.load_state_dict(blob["model"], strict=False)
    if missing:
        raise SystemExit(
            f"{len(missing)} parameters missing from {checkpoint} (first: {list(missing)[:5]}). "
            f"Refusing to evaluate a partially-restored model.")
    if unexpected:
        print(f"  note: {len(unexpected)} unexpected keys ignored (first: {list(unexpected)[:3]})")
    model.to(device).eval()
    model.assert_frozen_protocol()
    return model, cfg, blob.get("epoch")


@torch.no_grad()
def evaluate_dataset(model, loader, views, device: str, dataset: str,
                     max_batches: int = 0) -> pd.DataFrame:
    from tqdm import tqdm

    rows: list[dict] = []
    for i, batch in enumerate(tqdm(loader, desc=f"  {dataset}", leave=False)):
        if max_batches and i >= max_batches:
            break
        labels = torch.where(batch["label"] != 0, 1, 0)
        out = model(views(batch, device))
        branches = out["branches"]

        record = {
            "dataset": dataset,
            "key": batch["name"],
            "video_id": [video_of(p) for p in batch["name"]],
            "label": labels.numpy(),
            "prob_fused": out["prob"].cpu().numpy(),
            "u_fused": out["fused"].vacuity.squeeze(1).cpu().numpy(),
            "V": out["V"].cpu().numpy(),
            "C": out["C"].cpu().numpy(),
            "A": out["A"].cpu().numpy(),
            "ds_conflict": out["ds_conflict"].cpu().numpy(),
            "ds_degenerate": out["ds_degenerate"].cpu().numpy(),
        }
        for name in BRANCHES:
            if name not in branches:
                continue
            state = to_dirichlet(branches[name]["evidence"])
            record[f"p_{name}"] = state.fake_prob().cpu().numpy()
            record[f"u_{name}"] = state.vacuity.cpu().numpy()
            record[f"e_{name}_real"] = state.evidence[:, 0].cpu().numpy()
            record[f"e_{name}_fake"] = state.evidence[:, 1].cpu().numpy()
            record[f"valid_{name}"] = branches[name]["valid"].cpu().numpy()
        if "direct" in branches:
            direct = to_dirichlet(branches["direct"]["evidence"])
            record["p_direct"] = direct.fake_prob().cpu().numpy()
            record["u_direct"] = direct.vacuity.cpu().numpy()
        if "proc" in branches and "raw_stats" in branches["proc"]["diagnostics"]:
            stats = branches["proc"]["diagnostics"]["raw_stats"].cpu().numpy()
            for j, sname in enumerate(model.process.stat_names):
                record[f"proc_{sname}"] = stats[:, j]
        if "ref" in branches:
            diag = branches["ref"]["diagnostics"]
            for key, col in (("reference_residual_norm", "ref_residual_norm"),
                             ("reference_angle", "ref_angle")):
                if key in diag:
                    record[col] = diag[key].cpu().numpy()
        rows.append(pd.DataFrame(record))
    if not rows:
        raise RuntimeError(f"{dataset}: no batches produced predictions")
    return pd.concat(rows, ignore_index=True)


def fix_degenerate_video_ids(df: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, str | None]:
    """Fall back to per-file ids where the parent directory is not a video.

    `video_of()` takes the parent directory, which is the video for every frame-extracted corpus.
    DF40's whole-image generators are stored flat as `<method>/<half>/<half>/<n>.jpg`, so the
    parent directory IS the class: every real image becomes one "video" called `real` and every
    fake one called `fake`. Video AUROC over two points whose scores are the class means is then
    exactly 1.0 or 0.0 — a spectacular-looking number that measures nothing at all, which is
    worse than a missing one because it looks like a result.

    Detected from the data rather than from a method whitelist, so any flat corpus is caught:
    a handful of distinct ids covering many frames is not a video structure.
    """
    n_ids = df["video_id"].nunique()
    if n_ids > 4 or len(df) < 4 * max(n_ids, 1):
        return df, None
    # The FULL path minus its suffix, not the bare file stem. DF40's flat methods store the two
    # classes as .../<half>/<half>/<n>.jpg, so `real/real/1332.jpg` and `fake/fake/1332.jpg` share
    # the stem `1332`: grouping on it pairs each real with its fake, label=max() marks every group
    # fake, and the video-level array becomes single-class -> AUROC nan for every branch while the
    # frame AUROC still looks fine. The path is unique per file by construction.
    df = df.assign(video_id=[str(Path(k).with_suffix("")) for k in df["key"]])
    note = (f"{dataset}: only {n_ids} distinct parent directories over {len(df)} frames — the "
            f"layout is flat, so each file is its own sample and video aggregation now uses the "
            f"full file path. Video AUROC therefore EQUALS frame AUROC here: there is no video "
            f"level in this corpus. Grouping on the parent would have given a 2-point AUROC of "
            f"1.0 or 0.0, and grouping on the bare stem gives nan (real and fake share stems).")
    return df, note


def metrics_for(df: pd.DataFrame, score_col: str) -> dict:
    """Frame and video AUROC. Video aggregation is mean frame p(fake), per §21."""
    from sklearn.metrics import roc_auc_score

    def auc(y, s):
        return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")

    video = df.groupby(["dataset", "video_id"], as_index=False).agg(
        score=(score_col, "mean"), label=("label", "max"))
    # DF40's whole-image methods list the same authentic image under several entries, so the
    # frame-level number double-counts those reals while the grouped number counts each file
    # once. Reported so the gap between the two is explained rather than puzzling.
    duplicates = int(len(df) - df["key"].nunique())
    conflicting = int((df.groupby("key")["label"].nunique() > 1).sum())
    return {
        "duplicate_keys": duplicates,
        "keys_with_conflicting_labels": conflicting,
        "frame_auroc": auc(df["label"].to_numpy(), df[score_col].to_numpy()),
        "video_auroc": auc(video["label"].to_numpy(), video["score"].to_numpy()),
        "n_frames": int(len(df)),
        "n_videos": int(len(video)),
        "n_real_videos": int((video["label"] == 0).sum()),
        "n_fake_videos": int((video["label"] == 1).sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--detector-config", type=Path,
                    default=REPO / "training/config/detector/nesy_defake_d1_v.yaml")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="pick the GPU explicitly; this script never chooses one for you")
    ap.add_argument("--df40", action="store_true",
                    help="treat --datasets as DF40 per-method names: use DF40's json folder, "
                         "inject its per-method label_dict, and remap its relative frame paths")
    ap.add_argument("--dataset-json-folder", type=Path, default=None,
                    help="override the json folder (DF40 keeps its own)")
    ap.add_argument("--overwrite", action="store_true",
                    help="discard existing results for this epoch in --output instead of refusing")
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    model, cfg, epoch = load_model(args.checkpoint, args.device)
    print(f"loaded epoch {epoch} from {args.checkpoint}")

    # Outputs are named by epoch, so a second invocation for the SAME epoch into the SAME
    # directory silently overwrites the first run's results and per-sample export — which is
    # exactly what a follow-up run (say, adding DF40) looks like. Refused rather than clobbered;
    # domain_audit.py accepts several parquets, so separate directories are the natural pattern.
    existing = [p for p in (args.output / f"results_epoch_{epoch}.json",
                            args.output / f"per_sample_epoch_{epoch}.parquet") if p.exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            f"{args.output} already holds results for epoch {epoch} "
            f"({', '.join(p.name for p in existing)}). Use a different --output (e.g. "
            f"{args.output}_df40) and pass both parquets to domain_audit.py, or --overwrite to "
            f"discard what is there.")

    data_cfg = prepare_dataset_config(args.detector_config, args.batch_size, args.workers)
    clip_norm = data_cfg["foundation_models"]["spatial"]["normalization"]
    views = ViewMaker(clip_norm["mean"], clip_norm["std"], args.device, augment=False)

    df40 = None
    if args.df40:
        from dataset import df40_paths as df40

        json_dir = args.dataset_json_folder or df40.DF40_JSON_DIR
        data_cfg["dataset_json_folder"] = str(json_dir)
        # DF40 labels are per method and absent from the shared config, which RAISES on unknown
        # keys; injected for this run only so test_config.yaml stays untouched.
        data_cfg["label_dict"] = {**data_cfg.get("label_dict", {}),
                                  **df40.label_dict_for(args.datasets, json_dir)}
        print(f"DF40 mode: json folder {json_dir}, "
              f"{len(data_cfg['label_dict'])} label keys after injection")
    elif args.dataset_json_folder:
        data_cfg["dataset_json_folder"] = str(args.dataset_json_folder)

    from dataset.nesy_defake_dataset import NeSyDeFakeDataset

    results: dict[str, dict] = {}
    resolution: dict[str, dict] = {}
    aggregation_notes: dict[str, str] = {}
    frames: list[pd.DataFrame] = []
    for name in args.datasets:
        print(f"\n=== {name} ===")
        cfg_ds = {**data_cfg, "test_dataset": name}
        dataset = NeSyDeFakeDataset(cfg_ds, mode="test")
        if df40 is not None:
            report = df40.remap_dataset(dataset)
            resolution[name] = {**report, "family": df40.family_of(name)}
            print(f"  path remap: kept {report['kept']}, dropped {report['dropped']} "
                  f"(rate {report['resolution_rate']:.3f}, family {resolution[name]['family']})")
            if report["resolution_rate"] < 0.99:
                print(f"  WARNING: {name} is missing "
                      f"{(1 - report['resolution_rate']) * 100:.1f}% of its frames — the AUROC "
                      f"below is computed over a partial method")
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=int(data_cfg["workers"]), collate_fn=dataset.collate_fn)
        df = evaluate_dataset(model, loader, views, args.device, name, args.max_batches)
        df, note = fix_degenerate_video_ids(df, name)
        if note:
            print(f"  NOTE {note}")
            aggregation_notes[name] = note
        frames.append(df)

        per_source = {"fused_ungated": metrics_for(df, "prob_fused")}
        for branch in BRANCHES:
            if f"p_{branch}" in df:
                per_source[branch] = metrics_for(df, f"p_{branch}")
        if "p_direct" in df:
            per_source["direct_probe_control"] = metrics_for(df, "p_direct")
        per_source["reliability_means"] = {k: float(df[k].mean()) for k in ("V", "C", "A")}
        per_source["ds_degenerate_rate"] = float(df["ds_degenerate"].mean())
        per_source["branch_validity"] = {
            f"valid_{b}": float(df[f"valid_{b}"].mean()) for b in BRANCHES
            if f"valid_{b}" in df}
        results[name] = per_source

        f = per_source["fused_ungated"]
        print(f"  fused (ungated)  video AUROC {f['video_auroc']:.4f}  "
              f"frame {f['frame_auroc']:.4f}  ({f['n_videos']} videos)")
        for key in [*BRANCHES, "direct_probe_control"]:
            if key in per_source:
                print(f"  {key:22s} video AUROC {per_source[key]['video_auroc']:.4f}")

    all_frames = pd.concat(frames, ignore_index=True)
    all_frames.to_parquet(args.output / f"per_sample_epoch_{epoch}.parquet", index=False)

    payload = {
        "checkpoint": str(args.checkpoint),
        "epoch": epoch,
        "IMPORTANT": ("Stage-B checkpoint scored with UNGATED fusion (q=1): no applicability gate "
                      "(Stage D) and no defer policy (Stage E) exist yet. These are plain-DS "
                      "numbers over three ungated experts — the baseline the applicability layer "
                      "must beat, NOT the V1 system."),
        "selection_note": ("§11: OOD scores are for later analysis only and must not select an "
                           "epoch, a hyperparameter, or a threshold."),
        "video_aggregation": "mean frame p(fake) (§21)",
        "df40_path_resolution": resolution or None,
        "video_aggregation_notes": aggregation_notes or None,
        "results": results,
    }
    (args.output / f"results_epoch_{epoch}.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.output}/results_epoch_{epoch}.json and the per-sample parquet")
    print("Reminder: this cannot select the checkpoint — §11 keeps that on VAL_select.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

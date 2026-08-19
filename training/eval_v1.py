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


def metrics_for(df: pd.DataFrame, score_col: str) -> dict:
    """Frame and video AUROC. Video aggregation is mean frame p(fake), per §21."""
    from sklearn.metrics import roc_auc_score

    def auc(y, s):
        return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")

    video = df.groupby(["dataset", "video_id"], as_index=False).agg(
        score=(score_col, "mean"), label=("label", "max"))
    return {
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
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    model, cfg, epoch = load_model(args.checkpoint, args.device)
    print(f"loaded epoch {epoch} from {args.checkpoint}")

    data_cfg = prepare_dataset_config(args.detector_config, args.batch_size, args.workers)
    clip_norm = data_cfg["foundation_models"]["spatial"]["normalization"]
    views = ViewMaker(clip_norm["mean"], clip_norm["std"], args.device, augment=False)

    from dataset.nesy_defake_dataset import NeSyDeFakeDataset

    results: dict[str, dict] = {}
    frames: list[pd.DataFrame] = []
    for name in args.datasets:
        print(f"\n=== {name} ===")
        cfg_ds = {**data_cfg, "test_dataset": name}
        dataset = NeSyDeFakeDataset(cfg_ds, mode="test")
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=int(data_cfg["workers"]), collate_fn=dataset.collate_fn)
        df = evaluate_dataset(model, loader, views, args.device, name, args.max_batches)
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
        "results": results,
    }
    (args.output / f"results_epoch_{epoch}.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.output}/results_epoch_{epoch}.json and the per-sample parquet")
    print("Reminder: this cannot select the checkpoint — §11 keeps that on VAL_select.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

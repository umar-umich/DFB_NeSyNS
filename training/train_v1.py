#!/usr/bin/env python3
"""Stage B — train the V1 experts (build spec §9 B, §10, §11, §19).

    🔴 UMAR-RUNS. This script launches nothing by itself; it is the entry point Umar calls.

    python training/train_v1.py \
        --config training/config/discern_v2/V1_CONFIG.yaml \
        --detector-config training/config/detector/nesy_defake_d1_v.yaml \
        --output logs/v1/stage_b_run1

What trains: CLIP's LoRA adapters, `H_sem`, `H_ref`, `H_proc`, and the §4.2 control head.
What does not: FS-VFM, `P_R`, the SDXL VAE — asserted at start-up via
`model.assert_frozen_protocol()`, not assumed.

One augmented image, three normalizations (§6)
----------------------------------------------
The three branches must describe the *same observation*. The dataset is therefore run with its
own augmentation OFF, augmentation is applied once per batch to the raw [0,1] tensor using the
ported DiCoME pipeline, and the three views are derived from that single augmented image:

    aug = DiCoME_transform(raw_frames)          # one image
      |-- CLIP normalization  -> spatial_frames  (Branch A)
      |-- as-is [0,1]         -> fsvfm_frames    (Branch B; FS-VFM normalizes internally)
      `-- as-is [0,1]         -> process_frames  (Branch C; the VAE normalizes internally)

Augmenting after the dataset produced independently-normalized tensors would give each branch a
*differently* augmented image, which is the clean-feature/augmented-image mismatch
`process_residual.py` warns about at length — invisible in training, and it shows up only as an
unexplained gap between branch agreement in training and at inference.

The loss, and a fork the spec leaves open
-----------------------------------------
§9 B says "per-branch auxiliary EDL losses so each branch is individually usable" and does not
mention a fused loss in Stage B. DiCoME, by contrast, supervises the *fused* evidence
(`src/model/dicome_module.py:114-140`). Both are defensible, and they are not equivalent:

* **per-branch only (default here)** — the branches are trained independently and the reasoning
  layer is *applied*, never trained. "Plain DS vs DS + applicability" then compares two ways of
  combining fixed experts, which is the cleaner test of the contribution.
* **adding a fused term** (`training.fused_loss_weight > 0`) couples the branches through DS
  during Stage B, so by Stage D the gate is choosing among experts that already co-adapted to
  each other's presence.

Default is the spec's literal reading; the alternative is one config value away and is flagged
🟡 ASK-UMAR rather than chosen silently.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))
# The FF++ split loader lives with the caching tool because that is where it was validated;
# imported rather than copied so the two cannot drift into loading different frames.
sys.path.insert(0, str(REPO / "analysis" / "discern_v2"))

from dataset.nesy_defake_dataset import NeSyDeFakeDataset  # noqa: E402
from networks.discern_v2.discern_v1_model import build_v1_model  # noqa: E402
from networks.discern_v2.semantic_branch import (  # noqa: E402
    dicome_param_groups, dicome_train_transform)


# ---------------------------------------------------------------------------
# EDL loss — ported from DiCoME src/losses/evidential_loss.py
# ---------------------------------------------------------------------------


def _kl_dirichlet_to_uniform(alpha: torch.Tensor, num_classes: int) -> torch.Tensor:
    """KL(Dir(alpha) || Dir(1)). Ported from DiCoME `_kl_dirichlet_to_uniform`."""
    beta = torch.ones_like(alpha)
    sum_alpha = alpha.sum(dim=1, keepdim=True)
    sum_beta = beta.sum(dim=1, keepdim=True)
    lnA = torch.lgamma(sum_alpha) - torch.lgamma(alpha).sum(dim=1, keepdim=True)
    lnB = torch.lgamma(beta).sum(dim=1, keepdim=True) - torch.lgamma(sum_beta)
    digamma_sum_alpha = torch.digamma(sum_alpha)
    digamma_alpha = torch.digamma(alpha)
    return lnA + lnB + ((alpha - beta) * (digamma_alpha - digamma_sum_alpha)).sum(
        dim=1, keepdim=True)


def edl_loss(evidence: torch.Tensor, labels: torch.Tensor, num_classes: int = 2,
             current_epoch: int = 0, total_epochs: int = 20,
             reduction: str = "mean") -> torch.Tensor:
    """DiCoME's evidential loss, ported, with per-sample reduction available.

    Identical formulation to `evidential_loss_dicome` — digamma classification term plus an
    annealed KL that discourages unsupported evidence for non-target classes. The only change is
    `reduction`: §9 B requires masking the loss on samples a branch could not process, which a
    batch-mean-only function cannot express.
    """
    alpha = evidence + 1
    alpha_sum = alpha.sum(dim=1, keepdim=True)
    one_hot = F.one_hot(labels, num_classes=num_classes).float()

    classification = (one_hot * (torch.digamma(alpha_sum) - torch.digamma(alpha))).sum(
        dim=1, keepdim=True)
    anneal = 1.0 if total_epochs <= 0 else min(1.0, current_epoch / total_epochs)
    non_target_alpha = (alpha - 1) * (1 - one_hot) + 1
    regularization = anneal * _kl_dirichlet_to_uniform(non_target_alpha, num_classes)

    per_sample = (classification + regularization).squeeze(1)
    return per_sample.mean() if reduction == "mean" else per_sample


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def load_split(config: dict, dataset_name: str, split: str) -> NeSyDeFakeDataset:
    import cache_encoder_features as CE

    return CE.load_split(config, dataset_name, split)


def prepare_dataset_config(detector_config: Path, batch_size: int, workers: int | None) -> dict:
    import cache_encoder_features as CE

    config = CE.strip_unused_loading(CE.load_config(detector_config))
    config["test_batchSize"] = batch_size
    config["train_batchSize"] = batch_size
    if workers is not None:
        config["workers"] = workers
    # augmentation is applied in the training loop instead, on one image for all three branches
    config["use_data_augmentation"] = False
    return config


class ViewMaker:
    """One augmented image -> the three branch-specific views (§6)."""

    def __init__(self, clip_mean, clip_std, device: str, augment: bool):
        self.mean = torch.tensor(clip_mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(clip_std, device=device).view(1, 3, 1, 1)
        self.transform = dicome_train_transform() if augment else None

    def __call__(self, batch: dict, device: str) -> dict:
        raw = batch["raw_frames"].to(device)
        if raw.min() < -0.01:
            raise ValueError(
                "raw_frames are not in [0, 1]; the three views must be derived from unnormalized "
                "pixels or each branch receives another branch's normalization")
        if self.transform is not None:
            # Per SAMPLE, not per batch. torchvision transforms applied to a batched tensor draw
            # their random parameters ONCE and apply the same flip/affine/blur/jitter to every
            # image in the batch, which is far less augmentation than DiCoME's per-image pipeline
            # and would make the ported recipe weaker than the one it reproduces. The cost is one
            # transform call per image, which is noise next to a ViT-L forward pass.
            raw = torch.stack([self.transform(img) for img in raw])
        return {
            "spatial_frames": (raw - self.mean) / self.std,   # Branch A: CLIP statistics
            "fsvfm_frames": raw,                              # Branch B normalizes internally
            "process_frames": raw,                            # Branch C normalizes internally
        }


def video_of(path: str) -> str:
    parts = Path(path).parts
    return parts[-2] if len(parts) >= 2 else path


def video_level(probs: np.ndarray, labels: np.ndarray, videos: list[str]
                ) -> tuple[np.ndarray, np.ndarray]:
    """Mean frame probability per video — §21's aggregation rule."""
    order: dict[str, list[int]] = {}
    for i, v in enumerate(videos):
        order.setdefault(v, []).append(i)
    keys = sorted(order)
    return (np.array([probs[order[k]].mean() for k in keys]),
            np.array([labels[order[k]][0] for k in keys]))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs > lo) & (probs <= hi)
        if not m.any():
            continue
        ece += m.mean() * abs((probs[m] > 0.5).astype(float).mean() - labels[m].mean())
    return float(ece)


# ---------------------------------------------------------------------------
# train / validate
# ---------------------------------------------------------------------------


def run_epoch(model, loader, views, device, optimizer, scheduler, epoch, cfg, train: bool):
    model.train(train)
    totals: dict[str, float] = {}
    n_batches = 0
    probs, labels_all, video_names = [], [], []
    branch_w = float(cfg["training"].get("branch_loss_weight", 1.0))
    fused_w = float(cfg["training"].get("fused_loss_weight", 0.0))
    direct_w = float(cfg["training"].get("direct_probe_loss_weight", 1.0))
    total_epochs = int(cfg["training"].get("max_epochs", 20))

    from tqdm import tqdm
    for batch in tqdm(loader, desc=("train" if train else "val"), leave=False):
        labels = torch.where(batch["label"] != 0, 1, 0).to(device)
        inputs = views(batch, device)

        with torch.set_grad_enabled(train):
            out = model(inputs)
            branches = out["branches"]

            loss = torch.zeros((), device=device)
            parts: dict[str, torch.Tensor] = {}
            for name, b in branches.items():
                valid = b["valid"]
                if not valid.any():
                    continue
                per_sample = edl_loss(b["evidence"][valid], labels[valid],
                                      current_epoch=epoch, total_epochs=total_epochs,
                                      reduction="none")
                term = per_sample.mean()
                parts[f"loss_{name}"] = term
                # the §4.2 control is optimised so the comparison is fair, but it is reported
                # separately and never enters fusion
                loss = loss + (direct_w if name == "direct" else branch_w) * term

            if fused_w > 0:
                fused_term = edl_loss(out["evidence"], labels, current_epoch=epoch,
                                      total_epochs=total_epochs)
                parts["loss_fused"] = fused_term
                loss = loss + fused_w * fused_term

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        n_batches += 1
        totals["loss"] = totals.get("loss", 0.0) + float(loss.detach())
        for k, v in parts.items():
            totals[k] = totals.get(k, 0.0) + float(v.detach())
        if not train:
            probs.append(out["prob"].detach().cpu().numpy())
            labels_all.append(labels.cpu().numpy())
            video_names.extend(video_of(p) for p in batch["name"])

    metrics = {k: v / max(1, n_batches) for k, v in totals.items()}
    if not train and probs:
        from sklearn.metrics import roc_auc_score

        p = np.concatenate(probs)
        y = np.concatenate(labels_all)
        vp, vy = video_level(p, y, video_names)
        metrics["frame_auroc"] = (float(roc_auc_score(y, p)) if len(np.unique(y)) > 1
                                  else float("nan"))
        metrics["video_auroc"] = (float(roc_auc_score(vy, vp)) if len(np.unique(vy)) > 1
                                  else float("nan"))
        metrics["ece"] = expected_calibration_error(p, y)
        metrics["n_videos"] = int(len(vy))
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path,
                    default=REPO / "training/config/discern_v2/V1_CONFIG.yaml")
    ap.add_argument("--detector-config", type=Path,
                    default=REPO / "training/config/detector/nesy_defake_d1_v.yaml")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-batches", type=int, default=0, help="0 = full epoch; >0 smoke test")
    ap.add_argument("--overwrite", action="store_true",
                    help="discard an existing run in --output instead of refusing")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.epochs is not None:
        cfg["training"]["max_epochs"] = args.epochs
    epochs = int(cfg["training"]["max_epochs"])
    seed = int(cfg["meta"].get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    args.output.mkdir(parents=True, exist_ok=True)
    # Refuse to append to a previous run's log. Silently continuing one would interleave two
    # runs' epochs in metrics.jsonl and leave checkpoints from different configurations sharing
    # one directory — §11 then selects a checkpoint whose provenance nobody can reconstruct.
    existing = sorted(args.output.glob("epoch_*.pth"))
    if (args.output / "metrics.jsonl").exists() or existing:
        if not args.overwrite:
            raise SystemExit(
                f"{args.output} already holds a run ({len(existing)} checkpoints). Pass a new "
                f"--output, or --overwrite to discard it. Appending would mix two runs' metrics "
                f"and checkpoints under one provenance.")
        (args.output / "metrics.jsonl").unlink(missing_ok=True)
        for old_ckpt in existing:
            old_ckpt.unlink()
        print(f"  --overwrite: discarded {len(existing)} checkpoints and the previous metrics log")
    (args.output / "config_snapshot.yaml").write_text(yaml.dump(cfg, sort_keys=False))

    data_cfg = prepare_dataset_config(args.detector_config, args.batch_size, args.workers)
    clip_norm = data_cfg["foundation_models"]["spatial"]["normalization"]

    print(f"building the V1 model on {args.device}")
    model = build_v1_model({
        "backbone": cfg["semantic"]["backbone"],
        "semantic_feature_dim": cfg["semantic"]["feature_dim"],
        "enable_lora": cfg["semantic"]["enable_lora"],
        "reference": {**cfg["reference"],
                      "artifact_path": str(REPO / cfg["reference"]["artifact_path"]),
                      "fsvfm_checkpoint": str(REPO / cfg["reference"]["fsvfm_checkpoint"])},
        "process": {**cfg["process"],
                    "stats_artifact": str(REPO / cfg["process"]["stats_artifact"])},
    }, stage="B").to(args.device)

    # §19: the mechanism check runs in the real build path, before a single step is taken
    model.assert_frozen_protocol()
    trainable = model.trainable_parameters()
    print(f"  trainable: {trainable} (total {sum(trainable.values()):,})")

    train_set = load_split(data_cfg, "FaceForensics++", "train")
    val_set = load_split(data_cfg, "FaceForensics++", "val")
    val_select = _restrict_to_val_select(val_set, cfg)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=int(data_cfg["workers"]), collate_fn=train_set.collate_fn, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_select, batch_size=args.batch_size, shuffle=False,
        num_workers=int(data_cfg["workers"]), collate_fn=val_select.collate_fn)

    optimizer = torch.optim.AdamW(
        dicome_param_groups(model, weight_decay=float(cfg["training"]["weight_decay"])),
        lr=float(cfg["training"]["lr"]), betas=tuple(cfg["training"]["betas"]))
    steps = max(1, len(train_loader)) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=float(cfg["training"]["scheduler"]["min_lr"]))

    train_views = ViewMaker(clip_norm["mean"], clip_norm["std"], args.device, augment=True)
    eval_views = ViewMaker(clip_norm["mean"], clip_norm["std"], args.device, augment=False)

    log_path = args.output / "metrics.jsonl"
    print(f"training {epochs} epochs; every epoch is checkpointed (§11 forbids a fixed window)")
    for epoch in range(epochs):
        started = time.time()
        train_metrics = run_epoch(model, _capped(train_loader, args.max_batches), train_views,
                                  args.device, optimizer, scheduler, epoch, cfg, train=True)
        val_metrics = run_epoch(model, _capped(val_loader, args.max_batches), eval_views,
                                args.device, None, None, epoch, cfg, train=False)

        record = {"epoch": epoch, "seconds": round(time.time() - started, 1),
                  "train": train_metrics, "val_select": val_metrics,
                  "lr": optimizer.param_groups[0]["lr"]}
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"epoch {epoch:3d}  train loss {train_metrics['loss']:.4f}  "
              f"VAL_select video AUROC {val_metrics.get('video_auroc', float('nan')):.4f}  "
              f"ECE {val_metrics.get('ece', float('nan')):.4f}")

        # §11: save EVERY epoch. Selection happens afterwards, on VAL_select only, so the run
        # must not decide in advance which epochs are worth keeping.
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "metrics": record,
                    "config": cfg},
                   args.output / f"epoch_{epoch:03d}.pth")

    print(f"\nwrote {log_path} and {epochs} checkpoints to {args.output}")
    print("Next: §11 checkpoint selection on VAL_select ONLY — never on an OOD source.")
    return 0


def _capped(loader, max_batches: int):
    if not max_batches:
        return loader

    class _Capped:
        def __iter__(self):
            for i, b in enumerate(loader):
                if i >= max_batches:
                    return
                yield b

        def __len__(self):
            return min(max_batches, len(loader))

    return _Capped()


def _restrict_to_val_select(val_set: NeSyDeFakeDataset, cfg: dict) -> NeSyDeFakeDataset:
    """Keep only VAL_select videos (§11/§12).

    Validating on all of FF++ val would select the checkpoint using VAL_meta as well, and VAL_meta
    is what the gates and the risk model are later fit on — the checkpoint would then have been
    chosen partly on the data used to calibrate its own defer policy.
    """
    split_file = REPO / cfg["meta_split"]["file"]
    if not split_file.is_file():
        raise SystemExit(
            f"{split_file} not found. Run analysis/discern_v2/meta_split.py first — without it "
            f"selection would silently use VAL_meta too, which the gates are fit on.")
    mapping = json.loads(split_file.read_text())["video_to_partition"]
    keep = [i for i, p in enumerate(val_set.image_list)
            if mapping.get(video_of(p if isinstance(p, str) else p[0])) == "VAL_select"]
    if not keep:
        raise SystemExit("no VAL_select frames matched the split file; check the video naming")
    val_set.image_list = [val_set.image_list[i] for i in keep]
    val_set.label_list = [val_set.label_list[i] for i in keep]
    val_set.data_dict = {"image": val_set.image_list, "label": val_set.label_list}
    val_set._build_source_video_maps()
    print(f"  VAL_select: {len(keep)} frames")
    return val_set


if __name__ == "__main__":
    raise SystemExit(main())

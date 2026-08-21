#!/usr/bin/env python3
"""FPAD two-stage training (brief Stages 1 and 3).

    🔴 UMAR-RUNS. Stage A trains the student; Stage B trains H_traj on the frozen student.

    # the two MATCHED students the Stage-2 gate compares (only lambda_preserve differs)
    python training/train_fpad.py --stage A --lambda-preserve 0.0 \
        --output logs/fpad/studentA_ordinary_seed42 --device cuda:N
    python training/train_fpad.py --stage A --lambda-preserve 1.0 \
        --output logs/fpad/studentA_preserve_seed42 --device cuda:N

    # then, per rung
    python training/train_fpad.py --stage B --student logs/fpad/studentA_ordinary_seed42 \
        --output logs/fpad/B2_seed42 --device cuda:N

Why two stages
--------------
At initialization the teacher and student are bit-identical, so `d_l^TS = 0` everywhere and the
gradient of any cosine-distance objective is zero there. Driving LoRA through
`D(x) -> H_traj -> L_cls` alone therefore starts in a flat region and cannot get out. So:

  **Stage A, adapt.**     LoRA + a direct head, `L_adapt = L_cls_direct + lambda * L_preserve`.
                          LoRA gets an ordinary classification gradient, not a degenerate one.
  **Stage B, interpret.** Student frozen; only `H_traj` trains on `D(x)`.

It also makes the causal claim honest: the adaptation is created by an independent objective, and
the trajectory only measures where it departed from the prior, rather than both creating and
classifying itself.

`L_preserve` is computed from the SAME `D` tensor the trajectory readout uses (authentic rows
only), so the quantity being preserved is provably the quantity being measured. There is no second
distance with its own pooling choice.

Matched students, proven rather than promised
---------------------------------------------
The Stage-2 gate compares two Stage-A students that must differ ONLY in `lambda_preserve`. Shared
seed, LoRA targets, batch order, augmentation draws, optimizer and step count. Seeding alone makes
that true — `L_preserve` does not touch data order — but "true by construction" is what every
silent mismatch also looks like, so each run records a `batch_order_fingerprint` per epoch. Two
runs whose fingerprints differ were not matched, and `assert_matched.py` says so instead of the
gate quietly comparing training maturity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(REPO / "analysis" / "discern_v2"))

from dataset.nesy_defake_dataset import NeSyDeFakeDataset  # noqa: E402
from dataset.paired_sampler import (  # noqa: E402
    MatchedAugment, PairedBatchSampler, pair_keys_for_paths, source_ids_for)
from networks.fpad import (  # noqa: E402
    DEFAULT_LAYERS, DirectEvidenceHead, FPADTeacherStudent, TrajectoryEvidenceHead)
from train_v1 import (  # noqa: E402
    _restrict_to_val_select, edl_loss, expected_calibration_error, load_split,
    prepare_dataset_config, video_level, video_of)

DEFAULT_CONFIG = REPO / "training/config/fpad/FPAD_CONFIG.yaml"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

class FpadViews:
    """One augmented image in [0,1]; the model applies FS-VFM's own statistics internally.

    Deliberately thinner than V1's `ViewMaker`: there is one encoder here, so there is one
    normalization, and it lives in `FPADTeacherStudent.normalize` where the checkpoint's
    statistics are read from file. Nothing here can hand the encoder another model's constants.
    """

    def __init__(self, augment: bool, matched: bool, seed: int = 42):
        self.augment = augment
        self.matched = matched and augment
        self.matched_augment = MatchedAugment(seed=seed) if self.matched else None
        self.transform = None
        if augment and not self.matched:
            from networks.discern_v2.semantic_branch import dicome_train_transform
            self.transform = dicome_train_transform()
        self._extract = NeSyDeFakeDataset._extract_source_video
        self.step = 0
        self.matched_seen = 0.0
        self.batches = 0

    def set_epoch(self, epoch: int) -> None:
        if self.matched_augment is not None:
            self.matched_augment.set_epoch(epoch)
        self.step = 0
        self.matched_seen = 0.0
        self.batches = 0

    def matched_fraction(self) -> float:
        return self.matched_seen / self.batches if self.batches else 0.0

    def __call__(self, batch: dict, device: str) -> torch.Tensor:
        raw = batch["raw_frames"].to(device)
        if raw.min() < -0.01:
            raise ValueError("raw_frames are not in [0, 1]; FS-VFM must apply its own statistics")
        if self.matched_augment is not None:
            keys = pair_keys_for_paths(batch["name"], self._extract)
            self.matched_seen += self.matched_augment.matched_fraction(keys)
            self.batches += 1
            raw = self.matched_augment(raw, keys, step=self.step)
            self.step += 1
        elif self.transform is not None:
            raw = torch.stack([self.transform(img) for img in raw])
        return raw


def batch_fingerprint(names: list[str]) -> str:
    """A stable hash of one epoch's sample order, so 'matched' is checkable after the fact."""
    h = hashlib.sha256()
    for n in names:
        h.update(str(n).encode())
    return h.hexdigest()[:16]


def make_loaders(cfg: dict, data_cfg: dict, batch_size: int, sampling: str, seed: int,
                 frames_per_video: int | None = None):
    """Build the FF++ loaders. `frames_per_video` subsamples the TRAIN split only.

    Train only, deliberately: validation keeps its full frame count so the selection metric
    means the same thing across runs with different training budgets. Subsampling is EVEN over
    each video (`abstract_dataset.py`), so 16 of 32 is every second frame rather than the first
    sixteen — contiguous frames are near-duplicates and would not halve the information.
    """
    cfg_val_data = data_cfg
    if frames_per_video:
        data_cfg = {**data_cfg,
                    "frame_num": {**data_cfg["frame_num"], "train": int(frames_per_video)}}
    train_set = load_split(data_cfg, "FaceForensics++", "train")
    # val is built from the ORIGINAL config, so its frame count is untouched
    val_set = load_split(cfg_val_data, "FaceForensics++", "val")
    val_select = _restrict_to_val_select(val_set, cfg)

    sampler = None
    if sampling == "paired":
        sampler = PairedBatchSampler(labels=train_set.label_list,
                                     source_ids=source_ids_for(train_set),
                                     batch_size=batch_size, seed=seed, drop_last=True)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_sampler=sampler, num_workers=int(data_cfg["workers"]),
            collate_fn=train_set.collate_fn)
        print(f"  sampling: source-paired — {sampler.n_pairs} pairs, "
              f"{sampler.pair_fraction():.1%} of samples have a partner")
    else:
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True,
            num_workers=int(data_cfg["workers"]), collate_fn=train_set.collate_fn,
            drop_last=True)
        print("  sampling: random (control)")
    val_loader = torch.utils.data.DataLoader(
        val_select, batch_size=batch_size, shuffle=False,
        num_workers=int(data_cfg["workers"]), collate_fn=val_select.collate_fn)
    return train_loader, val_loader, sampler


# ---------------------------------------------------------------------------
# Stage A — adapt the student
# ---------------------------------------------------------------------------

def stage_a_epoch(model, direct, loader, views, device, optimizer, scheduler, epoch, cfg,
                  lambda_preserve: float, train: bool, max_batches: int = 0,
                  amp: bool = False) -> dict:
    from tqdm import tqdm

    model.train(train)
    direct.train(train)
    totals: dict[str, float] = {}
    n = 0
    names_seen: list[str] = []
    probs, labels_all, videos = [], [], []
    total_epochs = int(cfg["training"]["max_epochs"])

    for i, batch in enumerate(tqdm(loader, desc=("A/train" if train else "A/val"), leave=False)):
        if max_batches and i >= max_batches:
            break
        labels = torch.where(batch["label"] != 0, 1, 0).to(device)
        pixels = views(batch, device)
        names_seen.extend(str(k) for k in batch["name"])

        with torch.set_grad_enabled(train):
            with torch.autocast(pixels.device.type, dtype=torch.bfloat16, enabled=amp):
                out = model(pixels)
            # D is the measurement, so it is always read in fp32 regardless of autocast. This
            # does NOT undo bf16's damage — both representations were already rounded before the
            # difference was taken — it only keeps the pooling and cosine from adding more.
            out["D"] = out["D"].float()
            head = direct(out["h_student"].float())
            loss_cls = edl_loss(head["evidence"], labels, current_epoch=epoch,
                                total_epochs=total_epochs)
            parts = {"loss_cls_direct": loss_cls}
            loss = loss_cls

            if lambda_preserve > 0:
                # AUTHENTIC faces only, and from the SAME `D` the trajectory readout uses.
                # Preserving the teacher's prior on reals is what forces adaptation to
                # concentrate on manipulation rather than on domain.
                real = labels == 0
                if real.any():
                    preserve = out["D"][real].sum(dim=1).mean()
                else:
                    preserve = torch.zeros((), device=device)
                parts["loss_preserve"] = preserve
                loss = loss + lambda_preserve * preserve

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        n += 1
        totals["loss"] = totals.get("loss", 0.0) + float(loss.detach())
        for k, v in parts.items():
            totals[k] = totals.get(k, 0.0) + float(v.detach())
        # the mean trajectory, logged every epoch: this IS the depth profile Stage 4 plots, and
        # watching it during Stage A is how a dead or saturated adaptation shows up early
        d_mean = out["D"].detach().mean(dim=0)
        for li, layer in enumerate(model.layers):
            key = f"d_layer{layer + 1}"
            totals[key] = totals.get(key, 0.0) + float(d_mean[li])
        if not train:
            probs.append(head["prob"].detach().cpu().numpy())
            labels_all.append(labels.cpu().numpy())
            videos.extend(video_of(p) for p in batch["name"])

    metrics = {k: v / max(1, n) for k, v in totals.items()}
    metrics["batch_order_fingerprint"] = batch_fingerprint(names_seen)
    metrics["n_batches"] = n
    if not train and probs:
        from sklearn.metrics import roc_auc_score
        p = np.concatenate(probs)
        y = np.concatenate(labels_all)
        vp, vy = video_level(p, y, videos)
        metrics["frame_auroc"] = (float(roc_auc_score(y, p)) if len(np.unique(y)) > 1
                                  else float("nan"))
        metrics["video_auroc"] = (float(roc_auc_score(vy, vp)) if len(np.unique(vy)) > 1
                                  else float("nan"))
        metrics["ece"] = expected_calibration_error(p, y)
    return metrics


# ---------------------------------------------------------------------------
# Stage B — interpret the frozen adaptation
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_trajectories(model, loader, views, device, max_batches: int = 0):
    """`D(x)` for every frame, with the student frozen. Used to fit H_traj's calibrator."""
    from tqdm import tqdm

    model.eval()
    D, labels, videos, names = [], [], [], []
    for i, batch in enumerate(tqdm(loader, desc="  D(x)", leave=False)):
        if max_batches and i >= max_batches:
            break
        out = model(views(batch, device))
        D.append(out["D"].cpu())
        labels.append(torch.where(batch["label"] != 0, 1, 0))
        videos.extend(video_of(p) for p in batch["name"])
        names.extend(str(k) for k in batch["name"])
    return torch.cat(D), torch.cat(labels), videos, names


def stage_b_epoch(head, D, labels, device, optimizer, epoch, total_epochs, batch: int,
                  train: bool, videos=None) -> dict:
    head.train(train)
    perm = torch.randperm(len(D)) if train else torch.arange(len(D))
    totals, n = 0.0, 0
    probs = []
    for i in range(0, len(perm), batch):
        idx = perm[i:i + batch]
        d, y = D[idx].to(device), labels[idx].to(device)
        with torch.set_grad_enabled(train):
            out = head(d)
            loss = edl_loss(out["evidence"], y, current_epoch=epoch, total_epochs=total_epochs)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        totals += float(loss.detach())
        n += 1
        if not train:
            probs.append(out["prob"].detach().cpu().numpy())
    metrics = {"loss": totals / max(1, n)}
    if not train and probs:
        from sklearn.metrics import roc_auc_score
        p = np.concatenate(probs)
        y = labels.numpy()
        metrics["frame_auroc"] = (float(roc_auc_score(y, p)) if len(np.unique(y)) > 1
                                  else float("nan"))
        if videos is not None:
            vp, vy = video_level(p, y, videos)
            metrics["video_auroc"] = (float(roc_auc_score(vy, vp)) if len(np.unique(vy)) > 1
                                      else float("nan"))
        metrics["ece"] = expected_calibration_error(p, y)
    return metrics


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=("A", "B"), required=True)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--detector-config", type=Path,
                    default=REPO / "training/config/detector/nesy_defake_d1_v.yaml")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--student", type=Path, default=None,
                    help="Stage B only: the Stage-A run whose frozen student to interpret")
    ap.add_argument("--lambda-preserve", type=float, default=None,
                    help="Stage A only. 0.0 = the ordinary-LoRA student (rung B2's); >0 = the "
                         "preservation student (rung B3's). These two, at the same seed, are the "
                         "matched pair the Stage-2 gate compares.")
    ap.add_argument("--sampling", choices=("paired", "random"), default="paired")
    ap.add_argument("--frames-per-video", type=int, default=None,
                    help="subsample the TRAIN split to N evenly-spaced frames per video "
                         "(validation is left at its full count so the selection metric keeps "
                         "one definition). 32 are extracted, so 16 halves the epoch.")
    ap.add_argument("--freeze-student", action="store_true",
                    help="rung B0: train ONLY the direct head on the frozen FS-VFM feature, with "
                         "no LoRA adaptation at all. `D` stays identically zero, so B0 has no "
                         "trajectory readout by construction — it is the frozen-prior baseline "
                         "the ladder starts from, not a degenerate B1.")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the latest epoch_*.pth in --output, restoring the "
                         "optimizer, scheduler and RNG state. Mutually exclusive with "
                         "--overwrite.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--amp", action="store_true",
                    help="bf16 autocast. OFF by default and it should usually stay off: measured "
                         "on this model, bf16 introduces up to 80.8%% RELATIVE error in `D`, "
                         "because the teacher and student representations differ by ~1e-5 early "
                         "in training and bf16 rounds both to ~3 significant digits before the "
                         "difference is taken. Casting up afterwards cannot recover it. It buys "
                         "2.1x (37 -> 18 min/epoch); take that trade only once a run has shown "
                         "`D` reaching a scale where 0.4%% relative precision is harmless, and "
                         "the per-epoch d_layer* values in metrics.jsonl are how to check.")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.epochs is not None:
        cfg["training"]["max_epochs"] = args.epochs
    epochs = int(cfg["training"]["max_epochs"])
    seed = int(cfg["meta"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    # `random` too, and it is NOT redundant: abstract_dataset.py:346 shuffles the collected
    # (label, path, video) lists with `random.shuffle` on the GLOBAL module RNG. Seeding only
    # torch and numpy leaves the dataset ORDER process-dependent — measured: the same 115,198
    # FF++ train frames hashed to two different orders across two runs. Every "same seed" claim
    # downstream, matched students included, rests on this line.
    random.seed(seed)

    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite are contradictory; pick one")
    args.output.mkdir(parents=True, exist_ok=True)
    existing = sorted(args.output.glob("epoch_*.pth"))
    if existing and not (args.overwrite or args.resume):
        raise SystemExit(
            f"{args.output} already holds {len(existing)} checkpoints. Appending would interleave "
            f"two runs' epochs in metrics.jsonl and leave checkpoints from different "
            f"configurations sharing one directory. Pass --overwrite, --resume, or choose a new "
            f"directory.")

    model = FPADTeacherStudent(
        checkpoint=cfg["encoder"]["checkpoint"] if cfg["encoder"].get("checkpoint") else
        (REPO / "weights/FS-VFM/checkpoint-599.pth"),
        layers=tuple(cfg["encoder"]["layers"]),
        lora=cfg["lora"],
        img_size=int(cfg["encoder"].get("img_size", 224))).to(args.device)
    if args.freeze_student:
        # B0. Disabling the adapters' gradient rather than rebuilding without LoRA keeps the
        # module identical to B1's, so the two rungs differ ONLY in whether the student adapts.
        for name, param in model.peft.named_parameters():
            param.requires_grad_(False)
        if args.lambda_preserve not in (None, 0.0):
            raise SystemExit(
                "--freeze-student with a non-zero --lambda-preserve is contradictory: there is no "
                "adaptation to preserve against. B0 takes --lambda-preserve 0.")
    model.assert_teacher_frozen()
    counts = model.trainable_parameters()
    print(f"FPAD teacher-student: layers {model.layers} "
          f"(1-indexed {tuple(i + 1 for i in model.layers)})")
    print(f"  LoRA {counts['lora']:,} trainable / {counts['frozen']:,} frozen")

    data_cfg = prepare_dataset_config(args.detector_config, args.batch_size, args.workers)
    frames_per_video = args.frames_per_video or cfg["training"].get("frames_per_video")
    train_loader, val_loader, sampler = make_loaders(cfg, data_cfg, args.batch_size,
                                                     args.sampling, seed, frames_per_video)
    train_views = FpadViews(augment=True, matched=(args.sampling == "paired"), seed=seed)
    eval_views = FpadViews(augment=False, matched=False)

    run_meta = {
        "stage": args.stage, "seed": seed, "sampling": args.sampling,
        "layers": list(model.layers), "layers_one_indexed": [i + 1 for i in model.layers],
        "lora": model.lora_config, "readout": "mean-pooled patch tokens (CLS reported beside)",
        "encoder": model.provenance, "batch_size": args.batch_size, "epochs": epochs,
        "config": str(args.config), "amp": bool(args.amp),
        "frames_per_video": frames_per_video,
        "precision_note": ("bf16 autocast ENABLED — measured up to 80.8% relative error in D on "
                           "this model; verify the d_layer* scale justifies it"
                           if args.amp else "fp32 throughout; D is the measurement"),
    }

    if args.stage == "A":
        if args.lambda_preserve is None:
            raise SystemExit(
                "--lambda-preserve is required for Stage A and has no default on purpose: it is "
                "the ONLY thing that distinguishes the two matched students the Stage-2 gate "
                "compares, so it must be stated at the call site.")
        lam = args.lambda_preserve
        run_meta["lambda_preserve"] = lam
        run_meta["arm"] = ("frozen_b0" if args.freeze_student else
                           "ordinary_lora" if lam == 0 else "preservation")
        run_meta["freeze_student"] = bool(args.freeze_student)
        print(f"  Stage A [{run_meta['arm']}]: lambda_preserve = {lam}")

        direct = DirectEvidenceHead(feature_dim=model.embed_dim,
                                    hidden_dim=int(cfg["heads"]["hidden_dim"])).to(args.device)
        params = [p for p in model.parameters() if p.requires_grad] + list(direct.parameters())
        if not params:
            raise SystemExit("nothing to optimize")
        optimizer = torch.optim.AdamW(params, lr=float(cfg["training"]["lr"]),
                                      betas=tuple(cfg["training"]["betas"]),
                                      weight_decay=float(cfg["training"]["weight_decay"]))
        steps = max(1, len(train_loader)) * epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=steps, eta_min=float(cfg["training"]["min_lr"]))

        log = args.output / "metrics.jsonl"

        # ---- resume -----------------------------------------------------------------------
        start_epoch = 0
        if args.resume and existing:
            blob = torch.load(str(existing[-1]), map_location=args.device, weights_only=False)
            prior = blob.get("run_meta", {})
            # A resume that silently changes the recipe is worse than no resume: the run would
            # carry one run_meta while half its epochs were trained under another.
            for field in ("seed", "sampling", "layers", "lora", "lambda_preserve",
                          "frames_per_video", "batch_size"):
                if prior.get(field) != run_meta.get(field):
                    raise SystemExit(
                        f"cannot resume: `{field}` differs from the checkpoint "
                        f"({prior.get(field)!r} vs {run_meta.get(field)!r}). Resuming into a "
                        f"different recipe would produce a run whose epochs were not trained "
                        f"under one configuration.")
            model.load_state_dict(blob["lora"], strict=False)
            direct.load_state_dict(blob["direct_head"])
            optimizer.load_state_dict(blob["optimizer"])
            if blob.get("scheduler") is not None:
                scheduler.load_state_dict(blob["scheduler"])
            rng = blob.get("rng", {})
            if rng:
                # Restored so the remaining epochs are the ones the run WOULD have produced.
                # The paired sampler and MatchedAugment are seeded per epoch and so are exact
                # regardless; this covers the random-sampling arm's DataLoader shuffle.
                torch.set_rng_state(rng["torch"].cpu() if hasattr(rng["torch"], "cpu")
                                    else rng["torch"])
                np.random.set_state(rng["numpy"])
                random.setstate(rng["python"])
            start_epoch = int(blob["epoch"]) + 1
            print(f"  RESUMED from {existing[-1].name}; continuing at epoch {start_epoch}"
                  f"/{epochs}")
            if start_epoch >= epochs:
                print("  nothing to do: the run already reached --epochs")
                return 0

        (args.output / "run_meta.json").write_text(json.dumps(run_meta, indent=2, default=str))
        for epoch in range(start_epoch, epochs):
            started = time.time()
            if sampler is not None:
                sampler.set_epoch(epoch)
            train_views.set_epoch(epoch)
            tr = stage_a_epoch(model, direct, train_loader, train_views, args.device, optimizer,
                               scheduler, epoch, cfg, lam, train=True,
                               max_batches=args.max_batches, amp=args.amp)
            va = stage_a_epoch(model, direct, val_loader, eval_views, args.device, None, None,
                               epoch, cfg, lam, train=False, max_batches=args.max_batches,
                               amp=args.amp)
            if sampler is not None:
                tr["pair_fraction"] = round(sampler.pair_fraction(), 4)
                tr["matched_aug_fraction"] = round(train_views.matched_fraction(), 4)
            record = {"epoch": epoch, "seconds": round(time.time() - started, 1),
                      "lambda_preserve": lam, "train": tr, "val_select": va}
            with open(log, "a") as f:
                f.write(json.dumps(record) + "\n")
            profile = " ".join(f"{tr[f'd_layer{l + 1}']:.4f}" for l in model.layers)
            print(f"epoch {epoch:3d}  loss {tr['loss']:.4f}  "
                  f"VAL video AUROC {va.get('video_auroc', float('nan')):.4f}  "
                  f"D=[{profile}]  fp {tr['batch_order_fingerprint']}")
            torch.save({"epoch": epoch,
                        "lora": {k: v for k, v in model.state_dict().items() if "lora_" in k},
                        "direct_head": direct.state_dict(),
                        # optimizer/scheduler/RNG so a resume continues the run rather than
                        # restarting the trajectory with a cold optimizer
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler is not None else None,
                        "rng": {"torch": torch.get_rng_state(),
                                "numpy": np.random.get_state(),
                                "python": random.getstate()},
                        "run_meta": run_meta, "metrics": record},
                       args.output / f"epoch_{epoch:03d}.pth")
        print(f"\nwrote {log} and {epochs} checkpoints to {args.output}")
        print("Student is now trainable-frozen for Stage B. Select on FF++ VAL_select ONLY; "
              "in-domain validation saturates, so treat small differences as noise.")
        return 0

    # ---------------- Stage B ----------------
    if args.student is None:
        raise SystemExit("--student is required for Stage B: the frozen Stage-A run to interpret")
    selected = sorted(args.student.glob("epoch_*.pth"))
    if not selected:
        raise SystemExit(f"no checkpoints in {args.student}")
    blob = torch.load(str(selected[-1]), map_location="cpu", weights_only=False)
    missing = model.load_state_dict(blob["lora"], strict=False)
    if any("lora_" in k for k in missing.unexpected_keys):
        raise SystemExit(f"unexpected LoRA keys: {missing.unexpected_keys[:3]}")
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    run_meta["student"] = {"run": str(args.student), "checkpoint": str(selected[-1]),
                           "arm": blob["run_meta"].get("arm"),
                           "lambda_preserve": blob["run_meta"].get("lambda_preserve")}
    print(f"  Stage B: interpreting the FROZEN student from {selected[-1].name} "
          f"[{run_meta['student']['arm']}]")

    D_tr, y_tr, v_tr, _ = collect_trajectories(model, train_loader, eval_views, args.device,
                                               args.max_batches)
    D_va, y_va, v_va, _ = collect_trajectories(model, val_loader, eval_views, args.device,
                                               args.max_batches)
    head = TrajectoryEvidenceHead(n_layers=len(model.layers),
                                  hidden_dim=int(cfg["heads"]["hidden_dim"]),
                                  use_slopes=bool(cfg["heads"].get("use_slopes", True)))
    # AUTHENTIC training trajectories only — the yardstick must be authentic video
    head.fit_calibrator(D_tr[y_tr == 0])
    head = head.to(args.device)
    print(f"  H_traj: {head.describe()} · calibrated on {int((y_tr == 0).sum())} authentic "
          f"trajectories")

    optimizer = torch.optim.AdamW(head.parameters(), lr=float(cfg["heads"]["lr"]),
                                  weight_decay=float(cfg["training"]["weight_decay"]))
    log = args.output / "metrics.jsonl"
    (args.output / "run_meta.json").write_text(json.dumps(run_meta, indent=2, default=str))
    head_epochs = int(cfg["heads"]["epochs"])
    for epoch in range(head_epochs):
        tr = stage_b_epoch(head, D_tr, y_tr, args.device, optimizer, epoch, head_epochs,
                           batch=1024, train=True)
        va = stage_b_epoch(head, D_va, y_va, args.device, None, epoch, head_epochs,
                           batch=1024, train=False, videos=v_va)
        record = {"epoch": epoch, "train": tr, "val_select": va}
        with open(log, "a") as f:
            f.write(json.dumps(record) + "\n")
        if epoch % max(1, head_epochs // 10) == 0 or epoch == head_epochs - 1:
            print(f"  H_traj epoch {epoch:3d}  loss {tr['loss']:.4f}  "
                  f"VAL video AUROC {va.get('video_auroc', float('nan')):.4f}")
        torch.save({"epoch": epoch, "traj_head": head.state_dict(),
                    "head_describe": head.describe(), "run_meta": run_meta,
                    "metrics": record}, args.output / f"epoch_{epoch:03d}.pth")
    print(f"\nwrote {log} and {head_epochs} checkpoints to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

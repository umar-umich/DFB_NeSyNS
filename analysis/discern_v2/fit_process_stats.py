#!/usr/bin/env python3
"""Fit the process branch's standardisation on FF++ authentic training frames (V1 spec §5).

§5 requires the process diagnostics to be standardised with FF++ REAL TRAINING statistics. Those
statistics have to come from somewhere, and the two convenient options are both wrong:

* per-batch statistics would make the yardstick move with whatever is in the batch, so a batch of
  fakes would look normal;
* statistics over all training data would fold fake reconstruction errors into the reference
  distribution and shrink the very deviation the branch measures.

So they are fit once, offline, on authentic frames only, and frozen — the same protocol and the
same `ResidualCalibrator` the reference branch uses, so the two specialists' inputs are
comparable by construction.

Reals-only is enforced here rather than trusted to the caller, for the same reason as Stage A: a
calibrator fit on fakes still produces plausible numbers and nothing downstream reveals it.

🔴 UMAR-RUNS. This runs the frozen VAE forward over FF++ train frames; nothing trains.

    python analysis/discern_v2/fit_process_stats.py \
        --config training/config/detector/nesy_defake_d1_v.yaml \
        --vae-path /data/umar/Repos/DiCoME/eval_adaptation/data/models/sdxl-vae \
        --out configs/discern_v2/process/process_stats.pt
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cache_encoder_features as CE  # noqa: E402  — dataset/split loading, shared
from networks.discern_v2.process_branch import RAW_MEAN, RAW_STD  # noqa: E402
from networks.discern_v2.process_residual import STAT_NAMES, build_operator  # noqa: E402
from networks.discern_v2.reference import ResidualCalibrator  # noqa: E402


@torch.no_grad()
def collect_real_stats(operator, loader, device: str, max_batches: int = 0) -> np.ndarray:
    from tqdm import tqdm

    out = []
    n_seen = n_real = 0
    for i, batch in enumerate(tqdm(loader, desc="  process stats")):
        if max_batches and i >= max_batches:
            break
        labels = torch.where(batch["label"] != 0, 1, 0)
        real = labels == 0
        n_seen += len(labels)
        if not real.any():
            continue
        images = batch["raw_frames"][real].to(device)
        if images.min() < -0.01:
            raise ValueError(
                "raw_frames appear pre-normalised; the operator is configured for [0, 1] input "
                "(mean 0 / std 1) and would un-normalise with the wrong constants")
        stats = operator(images)
        out.append(stats.float().cpu().numpy())
        n_real += int(real.sum())
    if not out:
        raise RuntimeError("no authentic frames found — refusing to fit on nothing")
    print(f"  kept {n_real} authentic frames of {n_seen} seen")
    return np.concatenate(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--vae-path", type=Path, required=True,
                    help="local stabilityai/sdxl-vae checkout")
    ap.add_argument("--dataset", default="FaceForensics++")
    ap.add_argument("--split", default="train",
                    help="must be a TRAINING split: these are training statistics")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path,
                    default=Path("configs/discern_v2/process/process_stats.pt"))
    args = ap.parse_args()

    if args.split != "train":
        print(f"  WARNING: fitting on the {args.split!r} split. §5 specifies FF++ real TRAINING "
              f"statistics; anything else has seen data the audits later report on.")

    config = CE.strip_unused_loading(CE.load_config(args.config))
    config["test_batchSize"] = args.batch_size
    if args.workers is not None:
        config["workers"] = args.workers

    print(f"building frozen VAE operator from {args.vae_path}")
    operator = build_operator(vae_path=args.vae_path, resolution=args.resolution,
                              input_mean=RAW_MEAN, input_std=RAW_STD).to(args.device)
    operator.eval()
    check = operator.sanity_check()
    if not check["vae_all_frozen"]:
        raise SystemExit("the VAE is not fully frozen; refusing to fit statistics against it")
    print(f"  {check}")

    dataset = CE.load_split(config, args.dataset, args.split)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=int(config["workers"]), collate_fn=dataset.collate_fn, drop_last=False)

    stats = collect_real_stats(operator, loader, args.device, args.max_batches)
    print(f"  statistics {stats.shape} over {len(STAT_NAMES)} named quantities")

    calibrator = ResidualCalibrator(stats.shape[1]).fit(torch.from_numpy(stats).float())
    provenance = {
        "n_real": int(stats.shape[0]),
        "dataset": f"{args.dataset}/{args.split}",
        "vae_path": str(args.vae_path),
        "resolution": args.resolution,
        "stat_names": list(STAT_NAMES),
        "input_normalization": {"mean": list(RAW_MEAN), "std": list(RAW_STD),
                                "note": "raw_frames in [0,1]; the operator's un-normalisation "
                                        "is the identity and the VAE sees its native scale"},
        "mean": stats.mean(axis=0).tolist(),
        "std": stats.std(axis=0).tolist(),
        "partial": bool(args.max_batches),
        "git_commit": _git_commit(),
    }
    for name, mu, sd in zip(STAT_NAMES, provenance["mean"], provenance["std"]):
        print(f"    {name:14s} mu={mu:+.5f}  sigma={sd:.5f}")
        if sd < 1e-8:
            print(f"      WARNING: {name} is constant on authentic frames; standardising it "
                  f"amplifies numerical noise and the head should probably not read it")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"n_stats": int(stats.shape[1]), "calibrator_state": calibrator.state_dict(),
                "provenance": provenance}, args.out)
    args.out.with_suffix(".json").write_text(json.dumps(provenance, indent=2))
    print(f"\nwrote {args.out}")
    print("These statistics are FROZEN — Stage B loads them read-only.")
    return 0


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())

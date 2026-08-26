#!/usr/bin/env python
"""Export DiCoME's PER-VIEW evidence, not just the fused head.

    conda activate discern_ext
    python eval_adaptation/export_views.py \
        --checkpoint weights/dicome-best.ckpt \
        --config eval_adaptation/configs/CDFv2.yaml \
        --dataset CDFv2 \
        --out /data/umar/Repos/DFB_NeSyNS/logs/tbiom/step1/dicome_released_CDFv2.csv

WHY THIS EXISTS. The recovery brief's Step 1 asks whether the ~5-point Celeb-DF-v2 gap between
DiCoME and our CLIP port belongs to the CLIP branch or to DiCoME's extra machinery. DiCoME's
`core_model.forward` already returns `semantic_evidence` and `artifact_evidence` beside
`fused_evidence`, but `DiCoMEModule.test_step` throws the first two away
(`fused_evidence, _, _, _, _, _, _, _ = self(batch.images)`) and only the fused head is ever
written. So the CLIP-only readout is one line away and has never been exported.

This is ADDITIVE. It does not modify `DiCoMEModule`, so the released-checkpoint reproduction in
`runs/dicome_repro/` stays byte-for-byte the baseline it is, which the brief requires for Step 8.

What it writes, per frame: the h5 key, the label, and `p_fake` for each of the three readouts —
`p_fused`, `p_semantic`, `p_artifact` — plus each one's Dirichlet uncertainty. Probabilities are
the evidential expectation `alpha / S` with `alpha = evidence + 1`, exactly as
`_predict_from_evidence` computes them, so `p_fused` here must reproduce the existing eval path.
That equality is the check that this exporter is reading the model correctly, and it is asserted
against the shipped per-frame CSV when one is available.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.config import Config, load_config  # noqa: E402
from src.dataset.deepfake import H5DeepfakeDataModule  # noqa: E402
from src.model.dicome_module import DiCoMEModule  # noqa: E402


def probs_and_uncertainty(evidence: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """The evidential expectation, matching DiCoMEModule._predict_from_evidence exactly."""
    alpha = evidence + 1
    strength = torch.sum(alpha, dim=1, keepdim=True).clamp_min(1e-6)
    expected = alpha / strength
    uncertainty = alpha.shape[1] / strength.squeeze(1)
    return expected.detach().cpu().numpy(), uncertainty.detach().cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True,
                    help="key inside the config's tst_files mapping, e.g. CDFv2")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit-batches", type=int, default=0)
    args = ap.parse_args()

    if args.out.exists():
        print(f"{args.out} exists — skipping")
        return 0

    config = load_config(args.config)
    config = Config(**config.model_copy(update={
        "mini_batch_size": args.batch_size,
        "batch_size": args.batch_size,
    }).model_dump())

    dm = H5DeepfakeDataModule(config=config, vfm_model_name=config.backbone)
    dm.setup("test")
    loader = dm.test_dataloader()
    if isinstance(loader, dict):
        loader = loader[args.dataset]
    elif isinstance(loader, (list, tuple)):
        names = list(config.tst_files.keys())
        loader = loader[names.index(args.dataset)]

    module = DiCoMEModule.load_from_checkpoint(args.checkpoint, config=config)
    model = module.model.to(args.device).eval()

    # The h5 dataset's __getitem__ already returns `path` (`<h5>::<key>`) and a parsed `video`,
    # so identity is read from the batch rather than reconstructed by indexing back into
    # `dataset.files`. That indexing is what `_files_from_indices` does and it silently returns
    # None when an index falls outside the file list; taking the batch's own fields cannot
    # misalign.
    rows = []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if args.limit_batches and bi >= args.limit_batches:
                break
            images = batch["image"] if isinstance(batch, dict) else batch[0]
            labels = batch["label"] if isinstance(batch, dict) else batch[1]
            out = model(images.to(args.device))
            fused_e, sem_e, art_e = out[0], out[1], out[2]

            p_f, u_f = probs_and_uncertainty(fused_e)
            p_s, u_s = probs_and_uncertainty(sem_e)
            p_a, u_a = probs_and_uncertainty(art_e)

            n = len(p_f)
            keys = list(batch["path"])[:n] if isinstance(batch, dict) and "path" in batch \
                else [f"{args.dataset}#{bi * args.batch_size + j}" for j in range(n)]
            videos = list(batch["video"])[:n] if isinstance(batch, dict) and "video" in batch \
                else keys
            rows.append(pd.DataFrame({
                "key": keys,
                "video": videos,
                "label": np.asarray(labels).reshape(-1)[:n],
                # class 1 is fake, matching the module's binary_labels convention
                "p_fused": p_f[:, 1], "u_fused": u_f,
                "p_semantic": p_s[:, 1], "u_semantic": u_s,
                "p_artifact": p_a[:, 1], "u_artifact": u_a,
            }))
            if bi % 50 == 0:
                print(f"  batch {bi}", flush=True)

    table = pd.concat(rows, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # CSV, not parquet: the `discern_ext` env has no pyarrow, and it must not be modified — its
    # peft is pinned at 0.14.0 because every P0/P1 checkpoint was trained under that LoRA
    # implementation. The consumer side reads CSV just as easily.
    table.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(table)} frames)")
    for col in ("p_fused", "p_semantic", "p_artifact"):
        r = table.loc[table.label == 0, col]
        f = table.loc[table.label == 1, col]
        print(f"  {col:12s} mean real {r.mean():.4f}  mean fake {f.mean():.4f}  "
              f"sep {f.mean() - r.mean():+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

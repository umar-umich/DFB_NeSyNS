#!/usr/bin/env python3
"""Score DiCoME over the COMBINED DF40 configs: one row per generator, both halves pooled.

    CUDA_VISIBLE_DEVICES=1 <discern_ext_timm python> \
        analysis/tbiom/baselines/dicome_df40_run.py <ckpt> --out <json>

DiCoME's own `eval_df40.py` takes its row list from `results/df40_rows.json` and `--only` only
FILTERS that list, so a config it does not know about is silently dropped. Rather than edit a
registry inside the baseline repo, this driver reuses that script's `score()` and its module
class and iterates our combined configs directly -- so the scoring path is DiCoME's own and
nothing in their tree changes except the configs we added under `ours_combined/`.

The model is built once and the datamodule swapped per row, which is the whole reason their
driver exists: constructing CLIP ViT-L/14 per row costs ~15 s × 21 rows of pure startup.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

REPO = Path("/data/umar/Repos/DiCoME")
# Every path inside a DiCoME config -- splits, h5, run_dir -- is relative to ITS repo root, so
# the driver has to run from there. Without this the split files resolve against whatever
# directory the lane happened to cd into and the loader reports "No valid H5 keys found" about
# files that are sitting right where the config says they are.
import os
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval_adaptation"))

from src.config import load_config                                  # noqa: E402
from src.dataset.deepfake import H5DeepfakeDataModule             # noqa: E402
from src.model.dicome_module import DiCoMEModule                    # noqa: E402
from eval_df40 import score                                         # noqa: E402

CFG = REPO / "eval_adaptation/configs/DF40/ours_combined"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-workers", type=int, default=12)
    a = ap.parse_args()

    names = sorted(p.stem for p in CFG.glob("*.yaml"))
    if not names:
        print(f"no combined configs under {CFG}")
        return 1
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base = load_config(str(CFG / names[0]) + ".yaml")
    model = DiCoMEModule.load_from_checkpoint(a.checkpoint, config=base).to(dev).eval()
    print(f"loaded {a.checkpoint}\n  rows={len(names)} device={dev}\n", flush=True)

    out = Path(a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out.read_text()) if out.exists() else {}
    for i, name in enumerate(names, 1):
        if name in results:
            print(f"[{i}/{len(names)}] {name}: cached", flush=True)
            continue
        cfg = load_config(str(CFG / name) + ".yaml")
        cfg.num_workers = a.num_workers
        dm = H5DeepfakeDataModule(config=cfg, vfm_model_name=cfg.backbone)
        dm.setup("test")
        probs, labels, keys = score(model, dm.test_dataloader(), dev)
        # The project's video rule: the full key path minus the frame. NOT a bare video id,
        # which merges a fake and a real video wherever the two share one.
        vids = [k.rsplit("/", 1)[0] for k in keys]
        order = {}
        for v, p, l in zip(vids, probs, labels):
            order.setdefault(v, []).append((p, l))
        vp = np.array([np.mean([x[0] for x in g]) for g in order.values()])
        vl = np.array([g[0][1] for g in order.values()])
        results[name] = {"auc_video": float(roc_auc_score(vl, vp)),
                         "auc_frame": float(roc_auc_score(labels, probs)),
                         "n_videos": int(len(vl)), "n_real": int((vl == 0).sum()),
                         "n_fake": int((vl == 1).sum())}
        out.write_text(json.dumps(results, indent=2))
        print(f"[{i}/{len(names)}] {name}: AUC={results[name]['auc_video']:.4f} "
              f"({results[name]['n_real']} real / {results[name]['n_fake']} fake)", flush=True)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

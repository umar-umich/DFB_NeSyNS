#!/usr/bin/env python3
"""Part 6c: does REGIONAL temporal evidence beat global pooling, or is it just extra capacity?

    CUDA_VISIBLE_DEVICES=3 <temporal_vjepa2 python> upgrade/part6c_regional.py --row R-REG

THREE ROWS, and the two controls are the point.
  R-REG   five regional features per temporal token group -> a two-layer transformer of width 128
          over region-by-time tokens with learned region and time embeddings, mean pooled, then the
          evidential head.
  R-GLOB  global pooling with a PARAMETER-MATCHED MLP head. Controls for capacity: a transformer
          over 5x T tokens has more parameters than the deployed head, and without this control a
          win could be capacity rather than regional structure.
  R-POOL  regional features MEAN-POOLED over regions, same head capacity as R-REG. Controls for
          the REGIONS: if R-REG only beats this by aggregating five views of the same thing, the
          structure is not what is doing the work.

DIAGNOSTICS, both required by the brief.
  - Frame order shuffled at evaluation, 5 permutations. A gain that SURVIVES shuffling is regional
    aggregation, not temporal modelling, and the report must describe it that way.
  - One region ablated at a time, so a win cannot rest on a single region.

Keep rule: R-REG is kept only if it beats BOTH R-GLOB and R-POOL on dev v3.1 with intervals above
zero.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(ROOT / "upgrade"), str(ROOT / "branches/ijepa"), "/data/umar/Repos/DISCERN_Ext"]
import v31 as V31                                                   # noqa: E402
from src.losses.evidential_loss import evidential_loss_discern_ext  # noqa: E402

torch.set_num_threads(4)
VF = ROOT / "logs/upgrade/vfeats"
OUT = ROOT / "analysis/upgrade/regional"
REGIONS = ["eyes_l", "eyes_r", "nose", "mouth", "jaw"]      # the five from the I-JEPA brief
EPOCHS, BATCH, LR, WD = 20, 32, 1e-4, 0.01
DEV_SETS = ("VALmix", "VALmixV2", "FFpp_val")
V2_SOURCES = {"CDFv3valG", "DF40train", "DS20val", "DS11val", "FFIWval"}


class RegionalTransformer(nn.Module):
    """Two layers, width 128, over region-by-time tokens. No positional shortcut beyond the learned
    region and time embeddings, which is what the brief fixes: a token's identity is (which region,
    which time group) and nothing else, so the model cannot read raw position."""

    def __init__(self, d_in, n_reg, n_time, width=128, layers=2, heads=4):
        super().__init__()
        self.proj = nn.Linear(d_in, width)
        self.reg_emb = nn.Parameter(torch.zeros(n_reg, width))
        self.time_emb = nn.Parameter(torch.zeros(n_time, width))
        nn.init.normal_(self.reg_emb, std=0.02); nn.init.normal_(self.time_emb, std=0.02)
        enc = nn.TransformerEncoderLayer(width, heads, width * 4, dropout=0.1,
                                         batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 2))

    def forward(self, x, region_mask=None):
        # x: (B, T, R, D)
        B, T, R, _ = x.shape
        h = self.proj(x) + self.reg_emb.view(1, 1, R, -1) + self.time_emb.view(1, T, 1, -1)
        h = h.reshape(B, T * R, -1)
        if region_mask is not None:                  # ablate one region: drop its tokens entirely
            keep = region_mask.repeat(T).bool()
            h = h[:, keep]
        return F.softplus(self.head(self.tr(h).mean(1)))


class MatchedMLP(nn.Module):
    """R-GLOB / R-POOL head, parameter-matched to the transformer so capacity is not the variable."""

    def __init__(self, d_in, target_params):
        super().__init__()
        w = max(64, int((target_params / max(1, d_in + 2)) ** 0.5) * 2)
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, w), nn.GELU(),
                                 nn.Dropout(0.1), nn.Linear(w, w), nn.GELU(), nn.Linear(w, 2))

    def forward(self, x, region_mask=None):
        return F.softplus(self.net(x))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", required=True, choices=["R-REG", "R-GLOB", "R-POOL"])
    ap.add_argument("--model", default="vjepa2_h")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle-eval", type=int, default=5, help="frame-order control permutations")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{a.model}_{a.row}_s{a.seed}"
    if (OUT / f"{tag}.json").exists():
        print(f"{tag} exists"); return 0

    need = VF / f"{a.model}_regional_FFpp_train.npz"
    if not need.exists():
        print(f"regional cache missing: {need}\n"
              f"Part 6a cached GLOBAL pooled features only; the dense per-region pass has not been "
              f"run. This row cannot be scored yet -- recorded as TODO(run) rather than approximated "
              f"from global features, which would not be the experiment the brief asks for.")
        json.dump({"row": a.row, "model": a.model, "status": "TODO(run)",
                   "reason": "regional feature cache absent; dense per-region extraction not yet run"},
                  open(OUT / f"{tag}.TODO.json", "w"), indent=1)
        return 3
    print("regional cache present; training", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

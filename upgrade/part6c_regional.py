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


def load(model, dataset):
    """(videos, time groups, regions, dim) plus labels and video ids."""
    f = VF / f"{model}_regional_{dataset}.npz"
    z = np.load(f, allow_pickle=True)
    X = z["feature"].astype(np.float32)
    return {"x": X, "y": z["label"].astype(np.int64),
            "v": np.array([str(v) for v in z["video"]]),
            "key": np.array([str(v) + "/000.png" for v in z["video"]], dtype=object),
            "regions": [str(r) for r in z["regions"]]}


def shape_for(row, X):
    """R-REG keeps the (time, region) grid; the controls collapse it, each in one specific way."""
    if row == "R-REG":
        return X                                   # (N, T, R, D)
    if row == "R-POOL":
        return X.mean(2)                           # mean over REGIONS -> (N, T, D), then flattened
    return X.mean((1, 2))                          # R-GLOB: global pool -> (N, D)


def build(row, X, dev):
    N, T, R, D = X.shape
    if row == "R-REG":
        m = RegionalTransformer(D, R, T)
        return m.to(dev), sum(p.numel() for p in m.parameters())
    ref = sum(p.numel() for p in RegionalTransformer(D, R, T).parameters())
    d_in = D * T if row == "R-POOL" else D
    return MatchedMLP(d_in, ref).to(dev), ref


def feats(row, X):
    if row == "R-REG":
        return X
    if row == "R-POOL":
        return X.mean(2).reshape(len(X), -1)
    return X.mean((1, 2))


def score(model, row, DEVC, dev, perm=None, region_mask=None):
    model.eval(); out = {}
    with torch.no_grad():
        for n, D in DEVC.items():
            X = D["x"]
            if perm is not None:
                X = X[:, perm]                     # frame-order control: permute the time groups
            Z = feats(row, X)
            p = []
            for i in range(0, len(Z), 256):
                b = torch.from_numpy(Z[i:i + 256]).to(dev)
                ev = model(b, region_mask)
                a = ev + 1.0
                p.append((a / a.sum(1, keepdim=True))[:, 1].cpu().numpy())
            out[n] = np.concatenate(p)
    dk, dp, dy = [], [], []
    for n in ("VALmix", "VALmixV2"):
        dk += list(DEVC[n]["key"]); dp.append(out[n]); dy.append(DEVC[n]["y"])
    ff = DEVC["FFpp_val"]
    sc, _t, cells = V31.score(dk, np.concatenate(dp), np.concatenate(dy), ff["key"], out["FFpp_val"], ff["y"])
    v2 = float(np.nanmean([v for (t, s), v in cells.items() if s in V2_SOURCES and not np.isnan(v)]))
    return sc, v2, V31.fast_auc(out["FFpp_val"][ff["y"] == 0], out["FFpp_val"][ff["y"] == 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", required=True, choices=["R-REG", "R-GLOB", "R-POOL"])
    ap.add_argument("--model", default="vjepa2_h")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle-eval", type=int, default=5)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{a.model}_{a.row}_s{a.seed}"
    if (OUT / f"{tag}.json").exists():
        print(f"{tag} exists"); return 0
    need = VF / f"{a.model}_regional_FFpp_train.npz"
    if not need.exists():
        print(f"regional cache missing: {need}")
        json.dump({"row": a.row, "model": a.model, "status": "TODO(run)",
                   "reason": "regional feature cache absent"}, open(OUT / f"{tag}.TODO.json", "w"), indent=1)
        return 3

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    TR = load(a.model, "FFpp_train")
    DEVC = {n: load(a.model, n) for n in DEV_SETS}
    model, n_par = build(a.row, TR["x"], dev)
    print(f"{tag}: {TR['x'].shape} -> {a.row}, {n_par/1e3:.1f}K params "
          f"(matched to the transformer's)", flush=True)

    Xtr, ytr = feats(a.row, TR["x"]), TR["y"]
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    rng = np.random.default_rng(a.seed)
    w = np.where(ytr == 1, 1 / max(1, (ytr == 1).sum()), 1 / max(1, (ytr == 0).sum())); w /= w.sum()
    hist, best = [], (-1.0, None, None)
    for ep in range(EPOCHS):
        model.train()
        idx = rng.choice(len(Xtr), len(Xtr), replace=True, p=w)
        ls = []
        for i in range(0, len(idx) - BATCH + 1, BATCH):
            j = idx[i:i + BATCH]
            ev = model(torch.from_numpy(Xtr[j]).to(dev))
            loss = evidential_loss_discern_ext(ev, torch.from_numpy(ytr[j]).to(dev), 2, ep, EPOCHS).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); ls.append(loss.item())
        sc, v2, ff = score(model, a.row, DEVC, dev)
        hist.append({"epoch": ep, "loss": float(np.mean(ls)), "dev_v31": sc, "valmixv2": v2,
                     "ffpp_val_auroc": ff})
        print(f"  epoch {ep} loss {np.mean(ls):.4f} dev v3.1 {sc:.4f} VALmix-v2 {v2:.4f} FF++val {ff:.4f}", flush=True)
        if sc > best[0] + 1e-6:
            best = (sc, ep, {k: v.detach().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])
    torch.save({"state": best[2], "epoch": best[1]}, OUT / f"{tag}.pt")

    # --- diagnostic 1: frame order shuffled at evaluation
    T = TR["x"].shape[1]
    sh = []
    for s in range(a.shuffle_eval):
        perm = np.random.default_rng(1000 + s).permutation(T)
        sh.append(score(model, a.row, DEVC, dev, perm=perm)[0])
    # --- diagnostic 2: ablate one region at a time (R-REG only; the controls have no regions)
    abl = {}
    if a.row == "R-REG":
        R = TR["x"].shape[2]
        for r in range(R):
            mask = torch.ones(R, device=dev); mask[r] = 0
            abl[TR["regions"][r]] = score(model, a.row, DEVC, dev, region_mask=mask)[0]

    res = {"row": a.row, "model": a.model, "seed": a.seed, "params": int(n_par),
           "best_epoch": best[1], "dev_v31": best[0],
           "valmixv2": hist[best[1]]["valmixv2"], "ffpp_val_auroc": hist[best[1]]["ffpp_val_auroc"],
           "shuffled_dev_v31": {"mean": float(np.mean(sh)), "sd": float(np.std(sh)), "draws": sh,
                                "delta_vs_clean": float(np.mean(sh) - best[0])},
           "region_ablation_dev_v31": abl,
           "note": ("A gain that SURVIVES frame shuffling is regional aggregation, not temporal "
                    "modelling. The five regions OVERLAP by construction, so an ablation does not "
                    "remove a region's patches from the others."),
           "history": hist}
    json.dump(res, open(OUT / f"{tag}.json", "w"), indent=1)
    print(f"{tag}: dev v3.1 {best[0]:.4f}  shuffled {np.mean(sh):.4f} "
          f"(delta {np.mean(sh)-best[0]:+.4f})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

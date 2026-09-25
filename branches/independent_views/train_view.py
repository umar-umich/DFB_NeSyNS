#!/usr/bin/env python3
"""Part 3: train an evidential head on a frozen backbone embedding, as a candidate Appearance view.

    CUDA_VISIBLE_DEVICES=2 <discern_ext python> branches/independent_views/train_view.py \
        --backbone dinov3 --seed 0

Head and recipe follow the brief: Linear to 256, GELU, dropout 0.1, Linear to 2, softplus evidence;
the deployed EDL variant and KL ramp; FF++ c23 train with the deployed Appearance view's augmentation
(none, so the features are extracted clean); frame-averaged to video as the Appearance view does.

SELECTION. "VALmix-v2 checkpoint selection" is read as the deployed convention, which
`analysis/tbiom/temporal/train_branch.py` spells out as "the v3.1 development score on VALmix +
VALmix-v2 + FF++ val". Both that score and the VALmix-v2-only score are reported, so the reading is
visible rather than buried.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(ROOT / "branches/ijepa"), "/data/umar/Repos/DISCERN_Ext"]
import v31 as V31  # noqa: E402
from src.losses.evidential_loss import evidential_loss_discern_ext  # noqa: E402

torch.set_num_threads(4)
FEATS = ROOT / "logs/independent_views/feats"
OUT = ROOT / "analysis/independent_views"
EPOCHS, BATCH, LR, WD, MIN_LR = 20, 128, 1e-4, 0.01, 1e-6
DEV_SETS = ("VALmix", "VALmixV2", "FFpp_val")
V2_SOURCES = {"CDFv3valG", "DF40train", "DS20val", "DS11val", "FFIWval"}


class Head(nn.Module):
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.fc1, self.drop, self.fc2 = nn.Linear(d, 256), nn.Dropout(dropout), nn.Linear(256, 2)

    def forward(self, x):
        return F.softplus(self.fc2(self.drop(F.gelu(self.fc1(x)))))


def load(backbone, dataset):
    f = FEATS / f"{backbone}_{dataset}.npz"
    if not f.exists():
        raise FileNotFoundError(f)
    z = np.load(f, allow_pickle=True)
    vid = np.array([str(v) for v in z["video"]])
    order, inv = np.unique(vid, return_inverse=True)
    first = {}
    for k, v in zip(z["key"], vid):
        first.setdefault(v, str(k))
    return {"x": z["feature"].astype(np.float32), "y": z["label"].astype(np.int64),
            "inv": inv, "order": order, "key": np.array([first[v] for v in order], dtype=object),
            "yv": np.array([int(z["label"][inv == i].max()) for i in range(len(order))])}


@torch.no_grad()
def video_p(head, D, dev, bs=8192):
    head.eval()
    out = []
    for i in range(0, len(D["x"]), bs):
        ev = head(torch.from_numpy(D["x"][i:i + bs]).to(dev))
        a = ev + 1.0; s = a.sum(1, keepdim=True)
        out.append((a / s)[:, 1].float().cpu().numpy())
    p = np.concatenate(out)
    n = len(D["order"])
    cnt = np.bincount(D["inv"], minlength=n).astype(float)
    return np.bincount(D["inv"], weights=p, minlength=n) / cnt      # frame mean, as the APP view does


def scores(head, DEV, dev):
    ff = DEV["FFpp_val"]; fp = video_p(head, ff, dev)
    dk, dp, dy = [], [], []
    for n in ("VALmix", "VALmixV2"):
        D = DEV[n]; dp.append(video_p(head, D, dev)); dk += list(D["key"]); dy.append(D["yv"])
    sc, types, cells = V31.score(dk, np.concatenate(dp), np.concatenate(dy), ff["key"], fp, ff["yv"])
    v2 = float(np.nanmean([v for (t, s), v in cells.items() if s in V2_SOURCES and not np.isnan(v)]))
    ffauc = V31.fast_auc(fp[ff["yv"] == 0], fp[ff["yv"] == 1])
    return sc, v2, ffauc, types


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    TR = load(a.backbone, "FFpp_train")
    DEV = {n: load(a.backbone, n) for n in DEV_SETS}
    d = TR["x"].shape[1]
    print(f"{a.backbone} seed {a.seed}: {len(TR['y'])} train frames, dim {d}", flush=True)

    head = Head(d).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.999))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=MIN_LR)
    y = torch.from_numpy(TR["y"]).to(dev)
    n, hist, best = len(TR["y"]), [], (-1.0, -1)
    for ep in range(EPOCHS):
        head.train(); perm = np.random.permutation(n); losses = []; t0 = time.time()
        for i in range(0, n, BATCH):
            j = perm[i:i + BATCH]
            ev = head(torch.from_numpy(TR["x"][j]).to(dev))
            loss = evidential_loss_discern_ext(ev, y[torch.from_numpy(j).to(dev)], 2, ep, EPOCHS)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(float(loss))
        sch.step()
        sc, v2, ffauc, types = scores(head, DEV, dev)
        hist.append({"epoch": ep, "loss": float(np.mean(losses)), "dev_v31": sc,
                     "valmixv2": v2, "ffpp_val_auroc": ffauc, "types": types})
        print(f"  epoch {ep:2d} loss {np.mean(losses):.4f} dev v3.1 {sc:.4f} "
              f"VALmix-v2 {v2:.4f} FF++val {ffauc:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if sc > best[0] + 1e-6:
            best = (sc, ep)
            torch.save({"state": head.state_dict(), "epoch": ep, "dev_v31": sc,
                        "backbone": a.backbone, "seed": a.seed, "dim": d},
                       OUT / f"view_{a.backbone}_s{a.seed}.pt")
    h = [x for x in hist if x["epoch"] == best[1]][0]
    json.dump({"backbone": a.backbone, "seed": a.seed, "dim": d, "best_epoch": best[1],
               "dev_v31": h["dev_v31"], "valmixv2": h["valmixv2"],
               "ffpp_val_auroc": h["ffpp_val_auroc"], "types": h["types"], "history": hist},
              open(OUT / f"view_{a.backbone}_s{a.seed}.json", "w"), indent=1)
    print(f"{a.backbone} seed {a.seed}: best epoch {best[1]}, dev v3.1 {best[0]:.4f}, "
          f"VALmix-v2 {h['valmixv2']:.4f}, FF++val {h['ffpp_val_auroc']:.4f}")


if __name__ == "__main__":
    main()

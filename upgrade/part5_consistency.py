#!/usr/bin/env python3
"""Part 5: label-preserving consistency. Does a clean/processed agreement term help, beyond the
augmentation it rides on?

    <discern_ext python> upgrade/part5_consistency.py --feats pe_core_l --readout last+mid+half --row C-CONS

PRINCIPLE. Moderate processing must not change whether a video is authentic; it may well change how
much EVIDENCE there is. So the consistency term is applied to the class PROBABILITIES only, never to
evidence magnitude or vacuity -- matching those would tell the model that a degraded frame should be
as informative as a clean one, which is false and would damage exactly the uncertainty signal the
fusion depends on.

ROWS.
  C-AUG   EDL on clean and processed versions together. This is the AUGMENTATION CONTROL, and it is
          the row that matters: most of any gain from C-CONS could come from simply having seen
          processed data, so C-CONS must be compared against this and not against a clean-only head.
  C-CONS  C-AUG plus a symmetric KL between the projected class probabilities of a clean/processed
          pair, weight 0.1.

Keep rule: C-CONS is kept only if it beats C-AUG on dev v3.1 with a bootstrap interval above zero
AND passes Part 4's conditions 2 to 4 (clean FF++ val within 0.01, processed not worse, fused real
FPR within 0.01).
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
from part4_suppression import load_readout_cache, SETTINGS          # noqa: E402
from train_head import Head, DEV_SETS, V2_SOURCES                   # noqa: E402
from src.losses.evidential_loss import evidential_loss_discern_ext  # noqa: E402

torch.set_num_threads(4)
OUT = ROOT / "analysis/upgrade/consistency"
EPOCHS, BATCH, LR, WD = 20, 128, 1e-4, 0.01
KL_WEIGHT = 0.1


def sym_kl(p, q, eps=1e-6):
    """Symmetric KL between two Bernoulli class distributions, on PROBABILITIES only."""
    p = p.clamp(eps, 1 - eps); q = q.clamp(eps, 1 - eps)
    kl = lambda a, b: a * (a / b).log() + (1 - a) * ((1 - a) / (1 - b)).log()
    return (kl(p, q) + kl(q, p)).mean()


def probs(ev):
    a = ev + 1.0
    return (a / a.sum(1, keepdim=True))[:, 1]


def build_pairs(feats, readout):
    """Clean features plus one processed view per frame, aligned by key."""
    clean = load_readout_cache(feats, "FFpp_train", readout, "none")
    pos = {k: i for i, k in enumerate(clean["key"])}
    Xp, Xc, Y = [], [], []
    for s in SETTINGS[1:]:
        c = load_readout_cache(feats, "FFpp_train", readout, s)
        if c is None:
            continue
        idx = np.array([pos[k] for k in c["key"] if k in pos])
        jdx = np.array([j for j, k in enumerate(c["key"]) if k in pos])
        Xc.append(clean["x"][idx]); Xp.append(c["x"][jdx]); Y.append(clean["label"][idx])
    if not Xc:
        raise SystemExit("no processed renders; run Part 4a first")
    return np.concatenate(Xc), np.concatenate(Xp), np.concatenate(Y)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="pe_core_l"); ap.add_argument("--readout", default="last+mid+half")
    ap.add_argument("--row", required=True, choices=["C-AUG", "C-CONS"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{a.feats}_{a.readout.replace('+','-')}_{a.row}_s{a.seed}"
    if (OUT / f"{tag}.json").exists():
        print(f"{tag} exists"); return 0

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    Xc, Xp, Y = build_pairs(a.feats, a.readout)
    print(f"{tag}: {len(Y)} clean/processed pairs, dim {Xc.shape[1]}", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    head = Head(Xc.shape[1]).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)

    # development sets, read through the SAME readout so selection matches the screen
    import train_layer_head as TLH
    DEVC = {n: TLH.load_readout(a.feats, n, a.readout) for n in DEV_SETS}

    def score():
        head.eval()
        out = {}
        with torch.no_grad():
            for n, D in DEVC.items():
                p = []
                for i in range(0, len(D["x"]), 8192):
                    p.append(probs(head(torch.from_numpy(D["x"][i:i + 8192]).to(dev))).cpu().numpy())
                p = np.concatenate(p)
                cnt = np.bincount(D["inv"], minlength=len(D["order"])).astype(float)
                out[n] = np.bincount(D["inv"], weights=p, minlength=len(D["order"])) / cnt
        dk, dp, dy = [], [], []
        for n in ("VALmix", "VALmixV2"):
            dk += list(DEVC[n]["key"]); dp.append(out[n]); dy.append(DEVC[n]["yv"])
        ff = DEVC["FFpp_val"]
        sc, _t, cells = V31.score(dk, np.concatenate(dp), np.concatenate(dy), ff["key"], out["FFpp_val"], ff["yv"])
        v2 = float(np.nanmean([v for (t, s), v in cells.items() if s in V2_SOURCES and not np.isnan(v)]))
        return sc, v2, V31.fast_auc(out["FFpp_val"][ff["yv"] == 0], out["FFpp_val"][ff["yv"] == 1])

    rng = np.random.default_rng(a.seed)
    hist, best = [], (-1.0, None)
    for ep in range(EPOCHS):
        head.train()
        order = rng.permutation(len(Y))
        ls = []
        for i in range(0, len(order) - BATCH + 1, BATCH):
            j = order[i:i + BATCH]
            xc = torch.from_numpy(Xc[j]).to(dev); xp = torch.from_numpy(Xp[j]).to(dev)
            y = torch.from_numpy(Y[j]).to(dev)
            ec, ep_ = head(xc), head(xp)
            # EDL on BOTH views: this is the augmentation, and it is what C-AUG is
            loss = (evidential_loss_discern_ext(ec, y, 2, ep, EPOCHS).mean()
                    + evidential_loss_discern_ext(ep_, y, 2, ep, EPOCHS).mean()) / 2
            if a.row == "C-CONS":
                loss = loss + KL_WEIGHT * sym_kl(probs(ec), probs(ep_))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); ls.append(loss.item())
        sc, v2, ff = score()
        hist.append({"epoch": ep, "loss": float(np.mean(ls)), "dev_v31": sc, "valmixv2": v2,
                     "ffpp_val_auroc": ff})
        print(f"  epoch {ep} loss {np.mean(ls):.4f} dev v3.1 {sc:.4f} VALmix-v2 {v2:.4f} FF++val {ff:.4f}", flush=True)
        if sc > best[0] + 1e-6:
            best = (sc, ep); torch.save({"state": head.state_dict(), "epoch": ep}, OUT / f"{tag}.pt")
    json.dump({"feats": a.feats, "readout": a.readout, "row": a.row, "seed": a.seed,
               "kl_weight": KL_WEIGHT if a.row == "C-CONS" else 0.0,
               "pairs": int(len(Y)), "dim": int(Xc.shape[1]),
               "best_epoch": best[1], "dev_v31": best[0],
               "valmixv2": hist[best[1]]["valmixv2"], "ffpp_val_auroc": hist[best[1]]["ffpp_val_auroc"],
               "history": hist}, open(OUT / f"{tag}.json", "w"), indent=1)
    print(f"{tag}: best epoch {best[1]}, dev v3.1 {best[0]:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

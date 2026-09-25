#!/usr/bin/env python3
"""Train one I-JEPA row's evidential head and select its epoch on the v3.1 development score.

    python branches/ijepa/train_heads.py --row R4
    python branches/ijepa/train_heads.py --row R5 --seed 42
    python branches/ijepa/train_heads.py --dry-run          # shapes and loop, on synthetic data

RECIPE, confirmed from the deployed configuration rather than assumed (see heads.py): digamma
evidential loss with the KL-to-uniform term on non-target evidence annealed as min(1, epoch/20),
AdamW at lr 1e-4, weight decay 0.01, betas (0.9, 0.999), cosine to 1e-6, 20 epochs, batch 128,
FF++ c23 train under the B mixture, natural class ratio.

SELECTION. After every epoch the head scores VALmix, VALmix-v2 and FF++ validation; frame
probabilities are averaged within a video, as for the other spatial views, and the v3.1 score is
computed over the three forgery types. Best epoch wins, ties go to the earliest. Test sets are never
touched here.

WHAT IS FITTED ON TRAINING DATA ONLY. The quality standardiser (mu_k, sigma_k) and the R5 validity
floors tau_k are fitted on REAL FF++ c23 training frames under the B mixture and then frozen, so no
development or test frame contributes to them.
"""
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/data/umar/Repos/DISCERN_Ext")

import data as D            # noqa: E402
import features as FX       # noqa: E402
import v31 as V31           # noqa: E402
from heads import IJepaHead, ROWS, probs  # noqa: E402
from src.losses.evidential_loss import evidential_loss_discern_ext  # noqa: E402

torch.set_num_threads(4)
OUT = Path("/data/umar/Repos/DFB_NeSyNS/analysis/ijepa")
EPOCHS, BATCH, LR, WD, MIN_LR = 20, 128, 1e-4, 0.01, 1e-6
TRAIN_STEM, VAL_STEM = "FFpp_train_B", "FFpp_val"
# G1 established that the standardiser has to be fitted where the evaluation severities are applied,
# or it extrapolates on the JPEG cells. The head rows still TRAIN on FFpp_train_B, which matches
# training conditions; only mu_k, sigma_k and tau_k come from the matched-placement cache.
STD_STEM = "FFpp_train_B224"
DEV_STEMS = ("VALmix", "VALmixV2")


def arrays_for(row):
    """Only load what the row consumes; rbar and sbar are 12.8 KB per frame each."""
    cfg = ROWS[row]
    need = {"q"}
    need.add({"pooled": "pooled", "sbar": "sbar", "rbar": "rbar"}[cfg["vec"]])
    if cfg["scal"] in ("logeps", "a") or row == "R5":
        need.add("eps")
    if row == "R5":
        need.add("rbar")
    return tuple(sorted(need))


def row_inputs(row, d, st, tau):
    """Returns (scalars or None, vectors, extra or None) as float32 numpy."""
    cfg = ROWS[row]
    vec = d[cfg["vec"]].astype(np.float32)
    if cfg["scal"] is None:
        scal = None
    elif cfg["scal"] == "logeps":
        scal = np.log(d["eps"].astype(np.float64)).astype(np.float32)
    elif cfg["scal"] == "a":
        scal = FX.apply_standardizer(d["eps"], d["q"], st)
    elif cfg["scal"] == "snorm":
        scal = np.log(np.linalg.norm(d["sbar"].astype(np.float32), axis=2) + 1e-8).astype(np.float32)
    else:
        raise ValueError(cfg["scal"])
    extra = None
    if cfg["extra"]:
        order, X = FX.video_agreement(d["rbar"], d["keys"], tau)   # video level, 4 columns
        idx = {v: i for i, v in enumerate(order)}
        extra = X[[idx[v] for v in d["video"]]]                    # broadcast back to frames
    return scal, vec, extra


@torch.no_grad()
def frame_probs(model, scal, vec, extra, dev, bs=4096):
    model.eval()
    out = []
    for i in range(0, len(vec), bs):
        s = torch.from_numpy(scal[i:i + bs]).to(dev) if scal is not None else None
        v = torch.from_numpy(vec[i:i + bs]).to(dev)
        e = torch.from_numpy(extra[i:i + bs]).to(dev) if extra is not None else None
        p, _ = probs(model(s, v, e))
        out.append(p[:, 1].float().cpu().numpy())      # p(fake)
    return np.concatenate(out)


def video_scores(model, packs, dev):
    keys, p, y = [], [], []
    for d, s, v, e in packs:
        fp = frame_probs(model, s, v, e, dev)
        order, m = D.video_mean(fp, d["video"])
        first = {}
        for k, vid in zip(d["keys"], d["video"]):
            first.setdefault(vid, k)
        _, yv = D.video_mean(d["y"].astype(float), d["video"])
        keys += [first[o] for o in order]
        p.append(m)
        y.append((yv > 0.5).astype(int))
    return keys, np.concatenate(p), np.concatenate(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", default="R4", choices=list(ROWS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        return dry_run()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    arrays = arrays_for(a.row)
    print(f"row {a.row} ({ROWS[a.row]['desc']}), loading {arrays}", flush=True)

    tr = D.load(TRAIN_STEM, arrays=tuple(set(arrays) | {"eps", "q", "rbar"}))
    real = tr["y"] == 0
    print(f"  train {len(tr['keys'])} frames, {real.sum()} real, {(~real).sum()} fake", flush=True)
    std_stem = STD_STEM if D.shard_files(STD_STEM) else TRAIN_STEM
    sd_ = D.load(std_stem, arrays=("eps", "q", "rbar"))
    sr = sd_["y"] == 0
    st = FX.fit_standardizer(sd_["eps"][sr], sd_["q"][sr])         # real training frames only
    tau = FX.fit_tau(sd_["rbar"][sr])
    print(f"  standardiser and tau from {std_stem} ({int(sr.sum())} real frames)", flush=True)
    del sd_
    json.dump({"sigma": st["sigma"].tolist(), "tau": tau.tolist(),
               "n_real_train_frames": int(real.sum())},
              open(OUT / f"standardizer_{a.row}.json", "w"), indent=1)

    packs = []
    for stem in (VAL_STEM,) + DEV_STEMS:
        d = D.load(stem, arrays=arrays if stem != VAL_STEM else arrays)
        s, v, e = row_inputs(a.row, d, st, tau)
        packs.append((d, s, v, e))
    ffpack, devpacks = packs[0], packs[1:]

    s_tr, v_tr, e_tr = row_inputs(a.row, tr, st, tau)
    y_tr = torch.from_numpy(tr["y"]).to(dev)
    model = IJepaHead(ROWS[a.row]["mode"], ROWS[a.row]["extra"]).to(dev)
    print(f"  trainable {model.n_trainable():,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=MIN_LR)

    n = len(v_tr)
    hist, best = [], (-1.0, -1)
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(n)
        losses = []
        t0 = time.time()
        for i in range(0, n, BATCH):
            j = perm[i:i + BATCH]
            s = torch.from_numpy(s_tr[j]).to(dev) if s_tr is not None else None
            v = torch.from_numpy(v_tr[j]).to(dev)
            e = torch.from_numpy(e_tr[j]).to(dev) if e_tr is not None else None
            ev = model(s, v, e)
            loss = evidential_loss_discern_ext(ev, y_tr[torch.from_numpy(j).to(dev)], 2, ep, EPOCHS)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(float(loss))
        sched.step()

        fk, fp, fy = video_scores(model, [ffpack], dev)
        dk, dp, dy = video_scores(model, devpacks, dev)
        sc, types, cells = V31.score(dk, dp, dy, fk, fp, fy)
        ff_auc = V31.fast_auc(fp[fy == 0], fp[fy == 1])
        hist.append({"epoch": ep, "loss": float(np.mean(losses)), "v31": sc,
                     "types": types, "ffpp_val_auroc": ff_auc})
        print(f"  epoch {ep:2d} loss {np.mean(losses):.4f} v3.1 {sc:.4f} "
              f"FF++val {ff_auc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if sc > best[0] + 1e-6:                      # ties go to the earliest epoch
            best = (sc, ep)
            torch.save({"state": model.state_dict(), "epoch": ep, "v31": sc,
                        "row": a.row, "seed": a.seed}, OUT / f"head_{a.row}.pt")

    json.dump({"row": a.row, "seed": a.seed, "best_epoch": best[1], "best_v31": best[0],
               "rule": V31.RULE_VERSION, "history": hist},
              open(OUT / f"train_{a.row}.json", "w"), indent=1)
    print(f"row {a.row}: best epoch {best[1]} at v3.1 {best[0]:.4f}", flush=True)


def dry_run():
    """Exercise every row's shapes and one optimisation step on synthetic data."""
    rng = np.random.default_rng(0)
    N, V = 512, 1280
    d = {"eps": np.exp(rng.normal(size=(N, 5))).astype(np.float16),
         "rbar": rng.normal(size=(N, 5, V)).astype(np.float16),
         "sbar": rng.normal(size=(N, 5, V)).astype(np.float16),
         "pooled": rng.normal(size=(N, V)).astype(np.float16),
         "q": rng.normal(size=(N, 3)).astype(np.float16),
         "keys": np.array([f"VALmixV2/CDFv3valG/AniTalker/vid{i // 8}/f{i}.png" for i in range(N)]),
         }
    d["video"] = D.videos_of(d["keys"]); d["y"] = D.labels_of(d["keys"])
    st = FX.fit_standardizer(d["eps"].astype(np.float64), d["q"].astype(np.float64))
    tau = FX.fit_tau(d["rbar"])
    for row in ROWS:
        s, v, e = row_inputs(row, d, st, tau)
        m = IJepaHead(ROWS[row]["mode"], ROWS[row]["extra"])
        ts = torch.from_numpy(s) if s is not None else None
        te = torch.from_numpy(e) if e is not None else None
        ev = m(ts, torch.from_numpy(v), te)
        loss = evidential_loss_discern_ext(ev, torch.from_numpy(d["y"]), 2, 0, EPOCHS)
        loss.backward()
        g = sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None)
        assert ev.shape == (N, 2) and math.isfinite(float(loss)) and g > 0
        print(f"  {row:4s} scal={None if s is None else s.shape} vec={v.shape} "
              f"extra={None if e is None else e.shape} loss={float(loss):.4f} grad_sum={g:.2f}")
    print("train_heads.py dry run OK")


if __name__ == "__main__":
    main()

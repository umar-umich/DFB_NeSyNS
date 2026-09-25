#!/usr/bin/env python3
"""Train a sequence head (T0-T3) and run the shuffle check (Amendment 2).

    python branches/ijepa/train_seq.py --head T2 --seed 0
    python branches/ijepa/train_seq.py --dry-run

The frames of a video are ordered by their ORIGINAL frame index, recovered from the stored crop
filename, not by their order in the manifest. Videos are right-padded to the batch's longest
sequence with a validity mask.

Everything about the recipe follows the brief's heads: the digamma evidential loss with the KL term
annealed as min(1, epoch/20), AdamW at 1e-4 with weight decay 0.01, cosine to 1e-6, 20 epochs, and
checkpoint selection on the v3.1 development score. Three seeds per head.

THE SHUFFLE CHECK permutes VALID frames only, keeping each frame's descriptor intact, five
permutations per video, drawn from a per-video seed so that the SAME permutations are used for every
head and for corresponding seeds. Shuffled predictions are never averaged into an ensemble: each
permutation is scored on its own and the five AUROCs are averaged. Dropout is off.

As established in `analysis/ijepa/amendment2_cache_check.json`, cached frames are 0.17-1.48 s apart,
so an order effect here is sensitivity to the ORDER OF SAMPLED FRAMES and is not evidence of motion.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/data/umar/Repos/DISCERN_Ext")

import data as D            # noqa: E402
import features as FX       # noqa: E402
import v31 as V31           # noqa: E402
from cache_check import frame_index          # noqa: E402
from seq_heads import SeqHead, SEQS, SEQ_BACKEND, ORDER_INVARIANT  # noqa: E402
from heads import probs                       # noqa: E402
from src.losses.evidential_loss import evidential_loss_discern_ext  # noqa: E402

torch.set_num_threads(4)
OUT = Path("/data/umar/Repos/DFB_NeSyNS/analysis/ijepa")
EPOCHS, BATCH, LR, WD, MIN_LR = 20, 64, 1e-4, 0.01, 1e-6
N_PERM = 5
TRAIN_STEM, VAL_STEM, DEV_STEMS = "FFpp_train_B", "FFpp_val", ("VALmix", "VALmixV2")


def build_sequences(d, st, tau, use_extra=False):
    """Group cached frames into per-video sequences, ordered by original frame index."""
    a = FX.apply_standardizer(d["eps"], d["q"], st)
    rb = d["rbar"]
    ex = None
    if use_extra:
        C, V = FX.frame_agreement(rb, tau)
        ex = np.concatenate([C, V], axis=1).astype(np.float32)   # per frame, 4 columns

    vids = d["video"]
    idx = np.array([frame_index(k) or 0 for k in d["keys"]])
    order = np.unique(vids)
    pos = {v: i for i, v in enumerate(order)}
    buckets = [[] for _ in order]
    for i, v in enumerate(vids):
        buckets[pos[v]].append(i)
    for b in buckets:
        b.sort(key=lambda i: idx[i])                             # temporal order

    L = max(len(b) for b in buckets)
    N, NB, VD = len(order), a.shape[1], rb.shape[2]
    scal = np.zeros((N, L, NB), np.float32)
    vecs = np.zeros((N, L, NB, VD), np.float16)
    mask = np.zeros((N, L), np.float32)
    extra = np.zeros((N, L, 4), np.float32) if use_extra else None
    y = np.zeros(N, np.int64)
    first_key = np.empty(N, object)
    for j, b in enumerate(buckets):
        n = len(b)
        scal[j, :n] = a[b]
        vecs[j, :n] = rb[b]
        mask[j, :n] = 1.0
        if use_extra:
            extra[j, :n] = ex[b]
        y[j] = int(d["y"][b].max())
        first_key[j] = d["keys"][b[0]]
    return {"scal": scal, "vecs": vecs, "mask": mask, "extra": extra, "y": y,
            "keys": first_key, "video": order, "L": L}


def to_t(x, dev):
    return None if x is None else torch.from_numpy(x).to(dev)


@torch.no_grad()
def predict(model, S, dev, bs=64, perm_seed=None):
    """Video-level p(fake). With `perm_seed`, permute VALID frames only, per video."""
    model.eval()
    out = []
    for i in range(0, len(S["y"]), bs):
        sl = slice(i, i + bs)
        sc = torch.from_numpy(S["scal"][sl]).to(dev)
        vc = torch.from_numpy(S["vecs"][sl]).float().to(dev)
        mk = torch.from_numpy(S["mask"][sl]).to(dev)
        ex = to_t(S["extra"][sl], dev) if S["extra"] is not None else None
        if perm_seed is not None:
            for r in range(sc.shape[0]):
                n = int(mk[r].sum())
                if n > 1:
                    g = np.random.default_rng([perm_seed, i + r])
                    p = torch.from_numpy(g.permutation(n)).to(dev)
                    sc[r, :n], vc[r, :n] = sc[r, :n][p], vc[r, :n][p]
                    if ex is not None:
                        ex[r, :n] = ex[r, :n][p]
        p, _ = probs(model(sc, vc, mk, ex))
        out.append(p[:, 1].float().cpu().numpy())
    return np.concatenate(out)


def v31_of(model, ffS, devS, dev, perm_seed=None):
    fp = predict(model, ffS, dev, perm_seed=perm_seed)
    dk, dp, dy = [], [], []
    for S in devS:
        dp.append(predict(model, S, dev, perm_seed=perm_seed))
        dk += list(S["keys"]); dy.append(S["y"])
    return V31.score(dk, np.concatenate(dp), np.concatenate(dy), ffS["keys"], fp, ffS["y"]), fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="T0", choices=list(SEQS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--r5", action="store_true", help="base row is R5: add the 4 per-frame inputs")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        return dry_run()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    arrays = ("eps", "q", "rbar")
    tr = D.load(TRAIN_STEM, arrays=arrays)
    real = tr["y"] == 0
    st = FX.fit_standardizer(tr["eps"][real], tr["q"][real])
    tau = FX.fit_tau(tr["rbar"][real])

    Str = build_sequences(tr, st, tau, a.r5)
    ffS = build_sequences(D.load(VAL_STEM, arrays=arrays), st, tau, a.r5)
    devS = [build_sequences(D.load(s, arrays=arrays), st, tau, a.r5) for s in DEV_STEMS]
    print(f"{a.head} seed {a.seed}: train {len(Str['y'])} videos (L={Str['L']}), "
          f"backend {SEQ_BACKEND}", flush=True)

    model = SeqHead(a.head, extra=4 if a.r5 else 0).to(dev)
    pc = model.param_counts()
    print(f"  parameters {pc}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.999))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=MIN_LR)

    n, hist, best = len(Str["y"]), [], (-1.0, -1)
    yt = torch.from_numpy(Str["y"]).to(dev)
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(n)
        losses, t0 = [], time.time()
        for i in range(0, n, BATCH):
            j = perm[i:i + BATCH]
            ev = model(torch.from_numpy(Str["scal"][j]).to(dev),
                       torch.from_numpy(Str["vecs"][j]).float().to(dev),
                       torch.from_numpy(Str["mask"][j]).to(dev),
                       to_t(Str["extra"][j], dev) if Str["extra"] is not None else None)
            loss = evidential_loss_discern_ext(ev, yt[torch.from_numpy(j).to(dev)], 2, ep, EPOCHS)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(float(loss))
        sch.step()
        (sc, types, _), fp = v31_of(model, ffS, devS, dev)
        ffauc = V31.fast_auc(fp[ffS["y"] == 0], fp[ffS["y"] == 1])
        hist.append({"epoch": ep, "loss": float(np.mean(losses)), "v31": sc, "types": types,
                     "ffpp_val_auroc": ffauc})
        print(f"  epoch {ep:2d} loss {np.mean(losses):.4f} v3.1 {sc:.4f} FF++val {ffauc:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if sc > best[0] + 1e-6:
            best = (sc, ep)
            torch.save({"state": model.state_dict(), "epoch": ep, "v31": sc,
                        "head": a.head, "seed": a.seed}, OUT / f"seq_{a.head}_s{a.seed}.pt")

    model.load_state_dict(torch.load(OUT / f"seq_{a.head}_s{a.seed}.pt")["state"])
    ordered, _ = v31_of(model, ffS, devS, dev)
    shuf = [v31_of(model, ffS, devS, dev, perm_seed=1000 + p)[0][0] for p in range(N_PERM)]
    t0 = time.time(); predict(model, ffS, dev); infer = (time.time() - t0) / max(len(ffS["y"]), 1)

    res = {"head": a.head, "seed": a.seed, "r5_base": a.r5, "backend": SEQ_BACKEND,
           "params": pc, "best_epoch": best[1], "best_v31": best[0],
           "ordered_v31": ordered[0], "shuffled_v31": shuf,
           "shuffle_drop_v31": float(ordered[0] - np.mean(shuf)),
           "order_invariant_by_design": a.head in ORDER_INVARIANT,
           "inference_seconds_per_video": infer,
           "frames_are_sampled_not_consecutive": True, "history": hist}
    json.dump(res, open(OUT / f"seq_{a.head}_s{a.seed}.json", "w"), indent=1)
    print(f"{a.head} seed {a.seed}: best epoch {best[1]}, v3.1 {best[0]:.4f}, "
          f"shuffle drop {res['shuffle_drop_v31']:+.4f}")


def dry_run():
    rng = np.random.default_rng(0)
    N, NB, V = 640, 5, 1280
    keys = np.array([f"VALmixV2/CDFv3valG/AniTalker/vid{i // 16}/frame_{i % 16:04d}.png"
                     for i in range(N)])
    d = {"eps": np.exp(rng.normal(size=(N, NB))).astype(np.float16),
         "rbar": rng.normal(size=(N, NB, V)).astype(np.float16),
         "q": rng.normal(size=(N, 3)).astype(np.float16), "keys": keys}
    d["video"], d["y"] = D.videos_of(keys), D.labels_of(keys)
    st = FX.fit_standardizer(d["eps"].astype(np.float64), d["q"].astype(np.float64))
    tau = FX.fit_tau(d["rbar"])
    for r5 in (False, True):
        S = build_sequences(d, st, tau, r5)
        assert S["scal"].shape == (N // 16, 16, NB) and S["mask"].all()
        for k in SEQS:
            m = SeqHead(k, extra=4 if r5 else 0)
            p = predict(m, S, "cpu")
            assert p.shape == (N // 16,) and np.isfinite(p).all()
        print(f"  r5={r5}: {S['scal'].shape[0]} videos, L={S['L']}, all four heads predict")
    # frames really are ordered by the original index, not by manifest order
    shuffled = np.random.default_rng(1).permutation(N)
    d2 = {k: (v[shuffled] if hasattr(v, "__getitem__") and len(v) == N else v) for k, v in d.items()}
    S2 = build_sequences(d2, st, tau, False)
    j = list(S2["video"]).index(S["video"][0])
    assert np.allclose(S["scal"][0], S2["scal"][j], atol=1e-6), "ordering must not depend on row order"
    print("  frame ordering is by original index, independent of cache row order")
    print("train_seq.py dry run OK")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Off-manifold with the ENCODER IN THE LOOP: train the LoRA under the OM objective itself.

    CUDA_VISIBLE_DEVICES=1 <discern_ext_timm python> upgrade/adapt_om.py \
        --backbone pe_core_l --recipe LN --seed 0 --readout b16

WHY THIS EXISTS, and why `om_branch.py` does not answer the same question.

`om_branch.py` fits the beta-VAE residual on a CACHED feature. The encoder is not in the graph, so
every quantity it produces is a deterministic function g(f_s) of a frozen embedding:

    f_t = Linear(LayerNorm(f_s)) -> f_c = bVAE(f_t) -> f_a = (f_t - f_c) - proj_{f_t}(f_t - f_c)

If f_s does not carry the evidence, no choice of g recovers it, and the branch cannot do anything
its host view could not already do. That is the whole content of arms 1 and 2: Y landed on top of A
in the per-video error correlation (phi +0.789 dev / +0.786 test) and substituted for it rather than
adding, and the DINOv3 arm earned zero weight. Neither result tests whether OM is CLIP-specific.
They test whether a post-hoc residual on a frozen feature adds anything, and the answer to that was
determined before the runs started.

The DEPLOYED Off-manifold view was not built that way. `bf_audit.B0_CLAIM["off_manifold"]` records
its encoder as `CLIP ViT-L/14 + LoRA(q,v)`, trained warm+gate to epoch 19 WITH the beta-VAE in the
same loop. CLIP's adapters moved to make the residual informative. The honest analogue on another
trunk therefore has to put that trunk's adapters in the same loop, which is what this file does and
what makes it cost GPU-hours instead of minutes.

WHAT CHANGES against `adapt_backbone.py`, and nothing else does. Same backbones, same LoRA/LN
recipes, same source-paired FF++ c23 train, same 20 epochs, same cosine schedule, same v3.1
selection protocol, same full-dev re-score of the selected epoch. Two substitutions:

    head   Head(d)                     ->  TemporalResidualHead(d, operator)
    loss   EDL(head(f))                ->  head.loss(...) = EDL(fused) + 0.5[EDL(t) + EDL(r)]
                                            + 0.7[alignment + 2.0 KL]

SELECTION FOLLOWS THE RESIDUAL ARM, for the reason `om_branch.py` gives: the fused arm mixes in the
plain appearance opinion, so selecting on it picks a checkpoint that is good because its appearance
half is good -- and the appearance half is the branch we already deploy as A. `--select` can ask for
`fused` or `t` explicitly, which is how the appearance arm of this same run is read back out.

READOUT. `--readout b16` taps block 16 rather than the last block, per UMAR's instruction that
PE-Core's best visual embeddings are not at the output. The OM objective then sees the block-16
feature, which is the combination the frozen-feature arms could not express.
"""
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(ROOT / "upgrade"), str(ROOT / "analysis/tbiom/temporal")]

import adapt_backbone as AB                                         # noqa: E402
import v31 as V31                                                   # noqa: E402
from residual_head import TemporalResidualHead, to_pu               # noqa: E402

OUT = ROOT / "analysis/upgrade/om"
CKPT = ROOT / "logs/upgrade/om"


ARMS = ("r", "t", "fused")


def dev_scores(m, spec, head, dev, recipe, cache, readout="last", bs=64):
    """Score ALL THREE arms of the head in ONE pass over the development frames.

    "r" is the off-manifold residual and the branch proper, "t" the appearance half of the same
    head, "fused" their Dempster-Shafer combination. They share an encoder forward, so scoring them
    one at a time -- which is what the first version of this file did -- paid for `embed` three
    times per batch and tripled the per-epoch cost of the only expensive thing here. The per-video
    reduction, the video-id parsing and the v3.1 call are AB's, so a difference between the two
    files is the head and nothing else.
    """
    m.eval(); head.eval()
    out = {a: {} for a in ARMS}
    for name, (X, K, mean, std) in cache.items():
        ps = {a: [] for a in ARMS}
        for i in range(0, len(X), bs):
            x = torch.from_numpy(X[i:i + bs]).permute(0, 3, 1, 2).float().div_(255).to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                z = AB.embed(m, spec, (x - mean) / std, recipe, readout).float()
                o = head(z.float())
            for a in ARMS:
                ps[a].append(to_pu(o[a])[0])
        vid = np.array([AB.video_of_key(k) for k in K])
        order, inv = np.unique(vid, return_inverse=True)
        cnt = np.bincount(inv, minlength=len(order)).astype(float)
        yv = np.array([0 if "real" in str(v).split("/")[-2].lower() else 1 for v in order])
        first = {}
        for k, v in zip(K, vid):
            first.setdefault(v, str(k))
        keys = np.array([first[v] for v in order], dtype=object)
        for a in ARMS:
            pv = np.bincount(inv, weights=np.concatenate(ps[a]), minlength=len(order)) / cnt
            out[a][name] = (keys, pv, yv)
    res = {}
    for a in ARMS:
        dk, dp, dy = [], [], []
        for n in ("VALmix", "VALmixV2"):
            k, p, y = out[a][n]; dk += list(k); dp.append(p); dy.append(y)
        fk, fp, fy = out[a]["FFpp_val"]
        sc, types, cells = V31.score(dk, np.concatenate(dp), np.concatenate(dy), fk, fp, fy)
        v2 = float(np.nanmean([v for (t, s), v in cells.items()
                               if s in AB.V2_SOURCES and not np.isnan(v)]))
        res[a] = {"dev_v31": sc, "valmixv2": v2,
                  "ffpp_val_auroc": V31.fast_auc(fp[fy == 0], fp[fy == 1]), "types": types}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=list(AB.SPECS))
    ap.add_argument("--recipe", required=True, choices=["L", "N", "LN"])
    ap.add_argument("--operator", default="bvae", choices=["bvae", "bvae_real", "mrvae"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=AB.BATCH)
    ap.add_argument("--train-frames", type=int, default=AB.MAX_TRAIN_FRAMES)
    ap.add_argument("--epochs", type=int, default=AB.EPOCHS)
    ap.add_argument("--readout", default="last", help="last | mid | half | bN (e.g. b16)")
    ap.add_argument("--select", default="r", choices=["r", "t", "fused"],
                    help="which arm drives epoch selection; the residual arm is the branch")
    # EVERY failure in this campaign has been in code that only runs AFTER an epoch, so a smoke
    # test that stops at the first batch proves nothing. `--smoke` runs the WHOLE file -- epoch
    # loop, checkpoint save, reload, full-dev re-score, record write -- on a few hundred frames.
    ap.add_argument("--smoke", type=int, default=0, metavar="N",
                    help="end-to-end rehearsal on N train frames and N//2 dev frames per set")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); CKPT.mkdir(parents=True, exist_ok=True)
    if a.smoke:
        a.epochs, a.train_frames = max(1, a.epochs if a.epochs < 3 else 2), a.smoke

    tag = ("smoke_" if a.smoke else "") + f"om_{a.backbone}_{a.recipe}_{a.operator}_s{a.seed}"
    if a.readout != "last":
        tag += f"_{a.readout.replace('+', '-')}"
    if a.select != "r":
        tag += f"_sel{a.select}"
    if (AB.LORA_R, AB.LORA_ALPHA, AB.LORA_DROPOUT) != (8, 16.0, 0.1):
        tag += f"_r{AB.LORA_R}a{AB.LORA_ALPHA:g}d{AB.LORA_DROPOUT:g}"
    if (OUT / f"{tag}.json").exists():
        print(f"{tag} exists"); return 0

    dev = "cuda"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    m, mean, std, spec, params, fused = AB.build(a.backbone, dev, a.recipe)
    depth_idx = AB.attach_depth_hooks(m, spec, a.readout)
    print(f"{tag}: input {spec['size']}, readout {a.readout} {depth_idx}, "
          f"trainable {sum(p.numel() for p in params)/1e6:.2f} M, select on '{a.select}'"
          + (" (fused qkv: key adapted too)" if fused else ""), flush=True)

    t0 = time.time()
    Xtr, Ktr = AB.frames_of("FFpp_train", spec["size"], a.train_frames, a.seed)
    ytr = np.array([0 if "real" in str(k).split("/")[-3].lower() else 1 for k in Ktr])
    cache, full = {}, {}
    per_set = AB.MAX_DEV_FRAMES // len(AB.DEV_SETS)
    for n in AB.DEV_SETS:
        X, K = AB.frames_of(n, spec["size"])
        if a.smoke:
            # The smoke run must still exercise BOTH scoring passes, so `full` is subsampled too --
            # at a different budget, so a bug that confuses the two caches still shows up.
            X, K = AB.subsample(X, K, a.smoke, a.seed)
            per_set = max(16, a.smoke // 2)
        full[n] = (X, K, mean, std)
        Xs, Ks = AB.subsample(X, K, per_set, a.seed)
        cache[n] = (Xs, Ks, mean, std)
    print(f"  data ready in {(time.time()-t0)/60:.1f} min: train {len(Xtr)}, selection on "
          + ", ".join(f"{n} {len(cache[n][0])}/{len(full[n][0])}" for n in AB.DEV_SETS), flush=True)

    d = AB.embed(m, spec, torch.zeros(1, 3, spec["size"], spec["size"], device=dev),
                 a.recipe, a.readout).shape[-1]
    head = TemporalResidualHead(d, a.operator).to(dev)
    opt = torch.optim.AdamW([{"params": params, "lr": AB.LR_ADAPT},
                             {"params": head.parameters(), "lr": AB.LR_HEAD}], weight_decay=AB.WD)
    steps = a.epochs * max(1, len(Xtr) // a.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: max(1e-6 / AB.LR_ADAPT, 0.5 * (1 + math.cos(math.pi * s / steps))))
    w = np.where(ytr == 1, 1.0 / max(1, (ytr == 1).sum()), 1.0 / max(1, (ytr == 0).sum()))
    w = w / w.sum()
    rng = np.random.default_rng(a.seed)

    hist, best = [], (-1.0, None)
    for ep in range(a.epochs):
        m.train(); head.train()
        idx = rng.choice(len(Xtr), size=len(Xtr), replace=True, p=w)
        losses = []
        for i in range(0, len(idx) - a.batch + 1, a.batch):
            j = idx[i:i + a.batch]
            x = torch.from_numpy(Xtr[j]).permute(0, 3, 1, 2).float().div_(255).to(dev)
            y = torch.from_numpy(ytr[j]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = AB.embed(m, spec, (x - mean) / std, a.recipe, a.readout).float()
            # The head runs in fp32: the beta-VAE's KL and the orthogonal projection both divide by
            # small quantities, and bf16 there produced non-finite losses in the frozen-feature arm.
            loss = head.loss(head(z.float()), y, ep, a.epochs)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            losses.append(loss.item())
        row = {"epoch": ep, "loss": float(np.mean(losses)), "minutes": (time.time() - t0) / 60}
        row.update(dev_scores(m, spec, head, dev, a.recipe, cache, a.readout))
        hist.append(row)
        print(f"  epoch {ep:2d} loss {row['loss']:.4f} residual {row['r']['dev_v31']:.4f} "
              f"appearance {row['t']['dev_v31']:.4f} fused {row['fused']['dev_v31']:.4f} "
              f"({row['minutes']:.0f} min)", flush=True)
        if row[a.select]["dev_v31"] > best[0] + 1e-6:
            best = (row[a.select]["dev_v31"], ep)
            torch.save({"epoch": ep, "head": head.state_dict(), "operator": a.operator,
                        "readout": a.readout, "in_dim": d,
                        "adapt": {k: v.detach().cpu() for k, v in m.state_dict().items()
                                  if any(t in k for t in (".a.", ".b.", "norm", "ln"))}},
                       CKPT / f"{tag}.pt")

    ck = torch.load(CKPT / f"{tag}.pt", map_location=dev)
    head.load_state_dict(ck["head"])
    m.load_state_dict({**m.state_dict(), **{k: v.to(dev) for k, v in ck["adapt"].items()}},
                      strict=False)
    final = dev_scores(m, spec, head, dev, a.recipe, full, a.readout)
    print(f"  selected epoch {ck['epoch']} re-scored on FULL dev: residual "
          f"{final['r']['dev_v31']:.4f}  appearance {final['t']['dev_v31']:.4f}  "
          f"fused {final['fused']['dev_v31']:.4f}", flush=True)

    rec = {"tag": tag, "backbone": a.backbone, "recipe": a.recipe, "operator": a.operator,
           "readout": a.readout, "depth_idx": depth_idx, "seed": a.seed, "select": a.select,
           "in_dim": d, "epochs": a.epochs, "train_frames": len(Xtr),
           "lora": {"r": AB.LORA_R, "alpha": AB.LORA_ALPHA, "dropout": AB.LORA_DROPOUT},
           "encoder_in_loop": True, "best_epoch": ck["epoch"], "history": hist, "full_dev": final}
    (OUT / f"{tag}.json").write_text(json.dumps(rec, indent=2))
    print(f"-> {OUT / f'{tag}.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

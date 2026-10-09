#!/usr/bin/env python3
"""Off-manifold branches on encoders other than CLIP.

    CUDA_VISIBLE_DEVICES=1 <discern_ext python> upgrade/branchfusion/om_branch.py \
        --backbone pe_core_lLN --operator bvae --seed 0 --train --export

THE QUESTION. The deployed Off-manifold view is built on CLIP and is one of the strongest branches
we have -- the oracle's choice on CDFv2, DFDC and DFDM. Nothing says the construction is CLIP-
specific, so it is worth asking what it does on the other trunks.

WHAT IS ALREADY ANSWERED, so this does not re-run it. `tbiom/reports/EXP4B_TEMPORAL_RESIDUAL.md`
applied this exact residual family to X-CLIP, TimeSformer, VideoMAE and V-JEPA 2 TEMPORAL features.
V-JEPA 2 went 0.7129 plain -> 0.7151 bvae -> 0.7166 bvae_real -> 0.7201 mrvae: nothing. UMAR's
decision of 2026-09-17 closed residual operators as reported negatives and lists "run more
residual-operator variants" under Do-not. The one thing that result does NOT settle is whether the
failure was the operator or the FROZEN PROBE it was measured on, which topped out at 0.71 while the
deployed temporal branch scores far higher. A video-feature arm here must say which it is testing.

WHAT THIS ADDS. The IMAGE trunks, which EXP4B never touched:

  pe_core_lLN  the APP trunk. PREDICTED TO FAIL, and that is why it is worth running. A and S share
               this trunk and the fusion fit zeroes S in 4-5 of 5 folds whenever A is present, with
               the subset macro moving exactly 0.0000 in 11 of 13 cases. OM-on-PE shares it the same
               way. The deployed OM works partly BECAUSE it sits on CLIP while APP moved to PE: even
               across two different backbones their errors correlate at phi ~ +0.41 to +0.46, by far
               the highest pair in the system (APP-T is ~ +0.12). If this arm earns non-zero weight
               beside A anyway, the shared-trunk explanation is WRONG and that is worth knowing.
  dinov3       the mechanism test, and the reason to run the pair. OM's story is that the beta-VAE
               captures what a caption would name and the orthogonal residual keeps what text cannot
               describe. DINOv3 is the only text-free trunk available -- self-distillation, no
               captions, 4 register tokens besides CLS. If OM works here too, the image-text story is
               wrong and OM is a generic residual detector. If it works on CLIP and fails on DINOv3,
               the contrastive image-text objective IS the mechanism.

HOW IT IS JUDGED. Standalone dev v3.1 is reported but is NOT the criterion. The criterion is whether
the branch earns non-zero weight beside the others across folds -- the independent-views study found
a view that improved standalone while costing the fusion, and S improved nothing while looking
reasonable alone. Run `subsets.py --screen` with the branch registered and read `folds_weight_zero`.

The transform itself is `analysis/tbiom/temporal/residual_head.py`, unchanged: it already takes an
arbitrary feature dimension, so no new maths is introduced here and the CLIP recipe's constants
(64-d projection, 32-d latent, beta 2.0, the alignment and auxiliary weights) are inherited intact.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                      # noqa: E402

sys.path[:0] = [str(C.ROOT / "analysis/tbiom/temporal"), str(C.ROOT / "branches/ijepa"),
                "/data/umar/Repos/DISCERN_Ext"]
import residual_head as RH                              # noqa: E402
import v31 as V31                                       # noqa: E402

C.thread_caps()

OUT = C.OUT / "om_branches"
CKPT = C.ROOT / "logs/branchfusion/om"
EPOCHS, BATCH, LR, WD, MIN_LR = 20, 128, 1e-4, 0.01, 1e-6
V2_SOURCES = {"CDFv3valG", "DF40train", "DS20val", "DS11val", "FFIWval"}
for _d in (OUT, CKPT):
    _d.mkdir(parents=True, exist_ok=True)


def tag(backbone: str, operator: str, seed: int) -> str:
    return f"OM-{backbone}-{operator}_s{seed}"


@torch.no_grad()
def video_pu(model, D, dev, which="fused", bs=8192):
    """Frame opinions -> frame-mean video opinion, the reduction every other image branch uses."""
    model.eval()
    ps, us = [], []
    for i in range(0, len(D["x"]), bs):
        out = model(torch.from_numpy(D["x"][i:i + bs]).to(dev))
        p, u = RH.to_pu(out[which])
        ps.append(p); us.append(u)
    return C.video_mean(D, np.concatenate(ps)), C.video_mean(D, np.concatenate(us))


def dev_scores(model, DEV, dev, which="fused"):
    """The deployed selection convention: v3.1 over VALmix + VALmix-v2 + FF++ val."""
    ff = DEV["FFpp_val"]
    fp, _ = video_pu(model, ff, dev, which)
    dk, dp, dy = [], [], []
    for n in ("VALmix", "VALmixV2"):
        D = DEV[n]
        p, _ = video_pu(model, D, dev, which)
        dp.append(p); dk += list(D["key"]); dy.append(D["yv"])
    sc, types, cells = V31.score(dk, np.concatenate(dp), np.concatenate(dy), ff["key"], fp, ff["yv"])
    v2 = float(np.nanmean([v for (t, s), v in cells.items() if s in V2_SOURCES and not np.isnan(v)]))
    return sc, v2, float(V31.fast_auc(fp[ff["yv"] == 0], fp[ff["yv"] == 1])), types


def train(backbone: str, operator: str, seed: int, epochs: int = EPOCHS) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    TR = C.load_frames(backbone, "FFpp_train")
    DEV = {n: C.load_frames(backbone, n) for n in C.DEV_SETS}
    d = TR["x"].shape[1]
    t = tag(backbone, operator, seed)
    print(f"{t}: {len(TR['y'])} train frames, dim {d}", flush=True)

    model = RH.TemporalResidualHead(d, operator).to(dev)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=MIN_LR)
    y = torch.from_numpy(TR["y"]).to(dev)
    n, hist, best = len(TR["y"]), [], (-1.0, -1)
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(n); losses = []; t0 = time.time()
        for i in range(0, n, BATCH):
            j = perm[i:i + BATCH]
            out = model(torch.from_numpy(TR["x"][j]).to(dev))
            loss = model.loss(out, y[torch.from_numpy(j).to(dev)], ep, epochs)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(float(loss))
        sch.step()
        # The RESIDUAL arm is what "off-manifold" means, so selection follows it, not the fused
        # head: the fused head mixes in the plain appearance opinion and would select a checkpoint
        # that is good because its appearance half is good -- which is the branch we already have.
        sc, v2, ffauc, types = dev_scores(model, DEV, dev, "r")
        fsc, _, _, _ = dev_scores(model, DEV, dev, "fused")
        tsc, _, _, _ = dev_scores(model, DEV, dev, "t")
        hist.append({"epoch": ep, "loss": float(np.mean(losses)), "dev_v31_residual": sc,
                     "dev_v31_fused": fsc, "dev_v31_appearance": tsc, "valmixv2": v2,
                     "ffpp_val_auroc": ffauc, "types": types})
        print(f"  epoch {ep:2d} loss {np.mean(losses):.4f} residual {sc:.4f} "
              f"fused {fsc:.4f} appearance {tsc:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if sc > best[0] + 1e-6:
            best = (sc, ep)
            torch.save({"state": model.state_dict(), "epoch": ep, "dim": d, "operator": operator,
                        "backbone": backbone, "seed": seed, "dev_v31_residual": sc},
                       CKPT / f"{t}.pt")
    h = [x for x in hist if x["epoch"] == best[1]][0]
    rec = {"tag": t, "backbone": backbone, "operator": operator, "seed": seed, "dim": d,
           "params": int(n_par), "best_epoch": best[1], "selected_on": "residual arm",
           **{k: h[k] for k in ("dev_v31_residual", "dev_v31_fused", "dev_v31_appearance",
                                "valmixv2", "ffpp_val_auroc", "types")},
           "history": hist}
    json.dump(rec, open(OUT / f"{t}.json", "w"), indent=1, default=str)
    print(f"{t}: best epoch {best[1]}, residual v3.1 {best[0]:.4f}, "
          f"fused {h['dev_v31_fused']:.4f}, appearance-arm {h['dev_v31_appearance']:.4f}")
    return rec


def export(backbone: str, operator: str, seed: int, which: str = "r") -> dict:
    """Write v,y,p,u per dataset so the branch ledger can read this as an `export` branch."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t = tag(backbone, operator, seed)
    ck = torch.load(CKPT / f"{t}.pt", map_location=dev)
    model = RH.TemporalResidualHead(ck["dim"], ck["operator"]).to(dev)
    model.load_state_dict(ck["state"]); model.eval()
    d = C.BRANCH_OUT / t
    d.mkdir(parents=True, exist_ok=True)
    done = {}
    for ds in list(C.DEV_SETS) + list(C.TEST_SETS):
        try:
            D = C.load_frames(backbone, ds)
        except FileNotFoundError:
            print(f"  {ds}: no {backbone} cache -- SKIPPED", flush=True); continue
        p, u = video_pu(model, D, dev, which)
        pd.DataFrame({"v": [str(x) for x in D["order"]], "y": D["yv"], "p": p, "u": u}) \
            .to_csv(d / f"{ds}.csv", index=False)
        done[ds] = int(len(D["order"]))
        print(f"  {ds}: {len(D['order'])} videos", flush=True)
    json.dump({"tag": t, "arm": which, "datasets": done, "dir": str(d)},
              open(OUT / f"{t}_export.json", "w"), indent=1)
    missing = [s for s in list(C.DEV_SETS) + list(C.TEST_SETS) if s not in done]
    if missing:
        print(f"INCOMPLETE -- no features for {missing}. The branch cannot enter a screen that "
              f"requires full coverage until those caches exist.", flush=True)
    return {"tag": t, "datasets": done, "missing": missing}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, help="feature-cache prefix, e.g. pe_core_lLN")
    ap.add_argument("--operator", default="bvae", choices=["bvae", "bvae_real", "mrvae"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--arm", default="r", choices=["r", "fused", "t"],
                    help="which opinion to export: r = the off-manifold residual (the point)")
    a = ap.parse_args()
    if a.train:
        train(a.backbone, a.operator, a.seed, a.epochs)
    if a.export:
        export(a.backbone, a.operator, a.seed, a.arm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

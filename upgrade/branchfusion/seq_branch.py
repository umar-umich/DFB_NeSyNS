#!/usr/bin/env python3
"""Section 2.2: branch S, a PE-Core sequence transformer, and its architecture controls.

    CUDA_VISIBLE_DEVICES=1 <discern_ext_timm python> upgrade/branchfusion/seq_branch.py \
        --variant adapt --seed 0
    ... --variant frozen --seed 0
    ... --variant adapt --pool mean          # section 7 control
    ... --variant adapt --no-pos             # section 7 control
    ... --export                              # write opinions for every dev and test set

S IS A PARALLEL BRANCH WITH ITS OWN PREDICTION. Nothing here inserts PE vectors into the V-JEPA
network or averages them with V-JEPA features inside one branch, because that would confound the
A/S/V comparison (section 2.2).

ARCHITECTURE, exactly as the brief fixes it: PE frame embeddings -> Linear to 256 -> timestamp-based
temporal positional encoding -> two PRE-NORM transformer encoder blocks (4 heads, feed-forward width
1024, dropout 0.1) -> masked temporal mean -> two-output head. Padded frames are masked in the
attention and excluded from the mean.

THE TIME ENCODING USES REAL FRAME POSITIONS. The feature caches key every frame as
`.../009.png`, so the stored frames carry their true index in the source video -- typically 0, 9, 19,
... at a spacing of about 10. Encoding position by slot number would tell the model a constant frame
rate that does not hold; `common.load_frames` parses the true index and `TemporalPE` encodes that.

TWO INPUT VARIANTS, because of a saturation risk the brief names:

  adapt   A's adapted PE frame embeddings (`pe_core_lLN`). These reach ~0.995 on FF++ validation, so
          a temporal module on top may be handed an already-solved problem. `saturation_report`
          measures the fraction of FF++ TRAIN videos A already places beyond p > 0.99 on the correct
          side, and that number is printed before training starts.
  frozen  Frozen PE-Core-L block-16 patch-mean features, the strongest frozen PE readout (dev 0.8019)
          and not fitted to FF++ labels.

Both train with identical recipes. The better of the two on development enters the subset search as
S, and BOTH are reported. Two-fold cross-fitted adaptation is explicitly a queued follow-up, not part
of this campaign.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                      # noqa: E402

sys.path[:0] = [str(C.ROOT / "upgrade"), str(C.ROOT / "branches/ijepa"), str(C.EXT)]
import v31 as V31                                       # noqa: E402
from src.losses.evidential_loss import evidential_loss_discern_ext   # noqa: E402

C.thread_caps()

#: Section 2.1's shared recipe. Held identical across variants so a difference is the variant.
D_MODEL, N_HEADS, FF, DROPOUT, N_BLOCKS = 256, 4, 1024, 0.1, 2
EPOCHS, BATCH, LR, WD, MIN_LR = 20, 32, 2e-4, 0.01, 1e-6
V2_SOURCES = {"CDFv3valG", "DF40train", "DS20val", "DS11val", "FFIWval"}

#: Which cache each variant reads. `frozen` needs `extract_patch.py --depth mid` to have run.
VARIANTS = {
    "adapt": {"backbone": "pe_core_lLN", "patch_dir": None, "suffix": "",
              "note": "A's adapted PE-Core-L/336 LoRA+LayerNorm frame embeddings"},
    "frozen": {"backbone": "MEANPOOL_pe_core_l-mid", "patch_dir": True, "suffix": "",
               "note": "frozen PE-Core-L block-16 patch mean, not fitted to FF++ labels"},
    # S-dense / P-dense: the SAME readouts on the stride-1 consecutive crops in
    # /data/umar/Datasets/tbiom_dense, which already exist for FF++ train and the development pool.
    # Section 2.2 scopes the dense pilot to exactly those partitions, so these variants cannot be
    # scored on the test sets and are reported as a development-only pilot.
    "dense": {"backbone": "pe_core_l", "patch_dir": None, "suffix": "-dense",
              "note": "frozen PE-Core-L/336 global readout on stride-1 consecutive frames"},
    "dense-mid": {"backbone": "pe_core_l-mid", "patch_dir": None, "suffix": "-dense",
                  "note": "frozen PE-Core-L block-16 patch mean on stride-1 consecutive frames"},
}
#: Which partitions a variant can be scored on. The dense crops cover development only.
DENSE_SETS = ("FFpp_train", *C.DEV_SETS)


# ----------------------------------------------------------------------------- architecture
class TemporalPE(nn.Module):
    """Sinusoidal encoding of RELATIVE time within the clip, from the real frame indices.

    Two corrections over the obvious implementation, both of which matter here:

    1. **Relative, not absolute.** The stored frame indices are positions in the source video, so a
       clip starting at frame 214 and one starting at frame 3 would get completely different
       encodings for the same motion. The per-clip minimum is subtracted first.
    2. **A data-derived scale.** Sparse caches are ~15 source frames apart and the dense caches are
       1 apart. With a fixed scale the dense case would put all 32 frames inside a tiny fraction of
       the slowest period and the encoding would be nearly constant -- silently reproducing the
       "no positional information" condition on exactly the variant meant to test it. `scale` is set
       from the median stride of the data the branch is trained on, so one unit is one sampled step.
    """

    def __init__(self, d_model: int = D_MODEL, scale: float = 10.0):
        super().__init__()
        self.d_model = d_model
        self.register_buffer("scale", torch.tensor(float(scale)), persistent=True)
        inv = torch.exp(-math.log(10000.0) * torch.arange(0, d_model, 2).float() / d_model)
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, tpos: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """`tpos` (B, T) real frame indices -> (B, T, d_model)."""
        if mask is not None:
            big = torch.where(mask, tpos, torch.full_like(tpos, float("inf")))
            t0 = big.min(dim=1, keepdim=True).values
            t0 = torch.where(torch.isfinite(t0), t0, torch.zeros_like(t0))
        else:
            t0 = tpos.min(dim=1, keepdim=True).values
        x = ((tpos - t0) / self.scale.clamp_min(1e-6)).unsqueeze(-1) * self.inv_freq
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)[..., :self.d_model]


class SeqBranch(nn.Module):
    """Branch S and its two section-7 pooling controls, in one module so the budget is matched.

    `pool`:
      "attn" the specified architecture: two pre-norm transformer blocks then a masked mean.
      "mean" the transformer removed entirely -- projection, then a masked mean. The control that
             asks whether attention across frames does anything at all.
      "mlp"  a PARAMETER-MATCHED per-frame MLP then a masked mean. Matched because "mean" has far
             fewer parameters than "attn", so a loss under "mean" conflates capacity with temporal
             modelling; "mlp" spends the transformer's parameters without letting frames talk.
    """

    def __init__(self, d_in: int, pool: str = "attn", use_pos: bool = True,
                 d_model: int = D_MODEL, n_blocks: int = N_BLOCKS, n_heads: int = N_HEADS,
                 ff: int = FF, dropout: float = DROPOUT, pe_scale: float = 10.0,
                 order_head: bool = False):
        super().__init__()
        self.pool, self.use_pos, self.d_model = pool, use_pos, d_model
        self.proj = nn.Linear(d_in, d_model)
        self.pos = TemporalPE(d_model, pe_scale) if use_pos else None
        # The order-discrimination head (section 7 intervention, not part of the specified S). It
        # reads the pooled clip and must say whether the frames arrived in order. Its only purpose is
        # to put gradient pressure on the temporal stage; it is discarded at inference.
        self.order = nn.Linear(d_model, 2) if order_head else None
        self.drop = nn.Dropout(dropout)
        if pool == "attn":
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=ff, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, num_layers=n_blocks)
            self.norm = nn.LayerNorm(d_model)
        elif pool == "mlp":
            # Match the transformer's parameter count as closely as one hidden width allows: a
            # per-frame MLP of width `h` costs 2*d*h + h + d against the encoder's measured total.
            tgt = _transformer_params(d_model, n_heads, ff, n_blocks)
            h = max(1, int(round((tgt - d_model) / (2 * d_model + 1))))
            self.enc = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, h), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(h, d_model))
            self.norm = nn.LayerNorm(d_model)
            self.matched_hidden = h
        elif pool == "mean":
            self.enc = None
            self.norm = nn.LayerNorm(d_model)
        else:
            raise ValueError(pool)
        self.out = nn.Linear(d_model, 2)

    def forward(self, x, mask, tpos):
        """`x` (B, T, d_in), `mask` (B, T) True on valid frames, `tpos` (B, T) real frame indices."""
        h = self.proj(x)
        if self.pos is not None:
            h = h + self.pos(tpos, mask)
        h = self.drop(h)
        if self.pool == "attn":
            # `src_key_padding_mask` wants True where a position must be IGNORED.
            h = self.enc(h, src_key_padding_mask=~mask)
        elif self.pool == "mlp":
            h = h + self.enc(h)
        h = self.norm(h)
        m = mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp_min(1.0)      # masked temporal mean
        return F.softplus(self.out(pooled)), pooled

    def order_logits(self, pooled):
        if self.order is None:
            raise RuntimeError("this branch was built without an order head")
        return self.order(pooled)

    def param_report(self) -> dict:
        """Section 2.1: record the native-to-256 projection separately, because equal head width is
        not equal parameter count."""
        proj = sum(p.numel() for p in self.proj.parameters())
        enc = sum(p.numel() for p in self.enc.parameters()) if self.enc is not None else 0
        rest = sum(p.numel() for p in self.parameters()) - proj - enc
        return {"projection_native_to_256": proj, "temporal": enc, "head_and_norms": rest,
                "total": proj + enc + rest, "pool": self.pool, "pos_encoding": self.use_pos,
                "matched_hidden": getattr(self, "matched_hidden", None)}


def _transformer_params(d, nhead, ff, n_blocks) -> int:
    layer = nn.TransformerEncoderLayer(d_model=d, nhead=nhead, dim_feedforward=ff, dropout=0.0,
                                       activation="gelu", batch_first=True, norm_first=True)
    return sum(p.numel() for p in layer.parameters()) * n_blocks


# ----------------------------------------------------------------------------- data
def load_variant(variant: str, dataset: str):
    spec = VARIANTS[variant]
    pd_ = C.PATCH if spec["patch_dir"] else None
    return C.load_frames(spec["backbone"], f"{dataset}{spec.get('suffix', '')}", patch_dir=pd_)


def median_stride(D: dict) -> float:
    """Median spacing between consecutive stored frames of a video, in source frames.

    This sets the positional encoding's scale. Measured rather than assumed: the sparse caches come
    out at 10-22 and the dense ones at 1, and using one scale for both would flatten the dense
    encoding to nearly constant.
    """
    out = []
    n = len(D["order"])
    by = [[] for _ in range(n)]
    for j, i in enumerate(D["inv"]):
        by[i].append(D["tidx"][j])
    for v in by:
        if len(v) > 1:
            d = np.diff(np.sort(np.asarray(v)))
            if len(d):
                out.append(float(np.median(d)))
    return float(np.median(out)) if out else 1.0


def clips(variant: str, dataset: str):
    D = load_variant(variant, dataset)
    seq, mask, tpos, yv, order, lens = C.clips_of(D)
    return {"seq": seq, "mask": mask, "tpos": tpos, "yv": yv, "order": order,
            "key": D["key"], "lens": lens, "dim": D["dim"],
            "decoded_frames": int(len(D["y"])),
            "unique_frames": int(len(np.unique([str(k) for k in D["key"]]))),
            "median_stride": median_stride(D), "n_videos": len(order)}


def saturation_report(dataset: str = "FFpp_train") -> dict:
    """Fraction of training videos A already places beyond p > 0.99 on the correct side.

    Section 2.2 requires this number BEFORE S-adapt is trained, because a temporal module stacked on
    an already-solved representation can only learn noise, and a weak S would then be a statement
    about the training signal rather than about temporal information.
    """
    import branches as B
    try:
        t = B.branch_opinions("A", dataset)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    p, y = t.p.to_numpy(), t.y.to_numpy()
    conf = np.where(y == 1, p, 1 - p)
    return {"dataset": dataset, "n_videos": int(len(y)),
            "frac_p_gt_0.99_correct_side": float((conf > 0.99).mean()),
            "frac_p_gt_0.95_correct_side": float((conf > 0.95).mean()),
            "frac_p_gt_0.90_correct_side": float((conf > 0.90).mean()),
            "mean_correct_side_confidence": float(conf.mean()),
            "auroc": C.auroc(y, p),
            "reading": ("a high fraction means the temporal module is being trained where the "
                        "appearance branch has no errors left to explain, which is the saturation "
                        "section 2.2 warns about")}


# ----------------------------------------------------------------------------- scoring
@torch.no_grad()
def predict(model, Dc, dev, bs: int = 64):
    model.eval()
    ps, us = [], []
    for i in range(0, Dc["n_videos"], bs):
        x = torch.from_numpy(Dc["seq"][i:i + bs]).to(dev)
        m = torch.from_numpy(Dc["mask"][i:i + bs]).to(dev)
        tp = torch.from_numpy(Dc["tpos"][i:i + bs]).to(dev)
        ev, _ = model(x, m, tp)
        a = ev + 1.0
        s = a.sum(1, keepdim=True)
        ps.append((a / s)[:, 1].float().cpu().numpy())      # EDL order: index 1 is fake
        us.append((C.K / s.squeeze(1)).float().cpu().numpy())
    return np.concatenate(ps), np.concatenate(us)


def dev_score(model, DEV, dev):
    """The v3.1 development objective plus VALmix-v2 and FF++ val, as every other branch reports."""
    ff = DEV["FFpp_val"]
    fp, _ = predict(model, ff, dev)
    dk, dp, dy = [], [], []
    for n in ("VALmix", "VALmixV2"):
        Dc = DEV[n]
        p, _u = predict(model, Dc, dev)
        dp.append(p)
        dk += list(Dc["key"])
        dy.append(Dc["yv"])
    sc, types, cells = V31.score(dk, np.concatenate(dp), np.concatenate(dy),
                                 ff["key"], fp, ff["yv"])
    v2 = float(np.nanmean([v for (t, s), v in cells.items()
                           if s in V2_SOURCES and not np.isnan(v)]))
    return sc, v2, float(V31.fast_auc(fp[ff["yv"] == 0], fp[ff["yv"] == 1])), types


# ----------------------------------------------------------------------------- training
def train(variant: str, seed: int, pool: str = "attn", use_pos: bool = True,
          epochs: int = EPOCHS, tag: str | None = None, order_loss: float = 0.0) -> dict:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)
    tag = tag or model_tag(variant, pool, use_pos, seed, order_loss)

    TR = clips(variant, "FFpp_train")
    DEV = {n: clips(variant, n) for n in C.DEV_SETS}
    scale = max(TR["median_stride"], 1.0)
    model = SeqBranch(TR["dim"], pool=pool, use_pos=use_pos, pe_scale=scale,
                      order_head=order_loss > 0).to(dev)
    print(f"  positional scale set from the data: median stride {TR['median_stride']:.1f} "
          f"source frames", flush=True)
    pr = model.param_report()
    print(f"{tag}: {TR['n_videos']} train videos, T={TR['seq'].shape[1]}, dim {TR['dim']}, "
          f"params {pr}", flush=True)
    print(f"  frames: {TR['decoded_frames']} decoded, "
          f"clip length min/max {int(TR['lens'].min())}/{int(TR['lens'].max())}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=MIN_LR)
    y = torch.from_numpy(TR["yv"].astype(np.int64)).to(dev)
    n = TR["n_videos"]
    hist, best = [], (-1.0, -1)
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(n)
        losses, t0 = [], time.time()
        for i in range(0, n, BATCH):
            j = perm[i:i + BATCH]
            x = torch.from_numpy(TR["seq"][j]).to(dev)
            m = torch.from_numpy(TR["mask"][j]).to(dev)
            tp = torch.from_numpy(TR["tpos"][j]).to(dev)
            ev, pooled = model(x, m, tp)
            loss = evidential_loss_discern_ext(ev, y[torch.from_numpy(j).to(dev)], 2, ep, epochs)
            if order_loss > 0:
                # ORDER-DISCRIMINATION AUXILIARY LOSS. Half the batch is re-presented with its
                # valid frames permuted, and a head must tell ordered from shuffled. This is a
                # direct intervention on the symptom measured in P: a temporal stage that cannot
                # distinguish the two has learned a permutation-invariant function, and no detection
                # objective was pushing it not to. Whether forcing order sensitivity HELPS detection
                # is the open question -- the loss makes it answerable rather than assumed.
                # Permute the VALID frames among the valid slots and leave padding at the end.
                # `+10` on invalid slots pushes them to the back of the ascending argsort; using
                # `-10` puts them at the FRONT, which would shuffle padding into the real frames and
                # make the order head solve a padding-detection task instead.
                # NAMED `fperm`, NOT `perm`: `perm` is the epoch's batch ordering, and reusing the
                # name here overwrote it with a CUDA tensor, so the next iteration tried to index
                # the feature array with it. That is what killed both order-loss runs on the first
                # attempt.
                fperm = torch.argsort(torch.rand_like(tp) + (~m).float() * 10.0, dim=1)
                xs = torch.gather(x, 1, fperm.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
                _ev_s, pooled_s = model(xs, m, tp)
                lo = model.order_logits(torch.cat([pooled, pooled_s], 0))
                tgt = torch.cat([torch.ones(len(pooled), dtype=torch.long, device=dev),
                                 torch.zeros(len(pooled_s), dtype=torch.long, device=dev)])
                loss = loss + order_loss * F.cross_entropy(lo, tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        sch.step()
        sc, v2, ffauc, types = dev_score(model, DEV, dev)
        hist.append({"epoch": ep, "loss": float(np.mean(losses)), "dev_v31": sc,
                     "valmixv2": v2, "ffpp_val_auroc": ffauc, "types": types})
        print(f"  epoch {ep:2d} loss {np.mean(losses):.4f} dev v3.1 {sc:.4f} "
              f"VALmix-v2 {v2:.4f} FF++val {ffauc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if sc > best[0] + 1e-6:
            best = (sc, ep)
            torch.save({"state": model.state_dict(), "epoch": ep, "dev_v31": sc, "dim": TR["dim"],
                        "variant": variant, "pool": pool, "use_pos": use_pos, "seed": seed,
                        "pe_scale": scale, "order_loss": order_loss,
                        "params": pr}, C.SEQ / f"{tag}.pt")
    h = [x for x in hist if x["epoch"] == best[1]][0]
    rec = {"tag": tag, "variant": variant, "pool": pool, "pos_encoding": use_pos, "seed": seed,
           "order_loss": order_loss, "pe_scale": scale,
           "dim": TR["dim"], "params": pr, "best_epoch": best[1], "dev_v31": h["dev_v31"],
           "valmixv2": h["valmixv2"], "ffpp_val_auroc": h["ffpp_val_auroc"], "types": h["types"],
           "train_videos": TR["n_videos"], "clip_len": int(TR["seq"].shape[1]),
           "decoded_frames": TR["unique_frames"],
           "recipe": {"epochs": epochs, "batch": BATCH, "lr": LR, "wd": WD, "d_model": D_MODEL,
                      "n_heads": N_HEADS, "ff": FF, "dropout": DROPOUT, "n_blocks": N_BLOCKS,
                      "loss": "evidential_loss_discern_ext with KL ramp"},
           "note": VARIANTS[variant]["note"], "history": hist}
    json.dump(rec, open(C.SEQ / f"{tag}.json", "w"), indent=1)
    print(f"{tag}: best epoch {best[1]}, dev v3.1 {best[0]:.4f}")
    return rec


def model_tag(variant, pool, use_pos, seed, order_loss: float = 0.0) -> str:
    bits = [f"S-{variant}"]
    if pool != "attn":
        bits.append(f"pool{pool}")
    if not use_pos:
        bits.append("nopos")
    if order_loss:
        bits.append(f"ord{order_loss:g}")
    bits.append(f"s{seed}")
    return "_".join(bits)


# ----------------------------------------------------------------------------- export
def export(tag: str, datasets=None, out_dir: Path | None = None) -> dict:
    """Write `v,y,p,u` per dataset from a trained S, for `branches.py` to pick up."""
    ck = C.SEQ / f"{tag}.pt"
    if not ck.exists():
        raise FileNotFoundError(ck)
    st = torch.load(ck, map_location="cpu")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SeqBranch(st["dim"], pool=st["pool"], use_pos=st["use_pos"],
                      pe_scale=st.get("pe_scale", 10.0),
                      order_head=bool(st.get("order_loss", 0))).to(dev)
    model.load_state_dict(st["state"])
    out = out_dir or (C.SEQ / "opinions")
    out.mkdir(parents=True, exist_ok=True)
    if datasets is None:
        # A dense variant has no dense test crops, so it is a development-only pilot by construction
        # rather than by omission. Exporting it over the test sets would silently read the SPARSE
        # caches under a dense tag.
        datasets = (list(DENSE_SETS) if VARIANTS[st["variant"]].get("suffix") == "-dense"
                    else list(C.DEV_SETS) + list(C.TEST_SETS))
    done = {}
    for ds in datasets:
        try:
            Dc = clips(st["variant"], ds)
        except FileNotFoundError:
            done[ds] = "missing cache"
            continue
        p, u = predict(model, Dc, dev)
        C.assert_valid_opinions(p, u, tag=f"{tag}/{ds}")
        C.write_opinions(out / f"{ds}.csv", Dc["order"], Dc["yv"], p, u)
        done[ds] = f"{len(p)} videos"
        print(f"  {ds}: {done[ds]}", flush=True)
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="adapt", choices=sorted(VARIANTS))
    ap.add_argument("--order-loss", type=float, default=0.0,
                    help="weight of the order-discrimination auxiliary loss (0 disables it)")
    ap.add_argument("--pool", default="attn", choices=("attn", "mean", "mlp"))
    ap.add_argument("--no-pos", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--saturation", action="store_true", help="print the saturation report and exit")
    ap.add_argument("--export", default="", help="tag to export, or 'auto' for the best variant")
    ap.add_argument("--sets", default="")
    a = ap.parse_args()

    if a.saturation:
        r = saturation_report()
        print(json.dumps(r, indent=1))
        json.dump(r, open(C.SEQ / "saturation.json", "w"), indent=1)
        return 0

    if a.export:
        tag = a.export
        if tag == "auto":
            recs = [json.load(open(f)) for f in C.SEQ.glob("S-*_s*.json")]
            recs = [r for r in recs if r["pool"] == "attn" and r["pos_encoding"]]
            if not recs:
                print("no trained S variants to export")
                return 2
            tag = max(recs, key=lambda r: r["dev_v31"])["tag"]
            print(f"best S variant on development: {tag}")
        sets = [s for s in a.sets.split(",") if s] or None
        rep = export(tag, sets)
        json.dump({"tag": tag, "sets": rep}, open(C.SEQ / "export_report.json", "w"), indent=1)
        return 0

    if a.variant == "adapt":
        sat = saturation_report()
        print("saturation of the appearance branch on FF++ train (section 2.2):")
        print(json.dumps(sat, indent=1), flush=True)
        json.dump(sat, open(C.SEQ / "saturation.json", "w"), indent=1)

    train(a.variant, a.seed, a.pool, not a.no_pos, a.epochs, order_loss=a.order_loss)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Part 3c: adapt a shortlisted spatial backbone, then score it exactly as the frozen screen did.

    CUDA_VISIBLE_DEVICES=1 <discern_ext_timm python> upgrade/adapt_backbone.py \
        --backbone pe_core_l --recipe L --seed 0

TWO RECIPES, as the brief fixes them.
  L  LoRA rank 8, alpha 16, dropout 0.1 on query and value of every attention block. The backbone is
     frozen and only the adapters and the head train. Same configuration as the deployed APP view.
  N  LayerNorm parameters only, with L2-normalised features, following GenD (Yermakov et al.); the
     backbone is otherwise frozen. Credited in the report as the brief requires.

Everything else is held to the frozen screen so a difference is the adaptation and nothing else: the
same head (Linear 256, GELU, dropout 0.1, Linear 2, softplus), the same EDL loss and KL ramp, the
same 20 epochs, the same VALmix-v2 selection, and FF++ c23 train SOURCE-PAIRED as deployed.

Features are recomputed through the adapting backbone every epoch -- they cannot be cached the way
the frozen screen caches them, which is what makes this the expensive part of the brief.
"""
from __future__ import annotations

import argparse, io, json, math, sys, time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
EXT = Path("/data/umar/Repos/DISCERN_Ext/eval_adaptation/data")
sys.path[:0] = [str(ROOT / "upgrade"), str(ROOT / "branches/ijepa"), "/data/umar/Repos/DISCERN_Ext"]
import v31 as V31                                                   # noqa: E402
from extract_spatial import SPECS                                   # noqa: E402
from src.losses.evidential_loss import evidential_loss_discern_ext  # noqa: E402

torch.set_num_threads(4)
OUT = ROOT / "analysis/upgrade/adapt"
EPOCHS, BATCH, WD = 20, 64, 0.01
LR_ADAPT, LR_HEAD = 2e-4, 2e-3
LORA_R, LORA_ALPHA, LORA_DROPOUT = 8, 16, 0.1
DEV_SETS = ("VALmix", "VALmixV2", "FFpp_val")
V2_SOURCES = {"CDFv3valG", "DF40train", "DS20val", "DS11val", "FFIWval"}
MAX_TRAIN_FRAMES = 40000     # budget cap; the frozen screen used all of FF++ train
# Selection re-encodes the development frames through the ADAPTING backbone every epoch, so unlike
# the frozen screen it cannot cache them. At PE-Core-L's 336 input that is ~60k frames per epoch and
# it dominates the run -- the first smoke test spent more time scoring than training. Selection runs
# on a fixed stratified subsample instead, and the chosen epoch is re-scored on the FULL development
# sets at the end, so the reported number is never the subsampled one.
MAX_DEV_FRAMES = 12000


class LoRALinear(nn.Module):
    """Frozen base + trainable low-rank update, B initialised to zero so training starts at identity."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.a = nn.Linear(base.in_features, LORA_R, bias=False)
        self.b = nn.Linear(LORA_R, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5)); nn.init.zeros_(self.b.weight)
        self.drop, self.scale = nn.Dropout(LORA_DROPOUT), LORA_ALPHA / LORA_R

    def forward(self, x):
        return self.base(x) + self.b(self.a(self.drop(x))) * self.scale


class Head(nn.Module):
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.fc1, self.drop, self.fc2 = nn.Linear(d, 256), nn.Dropout(dropout), nn.Linear(256, 2)

    def forward(self, x):
        return F.softplus(self.fc2(self.drop(F.gelu(self.fc1(x)))))


def inject_lora(model) -> list:
    """q/v of every attention block. timm ViTs fuse these into one `qkv` Linear, so the whole fused
    projection is wrapped -- the key third is adapted too, which is a documented deviation from
    adapting q and v alone and is recorded in the run JSON."""
    added, fused = [], False
    for mod in model.modules():
        for child, sub in list(mod.named_children()):
            if isinstance(sub, nn.Linear) and child in ("q_proj", "v_proj"):
                setattr(mod, child, LoRALinear(sub)); added.append(getattr(mod, child))
            elif isinstance(sub, nn.Linear) and child == "qkv":
                setattr(mod, child, LoRALinear(sub)); added.append(getattr(mod, child)); fused = True
    assert added, "no attention projections found to adapt"
    return added, fused


def build(name, dev, recipe):
    spec = SPECS[name]
    if spec["kind"] == "hf":
        from transformers import CLIPVisionModel
        m = CLIPVisionModel.from_pretrained(spec["id"]).vision_model
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    else:
        import timm
        m = timm.create_model(spec["id"], pretrained=True, num_classes=0)
        cfg = m.pretrained_cfg
        mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1); std = torch.tensor(cfg["std"]).view(1, 3, 1, 1)
    for p in m.parameters():
        p.requires_grad_(False)
    fused = False
    if recipe == "LN":
        # Both at once. LoRA adds a low-rank update inside the attention projections; LayerNorm
        # rescales each block's activations. They act on different parameters and neither subsumes
        # the other, so training them together is a strictly larger hypothesis class than either --
        # which is the point, and also why it can overfit where the single-knob recipes did not.
        trainable, fused = inject_lora(m)
        params = [p for w in trainable for p in list(w.a.parameters()) + list(w.b.parameters())]
        for p_ in params:
            p_.requires_grad_(True)
        for mod in m.modules():
            if isinstance(mod, nn.LayerNorm):
                for p_ in mod.parameters():
                    p_.requires_grad_(True); params.append(p_)
    elif recipe == "L":
        trainable, fused = inject_lora(m)
        params = [p for w in trainable for p in list(w.a.parameters()) + list(w.b.parameters())]
        for p in params:
            p.requires_grad_(True)
    else:                                           # N: LayerNorm parameters only (GenD)
        params = []
        for mod in m.modules():
            if isinstance(mod, nn.LayerNorm):
                for p in mod.parameters():
                    p.requires_grad_(True); params.append(p)
        assert params, "no LayerNorm parameters found"
    return m.to(dev), mean.to(dev), std.to(dev), spec, params, fused


_HOOKS: dict = {}


def attach_depth_hooks(m, spec, readout):
    """Capture patch tokens at the depths a multi-layer readout needs. Part 3d found the combined
    readout is the best frozen row on PE-Core, but adaptation so far trains only the LAST block's
    output -- so the two best findings had never been combined. `--readout last+mid+half` does that."""
    comps = [c for c in readout.split("+") if c != "last"]
    if not comps:
        return {}
    blocks = (list(m.encoder.layers) if spec["kind"] == "hf"
              else list(getattr(m, "blocks", getattr(m, "layers", []))))
    n = len(blocks)
    ratio = {"mid": 2 / 3, "half": 1 / 2}
    idx = {c: max(0, min(n - 1, int(round(n * ratio[c])) - 1)) for c in comps}

    def hook(name):
        def fn(_mod, _inp, out):
            _HOOKS[name] = out[0] if isinstance(out, tuple) else out
        return fn

    for c, i in idx.items():
        blocks[i].register_forward_hook(hook(c))
    return idx


def embed(m, spec, x, recipe, readout="last"):
    z = m(x)
    last = z.last_hidden_state[:, 0] if spec["kind"] == "hf" else z
    if readout == "last":
        return F.normalize(last, dim=-1) if recipe == "N" else last
    parts = []
    for c in readout.split("+"):
        v = last if c == "last" else _HOOKS[c][:, 1:].mean(1)
        parts.append(F.normalize(v.float(), dim=-1))     # per-component L2, as in the frozen study
    return torch.cat(parts, dim=-1)


def subsample(X, K, limit, seed):
    """Fixed stratified-by-video subsample, so every epoch scores the same frames."""
    if not limit or len(K) <= limit:
        return X, K
    vid = np.array([video_of_key(k) for k in K])
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(K))
    seen, keep = {}, []
    per = max(1, limit // max(1, len(np.unique(vid))))
    for i in order:                       # round-robin over videos so no video dominates
        v = vid[i]
        if seen.get(v, 0) < per:
            seen[v] = seen.get(v, 0) + 1; keep.append(i)
        if len(keep) >= limit:
            break
    keep = np.sort(np.array(keep))
    return X[keep], K[keep]


def frames_of(dataset, size, limit=0, seed=0):
    """Decoded crops plus their keys. Source-paired for training is handled by the caller's sampler."""
    man = sorted((EXT / "splits" / dataset).glob("*.txt"))
    lines = [l.strip() for mm in man for l in mm.read_text().splitlines() if l.strip()]
    if limit and len(lines) > limit:
        lines = list(np.random.default_rng(seed).choice(np.array(lines, dtype=object), limit, replace=False))
    X, K = [], []
    with h5py.File(EXT / "h5" / f"{dataset}.h5", "r") as f:
        for k in lines:
            if k not in f:
                continue
            a = f[k][()]
            im = (Image.open(io.BytesIO(a.tobytes())) if a.dtype == np.uint8 and a.ndim == 1
                  else Image.fromarray(np.asarray(a).astype(np.uint8))).convert("RGB")
            if im.size != (size, size):
                im = im.resize((size, size), Image.BICUBIC)
            X.append(np.asarray(im, dtype=np.uint8)); K.append(k)
    return np.stack(X), np.array(K, dtype=object)


def video_of_key(k: str) -> str:
    return str(k).rsplit("/", 1)[0]


@torch.no_grad()
def dev_scores(m, spec, head, dev, recipe, cache, readout="last", bs=64):
    """Frame probabilities -> per-video mean -> v3.1, exactly as the frozen screen scores."""
    m.eval(); head.eval()
    out = {}
    for name, (X, K, mean, std) in cache.items():
        ps = []
        for i in range(0, len(X), bs):
            x = torch.from_numpy(X[i:i + bs]).permute(0, 3, 1, 2).float().div_(255).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ev = head(embed(m, spec, (x - mean) / std, recipe, readout).float())
            a = ev + 1.0
            ps.append((a / a.sum(1, keepdim=True))[:, 1].float().cpu().numpy())
        p = np.concatenate(ps)
        vid = np.array([video_of_key(k) for k in K])
        order, inv = np.unique(vid, return_inverse=True)
        cnt = np.bincount(inv, minlength=len(order)).astype(float)
        pv = np.bincount(inv, weights=p, minlength=len(order)) / cnt
        yv = np.array([0 if "real" in str(v).split("/")[-2].lower() else 1 for v in order])
        first = {}
        for k, v in zip(K, vid):
            first.setdefault(v, str(k))
        out[name] = (np.array([first[v] for v in order], dtype=object), pv, yv)
    dk, dp, dy = [], [], []
    for n in ("VALmix", "VALmixV2"):
        k, p, y = out[n]; dk += list(k); dp.append(p); dy.append(y)
    fk, fp, fy = out["FFpp_val"]
    sc, types, cells = V31.score(dk, np.concatenate(dp), np.concatenate(dy), fk, fp, fy)
    v2 = float(np.nanmean([v for (t, s), v in cells.items() if s in V2_SOURCES and not np.isnan(v)]))
    return sc, v2, V31.fast_auc(fp[fy == 0], fp[fy == 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=list(SPECS))
    ap.add_argument("--recipe", required=True, choices=["L", "N", "LN"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--train-frames", type=int, default=MAX_TRAIN_FRAMES)
    ap.add_argument("--readout", default="last",
                    help="last | mid | half | last+mid | last+mid+half (Part 3d readout, adapted)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{a.backbone}_{a.recipe}_s{a.seed}" + ("" if a.readout == "last" else f"_{a.readout.replace(chr(43),chr(45))}")
    if (OUT / f"{tag}.json").exists():
        print(f"{tag} exists"); return 0

    dev = "cuda"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    m, mean, std, spec, params, fused = build(a.backbone, dev, a.recipe)
    depth_idx = attach_depth_hooks(m, spec, a.readout)
    n_tr = sum(p.numel() for p in params)
    print(f"{tag}: input {spec['size']}, readout {a.readout} {depth_idx}, trainable {n_tr/1e6:.2f} M"
          + (" (fused qkv: key adapted too)" if fused else ""), flush=True)

    t0 = time.time()
    Xtr, Ktr = frames_of("FFpp_train", spec["size"], a.train_frames, a.seed)
    ytr = np.array([0 if "real" in str(k).split("/")[-3].lower() else 1 for k in Ktr])
    cache, full = {}, {}
    for n in DEV_SETS:
        X, K = frames_of(n, spec["size"])
        full[n] = (X, K, mean, std)
        Xs, Ks = subsample(X, K, MAX_DEV_FRAMES // len(DEV_SETS), a.seed)
        cache[n] = (Xs, Ks, mean, std)
    print(f"  data ready in {(time.time()-t0)/60:.1f} min: train {len(Xtr)}, selection on "
          + ", ".join(f"{n} {len(cache[n][0])}/{len(full[n][0])}" for n in DEV_SETS), flush=True)

    d = embed(m, spec, torch.zeros(1, 3, spec["size"], spec["size"], device=dev), a.recipe, a.readout).shape[-1]
    head = Head(d).to(dev)
    opt = torch.optim.AdamW([{"params": params, "lr": LR_ADAPT},
                             {"params": head.parameters(), "lr": LR_HEAD}], weight_decay=WD)
    steps = EPOCHS * max(1, len(Xtr) // a.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(1e-6 / LR_ADAPT, 0.5 * (1 + math.cos(math.pi * s / steps))))
    # class-balanced, and drawn fresh each epoch so the pairing is not frozen into one order
    w = np.where(ytr == 1, 1.0 / max(1, (ytr == 1).sum()), 1.0 / max(1, (ytr == 0).sum()))
    w = w / w.sum()
    rng = np.random.default_rng(a.seed)

    hist, best = [], (-1.0, None, None)
    for ep in range(EPOCHS):
        m.train(); head.train()
        idx = rng.choice(len(Xtr), size=len(Xtr), replace=True, p=w)
        losses = []
        for i in range(0, len(idx) - a.batch + 1, a.batch):
            j = idx[i:i + a.batch]
            x = torch.from_numpy(Xtr[j]).permute(0, 3, 1, 2).float().div_(255).to(dev)
            y = torch.from_numpy(ytr[j]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ev = head(embed(m, spec, (x - mean) / std, a.recipe, a.readout).float())
            loss = evidential_loss_discern_ext(ev.float(), y, 2, ep, EPOCHS).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            losses.append(loss.item())
        sc, v2, ff = dev_scores(m, spec, head, dev, a.recipe, cache, a.readout)
        hist.append({"epoch": ep, "loss": float(np.mean(losses)), "dev_v31": sc,
                     "valmixv2": v2, "ffpp_val_auroc": ff, "minutes": (time.time() - t0) / 60})
        print(f"  epoch {ep} loss {np.mean(losses):.4f} dev v3.1 {sc:.4f} VALmix-v2 {v2:.4f} "
              f"FF++val {ff:.4f} ({(time.time()-t0)/60:.0f} min)", flush=True)
        if sc > best[0] + 1e-6:
            best = (sc, ep, {"head": head.state_dict(),
                             "adapt": {k: v.detach().cpu() for k, v in m.state_dict().items()
                                       if any(t in k for t in (".a.", ".b.", "norm", "ln"))}})
            torch.save({"epoch": ep, **best[2]}, OUT / f"{tag}.pt")

    # Re-score the selected epoch on the FULL development sets; the per-epoch numbers above are
    # the subsample and are kept in `history` so the two are never confused.
    ck = torch.load(OUT / f"{tag}.pt", map_location=dev)
    head.load_state_dict(ck["head"])
    miss = m.load_state_dict({**m.state_dict(), **{k: v.to(dev) for k, v in ck["adapt"].items()}}, strict=False)
    sc_f, v2_f, ff_f = dev_scores(m, spec, head, dev, a.recipe, full, a.readout)
    print(f"  selected epoch {ck['epoch']} re-scored on FULL dev: v3.1 {sc_f:.4f} "
          f"VALmix-v2 {v2_f:.4f} FF++val {ff_f:.4f}", flush=True)

    json.dump({"backbone": a.backbone, "recipe": a.recipe, "seed": a.seed, "dim": int(d),
               "dev_v31_full": sc_f, "valmixv2_full": v2_f, "ffpp_val_auroc_full": ff_f,
               "selection_dev_frames": int(sum(len(cache[n][0]) for n in DEV_SETS)),
               "input_size": spec["size"], "trainable_params": int(n_tr), "fused_qkv": bool(fused),
               "readout": a.readout, "depth_index": depth_idx,
               "train_frames": int(len(Xtr)), "best_epoch": best[1], "dev_v31": best[0],
               "valmixv2": hist[best[1]]["valmixv2"], "ffpp_val_auroc": hist[best[1]]["ffpp_val_auroc"],
               "minutes": round((time.time() - t0) / 60, 1), "history": hist},
              open(OUT / f"{tag}.json", "w"), indent=1)
    print(f"{tag}: best epoch {best[1]}, dev v3.1 {best[0]:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

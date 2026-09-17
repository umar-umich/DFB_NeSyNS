#!/usr/bin/env python3
"""Bounded adaptation of a temporal branch (brief Experiment 4): VideoMAE or V-JEPA 2 on our stored face-crop frames.

    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 python analysis/tbiom/temporal/train_branch.py --model videomae
    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 <temporal_vjepa2 python> analysis/tbiom/temporal/train_branch.py --model vjepa2

BUDGET AND RECIPE, FIXED BEFORE LAUNCH (2026-09-17)
  data        FF++ c23 train only (3,600 videos), 16 frames per video sampled uniformly over the stored crops, in order;
              class-balanced sampling; no augmentation (the matched R0 recipe has none)
  trainable   the LAST 2 transformer blocks (+ V-JEPA 2's final encoder LayerNorm) and an evidential head
              (LayerNorm -> 128 GELU -> 2, softplus); everything else frozen. VideoMAE: 2 of 12 blocks (14.2M params);
              V-JEPA 2 ViT-L: 2 of 24 blocks + LayerNorm (25.2M). Token mean-pooling as in the frozen probes.
  optimiser   AdamW, weight decay 0.01, cosine to 1e-6, bf16 autocast, 10 epochs, seed 42; batch 32 videos with learning
              rates scaled linearly from the batch-8 plan (blocks 5e-5 -> 2e-4; head 1e-3 -> 2e-3, capped).
              REVISED 2026-09-17 before any epoch completed (UMAR: larger batch to cut training time); the batch-8 runs
              were stopped after ~3 minutes and discarded. Epochs, trainable modules, loss and selection unchanged.
  loss        DISCERN's evidential loss (evidential_loss_discern_ext), KL annealed over the 10 epochs
  selection   after every epoch, the v3.1 development score on VALmix + VALmix-v2 + FF++ val (clean; same cells and
              scope as detector checkpoints); best epoch kept, ties -> earliest. Only the selected epoch is exported
              on processed development and the test sets.
The Appearance / Off-manifold residual (beta-VAE) / face-prior paths are not touched; fusion happens afterwards on cached
opinions (analysis/tbiom/temporal/branch_fusion.py) with the audit's grouped fitting folds.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import h5py, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
torch.set_num_threads(4)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "xclip")); sys.path.insert(0, "/data/umar/Repos/DISCERN_Ext")
from video_id import video_id  # noqa: E402
from extract_xclip import apply_processing, dataset_spec, OUT_OF_SCOPE, EXT  # noqa: E402
from src.losses.evidential_loss import evidential_loss_discern_ext  # noqa: E402

# name -> (HF id, feature dim, frames per clip)
MODELS = {"videomae": ("MCG-NJU/videomae-base-finetuned-kinetics", 768, 16),
          "vjepa2": ("facebook/vjepa2-vitl-fpc64-256", 1024, 16),
          # added 2026-09-17 (same budget): motion-centric SSv2 fine-tunes, a larger VideoMAE, and three other designs
          "videomae_ssv2": ("MCG-NJU/videomae-base-finetuned-ssv2", 768, 16),
          "vjepa2_ssv2": ("facebook/vjepa2-vitl-fpc16-256-ssv2", 1024, 16),
          "videomae_large": ("MCG-NJU/videomae-large-finetuned-kinetics", 1024, 16),
          "timesformer": ("facebook/timesformer-base-finetuned-k400", 768, 8),
          "xclip": ("microsoft/xclip-base-patch16-16-frames", 512, 16),
          "vivit": ("google/vivit-b-16x2-kinetics400", 768, 32)}
EPOCHS, UNFREEZE, BATCH = 10, 2, 32
LR_BLOCKS, LR_HEAD = 2e-4, 2e-3
DEV = ["VALmix", "VALmixV2", "FFpp_val"]
TESTS = ["CDFv2", "CDFv3", "DFD", "DFDC", "DFDCP", "DFEval24", "FFIW", "KoDF", "IDForge", "DFDM"]
S23 = Path("/data/umar/Repos/DFB_NeSyNS/logs/tbiom/stage23")


def videos_of(ds):
    h5rel, lists = dataset_spec(ds)
    keys = [k for f in lists for k in (EXT / f).read_text().split("\n") if k]
    keys = [k for k in keys if not (k.split("/")[-3] in OUT_OF_SCOPE or k.split("/")[-3].split("-")[-1] in OUT_OF_SCOPE)]
    vids: dict[str, list[str]] = {}
    for k in keys:
        vids.setdefault(k.rsplit("/", 1)[0], []).append(k)
    return h5rel, [(sorted(vids[v]), 0 if "real" in v.split("/")[-2].lower() else 1,
                    video_id(pd.Series([f"{h5rel}::{v}/x.png"])).iloc[0]) for v in sorted(vids)]


class Clips(torch.utils.data.Dataset):
    def __init__(self, ds, proc, T, shuffle=False):
        self.h5rel, self.items = videos_of(ds); self.proc = proc; self.h = None; self.T = T; self.shuffle = shuffle

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        if self.h is None:
            self.h = h5py.File(EXT / self.h5rel, "r")
        ks, y, v = self.items[i]
        idx = np.linspace(0, len(ks) - 1, self.T).round().astype(int)
        if self.shuffle:                                   # order-sensitivity control: fixed permutation per video
            idx = idx[np.random.default_rng(i).permutation(self.T)]
        fr = [np.asarray(self.h[ks[j]]) for j in idx]
        fr = [f if f.shape[:2] == (256, 256) else np.asarray(Image.fromarray(f).resize((256, 256), Image.BILINEAR)) for f in fr]
        return self.proc(fr), y, i


class Branch(nn.Module):
    def __init__(self, name):
        super().__init__(); mid, d, self.T = MODELS[name]; self.name = name
        if name.startswith("vjepa2"):
            from transformers import AutoVideoProcessor, VJEPA2Model
            self.backbone = VJEPA2Model.from_pretrained(mid); self.processor = AutoVideoProcessor.from_pretrained(mid)
            blocks = list(self.backbone.encoder.layer[-UNFREEZE:]) + [self.backbone.encoder.layernorm]
        elif name.startswith("videomae"):
            from transformers import AutoImageProcessor, VideoMAEModel
            self.backbone = VideoMAEModel.from_pretrained(mid); self.processor = AutoImageProcessor.from_pretrained(mid)
            blocks = list(self.backbone.encoder.layer[-UNFREEZE:])
        elif name == "timesformer":
            from transformers import AutoImageProcessor, TimesformerModel
            self.backbone = TimesformerModel.from_pretrained(mid); self.processor = AutoImageProcessor.from_pretrained(mid)
            blocks = list(self.backbone.encoder.layer[-UNFREEZE:]) + [self.backbone.layernorm]
        elif name == "vivit":
            from transformers import VivitImageProcessor, VivitModel     # the hub config lacks image_processor_type
            self.backbone = VivitModel.from_pretrained(mid, add_pooling_layer=False); self.processor = VivitImageProcessor.from_pretrained(mid)
            blocks = list(self.backbone.encoder.layer[-UNFREEZE:]) + [self.backbone.layernorm]
        elif name == "xclip":
            from transformers import AutoProcessor, XCLIPModel
            self.backbone = XCLIPModel.from_pretrained(mid); self.processor = AutoProcessor.from_pretrained(mid).image_processor
            v = self.backbone.vision_model
            blocks = list(v.encoder.layers[-UNFREEZE:]) + [v.post_layernorm, self.backbone.visual_projection, self.backbone.mit]
        else:
            raise ValueError(name)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.trainable_blocks = nn.ModuleList(blocks)
        for p in self.trainable_blocks.parameters():
            p.requires_grad_(True)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 2))
        self.mean, self.std = self.processor.image_mean, self.processor.image_std

    def preprocess(self, frames):
        if self.name.startswith("vjepa2"):
            return self.processor(np.stack(frames), return_tensors="pt")["pixel_values_videos"][0]
        out = self.processor([Image.fromarray(f) for f in frames], return_tensors="pt")["pixel_values"]
        return out.reshape(-1, *out.shape[-3:])                  # (T, 3, H, W) whatever the processor's leading dims

    def forward(self, px):
        if self.name.startswith("vjepa2"):
            z = self.backbone(pixel_values_videos=px, skip_predictor=True).last_hidden_state.mean(1)
        elif self.name == "xclip":
            B, T_ = px.shape[:2]
            cls = self.backbone.visual_projection(self.backbone.vision_model(pixel_values=px.flatten(0, 1))[1]).view(B, T_, -1)
            z = self.backbone.mit(cls)[1]
        elif self.name == "timesformer":
            z = self.backbone(pixel_values=px).last_hidden_state[:, 0]
        else:
            z = self.backbone(pixel_values=px).last_hidden_state.mean(1)
        return F.softplus(self.head(z.float()))


def opinions(model, ds, dev, process=None, workers=4, shuffle=False):
    loader = torch.utils.data.DataLoader(Clips(ds, model.preprocess, model.T, shuffle), batch_size=BATCH, num_workers=workers,
                                         worker_init_fn=lambda _: torch.set_num_threads(1))
    out, model_items = [], loader.dataset.items
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for px, y, i in loader:
            px = px.to(dev)
            if process:
                B_, T_ = px.shape[:2]
                px = apply_processing(px.flatten(0, 1).float(), process, model.mean, model.std).view(B_, T_, *px.shape[2:])
            a = model(px).double().cpu().numpy() + 1
            for k, ii in enumerate(i.tolist()):
                out.append((model_items[ii][2], int(y[k]), a[k, 1] / a[k].sum(), 2 / a[k].sum()))
    return pd.DataFrame(out, columns=["v", "y", "p", "u"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--export-shuffled-only", action="store_true",
                    help="load best.pt of a finished run and write only the shuffled-frame control exports")
    a = ap.parse_args()
    import exp1_fusion_audit as A
    out_dir = Path(f"/data/umar/Repos/DFB_NeSyNS/logs/tbiom/temporal/branches/{a.model}_seed{a.seed}")
    (out_dir / "opinions").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed); dev = "cuda"
    model = Branch(a.model).to(dev)
    n_blocks = sum(p.numel() for p in model.trainable_blocks.parameters()); n_head = sum(p.numel() for p in model.head.parameters())
    print(f"{a.model}: trainable blocks {n_blocks / 1e6:.1f}M, head {n_head / 1e3:.1f}K", flush=True)

    if a.export_shuffled_only:
        ck = torch.load(out_dir / "best.pt", map_location=dev)
        model.trainable_blocks.load_state_dict(ck["trainable_blocks"]); model.head.load_state_dict(ck["head"])
        export_shuffled(model, dev, out_dir, a.workers); return
    devtab = A.dev_tables({n: S23 / f"warmgate_e19_{n}.csv" for n in DEV}, [1.0])
    cl = A.cells(devtab)
    train = Clips("FFpp_train", model.preprocess, model.T)
    ys = torch.tensor([it[1] for it in train.items])
    w = torch.where(ys == 1, 1.0 / (ys == 1).sum(), 1.0 / (ys == 0).sum()).double()
    sampler = torch.utils.data.WeightedRandomSampler(w, len(ys), replacement=True, generator=torch.Generator().manual_seed(a.seed))
    loader = torch.utils.data.DataLoader(train, batch_size=BATCH, sampler=sampler, num_workers=a.workers, drop_last=True,
                                         worker_init_fn=lambda _: torch.set_num_threads(1), persistent_workers=True)
    opt = torch.optim.AdamW([{"params": model.trainable_blocks.parameters(), "lr": LR_BLOCKS},
                             {"params": model.head.parameters(), "lr": LR_HEAD}], weight_decay=0.01)
    total = EPOCHS * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(1e-6 / LR_BLOCKS, 0.5 * (1 + math.cos(math.pi * s / total))))
    history, best = [], (-1.0, None)
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train(); model.backbone.eval()                  # frozen parts deterministic (no dropout)
        model.trainable_blocks.train(); losses = []
        for px, y, _ in loader:
            px, y = px.to(dev, non_blocking=True), y.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ev = model(px)
            loss = evidential_loss_discern_ext(ev.float(), y, 2, ep, EPOCHS).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); losses.append(loss.item())
        o = pd.concat([opinions(model, n, dev, workers=a.workers) for n in DEV])
        t = devtab[["v", "source", "subset", "y"]].merge(o[["v", "p"]], on="v", how="left")
        assert t.p.notna().all(), "development videos without branch opinions"
        score, types = A.v31(t.p.to_numpy(), A.cells(t))
        history.append({"epoch": ep, "train_loss": float(np.mean(losses)), "dev_v31": score, "types": types,
                        "minutes": (time.time() - t0) / 60})
        print(f"epoch {ep}: loss {np.mean(losses):.4f} dev v3.1 {score:.4f} {types} ({(time.time() - t0) / 60:.0f} min)", flush=True)
        if score > best[0] + 1e-6:
            best = (score, ep)
            torch.save({"epoch": ep, "trainable_blocks": model.trainable_blocks.state_dict(), "head": model.head.state_dict()},
                       out_dir / "best.pt")
            o.to_csv(out_dir / "opinions" / "dev_clean.csv", index=False)
        json.dump({"model": a.model, "seed": a.seed, "history": history, "best_epoch": best[1], "best_dev_v31": best[0],
                   "budget": {"lr_blocks": LR_BLOCKS, "lr_head": LR_HEAD, "epochs": EPOCHS, "unfrozen_blocks": UNFREEZE, "trainable_block_params": n_blocks,
                              "head_params": n_head, "batch": BATCH, "frames": model.T}},
                  open(out_dir / "train_record.json", "w"), indent=1)

    ck = torch.load(out_dir / "best.pt", map_location=dev)
    model.trainable_blocks.load_state_dict(ck["trainable_blocks"]); model.head.load_state_dict(ck["head"])
    print(f"selected epoch {ck['epoch']}; exporting opinions", flush=True)
    for n in DEV:
        opinions(model, n, dev, workers=a.workers).to_csv(out_dir / "opinions" / f"{n}.csv", index=False)
        for c, spec in (("jpeg40", "jpeg:40"), ("rs06", "resize:0.6")):
            opinions(model, n, dev, spec, a.workers).to_csv(out_dir / "opinions" / f"{n}_{c}.csv", index=False)
    for n in TESTS:
        opinions(model, n, dev, workers=a.workers).to_csv(out_dir / "opinions" / f"{n}.csv", index=False)
    for n in ("CDFv2", "DFDCP"):
        for c, spec in (("jpeg40", "jpeg:40"), ("rs06", "resize:0.6")):
            opinions(model, n, dev, spec, a.workers).to_csv(out_dir / "opinions" / f"{n}_{c}.csv", index=False)
    export_shuffled(model, dev, out_dir, a.workers)
    (out_dir / "DONE").touch()
    print(f"done in {(time.time() - t0) / 60:.0f} min", flush=True)


def export_shuffled(model, dev, out_dir, workers):
    """Order-sensitivity control for the TRAINED branch: same selected weights, frames in a fixed shuffled order."""
    for n in DEV + ["CDFv3", "KoDF"]:
        opinions(model, n, dev, workers=workers, shuffle=True).to_csv(out_dir / "opinions" / f"{n}_shuffled.csv", index=False)
    (out_dir / "SHUFFLED_DONE").touch()


if __name__ == "__main__":
    main()

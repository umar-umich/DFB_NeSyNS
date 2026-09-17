#!/usr/bin/env python3
"""Experiment 6: the joint-training ablation of the staged three-view system (finish-line brief, optional).

    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 <timm python> analysis/tbiom/joint/train_joint.py [--epochs 5]

THE QUESTION. The deployed reference is STAGED: the Appearance and Off-manifold views come from the warm-start +
gated detector (their CLIP path frozen, warm-started from our DiCoME run), the temporal branch was adapted separately,
and the three are combined afterwards by DS-R fitted on development folds. This trains all of it TOGETHER -- every
branch adapter and head at once, with branch-level supervision plus a fusion loss -- starting from those same staged
checkpoints, and asks what joint training does to performance and to the per-branch contributions. The staged system
remains the reference; this is the ablation, not a replacement, and it is reported either way.

BUDGET AND RECIPE, FIXED BEFORE LAUNCH (2026-09-17)
  init        temporal = the reference B-augmented V-JEPA 2 LoRA branch (`vjepa2_lora_B_seed42/best.pt`);
              spatial  = the development-selected warm+gate checkpoint (epoch 19), loaded with `freeze_views` OFF
  trainable   temporal LoRA adapters + evidential head; spatial CLIP LoRA adapters, the semantic manifold projector
              (beta-VAE) and the semantic/artifact heads. The face-prior branch and its gate stay frozen and are
              excluded from every loss and from the fusion: they are not part of the reference triple.
  data        FF++ c23 train videos, 16 frames per clip sampled uniformly in order (the temporal recipe); the spatial
              views see 4 of those 16 frames per clip, which bounds the CLIP cost at 4x the temporal batch.
  losses      EDL on each view (lambda_aux 0.5, the staged spatial recipe's weight), EDL on the DS fusion of the three
              views with EQUAL trust, and the staged beta-VAE term (beta 2.0, align 1.0, lambda_vae 0.7).
              Equal trust on purpose: DS-R's reliability weights are fitted on development, and putting them inside
              the training objective would leak development into training.
  optimiser   AdamW, weight decay 0.01, cosine to 1e-6, bf16 autocast, 5 epochs, batch 8 clips, class-balanced
              sampling, lr 2e-4 for every adapter and 2e-3 for the heads -- the staged temporal branch's rates.
  selection   the v3.1 development score of the equal-trust three-view fusion after every epoch (same cells, same
              scope, ties -> earliest epoch). Only the selected epoch is exported.
Exports per-video opinions of all three views (clean development and the ten test sets) so the joint system can be
refitted and compared against the staged one on the SAME source-grouped folds by `exp6_analysis.py`.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
import h5py, numpy as np, pandas as pd, torch, torch.nn as nn
from PIL import Image
torch.set_num_threads(4)
HERE = Path(__file__).resolve().parent
DFB = Path("/data/umar/Repos/DFB_NeSyNS")
EXT = Path("/data/umar/Repos/DISCERN_Ext")
sys.path.insert(0, str(HERE.parent / "temporal")); sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "xclip")); sys.path.insert(0, str(EXT))
import train_branch as TB  # noqa: E402
import exp1_fusion_audit as A  # noqa: E402
from src.losses.evidential_loss import evidential_loss_discern_ext  # noqa: E402

SPATIAL_CKPT = EXT / ("runs/discern_ext_train/warm_dicome_gate_seed42/checkpoints/"
                      "discern_ext-best-epoch=19-val_auroc_video=0.9956.ckpt")
ARCH_CFG = EXT / "eval_adaptation/configs/warmgate_VALmix.yaml"
TEMPORAL = "vjepa2_lora_B_seed42"
DEV, TESTS = ["VALmix", "VALmixV2", "FFpp_val"], ["CDFv2", "CDFv3", "DFD", "DFDC", "DFDCP", "DFEval24", "FFIW", "KoDF", "IDForge", "DFDM"]
S23 = DFB / "logs/tbiom/stage23"
OUT = DFB / "logs/tbiom/joint"
EPOCHS, BATCH, S_FRAMES = 5, 8, 4
LR_ADAPT, LR_HEAD, K = 2e-4, 2e-3, 2.0
VIEWS = ("T", "APP", "OM")


# ----------------------------------------------------------------------------- data
class JointClips(torch.utils.data.Dataset):
    """One item = one video: the temporal clip and S_FRAMES CLIP frames drawn from that same clip."""

    def __init__(self, ds, t_proc, s_proc, T, augment=None, seed=0):
        self.h5rel, self.items = TB.videos_of(ds)
        self.t_proc, self.s_proc, self.T, self.h = t_proc, s_proc, T, None
        self.augment, self.seed, self._rng, self._owner = augment, seed, None, None
        if augment == "B":
            from src.dataset.training_arms import SharedProcessing
            self.processing = SharedProcessing(enabled=True)

    def _worker_rng(self):
        wi = torch.utils.data.get_worker_info(); wid = wi.id if wi is not None else -1
        if self._owner != wid:
            self._rng = np.random.default_rng([int(self.seed), wid + 1, (wi.seed % (2 ** 32)) if wi is not None else 0])
            self._owner = wid
        return self._rng

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        if self.h is None:
            self.h = h5py.File(TB.EXT / self.h5rel, "r")
        ks, y, v = self.items[i]
        idx = np.linspace(0, len(ks) - 1, self.T).round().astype(int)
        fr = [np.asarray(self.h[ks[j]]) for j in idx]
        fr = [f if f.shape[:2] == (256, 256) else np.asarray(Image.fromarray(f).resize((256, 256), Image.BILINEAR)) for f in fr]
        if self.augment == "B":
            setting = self.processing.sample(self._worker_rng())        # one draw per clip, label-independent
            if setting[0] != "none":
                fr = [self.processing.apply(f, setting) for f in fr]
        sidx = np.linspace(0, self.T - 1, S_FRAMES).round().astype(int)
        sp = self.s_proc(images=[Image.fromarray(fr[j]) for j in sidx], return_tensors="pt")["pixel_values"]
        return self.t_proc(fr), sp, y, i


# ----------------------------------------------------------------------------- model
class Joint(nn.Module):
    def __init__(self):
        super().__init__()
        from src.config import load_config
        from src.model.discern_ext_module import DiscernExtModule
        cwd = os.getcwd(); os.chdir(EXT)
        cfg = load_config(str(ARCH_CFG)); cfg.freeze_views = False
        module = DiscernExtModule.load_from_checkpoint(str(SPATIAL_CKPT), config=cfg, verbose=False)
        os.chdir(cwd)
        self.cfg = cfg
        self.spatial = module.model
        self.vae_loss = module.vae_loss
        self.temporal = TB.Branch("vjepa2", lora=True)
        ck = torch.load(DFB / "logs/tbiom/temporal/branches" / TEMPORAL / "best.pt", map_location="cpu")
        self.temporal.trainable_blocks.load_state_dict(ck["trainable_blocks"])
        self.temporal.head.load_state_dict(ck["head"])
        # the face prior is not part of the reference triple: frozen, and excluded from losses and fusion
        for name, p in self.spatial.named_parameters():
            if name.startswith(("fsvfm_branch.", "face_prior_gate.")):
                p.requires_grad_(False)
        from transformers import CLIPImageProcessor
        self.s_proc = CLIPImageProcessor.from_pretrained(cfg.backbone)

    def adapters_and_heads(self):
        ad, hd = [], []
        for n, p in self.spatial.named_parameters():
            if not p.requires_grad:
                continue
            (hd if ("head" in n or "norm" in n) else ad).append(p)
        ad += [p for p in self.temporal.trainable_blocks.parameters() if p.requires_grad]
        hd += list(self.temporal.head.parameters())
        return ad, hd

    def forward(self, px_t, px_s):
        B, S = px_s.shape[:2]
        out = self.spatial(px_s.flatten(0, 1))
        sem_ev, art_ev, feat, manifold, z, mu, log_var = out[1], out[2], out[3], out[4], out[5], out[6], out[7]
        a_app = (sem_ev + 1).view(B, S, 2).mean(1)          # per-video opinion: mean over the clip's frames
        a_om = (art_ev + 1).view(B, S, 2).mean(1)
        a_t = self.temporal(px_t) + 1
        return a_t, a_app, a_om, (sem_ev, art_ev, feat, manifold, mu, log_var)


def ds_fuse(alphas):
    """Dempster's rule over a list of alphas, equal trust -- the operator the deployed system uses."""
    f = alphas[0]
    for a in alphas[1:]:
        S1, S2 = f.sum(1, keepdim=True), a.sum(1, keepdim=True)
        b1, b2, u1, u2 = (f - 1) / S1, (a - 1) / S2, K / S1, K / S2
        c = (b1.sum(1) * b2.sum(1) - (b1 * b2).sum(1)).unsqueeze(1)
        b = (b1 * b2 + b1 * u2 + b2 * u1) / (1 - c); uu = u1 * u2 / (1 - c)
        f = b * (K / uu) + 1
    return f


def pu(a):
    S = a.sum(1)
    return (a[:, 1] / S).double().cpu().numpy(), (K / S).double().cpu().numpy()


@torch.no_grad()
def export(model, ds, dev, workers):
    loader = torch.utils.data.DataLoader(JointClips(ds, model.temporal.preprocess, model.s_proc, model.temporal.T),
                                         batch_size=BATCH * 2, num_workers=workers,
                                         worker_init_fn=lambda _: torch.set_num_threads(1))
    items = loader.dataset.items; rows = []
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for px_t, px_s, y, i in loader:
            a_t, a_app, a_om, _ = model(px_t.to(dev), px_s.to(dev))
            p = {v: pu(a) for v, a in zip(VIEWS, (a_t, a_app, a_om))}
            for k, idx in enumerate(i.tolist()):
                rows.append((items[idx][2], int(y[k]), *[x for v in VIEWS for x in (p[v][0][k], p[v][1][k])]))
    return pd.DataFrame(rows, columns=["v", "y"] + [f"{q}_{v}" for v in VIEWS for q in "pu"])


def dev_score(model, dev, workers, devtab):
    o = pd.concat([export(model, n, dev, workers) for n in DEV], ignore_index=True)
    t = devtab[["v", "source", "subset", "y"]].merge(o.drop(columns=["y"]), on="v", how="left")
    assert t.p_T.notna().all(), "development videos without joint opinions"
    al = [torch.tensor(np.stack([(1 - t[f"p_{v}"]) * K / t[f"u_{v}"], t[f"p_{v}"] * K / t[f"u_{v}"]], 1)) for v in VIEWS]
    f = ds_fuse(al); p = (f[:, 1] / f.sum(1)).numpy()
    return A.v31(p, A.cells(t)), o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS); ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--workers", type=int, default=10); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--augment", default="B", choices=["B", "none"])
    ap.add_argument("--limit-steps", type=int, default=0, help="smoke run: stop the epoch after N steps")
    a = ap.parse_args()
    run = OUT / f"joint_seed{a.seed}"; (run / "opinions").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed); dev = torch.device("cuda")
    model = Joint().to(dev)
    ad, hd = model.adapters_and_heads()
    n_ad, n_hd = sum(p.numel() for p in ad), sum(p.numel() for p in hd)
    print(f"joint: adapters {n_ad / 1e6:.2f}M, heads/norms {n_hd / 1e6:.2f}M trainable", flush=True)

    devtab = A.dev_tables({n: S23 / f"warmgate_e19_{n}.csv" for n in DEV}, [1.0])
    train = JointClips("FFpp_train", model.temporal.preprocess, model.s_proc, model.temporal.T,
                       augment=None if a.augment == "none" else "B", seed=a.seed)
    ys = torch.tensor([it[1] for it in train.items])
    w = torch.where(ys == 1, 1.0 / (ys == 1).sum(), 1.0 / (ys == 0).sum()).double()
    sampler = torch.utils.data.WeightedRandomSampler(w, len(ys), replacement=True,
                                                     generator=torch.Generator().manual_seed(a.seed))
    loader = torch.utils.data.DataLoader(train, batch_size=a.batch, sampler=sampler, num_workers=a.workers,
                                         drop_last=True, persistent_workers=True,
                                         worker_init_fn=lambda _: torch.set_num_threads(1))
    opt = torch.optim.AdamW([{"params": ad, "lr": LR_ADAPT}, {"params": hd, "lr": LR_HEAD}], weight_decay=0.01)
    total = a.epochs * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(1e-6 / LR_ADAPT, 0.5 * (1 + math.cos(math.pi * s / total))))
    lam_aux, lam_vae = float(model.cfg.lambda_aux), float(model.cfg.lambda_vae)
    hist, best, t0 = [], (-1.0, None), time.time()
    for ep in range(a.epochs):
        model.train(); model.temporal.backbone.eval(); losses = []
        for step, (px_t, px_s, y, _) in enumerate(loader):
            if a.limit_steps and step >= a.limit_steps:
                break
            px_t, px_s, y = px_t.to(dev, non_blocking=True), px_s.to(dev, non_blocking=True), y.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                a_t, a_app, a_om, extra = model(px_t, px_s)
            sem_ev, art_ev, feat, manifold, mu, log_var = extra
            yf = y.repeat_interleave(px_s.shape[1])
            edl = lambda ev, lab: evidential_loss_discern_ext(ev.float(), lab, 2, ep, a.epochs).mean()
            branch = edl(sem_ev, yf) + edl(art_ev, yf) + edl(a_t - 1, y)
            fused = edl(ds_fuse([a_t.float(), a_app.float(), a_om.float()]) - 1, y)
            vae = model.vae_loss(z_original=feat.float(), z_reconstructed=manifold.float(), mu=mu.float(),
                                 log_var=log_var.float(), beta_kld=model.cfg.beta_kld,
                                 lambda_align=model.cfg.lambda_align)
            loss = fused + lam_aux * branch + lam_vae * vae
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); losses.append(loss.item())
            if step == 0:
                print(f"  step 0: loss {loss.item():.4f} (fused {fused.item():.4f} branch {branch.item():.4f} "
                      f"vae {vae.item():.4f})", flush=True)
        (score, types), o = dev_score(model, dev, a.workers, devtab)
        hist.append({"epoch": ep, "train_loss": float(np.mean(losses)), "dev_v31": score, "types": types,
                     "minutes": (time.time() - t0) / 60})
        print(f"epoch {ep}: loss {np.mean(losses):.4f} dev v3.1 {score:.4f} {types} ({(time.time() - t0) / 60:.0f} min)", flush=True)
        if score > best[0] + 1e-6:
            best = (score, ep)
            torch.save({"epoch": ep, "spatial": {k: v for k, v in model.spatial.state_dict().items()},
                        "temporal_blocks": model.temporal.trainable_blocks.state_dict(),
                        "temporal_head": model.temporal.head.state_dict()}, run / "best.pt")
            o.to_csv(run / "opinions" / "dev_clean.csv", index=False)
        json.dump({"seed": a.seed, "history": hist, "best_epoch": best[1], "best_dev_v31": best[0],
                   "budget": {"epochs": a.epochs, "batch": a.batch, "spatial_frames_per_clip": S_FRAMES,
                              "lr_adapters": LR_ADAPT, "lr_heads": LR_HEAD, "augment": a.augment,
                              "lambda_aux": lam_aux, "lambda_vae": lam_vae, "trainable_adapters": n_ad,
                              "trainable_heads": n_hd, "init_spatial": str(SPATIAL_CKPT), "init_temporal": TEMPORAL}},
                  open(run / "train_record.json", "w"), indent=1)
    ck = torch.load(run / "best.pt", map_location=dev)
    model.spatial.load_state_dict(ck["spatial"]); model.temporal.trainable_blocks.load_state_dict(ck["temporal_blocks"])
    model.temporal.head.load_state_dict(ck["temporal_head"])
    print(f"selected epoch {ck['epoch']}; exporting", flush=True)
    for n in DEV + TESTS:
        f = run / "opinions" / f"{n}.csv"
        if not f.exists():
            export(model, n, dev, a.workers).to_csv(f, index=False); print("  exported", n, flush=True)
    (run / "DONE").write_text(f"{best[0]:.6f} epoch {best[1]}\n")


if __name__ == "__main__":
    main()

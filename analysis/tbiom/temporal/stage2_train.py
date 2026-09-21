#!/usr/bin/env python3
"""Stage 2 (executable brief): train one bounded temporal configuration and export its opinions.

    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 <temporal_vjepa2 python> analysis/tbiom/temporal/stage2_train.py \
        --sampling {sparse,dense,dsub} --head {pool,gru}

Frozen in tbiom/docs/stage2-temporal-preregistration.md: every configuration uses the reference recipe (V-JEPA 2
ViT-L, LoRA r8 alpha16 dropout 0.1 on query/value of every layer, 10 epochs, batch 32, lr 2e-4 adapters / 2e-3 heads,
arm-B processing drawn once per clip, seed 42, FF++ train videos). Only the sampling of frames and the head change.

  sparse  16 frames at linspace over the video's stored stills (the reference sampling)
  dense   16 frames at stride 2 from the 32 consecutive source frames centred on the midpoint (dense h5)
  dsub    4 frames at stride 8 from that same window, each repeated 4 times (same duration, quarter the density)
  pool    token mean -> evidential head (the reference head)
  gru     tokens averaged over space per temporal index (8 steps) -> GRU(256) -> mean over steps -> evidential head

Epochs are selected on the fitting + selection parts of the bounded development subset only (subset v3.1 of the
branch's own opinions); the confirmation part is never read here except to export opinions after selection.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import h5py, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
torch.set_num_threads(4)
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent / "xclip"), "/data/umar/Repos/DISCERN_Ext"]
import train_branch as TB  # noqa: E402
from extract_xclip import apply_processing  # noqa: E402
from src.losses.evidential_loss import evidential_loss_discern_ext  # noqa: E402

DENSE = Path("/data/umar/Datasets/tbiom_dense")
OUT = Path("/data/umar/Repos/DFB_NeSyNS/logs/tbiom/temporal_stage2")
PART = OUT / "partition.json"
EPOCHS, BATCH, T = 10, 32, 16
SPECS = {"clean": None, "jpeg40": "jpeg:40", "rs06": "resize:0.6"}


# ----------------------------------------------------------------------------- the bounded subset and its partition
def partition():
    """v -> fit / select / confirm, source-grouped, seed 2026. Written once and reused by every consumer."""
    if PART.exists():
        return json.load(open(PART))
    import dsr_system as D, exp1_fusion_audit as A
    from sklearn.model_selection import GroupShuffleSplit
    man = json.load(open(DENSE / "dev2_dense_manifest.json"))["videos"]
    ok = {v for v, m in man.items() if "error" not in m and m.get("fail_rate", 1) <= 0.25}
    dev = D.load("vjepa2_lora_B_seed42")[0]
    dev = dev[dev.v.isin(ok)].reset_index(drop=True)
    g = A.groups_for(dev)
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.6, random_state=2026).split(dev, dev.y, g))
    rest = dev.iloc[b].reset_index(drop=True); gr = g[b]
    c, d = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=2026).split(rest, rest.y, gr))
    part = {**{v: "fit" for v in dev.iloc[a].v}, **{v: "select" for v in rest.iloc[c].v}, **{v: "confirm" for v in rest.iloc[d].v}}
    OUT.mkdir(parents=True, exist_ok=True); json.dump(part, open(PART, "w"))
    return part


# ----------------------------------------------------------------------------- data
class DenseClips(torch.utils.data.Dataset):
    """Clips from the dense h5: the 32-frame window per video, sampled `dense` or `dsub`."""

    def __init__(self, which, videos, sampling, proc, augment=None, seed=0, labels=None):
        self.h5p = DENSE / f"{which}_dense.h5"
        keys = [k for k in open(DENSE / f"{which}_dense_keys.txt").read().split("\n") if k]
        by_v = {}
        for k in keys:
            by_v.setdefault(k.rsplit("/", 1)[0], []).append(k)
        self.items = [(sorted(by_v[v]), labels[v], v) for v in videos if v in by_v]
        self.sampling, self.proc, self.h = sampling, proc, None
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

    def idx(self, n):
        if self.sampling == "dense":
            return np.arange(0, 2 * T, 2)[:T].clip(0, n - 1)
        return np.repeat(np.arange(0, 32, 8), 4).clip(0, n - 1)                 # dsub: 4 frames, same window

    def __getitem__(self, i):
        if self.h is None:
            self.h = h5py.File(self.h5p, "r")
        ks, y, _v = self.items[i]
        fr = [np.asarray(self.h[ks[j]]) for j in self.idx(len(ks))]
        if self.augment == "B":
            setting = self.processing.sample(self._worker_rng())               # one draw per clip, label-independent
            if setting[0] != "none":
                fr = [self.processing.apply(f, setting) for f in fr]
        return self.proc(fr), y, i


def ffpp_train_labels():
    return {v: (0 if "real" in v.split("/")[-2].lower() else 1)
            for v in {k.rsplit("/", 1)[0] for k in open(DENSE / "ffpp_train_dense_keys.txt").read().split("\n") if k}}


# ----------------------------------------------------------------------------- model
class Branch2(TB.Branch):
    def __init__(self, head="pool"):
        super().__init__("vjepa2", lora=True)
        self.head_type = head
        if head == "gru":
            self.tgru = nn.GRU(1024, 256, batch_first=True)
            self.head = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 2))

    def forward(self, px):
        if self.head_type == "pool":
            return super().forward(px)
        h = self.backbone(pixel_values_videos=px, skip_predictor=True).last_hidden_state   # (B, 8*256, 1024), t-major
        h = h.view(h.shape[0], T // 2, -1, h.shape[-1]).mean(2)                             # (B, 8, 1024)
        return F.softplus(self.head(self.tgru(h.float())[0].mean(1)))

    def head_params(self):
        return list(self.head.parameters()) + (list(self.tgru.parameters()) if self.head_type == "gru" else [])


# ----------------------------------------------------------------------------- opinions
@torch.no_grad()
def opinions(model, loader, dev, spec, order="ordered"):
    rows = []; model.eval(); items = loader.dataset.items
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for px, y, i in loader:
            px = px.to(dev); B_, T_ = px.shape[:2]
            if spec:
                px = apply_processing(px.flatten(0, 1).float(), spec, model.mean, model.std).view(B_, T_, *px.shape[2:])
            if order == "reversed":
                px = px.flip(1)
            elif order == "local":
                perm = torch.arange(T_).view(-1, 2).flip(1).reshape(-1); px = px[:, perm]
            elif order == "shuffled":
                px = px[:, torch.randperm(T_, generator=torch.Generator().manual_seed(0))]
            a = model(px).double().cpu().numpy() + 1
            for k, ii in enumerate(i.tolist()):
                rows.append((items[ii][2], int(y[k]), a[k, 1] / a[k].sum(), 2 / a[k].sum()))
    return pd.DataFrame(rows, columns=["v", "y", "p", "u"])


def dev_loader(sampling, model, videos, labels, workers):
    if sampling == "sparse":
        ds = []
        for n in ("VALmix", "VALmixV2", "FFpp_val"):
            c = TB.Clips(n, model.preprocess, T); keep = [it for it in c.items if it[2] in videos]
            c.items = keep; ds.append(c)
        data = torch.utils.data.ConcatDataset(ds)
        # ConcatDataset hides `items`; expose a flat list aligned with the global index
        data.items = [it for c in ds for it in c.items]
        class Wrap(torch.utils.data.Dataset):
            def __init__(s, d): s.d, s.items = d, d.items
            def __len__(s): return len(s.d)
            def __getitem__(s, j): px, y, _ = s.d[j]; return px, y, j
        data = Wrap(data)
    else:
        data = DenseClips("dev2", sorted(videos), sampling, model.preprocess, labels=labels)
    return torch.utils.data.DataLoader(data, batch_size=BATCH, num_workers=workers,
                                       worker_init_fn=lambda _: torch.set_num_threads(1))


def reference_exports(a):
    """The sparse x pool cell is the reference itself: export it through the same code path, never retrain it."""
    import dsr_system as D
    run = OUT / "reference_sparse_pool"; (run / "opinions").mkdir(parents=True, exist_ok=True); dev = "cuda"
    part = partition(); base = D.load("vjepa2_lora_B_seed42")[0]; base = base[base.v.isin(part)]
    model = Branch2("pool").to(dev)
    ck = torch.load(Path("/data/umar/Repos/DFB_NeSyNS/logs/tbiom/temporal/branches/vjepa2_lora_B_seed42/best.pt"), map_location=dev)
    model.trainable_blocks.load_state_dict(ck["trainable_blocks"]); model.head.load_state_dict(ck["head"])
    dl = dev_loader("sparse", model, set(base.v), dict(zip(base.v, base.y)), a.workers)
    tt = time.time()
    for c, spec in SPECS.items():
        opinions(model, dl, dev, spec).to_csv(run / "opinions" / f"dev2_{c}.csv", index=False)
    ms = (time.time() - tt) / 3 / len(base) * 1e3
    for order in ("reversed", "local", "shuffled"):
        opinions(model, dl, dev, None, order).to_csv(run / "opinions" / f"dev2_clean_{order}.csv", index=False)
    json.dump({"reference": "vjepa2_lora_B_seed42", "epoch": ck["epoch"], "export_ms_per_video": ms}, open(run / "train_record.json", "w"))
    (run / "DONE").write_text("reference\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampling", required=True, choices=["sparse", "dense", "dsub"]); ap.add_argument("--head", required=True, choices=["pool", "gru"])
    ap.add_argument("--workers", type=int, default=10); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--diagnostics", action="store_true", help="after training: ordered/reversed/local/shuffled exports")
    ap.add_argument("--reference", action="store_true",
                    help="no training: load the reference branch (vjepa2_lora_B_seed42) and write its subset exports")
    a = ap.parse_args()
    if a.reference:
        return reference_exports(a)
    import exp1_fusion_audit as A, dsr_system as D
    run = OUT / f"{a.sampling}_{a.head}_seed{a.seed}"; (run / "opinions").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed); dev = "cuda"
    part = partition()
    base = D.load("vjepa2_lora_B_seed42")[0]; base = base[base.v.isin(part)].reset_index(drop=True)
    labels = dict(zip(base.v, base.y))
    fs = base[base.v.map(part).isin(["fit", "select"])].reset_index(drop=True)
    model = Branch2(a.head).to(dev)
    if a.sampling == "sparse":
        train = TB.Clips("FFpp_train", model.preprocess, T, augment="B", seed=a.seed)
    else:
        lab = ffpp_train_labels(); train = DenseClips("ffpp_train", sorted(lab), a.sampling, model.preprocess, "B", a.seed, lab)
    ys = torch.tensor([it[1] for it in train.items])
    w = torch.where(ys == 1, 1.0 / (ys == 1).sum(), 1.0 / (ys == 0).sum()).double()
    sampler = torch.utils.data.WeightedRandomSampler(w, len(ys), replacement=True, generator=torch.Generator().manual_seed(a.seed))
    loader = torch.utils.data.DataLoader(train, batch_size=BATCH, sampler=sampler, num_workers=a.workers, drop_last=True,
                                         worker_init_fn=lambda _: torch.set_num_threads(1), persistent_workers=True)
    adapters = [p for p in model.trainable_blocks.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": adapters, "lr": TB.LR_LORA}, {"params": model.head_params(), "lr": TB.LR_HEAD}], weight_decay=0.01)
    total = EPOCHS * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(1e-6 / TB.LR_LORA, 0.5 * (1 + math.cos(math.pi * s / total))))
    dl = dev_loader(a.sampling, model, set(fs.v), labels, a.workers)
    hist, best, t0 = [], (-1.0, None), time.time()
    for ep in range(EPOCHS):
        model.train(); model.backbone.eval(); model.trainable_blocks.train(); losses = []
        for px, y, _ in loader:
            px, y = px.to(dev, non_blocking=True), y.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ev = model(px)
            loss = evidential_loss_discern_ext(ev.float(), y, 2, ep, EPOCHS).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); losses.append(loss.item())
        o = opinions(model, dl, dev, None)
        t = fs[["v", "source", "subset", "y"]].merge(o[["v", "p"]], on="v", how="left")
        assert t.p.notna().all(), "fitting/selection videos without opinions"
        score, types = A.v31(t.p.to_numpy(), A.cells(t))
        hist.append({"epoch": ep, "train_loss": float(np.mean(losses)), "dev_v31_fit_select": score, "types": types, "minutes": (time.time() - t0) / 60})
        print(f"epoch {ep}: loss {np.mean(losses):.4f} fit+select subset-v3.1 {score:.4f} {types} ({(time.time() - t0) / 60:.0f} min)", flush=True)
        if score > best[0] + 1e-6:
            best = (score, ep)
            torch.save({"epoch": ep, "trainable_blocks": model.trainable_blocks.state_dict(), "head": model.head.state_dict(),
                        "tgru": model.tgru.state_dict() if a.head == "gru" else None}, run / "best.pt")
        json.dump({"sampling": a.sampling, "head": a.head, "seed": a.seed, "history": hist, "best_epoch": best[1], "best": best[0],
                   "budget": {"epochs": EPOCHS, "batch": BATCH, "frames": T, "lora": [TB.LORA_R, TB.LORA_ALPHA, TB.LORA_DROPOUT],
                              "lr": [TB.LR_LORA, TB.LR_HEAD], "augment": "B"}}, open(run / "train_record.json", "w"), indent=1)
    ck = torch.load(run / "best.pt", map_location=dev)
    model.trainable_blocks.load_state_dict(ck["trainable_blocks"]); model.head.load_state_dict(ck["head"])
    if a.head == "gru":
        model.tgru.load_state_dict(ck["tgru"])
    print(f"selected epoch {ck['epoch']}; exporting the whole subset", flush=True)
    dl = dev_loader(a.sampling, model, set(base.v), labels, a.workers)
    tt = time.time()
    for c, spec in SPECS.items():
        opinions(model, dl, dev, spec).to_csv(run / "opinions" / f"dev2_{c}.csv", index=False)
    per_video_ms = (time.time() - tt) / 3 / len(base) * 1e3
    if a.diagnostics:
        for order in ("reversed", "local", "shuffled"):
            opinions(model, dl, dev, None, order).to_csv(run / "opinions" / f"dev2_clean_{order}.csv", index=False)
    rec = json.load(open(run / "train_record.json")); rec["export_ms_per_video"] = per_video_ms
    json.dump(rec, open(run / "train_record.json", "w"), indent=1)
    (run / "DONE").write_text(f"{best[0]:.6f} epoch {best[1]}\n")


if __name__ == "__main__":
    main()

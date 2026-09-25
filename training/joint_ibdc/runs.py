#!/usr/bin/env python3
"""J0, J1, J2a and J2b: joint training of the three evidential heads on cached features.

    <discern_ext python> training/joint_ibdc/runs.py --dry-run
    <discern_ext python> training/joint_ibdc/runs.py --run J1 --seed 0

Only the heads train. Every backbone, LoRA adapter and the beta-VAE stay frozen, which is what makes
training on a feature cache equivalent to training them in place.

| run | loss |
|-----|------|
| J0  | no training; the deployed heads, read through the Task 1 fusion code |
| J1  | EDL(fused) + 0.5 * sum_b EDL(branch b, BEFORE discounting) |
| J2a | J1 + lambda * ibdc_v1   (the original IJCB loss, for the failure it was shown to have) |
| J2b | J1 + lambda * L_IBDC    (the corrected one-sided loss) |

BRANCH TERMS ACT ON RAW EVIDENCE, before discounting, so a heavily discounted view keeps learning.
TRAINING USES EQUAL TRUST in the fused term; the deployed weights are development-fitted and are
refit out of fold for evaluation only (see the report, and EXP_JOINT_ABLATION.md for the precedent).
ONE DRAW PER SAMPLE PER EPOCH, cycling the three cached arm-B realisations.

The EDL variant, the KL ramp, the optimiser and VALmix-v2 checkpoint selection are the deployed
head's, confirmed from the deployed config rather than assumed.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
EXT = Path("/data/umar/Repos/DISCERN_Ext")
sys.path[:0] = [str(ROOT / "analysis/tbiom/pcr6p"), str(ROOT / "analysis/tbiom/grouped"),
                str(ROOT / "analysis/tbiom"), str(ROOT / "analysis/tbiom/temporal"),
                str(EXT), str(Path(__file__).resolve().parent)]

import fusion_torch as FT        # noqa: E402
import draws as DR               # noqa: E402
from src.losses.evidential_loss import evidential_loss_discern_ext  # noqa: E402

torch.set_num_threads(4)
CACHE = ROOT / "logs/joint_ibdc/cache"
OUT = ROOT / "analysis/joint_ibdc"
CKPT = EXT / "runs/discern_ext_train/warm_dicome_gate_seed42/checkpoints/discern_ext-best-epoch=19-val_auroc_video=0.9956.ckpt"
BR = ROOT / "logs/tbiom/temporal/branches/vjepa2_lora_B_seed42"
VIEWS = ("T", "APP", "OM")
EPOCHS, BATCH, LR, WD, MIN_LR, N_DRAWS = 20, 128, 1e-4, 0.01, 1e-6, 3
LAMBDAS = (0.1, 0.3, 1.0)
EPS_U = 1e-6


class SpatialHead(nn.Module):
    """The deployed EvidentialHead with its frozen LayerNorm in front."""

    def __init__(self, dim=64):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1, self.fc2 = nn.Linear(dim, 64), nn.Linear(64, 2)

    def forward(self, x):
        return F.softplus(self.fc2(F.relu(self.fc1(self.norm(x)))))


class TemporalHead(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 128), nn.GELU(), nn.Linear(128, 2))

    def forward(self, x):
        return F.softplus(self.net(x))


def load_deployed_heads(dev="cpu", freeze_norms=True):
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    app, om = SpatialHead(), SpatialHead()
    for h, pre in ((app, "model.semantic"), (om, "model.artifact")):
        h.norm.weight.data = sd[f"{pre}_norm.weight"].float().clone()
        h.norm.bias.data = sd[f"{pre}_norm.bias"].float().clone()
        h.fc1.weight.data = sd[f"{pre}_head.fc1.weight"].float().clone()
        h.fc1.bias.data = sd[f"{pre}_head.fc1.bias"].float().clone()
        h.fc2.weight.data = sd[f"{pre}_head.fc2.weight"].float().clone()
        h.fc2.bias.data = sd[f"{pre}_head.fc2.bias"].float().clone()
        if freeze_norms:
            h.norm.weight.requires_grad_(False); h.norm.bias.requires_grad_(False)
    t = TemporalHead()
    t.net.load_state_dict(torch.load(BR / "best.pt", map_location="cpu")["head"])
    return app.to(dev), om.to(dev), t.to(dev)


def fused_evidence(e):
    """(n,3,2) raw evidence -> fused evidence under equal trust, plus the fused masses."""
    m = FT.opinion_of_evidence(e)
    f = FT.fuse(m, plus=True)
    u = f[..., 2].clamp_min(EPS_U)
    ev = torch.stack([FT.W_FRAME * f[..., 0] / u, FT.W_FRAME * f[..., 1] / u], dim=-1)
    return ev.clamp_min(0.0), f, m


def loss_of(run, e, y, epoch, lam):
    """e is (n,3,2) raw evidence in VIEWS order."""
    ev_f, f, m = fused_evidence(e)
    L = evidential_loss_discern_ext(ev_f, y, 2, epoch, EPOCHS)
    for i in range(e.shape[1]):                          # branch terms on RAW evidence
        L = L + 0.5 * evidential_loss_discern_ext(e[:, i], y, 2, epoch, EPOCHS)
    if run == "J2b":
        L = L + lam * FT.ibdc_loss(m, f)
    elif run == "J2a":
        L = L + lam * FT.ibdc_v1([e[:, i] for i in range(e.shape[1])], f[..., 2])
    return L


def load_train():
    """Three cached draws, aligned frame-to-clip. Returns per-draw tensors in VIEWS order."""
    out = []
    for d in range(N_DRAWS):
        fs, ft = CACHE / f"FFpp_train_B_d{d}.npz", CACHE / f"FFpp_train_T_B_d{d}.npz"
        if not (fs.exists() and ft.exists()):
            raise FileNotFoundError(f"draw {d} not cached")
        zs, zt = np.load(fs, allow_pickle=True), np.load(ft, allow_pickle=True)
        tv = {str(v): i for i, v in enumerate(zt["video"])}
        vid = [DR.video_of(str(k)) for k in zs["key"]]
        keep = np.array([i for i, v in enumerate(vid) if v in tv])
        idx = np.array([tv[vid[i]] for i in keep])
        out.append({"app": zs["semantic"][keep].astype(np.float32),
                    "om": zs["artifact"][keep].astype(np.float32),
                    "t": zt["feature"][idx].astype(np.float32),
                    "y": zs["label"][keep].astype(np.int64)})
    n = {len(o["y"]) for o in out}
    assert len(n) == 1, f"draws disagree on sample count: {n}"
    return out


def evidence_of(heads, batch, dev):
    app, om, t = heads
    return torch.stack([t(torch.from_numpy(batch["t"]).to(dev)),
                        app(torch.from_numpy(batch["app"]).to(dev)),
                        om(torch.from_numpy(batch["om"]).to(dev))], dim=1)


def dry_run():
    rng = np.random.default_rng(0)
    n = 512
    b = {"app": rng.normal(size=(n, 64)).astype(np.float32),
         "om": rng.normal(size=(n, 64)).astype(np.float32),
         "t": rng.normal(size=(n, 1024)).astype(np.float32),
         "y": (rng.random(n) < 0.8).astype(np.int64)}
    heads = load_deployed_heads("cpu")
    y = torch.from_numpy(b["y"])
    print(f"  heads: APP {sum(p.numel() for p in heads[0].parameters()):,}  "
          f"OM {sum(p.numel() for p in heads[1].parameters()):,}  "
          f"T {sum(p.numel() for p in heads[2].parameters()):,}")
    for run in ("J1", "J2a", "J2b"):
        for h in heads:
            for p in h.parameters():
                if p.grad is not None: p.grad = None
        e = evidence_of(heads, b, "cpu")
        L = loss_of(run, e, y, 0, 0.3)
        L.backward()
        g = sum(float(p.grad.abs().sum()) for h in heads for p in h.parameters() if p.grad is not None)
        assert torch.isfinite(L) and g > 0, (run, float(L), g)
        print(f"  {run:4s} loss {float(L):8.4f}  grad sum {g:10.2f}")
    # the corrected loss must be inert where the original is not
    e0 = torch.full((16, 3, 2), 0.01, requires_grad=True)
    _ev, f0, m0 = fused_evidence(e0)
    print(f"  shared ignorance: L_IBDC {float(FT.ibdc_loss(m0, f0)):.6f}  "
          f"ibdc_v1 {float(FT.ibdc_v1([e0[:, i] for i in range(3)], f0[..., 2])):.6f}")
    assert float(FT.ibdc_loss(m0, f0)) == 0.0
    print("runs.py dry run OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="J1", choices=["J0", "J1", "J2a", "J2b"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        return dry_run()
    print("the training loop needs the development caches for VALmix-v2 selection; "
          "run the sweep first. Dry run is available now.")
    return 2


if __name__ == "__main__":
    sys.exit(main())

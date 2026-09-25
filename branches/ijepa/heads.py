#!/usr/bin/env python3
"""Evidential heads for the I-JEPA rows R2, R2b, R3, R4 and R5.

ARCHITECTURE is the brief's: a shared linear map takes each block vector to 64 dimensions, the five
projections are concatenated with five scalars, and the head is Linear(.,256) -> GELU -> Dropout(0.1)
-> Linear(256,2) -> softplus. R2 gets a 1280->325 map first so the three controls have matched
capacity. R5 appends the amendment's 4 video-level agreement inputs.

| row | scalars                | vectors        | head input |
|-----|------------------------|----------------|------------|
| R2  | --                     | pooled (1280)  | 325        |
| R2b | log \\|\\|s_bar_k\\|\\|       | s_bar_k        | 325        |
| R3  | log eps_k              | r_bar_k        | 325        |
| R4  | a_k (standardised)     | r_bar_k        | 325        |
| R5  | a_k + 4 agreement      | r_bar_k        | 329        |

TRAINING RECIPE is the deployed Off-manifold head's, confirmed from the code rather than assumed:
the digamma evidential loss with a KL-to-uniform term on non-target evidence annealed as
min(1, epoch/20) (`DISCERN_Ext/src/losses/evidential_loss.py`), AdamW at lr 1e-4, weight decay 0.01,
betas (0.9, 0.999), cosine to 1e-6, 20 epochs, batch 128 (`warm_dicome_gate_seed42.yaml`).

CLASS RATIO IS LEFT ALONE. FF++ train is 23,039 real against 92,159 fake and the deployed pipeline
has no balanced sampler -- `src/dataset/deepfake.py` only reports the ratio. Balancing here would be
a different recipe from the head this row is supposed to be compared against.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NBLOCK, VEC, PROJ = 5, 1280, 64
BASE_IN = NBLOCK + NBLOCK * PROJ          # 325


class IJepaHead(nn.Module):
    """One row's head.

    `mode` is 'blocks' (R2b, R3, R4, R5), 'pooled' (R2), 'scalars' (no residual directions at all)
    or 'both' (the pooled backbone feature AND the residual directions, which asks whether the
    residual ADDS to the backbone rather than whether it BEATS it).

    `proj` sets the shared projection width. G2 showed the head overfits FF++-specific residual
    directions -- in-domain AUROC climbing for twenty epochs while the cross-dataset score turns over
    at epoch 2 -- so the width is the capacity knob for that.
    """

    def __init__(self, mode: str = "blocks", extra: int = 0, dropout: float = 0.1, proj: int = PROJ):
        super().__init__()
        if mode not in ("blocks", "pooled", "scalars", "both"):
            raise ValueError(mode)
        self.mode, self.extra, self.proj_dim = mode, extra, proj
        if mode in ("blocks", "both"):
            self.proj = nn.Linear(VEC, proj)          # shared across the five blocks
        if mode in ("pooled", "both"):
            self.pre = nn.Linear(VEC, BASE_IN if mode == "pooled" else proj)
        in_dim = {"blocks": NBLOCK + NBLOCK * proj,
                  "pooled": BASE_IN,
                  "scalars": NBLOCK,
                  "both": NBLOCK + NBLOCK * proj + proj}[mode]
        self.fc1 = nn.Linear(in_dim + extra, 256)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, scal, vec, extra=None, pooled=None):
        """scal (B,5); vec (B,5,1280) block vectors, or (B,1280) for 'pooled'; pooled (B,1280)."""
        if self.mode == "blocks":
            B = vec.shape[0]
            z = self.proj(vec.reshape(B * NBLOCK, VEC)).reshape(B, NBLOCK * self.proj_dim)
            x = torch.cat([scal, z], dim=1)
        elif self.mode == "scalars":
            x = scal
        elif self.mode == "both":
            B = vec.shape[0]
            z = self.proj(vec.reshape(B * NBLOCK, VEC)).reshape(B, NBLOCK * self.proj_dim)
            x = torch.cat([scal, z, self.pre(pooled)], dim=1)
        else:
            x = self.pre(vec)
        if self.extra:
            if extra is None:
                raise ValueError("this row expects extra inputs")
            x = torch.cat([x, extra], dim=1)
        return F.softplus(self.fc2(self.drop(F.gelu(self.fc1(x)))))

    def n_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def probs(evidence: torch.Tensor):
    """DISCERN's read-out: alpha = evidence + 1, p = alpha / sum(alpha), u = K / sum(alpha)."""
    alpha = evidence + 1.0
    s = alpha.sum(1, keepdim=True)
    return (alpha / s), (alpha.shape[1] / s).squeeze(1)


ROWS = {
    "R2":  dict(mode="pooled", extra=0, scal=None,   vec="pooled", desc="plain I-JEPA probe"),
    "R2b": dict(mode="blocks", extra=0, scal="snorm", vec="sbar",  desc="regional feature control"),
    "R3":  dict(mode="blocks", extra=0, scal="logeps", vec="rbar", desc="residual, unstandardised"),
    "R4":  dict(mode="blocks", extra=0, scal="a",     vec="rbar",  desc="residual, standardised"),
    "R5":  dict(mode="blocks", extra=4, scal="a",     vec="rbar",  desc="R4 + cross-block agreement"),
}


def _selftest():
    torch.manual_seed(0)
    B = 7
    for row, cfg in ROWS.items():
        m = IJepaHead(cfg["mode"], cfg["extra"])
        if cfg["mode"] == "pooled":
            ev = m(None, torch.randn(B, VEC))
        else:
            ex = torch.randn(B, cfg["extra"]) if cfg["extra"] else None
            ev = m(torch.randn(B, NBLOCK), torch.randn(B, NBLOCK, VEC), ex)
        assert ev.shape == (B, 2), (row, ev.shape)
        assert (ev >= 0).all(), f"{row}: softplus must give non-negative evidence"
        p, u = probs(ev)
        assert torch.allclose(p.sum(1), torch.ones(B), atol=1e-5)
        assert ((u > 0) & (u <= 1.0)).all(), u
        print(f"  {row:4s} {cfg['desc']:28s} input {BASE_IN + cfg['extra']:3d}  "
              f"trainable {m.n_trainable():,}")

    # the shared projection really is shared: one weight, applied to all five blocks
    m = IJepaHead("blocks")
    assert m.proj.weight.shape == (PROJ, VEC)
    # and permuting blocks changes the output, i.e. block identity is preserved by concatenation
    v = torch.randn(3, NBLOCK, VEC); s = torch.randn(3, NBLOCK)
    m.eval()
    a = m(s, v)
    b = m(s[:, [1, 0, 2, 3, 4]], v[:, [1, 0, 2, 3, 4]])
    assert not torch.allclose(a, b), "block order must matter"

    # maximum uncertainty when there is no evidence
    p, u = probs(torch.zeros(1, 2))
    assert torch.allclose(u, torch.tensor([1.0])) and torch.allclose(p, torch.tensor([[0.5, 0.5]]))
    print("heads.py self-test OK")


if __name__ == "__main__":
    _selftest()

#!/usr/bin/env python3
"""Sequence heads T0-T3 over the frozen I-JEPA residual cache (Amendment 2).

    python branches/ijepa/seq_heads.py            # self-test, including padding equivalence

WHAT THESE ARE NOT. The cached frames are 0.17 to 1.48 seconds apart depending on the dataset
(`analysis/ijepa/amendment2_cache_check.json`), so nothing here observes motion. T2 and T3 aggregate
over SAMPLED frames in order. They are never to be described as temporal dynamics.

PER-FRAME INPUT. h_t = [a_1..a_5, shared 1280->64 projection of r_bar_1..r_bar_5] and, when R5 was
kept, that row's per-frame cosine and validity inputs. A linear layer takes h_t to 64. The five
REGIONS are a fixed vector, never a sequence; only frames are a sequence.

HEADS. Each produces a 64-d video representation, then the same classifier
(64 -> 256, GELU, dropout 0.1, 256 -> 2, softplus evidence).
  T0  per-frame projection, masked mean pooling                     order-invariant, sanity check
  T1  two-layer residual per-frame MLP, masked attention pooling    order-invariant, learned
  T2  two residual 1-D convolutions, kernel 3, masked mean pooling  order-sensitive
  T3  two Mamba layers, d_state 16, masked mean pooling             order-sensitive

PARAMETER MATCHING. Widths are solved so the sequence-and-aggregation modules agree to within 0.96%,
against the amendment's 15% limit: T1 hidden 468, T2 channels 84, T3 expand 2 -> 64,661 / 64,808 /
65,280 parameters. T0 has no sequence parameters by construction.

PADDING. Right padding only, with a validity mask.
  * pooling excludes padded positions;
  * T2 zeroes padded positions BEFORE each convolution, so a padded column cannot leak into a valid
    one through a symmetric kernel;
  * T3's convolution is causal (left pad only) and its scan runs forward with state reset per video,
    so a right-padded tail cannot reach an earlier position.
`test_padding_equivalence` checks the consequence rather than trusting the argument: every head must
give the same video prediction alone and inside a padded batch.

T3 USES THE REFERENCE SELECTIVE SCAN. mamba_ssm imports in none of the environments on this machine,
which the amendment anticipates; the choice is recorded in SEQ_BACKEND.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = 64
T1_HIDDEN, T1_ATT = 468, 64
T2_CH = 84
T3_LAYERS, T3_STATE, T3_CONV, T3_EXPAND = 2, 16, 4, 2

try:                                              # recorded, not assumed
    import mamba_ssm  # noqa: F401
    SEQ_BACKEND = "mamba_ssm"
except Exception:
    SEQ_BACKEND = "reference_pytorch_selective_scan"


# --------------------------------------------------------------------------- shared pieces
class InputLayer(nn.Module):
    """[a_k, shared projection of r_bar_k, optional R5 per-frame inputs] -> 64."""

    def __init__(self, n_block: int = 5, vec: int = 1280, proj: int = 64, extra: int = 0):
        super().__init__()
        self.n_block, self.vec, self.extra = n_block, vec, extra
        self.proj = nn.Linear(vec, proj)                       # shared across the five regions
        self.to_model = nn.Linear(n_block + n_block * proj + extra, D_MODEL)

    def forward(self, scal, vecs, extra=None):
        """scal (B,L,5); vecs (B,L,5,1280); extra (B,L,4) or None -> (B,L,64)."""
        B, L = scal.shape[:2]
        z = self.proj(vecs.reshape(B * L * self.n_block, self.vec)).reshape(B, L, -1)
        parts = [scal, z] + ([extra] if self.extra else [])
        return self.to_model(torch.cat(parts, dim=-1))


class Classifier(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.fc1, self.drop, self.fc2 = nn.Linear(D_MODEL, 256), nn.Dropout(dropout), nn.Linear(256, 2)

    def forward(self, v):
        return F.softplus(self.fc2(self.drop(F.gelu(self.fc1(v)))))


def masked_mean(x, mask):
    """x (B,L,D), mask (B,L) with 1 on valid frames."""
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(1) / m.sum(1).clamp_min(1.0)


# --------------------------------------------------------------------------- sequence modules
class T0Seq(nn.Module):
    """Masked mean pooling. No parameters, order-invariant."""

    def forward(self, x, mask):
        return masked_mean(x, mask)


class T1Seq(nn.Module):
    """Residual per-frame MLP, then masked attention pooling with no positional encoding."""

    def __init__(self, hidden=T1_HIDDEN, att=T1_ATT):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(D_MODEL, hidden), nn.Linear(hidden, D_MODEL)
        self.att1, self.att2 = nn.Linear(D_MODEL, att), nn.Linear(att, 1)

    def forward(self, x, mask):
        x = x + self.fc2(F.gelu(self.fc1(x)))                  # per frame, so still order-invariant
        s = self.att2(torch.tanh(self.att1(x))).squeeze(-1)    # (B,L)
        s = s.masked_fill(mask == 0, float("-inf"))
        w = torch.softmax(s, dim=1)
        w = torch.nan_to_num(w, nan=0.0)                       # a fully padded row cannot produce NaN
        return (x * w.unsqueeze(-1)).sum(1)


class T2Seq(nn.Module):
    """Two residual 1-D convolutions, kernel 3, then masked mean pooling.

    Padded positions are zeroed BEFORE each convolution. With a symmetric kernel a padded column at
    t+1 would otherwise contribute to the valid position t, which is exactly the leak the amendment
    forbids.
    """

    def __init__(self, ch=T2_CH):
        super().__init__()
        self.a1, self.a2 = nn.Conv1d(D_MODEL, ch, 3, padding=1), nn.Conv1d(ch, D_MODEL, 3, padding=1)
        self.b1, self.b2 = nn.Conv1d(D_MODEL, ch, 3, padding=1), nn.Conv1d(ch, D_MODEL, 3, padding=1)

    def forward(self, x, mask):
        m = mask.unsqueeze(-1).to(x.dtype)
        for c1, c2 in ((self.a1, self.a2), (self.b1, self.b2)):
            h = (x * m).transpose(1, 2)                        # zero the pad before the convolution
            h = c1(h)
            h = F.gelu(h)
            h = (h.transpose(1, 2) * m).transpose(1, 2)        # and again between the two
            h = c2(h).transpose(1, 2)
            x = x + h * m
        return masked_mean(x, mask)


class MambaBlock(nn.Module):
    """One Mamba layer with the reference PyTorch selective scan.

    The depthwise convolution is CAUSAL (left padding only) and the scan runs strictly forward, so a
    right-padded tail can never influence an earlier position. State starts at zero for every video,
    which is the per-video reset the amendment asks for.
    """

    def __init__(self, d_model=D_MODEL, d_state=T3_STATE, d_conv=T3_CONV, expand=T3_EXPAND):
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state, self.d_conv = d_state, d_conv
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)
                                            .repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x, mask):
        B, L, _ = x.shape
        m = mask.unsqueeze(-1).to(x.dtype)
        xz = self.in_proj(x * m)
        xi, z = xz.chunk(2, dim=-1)
        h = (xi * m).transpose(1, 2)
        h = F.pad(h, (self.d_conv - 1, 0))                     # causal: pad the PAST only
        xi = F.silu(self.conv1d(h)[:, :, :L].transpose(1, 2)) * m

        dbl = self.x_proj(xi)
        dt, Bm, Cm = torch.split(dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                      # (B,L,d_inner)
        A = -torch.exp(self.A_log)                             # (d_inner,d_state)

        dA = torch.exp(dt.unsqueeze(-1) * A)                   # (B,L,d_inner,d_state)
        dBx = dt.unsqueeze(-1) * Bm.unsqueeze(2) * xi.unsqueeze(-1)
        s = x.new_zeros(B, self.d_inner, self.d_state)
        ys = []
        for t in range(L):                                     # forward only; L is 32 here
            s = dA[:, t] * s + dBx[:, t]
            ys.append((s * Cm[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1) + self.D * xi
        return self.out_proj(y * F.silu(z)) * m


class T3Seq(nn.Module):
    def __init__(self, layers=T3_LAYERS):
        super().__init__()
        self.blocks = nn.ModuleList([MambaBlock() for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(D_MODEL) for _ in range(layers)])

    def forward(self, x, mask):
        for blk, nrm in zip(self.blocks, self.norms):
            x = x + blk(nrm(x), mask)
        return masked_mean(x, mask)


SEQS = {"T0": T0Seq, "T1": T1Seq, "T2": T2Seq, "T3": T3Seq}
ORDER_INVARIANT = {"T0", "T1"}


class SeqHead(nn.Module):
    """One run: its own input layers, its own sequence module, its own classifier. Nothing shared."""

    def __init__(self, kind: str, extra: int = 0, dropout: float = 0.1):
        super().__init__()
        if kind not in SEQS:
            raise ValueError(kind)
        self.kind = kind
        self.inp = InputLayer(extra=extra)
        self.seq = SEQS[kind]()
        self.cls = Classifier(dropout)

    def forward(self, scal, vecs, mask, extra=None):
        return self.cls(self.seq(self.inp(scal, vecs, extra), mask))

    def param_counts(self):
        tot = sum(p.numel() for p in self.parameters() if p.requires_grad)
        seq = sum(p.numel() for p in self.seq.parameters() if p.requires_grad)
        return {"total": tot, "sequence_and_aggregation": seq,
                "input": sum(p.numel() for p in self.inp.parameters()),
                "classifier": sum(p.numel() for p in self.cls.parameters())}


# --------------------------------------------------------------------------- self-test
def _selftest():
    torch.manual_seed(0)
    B, L, NB, V = 6, 32, 5, 1280
    print(f"sequence backend for T3: {SEQ_BACKEND}")

    counts = {}
    for k in SEQS:
        m = SeqHead(k)
        counts[k] = m.param_counts()
        print(f"  {k}  total {counts[k]['total']:>8,}   sequence+aggregation "
              f"{counts[k]['sequence_and_aggregation']:>7,}")
    matched = [counts[k]["sequence_and_aggregation"] for k in ("T1", "T2", "T3")]
    spread = (max(matched) - min(matched)) / min(matched)
    assert spread <= 0.15, f"T1/T2/T3 differ by {spread:.1%}, limit is 15%"
    print(f"  T1/T2/T3 sequence-module spread {spread:.2%} (limit 15%)")
    assert counts["T0"]["sequence_and_aggregation"] == 0

    scal = torch.randn(B, L, NB)
    vecs = torch.randn(B, L, NB, V)
    lens = torch.tensor([32, 31, 20, 8, 3, 1])
    mask = (torch.arange(L)[None, :] < lens[:, None]).float()

    # padding equivalence: a video alone must equal the same video inside a padded batch
    print("\n  padding equivalence (video alone vs inside a padded batch):")
    for k in SEQS:
        m = SeqHead(k).eval()
        with torch.no_grad():
            full = m(scal, vecs, mask)
            worst = 0.0
            for i in range(B):
                n = int(lens[i])
                one = m(scal[i:i + 1, :n], vecs[i:i + 1, :n], torch.ones(1, n))
                worst = max(worst, float((one - full[i:i + 1]).abs().max()))
        assert worst < 1e-5, f"{k}: padded batch changes the prediction by {worst:.2e}"
        print(f"    {k}  max abs diff {worst:.2e}")

    # order sensitivity: T0 and T1 must be invariant, T2 and T3 must not be
    print("\n  order sensitivity (same frames, permuted):")
    perm = torch.randperm(L)
    for k in SEQS:
        m = SeqHead(k).eval()
        with torch.no_grad():
            a = m(scal, vecs, torch.ones(B, L))
            b = m(scal[:, perm], vecs[:, perm], torch.ones(B, L))
        d = float((a - b).abs().max())
        if k in ORDER_INVARIANT:
            assert d < 1e-5, f"{k} must be order-invariant, got {d:.2e}"
            print(f"    {k}  invariant, max abs diff {d:.2e}")
        else:
            assert d > 1e-4, f"{k} should depend on order, got {d:.2e}"
            print(f"    {k}  order-sensitive, max abs diff {d:.2e}")

    # evidence is non-negative and gradients reach every module
    for k in SEQS:
        m = SeqHead(k)
        ev = m(scal, vecs, mask)
        assert ev.shape == (B, 2) and (ev >= 0).all()
        ev.sum().backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.inp.parameters())
        if k != "T0":
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.seq.parameters()), k

    # the R5 variant carries four extra per-frame inputs
    m = SeqHead("T2", extra=4)
    ev = m(scal, vecs, mask, torch.randn(B, L, 4))
    assert ev.shape == (B, 2)
    print("\nseq_heads.py self-test OK")


if __name__ == "__main__":
    _selftest()

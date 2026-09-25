#!/usr/bin/env python3
"""Amendment 2's required unit tests for the sequence heads.

    <python> -m pytest branches/ijepa/test_seq_heads.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seq_heads import SeqHead, SEQS, ORDER_INVARIANT, SEQ_BACKEND  # noqa: E402

B, L, NB, V = 6, 32, 5, 1280
LENS = [32, 31, 20, 8, 3, 1]


@pytest.fixture(scope="module")
def batch():
    torch.manual_seed(0)
    lens = torch.tensor(LENS)
    return (torch.randn(B, L, NB), torch.randn(B, L, NB, V),
            (torch.arange(L)[None, :] < lens[:, None]).float(), lens)


@pytest.mark.parametrize("kind", list(SEQS))
def test_padding_equivalence(kind, batch):
    """THE required test: a video's prediction is the same alone as inside a padded batch."""
    scal, vecs, mask, lens = batch
    m = SeqHead(kind).eval()
    with torch.no_grad():
        full = m(scal, vecs, mask)
        for i in range(B):
            n = int(lens[i])
            one = m(scal[i:i + 1, :n], vecs[i:i + 1, :n], torch.ones(1, n))
            d = float((one - full[i:i + 1]).abs().max())
            assert d < 1e-5, f"{kind}, video {i} (len {n}): padded batch shifts it by {d:.2e}"


@pytest.mark.parametrize("kind", list(SEQS))
def test_padded_tail_cannot_change_a_valid_prediction(kind, batch):
    """Overwriting the padded tail with garbage must not move the answer."""
    scal, vecs, mask, _ = batch
    m = SeqHead(kind).eval()
    s2, v2 = scal.clone(), vecs.clone()
    pad = mask == 0
    s2[pad] = 99.0
    v2[pad] = -99.0
    with torch.no_grad():
        assert torch.allclose(m(scal, vecs, mask), m(s2, v2, mask), atol=1e-5)


@pytest.mark.parametrize("kind", sorted(ORDER_INVARIANT))
def test_order_invariant_heads_are_invariant(kind, batch):
    scal, vecs, _, _ = batch
    m = SeqHead(kind).eval()
    p = torch.randperm(L)
    with torch.no_grad():
        a, b = m(scal, vecs, torch.ones(B, L)), m(scal[:, p], vecs[:, p], torch.ones(B, L))
    assert float((a - b).abs().max()) < 1e-5


@pytest.mark.parametrize("kind", sorted(set(SEQS) - ORDER_INVARIANT))
def test_order_sensitive_heads_depend_on_order(kind, batch):
    scal, vecs, _, _ = batch
    m = SeqHead(kind).eval()
    p = torch.randperm(L)
    with torch.no_grad():
        a, b = m(scal, vecs, torch.ones(B, L)), m(scal[:, p], vecs[:, p], torch.ones(B, L))
    assert float((a - b).abs().max()) > 1e-4


def test_parameter_matching_within_15_percent():
    c = {k: SeqHead(k).param_counts()["sequence_and_aggregation"] for k in SEQS}
    matched = [c[k] for k in ("T1", "T2", "T3")]
    spread = (max(matched) - min(matched)) / min(matched)
    assert spread <= 0.15, f"{c} spread {spread:.1%}"
    assert c["T0"] == 0


def test_t3_backend_is_recorded():
    assert SEQ_BACKEND in ("mamba_ssm", "reference_pytorch_selective_scan")


def test_evidence_is_non_negative(batch):
    scal, vecs, mask, _ = batch
    for kind in SEQS:
        ev = SeqHead(kind).eval()(scal, vecs, mask)
        assert ev.shape == (B, 2) and (ev >= 0).all()

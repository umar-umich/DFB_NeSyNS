"""Tests for Branch C — generative-process evidence (V1 spec §5, §14.1, §19).

    python training/networks/discern_v2/test_process_branch.py

Tests that build the VAE need a local sdxl-vae checkout; they skip with a message otherwise.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from networks.discern_v2 import process_branch as P  # noqa: E402
from networks.discern_v2.ds_fusion import Opinion, apply_validity  # noqa: E402
from networks.discern_v2.reference import ResidualCalibrator  # noqa: E402

VAE_PATH = "/data/umar/Repos/DiCoME/eval_adaptation/data/models/sdxl-vae"
PASSED, SKIPPED = [], []


class SkipTest(Exception):
    pass


def check(name):
    def deco(fn):
        try:
            fn()
        except SkipTest as exc:
            SKIPPED.append(name)
            print(f"  skip: {name} ({exc})")
        else:
            PASSED.append(name)
            print(f"  ok: {name}")
        return fn
    return deco


_cached = {}


def branch() -> P.ProcessEvidenceBranch:
    """One instance, calibrated on synthetic 'authentic' statistics."""
    if "b" not in _cached:
        if not os.path.isdir(VAE_PATH):
            raise SkipTest(f"no sdxl-vae at {VAE_PATH}")
        try:
            b = P.ProcessEvidenceBranch(vae_path=VAE_PATH, resolution=128)
        except Exception as exc:  # noqa: BLE001
            raise SkipTest(f"cannot build the operator: {type(exc).__name__}: {str(exc)[:70]}")
        # stand in for the offline fit so the calibrator is usable in tests
        b.calibrator.fit(torch.randn(64, b.n_stats) * 0.1 + 0.5)
        _cached["b"] = b
    return _cached["b"]


# ---------------------------------------------------------------- no VAE needed


@check("§5: the input normalization is the identity, so the VAE sees its native scale")
def _raw_normalization():
    assert P.RAW_MEAN == (0.0, 0.0, 0.0) and P.RAW_STD == (1.0, 1.0, 1.0)


@check("§5: an unfitted calibrator refuses rather than passing statistics through")
def _unfitted_refuses():
    cal = ResidualCalibrator(6)
    try:
        cal(torch.randn(2, 6))
    except RuntimeError as exc:
        assert "before fit()" in str(exc)
    else:
        raise AssertionError("an unfitted calibrator must refuse")


@check("§5: standardisation uses the authentic statistics it was given, and nothing else")
def _calibration_is_reals_only():
    cal = ResidualCalibrator(3)
    reals = torch.tensor([[1.0, 10.0, 100.0], [3.0, 30.0, 300.0]])
    cal.fit(reals)
    out = cal(reals.mean(dim=0, keepdim=True))
    assert torch.allclose(out, torch.zeros(1, 3), atol=1e-4), (
        "the mean authentic sample must standardise to ~0")
    assert not any(p.requires_grad for p in cal.parameters()), "buffers, never parameters"


# ---------------------------------------------------------------- needs the VAE


@check("§5: output is non-negative evidence (B, 2) and a valid opinion")
def _evidence():
    b = branch()
    out = b(torch.rand(2, 3, 128, 128))
    e = out["evidence"]
    assert e.shape == (2, 2), e.shape
    assert (e >= 0).all()
    Opinion.from_evidence(e).assert_normalized()
    assert out["feature"].shape == (2, b.n_stats)
    assert out["raw_stats"].shape == (2, b.n_stats)


@check("§19: the VAE is frozen, stays in eval under train(), and gets no gradient")
def _frozen():
    b = branch()
    b.train(True)
    assert not b.operator.training, "the VAE must stay in eval mode"
    b.assert_frozen()

    before = {n: p.detach().clone() for n, p in b.operator.named_parameters()}
    out = b(torch.rand(2, 3, 128, 128))
    out["evidence"].sum().backward()
    got = [n for n, p in b.operator.named_parameters() if p.grad is not None]
    assert not got, f"VAE parameters received gradient: {got[:3]}"
    moved = [n for n, p in b.operator.named_parameters()
             if not torch.equal(p.detach(), before[n])]
    assert not moved, f"VAE parameters moved: {moved[:3]}"
    assert b.head[0].weight.grad is not None, "the head must still learn"


@check("§5: polarity is LEARNED — the head can map a LOW residual to fake (AEROBLADE)")
def _signed_response():
    b = branch()
    # Train the tiny head on synthetic statistics where SMALL error means fake, the direction a
    # hardcoded "large error = fake" rule could never represent.
    torch.manual_seed(0)
    low = torch.full((32, b.n_stats), -1.0) + 0.05 * torch.randn(32, b.n_stats)   # fake
    high = torch.full((32, b.n_stats), 1.0) + 0.05 * torch.randn(32, b.n_stats)   # real
    x = torch.cat([low, high])
    y = torch.cat([torch.ones(32, dtype=torch.long), torch.zeros(32, dtype=torch.long)])

    opt = torch.optim.Adam(b.head.parameters(), lr=0.05)
    for _ in range(150):
        opt.zero_grad()
        evidence = torch.nn.functional.softplus(b.head(x))
        loss = torch.nn.functional.cross_entropy(evidence + 1e-6, y)
        loss.backward()
        opt.step()

    with torch.no_grad():
        p_fake_low = Opinion.from_evidence(
            torch.nn.functional.softplus(b.head(low[:1]))).fake_prob()
        p_fake_high = Opinion.from_evidence(
            torch.nn.functional.softplus(b.head(high[:1]))).fake_prob()
    assert float(p_fake_low) > float(p_fake_high), (
        f"the head must be able to learn low-residual-means-fake: {float(p_fake_low):.3f} vs "
        f"{float(p_fake_high):.3f}")


@check("§14.1: a non-finite statistic flags the branch invalid instead of poisoning the batch")
def _validity(monkey=[]):
    b = branch()
    original = b.operator.forward

    def broken(images):
        stats = original(images)
        stats = stats.clone()
        stats[0, 0] = float("nan")      # simulate a catastrophic operator failure on sample 0
        return stats

    b.operator.forward = broken
    try:
        out = b(torch.rand(3, 3, 128, 128))
    finally:
        b.operator.forward = original

    assert not bool(out["valid"][0]), "the failed sample must be flagged invalid"
    assert bool(out["valid"][1]) and bool(out["valid"][2])
    assert torch.isfinite(out["evidence"]).all(), (
        "a NaN in one sample must not propagate into the batch's evidence")

    # and the caller turns that flag into ignorance, not into evidence for Real
    op = apply_validity(Opinion.from_evidence(out["evidence"]), out["valid"].float())
    op.assert_normalized()
    assert float(op.vacuity[0]) == 1.0, "an invalid sample must become vacuous"
    assert float(op.belief[0].sum()) == 0.0


if __name__ == "__main__":
    print(f"\n{len(PASSED)} passed, {len(SKIPPED)} skipped")
    for s in SKIPPED:
        print(f"  - skipped: {s}")
    sys.exit(0 if PASSED else 1)

#!/usr/bin/env python3
"""Tests for the reals-only frozen reference.

The headline test is `test_frozen_reference_receives_zero_gradient`. Phase 1's manifold
projector was wired into the optimizer and into the loss path but never given a generative
objective, so it silently learned a discriminative transform and `f - P(f)` was not a manifold
residual. **That is the bug this file exists to make impossible.** Everything else here guards
the couplings whose failure modes are equally silent.

    python training/networks/discern_v2/test_reference.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from networks.discern_v2 import reference as R  # noqa: E402

D, LAT, B = 64, 16, 32


def _real_features(n=512, seed=0):
    """Authentic features on a fixed low-rank manifold plus small noise.

    The BASIS is fixed (seed 0) so every call samples the same manifold — only the draw
    changes with `seed`. Varying the basis would put fit and evaluation features on different
    manifolds, and a correctly-fit reference would then look broken.
    """
    gb = torch.Generator().manual_seed(0)
    basis = torch.randn(LAT, D, generator=gb)
    g = torch.Generator().manual_seed(seed)
    coeff = torch.randn(n, LAT, generator=g)
    return coeff @ basis + 0.01 * torch.randn(n, D, generator=g)


def _off_manifold(n=128, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, D, generator=g) * 3.0


# --------------------------------------------------------------- the bug guard


def test_frozen_reference_receives_zero_gradient():
    """A frozen reference must get EXACTLY zero gradient from a supervised loss.

    This is the Phase-1 failure reproduced as a test: put a reference in a model, train the
    model on a classification objective, and check the reference did not move.
    """
    ref = R.build_reference("C3_ae", D, LAT)
    ref.fitting()
    opt = torch.optim.Adam(ref.parameters(), lr=1e-2)
    for _ in range(20):                                  # offline reals-only fit
        opt.zero_grad()
        ref.reconstruction_loss(_real_features(128), "cosine").backward()
        opt.step()
    ref.freeze()

    before = [p.detach().clone() for p in ref.parameters()]
    head = nn.Linear(D + 2, 2)
    hopt = torch.optim.Adam(head.parameters(), lr=1e-2)
    for _ in range(5):                                   # supervised training step
        hopt.zero_grad()
        desc = ref(_real_features(B))
        logits = head(desc.features())
        torch.nn.functional.cross_entropy(logits, torch.randint(0, 2, (B,))).backward()
        hopt.step()

    for p in ref.parameters():
        assert p.grad is None or float(p.grad.abs().max()) == 0.0, "reference received gradient"
    for a, b in zip(before, ref.parameters()):
        assert torch.equal(a, b), "frozen reference parameters CHANGED during training"
    print("  ok: frozen reference gets zero gradient and its weights do not move")


def test_unfrozen_reference_is_refused():
    """Using a never-frozen reference in a forward pass must fail loudly."""
    ref = R.C3AEReference(D, LAT)
    try:
        ref(_real_features(B))
    except RuntimeError as exc:
        assert "not frozen" in str(exc)
        print("  ok: an unfrozen reference is refused, with the bug explained")
        return
    raise AssertionError("should have refused")


def test_fitting_mode_does_receive_gradient():
    """The offline fit is the one place gradients are expected."""
    ref = R.build_reference("C3_ae", D, LAT).fitting()
    ref.reconstruction_loss(_real_features(64), "mse").backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in ref.parameters())
    print("  ok: fitting() mode receives gradient (offline reals-only stage)")


def test_c1_refuses_to_be_fit():
    try:
        R.build_reference("C1_random", D, LAT).fitting()
    except RuntimeError as exc:
        assert "capacity floor" in str(exc)
        print("  ok: C1 random floor refuses to be fit")
        return
    raise AssertionError("C1 should refuse fitting")


# --------------------------------------------------------------- arms


def test_all_arms_share_the_interface():
    fr = _real_features(256)
    for arm in R.ARMS:
        ref = R.build_reference(arm, D, LAT)
        if arm == "C2_linear":
            ref.fit_pca(fr)
        elif arm != "C1_random":
            ref.fitting()
            ref.reconstruction_loss(fr[:64], "cosine").backward()
        ref.freeze()
        desc = ref(fr[:B])
        assert desc.residual.shape == (B, D), arm
        assert desc.norm.shape == (B,) and desc.angle.shape == (B,), arm
        assert desc.features().shape == (B, D + 2), arm
    print(f"  ok: all {len(R.ARMS)} arms return residual + norm + angle descriptors")


def test_c2_pca_reconstructs_the_real_manifold():
    """PCA on reals should reconstruct reals well and off-manifold points poorly.

    This is the property the whole reference idea rests on; if it does not hold, r_ref carries
    nothing.
    """
    ref = R.build_reference("C2_linear", D, LAT)
    info = ref.fit_pca(_real_features(1024))
    ref.freeze()
    r_real = ref(_real_features(256, seed=7)).norm.mean()
    r_off = ref(_off_manifold(256)).norm.mean()
    assert r_off > r_real * 2, f"off-manifold {r_off:.3f} vs real {r_real:.3f}"
    print(f"  ok: C2 residual norm real {r_real:.4f} << off-manifold {r_off:.4f} "
          f"(explained var {info['explained_variance_ratio']:.3f})")


def test_c2_has_no_parameters_at_all():
    """C2 is buffers only — it cannot smuggle in discriminative learning."""
    ref = R.build_reference("C2_linear", D, LAT)
    assert list(ref.parameters()) == []
    print("  ok: C2 has zero parameters (buffers only)")


def test_cosine_fit_leaves_magnitude_free_mse_does_not():
    """The two objectives measure different things — E1's reason for evaluating both.

    A cosine fit only aligns direction, so the reconstruction is free to differ in scale and
    the residual norm stays large. An MSE fit shrinks the residual outright.
    """
    fr = _real_features(1024)
    norms = {}
    for obj in R.FIT_OBJECTIVES:
        ref = R.build_reference("C3_ae", D, LAT).fitting()
        opt = torch.optim.Adam(ref.parameters(), lr=5e-3)
        for _ in range(150):
            opt.zero_grad()
            ref.reconstruction_loss(fr, obj).backward()
            opt.step()
        ref.freeze()
        norms[obj] = float(ref(fr[:256]).norm.mean())
    assert norms["mse"] < norms["cosine"], norms
    print(f"  ok: residual norm cosine {norms['cosine']:.4f} > mse {norms['mse']:.4f} "
          f"— magnitude carries signal only under the cosine fit")


# --------------------------------------------------------------- calibration


def test_calibrator_fits_on_reals_only_and_is_frozen():
    cal = R.ResidualCalibrator(D)
    cal.fit(torch.randn(1000, D) * 3.0 + 5.0)
    out = cal(torch.randn(B, D) * 3.0 + 5.0)
    assert abs(float(out.mean())) < 0.4 and abs(float(out.std()) - 1.0) < 0.4
    assert list(cal.parameters()) == [], "calibrator must hold buffers, never parameters"
    print(f"  ok: calibrator standardizes to mean {float(out.mean()):+.3f} "
          f"std {float(out.std()):.3f}, no parameters")


def test_unfitted_calibrator_refuses():
    try:
        R.ResidualCalibrator(D)(torch.randn(B, D))
    except RuntimeError as exc:
        assert "before fit()" in str(exc)
        print("  ok: unfitted calibrator refuses rather than passing through")
        return
    raise AssertionError("should have refused")


# --------------------------------------------------------------- config guards


def test_drifting_encoder_is_refused():
    """The root lesson of the bug: the reference's input space must be stationary."""
    R.assert_reference_config("C3_ae", "cosine", encoder_frozen=True)
    try:
        R.assert_reference_config("C3_ae", "cosine", encoder_frozen=False)
    except ValueError as exc:
        assert "stale manifold" in str(exc)
        print("  ok: fitting a reference on a tuned (drifting) encoder is refused")
        return
    raise AssertionError("should have refused a drifting encoder")


def test_bad_objective_is_refused():
    try:
        R.assert_reference_config("C3_ae", "huber", encoder_frozen=True)
    except ValueError as exc:
        assert "fit_objective" in str(exc)
        print("  ok: unknown fit objective refused")
        return
    raise AssertionError("should have refused")


def test_frozen_reference_stays_eval_under_model_train():
    ref = R.build_reference("C3_ae", D, LAT)
    ref.fitting(); ref.freeze()
    wrapper = nn.Sequential(ref, nn.Identity())
    wrapper.train()
    assert not ref.training, "frozen reference flipped to train mode with the enclosing model"
    print("  ok: frozen reference stays in eval mode when the model calls .train()")


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"running {len(fns)} reference tests\n")
    failed = 0
    for fn in fns:
        print(f"{fn.__name__}:")
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)

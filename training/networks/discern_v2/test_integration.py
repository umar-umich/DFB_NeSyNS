#!/usr/bin/env python3
"""Tests for the DISCERN v2 -> detector integration.

The property everything else depends on: **with the v2 flags off, nothing changes**. D0 is
the reference the whole ladder is measured against, so if enabling the integration perturbed
the v1 path even slightly, every D1-D5 delta would be measuring the refactor rather than the
mechanism. That is checked here rather than argued.

    python training/networks/discern_v2/test_integration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from networks.discern_v2.integration import DiscernV2Stack, build_v2_stack  # noqa: E402

VIS_DIM, FEAT_DIM, BATCH = 1024, 64, 6


def _cfg(manifold=False, process=False, projector="beta_tcvae"):
    return {
        "manifold_v2": manifold,
        "process_v2": process,
        "discern_v2": {
            "visual": {"enabled": True, "feature_dim": VIS_DIM},
            "manifold": {"enabled": True, "feature_dim": FEAT_DIM, "latent_dim": 32,
                         "projector": projector, "input_dim": VIS_DIM,
                         "export_rate_response": False},
            "process": {"enabled": True, "vae_path": None},
        },
    }


def test_v1_run_builds_no_stack():
    """Both flags off -> None, so the detector's v2 path is skipped entirely."""
    assert build_v2_stack(_cfg(False, False)) is None
    print("  ok: v1 config produces no v2 stack (zero added params, zero added ops)")


def test_flags_are_master_switches():
    """A branch section cannot enable itself behind the top-level flag's back."""
    stack = build_v2_stack(_cfg(manifold=True, process=False))
    assert set(stack.branches) == {"manifold"}, set(stack.branches)
    print("  ok: process section stays off when process_v2 is false")


def test_visual_branch_never_doubles_the_spatial_head():
    """The v1 spatial head already supplies visual evidence."""
    stack = build_v2_stack(_cfg(manifold=True))
    assert "visual" not in stack.branches
    print("  ok: v2 visual branch is suppressed — no double-counted visual evidence")


def test_additive_and_gated():
    """v2 evidence is added on top of the fused v1 evidence, scaled by a sigmoid gate."""
    stack = build_v2_stack(_cfg(manifold=True))
    base = torch.rand(BATCH, 2) * 3
    vis = torch.randn(BATCH, VIS_DIM)
    out, diag = stack(base.clone(), visual_feature=vis)
    assert out.shape == base.shape
    assert (out >= base - 1e-6).all(), "v2 evidence must be additive, never subtractive"
    gate = diag["manifold_gate"]
    expected = base + gate * diag["manifold_evidence"]
    assert torch.allclose(out, expected, atol=1e-5)
    print(f"  ok: total = v1 + sigmoid(gate) * v2, gate={gate.item():.4f}")


def test_gate_starts_small():
    """A freshly enabled branch must not swamp the CLIP baseline in early epochs."""
    stack = build_v2_stack(_cfg(manifold=True))
    g = torch.sigmoid(stack.gates["manifold"]).item()
    assert g < 0.2, f"gate opens at {g:.3f}; a new branch should start as a perturbation"
    print(f"  ok: gate opens at {g:.4f} — the branch has to earn its weight")


def test_input_projection_matches_pilot_regime():
    """1024-d DISCERN feature is projected to the 64-d regime the projectors were built for."""
    stack = build_v2_stack(_cfg(manifold=True))
    man = stack.branches["manifold"]
    assert isinstance(man.input_proj, torch.nn.Linear)
    assert man.input_proj.in_features == VIS_DIM and man.input_proj.out_features == FEAT_DIM
    assert man.projector.feature_dim == FEAT_DIM
    out = man(torch.randn(BATCH, VIS_DIM))
    assert out.features.shape == (BATCH, FEAT_DIM), "residual must live in the pilot's width"
    print(f"  ok: {VIS_DIM} -> {FEAT_DIM} input projection, projector runs at pilot width")


def test_identity_projection_when_widths_match():
    cfg = _cfg(manifold=True)
    cfg["discern_v2"]["manifold"]["input_dim"] = FEAT_DIM
    man = build_v2_stack(cfg).branches["manifold"]
    assert isinstance(man.input_proj, torch.nn.Identity)
    print("  ok: identity projection when input_dim == feature_dim (pilot-equivalent)")


def test_gradients_reach_the_branch():
    stack = build_v2_stack(_cfg(manifold=True))
    base = torch.rand(BATCH, 2)
    out, _ = stack(base, visual_feature=torch.randn(BATCH, VIS_DIM))
    out.sum().backward()
    assert stack.gates["manifold"].grad is not None
    assert any(p.grad is not None for p in stack.branches["manifold"].parameters())
    print("  ok: gradients reach both the gate and the branch")


def test_process_without_images_fails_loudly():
    """The process residual is computed from pixels; a silent skip would be worse."""
    stack = DiscernV2Stack({"discern_v2": {
        "manifold": {"enabled": True, "feature_dim": FEAT_DIM, "input_dim": VIS_DIM,
                     "projector": "beta_tcvae", "export_rate_response": False}}})
    stack.branches["process"] = torch.nn.Identity()  # force the branch name in
    try:
        stack(torch.rand(BATCH, 2), visual_feature=torch.randn(BATCH, VIS_DIM), images=None)
    except (ValueError, KeyError) as exc:
        print(f"  ok: missing images raises ({type(exc).__name__}) rather than skipping")
        return
    raise AssertionError("should have raised")


def test_every_projector_integrates():
    for proj in ("beta_vae", "deterministic_ae", "beta_tcvae", "mr_vae"):
        stack = build_v2_stack(_cfg(manifold=True, projector=proj))
        out, diag = stack(torch.rand(BATCH, 2), visual_feature=torch.randn(BATCH, VIS_DIM))
        assert out.shape == (BATCH, 2), proj
        assert "manifold_evidence" in diag, proj
    print("  ok: all four projectors integrate — D1 projector swap stays a config change")


def test_tc_extra_loss_flows_through_projection():
    stack = build_v2_stack(_cfg(manifold=True, projector="beta_tcvae"))
    stack.train()
    losses = stack.extra_losses(torch.randn(BATCH, VIS_DIM), dataset_size=10000)
    assert "manifold_tc" in losses and torch.isfinite(losses["manifold_tc"])
    print(f"  ok: beta-TCVAE TC term reaches the loss ({losses['manifold_tc'].item():.3f})")


def test_no_extra_loss_for_non_tcvae():
    stack = build_v2_stack(_cfg(manifold=True, projector="mr_vae"))
    assert stack.extra_losses(torch.randn(BATCH, VIS_DIM), 10000) == {}
    print("  ok: projectors without an extra objective contribute no loss term")


if __name__ == "__main__":
    torch.manual_seed(0)
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"running {len(fns)} integration tests\n")
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

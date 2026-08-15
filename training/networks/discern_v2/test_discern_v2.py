#!/usr/bin/env python3
"""Contract and port-parity tests for the DISCERN v2 scaffolding.

These pin the properties that are easy to break silently during integration: the shared
Dirichlet definition, the projector interface, the parameter-count relationships that make
the P1 comparison fair, and the config couplings whose failure mode is a wrong experiment
rather than an exception.

    python training/networks/discern_v2/test_discern_v2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from networks.discern_v2 import branches as B          # noqa: E402
from networks.discern_v2 import dirichlet as D         # noqa: E402
from networks.discern_v2 import projectors as P        # noqa: E402

FEATURE_DIM, LATENT_DIM, BATCH = 64, 32, 8


# --------------------------------------------------------------------------- dirichlet


def test_dirichlet_definitions():
    e = torch.rand(BATCH, 2) * 5
    s = D.to_dirichlet(e)
    assert torch.allclose(s.alpha, e + 1)
    assert torch.allclose(s.strength, s.alpha.sum(1, keepdim=True))
    assert torch.allclose(s.p, s.alpha / s.strength)
    assert torch.allclose(s.p.sum(1), torch.ones(BATCH), atol=1e-6)
    assert torch.allclose(s.vacuity, (2.0 / s.strength).squeeze(1))
    print("  ok: alpha = e+1, p = alpha/S, u = K/S, p sums to 1")


def test_vacuity_bounds():
    """Zero evidence -> maximal vacuity 1.0; large evidence -> vacuity toward 0."""
    none = D.to_dirichlet(torch.zeros(4, 2))
    lots = D.to_dirichlet(torch.full((4, 2), 100.0))
    assert torch.allclose(none.vacuity, torch.ones(4)), none.vacuity
    assert (lots.vacuity < 0.02).all(), lots.vacuity
    print("  ok: vacuity is 1.0 with no evidence and ->0 with strong evidence")


def test_negative_evidence_clamped():
    """A negative entry would make alpha < 1 and vacuity > K -- not a valid Dirichlet."""
    s = D.to_dirichlet(torch.tensor([[-5.0, 2.0]]))
    assert (s.alpha >= 1).all()
    assert (s.vacuity <= 2.0).all()
    print("  ok: negative evidence is clamped rather than producing an invalid Dirichlet")


def test_evidence_activations_non_negative():
    logits = torch.randn(BATCH, 2) * 10
    for act in ("softplus", "relu", "exp"):
        e = D.logits_to_evidence(logits, act)
        assert (e >= 0).all() and torch.isfinite(e).all(), act
    print("  ok: every evidence activation is non-negative and finite")


# --------------------------------------------------------------------------- projectors


def test_all_projectors_share_the_contract():
    x = torch.randn(BATCH, FEATURE_DIM)
    for name in P.PROJECTORS:
        proj = P.build_projector(name, FEATURE_DIM, LATENT_DIM)
        proj.eval()
        z, mu, log_var, recon = proj(x)
        assert z.shape == (BATCH, LATENT_DIM), name
        assert mu.shape == (BATCH, LATENT_DIM), name
        assert log_var.shape == (BATCH, LATENT_DIM), name
        assert recon.shape == (BATCH, FEATURE_DIM), name
    print(f"  ok: {len(P.PROJECTORS)} projectors all return (z, mu, log_var, recon)")


def test_betatcvae_matches_betavae_param_count():
    """P1b changes only the objective, so the parameter count must be identical to P0-DS."""
    a = sum(p.numel() for p in P.build_projector("beta_vae", FEATURE_DIM, LATENT_DIM).parameters())
    b = sum(p.numel() for p in P.build_projector("beta_tcvae", FEATURE_DIM, LATENT_DIM).parameters())
    assert a == b, f"beta_vae {a} vs beta_tcvae {b}"
    print(f"  ok: beta_tcvae parameter count identical to beta_vae ({a})")


def test_deterministic_ae_drops_exactly_fc_log_var():
    """P1a removes fc_log_var: Linear(32,32) = 32*32 + 32 = 1056 parameters."""
    a = sum(p.numel() for p in P.build_projector("beta_vae", FEATURE_DIM, LATENT_DIM).parameters())
    d = sum(p.numel() for p in P.build_projector("deterministic_ae", FEATURE_DIM, LATENT_DIM).parameters())
    assert a - d == 1056, f"expected a 1056-param drop, got {a - d}"
    print(f"  ok: deterministic_ae is exactly 1056 params smaller ({a} -> {d})")


def test_deterministic_ae_is_deterministic():
    proj = P.build_projector("deterministic_ae", FEATURE_DIM, LATENT_DIM)
    proj.train()                       # even in train mode there is no sampling
    x = torch.randn(BATCH, FEATURE_DIM)
    z1, mu1, lv1, r1 = proj(x)
    z2, _, _, r2 = proj(x)
    assert torch.allclose(z1, z2) and torch.allclose(r1, r2)
    assert torch.allclose(z1, mu1), "z and mu must be the same tensor for a deterministic AE"
    assert torch.allclose(lv1, torch.zeros_like(lv1))
    print("  ok: deterministic_ae samples nothing in train mode, log_var is zeros")


def test_stochastic_projectors_sample_in_train_only():
    for name in ("beta_vae", "beta_tcvae"):
        proj = P.build_projector(name, FEATURE_DIM, LATENT_DIM)
        x = torch.randn(BATCH, FEATURE_DIM)
        proj.train()
        assert not torch.allclose(proj(x)[0], proj(x)[0]), f"{name} did not sample in train"
        proj.eval()
        assert torch.allclose(proj(x)[0], proj(x)[0]), f"{name} sampled in eval"
    print("  ok: beta_vae / beta_tcvae sample in train mode and are deterministic in eval")


def test_mr_vae_rate_response_shape():
    proj = P.build_projector("mr_vae", FEATURE_DIM, LATENT_DIM)
    r = proj.rate_distortion_response(torch.randn(BATCH, FEATURE_DIM))
    assert r.shape == (BATCH, P.RATE_RESPONSE_K), r.shape
    assert torch.isfinite(r).all()
    print(f"  ok: mr_vae rate response is (B, {P.RATE_RESPONSE_K})")


def test_mr_vae_film_is_identity_at_init():
    """FiLM starts at gamma = delta = 0, so an untrained MR-VAE ignores beta.

    The response must therefore be flat across the grid at init -- if it is not, FiLM was
    not zero-initialised and the model is not equivalent to an unconditioned VAE at step 0.
    """
    proj = P.build_projector("mr_vae", FEATURE_DIM, LATENT_DIM)
    r = proj.rate_distortion_response(torch.randn(BATCH, FEATURE_DIM))
    spread = (r.max(dim=1).values - r.min(dim=1).values).abs().max()
    assert spread < 1e-5, f"response varies by {spread} at init; FiLM is not identity"
    print("  ok: MR-VAE FiLM is identity at init (flat response across the beta grid)")


def test_only_mr_vae_advertises_a_rate_response():
    for name, cls in P.PROJECTORS.items():
        expected = (name == "mr_vae")
        assert cls.has_rate_response == expected, name
    print("  ok: has_rate_response is true for mr_vae only")


def test_wae_is_unported_with_an_informative_error():
    try:
        P.build_projector("wae", FEATURE_DIM, LATENT_DIM)
    except NotImplementedError as exc:
        assert "P1c" in str(exc)
        print("  ok: wae raises an informative NotImplementedError, not a bare KeyError")
        return
    raise AssertionError("wae should not build")


def test_deterministic_ae_requires_zero_kl():
    """The silent-failure guard: beta_kld != 0 turns the KL into L2 on the latent."""
    P.assert_projector_config("deterministic_ae", 0.0)          # fine
    P.assert_projector_config("beta_vae", 2.0)                  # unrelated
    try:
        P.assert_projector_config("deterministic_ae", 2.0)
    except ValueError as exc:
        assert "0.5 * sum(mu^2)" in str(exc)
        print("  ok: deterministic_ae with non-zero beta_kld is rejected")
        return
    raise AssertionError("should have rejected deterministic_ae with beta_kld = 2.0")


def test_tc_decomposition_is_finite_and_differentiable():
    proj = P.build_projector("beta_tcvae", FEATURE_DIM, LATENT_DIM)
    proj.train()
    x = torch.randn(BATCH, FEATURE_DIM)
    z, mu, log_var, _ = proj(x)
    loss = P.tc_decomposed_kl(z, mu, log_var, dataset_size=10000, beta_tc=2.0)
    assert torch.isfinite(loss), loss
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in proj.parameters())
    print(f"  ok: TC-decomposed KL is finite ({loss.item():.4f}) and backprops")


# --------------------------------------------------------------------------- branches


def test_branch_outputs_share_the_contract():
    vis = B.VisualEvidenceBranch(in_dim=FEATURE_DIM)
    man = B.ManifoldEvidenceBranch(feature_dim=FEATURE_DIM, projector="mr_vae")
    for name, out in (("visual", vis(torch.randn(BATCH, FEATURE_DIM))),
                      ("manifold", man(torch.randn(BATCH, FEATURE_DIM)))):
        assert out.evidence.shape == (BATCH, 2), name
        assert (out.evidence >= 0).all(), name
        s = out.state
        assert s.p.shape == (BATCH, 2) and s.vacuity.shape == (BATCH,), name
        assert torch.allclose(s.alpha, out.evidence + 1), name
    print("  ok: branches return (B,2) evidence and a centrally-derived Dirichlet state")


def test_manifold_branch_residual_and_diagnostics():
    man = B.ManifoldEvidenceBranch(feature_dim=FEATURE_DIM, projector="mr_vae")
    man.eval()
    x = torch.randn(BATCH, FEATURE_DIM)
    out = man(x)
    assert out.features.shape == (BATCH, FEATURE_DIM), "residual must match feature width"
    recon = out.diagnostics["reconstruction"]
    assert torch.allclose(out.features, x - recon, atol=1e-6), "features must be f_s - f_c"
    assert out.diagnostics["rate_response"].shape == (BATCH, P.RATE_RESPONSE_K)
    print("  ok: manifold branch exposes f_r = f_s - f_c plus the rate response")


def test_rate_response_absent_for_non_mrvae():
    man = B.ManifoldEvidenceBranch(feature_dim=FEATURE_DIM, projector="beta_tcvae")
    out = man(torch.randn(BATCH, FEATURE_DIM))
    assert "rate_response" not in out.diagnostics
    print("  ok: a non-MR-VAE projector exports no rate response, and nothing breaks")


def test_projector_swap_is_config_only():
    """Every projector must drop into the manifold branch with no other change."""
    for name in P.PROJECTORS:
        man = B.ManifoldEvidenceBranch(feature_dim=FEATURE_DIM, projector=name)
        out = man(torch.randn(BATCH, FEATURE_DIM))
        assert out.evidence.shape == (BATCH, 2), name
    print(f"  ok: all {len(P.PROJECTORS)} projectors drop into the branch unchanged")


def test_branch_toggling_is_config_driven():
    """The D1 sub-ablation must be a config switch, not a code change."""
    cfg = {"discern_v2": {
        "visual": {"enabled": True, "feature_dim": FEATURE_DIM},
        "manifold": {"enabled": True, "feature_dim": FEATURE_DIM, "projector": "mr_vae"},
        "process": {"enabled": False}}}
    assert set(B.build_branches(cfg)) == {"visual", "manifold"}          # D1-VM

    cfg["discern_v2"]["manifold"]["enabled"] = False
    assert set(B.build_branches(cfg)) == {"visual"}                      # D1-V

    cfg["discern_v2"]["manifold"]["enabled"] = True
    cfg["discern_v2"]["visual"]["enabled"] = False
    assert set(B.build_branches(cfg)) == {"manifold"}                    # D1-M
    print("  ok: D1-V / D1-M / D1-VM are reachable by config alone")


def test_all_branches_off_is_refused():
    cfg = {"discern_v2": {"visual": {"enabled": False}, "manifold": {"enabled": False},
                          "process": {"enabled": False}}}
    try:
        B.build_branches(cfg)
    except ValueError as exc:
        assert "no DISCERN v2 branch is enabled" in str(exc)
        print("  ok: an all-off config is refused rather than silently training a stub")
        return
    raise AssertionError("all-off config should raise")


def test_conflict_is_symmetric_and_bounded():
    a = D.to_dirichlet(torch.rand(BATCH, 2) * 5)
    b = D.to_dirichlet(torch.rand(BATCH, 2) * 5)
    c1, c2 = D.conflict(a, b), D.conflict(b, a)
    assert torch.allclose(c1, c2)
    assert ((c1 >= 0) & (c1 <= 1)).all()
    print("  ok: cross-branch conflict is symmetric and in [0, 1]")


def test_cache_hazard_is_enforced():
    from networks.discern_v2.process_residual import assert_cache_safe
    assert_cache_safe(process_input_is_deterministic=True, visual_input_is_augmented=True)
    try:
        assert_cache_safe(process_input_is_deterministic=False,
                          visual_input_is_augmented=True)
    except ValueError as exc:
        assert "unsafe" in str(exc)
        print("  ok: cached clean residual + augmented visual input is refused")
        return
    raise AssertionError("the caching hazard should have been refused")


if __name__ == "__main__":
    torch.manual_seed(0)
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"running {len(fns)} DISCERN v2 tests\n")
    failed = 0
    for fn in fns:
        print(f"{fn.__name__}:")
        try:
            fn()
        except Exception as exc:                      # noqa: BLE001
            failed += 1
            print(f"  FAIL: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)

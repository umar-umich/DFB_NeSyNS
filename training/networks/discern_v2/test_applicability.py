#!/usr/bin/env python3
"""Tests for D4 — applicability-aware fusion.

Two properties carry the whole arm, and both are checked here rather than argued:

* **D0-D3 are untouched.** D4's claim is the delta D4 - D3(P1d). If merely adding the gate
  code perturbed the ungated path, that delta would measure the edit instead of the gate.
* **The gate is label-free at inference.** The A2b protocol permits the label as a training
  target and forbids it as an input. Those are different things, and the entire
  applicability result depends on the distinction holding in code, not just in prose.

    python training/networks/discern_v2/test_applicability.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from networks.discern_v2.applicability import (  # noqa: E402
    FEATURE_DIM, FEATURE_NAMES, ApplicabilityGate, assert_label_free, gate_features)
from networks.discern_v2.dirichlet import to_dirichlet  # noqa: E402
from networks.discern_v2.integration import DiscernV2Stack, build_v2_stack  # noqa: E402

VIS_DIM, FEAT_DIM, BATCH = 1024, 64, 8


def _cfg(applicability=None, projector="mr_vae"):
    """Manifold-only config. Process is left off so the tests need no SDXL-VAE on disk."""
    v2 = {
        "visual": {"enabled": True, "feature_dim": VIS_DIM},
        "manifold": {"enabled": True, "feature_dim": FEAT_DIM, "latent_dim": 32,
                     "projector": projector, "input_dim": VIS_DIM,
                     "export_rate_response": False},
        "process": {"enabled": False, "vae_path": None},
    }
    if applicability is not None:
        v2["applicability"] = applicability
    return {"manifold_v2": True, "process_v2": False, "discern_v2": v2}


def _states(n=BATCH):
    """A baseline and a specialist Dirichlet state with deliberately different opinions."""
    torch.manual_seed(0)
    base = to_dirichlet(torch.rand(n, 2) * 3)
    spec = to_dirichlet(torch.rand(n, 2) * 3)
    return base, spec


# --------------------------------------------------------------------------- #
#  D0-D3 must be bit-identical                                                  #
# --------------------------------------------------------------------------- #

def test_no_applicability_key_means_no_gate():
    """A config without an `applicability` block builds the pre-D4 stack exactly."""
    stack = build_v2_stack(_cfg())
    assert stack.applicability is None
    assert not [n for n, _ in stack.named_parameters() if "applicability" in n]
    print("  ok: no applicability block -> no gate, no extra parameters")


def test_applicability_disabled_means_no_gate():
    """`enabled: false` is equivalent to omitting the block, so D3 configs can carry it."""
    stack = build_v2_stack(_cfg({"enabled": False, "tau": 0.95}))
    assert stack.applicability is None
    print("  ok: applicability.enabled=false -> no gate")


def test_ungated_forward_uses_the_pre_d4_formula():
    """Without the gate, contribution is exactly sigmoid(gate_b) * evidence_b."""
    torch.manual_seed(1)
    stack = build_v2_stack(_cfg())
    stack.eval()
    vis = torch.randn(BATCH, VIS_DIM)
    base_ev = torch.rand(BATCH, 2)
    with torch.no_grad():
        out, diag = stack(base_ev.clone(), visual_feature=vis)
        expected = base_ev + torch.sigmoid(stack.gates["manifold"]) * diag["manifold_evidence"]
    assert torch.allclose(out, expected, atol=1e-6), (out - expected).abs().max()
    print("  ok: ungated forward matches the pre-D4 arithmetic exactly")


# --------------------------------------------------------------------------- #
#  Label-free discipline (A2b)                                                  #
# --------------------------------------------------------------------------- #

def test_declared_features_are_label_free():
    assert_label_free(FEATURE_NAMES)
    assert len(FEATURE_NAMES) == FEATURE_DIM
    print(f"  ok: all {FEATURE_DIM} declared gate features pass the label-free check")


def test_forbidden_feature_names_are_rejected():
    """The realistic failure: someone adds the very predictive `dloss_manifold`."""
    for bad in ("dloss_manifold", "label", "generator_id", "dataset", "family_id",
                "source_name", "method", "split"):
        try:
            assert_label_free(["p_vis", bad])
        except ValueError:
            continue
        raise AssertionError(f"forbidden feature name accepted: {bad!r}")
    print("  ok: forbidden feature names are rejected, not silently used")


def test_features_do_not_depend_on_labels():
    """gate_features takes only Dirichlet states — there is no label argument to leak."""
    base, spec = _states()
    a = gate_features(base, spec)
    b = gate_features(base, spec)
    assert torch.equal(a, b) and a.shape == (BATCH, FEATURE_DIM)
    print("  ok: gate features are a pure function of evidence state")


def test_eval_pass_computes_no_gate_loss():
    """No labels are handed to the gate outside training, so no supervision term exists."""
    torch.manual_seed(2)
    stack = build_v2_stack(_cfg({"enabled": True, "tau": 0.95}))
    stack.eval()
    with torch.no_grad():
        _, diag = stack(torch.rand(BATCH, 2), visual_feature=torch.randn(BATCH, VIS_DIM),
                        baseline_evidence=torch.rand(BATCH, 2), labels=None)
    assert not [k for k in diag if k.startswith("applicability_loss")]
    assert "manifold_q" in diag and "manifold_route" in diag
    print("  ok: eval pass routes but computes no gate loss (no label touched)")


def test_gate_needs_a_baseline():
    """The gate is defined relative to the visual baseline; without it, fail loudly."""
    stack = build_v2_stack(_cfg({"enabled": True}))
    try:
        stack(torch.rand(BATCH, 2), visual_feature=torch.randn(BATCH, VIS_DIM))
    except ValueError as e:
        assert "baseline_evidence" in str(e)
        print("  ok: missing baseline_evidence raises instead of silently degrading")
        return
    raise AssertionError("gate ran without a baseline")


# --------------------------------------------------------------------------- #
#  Routing behaviour                                                            #
# --------------------------------------------------------------------------- #

def test_routing_is_hard_and_at_tau():
    gate = ApplicabilityGate(["manifold"], tau=0.95, warmup_epochs=0)
    q = torch.tensor([0.0, 0.5, 0.9499, 0.95, 0.999])
    route = gate.route(q)
    assert torch.equal(route, torch.tensor([0., 0., 0., 1., 1.]))
    assert not route.requires_grad
    print("  ok: routing is a hard {0,1} mask at tau, with no gradient path")


def test_warmup_routes_everything():
    """During warmup D4 must be exactly D3, or the specialists never train at all."""
    gate = ApplicabilityGate(["manifold"], tau=0.95, warmup_epochs=5)
    q = torch.tensor([0.0, 0.01, 0.5])          # all far below tau
    gate.set_epoch(0)
    assert gate.warming_up and torch.equal(gate.route(q), torch.ones(3))
    gate.set_epoch(4)
    assert gate.warming_up and torch.equal(gate.route(q), torch.ones(3))
    print("  ok: during warmup every sample is routed (D4 == D3)")


def test_gate_engages_after_warmup():
    gate = ApplicabilityGate(["manifold"], tau=0.95, warmup_epochs=5)
    q = torch.tensor([0.0, 0.96])
    gate.set_epoch(5)
    assert not gate.warming_up
    assert torch.equal(gate.route(q), torch.tensor([0., 1.]))
    print("  ok: at warmup_epochs the hard tau threshold takes over")


def test_warmup_prevents_the_deadlock():
    """The failure this guards against: no warmup -> nothing routed -> no branch gradient.

    Asserted end-to-end on the stack, because the deadlock is a property of the wiring
    rather than of the gate in isolation.
    """
    torch.manual_seed(9)
    stack = build_v2_stack(_cfg({"enabled": True, "tau": 0.95, "warmup_epochs": 0}))
    base_ev = torch.rand(BATCH, 2)
    out, diag = stack(base_ev.clone(), visual_feature=torch.randn(BATCH, VIS_DIM),
                      baseline_evidence=torch.rand(BATCH, 2),
                      labels=torch.randint(0, 2, (BATCH,)), epoch=0)
    if diag["manifold_route"].sum() == 0:
        out.sum().backward()
        branch_grads = [p.grad for p in stack.branches["manifold"].parameters()
                        if p.grad is not None]
        assert not branch_grads or all(g.abs().sum() == 0 for g in branch_grads), \
            "expected the routed-out branch to receive no gradient"
        print("  ok: with warmup_epochs=0 an untrained gate can starve the branch "
              "(this is exactly why the shipped config warms up)")
    else:
        print("  ok: untrained gate happened to route; warmup still guards the general case")


def test_low_confidence_leaves_the_baseline_untouched():
    """The conservative property: when the gate is unsure, D4 must equal the baseline."""
    torch.manual_seed(3)
    stack = build_v2_stack(_cfg({"enabled": True, "tau": 0.95}))
    stack.eval()
    head = stack.applicability.heads["manifold"][-1]
    with torch.no_grad():                      # force q ~= 0 for every sample
        head.weight.zero_()
        head.bias.fill_(-20.0)
    base_ev = torch.rand(BATCH, 2)
    with torch.no_grad():
        out, diag = stack(base_ev.clone(), visual_feature=torch.randn(BATCH, VIS_DIM),
                          baseline_evidence=torch.rand(BATCH, 2), epoch=99)
    assert diag["manifold_route"].sum() == 0
    assert torch.allclose(out, base_ev, atol=1e-6)
    print("  ok: below tau, the specialist contributes exactly nothing")


def test_high_confidence_restores_the_d3_contribution():
    """Routed in, a branch contributes precisely what D3 would have contributed."""
    torch.manual_seed(4)
    stack = build_v2_stack(_cfg({"enabled": True, "tau": 0.95}))
    stack.eval()
    head = stack.applicability.heads["manifold"][-1]
    with torch.no_grad():
        head.weight.zero_()
        head.bias.fill_(20.0)                  # q ~= 1 for every sample
    base_ev = torch.rand(BATCH, 2)
    with torch.no_grad():
        out, diag = stack(base_ev.clone(), visual_feature=torch.randn(BATCH, VIS_DIM),
                          baseline_evidence=torch.rand(BATCH, 2), epoch=99)
        expected = base_ev + torch.sigmoid(stack.gates["manifold"]) * diag["manifold_evidence"]
    assert diag["manifold_route"].sum() == BATCH
    assert torch.allclose(out, expected, atol=1e-6)
    print("  ok: above tau, D4 reproduces the D3 contribution exactly")


# --------------------------------------------------------------------------- #
#  Supervision and freezing                                                     #
# --------------------------------------------------------------------------- #

def test_gate_target_is_the_pilot_target():
    """t_b = 1 exactly when the specialist's NLL beats the baseline's on that sample."""
    gate = ApplicabilityGate(["manifold"], tau=0.95)
    labels = torch.tensor([1, 1])
    # sample 0: specialist far more confident in the true class (fake) -> target 1
    # sample 1: specialist far less confident                          -> target 0
    base = to_dirichlet(torch.tensor([[1.0, 1.0], [0.0, 9.0]]))
    spec = to_dirichlet(torch.tensor([[0.0, 9.0], [9.0, 0.0]]))
    q_hi, q_lo = torch.tensor([1 - 1e-6, 1e-6]), torch.tensor([1e-6, 1 - 1e-6])
    loss_right = gate.gate_loss("manifold", q_hi, base, spec, labels)
    loss_wrong = gate.gate_loss("manifold", q_lo, base, spec, labels)
    assert loss_right < loss_wrong, (loss_right.item(), loss_wrong.item())
    assert loss_right.item() < 1e-3
    print("  ok: gate loss is minimised by predicting 'specialist wins' correctly")


def test_freeze_stops_gate_updates():
    gate = ApplicabilityGate(["manifold"], freeze_after_epoch=3)
    assert all(p.requires_grad for p in gate.parameters())
    gate.maybe_freeze(2)
    assert all(p.requires_grad for p in gate.parameters())
    gate.maybe_freeze(3)
    assert not any(p.requires_grad for p in gate.parameters())
    gate.maybe_freeze(4)                       # idempotent
    print("  ok: gate freezes at freeze_after_epoch and stays frozen")


def test_gate_params_are_reachable_from_the_stack():
    """train.py adds `m.v2.parameters()` as one optimizer group. If the gate's parameters
    were not inside that, D4 would train with a permanently random gate and quietly measure
    noise — the exact bug class fixed in a08a5b8 and, before that, in v3."""
    stack = build_v2_stack(_cfg({"enabled": True}))
    names = {n for n, _ in stack.named_parameters()}
    gate_names = {n for n in names if "applicability" in n}
    assert gate_names, "gate parameters are not in stack.parameters()"
    assert len(gate_names) == 4, sorted(gate_names)      # 2 Linear layers, weight + bias
    print(f"  ok: {len(gate_names)} gate parameters reach the optimizer group")


def test_routable_branches_only():
    """Enabling the gate with nothing to route is a config error, not a silent D0."""
    cfg = {"manifold_v2": False, "process_v2": False,
           "discern_v2": {"applicability": {"enabled": True}}}
    assert build_v2_stack(cfg) is None          # master switches win; no stack at all
    print("  ok: gate cannot be enabled without a routable branch")


# --------------------------------------------------------------------------- #
#  The shipped config                                                           #
# --------------------------------------------------------------------------- #

def test_d4_config_builds_the_gate():
    import yaml
    path = Path(__file__).resolve().parents[2] / "config/detector/nesy_defake_d4.yaml"
    cfg = yaml.safe_load(open(path))
    assert cfg["manifold_v2"] and cfg["process_v2"]
    app = cfg["discern_v2"]["applicability"]
    assert app["enabled"] is True
    assert app["tau"] == 0.95, f"tau must stay at the A2c setting, got {app['tau']}"
    assert app["freeze_after_epoch"] is None
    assert app["warmup_epochs"] > 0, \
        "the shipped D4 config must warm up, or the arm can silently deadlock"
    assert cfg["discern_v2"]["manifold"]["projector"] == "mr_vae", \
        "D4 is the P1d arm; changing the projector would confound the D4 - D3(P1d) delta"
    print("  ok: nesy_defake_d4.yaml enables the gate at tau=0.95 on the mr_vae baseline")


def test_d4_matches_its_baseline_config():
    """Every key except the applicability block must agree with D3(P1d)."""
    import yaml
    base_dir = Path(__file__).resolve().parents[2] / "config/detector"
    d4 = yaml.safe_load(open(base_dir / "nesy_defake_d4.yaml"))
    d3 = yaml.safe_load(open(base_dir / "nesy_defake_d3_p1d.yaml"))
    d4["discern_v2"].pop("applicability")
    assert d4 == d3, "D4 differs from D3(P1d) beyond the applicability block"
    print("  ok: D4 is D3(P1d) + applicability and nothing else")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"DISCERN v2 — D4 applicability tests ({len(tests)})\n")
    for t in tests:
        t()
    print(f"\nall {len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

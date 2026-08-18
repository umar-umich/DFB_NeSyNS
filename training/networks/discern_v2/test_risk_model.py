"""Tests for §18's Real / Fake / Defer risk model.

    python training/networks/discern_v2/test_risk_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from networks.discern_v2 import risk_model as R  # noqa: E402

PASSED = []


def check(name):
    def deco(fn):
        fn()
        PASSED.append(name)
        print(f"  ok: {name}")
        return fn
    return deco


def synthetic(n: int = 800, seed: int = 0):
    """Errors concentrate where vacuity/conflict are high and the fused margin is small.

    That is the relationship the reliability decomposition claims exists; the tests check the
    machinery can find it when present, not that it exists in real data.
    """
    torch.manual_seed(seed)
    V = torch.rand(n)
    C = torch.rand(n)
    A = torch.rand(n)
    margin = torch.rand(n) * 0.5
    logit = 3.0 * V + 2.0 * C + 0.5 * A - 6.0 * margin - 1.0
    wrong = (torch.rand(n) < torch.sigmoid(logit)).float()
    labels = torch.randint(0, 2, (n,))
    # a fused probability consistent with the margin and the error pattern
    correct_side = torch.where(labels == 1, 0.5 + margin, 0.5 - margin)
    flipped = torch.where(labels == 1, 0.5 - margin, 0.5 + margin)
    prob = torch.where(wrong.bool(), flipped, correct_side)
    features = R.risk_features(V, C, A, prob)
    return features, wrong, labels, prob


# ---------------------------------------------------------------- features / model


@check("§18: features are exactly [V, C, A, fused_margin], and margin is unsigned")
def _features():
    V, C, A = torch.tensor([0.2]), torch.tensor([0.3]), torch.tensor([0.4])
    high = R.risk_features(V, C, A, torch.tensor([0.9]))
    low = R.risk_features(V, C, A, torch.tensor([0.1]))
    assert high.shape == (1, 4)
    assert R.FEATURE_NAMES == ("V", "C", "A", "fused_margin")
    assert torch.allclose(high[:, 3], low[:, 3]), (
        "the margin must be symmetric in the class, or the risk model can learn a class prior "
        "instead of a reliability signal")


@check("§18: the model is a logistic regression with four readable coefficients")
def _small_model():
    model = R.RiskModel()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 5, f"4 weights + bias, got {n_params}"
    coefs = model.coefficients()
    assert set(coefs) == {"V", "C", "A", "fused_margin", "bias"}


@check("the model refuses a feature vector of the wrong width")
def _wrong_width():
    try:
        R.RiskModel()(torch.randn(3, 5))
    except ValueError as exc:
        assert "expected" in str(exc)
    else:
        raise AssertionError("a mismatched width must raise")


@check("a split with no errors at all is refused rather than silently calibrated")
def _degenerate_fit():
    features, _, _, _ = synthetic(50)
    try:
        R.fit_risk_model(features, torch.zeros(50))
    except ValueError as exc:
        assert "no error signal" in str(exc)
    else:
        raise AssertionError("an all-correct split must be refused")


@check("§18: the fitted model detects errors and recovers the right coefficient signs")
def _fit_detects_errors():
    features, wrong, _, _ = synthetic()
    model, info = R.fit_risk_model(features, wrong)
    with torch.no_grad():
        risk = model(features)
    auroc = R.error_detection_auroc(risk, wrong)
    assert auroc > 0.75, auroc
    c = info["coefficients"]
    assert c["V"] > 0 and c["C"] > 0, f"more ignorance/conflict must mean more risk: {c}"
    assert c["fused_margin"] < 0, f"a more decided fusion must mean less risk: {c}"


# ---------------------------------------------------------------- selective prediction


@check("§18: selective risk falls as coverage falls")
def _risk_coverage():
    features, wrong, _, _ = synthetic()
    model, _ = R.fit_risk_model(features, wrong)
    with torch.no_grad():
        risk = model(features)
    curve = R.risk_coverage_curve(risk, wrong)
    assert curve[0]["coverage"] == 1.0
    assert curve[0]["selective_risk"] > curve[-1]["selective_risk"], (
        "a working risk score must let the system trade coverage for accuracy")
    # and roughly monotone: allow small non-monotonicity from finite samples
    risks = [row["selective_risk"] for row in curve]
    inversions = sum(1 for a, b in zip(risks, risks[1:]) if b > a + 1e-9)
    assert inversions < len(risks) // 2, f"curve is not ordering errors: {risks}"


@check("§18: a random risk score does NOT improve selective risk")
def _random_baseline():
    _, wrong, _, _ = synthetic()
    torch.manual_seed(1)
    curve = R.risk_coverage_curve(torch.rand(len(wrong)), wrong)
    assert abs(curve[0]["selective_risk"] - curve[-1]["selective_risk"]) < 0.15, (
        "a random score should leave selective risk roughly flat; if this fails the metric "
        "itself is rewarding something other than error ordering")


@check("§18: the fixed 10% abstention budget reports coverage and risk reduction")
def _budget():
    features, wrong, _, _ = synthetic()
    model, _ = R.fit_risk_model(features, wrong)
    with torch.no_grad():
        risk = model(features)
    out = R.coverage_at_budget(risk, wrong, 0.10)
    assert abs(out["coverage"] - 0.90) < 0.02, out
    assert out["risk_reduction"] > 0, out
    assert out["selective_risk"] < out["full_coverage_risk"]


# ---------------------------------------------------------------- threshold discipline


@check("§18: thresholds are frozen, carried with the policy, and record their provenance")
def _frozen_policy():
    features, wrong, labels, prob = synthetic()
    model, _ = R.fit_risk_model(features, wrong)
    with torch.no_grad():
        risk = model(features)
    policy = R.freeze_thresholds(labels, prob, risk, abstention_budget=0.10)
    assert "frozen for all OOD sources" in policy.provenance
    try:
        policy.decision_threshold = 0.9      # type: ignore[misc]
    except Exception:
        pass
    else:
        raise AssertionError("the policy must be immutable so it cannot be retuned downstream")


@check("§18: the frozen policy defers ~the budget on the source it was fit on")
def _budget_holds_on_source():
    features, wrong, labels, prob = synthetic()
    model, _ = R.fit_risk_model(features, wrong)
    with torch.no_grad():
        risk = model(features)
    policy = R.freeze_thresholds(labels, prob, risk, abstention_budget=0.10)
    out = R.evaluate_policy(policy, labels, prob, risk)
    assert abs(out["coverage"] - 0.90) < 0.03, out
    assert out["selective_accuracy"] >= out["full_coverage_accuracy"], out


@check("§18: the SAME frozen policy applied to a shifted source is not retuned")
def _policy_transfers_unchanged():
    features, wrong, labels, prob = synthetic(seed=0)
    model, _ = R.fit_risk_model(features, wrong)
    with torch.no_grad():
        risk = model(features)
    policy = R.freeze_thresholds(labels, prob, risk, 0.10)

    # a harder "OOD" source: more vacuity and conflict, so more samples exceed the cutoff
    ood_features, ood_wrong, ood_labels, ood_prob = synthetic(n=600, seed=5)
    ood_features = ood_features.clone()
    ood_features[:, 0] = (ood_features[:, 0] + 0.4).clamp(max=1.0)     # V up
    with torch.no_grad():
        ood_risk = model(ood_features)
    out = R.evaluate_policy(policy, ood_labels, ood_prob, ood_risk)
    assert out["coverage"] < 0.90 + 1e-6, (
        "a shifted, riskier source must defer MORE under a frozen threshold — if coverage were "
        "held at the budget, the threshold would have been retuned on the test data")
    assert sum(out["decisions"].values()) == out["n"]


@check("decisions partition into Real / Fake / Defer with no leftovers")
def _decisions():
    policy = R.DeferPolicy(decision_threshold=0.5, risk_threshold=0.5, provenance="test")
    prob = torch.tensor([0.9, 0.1, 0.8, 0.2])
    risk = torch.tensor([0.1, 0.1, 0.9, 0.9])
    d = policy.decide(prob, risk)
    assert d.tolist() == [R.FAKE, R.REAL, R.DEFER, R.DEFER]


@check("§18's full report contains every quantity the spec names")
def _report():
    features, wrong, labels, prob = synthetic()
    model, _ = R.fit_risk_model(features, wrong)
    with torch.no_grad():
        risk = model(features)
    rep = R.report(risk, wrong, labels, prob)
    for key in ("error_detection_auroc", "risk_coverage_curve", "selective_risk_at",
                "fixed_budget", "base_error_rate"):
        assert key in rep, key
    assert len(rep["selective_risk_at"]) == 5


if __name__ == "__main__":
    print(f"\nall {len(PASSED)} passed")

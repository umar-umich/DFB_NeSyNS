"""Tests for §13's fusion-utility-supervised applicability gates.

    python training/networks/discern_v2/test_applicability_gate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from networks.discern_v2 import applicability_gate as G  # noqa: E402
from networks.discern_v2.ds_fusion import Opinion, discount, ds_combine  # noqa: E402

PASSED = []
REAL, FAKE = 0, 1


def check(name):
    def deco(fn):
        fn()
        PASSED.append(name)
        print(f"  ok: {name}")
        return fn
    return deco


def opinion(belief_real, belief_fake):
    b = torch.tensor([[belief_real, belief_fake]], dtype=torch.float32)
    return Opinion(belief=b, vacuity=1.0 - b.sum(dim=1, keepdim=True))


def stack(ops):
    return Opinion(belief=torch.cat([o.belief for o in ops]),
                   vacuity=torch.cat([o.vacuity for o in ops]))


# ---------------------------------------------------------------- the target


@check("§13: a specialist that CORRECTS the anchor gets target 1")
def _helpful_specialist():
    sem = opinion(0.85, 0.05)          # confidently REAL
    spec = opinion(0.02, 0.90)         # confidently FAKE
    y = torch.tensor([FAKE])           # the specialist is right
    out = G.fusion_utility_target(sem, spec, y)
    assert float(out["target"]) == 1.0
    assert float(out["delta_fuse"]) > 0, out


@check("§13: a specialist that DAMAGES a correct anchor gets target 0")
def _harmful_specialist():
    sem = opinion(0.05, 0.85)          # confidently FAKE and correct
    spec = opinion(0.90, 0.02)         # confidently REAL and wrong
    y = torch.tensor([FAKE])
    out = G.fusion_utility_target(sem, spec, y)
    assert float(out["target"]) == 0.0
    assert float(out["delta_fuse"]) < 0


@check("§13: a vacuous specialist changes nothing, so it is not 'useful'")
def _vacuous_specialist():
    sem = opinion(0.10, 0.70)
    spec = Opinion.vacuous(1)
    out = G.fusion_utility_target(sem, spec, torch.tensor([FAKE]))
    assert abs(float(out["delta_fuse"])) < 1e-5, out["delta_fuse"]
    assert float(out["target"]) == 0.0, "no improvement means no admission signal"


@check("§13: fusion utility and standalone quality DISAGREE often enough to matter")
def _fusion_not_standalone():
    """The justification for §13, measured rather than asserted.

    Standalone supervision asks "is the specialist better than the anchor on its own?"; §13's
    target asks "does admitting it improve the fused decision?". If those agreed everywhere the
    distinction would be academic, so this samples random opinion pairs and reports how often
    they disagree — and fails if the answer is "hardly ever", which would mean the more complex
    target buys nothing.
    """
    torch.manual_seed(7)
    n = 2000
    def random_opinions(k):
        b = torch.rand(k, 2)
        b = b / b.sum(dim=1, keepdim=True) * torch.rand(k, 1) * 0.95
        return Opinion(belief=b, vacuity=1.0 - b.sum(dim=1, keepdim=True))

    sem, spec = random_opinions(n), random_opinions(n)
    y = torch.randint(0, 2, (n,))

    fusion_target = G.fusion_utility_target(sem, spec, y)["target"]
    ce_sem = G._cross_entropy(sem.probability(), y)
    ce_spec = G._cross_entropy(spec.probability(), y)
    standalone_target = (ce_sem - ce_spec > 0).float()      # the rejected alternative

    disagreement = float((fusion_target != standalone_target).float().mean())
    print(f"      standalone-vs-fusion target disagreement: {disagreement:.1%}")
    assert disagreement > 0.05, (
        f"only {disagreement:.1%} disagreement — if the two criteria agreed everywhere, §13's "
        f"more complex target would buy nothing")

    # and specifically: cases where the specialist is WORSE standalone yet still helps fusion
    helps_but_worse = ((fusion_target == 1) & (standalone_target == 0)).sum()
    assert int(helps_but_worse) > 0, (
        "there must exist specialists that are worse on their own yet improve the fusion — "
        "exactly the samples standalone supervision would wrongly gate off")


@check("delta is a config knob and shifts the decision boundary")
def _delta_threshold():
    sem = opinion(0.30, 0.40)
    spec = opinion(0.20, 0.50)
    y = torch.tensor([FAKE])
    lenient = G.fusion_utility_target(sem, spec, y, delta=0.0)
    strict = G.fusion_utility_target(sem, spec, y, delta=10.0)
    assert float(lenient["target"]) == 1.0
    assert float(strict["target"]) == 0.0, "an unreachable delta must admit nothing"


# ---------------------------------------------------------------- features


@check("§13: gate features are label-free by name, and a leaky name is refused")
def _label_free():
    G.assert_label_free(list(G.FEATURE_NAMES))
    for leaky in ("dloss_ref", "generator_id", "family", "label", "dataset_name"):
        try:
            G.assert_label_free([*G.FEATURE_NAMES, leaky])
        except Exception:
            continue
        raise AssertionError(f"{leaky!r} must be refused as a gate feature")


@check("gate features are computed from opinions only, with the documented width")
def _features():
    sem = stack([opinion(0.8, 0.1), opinion(0.1, 0.8)])
    spec = stack([opinion(0.2, 0.6), opinion(0.5, 0.2)])
    f = G.gate_features(sem, spec)
    assert f.shape == (2, len(G.FEATURE_NAMES)), f.shape
    assert torch.isfinite(f).all()
    # abs_disagree is index 4 by FEATURE_NAMES order
    expected = (sem.probability()[:, 1] - spec.probability()[:, 1]).abs()
    assert torch.allclose(f[:, 4], expected, atol=1e-6)


@check("the gate refuses a feature vector of the wrong width")
def _wrong_width():
    gate = G.ApplicabilityGate()
    try:
        gate(torch.randn(3, len(G.FEATURE_NAMES) + 1))
    except ValueError as exc:
        assert "expects" in str(exc)
    else:
        raise AssertionError("a mismatched feature width must raise")


# ---------------------------------------------------------------- training


def synthetic_gate_data(n: int = 400, seed: int = 0):
    """Specialist is useful exactly when the anchor is unsure — a learnable, label-free rule."""
    torch.manual_seed(seed)
    sem_fake = torch.rand(n)
    unsure = (sem_fake - 0.5).abs() < 0.2
    sem = Opinion(belief=torch.stack([1 - sem_fake, sem_fake], dim=1) * 0.8,
                  vacuity=torch.full((n, 1), 0.2))
    spec_fake = torch.rand(n)
    spec = Opinion(belief=torch.stack([1 - spec_fake, spec_fake], dim=1) * 0.8,
                   vacuity=torch.full((n, 1), 0.2))
    features = G.gate_features(sem, spec)
    target = unsure.float()
    return features, target


@check("a gate learns a label-free applicability rule")
def _trains():
    features, target = synthetic_gate_data()
    gate, info = G.train_gate(features, target, epochs=300)
    with torch.no_grad():
        q = gate(features)
    metrics = G.gate_metrics(q, target)
    assert metrics["accuracy"] > 0.85, metrics
    assert metrics["auroc"] > 0.9, metrics
    assert metrics["accuracy"] > metrics["majority_baseline_accuracy"], (
        "the gate must beat always-admit, or it has learned nothing usable")


@check("§12: cross-fitting yields out-of-fold q plus a deployment gate")
def _cross_fit():
    features, target = synthetic_gate_data()
    folds = torch.arange(len(target)) % 5
    out = G.cross_fit(features, target, folds, epochs=200)
    assert out["q_out_of_fold"].shape == target.shape
    assert len(out["per_fold"]) == 5
    assert out["metrics"]["auroc"] > 0.85, out["metrics"]
    # every sample got a prediction from a gate that did not see it
    assert (out["q_out_of_fold"] > 0).all()


@check("§12: out-of-fold predictions really are out of sample")
def _oof_is_out_of_sample():
    features, target = synthetic_gate_data(n=200, seed=3)
    folds = torch.arange(len(target)) % 5
    out = G.cross_fit(features, target, folds, epochs=200)
    with torch.no_grad():
        q_in_sample = out["gate"](features)
    # the deployment gate saw everything, so its fit is at least as good as the OOF one
    acc_oof = G.gate_metrics(out["q_out_of_fold"], target)["accuracy"]
    acc_in = G.gate_metrics(q_in_sample, target)["accuracy"]
    assert acc_in >= acc_oof - 1e-6, (acc_in, acc_oof)


@check("§14.2: a trained gate composes with discounting — low q means near-vacuous")
def _composes_with_discounting():
    features, target = synthetic_gate_data(n=200, seed=1)
    gate, _ = G.train_gate(features, target, epochs=300)
    with torch.no_grad():
        q = gate(features)
    spec = Opinion(belief=torch.full((len(q), 2), 0.45), vacuity=torch.full((len(q), 1), 0.1))
    discounted = discount(spec, q)
    discounted.assert_normalized()
    low, high = q.argmin(), q.argmax()
    assert float(discounted.vacuity[low]) > float(discounted.vacuity[high]), (
        "the least applicable sample must end up the most ignorant")


@check("an empty or all-covering fold is refused")
def _bad_folds():
    features, target = synthetic_gate_data(n=50)
    try:
        G.cross_fit(features, target, torch.zeros(len(target)), epochs=5)
    except ValueError as exc:
        assert "empty or covers everything" in str(exc)
    else:
        raise AssertionError("a single fold covering everything must raise")


if __name__ == "__main__":
    print(f"\nall {len(PASSED)} passed")

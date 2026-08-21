"""Stages 5-7 pure logic — the Shapley target, the arms, and the reliability rename.

    python training/test_stage567.py

Runs without a checkpoint, a GPU or a dataset: the parts that can be wrong in a way no training
run would reveal are all pure functions over synthetic opinions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

import stage567 as S  # noqa: E402
from networks.discern_v2.ds_fusion import Opinion  # noqa: E402

PASSED = []


def check(name):
    def deco(fn):
        fn()
        PASSED.append(name)
        print(f"  ok  {name}")
        return fn
    return deco


def op(p_fake: float, u: float, n: int = 1) -> Opinion:
    """A binomial opinion with the requested projected p(fake) and vacuity."""
    b_fake = p_fake - u / 2
    b_real = 1.0 - u - b_fake
    return Opinion(belief=torch.tensor([[b_real, b_fake]]).repeat(n, 1),
                   vacuity=torch.tensor([[u]]).repeat(n, 1))


def scene(n: int = 8):
    torch.manual_seed(0)
    return {
        "sem": op(0.70, 0.20, n),
        "ref": op(0.60, 0.30, n),
        "proc": op(0.45, 0.40, n),
    }


@check("Shapley weights sum to the full coalition gain (efficiency)")
def _efficiency():
    """The defining property of a Shapley value: the parts add up to U(all) - U(none). If the
    weights were wrong, phi would still look plausible per specialist and never be caught."""
    opinions = scene()
    labels = torch.ones(8, dtype=torch.long)
    specialists = ["ref", "proc"]
    for operator in ("ds", "ccf"):
        phi = S.marginal_utility(opinions, labels, specialists, operator)
        ones = {n: torch.ones(8) for n in opinions}
        u_all = -S._cross_entropy(
            S.fuse(opinions, ["sem", *specialists], ones, operator)["prob"], labels)
        u_none = -S._cross_entropy(S.fuse(opinions, ["sem"], ones, operator)["prob"], labels)
        total = sum(phi.values())
        assert torch.allclose(total, u_all - u_none, atol=1e-5), (
            f"{operator}: Shapley values sum to {float(total[0]):.6f}, coalition gain is "
            f"{float((u_all - u_none)[0]):.6f}")


@check("two specialists reduce to the brief's literal formula")
def _matches_brief_formula():
    """phi_ref = 0.5[U(A+ref) - U(A)] + 0.5[U(A+ref+rate) - U(A+rate)]."""
    opinions = scene()
    labels = torch.ones(8, dtype=torch.long)
    ones = {n: torch.ones(8) for n in opinions}

    def U(subset):
        return -S._cross_entropy(S.fuse(opinions, ["sem", *subset], ones, "ds")["prob"], labels)

    literal = 0.5 * (U(["ref"]) - U([])) + 0.5 * (U(["ref", "proc"]) - U(["proc"]))
    phi = S.marginal_utility(opinions, labels, ["ref", "proc"], "ds")
    assert torch.allclose(phi["ref"], literal, atol=1e-6)


@check("a specialist that agrees with the truth gets positive phi; one that opposes it negative")
def _phi_sign():
    labels = torch.ones(4, dtype=torch.long)             # everything is fake
    opinions = {"sem": op(0.55, 0.30, 4),
                "ref": op(0.95, 0.05, 4),                # confidently right
                "proc": op(0.05, 0.05, 4)}               # confidently wrong
    phi = S.marginal_utility(opinions, labels, ["ref", "proc"], "ds")
    assert float(phi["ref"].mean()) > 0, float(phi["ref"].mean())
    assert float(phi["proc"].mean()) < 0, float(phi["proc"].mean())


@check("redundant specialists are NOT both credited with the full gain")
def _redundancy():
    """The reason the brief replaces V1's pairwise target. Two identical helpful specialists each
    deserve about half the coalition gain; a pairwise target would give each of them all of it."""
    labels = torch.ones(4, dtype=torch.long)
    same = op(0.95, 0.05, 4)
    opinions = {"sem": op(0.55, 0.30, 4), "ref": same, "proc": same}
    phi = S.marginal_utility(opinions, labels, ["ref", "proc"], "ds")
    assert torch.allclose(phi["ref"], phi["proc"], atol=1e-6), "identical specialists, equal share"

    ones = {n: torch.ones(4) for n in opinions}
    pairwise = (-S._cross_entropy(S.fuse(opinions, ["sem", "ref"], ones, "ds")["prob"], labels)
                + S._cross_entropy(S.fuse(opinions, ["sem"], ones, "ds")["prob"], labels))
    assert float(phi["ref"].mean()) < float(pairwise.mean()) - 1e-4, (
        "the Shapley share must be SMALLER than the pairwise gain when specialists are redundant; "
        f"got shapley {float(phi['ref'].mean()):.6f} vs pairwise {float(pairwise.mean()):.6f}")


@check("discounting an inapplicable specialist adds ignorance, not evidence for Real")
def _discount_semantics():
    opinions = scene(4)
    names = ["sem", "ref", "proc"]
    admitted = S.fuse(opinions, names, {n: torch.ones(4) for n in names}, "ds")
    shut_off = S.fuse(opinions, names,
                      {"sem": torch.ones(4), "ref": torch.zeros(4), "proc": torch.zeros(4)}, "ds")
    anchor_only = S.fuse(opinions, ["sem"], {"sem": torch.ones(4)}, "ds")
    assert torch.allclose(shut_off["prob"], anchor_only["prob"], atol=1e-5), (
        "q = 0 must make a specialist VACUOUS, leaving the anchor's opinion untouched. If it "
        "instead pushed p(fake) down, the gate would be voting Real whenever it shut a "
        "specialist off.")
    assert float(shut_off["U_sup"].mean()) > float(admitted["U_sup"].mean())


@check("U_sup is exactly the brief's 1 - 0.5*sum over two specialists")
def _u_sup_identity():
    opinions = scene(4)
    names = ["sem", "ref", "proc"]
    q = {"sem": torch.ones(4), "ref": torch.full((4,), 0.7), "proc": torch.full((4,), 0.2)}
    arm = S.fuse(opinions, names, q, "ds")
    literal = 1.0 - 0.5 * sum(
        q[n] * (1.0 - opinions[n].vacuity.squeeze(1)) for n in ("ref", "proc"))
    assert torch.allclose(arm["U_sup"], literal, atol=1e-6), (
        f"U_sup {float(arm['U_sup'][0]):.6f} vs brief formula {float(literal[0]):.6f}")


@check("V is operator-dependent, so it must be recalibrated per arm")
def _v_per_operator():
    opinions = scene(4)
    names = ["sem", "ref", "proc"]
    ones = {n: torch.ones(4) for n in names}
    v_ds = float(S.fuse(opinions, names, ones, "ds")["V"].mean())
    v_ccf = float(S.fuse(opinions, names, ones, "ccf")["V"].mean())
    assert abs(v_ds - v_ccf) > 1e-4, (
        f"V is identical under both operators ({v_ds:.6f}); if that were true the per-arm "
        f"recalibration Stage 6 requires would be unnecessary — check the operator dispatch")


@check("C and U_sup depend on q but NOT on the operator")
def _c_carries_over():
    """The brief says C and U_sup are pre-fusion quantities that carry over unchanged. True across
    operators; NOT true across arms with different q, which is why they are recomputed per arm."""
    opinions = scene(4)
    names = ["sem", "ref", "proc"]
    q = {"sem": torch.ones(4), "ref": torch.full((4,), 0.6), "proc": torch.full((4,), 0.3)}
    a = S.fuse(opinions, names, q, "ds")
    b = S.fuse(opinions, names, q, "ccf")
    assert torch.allclose(a["C"], b["C"], atol=1e-6)
    assert torch.allclose(a["U_sup"], b["U_sup"], atol=1e-6)
    c = S.fuse(opinions, names, {n: torch.ones(4) for n in names}, "ds")
    assert not torch.allclose(a["U_sup"], c["U_sup"], atol=1e-4), (
        "U_sup must change with q, or the control arm would be indistinguishable from the "
        "applicability arm on this axis")


@check("every arm in ARMS is constructible and the control really uses q = 1")
def _arms_wellformed():
    opinions = scene(6)
    names = ["sem", "ref", "proc"]
    ones = {n: torch.ones(6) for n in names}
    learned = {"sem": torch.ones(6), "ref": torch.full((6,), 0.4), "proc": torch.full((6,), 0.9)}
    q_by_operator = {"ds": learned, "ccf": learned}
    seen = set()
    for label, operator, kind in S.ARMS:
        q = (ones if kind == "equal" else
             q_by_operator[operator] if kind == "own" else q_by_operator[kind])
        arm = S.fuse(opinions, names, q, operator)
        assert torch.isfinite(arm["prob"]).all()
        arm["fused"].assert_normalized()
        seen.add(label)
    assert {"applicability_ccf", "equal_ccf", "applicability_ds",
            "applicability_ds_shared_q"} <= seen, seen
    equal_arm = S.fuse(opinions, names, ones, "ccf")
    applied = S.fuse(opinions, names, learned, "ccf")
    assert not torch.allclose(equal_arm["prob"], applied["prob"], atol=1e-5), (
        "Equal-CCF and Applicability-CCF produced identical predictions, so the control is not "
        "controlling anything — check that q is actually reaching the discount")


@check("the risk-model sign expectations match the brief's pass condition")
def _risk_signs():
    import inspect
    src = inspect.getsource(S.fit_risk)
    assert '"V": +1' in src and '"C": +1' in src and '"U_sup": +1' in src, src[:200]
    assert '"fused_margin": -1' in src, (
        "a more DECIDED margin must LOWER risk; a positive expectation here would invert the "
        "brief's pass condition")


if __name__ == "__main__":
    print(f"\nall {len(PASSED)} passed")

"""§15 mandatory DS unit tests, plus §14/§16/§17 behaviour. Run before any real training.

    python training/networks/discern_v2/test_ds_fusion.py

The spec names the synthetic cases that must be covered: both vacuous; anchor confident with
specialist vacuous; specialist confident with anchor vacuous; confident agreement; confident
disagreement; near-total conflict; q=0; q=1. Each asserts finite outputs, a normalized opinion,
and sensible limiting behaviour.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from networks.discern_v2.ds_fusion import (  # noqa: E402
    CONFLICT_EPS, Opinion, apply_validity, assert_order_invariant, discount, ds_combine,
    js_divergence, reliability)

PASSED = []
REAL, FAKE = 0, 1


def check(name):
    def deco(fn):
        fn()
        PASSED.append(name)
        print(f"  ok: {name}")
        return fn
    return deco


def opinion(belief_real: float, belief_fake: float) -> Opinion:
    """One-sample opinion from explicit masses; vacuity is whatever is left."""
    b = torch.tensor([[belief_real, belief_fake]])
    return Opinion(belief=b, vacuity=1.0 - b.sum(dim=1, keepdim=True)).assert_normalized()


CONFIDENT_FAKE = opinion(0.02, 0.96)
CONFIDENT_REAL = opinion(0.96, 0.02)
VACUOUS = Opinion.vacuous(1)
MILD_FAKE = opinion(0.20, 0.50)


def assert_finite(op: Opinion, where: str) -> None:
    assert torch.isfinite(op.belief).all(), f"{where}: non-finite belief {op.belief}"
    assert torch.isfinite(op.vacuity).all(), f"{where}: non-finite vacuity {op.vacuity}"
    op.assert_normalized()


# ---------------------------------------------------------------- §7 / conversions


@check("evidence -> opinion -> evidence round-trips")
def _round_trip():
    e = torch.tensor([[3.0, 9.0], [0.0, 0.0], [50.0, 1.0]])
    op = Opinion.from_evidence(e)
    assert_finite(op, "from_evidence")
    back = op.to_dirichlet().evidence
    assert torch.allclose(back, e, atol=1e-4), (back, e)


@check("zero evidence is exactly the vacuous opinion")
def _zero_evidence():
    op = Opinion.from_evidence(torch.zeros(3, 2))
    assert torch.allclose(op.belief, torch.zeros(3, 2))
    assert torch.allclose(op.vacuity, torch.ones(3, 1))


# ---------------------------------------------------------------- §15 mandatory cases


@check("§15: both vacuous -> fused stays vacuous")
def _both_vacuous():
    fused, _ = ds_combine([VACUOUS, VACUOUS])
    assert_finite(fused, "both vacuous")
    assert torch.allclose(fused.vacuity, torch.ones(1, 1), atol=1e-6), fused.vacuity
    assert torch.allclose(fused.belief, torch.zeros(1, 2), atol=1e-6)


@check("§15: anchor confident + specialist vacuous -> anchor survives unchanged")
def _anchor_only():
    fused, _ = ds_combine([CONFIDENT_FAKE, VACUOUS])
    assert_finite(fused, "anchor + vacuous")
    assert torch.allclose(fused.belief, CONFIDENT_FAKE.belief, atol=1e-6), fused.belief
    assert torch.allclose(fused.vacuity, CONFIDENT_FAKE.vacuity, atol=1e-6)


@check("§15: specialist confident + anchor vacuous -> specialist carries the sample")
def _specialist_only():
    fused, _ = ds_combine([VACUOUS, CONFIDENT_FAKE])
    assert_finite(fused, "vacuous + specialist")
    assert torch.allclose(fused.belief, CONFIDENT_FAKE.belief, atol=1e-6)


@check("§15: confident agreement -> more confident than either input")
def _agreement():
    fused, diag = ds_combine([CONFIDENT_FAKE, CONFIDENT_FAKE])
    assert_finite(fused, "agreement")
    assert float(fused.vacuity) < float(CONFIDENT_FAKE.vacuity), (
        "agreeing views must reduce vacuity")
    assert float(fused.fake_prob()) > float(CONFIDENT_FAKE.fake_prob())
    # DS conflict is the off-diagonal belief mass, so two agreeing-but-not-certain opinions
    # still register a little (0.02*0.96*2 here). What must hold is that agreement is an order
    # of magnitude below disagreement, not that it is zero.
    _, disagree = ds_combine([CONFIDENT_FAKE, CONFIDENT_REAL])
    assert float(diag["ds_conflict_max"]) < 0.1 * float(disagree["ds_conflict_max"]), (
        float(diag["ds_conflict_max"]), float(disagree["ds_conflict_max"]))


@check("§15: confident disagreement -> finite, high conflict, no class dominates")
def _disagreement():
    fused, diag = ds_combine([CONFIDENT_FAKE, CONFIDENT_REAL])
    assert_finite(fused, "disagreement")
    assert float(diag["ds_conflict_max"]) > 0.9, diag["ds_conflict_max"]
    p = float(fused.fake_prob())
    assert 0.2 < p < 0.8, f"symmetric disagreement must not resolve to a confident class: {p}"


@check("§15: NEAR-TOTAL conflict stays finite (DiCoME's unguarded divisor would NaN)")
def _total_conflict():
    a = opinion(0.0, 1.0)      # certain fake
    b = opinion(1.0, 0.0)      # certain real
    fused, diag = ds_combine([a, b])
    assert_finite(fused, "total conflict")
    assert float(diag["ds_conflict_max"]) > 0.999
    # unguarded, belief would be 0/0; with the guard the result is finite and reports the
    # conflict rather than poisoning every downstream reliability signal
    unguarded_denominator = 1.0 - float(diag["ds_conflict_max"])
    assert unguarded_denominator < CONFLICT_EPS * 10, unguarded_denominator
    # Dempster's rule is UNDEFINED here (all mass on the empty set), so the fallback is the
    # vacuous opinion and the sample is flagged. Both halves matter: annihilating views must
    # read as "nothing is known", and a fallback nobody counts looks like one that never fires.
    assert bool(diag["degenerate"][0]), "total conflict must be flagged per sample"
    assert float(fused.vacuity) == 1.0, fused.vacuity
    # ... while C still reports the disagreement, from the pre-fusion opinions
    r = reliability({"sem": a, "ref": b}, {"sem": torch.ones(1), "ref": torch.ones(1)}, fused)
    assert float(r["V"]) == 1.0 and float(r["C"]) > 0.99, (float(r["V"]), float(r["C"]))


@check("a normal fusion is NOT flagged degenerate")
def _not_degenerate():
    _, diag = ds_combine([CONFIDENT_FAKE, MILD_FAKE])
    assert not bool(diag["degenerate"].any())


@check("§15: q = 0 -> the specialist has NO influence")
def _q_zero():
    with_spec, _ = ds_combine([CONFIDENT_REAL, discount(CONFIDENT_FAKE, torch.zeros(1))])
    alone, _ = ds_combine([CONFIDENT_REAL])
    assert_finite(with_spec, "q=0")
    assert torch.allclose(with_spec.belief, alone.belief, atol=1e-6), (
        "a fully discounted specialist must leave the fusion identical")


@check("§15: q = 1 -> discounting is the identity")
def _q_one():
    d = discount(MILD_FAKE, torch.ones(1))
    assert torch.allclose(d.belief, MILD_FAKE.belief, atol=1e-6)
    assert torch.allclose(d.vacuity, MILD_FAKE.vacuity, atol=1e-6)


@check("§15: combination order does not change the result beyond tolerance")
def _order():
    worst = assert_order_invariant([CONFIDENT_FAKE, MILD_FAKE, opinion(0.3, 0.1)])
    assert worst < 1e-4, worst


# ---------------------------------------------------------------- §14 discounting


@check("§14.2: discounting moves mass to IGNORANCE, not to Real")
def _discount_is_ignorance():
    half = discount(CONFIDENT_FAKE, torch.tensor([0.5]))
    half.assert_normalized()
    assert float(half.belief[0, FAKE]) < float(CONFIDENT_FAKE.belief[0, FAKE])
    assert float(half.belief[0, REAL]) < float(CONFIDENT_FAKE.belief[0, REAL]), (
        "discounting must not increase belief in Real — that would make an inapplicable "
        "specialist an argument for authenticity")
    assert float(half.vacuity) > float(CONFIDENT_FAKE.vacuity)


@check("§14.2: a discounted specialist cannot outvote the anchor as q falls")
def _monotone_influence():
    previous = None
    for q in (1.0, 0.75, 0.5, 0.25, 0.0):
        fused, _ = ds_combine([CONFIDENT_REAL, discount(CONFIDENT_FAKE, torch.tensor([q]))])
        p_fake = float(fused.fake_prob())
        if previous is not None:
            assert p_fake <= previous + 1e-6, f"p(fake) rose as q fell: {p_fake} > {previous}"
        previous = p_fake


@check("§14.1: an invalid branch becomes vacuous, per sample")
def _validity():
    op = Opinion(belief=torch.tensor([[0.1, 0.8], [0.1, 0.8]]),
                 vacuity=torch.tensor([[0.1], [0.1]]))
    out = apply_validity(op, torch.tensor([1.0, 0.0]))
    out.assert_normalized()
    assert torch.allclose(out.belief[0], op.belief[0]), "a valid sample must be untouched"
    assert torch.allclose(out.belief[1], torch.zeros(2)), "an invalid sample must be vacuous"
    assert float(out.vacuity[1]) == 1.0


# ---------------------------------------------------------------- §16 / §17


@check("§16: fused opinion converts back to a valid Dirichlet")
def _back_to_edl():
    fused, _ = ds_combine([CONFIDENT_FAKE, MILD_FAKE])
    state = fused.to_dirichlet()
    assert torch.isfinite(state.alpha).all()
    assert (state.alpha >= 1.0).all(), "alpha < 1 is not a valid evidential posterior"
    assert torch.allclose(state.p.sum(dim=1), torch.ones(1), atol=1e-5)


@check("§17: V/C/A separate ignorance, disagreement and missing support")
def _vca():
    q_all = {"sem": torch.ones(1), "ref": torch.ones(1), "proc": torch.ones(1)}

    # everyone ignorant -> V high, A high, C undefined-but-zero (no informative pair)
    ops = {"sem": VACUOUS, "ref": VACUOUS, "proc": VACUOUS}
    fused, _ = ds_combine(list(ops.values()))
    r = reliability(ops, q_all, fused)
    assert float(r["V"]) > 0.99 and float(r["A"]) > 0.99
    assert float(r["C"]) < 1e-3, "vacuous views must not read as agreement OR as conflict"

    # informed disagreement -> C high while V is low
    ops = {"sem": CONFIDENT_FAKE, "ref": CONFIDENT_REAL, "proc": VACUOUS}
    fused, _ = ds_combine(list(ops.values()))
    r = reliability(ops, q_all, fused)
    assert float(r["C"]) > 0.5, float(r["C"])
    assert float(r["V"]) < 0.5

    # anchor alone with specialists gated off -> A high, C zero
    ops = {"sem": CONFIDENT_FAKE, "ref": CONFIDENT_REAL, "proc": CONFIDENT_REAL}
    q_off = {"sem": torch.ones(1), "ref": torch.zeros(1), "proc": torch.zeros(1)}
    fused, _ = ds_combine([ops["sem"],
                           discount(ops["ref"], torch.zeros(1)),
                           discount(ops["proc"], torch.zeros(1))])
    r = reliability(ops, q_off, fused)
    assert float(r["A"]) > 0.99, "no applicable specialist means unsupported"
    assert float(r["C"]) < 1e-3, "an inapplicable specialist's disagreement is not conflict"


@check("§17: JS divergence is bounded in [0, 1] and zero for identical inputs")
def _js():
    p = torch.tensor([[0.5, 0.5], [1.0, 0.0], [0.3, 0.7]])
    q = torch.tensor([[0.5, 0.5], [0.0, 1.0], [0.3, 0.7]])
    d = js_divergence(p, q)
    assert torch.isfinite(d).all()
    assert float(d[0]) < 1e-6 and float(d[2]) < 1e-6
    assert 0.99 < float(d[1]) <= 1.0, f"maximal disagreement should be ~1 bit, got {float(d[1])}"


@check("batched fusion matches per-sample fusion")
def _batching():
    a = Opinion(belief=torch.tensor([[0.1, 0.7], [0.6, 0.2]]),
                vacuity=torch.tensor([[0.2], [0.2]]))
    b = Opinion(belief=torch.tensor([[0.5, 0.3], [0.05, 0.9]]),
                vacuity=torch.tensor([[0.2], [0.05]]))
    batched, _ = ds_combine([a, b])
    for i in range(2):
        single, _ = ds_combine([
            Opinion(a.belief[i:i + 1], a.vacuity[i:i + 1]),
            Opinion(b.belief[i:i + 1], b.vacuity[i:i + 1])])
        assert torch.allclose(batched.belief[i:i + 1], single.belief, atol=1e-6)


@check("an unnormalized opinion is refused rather than silently fused")
def _refuses_bad_opinion():
    bad = Opinion(belief=torch.tensor([[0.9, 0.9]]), vacuity=torch.tensor([[0.5]]))
    try:
        ds_combine([bad, VACUOUS])
    except ValueError as exc:
        assert "not normalized" in str(exc)
    else:
        raise AssertionError("sum(belief) + u = 2.3 must be refused")


if __name__ == "__main__":
    print(f"\nall {len(PASSED)} passed")

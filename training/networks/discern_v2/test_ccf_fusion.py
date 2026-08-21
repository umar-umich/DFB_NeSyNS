"""CCF is validated against the source paper's published numbers, not against our own reasoning.

    python training/networks/discern_v2/test_ccf_fusion.py

The binary specialisation in `ccf_fusion` was derived by working equation (7)'s four terms through
for a two-element domain with singleton-only input belief. A derivation is only as good as its
check, so the primary test reproduces **Table I of van der Heijden, Kopp & Kargl (FUSION 2018)**
exactly: if any of the four terms were mis-specialised, the fused triplet would miss the published
row, and three of the four numbers would have to coincide by accident for the error to hide.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from networks.discern_v2.ccf_fusion import ccf_combine, ccf_is_non_associative  # noqa: E402
from networks.discern_v2.ds_fusion import Opinion, ds_combine  # noqa: E402

PASSED = []


def check(name):
    def deco(fn):
        fn()
        PASSED.append(name)
        print(f"  ok  {name}")
        return fn
    return deco


def op(b_x: float, b_xbar: float, u: float) -> Opinion:
    """Paper convention: belief on (x, x̄) plus uncertainty. Column order here is (Real, Fake)."""
    return Opinion(belief=torch.tensor([[b_x, b_xbar]]), vacuity=torch.tensor([[u]]))


def close(a, b, tol=5e-4):
    return abs(float(a) - float(b)) <= tol


# Table I inputs, reused by several checks
A1, A2, A3 = op(0.10, 0.30, 0.60), op(0.40, 0.20, 0.40), op(0.70, 0.10, 0.20)


@check("reproduces Table I of the source paper")
def _table_i():
    fused = ccf_combine([A1, A2, A3])
    got = (float(fused.belief[0, 0]), float(fused.belief[0, 1]), float(fused.vacuity[0, 0]),
           float(fused.probability()[0, 0]))
    want = (0.629, 0.182, 0.189, 0.723)     # FUSION 2018, Table I, CCF column
    for g, w, label in zip(got, want, ("b(x)", "b(x̄)", "u", "P(x)")):
        assert close(g, w), f"{label}: got {g:.6f}, paper says {w}"


@check("output is a valid opinion")
def _valid():
    ccf_combine([A1, A2, A3]).assert_normalized()
    fused = ccf_combine([A1, A2, A3])
    assert (fused.belief >= 0).all() and (fused.vacuity >= 0).all()


@check("conflict becomes vagueness, not a winner")
def _conflict_to_vagueness():
    """The property CCF is chosen for, contrasted directly with Dempster on the same inputs."""
    a, b = op(0.90, 0.05, 0.05), op(0.05, 0.90, 0.05)
    ccf = ccf_combine([a, b])
    ds, _ = ds_combine([a, b])
    assert float(ccf.vacuity[0, 0]) > 0.5, "opposed specialists must raise vacuity"
    assert abs(float(ccf.belief[0, 0] - ccf.belief[0, 1])) < 0.1, "neither side may win"
    assert float(ds.vacuity[0, 0]) < float(ccf.vacuity[0, 0]), (
        "Dempster should renormalise the conflict away; if it does not, this comparison is not "
        "showing what it claims")


@check("identical opinions are returned unchanged (eta is undefined there)")
def _identical():
    one = op(0.6, 0.2, 0.2)
    fused = ccf_combine([one, one, one])
    assert torch.allclose(fused.belief, one.belief, atol=1e-5)
    assert torch.allclose(fused.vacuity, one.vacuity, atol=1e-5)


@check("vacuous inputs stay vacuous")
def _vacuous():
    vac = Opinion.vacuous(1, 2)
    assert close(ccf_combine([vac, vac]).vacuity[0, 0], 1.0, 1e-5)


@check("a vacuous source does not erase an informed one")
def _vacuous_neutral():
    fused = ccf_combine([op(0.7, 0.1, 0.2), Opinion.vacuous(1, 2)])
    assert float(fused.belief[0, 0]) > float(fused.belief[0, 1])


@check("dogmatic source (u = 0) does not divide by zero")
def _dogmatic():
    fused = ccf_combine([op(0.8, 0.2, 0.0), op(0.3, 0.3, 0.4)])
    assert torch.isfinite(fused.belief).all() and torch.isfinite(fused.vacuity).all()
    fused.assert_normalized()


@check("non-associativity is real on these inputs")
def _non_associative():
    """The reason the brief forbids sequential pairwise fusion, measured rather than asserted."""
    assert ccf_is_non_associative(A1, A2, A3)
    chained = ccf_combine([ccf_combine([A1, A2]), A3])
    proper = ccf_combine([A1, A2, A3])
    gap = float((chained.belief - proper.belief).abs().max())
    print(f"      pairwise-chained vs proper multi-source: max |delta b| = {gap:.4f}")


@check("order invariance: CCF is symmetric even though it is not associative")
def _order_invariant():
    import itertools
    ref = ccf_combine([A1, A2, A3])
    for perm in itertools.permutations([A1, A2, A3]):
        other = ccf_combine(list(perm))
        assert torch.allclose(ref.belief, other.belief, atol=1e-5)


@check("batched fusion matches per-sample fusion")
def _batched():
    rows = [(0.10, 0.30, 0.60), (0.90, 0.05, 0.05), (0.2, 0.2, 0.6)]
    second = [(0.40, 0.20, 0.40), (0.05, 0.90, 0.05), (0.3, 0.3, 0.4)]
    batch = [
        Opinion(belief=torch.tensor([[b, f] for b, f, _ in rows]),
                vacuity=torch.tensor([[u] for _, _, u in rows])),
        Opinion(belief=torch.tensor([[b, f] for b, f, _ in second]),
                vacuity=torch.tensor([[u] for _, _, u in second])),
    ]
    fused = ccf_combine(batch)
    for i, (r1, r2) in enumerate(zip(rows, second)):
        single = ccf_combine([op(*r1), op(*r2)])
        assert torch.allclose(fused.belief[i], single.belief[0], atol=1e-5)
        assert torch.allclose(fused.vacuity[i], single.vacuity[0], atol=1e-5)


@check("refuses a non-binary domain instead of silently mis-specialising")
def _rejects_multinomial():
    three = Opinion(belief=torch.tensor([[0.3, 0.3, 0.2]]), vacuity=torch.tensor([[0.2]]))
    try:
        ccf_combine([three, three])
    except NotImplementedError as exc:
        assert "general form" in str(exc)
    else:
        raise AssertionError("a 3-class domain must be refused, not fused with binary terms")


if __name__ == "__main__":
    print(f"\nall {len(PASSED)} passed")

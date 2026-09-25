#!/usr/bin/env python3
"""Unit tests for the corrected IBDC loss and the differentiable PCR6+ fusion (brief Task 1).

    python -m pytest training/joint_ibdc/test_ibdc.py -v
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fusion_torch as F  # noqa: E402

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")


def masses_from_evidence(ev_list, r=None):
    """ev_list: list of (n,2) evidence tensors -> (n, S, 3) discounted opinions."""
    e = torch.stack(ev_list, dim=1)                       # (n, S, 2)
    if r is not None:
        e = e * torch.as_tensor(r, dtype=e.dtype).view(1, -1, 1)
    return F.opinion_of_evidence(e)


# ---------------------------------------------------------------- the deployed spec agrees
def test_combinatorics_match_deployed_spec():
    """My independently derived tuples and keeping indexes must equal the deployed numpy ones."""
    sys.path[:0] = [str(ROOT / "analysis/tbiom/pcr6p"), str(ROOT / "analysis/tbiom/grouped"),
                    str(ROOT / "analysis/tbiom"), str(ROOT / "analysis/tbiom/temporal")]
    import ops_pcr6p as P
    for S in (2, 3):
        mine, theirs = F.spec(S), P.spec(S)
        assert len(mine) == len(theirs) == 3 ** S
        for a, b in zip(mine, theirs):
            assert a["tup"] == b["tup"] and a["inter"] == b["inter"]
            assert a["conflict"] == b["conflict"]
            assert a["kappa"] == b["kappa"], (a["tup"], a["kappa"], b["kappa"])


def test_theta_keeping_index_is_zero_in_conflict():
    """Remark 1: Theta never keeps mass in a conflicting tuple, which is PCR6+'s neutrality."""
    for t in F.spec(3):
        if t["conflict"] and F.THETA in set(t["tup"]):
            assert t["kappa"][F.THETA] == 0


# ---------------------------------------------------------------- the six required assertions
def test_shared_ignorance_is_inert():
    """All branches at evidence (0.01, 0.01): kappa < 1e-3, loss 0, and no gradient."""
    n = 16
    ev = [torch.full((n, 2), 0.01, requires_grad=True) for _ in range(3)]
    m = masses_from_evidence(ev)
    k = F.kappa(m)
    assert k.max() < 1e-3, k.max()
    loss = F.ibdc_loss(m, F.fuse(m))
    assert float(loss) == 0.0
    loss.backward()
    for e in ev:
        assert e.grad is None or torch.allclose(e.grad, torch.zeros_like(e.grad)), e.grad.abs().max()


def test_confident_disagreement_raises_vacuity():
    """(100,0) against (0,100): loss > 0, and one gradient step increases u_f."""
    n = 8
    e1 = torch.tensor([[100.0, 0.0]]).repeat(n, 1).clone().requires_grad_(True)
    e2 = torch.tensor([[0.0, 100.0]]).repeat(n, 1).clone().requires_grad_(True)
    opt = torch.optim.SGD([e1, e2], lr=50.0)
    m = masses_from_evidence([e1, e2])
    f = F.fuse(m)
    u0 = float(F.vacuity_of(f).mean())
    loss = F.ibdc_loss(m, f)
    assert float(loss) > 0.0
    opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        u1 = float(F.vacuity_of(F.fuse(masses_from_evidence([e1, e2]))).mean())
    assert u1 > u0, (u0, u1)


def test_confident_agreement_is_inert():
    """Two branches both at (100, 0): no conflict, so loss 0."""
    n = 8
    ev = [torch.tensor([[100.0, 0.0]]).repeat(n, 1) for _ in range(2)]
    m = masses_from_evidence(ev)
    assert float(F.kappa(m).max()) < 1e-6
    assert float(F.ibdc_loss(m, F.fuse(m))) == 0.0


def test_zero_trust_branch_contributes_no_conflict():
    """A branch at r = 0 is vacuous, so kappa_ij = 0 for every pair it is in."""
    n = 8
    e1 = torch.tensor([[100.0, 0.0]]).repeat(n, 1)
    e2 = torch.tensor([[0.0, 100.0]]).repeat(n, 1)
    e3 = torch.tensor([[50.0, 50.0]]).repeat(n, 1)
    m = masses_from_evidence([e1, e2, e3], r=[1.0, 0.0, 1.0])
    kij = F.pairwise_kappa(m)
    pairs = list(itertools.combinations(range(3), 2))
    for col, (i, j) in enumerate(pairs):
        if 1 in (i, j):                                   # every pair containing the zeroed source
            assert float(kij[:, col].abs().max()) < 1e-12, (i, j, kij[:, col])
    assert float(m[:, 1, 2].min()) == pytest.approx(1.0)  # that source is vacuous


def test_gradient_flows_only_through_vacuity():
    """kappa is detached, so d(loss)/d(evidence) must equal -d(mean u_f)/d(evidence) where active."""
    n = 32
    torch.manual_seed(0)
    base = [torch.rand(n, 2) * 20 for _ in range(3)]
    ev_a = [b.clone().requires_grad_(True) for b in base]
    m = masses_from_evidence(ev_a)
    f = F.fuse(m)
    active = (F.kappa(m) > F.vacuity_of(f))
    assert active.any(), "need active hinge rows for this test to mean anything"
    F.ibdc_loss(m, f).backward()

    ev_b = [b.clone().requires_grad_(True) for b in base]
    m2 = masses_from_evidence(ev_b)
    f2 = F.fuse(m2)
    (-(F.vacuity_of(f2) * active.float()).sum() / n).backward()
    for a, b in zip(ev_a, ev_b):
        assert torch.allclose(a.grad, b.grad, atol=1e-6), (a.grad - b.grad).abs().max()


def test_original_ibdc_lowers_vacuity_under_shared_ignorance():
    """Reviewer 3's failure case, reproduced: ibdc_v1 pushes u_f DOWN when everyone is ignorant."""
    n = 16
    ev = [torch.full((n, 2), 0.01).requires_grad_(True) for _ in range(3)]
    opt = torch.optim.SGD(ev, lr=1.0)
    m = masses_from_evidence(ev)
    f = F.fuse(m)
    u0 = float(F.vacuity_of(f).mean())
    loss = F.ibdc_v1([e for e in ev], F.vacuity_of(f))
    opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        u1 = float(F.vacuity_of(F.fuse(masses_from_evidence(ev))).mean())
    assert u1 < u0, f"expected the original loss to lower vacuity, got {u0} -> {u1}"

    # and the corrected loss does nothing at all on the same input
    ev2 = [torch.full((n, 2), 0.01).requires_grad_(True) for _ in range(3)]
    m2 = masses_from_evidence(ev2)
    assert float(F.ibdc_loss(m2, F.fuse(m2))) == 0.0


# ---------------------------------------------------------------- algebra sanity
def test_opinion_evidence_roundtrip():
    torch.manual_seed(0)
    e = torch.rand(64, 3, 2) * 30
    o = F.opinion_of_evidence(e)
    assert torch.allclose(o.sum(-1), torch.ones(64, 3), atol=1e-6)
    assert torch.allclose(F.evidence_of(o), e, atol=1e-4)


def test_discount_endpoints():
    torch.manual_seed(0)
    e = torch.rand(32, 2) * 10
    o = F.opinion_of_evidence(e)
    assert torch.allclose(F.discount(o, 1.0), o, atol=1e-6)
    z = F.discount(o, 0.0)
    assert torch.allclose(z[..., 2], torch.ones(32), atol=1e-6)      # r = 0 is vacuous


def test_fused_masses_are_a_distribution():
    torch.manual_seed(0)
    for S in (2, 3):
        m = torch.rand(50, S, 3); m = m / m.sum(-1, keepdim=True)
        f = F.fuse(m, plus=True)
        assert torch.allclose(f.sum(1), torch.ones(50), atol=1e-6)
        assert (f >= -1e-9).all()

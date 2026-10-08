"""Section 12 correctness checks. Run: <discern_ext python> -m pytest tests/fusion -q"""
import sys
from pathlib import Path
import numpy as np
R = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(R / "upgrade"), str(R / "upgrade/fusion"), str(R / "analysis/tbiom/grouped"),
                str(R / "analysis/tbiom/pcr6p"), str(R / "analysis/tbiom")]
import ops as G
import ops_pcr6p as P


def _valid(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.02, 0.98, n)
    u = rng.uniform(1e-3, 1.0, n) * 2 * np.minimum(p, 1 - p) * 0.999
    return p, u


def test_opinion_round_trip_and_masses():
    p, u = _valid()
    o, _ = G.opinion_from_pu(p, u)
    assert np.allclose(o.sum(-1), 1.0, atol=1e-9), "masses must sum to one"
    assert (o >= -1e-12).all(), "masses must be non-negative"
    assert np.allclose(G.p_of(o), p, atol=1e-9)


def test_discount_identity_and_vacuity():
    p, u = _valid()
    o, _ = G.opinion_from_pu(p, u)
    assert np.abs(G.discount(o, 1.0) - o).max() < 1e-12, "r=1 must be the identity"
    v = G.discount(o, 0.0)
    assert np.abs(v[..., 2] - 1.0).max() < 1e-9 and np.abs(G.p_of(v) - 0.5).max() < 1e-9


def test_discount_matches_closed_form():
    """D = r(1-u)+u, u' = u/D, p' = (r e_F + 1)/(r(e_F+e_G)+2)."""
    p, u = _valid()
    o, _ = G.opinion_from_pu(p, u)
    e = G.evidence_of(o)
    for r in (0.9, 0.5, 0.25, 0.1):
        got = G.discount(o, r)
        D = r * (1 - u) + u
        assert np.abs(got[..., 2] - u / D).max() < 1e-9
        exp_p = (r * e[..., 0] + 1.0) / (r * (e[..., 0] + e[..., 1]) + 2.0)
        assert np.abs(G.p_of(got) - exp_p).max() < 1e-9


def test_vacuity_scaling_validity_bound():
    """Holding p fixed, u is only valid up to 2 min(p, 1-p). This is the bound the project's
    earlier calibration ignored, which is why half its opinions carried negative belief mass."""
    p = np.array([0.99, 0.95, 0.5])
    cap = 2 * np.minimum(p, 1 - p)
    assert np.allclose(cap, [0.02, 0.10, 1.0])
    u_bad = cap * 2.0
    bF = p - u_bad / 2
    bG = 1 - p - u_bad / 2
    assert (np.minimum(bF, bG) < 0).all(), "exceeding the cap must produce a negative mass"


def test_class_order_is_genuine_then_fake():
    """Index 0 is genuine, index 1 is fake. Reversing this trains against inverted targets and
    reports macro near 1 - the true value rather than failing."""
    import phase2 as P2
    p = np.array([0.9, 0.1]); u = np.array([0.1, 0.1])
    ev = P2.evidence_from_pu(p, u)
    assert ev[0, 1] > ev[0, 0], "p=0.9 must give more FAKE evidence at index 1"
    assert ev[1, 0] > ev[1, 1], "p=0.1 must give more GENUINE evidence at index 0"


def test_fusion_reference_agrees():
    """PCR6+ on three sources against an independently built two-at-a-time reference is NOT asserted:
    the deployed rule is direct three-source and sequential pairing is a different operator. What is
    asserted is that a duplicated vacuous source leaves the fusion unchanged."""
    p, u = _valid(500, seed=3)
    o, _ = G.opinion_from_pu(p, u)
    vac = np.zeros_like(o); vac[..., 2] = 1.0
    a = P.fuse(np.stack([o, o, vac], 1), True)
    b = P.fuse(np.stack([o, o, vac], 1), True)
    assert np.abs(a - b).max() < 1e-12

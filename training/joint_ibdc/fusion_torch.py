#!/usr/bin/env python3
"""Differentiable discounting and PCR6/PCR6+ fusion, plus the corrected one-sided IBDC loss.

This is a PyTorch re-implementation of the deployed numpy fusion in
`analysis/tbiom/pcr6p/ops_pcr6p.py` and the shared algebra in `analysis/tbiom/grouped/ops.py`.
It is written INDEPENDENTLY of those modules rather than wrapping them, so that
`verify_fusion.py` comparing the two on real development videos is a genuine cross-check and not a
comparison of a function with itself. The tuple enumeration and keeping indexes are re-derived here
from the same equations (Dezert, Dezert & Smarandache, JAIF 16(1) 2021, eqs 15, 22-24, 26) and
`test_ibdc.py` asserts they agree element-for-element with the deployed spec.

THE DISCOUNT IS EVIDENCE SCALING, e' = r * e, applied in the evidence domain. It is NOT classical
belief-mass (Shafer) discounting; the two agree only at r = 0 and r = 1.

VACUITY. u = K/S is called evidential vacuity throughout, never epistemic uncertainty.
"""
from __future__ import annotations

import itertools

import torch

W_FRAME = 2.0                                  # binary frame: |Theta| = 2
EPS = 1e-12
FAKE, GEN, THETA = 0b01, 0b10, 0b11
ELEMS = (FAKE, GEN, THETA)
IDX = {FAKE: 0, GEN: 1, THETA: 2}              # column order of an opinion (b1, b0, u)


# --------------------------------------------------------------------- representation
def opinion_from_pu(p, u):
    """(p_fake, vacuity) -> (b1, b0, u)."""
    u = u.clamp(0.0, 1.0)
    return torch.stack([p - u / 2, 1 - p - u / 2, u], dim=-1)


def evidence_of(o):
    """(b1, b0, u) -> (e_fake, e_genuine). Undefined at u = 0."""
    u = o[..., 2]
    return torch.stack([W_FRAME * o[..., 0] / u, W_FRAME * o[..., 1] / u], dim=-1)


def opinion_of_evidence(e):
    S = e.sum(-1) + W_FRAME
    return torch.stack([e[..., 0] / S, e[..., 1] / S, W_FRAME / S.expand_as(e[..., 0])], dim=-1)


def discount(o, r):
    """Evidence scaling e' = r*e. `r` is a scalar or broadcasts over the source axis."""
    if r is None:
        return o
    return opinion_of_evidence(evidence_of(o) * r)


def discount_evidence(e, r):
    """The same discount applied directly to evidence, for heads that emit evidence."""
    return e if r is None else e * r


# --------------------------------------------------------------------- PCR6 / PCR6+ combinatorics
def _card(x):
    return bin(x).count("1")


def _subset(a, b):
    """delta_j(a, b) of eq. (22): 1 when a is a subset of b."""
    return int((a & b) == a)


def keeping_indexes(tup):
    """kappa_j over the distinct focal elements of one tuple, eqs. (22)-(24)."""
    Xj = sorted(set(tup))
    kap = {}
    for A in Xj:
        prod = 1
        for lp, l in itertools.permutations(Xj, 2):
            if _card(A) <= _card(l) and _card(lp) <= _card(l):
                prod *= _subset(lp, l)
        kap[A] = 1 - prod
    return kap


_SPEC: dict[int, list] = {}


def spec(S: int):
    """Every S-tuple of focal elements with its intersection, conflict flag and keeping indexes."""
    if S not in _SPEC:
        out = []
        for tup in itertools.product(ELEMS, repeat=S):
            inter = THETA
            for x in tup:
                inter &= x
            out.append({"tup": tup, "inter": inter, "conflict": inter == 0,
                        "kappa": keeping_indexes(tup) if inter == 0 else None})
        _SPEC[S] = out
    return _SPEC[S]


def fuse(masses: torch.Tensor, plus: bool = True):
    """masses (n, S, 3) of discounted opinions -> fused (n, 3). Differentiable in `masses`.

    `plus` selects PCR6+ (eq. 26) over PCR6 (eq. 15). On a binary frame PCR6+ is PCR6 with the
    conflicting mass never redistributed to Theta, because Theta's keeping index is 0 whenever it
    appears in a conflicting tuple (Remark 1).
    """
    if masses.dim() != 3 or masses.shape[-1] != 3:
        raise ValueError(f"expected (n, S, 3), got {tuple(masses.shape)}")
    n, S, _ = masses.shape
    m = {X: masses[:, :, IDX[X]] for X in ELEMS}                 # each (n, S)
    out = {X: masses.new_zeros(n) for X in ELEMS}
    for t in spec(S):
        pi = masses.new_ones(n)
        for i, x in enumerate(t["tup"]):
            pi = pi * m[x][:, i]
        if not t["conflict"]:
            out[t["inter"]] = out[t["inter"]] + pi
            continue
        kap = t["kappa"] if plus else {x: 1 for x in set(t["tup"])}
        hold = {X: sum(m[X][:, i] for i, x in enumerate(t["tup"]) if x == X) for X in set(t["tup"])}
        den = sum(kap[X] * hold[X] for X in hold)
        ok = den > EPS                                            # zero denominator: term discarded
        share = torch.where(ok, pi / torch.where(ok, den, torch.ones_like(den)),
                            torch.zeros_like(den))
        for A in hold:
            if kap[A]:
                out[A] = out[A] + kap[A] * hold[A] * share
    return torch.stack([out[FAKE], out[GEN], out[THETA]], dim=1)


def p_of(f):
    return (f[..., 0] + f[..., 2] / 2).clamp(0.0, 1.0)


def vacuity_of(f):
    """Fused evidential vacuity u_f, i.e. the mass left on Theta."""
    return f[..., 2]


# --------------------------------------------------------------------- conflict and IBDC
def pairwise_kappa(masses: torch.Tensor) -> torch.Tensor:
    """kappa_ij = m~_i(real)*m~_j(fake) + m~_i(fake)*m~_j(real) for every pair. Returns (n, n_pairs).

    A source discounted to r = 0 is vacuous, so both its singleton masses are 0 and it contributes
    kappa = 0 to every pair -- consistent with fusion ignoring it.
    """
    n, S, _ = masses.shape
    b1, b0 = masses[:, :, 0], masses[:, :, 1]
    cols = [b0[:, i] * b1[:, j] + b1[:, i] * b0[:, j] for i, j in itertools.combinations(range(S), 2)]
    return torch.stack(cols, dim=1)


def kappa(masses: torch.Tensor) -> torch.Tensor:
    """kappa = stop_gradient(max over pairs). Detached, as the brief specifies."""
    return pairwise_kappa(masses).max(dim=1).values.detach()


def ibdc_loss(masses: torch.Tensor, fused: torch.Tensor) -> torch.Tensor:
    """Corrected, one-sided IBDC: mean over x of max(0, kappa(x) - u_f(x)).

    kappa is detached, so gradient reaches the evidence only through u_f. Under shared ignorance all
    branches are vacuous, their singleton masses vanish, kappa is ~0 and the hinge is inert -- which
    is the failure of the symmetric IJCB form that this replaces.
    """
    return torch.relu(kappa(masses) - vacuity_of(fused)).mean()


def ibdc_v1(branch_evidences, fused_vacuity, eps: float = 1e-8) -> torch.Tensor:
    """The ORIGINAL IJCB IBDC, ported exactly from
    `training/networks/nesy_defake/losses/nesy_edl_loss.py::_disagreement_calibration` (version 'v1').

        d   = mean over pairs of (1 - cos(p_i, p_j)), uniform weights, clamped to [0, 1], DETACHED
        L   = -( d*log(u) + (1-d)*log(1-u) ),  u = fused vacuity clamped to [eps, 1-eps]

    Kept verbatim so the shared-ignorance failure can be demonstrated rather than asserted: when every
    branch is vacuous each p_i is ~uniform, so cos = 1, d = 0, and L = -log(1-u), whose gradient
    LOWERS u. That is Reviewer 3's objection, reproduced.
    """
    if len(branch_evidences) < 2:
        return fused_vacuity.new_zeros(())
    probs = []
    for ev in branch_evidences:
        alpha = ev + 1.0
        probs.append(alpha / alpha.sum(dim=1, keepdim=True))
    B = probs[0].shape[0]
    num = probs[0].new_zeros(B)
    den = probs[0].new_zeros(B)
    for i in range(len(probs)):
        for j in range(i + 1, len(probs)):
            num = num + (1.0 - torch.nn.functional.cosine_similarity(probs[i], probs[j], dim=1))
            den = den + 1.0
    d = (num / den.clamp_min(eps)).clamp(0, 1).detach()
    u = fused_vacuity.squeeze().clamp(eps, 1.0 - eps)
    return (-(d * torch.log(u) + (1 - d) * torch.log(1 - u))).mean()

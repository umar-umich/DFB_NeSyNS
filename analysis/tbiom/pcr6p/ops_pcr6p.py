#!/usr/bin/env python3
"""Direct multi-source PCR6 and PCR6+ on the binary frame, from the primary source.

Primary source, obtained and read for this implementation (not reconstructed from memory):

    T. Dezert, J. Dezert, F. Smarandache, "Improvement of Proportional Conflict Redistribution Rules of
    Combination of Basic Belief Assignments", Journal of Advances in Information Fusion, Vol. 16, No. 1,
    June 2021, pp. 48-73.
    PDF: https://isif.org/files/isif/2024-01/last.pdf   (sha256 d87ae493e7c409d1b194ed701b6fe746, 2026-09-23)
    Landing page: https://digitalrepository.unm.edu/math_fsp/598/

Equations used, with the paper's numbering.

(15) PCR6 for S > 2 sources. m_PCR6(empty) = 0 and for A != empty,

        m_PCR6(A) = m_Conj(A) + SUM_j [ A in X_j and pi_j(empty)>0 ]
                        ( SUM_{i : X_ji = A} m_i(X_ji) ) * pi_j(empty)
                        / ( SUM_{X in X_j} SUM_{i : X_ji = X} m_i(X_ji) )

    where X_j = (X_j1, ..., X_jS) is a tuple of focal elements, one per source, pi_j = PROD_i m_i(X_ji), and
    pi_j(empty) denotes a product whose intersection is empty. The denominator sums each distinct focal element of
    the tuple once, over the sources holding it, so it equals SUM_i m_i(X_ji).

(22) binary containing indicator  delta_j(X', X) = 1 if X' is a subset of X, else 0.

(23)/(24) binary keeping index kappa_j(X_ji): 1 unless the element contains every other element implied in that
    partial conflict, in which case 0. Remark 1 of the paper: kappa_j(Theta) = 0 whenever Theta is in X_j, because
    Theta includes every other focal element and has the highest cardinality.

(26) PCR6+ for S > 2 sources: equation (15) with kappa_j(A) multiplying the numerator and kappa_j(X) weighting each
    term of the denominator. Remark 4: (26) reduces to (15) when every keeping index is 1. Remark 3: for S = 2 all
    four rules PCR5, PCR5+, PCR6, PCR6+ coincide.

WHAT THIS MEANS ON OUR BINARY FRAME. Theta = {fake, genuine}; the focal elements a view can carry are {fake},
{genuine} and Theta. A tuple is conflicting exactly when it contains at least one {fake} and at least one {genuine},
because Theta never empties an intersection. In such a tuple the distinct focal elements are {fake} and {genuine},
plus Theta if any source is (partly) ignorant. By (23) neither singleton contains the other, so both keep index 1;
by Remark 1, Theta keeps index 0 whenever it appears. **PCR6+ on a binary frame is therefore exactly PCR6 with the
conflicting mass never redistributed to Theta**, which is the neutrality property the paper set out to restore. The
keeping indexes are computed here from the definition rather than hard-coded, and the audit checks the consequence.

Representation and the reliability discount are shared with the grouped study: opinions are (b1, b0, u) with
b1 + b0 + u = 1, and reliability is evidence scaling e' = r e, which agrees with classical belief-mass discounting
only at r = 0 and r = 1.
"""
from __future__ import annotations
import itertools
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "grouped"), str(HERE.parent), str(HERE.parent / "temporal")]
import ops as G  # noqa: E402  (opinion_from_pu, evidence_of, opinion_of_evidence, discount, p_of, VIEWS)

EPS = 1e-12
FAKE, GEN, THETA = 0b01, 0b10, 0b11          # focal elements as bit sets over {fake, genuine}
ELEMS = (FAKE, GEN, THETA)
NAME = {FAKE: "fake", GEN: "genuine", THETA: "Theta"}


def card(x):
    return bin(x).count("1")


def subset(a, b):
    """delta_j(a, b) of equation (22): 1 when a is a subset of b."""
    return (a & b) == a


def keeping_indexes(tup):
    """kappa_j for the distinct focal elements of one tuple, from equations (22)-(24).

    An element is discarded (index 0) when it contains every other distinct element implied in the partial conflict,
    i.e. when the product of the containing indicators over the qualifying ordered pairs is 1."""
    Xj = sorted(set(tup))
    kap = {}
    for A in Xj:
        prod = 1
        for lp, l in itertools.permutations(Xj, 2):
            if card(A) <= card(l) and card(lp) <= card(l):
                prod *= subset(lp, l)
        kap[A] = 1 - prod
    return kap


def tuples(S):
    """Every S-tuple of focal elements, with its intersection, conflict flag and keeping indexes (all mass-free)."""
    out = []
    for tup in itertools.product(ELEMS, repeat=S):
        inter = THETA
        for x in tup:
            inter &= x
        out.append({"tup": tup, "inter": inter, "conflict": inter == 0,
                    "kappa": keeping_indexes(tup) if inter == 0 else None})
    return out


_CACHE: dict[int, list] = {}


def spec(S):
    if S not in _CACHE:
        _CACHE[S] = tuples(S)
    return _CACHE[S]


def fuse(masses, plus: bool, attribute=False):
    """masses: (n, S, 3) array of (b1, b0, u) per source. Returns the fused (n, 3) opinion.

    `plus` selects PCR6+ (equation 26) over PCR6 (equation 15). With `attribute`, also returns the conflict mass
    broken down by the source pair that produced it."""
    n, S, _ = masses.shape
    m = {FAKE: masses[:, :, 0], GEN: masses[:, :, 1], THETA: masses[:, :, 2]}   # each (n, S)
    out = {FAKE: np.zeros(n), GEN: np.zeros(n), THETA: np.zeros(n)}
    K = np.zeros(n)
    att = {"K_TS": np.zeros(n), "K_AO": np.zeros(n), "K_all": np.zeros(n)}
    for t in spec(S):
        tup, inter = t["tup"], t["inter"]
        pi = np.ones(n)
        for i, x in enumerate(tup):
            pi = pi * m[x][:, i]
        if not t["conflict"]:
            out[inter] += pi
            continue
        K += pi
        if attribute:
            att[_bucket(tup)] += pi
        kap = t["kappa"] if plus else {x: 1 for x in set(tup)}
        # per distinct focal element, the mass its holders contribute
        hold = {X: sum(m[X][:, i] for i, x in enumerate(tup) if x == X) for X in set(tup)}
        den = sum(kap[X] * hold[X] for X in hold)
        ok = den > EPS                                            # zero denominator: the fraction is discarded
        share = np.where(ok, pi / np.where(ok, den, 1.0), 0.0)
        for A in hold:
            if kap[A]:
                out[A] += kap[A] * hold[A] * share
    f = np.stack([out[FAKE], out[GEN], out[THETA]], 1)
    return (f, {**{k: v for k, v in att.items()}, "K": K}) if attribute else f


def _bucket(tup):
    """Which source pair produced this conflicting tuple. Views are ordered (T, APP, OM) by `G.VIEWS`."""
    ts = any(tup[0] != tup[k] and (tup[0] & tup[k]) == 0 for k in (1, 2))
    ao = len(tup) > 2 and (tup[1] & tup[2]) == 0
    return "K_all" if (ts and ao) else ("K_TS" if ts else "K_AO")


def fuse_simple(masses, rule):
    """Direct S-source conjunctive, Dempster and Yager on the same tuple enumeration.

    Dempster is associative so a pairwise fold would agree, but Yager is not: its conflict must be collected over
    all S sources at once and assigned to Theta in one step, which is what the S-source rule does."""
    n, S, _ = masses.shape
    m = {FAKE: masses[:, :, 0], GEN: masses[:, :, 1], THETA: masses[:, :, 2]}
    out = {FAKE: np.zeros(n), GEN: np.zeros(n), THETA: np.zeros(n)}
    K = np.zeros(n)
    for t in spec(S):
        pi = np.ones(n)
        for i, x in enumerate(t["tup"]):
            pi = pi * m[x][:, i]
        if t["conflict"]:
            K += pi
        else:
            out[t["inter"]] += pi
    f = np.stack([out[FAKE], out[GEN], out[THETA]], 1)
    if rule == "conjunctive":
        return f, K
    if rule == "yager":                                  # all conflict to ignorance
        f[:, 2] += K
        return f
    if rule == "dempster":                               # conflict normalised away
        d = np.clip(1 - K, EPS, None)
        return f / d[:, None]
    raise ValueError(rule)


def masses_of(t, views, w=None):
    """(n, S, 3) discounted opinions for the named views, in the given order."""
    out = []
    for i, v in enumerate(views):
        o, _bad = G.opinion_from_pu(t[f"p_{v}"].to_numpy(), t[f"u_{v}"].to_numpy())
        out.append(G.discount(o, None if w is None else w[i]))
    return np.stack(out, 1)


def p_of(f):
    return np.clip(f[:, 0] + f[:, 2] / 2, 0.0, 1.0)

#!/usr/bin/env python3
"""Exact operator definitions for the grouped-fusion study (§5 of `tbiom/docs/grouped-fusion-brief.md`).

Binary frame {fake, genuine}, uniform base rate, W = 2. An opinion is carried as (b1, b0, u) with b1 + b0 + u = 1,
b1 the belief mass on `fake`. Evidence and opinion are related by

    alpha_k = e_k + 1,  S = sum(alpha) = 2 + e_1 + e_0,  b_k = e_k / S,  u = 2 / S,  p_fake = b1 + u/2,

so p is the Dirichlet mean, and the admissible region for a reconstruction from (p, u) is u/2 <= p <= 1 - u/2.
Reliability enters as evidence scaling e'_k = r * e_k, which is the discount this project already deploys; it is NOT
belief-mass (Shafer) discounting and this module never applies the latter.

Every rule below is written from its original definition and is exercised by `audit.py`. Two numerical policies are
fixed here and reported rather than hidden:

  * total conflict. Dempster and PCR6 divide by 1 - K. When 1 - K < TOTAL_CONFLICT_EPS the pair is treated as fully
    conflicting and the vacuous opinion (0, 0, 1) is returned. `combine` counts how often this fires.
  * dogmatic output. u_F = 0 is kept as the limiting dogmatic opinion; no uncertainty floor is introduced. Converting
    such an opinion back to evidence is undefined and is reported as such rather than clipped.

Missingness: a view is absent when its opinion is NaN. An absent view is dropped, never replaced by a vacuous one --
absence and an observed uncertain view are different. If a whole family is absent the other family passes through; if
all are absent the result is vacuous.
"""
from __future__ import annotations
import numpy as np

W_FRAME, EPS, TOTAL_CONFLICT_EPS = 2.0, 1e-12, 1e-9
VIEWS = ("T", "APP", "OM")


# ------------------------------------------------------------------ representation
def opinion_from_pu(p, u):
    """(p_fake, vacuity) -> (b1, b0, u). Returns the mask of rows that violate u/2 <= p <= 1 - u/2."""
    p, u = np.asarray(p, float), np.clip(np.asarray(u, float), 0.0, 1.0)
    bad = (p < u / 2 - 1e-9) | (p > 1 - u / 2 + 1e-9)
    b1, b0 = p - u / 2, 1 - p - u / 2
    return np.stack([b1, b0, u], -1), bad


def evidence_of(o):
    """(b1, b0, u) -> (e_1, e_0). Undefined where u = 0; those rows come back as inf and are reported, not clipped."""
    u = o[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.stack([W_FRAME * o[..., 0] / u, W_FRAME * o[..., 1] / u], -1)


def opinion_of_evidence(e):
    S = e.sum(-1) + W_FRAME
    return np.stack([e[..., 0] / S, e[..., 1] / S, W_FRAME / S], -1)


def discount(o, r):
    """Evidence scaling e' = r e, applied in the evidence domain, then back to an opinion."""
    if r is None or r == 1.0:
        return o
    return opinion_of_evidence(evidence_of(o) * r)


def p_of(o):
    return np.clip(o[..., 0] + o[..., 2] / 2, 0.0, 1.0)


# ------------------------------------------------------------------ two-source rules (§5)
def dempster2(a, b, stats=None):
    x1, x0, uS = a[..., 0], a[..., 1], a[..., 2]
    y1, y0, uT = b[..., 0], b[..., 1], b[..., 2]
    K = x1 * y0 + x0 * y1
    d = 1 - K
    tot = d < TOTAL_CONFLICT_EPS
    if stats is not None:
        stats["total_conflict"] = stats.get("total_conflict", 0) + int(tot.sum())
        stats["n"] = stats.get("n", 0) + int(np.size(tot))
    ds = np.where(tot, 1.0, d)
    out = np.stack([(x1 * y1 + x1 * uT + uS * y1) / ds,
                    (x0 * y0 + x0 * uT + uS * y0) / ds,
                    (uS * uT) / ds], -1)
    return np.where(tot[..., None], np.array([0.0, 0.0, 1.0]), out)


def yager2(a, b, stats=None):
    """Conflict returned to vacuity; no division, so no total-conflict case."""
    x1, x0, uS = a[..., 0], a[..., 1], a[..., 2]
    y1, y0, uT = b[..., 0], b[..., 1], b[..., 2]
    K = x1 * y0 + x0 * y1
    return np.stack([x1 * y1 + x1 * uT + uS * y1,
                     x0 * y0 + x0 * uT + uS * y0,
                     uS * uT + K], -1)


def pcr6_2(a, b, stats=None):
    """Two-source PCR6, which coincides with PCR5. Zero-denominator redistribution terms contribute zero."""
    x1, x0, uS = a[..., 0], a[..., 1], a[..., 2]
    y1, y0, uT = b[..., 0], b[..., 1], b[..., 2]
    safe = lambda num, den: np.where(den > EPS, num / np.where(den > EPS, den, 1.0), 0.0)
    b1 = x1 * y1 + x1 * uT + uS * y1 + safe(x1 * x1 * y0, x1 + y0) + safe(y1 * y1 * x0, y1 + x0)
    b0 = x0 * y0 + x0 * uT + uS * y0 + safe(x0 * x0 * y1, x0 + y1) + safe(y0 * y0 * x1, y0 + x1)
    return np.stack([b1, b0, uS * uT], -1)


# ------------------------------------------------------------------ cautious (Denoeux), normalised binary variant
def canonical_weights(o):
    """w_k = u / (b_k + u): the weight of the simple support function toward class k in Denoeux's decomposition."""
    u = np.clip(o[..., 2], EPS, 1.0)
    return u / np.clip(o[..., 0] + u, EPS, None), u / np.clip(o[..., 1] + u, EPS, None)


def from_weights(w1, w0):
    """Rebuild the separable belief function from its canonical weights and renormalise the empty mass away."""
    b1, b0, u = w0 * (1 - w1), w1 * (1 - w0), w1 * w0
    empty = (1 - w1) * (1 - w0)
    s = np.clip(b1 + b0 + u + empty, EPS, None)
    d = np.clip(1 - empty / s, EPS, None)
    return np.stack([b1 / s / d, b0 / s / d, u / s / d], -1)


def cautious_n(*os):
    """Normalised cautious rule: minimum of the canonical weights, which is idempotent."""
    ws = [canonical_weights(o) for o in os]
    return from_weights(np.min([w[0] for w in ws], 0), np.min([w[1] for w in ws], 0))


def evidence_average(*os):
    """Average the discounted EVIDENCE and add the Dirichlet prior once -- not an average of probabilities."""
    return opinion_of_evidence(np.mean([evidence_of(o) for o in os], 0))


OUTER = {"dempster": dempster2, "pcr6": pcr6_2, "yager": yager2}


# ------------------------------------------------------------------ the fixed matrix (§6)
ARMS = {
    # id                inner pair            inner rule    outer rule    passthrough view
    "grouped-R":      (("APP", "OM"), "cautious", "dempster", "T"),
    "grouped-TA-R":   (("T", "APP"), "cautious", "dempster", "OM"),
    "grouped-TO-R":   (("T", "OM"), "cautious", "dempster", "APP"),
    "GCP-AO-R":       (("APP", "OM"), "cautious", "pcr6", "T"),
    "GCY-AO-R":       (("APP", "OM"), "cautious", "yager", "T"),
    "GAP-AO-R":       (("APP", "OM"), "average", "dempster", "T"),
    "GAP-PCR6-AO-R":  (("APP", "OM"), "average", "pcr6", "T"),
}
FLAT = {"Dempster-R": "dempster", "PCR6-R": "pcr6", "Yager-R": "yager"}
SUBSETS = {"T only": ("T",), "T+APP": ("T", "APP"), "T+OM": ("T", "OM")}


def fuse(views, arm, stats=None):
    """views: {name: (n, 3) opinion array, already discounted}. Returns the fused (n, 3) opinion."""
    if arm in ARMS:
        pair, inner, outer, rest = ARMS[arm]
        have = [v for v in pair if v in views]
        if not have:
            return views[rest] if rest in views else vacuous(views)
        S = (cautious_n if inner == "cautious" else evidence_average)(*[views[v] for v in have]) \
            if len(have) == 2 else views[have[0]]
        return OUTER[outer](S, views[rest], stats) if rest in views else S
    if arm in FLAT:
        f = OUTER[FLAT[arm]]
        present = [views[v] for v in VIEWS if v in views]
        if not present:
            return vacuous(views)
        out = present[0]
        for o in present[1:]:
            out = f(out, o, stats)
        return out
    if arm == "CAUT-R":
        present = [views[v] for v in VIEWS if v in views]
        return cautious_n(*present) if present else vacuous(views)
    if arm == "AVG-R":
        present = [views[v] for v in VIEWS if v in views]
        return evidence_average(*present) if present else vacuous(views)
    if arm in SUBSETS:
        present = [views[v] for v in SUBSETS[arm] if v in views]
        if not present:
            return vacuous(views)
        out = present[0]
        for o in present[1:]:
            out = dempster2(out, o, stats)
        return out
    raise ValueError(arm)


def vacuous(views):
    n = len(next(iter(views.values())))
    return np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))


ALL_ARMS = ["grouped-R", "grouped-TA-R", "grouped-TO-R", "Dempster-R", "AVG-R", "CAUT-R", "PCR6-R",
            "GCP-AO-R", "GCY-AO-R", "GAP-AO-R", "T only", "T+APP", "T+OM"]

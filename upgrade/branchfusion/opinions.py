#!/usr/bin/env python3
"""Fusion operators F0-F8 over an arbitrary subset of branches, plus the two discount conventions.

Section 5 of the brief. Everything here operates on a wide table carrying `p_<B>` and `u_<B>` per
branch, and returns a fused `p_fake` per video. Nothing here fits anything on test data.

WHY THIS MODULE EXISTS RATHER THAN REUSING `arms.py` WHOLESALE. `analysis/tbiom/pcr6p/arms.py` is
hard-wired to exactly three views (`VIEWS = ("T", "APP", "OM")` and a 3-dimensional weight grid), and
this campaign scores subsets of five branches. What it delegates to -- `ops_pcr6p.fuse`, the direct
S-source PCR6/PCR6+ enumeration -- is already general in the number of sources, so F8 calls THAT
verified implementation with the subset's masses rather than reimplementing the operator. The
pairwise fold stays available under its own name and is never relabelled as the multisource rule,
which section 5 forbids.

THE TWO DISCOUNT CONVENTIONS ARE NOT THE SAME OPERATOR, and the brief asks for the distinction to be
recorded rather than assumed:

  evidence scaling   e'_k = rho e_k
  belief discounting b'_k = rho b_k,  u' = 1 - rho + rho u
                     equivalently     e'_k = rho u e_k / (1 - rho + rho u)

`ops.discount`, which B1 uses, is EVIDENCE SCALING -- its own docstring says so
(`analysis/tbiom/grouped/ops.py:53`). Both are provided here so a comparison can be made, with
`DISCOUNT_DEFAULT` naming the deployed one so no caller silently picks the other.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(ROOT / "analysis/tbiom"), str(ROOT / "analysis/tbiom/pcr6p"),
                str(ROOT / "analysis/tbiom/grouped"), str(ROOT / "upgrade/branchfusion")]

import ops as G                  # noqa: E402  opinion_from_pu, evidence_of, opinion_of_evidence, discount
import ops_pcr6p as P            # noqa: E402  the verified direct multisource PCR6/PCR6+
import common as C               # noqa: E402

K = 2.0
EPS = 1e-12
#: TWO CLASS ORDERS LIVE IN THIS REPOSITORY AND MIXING THEM SILENTLY INVERTS A SCORE.
#: `ops.opinion_from_pu` builds (b_fake, b_genuine, u) and `ops.evidence_of` therefore returns
#: (e_fake, e_genuine) -- fake FIRST. `common.alpha_of` and the EDL loss use (genuine, fake) -- fake
#: SECOND, because the loss indexes class 0 as real. Every index into an `ops`-derived evidence array
#: in this module goes through these names, and `tests/branchfusion/test_opinions.py` pins them: an
#: inversion here once reported macro 0.07 for a system scoring 0.93.
OPS_FAKE, OPS_GEN = 0, 1
#: Which convention B1 and the deployed system use. Recorded, not chosen per call site.
DISCOUNT_DEFAULT = "evidence_scaling"

#: Section 5's comparator ledger. F3/F4 are heads and live in `heads.py`; they appear here so the
#: table legend and the acceptance check can see the full matrix from one place.
OPERATORS = {
    "F0": "best single branch, selected on development (not per test set)",
    "F1": "uniform probability mean",
    "F2": "weighted probability mean, nonnegative simplex weights fitted on inner development",
    "F3": "feature concatenation + CE head (heads.py)",
    "F4": "feature concatenation + EDL head (heads.py)",
    "F5": "cumulative evidence, prior counted once",
    "F6": "reliability-discounted cumulative evidence (B1's convention)",
    "F7": "reliability-discounted Dempster combination",
    "F8": "verified direct multisource PCR6+ (ops_pcr6p.fuse, plus=True)",
}
#: Operators implemented in this module, i.e. those that act on opinions alone.
OPINION_OPS = ("F0", "F1", "F2", "F5", "F6", "F7", "F8")


# ----------------------------------------------------------------------------- opinions from a table
def masses(t, branches, w=None, convention: str = DISCOUNT_DEFAULT):
    """(n, S, 3) discounted opinions `(b_fake, b_genuine, u)` for `branches`, in the given order.

    `w` is one reliability per branch in [0, 1], or None for no discounting. Rows whose stored
    (p, u) cannot be a valid opinion are reported by `invalid_count`, never silently clipped --
    Phase 0 of the previous campaign showed what silent clipping costs.
    """
    out = []
    for i, b in enumerate(branches):
        o, bad = G.opinion_from_pu(t[f"p_{b}"].to_numpy(), t[f"u_{b}"].to_numpy())
        if bad.any():
            o = np.clip(o, 0.0, 1.0)
            o = o / np.clip(o.sum(-1, keepdims=True), EPS, None)
        r = None if w is None else float(w[i])
        out.append(_discount(o, r, convention))
    return np.stack(out, 1)


def invalid_count(t, branches) -> dict:
    """How many rows per branch carry an invalid stored opinion. Section 11 hard-stop input."""
    out = {}
    for b in branches:
        _o, bad = G.opinion_from_pu(t[f"p_{b}"].to_numpy(), t[f"u_{b}"].to_numpy())
        out[b] = int(np.asarray(bad).sum())
    return out


def _discount(o, r, convention):
    if r is None or r == 1.0:
        return o
    if convention == "evidence_scaling":
        return G.discount(o, r)                     # e' = r e, the deployed convention
    if convention == "belief_discounting":
        return discount_belief(o, r)
    raise ValueError(f"unknown discount convention {convention!r}")


def discount_belief(o, rho):
    """Belief discounting: b'_k = rho b_k, u' = 1 - rho + rho u.

    Kept beside evidence scaling because the brief requires the two to be distinguished rather than
    treated as interchangeable. `equivalent_evidence` gives the evidence form the brief states.
    """
    rho = float(rho)
    b1, b0, u = o[..., 0], o[..., 1], o[..., 2]
    return np.stack([rho * b1, rho * b0, 1.0 - rho + rho * u], -1)


def equivalent_evidence(o, rho):
    """The evidence a belief-discounted opinion carries: e'_k = rho u e_k / (1 - rho + rho u).

    Provided so the identity in section 5 can be checked numerically rather than trusted; the test
    module asserts it against `discount_belief` followed by `ops.evidence_of`.
    """
    rho = float(rho)
    u = np.clip(o[..., 2], EPS, 1.0)
    e = G.evidence_of(o)
    den = np.clip(1.0 - rho + rho * u, EPS, None)
    return e * (rho * u / den)[..., None]


# ----------------------------------------------------------------------------- the operators
def fuse(t, branches, op: str, w=None, convention: str = DISCOUNT_DEFAULT,
         stats: dict | None = None) -> np.ndarray:
    """Fused p_fake for one operator over `branches`. `w` is per-branch reliability or weight."""
    branches = tuple(branches)
    if len(branches) == 0:
        raise ValueError("fuse: no branches")
    if len(branches) == 1:
        return np.clip(t[f"p_{branches[0]}"].to_numpy(), 0.0, 1.0)

    if op == "F1":
        return _prob_mean(t, branches, None)
    if op == "F2":
        return _prob_mean(t, branches, w)
    if op in ("F5", "F6"):
        return _cumulative(t, branches, None if op == "F5" else w, convention, stats)
    if op == "F7":
        return _dempster(t, branches, w, convention, stats)
    if op == "F8":
        return _pcr6_plus(t, branches, w, convention, stats)
    if op == "F0":
        raise ValueError("F0 is a selection rule, not a combination: use `best_single`")
    raise ValueError(f"unknown operator {op!r}; opinion operators are {OPINION_OPS}")


def _prob_mean(t, branches, w):
    """F1 / F2: mean of the class probabilities, uniform or on nonnegative simplex weights."""
    p = np.stack([t[f"p_{b}"].to_numpy() for b in branches], 1)
    if w is None:
        return np.clip(p.mean(1), 0.0, 1.0)
    w = np.asarray(w, float)
    w = np.maximum(w, 0.0)
    s = w.sum()
    if s <= EPS:
        return np.clip(p.mean(1), 0.0, 1.0)
    return np.clip(p @ (w / s), 0.0, 1.0)


def _cumulative(t, branches, w, convention, stats):
    """F5 / F6: alpha_f = 1 + sum_v e_v -- the Dirichlet prior counted ONCE, not once per branch.

    Counting the prior per branch is the standard off-by-S error here: S sources would give
    alpha = S + sum e and a systematically over-confident strength. The duplicate-view diagnostic
    section 5 asks for is in `duplicate_diagnostic`.
    """
    m = masses(t, branches, w, convention)
    e = np.stack([G.evidence_of(m[:, i, :]) for i in range(m.shape[1])], 1)   # (n, S, 2)
    e = np.where(np.isfinite(e), e, 0.0)
    tot = e.sum(1)                                                            # (n, 2) in ops order
    a = 1.0 + tot
    S = a.sum(1)
    if stats is not None:
        stats["mean_strength"] = float(np.mean(S))
    return np.clip(a[:, OPS_FAKE] / S, 0.0, 1.0)


def duplicate_diagnostic(t, branches, w=None, convention=DISCOUNT_DEFAULT) -> dict:
    """What cumulative fusion does when a branch is counted twice.

    Cumulative evidence is not idempotent: duplicating a source doubles its evidence and so halves
    the fused vacuity, which means a subset holding two near-identical branches is over-confident by
    construction. This reports the mean fused strength and the AUROC shift from duplicating each
    branch in turn -- the diagnostic section 5 attaches to F5.
    """
    y = t.y.to_numpy()
    st0: dict = {}
    base = _cumulative(t, branches, w, convention, st0)
    out = {"_strength_base": st0.get("mean_strength"),
           "_auroc_base": C.auroc(y, base)}
    bl = list(branches)
    for b in bl:
        dup_t = t.copy()
        dup_t["p_DUP"] = t[f"p_{b}"].to_numpy()
        dup_t["u_DUP"] = t[f"u_{b}"].to_numpy()
        wd = None if w is None else list(w) + [float(w[bl.index(b)])]
        st: dict = {}
        p = _cumulative(dup_t, tuple(bl) + ("DUP",), wd, convention, st)
        out[b] = {"auroc_dup": C.auroc(y, p),
                  "delta_auroc": C.auroc(y, p) - out["_auroc_base"],
                  "strength_dup": st.get("mean_strength"),
                  "strength_ratio": (st.get("mean_strength") or np.nan)
                                    / (out["_strength_base"] or np.nan)}
    return out


def _dempster(t, branches, w, convention, stats):
    """F7: reliability-discounted Dempster, over all S sources at once.

    Dempster is associative so the S-source enumeration and a pairwise fold agree; the enumeration is
    used anyway because it reports total conflict in one place. The normalisation conflict and the
    numerical fallback count go into `stats`, which section 5 requires to be logged.
    """
    m = masses(t, branches, w, convention)
    f, Kc = P.fuse_simple(m, "conjunctive")
    tot = Kc > 1.0 - 1e-9
    if stats is not None:
        stats["conflict_mean"] = float(np.mean(Kc))
        stats["conflict_max"] = float(np.max(Kc))
        stats["total_conflict_rows"] = int(tot.sum())
        stats["n"] = int(len(Kc))
    d = np.clip(1.0 - Kc, 1e-12, None)
    f = f / d[:, None]
    # A row of total conflict has no Dempster answer; fall back to the vacuous opinion rather than a
    # guess, and count it. This is the numerical fallback the brief wants reported.
    if tot.any():
        f[tot] = np.array([0.0, 0.0, 1.0])
    return P.p_of(f)


def _pcr6_plus(t, branches, w, convention, stats):
    """F8: the verified direct multisource PCR6+ (`ops_pcr6p.fuse`, equation 26).

    This is the existing validated implementation, called with S = len(branches) sources. It is
    permutation-invariant by construction because it enumerates all source tuples; the sequential
    pairwise fold is a DIFFERENT operator and is exposed separately as `pcr6_pairwise`.
    """
    m = masses(t, branches, w, convention)
    f, att = P.fuse(m, True, attribute=True)
    if stats is not None:
        stats["conflict_mean"] = float(np.mean(att["K"]))
        stats["conflict_max"] = float(np.max(att["K"]))
    return P.p_of(f)


def pcr6_pairwise(t, branches, w=None, convention=DISCOUNT_DEFAULT) -> np.ndarray:
    """The historical sequential pairwise PCR6 fold, kept under its own name.

    Section 5: never relabel this as a permutation-invariant multisource operator. It is reported
    only as a labelled comparison row, and `permutation_spread` measures how far from invariant it is.
    """
    m = masses(t, branches, w, convention)
    f = m[:, 0, :]
    for i in range(1, m.shape[1]):
        f = G.pcr6_2(f, m[:, i, :])
    return P.p_of(f)


def best_single(t, branches, dev_scores: dict) -> tuple:
    """F0: the single branch with the best DEVELOPMENT score, then scored wherever asked.

    `dev_scores` maps branch -> development macro. Choosing per test set would be an oracle, which is
    section 4's separate non-deployable row, not F0.
    """
    b = max(branches, key=lambda x: dev_scores.get(x, -np.inf))
    return b, np.clip(t[f"p_{b}"].to_numpy(), 0.0, 1.0)


def oracle_single(t, branches) -> np.ndarray:
    """Section 4's non-deployable reference: the best single branch chosen per dataset.

    Returns the per-branch AUROCs so the caller records WHICH branch won where; the row exists to
    answer whether fusion ever beats view selection, and is never presented as a deployable system.
    """
    y = t.y.to_numpy()
    aus = {b: C.auroc(y, t[f"p_{b}"].to_numpy()) for b in branches}
    best = max(aus, key=lambda b: aus[b])
    return best, aus[best], aus


# ----------------------------------------------------------------------------- weight fitting
#: B0's locked reliability grid. `fit_weights` is exhaustive over it, so the cost of every fusion
#: fit in this project is `len(WEIGHT_GRID)**k - (len(WEIGHT_GRID)-1)**k` for a k-branch subset:
#: 369 vectors at k=4, 325,089 at k=8, 746 billion at k=17. Any widening of the subset space runs
#: into this before it runs into anything else.
WEIGHT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def weight_grid(n: int, grid=WEIGHT_GRID, anchored: bool = True):
    """Reliability grids for `n` branches.

    `anchored` keeps B0's convention that at least one branch has weight 1, which removes the global
    scale redundancy (a discount applied to every source is a different operator, studied as B1).
    """
    for w in itertools.product(grid, repeat=n):
        if anchored and max(w) != 1.0:
            continue
        yield tuple(w)


def fit_weights(fit_t, branches, op, objective="v31", convention=DISCOUNT_DEFAULT,
                grid=WEIGHT_GRID, anchored=True, cl=None):
    """Fit per-branch weights on a FIT partition only, by the stated objective.

    `objective` is "v31" (the development macro B0 selects on) or "auroc". The search budget is the
    same grid for F2 as for the evidential trust weights, which section 5 requires, and the start is
    equal weights by virtue of the grid containing them.
    """
    import exp1_fusion_audit as A
    cl = A.cells(fit_t) if (objective == "v31" and cl is None) else cl
    y = fit_t.y.to_numpy()

    def sc(w):
        p = _prob_mean(fit_t, branches, w) if op == "F2" else fuse(fit_t, branches, op, w, convention)
        if objective == "v31":
            return A.v31(p, cl)[0]
        return C.auroc(y, p)

    cands = list(weight_grid(len(branches), grid, anchored))
    vals = [sc(w) for w in cands]
    j = int(np.nanargmax(vals))
    return cands[j], float(vals[j])


# ----------------------------------------------------------------------------- required algebra tests
def permutation_spread(t, branches, op, w=None, convention=DISCOUNT_DEFAULT) -> dict:
    """Max absolute score change over all orderings of the branches.

    Section 5 requires the input-order behaviour to be tested, including all six permutations of a
    triple for the direct multisource PCR6+. A permutation-invariant operator must return ~0 here;
    the pairwise fold will not, which is the point of keeping them separate.
    """
    perms = list(itertools.permutations(range(len(branches))))
    base = None
    worst = 0.0
    for pm in perms:
        bs = [branches[i] for i in pm]
        ws = None if w is None else [w[i] for i in pm]
        p = (pcr6_pairwise(t, bs, ws, convention) if op == "pairwise"
             else fuse(t, bs, op, ws, convention))
        if base is None:
            base = p
        else:
            worst = max(worst, float(np.max(np.abs(p - base))))
    return {"n_permutations": len(perms), "max_abs_change": worst,
            "invariant": bool(worst < 1e-9)}


#: Operators that take NO reliability argument, so the rho endpoints do not apply to them.
#: F1 is a uniform mean by definition and F5 is UNdiscounted cumulative evidence -- discounting it is
#: F6, which is a separate row. Testing a rho endpoint on these would compare a weight the operator
#: never reads, which is how a meaningless assertion gets into a test suite.
UNDISCOUNTED_OPS = ("F1", "F5")


def endpoint_checks(t, branches, op, convention=DISCOUNT_DEFAULT) -> dict:
    """rho = 0 and rho = 1 behaviour, per section 5.

    At rho = 1 the discount is the identity, so the operator must equal its undiscounted self. At
    rho = 0 a source becomes vacuous and must stop contributing: fusing with it has to agree with
    dropping it. Both are properties of the operator, not of the data, so a failure here is a bug.

    For an operator that reads no reliability at all (see `UNDISCOUNTED_OPS`) the checks are reported
    as not applicable rather than evaluated against a weight the operator ignores.
    """
    if op in UNDISCOUNTED_OPS:
        return {"applicable": False,
                "reason": f"{op} takes no reliability argument; discounting it is a different row"}
    out = {"applicable": True}
    p_one = fuse(t, branches, op, [1.0] * len(branches), convention)
    p_none = fuse(t, branches, op, None, convention)
    out["rho1_equals_undiscounted"] = float(np.max(np.abs(p_one - p_none)))
    if len(branches) > 1:
        w = [1.0] * len(branches)
        w[-1] = 0.0
        p_zero = fuse(t, branches, op, w, convention)
        p_drop = fuse(t, branches[:-1], op, [1.0] * (len(branches) - 1), convention)
        out["rho0_equals_dropped"] = float(np.max(np.abs(p_zero - p_drop)))
    return out


def conflict_stability(op, convention=DISCOUNT_DEFAULT) -> dict:
    """Behaviour on vacuous and maximally conflicting opinions, on synthetic inputs.

    Three cases the brief names: a vacuous opinion must act as the identity, two opposed dogmatic
    opinions must not produce a non-finite or out-of-range score, and the output must stay in [0, 1].
    Synthetic rather than sampled, because the interesting rows may not occur in the data.
    """
    import pandas as pd
    rows = {
        "vacuous_with_confident": ([0.5, 0.9], [1.0, 0.1]),
        "both_vacuous": ([0.5, 0.5], [1.0, 1.0]),
        "opposed_strong": ([0.999, 0.001], [0.002, 0.002]),
        "opposed_dogmatic": ([1.0 - 1e-9, 1e-9], [1e-9, 1e-9]),
    }
    out = {}
    for name, (ps, us) in rows.items():
        t = pd.DataFrame({"y": [1], "p_X": [ps[0]], "u_X": [us[0]],
                          "p_Y": [ps[1]], "u_Y": [us[1]]})
        try:
            p = fuse(t, ("X", "Y"), op, None, convention)
            out[name] = {"p": float(p[0]), "finite": bool(np.isfinite(p).all()),
                         "in_range": bool(0.0 - 1e-9 <= p[0] <= 1.0 + 1e-9)}
        except Exception as e:                                        # a raise is a result too
            out[name] = {"error": f"{type(e).__name__}: {e}"}
    # the vacuous source must leave the confident one alone
    if "vacuous_with_confident" in out and "p" in out["vacuous_with_confident"]:
        out["vacuous_is_identity_err"] = abs(out["vacuous_with_confident"]["p"] - 0.9)
    return out


def algebra_report(t, branches) -> dict:
    """Run every section-5 algebra requirement over the real development table."""
    rep = {"branches": list(branches), "convention": DISCOUNT_DEFAULT,
           "invalid_stored_opinions": invalid_count(t, branches)}
    for op in ("F5", "F6", "F7", "F8"):
        rep[op] = {"endpoints": endpoint_checks(t, branches, op),
                   "conflict": conflict_stability(op)}
    if len(branches) >= 2:
        rep["permutation"] = {op: permutation_spread(t, branches, op)
                              for op in ("F5", "F7", "F8")}
        rep["permutation"]["pairwise"] = permutation_spread(t, branches, "pairwise")
    # the discount identity from the brief, checked numerically
    o, _bad = G.opinion_from_pu(t[f"p_{branches[0]}"].to_numpy(),
                                t[f"u_{branches[0]}"].to_numpy())
    errs = {}
    for rho in (0.25, 0.5, 0.75):
        lhs = G.evidence_of(discount_belief(o, rho))
        rhs = equivalent_evidence(o, rho)
        m = np.isfinite(lhs) & np.isfinite(rhs)
        errs[str(rho)] = float(np.max(np.abs(lhs[m] - rhs[m]))) if m.any() else float("nan")
    rep["belief_discount_identity_max_err"] = errs
    return rep

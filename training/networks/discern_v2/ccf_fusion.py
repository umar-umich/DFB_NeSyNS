"""Multi-source Consensus & Compromise Fusion (CCF) — brief Stage 6.

Implements Definition 5 of

    R. W. van der Heijden, H. Kopp, F. Kargl, "Multi-Source Fusion Operations in Subjective
    Logic", FUSION 2018, DOI 10.23919/ICIF.2018.8455615 (arXiv:1805.01388).

The brief requires the **genuine multi-source operator, not sequential pairwise fusion**, and
that is not a style preference: CCF is non-associative, so `ccf(ccf(a, b), c)` is a different
opinion from `ccf(a, b, c)`. The paper's whole contribution is that the multi-source form has to
be defined rather than chained, and the consensus step makes the reason concrete — `b_cons(x) =
min over ALL sources` is not recoverable from a sequence of pairwise minima applied to already
normalised intermediate results.

The operator, verbatim from the paper
-------------------------------------
Given actors A over domain X, each with belief mass `b_A` and uncertainty `u_A`:

  (6)  consensus     b_cons(x) = min_A b_A(x)          b_res_A(x) = b_A(x) - b_cons(x)
                     b_cons    = sum_{x in R(X)} b_cons(x)
  (7)  compromise    b_comp(x) = sum_A b_res_A(x) * prod_{A' != A} u_A'                  [term 1]
                               + sum_{y_1..y_n : cap y_i = x} prod_i b_res_i(y_i) *
                                                              a(y_i | y_j, j != i)       [term 2]
                               + sum_{y : cup y_i = x, cap y_i != 0}
                                     (1 - prod_i a(y_i | y_j)) prod_i b_res_i(y_i)       [term 3]
                               + sum_{y : cup y_i = x, cap y_i = 0} prod_i b_res_i(y_i)  [term 4]
  (8)  u_pre = prod_A u_A
  (9)  b_comp = sum_{x in P(X)} b_comp(x)
  (10) eta = (1 - b_cons - u_pre) / b_comp
  (11) u_fused = u_pre + eta * b_comp(X),    then b_comp(X) := 0
  (12) b_fused(x) = b_cons(x) + eta * b_comp(x)

The binary specialisation used here
-----------------------------------
Our domain is X = {Real, Fake} and every input opinion carries belief on singletons only (no
composite mass), which collapses (7) considerably. Working the sums through:

* **Singleton x.** `cap y_i = x` over singletons forces every `y_i = x`, and then every relative
  base rate `a(x | x) = a(x)/a(x) = 1`, so term 2 is `prod_A b_res_A(x)` and term 3 vanishes
  (its factor is `1 - 1`). Term 4 needs `cup y_i = x` with empty intersection, impossible when
  all `y_i` are the same singleton. So

      b_comp(x) = sum_A b_res_A(x) * prod_{A' != A} u_A'  +  prod_A b_res_A(x)

* **The composite X.** Terms 1 and 2 need composite input mass, which is zero. Term 3 needs
  `cup y_i = X` with non-empty intersection, impossible for two disjoint singletons. Only term 4
  survives: every *mixed* assignment, i.e. at least one source contributing Real and at least one
  Fake.

      b_comp(X) = prod_A (b_res_A(R) + b_res_A(F)) - prod_A b_res_A(R) - prod_A b_res_A(F)

  This is the operator's whole point in this application: **belief that two specialists split
  between Real and Fake becomes vagueness, not a winner.** Dempster's rule discards that conflict
  and renormalises; CCF converts it into uncertainty, which is exactly the quantity the risk model
  downstream is built to read.

Because the relative base rates all evaluate to 1 above, `a` does not enter the fusion in the
binary singleton-only case — it enters only the probability projection `p = b + a*u`, where the
brief fixes `a = [0.5, 0.5]`. A multinomial or hyper-opinion extension would need the general (7).

Validated against the paper, not against reasoning
--------------------------------------------------
`test_ccf_fusion.py` reproduces Table I of the paper — three sources
(b(x), b(x̄), u) = (.10,.30,.60), (.40,.20,.40), (.70,.10,.20) — and asserts the published CCF row
b(x)=0.629, b(x̄)=0.182, u=0.189, P(x)=0.723. All four match to the paper's rounding.
"""

from __future__ import annotations

import torch

from .ds_fusion import Opinion

CCF_EPS = 1e-12


def _degenerate_fallback(belief: torch.Tensor, vacuity: torch.Tensor,
                         b_cons: torch.Tensor, b_cons_total: torch.Tensor) -> Opinion:
    """The opinion to return where `b_comp` is zero and eta (10) is therefore undefined.

    `b_comp = 0` means every source assigned identical belief, so there is nothing to compromise
    over — the residues are all zero. The consensus already holds every source's belief, and the
    remaining mass `1 - b_cons` is what no source committed. Returning
    `(b_cons, 1 - b_cons_total)` therefore reproduces the common input opinion exactly, which is
    the only answer consistent with "consensus conserves the agreed weight of all inputs".

    Not a guard bolted on to avoid a division: the paper's eta is genuinely undefined here (its
    numerator `u - u^n` is positive while its denominator is zero), so the case has to be decided
    rather than clamped. Clamping the denominator would instead send eta to a huge number and
    scale zero belief by it, silently producing whatever the floating-point noise happened to be.
    """
    return Opinion(belief=b_cons, vacuity=(1.0 - b_cons_total).clamp(0.0, 1.0))


def ccf_combine(opinions: list[Opinion]) -> Opinion:
    """Fuse N binomial opinions with multi-source CCF. Genuinely N-ary, never chained pairwise.

    Every input must be a valid opinion over the same 2-class domain; this is asserted rather
    than assumed, because an unnormalised input makes eta (10) meaningless without erroring.
    """
    if not opinions:
        raise ValueError("ccf_combine needs at least one opinion")
    if len(opinions) == 1:
        return opinions[0]
    for op in opinions:
        op.assert_normalized()
    if any(op.num_classes != 2 for op in opinions):
        raise NotImplementedError(
            "ccf_combine implements the BINARY specialisation of equation (7). A domain with "
            "|X| > 2 needs the general form, including the relative base rates that cancel here.")

    beliefs = torch.stack([op.belief for op in opinions], dim=0)      # (N, B, 2)
    vacuities = torch.stack([op.vacuity for op in opinions], dim=0)   # (N, B, 1)

    # (6) consensus and residues
    b_cons = beliefs.min(dim=0).values                                # (B, 2)
    b_res = beliefs - b_cons.unsqueeze(0)                             # (N, B, 2)
    b_cons_total = b_cons.sum(dim=1, keepdim=True)                    # (B, 1)

    # (8) preliminary uncertainty: the common uncertainty all actors agree on
    u_pre = vacuities.prod(dim=0)                                     # (B, 1)

    # (7) term 1: each residue weighted by every OTHER actor's uncertainty.
    # prod_{A' != A} u_A' computed as total / u_A would divide by zero for a dogmatic source, so
    # it is built by leave-one-out multiplication instead.
    n = len(opinions)
    others_u = torch.stack(
        [torch.prod(torch.cat([vacuities[:i], vacuities[i + 1:]], dim=0), dim=0)
         for i in range(n)], dim=0)                                   # (N, B, 1)
    term1 = (b_res * others_u).sum(dim=0)                             # (B, 2)

    # (7) term 2 on singletons: all sources agreeing on the same singleton
    term2 = b_res.prod(dim=0)                                         # (B, 2)
    b_comp_singleton = term1 + term2                                  # (B, 2)

    # (7) term 4 on the composite X: every MIXED assignment across sources
    prod_all = (b_res.sum(dim=2)).prod(dim=0).unsqueeze(1)            # (B, 1)
    prod_real = b_res[:, :, 0].prod(dim=0).unsqueeze(1)               # (B, 1)
    prod_fake = b_res[:, :, 1].prod(dim=0).unsqueeze(1)               # (B, 1)
    b_comp_X = (prod_all - prod_real - prod_fake).clamp_min(0.0)      # (B, 1)

    # (9) total compromise mass
    b_comp_total = b_comp_singleton.sum(dim=1, keepdim=True) + b_comp_X

    # (10) normalization
    degenerate = b_comp_total < CCF_EPS
    eta = (1.0 - b_cons_total - u_pre) / b_comp_total.clamp_min(CCF_EPS)

    # (12) and (11)
    belief = b_cons + eta * b_comp_singleton
    vacuity = u_pre + eta * b_comp_X

    fallback = _degenerate_fallback(beliefs, vacuities, b_cons, b_cons_total)
    belief = torch.where(degenerate, fallback.belief, belief)
    vacuity = torch.where(degenerate, fallback.vacuity, vacuity)

    return Opinion(belief=belief.clamp_min(0.0),
                   vacuity=vacuity.clamp(0.0, 1.0)).assert_normalized(tol=1e-4)


def ccf_is_non_associative(a: Opinion, b: Opinion, c: Opinion, tol: float = 1e-4) -> bool:
    """Demonstrate, on real inputs, that pairwise chaining is not the multi-source operator.

    Exists so the claim in this module's docstring is checkable from data rather than taken on
    the paper's word — Stage 6 records the measured gap alongside its results.
    """
    chained = ccf_combine([ccf_combine([a, b]), c])
    proper = ccf_combine([a, b, c])
    gap = (chained.belief - proper.belief).abs().max()
    return bool(gap > tol)

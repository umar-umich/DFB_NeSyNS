# PCR6+ audit — algebraic unit tests and exact formulas

Every test states what it expects independently of the implementation. Test 6 additionally checks the direct
S = 2 fusion against the two-source PCR5/PCR6 routine written for the earlier grouped study, so the two independent
implementations must agree.

**17 of 17 tests pass.**

## 1. Primary source and the equations as implemented

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

## 2. Unit tests

| test | expected | result | detail |
|---|---|---|---|
| PCR6: output masses non-negative and sum to one | non-negative, sum to one | pass | min 1.04e-03, max |sum-1| 4.44e-16 |
| PCR6: probabilities in [0, 1] | valid probabilities | pass |  |
| PCR6+: output masses non-negative and sum to one | non-negative, sum to one | pass | min 2.63e-05, max |sum-1| 4.44e-16 |
| PCR6+: probabilities in [0, 1] | valid probabilities | pass |  |
| PCR6: permutation of the three views | identical output | pass | max dev 4.44e-16 |
| PCR6+: permutation of the three views | identical output | pass | max dev 4.44e-16 |
| PCR6 on (genuine=1), (fake=1), (Theta=1) | fake, genuine and Theta each 1/3 | pass | got [0.333333, 0.333333, 0.333333] |
| PCR6+ on the same three views | equals the two-source fusion of the first two | pass | got [0.5, 0.5, 0.0], two-source [0.5, 0.5, 0.0] |
| PCR6+: a view with r = 0 acts as if removed | equals the fusion of the other two views | pass | max dev 0.000e+00 |
| PCR6: a view with r = 0 acts as if removed | PCR6+ yes; PCR6 is expected to fail this | pass | max dev 2.789e-01  (a non-zero deviation here IS the flaw PCR6+ corrects) |
| two sources: PCR6 equals PCR6+ | identical (paper, Remark 3) | pass | max dev 0.00e+00 |
| two sources: direct rule equals the independent two-source PCR5 implementation | identical | pass | max dev 2.22e-16 |
| fully conflicting dogmatic views | finite, stated, documented | pass | PCR6 [0.666667, 0.333333, 0.0], PCR6+ [0.666667, 0.333333, 0.0] |
| very small vacuity | stable arithmetic, no NaN | pass | PCR6+ [0.553774, 0.446226, 0.0] |
| evidence to opinion and back | exact for u > 0 | pass | max dev 2.22e-16 |
| conflict attribution sums to K on every sample | K_TS + K_AO + K_all = K | pass | max dev 3.33e-16 |
| historical PCR6-R row is direct three-source PCR6 | expected NO — it is a pairwise fold | pass | max dev 2.082e-01; the historical row must be relabelled PCR6-pairwise-R |

## 3. What the correction actually changes

A view discounted to r = 0 is vacuous. Removing it should change nothing. Under **PCR6+** the maximum deviation between fusing three views and fusing the two survivors is 0.00e+00 — it is the same function. Under **PCR6** it is 2.789e-01: a discounted-away view still absorbs conflicting mass into Theta and shifts the result. That is the flaw the paper corrects, and it is the reason this study nominates PCR6+ rather than PCR6.

On our binary frame the whole correction reduces to one rule: **conflicting mass is never redistributed to Theta.** The keeping indexes are computed from equations (22)-(24) rather than assumed, and they come out as kappa(fake) = kappa(genuine) = 1, kappa(Theta) = 0, matching Remark 1 of the paper.

The neutrality example, on our frame: three views asserting genuine, fake and total ignorance.

| rule | m(fake) | m(genuine) | m(Theta) |
|---|---:|---:|---:|
| PCR6 | 0.333333 | 0.333333 | 0.333333 |
| PCR6+ | 0.500000 | 0.500000 | 0.000000 |
| two_source | 0.500000 | 0.500000 | 0.000000 |

PCR6 gives the ignorant view a third of the mass; PCR6+ ignores it and returns the two-source fusion, as the paper's theorem requires.

## 4. Fully conflicting dogmatic views, documented rather than avoided

| rule | m(fake) | m(genuine) | m(Theta) |
|---|---:|---:|---:|
| PCR6 | 0.666667 | 0.333333 | 0.000000 |
| PCR6+ | 0.666667 | 0.333333 | 0.000000 |

Two views assert fake with certainty and one asserts genuine with certainty; all vacuities are zero. Both rules return a finite, normalised result and no fraction is discarded, because every denominator is the sum of the conflicting sources' masses and is strictly positive. The zero-denominator convention of the paper ("if a denominator is zero, that fraction is discarded") is implemented and never fires on admissible inputs.

## 5. Historical labelling

The `PCR6-R` row in `EXP_FUSION_OPERATORS.md`, `EXP_DF40_OPERATORS.md` and `EXP_GROUPED_*.md` was computed by folding the two-source rule over the three views, not by the direct three-source formula (15). The two differ by up to 2.082e-01 on random admissible opinions. PCR6 is not associative, so the fold is a different operator. **Every new table in this study labels that construction `PCR6-pairwise-R`**, and the direct rule `PCR6-R`. The earlier reports are left unchanged.

## 6. Conflict attribution

Each conflicting tuple is assigned to exactly one bucket by which pair produced it: `K_TS` when the temporal view conflicts with a spatial view and the two spatial views agree, `K_AO` when the two spatial views conflict and the temporal view sides with neither in a way that adds a further conflict, and `K_all` when both kinds are present in the same tuple. Three-way products are therefore reported separately rather than split arbitrarily, and the shares sum to K exactly on every sample.

| quantity | mean on random admissible opinions |
|---|---:|
| K | 0.3217 |
| K_TS | 0.1742 |
| K_AO | 0.0676 |
| K_all | 0.0798 |

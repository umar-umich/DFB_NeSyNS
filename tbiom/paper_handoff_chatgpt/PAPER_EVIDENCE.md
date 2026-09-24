# PAPER_EVIDENCE.md -- consolidated scientific documentation for the T-BIOM manuscript

Snapshot 2026-09-23, commit `5f78c29aa124ad642800368f90b7aa578b262a48`, working tree DIRTY (4 paths).

Packaging document. Report boundaries are preserved with explicit source-path markers; no values were recomputed, reconciled or edited while assembling it.

## Selected method, fixed by the authors

| item | value |
|---|---|
| operator | **PCR6+-R** -- direct three-source PCR6+ after evidence scaling |
| operator source | Dezert, Dezert & Smarandache, JAIF 16(1) 2021, pp. 48--73, eqs. (15), (22)--(24), (26) |
| novelty claimed | the application and the measurement; **not** the operator, which is prior work |
| views | T (V-JEPA 2 ViT-L + LoRA r8/a16), APP (CLIP ViT-L/14 + LoRA r8/a16), OM (beta-VAE head, latent 32, beta_KLD 2.0, on the CLIP path) |
| configuration of record | `42`: warm+gate spatial epoch 19 + `vjepa2_lora_B_seed42` |
| spatial views | **frozen** (`freeze_views: true`) at the baseline's own values |
| trainable parameters added | 919,938 (786,432 LoRA over 48 projections + 133,506 head) |
| excluded branch | face-prior / FS-VFM student: tested, reliability fit zeroed it 5/5 folds |
| discount | evidence scaling e' = r e; NOT belief-mass discounting (agrees only at r in {0,1}) |
| threshold | EER on FF++ val, per arm, applied unchanged to every evaluation set |
| aggregation | per-frame opinions averaged per video, then fused; temporal view emits one opinion per clip |
| uncertainty outputs | fused vacuity u_f, retained conflict K with the K_TS / K_AO / K_all decomposition |

## Contents

1. [CURRENT] `tbiom/reports/PCR6P_REFERENCE_MANIFEST.md` -- Method identity, the discount definition, the four detector configurations, folds and the exposure history.
2. [CURRENT] `tbiom/reports/EXP_PCR6P_AUDIT.md` -- Exact operator equations from the primary source, 17 algebraic unit tests, and the resolution of the vacuous-source neutrality question.
3. [CURRENT] `tbiom/reports/EXP_PCR6P_DEVELOPMENT.md` -- The fixed arm matrix on development for all four configurations: detection, calibration, uncertainty scores, conflict cases, fitted weights.
4. [CURRENT] `tbiom/reports/EXP_PCR6P_EXPOSED.md` -- Ten test sets, processing conditions and all 41 DF40 entries, per arm and per configuration, with both required DF40 means.
5. [CURRENT] `tbiom/reports/PCR6P_OUTCOME.md` -- Pre-registered acceptance criteria applied; terminal outcome.
6. [CURRENT] `tbiom/reports/PCR6P_PRIOR_ART_AND_CLAIMS.md` -- Prior art with primary sources read, and the claim-to-evidence map separating established operators from our implementation choices and measured findings.
7. [CURRENT] `tbiom/reports/EXP_DF40_CROSS_MANIPULATION.md` -- Per-generator cross-manipulation against our trained DiCoME.
8. [CURRENT] `tbiom/reports/BASELINE_MATRIX.md` -- Comparator provenance and per-benchmark values, all reproduced locally.
9. [HISTORICAL] `tbiom/reports/REFERENCE_MANIFEST.md` -- SUPERSEDED as the method of record (it documents the DS-R operator). Retained because it contains the numerical comparison of evidence scaling against belief-mass discounting, which is the definition the current method still uses.
10. [HISTORICAL] `tbiom/reports/GROUPED_FUSION_OUTCOME.md` -- SUPERSEDED / closed line of work. Retained because it explains why a partitioned (grouped) t-norm operator is NOT the proposed method, and because its zero-weight finding is why the current operator was chosen for its vacuous-view behaviour.
11. [HISTORICAL] `tbiom/reports/EXP_THREEVIEW_ABLATION.md` -- SUPERSEDED for the view ablation: its numbers are DS-R, not PCR6+-R. The current view ladder is in EXP_PCR6P_DEVELOPMENT/EXPOSED. Retained for the paired bootstrap methodology and the original temporal-view effect size.

## Source inventory

| path | bytes | modified | status |
|---|---:|---|---|
| `tbiom/reports/PCR6P_REFERENCE_MANIFEST.md` | 5790 | 2026-09-23 | CURRENT |
| `tbiom/reports/EXP_PCR6P_AUDIT.md` | 8262 | 2026-09-23 | CURRENT |
| `tbiom/reports/EXP_PCR6P_DEVELOPMENT.md` | 12677 | 2026-09-23 | CURRENT |
| `tbiom/reports/EXP_PCR6P_EXPOSED.md` | 22191 | 2026-09-23 | CURRENT |
| `tbiom/reports/PCR6P_OUTCOME.md` | 6136 | 2026-09-23 | CURRENT |
| `tbiom/reports/PCR6P_PRIOR_ART_AND_CLAIMS.md` | 5379 | 2026-09-23 | CURRENT |
| `tbiom/reports/EXP_DF40_CROSS_MANIPULATION.md` | 7250 | 2026-09-22 | CURRENT |
| `tbiom/reports/BASELINE_MATRIX.md` | 8042 | 2026-09-15 | CURRENT |
| `tbiom/reports/REFERENCE_MANIFEST.md` | 7779 | 2026-09-20 | HISTORICAL / superseded |
| `tbiom/reports/GROUPED_FUSION_OUTCOME.md` | 8421 | 2026-09-23 | HISTORICAL / superseded |
| `tbiom/reports/EXP_THREEVIEW_ABLATION.md` | 5048 | 2026-09-17 | HISTORICAL / superseded |

Machine-readable companions to every report below live beside them as `*.json` and are included in `PAPER_PROJECT.zip`.

## Unresolved factual questions at this snapshot

1. **No untouched confirmation set exists.** CDFv2 and DeepfakeEval-2024 informed development; DF40 informed operator design in two prior studies. Every benchmark reported is exposed evaluation.
2. **Seed replication is partial.** The five development folds are reliability-fitting folds, not detector seeds. Four detector configurations exist, but two of them change the spatial training recipe, so they are not four draws of one procedure. The deployed spatial recipe exists at one seed.
3. **Operating-point safeguards.** The pre-registered calibrated FPR/TPR criteria hold in one of the four configurations; AUROC noninferiority holds in all four. See `PCR6P_OUTCOME.md` section 1.
4. **The conflict-by-source uncertainty score failed its pre-registered test** (beats every simple score in 1 of 4 configurations). The conflict decomposition is reported as a diagnostic only. The manuscript makes no uncertainty novelty claim.
5. **DISCERN page/DOI** are not yet available; the bib entry carries the arXiv identifier.


==============================================================================================================
# CURRENT EVIDENCE
==============================================================================================================


--------------------------------------------------------------------------------------------------------------
## BEGIN SOURCE: `tbiom/reports/PCR6P_REFERENCE_MANIFEST.md`

*Role in this handoff:* Method identity, the discount definition, the four detector configurations, folds and the exposure history.

--------------------------------------------------------------------------------------------------------------

# PCR6+ reference manifest

Reuses the grouped-fusion manifest where it is still accurate and says which parts were reused; everything specific
to this study -- the four detector configurations and their own development exports, the operator sources, the
historical PCR6 labelling -- is recomputed here from the live tree.

Repository commit `e58c751f53d21c1d99a2bb7f318be4de4b2fe92a`; working tree DIRTY at audit time (6 paths).

## 1. Reused from the grouped-fusion manifest

`GROUPED_REFERENCE_MANIFEST.md` is reused unchanged for: the branch order and the shared CLIP path (§2), the opinion algebra verified on the caches — 0 rows outside u/2 <= p <= 1-u/2 on all three views (§3), the five source-grouped folds and the DF40 source-grouping rule (§4), and the exposure history (§5). Nothing in those sections changed between the two studies; the checkpoints and caches are the same files, and their hashes are re-verified below.

## 2. The discount, stated exactly

Reliability enters as **evidence scaling**, e'_k = r * e_k, applied in the evidence domain and converted back to an opinion. This is the discount the project deploys and it is kept throughout this study.

It is **not** classical belief-mass (Shafer) discounting, m'(A) = r m(A) for A != Theta and m'(Theta) = 1 - r(1 - m(Theta)). The two agree at the endpoints r = 1 (both the identity) and r = 0 (both produce the vacuous opinion), and differ at every intermediate r. A worked comparison at r = 0.25 is in `REFERENCE_MANIFEST.md` §2, where the two differ by up to 0.29 in p.

The shared r = 0 endpoint is what makes the PCR6+ neutrality property load-bearing here: a view discounted to zero is vacuous under either convention, and PCR6+ is the rule under which a vacuous view is exactly a removed view (`EXP_PCR6P_AUDIT.md` §3).

## 3. The four detector configurations and their own development exports

| tag | spatial views | temporal branch | development exports used to fit the discount |
|---|---|---|---|
| `42` | deployed warm+gate, epoch 19 | `vjepa2_lora_B_seed42` | `warmgate_e19_*` — present |
| `T1337` | deployed warm+gate, epoch 19 | `vjepa2_lora_B_seed1337` | `warmgate_e19_*` — present |
| `armB42` | SBI arm B seed 42, v3.1-selected epoch 8 | `vjepa2_lora_B_seed42` | `sbiarmB_e08_*` — present |
| `armB1337` | SBI arm B seed 1337, v3.1-selected epoch 3 | `vjepa2_lora_B_seed1337` | `sbiarmBs1337_e03_*` — present |

**Arm B is a different training recipe, not merely a different seed.** Within arm B the seed is the only difference between seed 42 and seed 1337, but arm B differs from the deployed warm+gate recipe in more than a seed, and the four configurations are described that way wherever they are compared. The deployed recipe exists at one seed and this task forbids training another.

A configuration's reliability weights are fitted on **its own** development exports. An earlier run of the grouped study fitted the arm-B weights on the deployed system's development exports, because `dsr_system.load` hardcodes the `warmgate_e19` spatial tag; that was corrected in commit `e58c751` and this study inherits the corrected loader.

## 4. Checkpoint and cache hashes

| artefact | sha256 (16) |
|---|---|
| T, V-JEPA 2 LoRA B seed 42 | `6f0d9c63a6c052de` |
| T, V-JEPA 2 LoRA B seed 1337 | `cdb205184f70ceab` |
| APP + OM, warm+gate epoch 19 | `fdc924a7cc9e1579` |
| spatial, armB42 (`discern_ext-best-epoch=08-val_auroc_video=0.9951.ckpt`) | `bc2ca7e4d0b73513` |
| spatial, armB1337 (`discern_ext-best-epoch=03-val_auroc_video=0.9946.ckpt`) | `1a64945f6677a350` |

## 5. Operator sources

| operator | source | status |
|---|---|---|
| PCR6, direct S-source, eq. (15) | Dezert, Dezert & Smarandache, *Improvement of Proportional Conflict Redistribution Rules of Combination of Basic Belief Assignments*, JAIF 16(1), June 2021, pp. 48-73 | **obtained and read** — PDF sha256 `d87ae493e7c409d1b194ed701b6fe746`, https://isif.org/files/isif/2024-01/last.pdf |
| PCR6+, direct S-source, eq. (26), with the binary keeping index of eqs. (22)-(24) | same paper | **obtained and read**; implemented from the equations, not from a summary |
| Dempster, Yager, evidence averaging | as implemented in the grouped study | reused unchanged |

PCR6+ is established prior work by Dezert, Dezert and Smarandache and is credited as such everywhere in this study. It is not presented as a new operator.

## 6. Historical PCR6 labelling

The existing `PCR6-R` rows in `EXP_FUSION_OPERATORS.md`, `EXP_DF40_OPERATORS.md` and the grouped reports were produced by folding the two-source rule over the three views, not by the direct three-source formula. PCR6 is not associative, so these are different operators; they differ by up to 2.082e-01 on random admissible opinions. Those reports are left unchanged, and every table in this study labels the fold **`PCR6-pairwise-R`** and the direct rule **`PCR6-R`**.

## 7. Folds, threshold rule, grid

Five existing source-grouped folds (`GroupShuffleSplit` on the source id, seeds 0-4) and the v3.1 macro objective over swap / reenactment / in-the-wild cells. Threshold: EER on FF++ val, computed per arm on its own scores — the frozen rule already in the protocol. Reliability grid: (0, 0.25, 0.5, 0.75, 1) per view with max = 1, the existing grid and candidate cap, unchanged.

## 8. Exposure

Unchanged from `GROUPED_REFERENCE_MANIFEST.md` §5 and repeated because it governs every claim here: CDFv2 and DeepfakeEval2024 informed development; DF40 has already informed operator design in two prior studies; FaceShifter was read once after the recipe was frozen. **No untouched confirmation set remains**, so outcome 6 of the brief — independent confirmation pending — applies to every result in this study regardless of what else it finds.

## END SOURCE: `tbiom/reports/PCR6P_REFERENCE_MANIFEST.md`


--------------------------------------------------------------------------------------------------------------
## BEGIN SOURCE: `tbiom/reports/EXP_PCR6P_AUDIT.md`

*Role in this handoff:* Exact operator equations from the primary source, 17 algebraic unit tests, and the resolution of the vacuous-source neutrality question.

--------------------------------------------------------------------------------------------------------------

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

## END SOURCE: `tbiom/reports/EXP_PCR6P_AUDIT.md`


--------------------------------------------------------------------------------------------------------------
## BEGIN SOURCE: `tbiom/reports/EXP_PCR6P_DEVELOPMENT.md`

*Role in this handoff:* The fixed arm matrix on development for all four configurations: detection, calibration, uncertainty scores, conflict cases, fitted weights.

--------------------------------------------------------------------------------------------------------------

# PCR6+ on development

Four detector configurations, each fitting its reliability weights on its own development exports. The conflict-by-
source score is a logistic model on at most five inputs, fitted on the fitting fold only and applied to the selection
fold, so it never sees the data it is scored on. Paired source-group bootstrap, 2000 draws, materiality floor 1e-4.

Wall clock 8.1 minutes, CPU. Pre-registration: `tbiom/configs/pcr6p_fusion_preregister.yaml`.

## Configuration `42`

| arm | v3.1 clean | swap | reenact | in-the-wild | JPEG q40 | resize 0.60 | FPR | TPR | NLL | Brier | ECE | Δ v3.1 vs Dempster-R (95% CI) | folds with T/APP/OM weight 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **PCR6P-R** | 0.8794 | 0.9301 | 0.9111 | 0.7971 | 0.8104 | 0.8581 | 0.104 | 0.735 | 0.4795 | 0.1529 | 0.1547 | -0.0014 [-0.0040, +0.0012] | 0/3/0 |
| PCR6-R | 0.8793 | 0.9314 | 0.9078 | 0.7987 | 0.8093 | 0.8574 | 0.083 | 0.684 | 0.4710 | 0.1501 | 0.1506 | -0.0015 [-0.0038, +0.0008] | 0/2/0 |
| PCR6-pairwise-R | 0.8802 | 0.9333 | 0.9067 | 0.8006 | 0.8091 | 0.8574 | 0.078 | 0.675 | 0.4744 | 0.1491 | 0.1437 | -0.0006 [-0.0024, +0.0012] | 0/1/0 |
| Yager-R | 0.8787 | 0.9296 | 0.9085 | 0.7978 | 0.8129 | 0.8575 | 0.103 | 0.729 | 0.4744 | 0.1537 | 0.1642 | -0.0021 [-0.0055, +0.0011] | 0/4/0 |
| Averaging-R *(baseline)* | 0.8799 | 0.9322 | 0.9133 | 0.7942 | 0.8088 | 0.8592 | 0.109 | 0.748 | 0.4263 | 0.1370 | 0.1352 | -0.0009 [-0.0032, +0.0013] | 0/0/0 |
| Dempster-R *(baseline)* | 0.8808 | 0.9345 | 0.9085 | 0.7993 | 0.8096 | 0.8589 | 0.107 | 0.737 | 0.4706 | 0.1469 | 0.1310 | +0.0000 [+0.0000, +0.0000] | 0/0/0 |
| T+OM (PCR6+) | 0.8800 | 0.9284 | 0.9157 | 0.7958 | 0.8135 | 0.8595 | 0.113 | 0.749 | 0.4851 | 0.1564 | 0.1646 | -0.0008 [-0.0041, +0.0024] | 0/5/0 |
| T+APP (PCR6+) | 0.8693 | 0.9197 | 0.9065 | 0.7817 | 0.7937 | 0.8516 | 0.116 | 0.682 | 0.4300 | 0.1380 | 0.0990 | -0.0115 [-0.0156, -0.0074] | 0/0/1 |
| APP+OM (PCR6+) | 0.8262 | 0.9283 | 0.7644 | 0.7860 | 0.7423 | 0.7860 | 0.149 | 0.710 | 0.5231 | 0.1755 | 0.1427 | -0.0546 [-0.0689, -0.0403] | 0/5/0 |
| T only | 0.8094 | 0.8059 | 0.9135 | 0.7088 | 0.7596 | 0.8026 | 0.137 | 0.644 | 0.6092 | 0.1997 | 0.1742 | -0.0713 [-0.0855, -0.0573] | 0/5/0 |

### Uncertainty scores, configuration `42`

| score | error-detection AUROC | E-AURC | risk@90 | risk@80 | Δ error-AUROC vs conflict-by-source (95% CI) |
|---|---:|---:|---:|---:|---|
| vacuity | 0.7343 | 0.07190 | 0.1885 | 0.1641 | +0.0090 [-0.0103, +0.0290] (2/5 splits above the floor) |
| conflict_K | 0.6727 | 0.09083 | 0.2018 | 0.1830 | +0.0707 [+0.0450, +0.0958] (4/5 splits above the floor) |
| entropy | 0.7508 | 0.06445 | 0.1824 | 0.1642 | -0.0074 [-0.0238, +0.0086] (2/5 splits above the floor) |
| margin | 0.7508 | 0.06445 | 0.1824 | 0.1642 | -0.0074 [-0.0238, +0.0086] (2/5 splits above the floor) |
| **conflict_by_source** | 0.7432 | 0.06662 | 0.1886 | 0.1671 | — |

Conflict-by-source coefficients (standardised inputs, mean over folds): vacuity +0.863, K_TS +0.404, K_AO -0.126, K_all +0.183, margin -0.005.


### Conflict case analysis, configuration `42`

| case | videos | K_TS | K_AO | K_all | PCR6+-R correct |
|---|---:|---:|---:|---:|---:|
| temporal right, both spatial wrong | 362 | 0.4194 | 0.0072 | 0.0142 | 0.6552 |
| temporal wrong, both spatial right | 520 | 0.3740 | 0.0076 | 0.0106 | 0.6472 |
| all three agree and right | 1512 | 0.0324 | 0.0038 | 0.0095 | 1.0000 |
| all three wrong | 214 | 0.0439 | 0.0092 | 0.0135 | 0.0018 |

## Configuration `T1337`

| arm | v3.1 clean | swap | reenact | in-the-wild | JPEG q40 | resize 0.60 | FPR | TPR | NLL | Brier | ECE | Δ v3.1 vs Dempster-R (95% CI) | folds with T/APP/OM weight 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **PCR6P-R** | 0.8803 | 0.9303 | 0.9064 | 0.8044 | 0.8240 | 0.8555 | 0.080 | 0.687 | 0.4389 | 0.1425 | 0.1433 | +0.0002 [-0.0016, +0.0021] | 0/5/0 |
| PCR6-R | 0.8804 | 0.9310 | 0.9054 | 0.8048 | 0.8251 | 0.8552 | 0.082 | 0.688 | 0.4366 | 0.1421 | 0.1458 | +0.0003 [-0.0018, +0.0024] | 0/5/0 |
| PCR6-pairwise-R | 0.8796 | 0.9314 | 0.9036 | 0.8038 | 0.8216 | 0.8536 | 0.072 | 0.663 | 0.4456 | 0.1411 | 0.1347 | -0.0005 [-0.0025, +0.0015] | 0/3/0 |
| Yager-R | 0.8802 | 0.9314 | 0.9043 | 0.8048 | 0.8257 | 0.8545 | 0.082 | 0.688 | 0.4371 | 0.1425 | 0.1492 | +0.0001 [-0.0025, +0.0026] | 0/5/0 |
| Averaging-R *(baseline)* | 0.8783 | 0.9315 | 0.9042 | 0.7991 | 0.8193 | 0.8538 | 0.074 | 0.674 | 0.4156 | 0.1323 | 0.1351 | -0.0018 [-0.0042, +0.0006] | 0/3/0 |
| Dempster-R *(baseline)* | 0.8801 | 0.9312 | 0.9054 | 0.8036 | 0.8226 | 0.8555 | 0.067 | 0.653 | 0.4346 | 0.1401 | 0.1296 | +0.0000 [+0.0000, +0.0000] | 0/4/0 |
| T+OM (PCR6+) | 0.8803 | 0.9303 | 0.9064 | 0.8044 | 0.8240 | 0.8555 | 0.080 | 0.687 | 0.4389 | 0.1425 | 0.1433 | +0.0002 [-0.0016, +0.0021] | 0/5/0 |
| T+APP (PCR6+) | 0.8694 | 0.9182 | 0.9029 | 0.7871 | 0.8092 | 0.8471 | 0.101 | 0.651 | 0.4026 | 0.1283 | 0.0802 | -0.0106 [-0.0155, -0.0057] | 0/0/1 |
| APP+OM (PCR6+) | 0.8262 | 0.9283 | 0.7644 | 0.7860 | 0.7423 | 0.7860 | 0.149 | 0.710 | 0.5231 | 0.1755 | 0.1427 | -0.0539 [-0.0698, -0.0381] | 0/5/0 |
| T only | 0.8173 | 0.8191 | 0.9062 | 0.7265 | 0.7829 | 0.7995 | 0.097 | 0.612 | 0.5148 | 0.1698 | 0.1340 | -0.0627 [-0.0753, -0.0504] | 0/5/0 |

### Uncertainty scores, configuration `T1337`

| score | error-detection AUROC | E-AURC | risk@90 | risk@80 | Δ error-AUROC vs conflict-by-source (95% CI) |
|---|---:|---:|---:|---:|---|
| vacuity | 0.7502 | 0.07807 | 0.2052 | 0.1810 | +0.0546 [+0.0333, +0.0766] (3/5 splits above the floor) |
| conflict_K | 0.7062 | 0.08555 | 0.2137 | 0.1883 | +0.0985 [+0.0766, +0.1197] (5/5 splits above the floor) |
| entropy | 0.8087 | 0.05091 | 0.2015 | 0.1593 | -0.0040 [-0.0105, +0.0022] (1/5 splits above the floor) |
| margin | 0.8087 | 0.05091 | 0.2015 | 0.1593 | -0.0040 [-0.0105, +0.0022] (1/5 splits above the floor) |
| **conflict_by_source** | 0.8047 | 0.05134 | 0.2091 | 0.1718 | — |

Conflict-by-source coefficients (standardised inputs, mean over folds): vacuity +0.733, K_TS +0.445, K_AO +0.000, K_all +0.000, margin +0.550.


### Conflict case analysis, configuration `T1337`

| case | videos | K_TS | K_AO | K_all | PCR6+-R correct |
|---|---:|---:|---:|---:|---:|
| temporal right, both spatial wrong | 340 | 0.4116 | 0.0000 | 0.0000 | 0.6052 |
| temporal wrong, both spatial right | 517 | 0.2578 | 0.0000 | 0.0000 | 0.5791 |
| all three agree and right | 1515 | 0.0347 | 0.0000 | 0.0000 | 0.9978 |
| all three wrong | 236 | 0.0522 | 0.0000 | 0.0000 | 0.0008 |

## Configuration `armB42`

| arm | v3.1 clean | swap | reenact | in-the-wild | JPEG q40 | resize 0.60 | FPR | TPR | NLL | Brier | ECE | Δ v3.1 vs Dempster-R (95% CI) | folds with T/APP/OM weight 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **PCR6P-R** | 0.8671 | 0.9099 | 0.9083 | 0.7832 | 0.8101 | 0.8493 | 0.063 | 0.621 | 0.5479 | 0.1750 | 0.1926 | +0.0004 [-0.0010, +0.0017] | 0/5/0 |
| PCR6-R | 0.8671 | 0.9103 | 0.9072 | 0.7837 | 0.8090 | 0.8496 | 0.063 | 0.622 | 0.5385 | 0.1734 | 0.1939 | +0.0003 [-0.0015, +0.0021] | 0/5/0 |
| PCR6-pairwise-R | 0.8671 | 0.9099 | 0.9083 | 0.7832 | 0.8101 | 0.8493 | 0.063 | 0.621 | 0.5479 | 0.1750 | 0.1926 | +0.0004 [-0.0010, +0.0017] | 0/5/0 |
| Yager-R | 0.8671 | 0.9105 | 0.9062 | 0.7845 | 0.8075 | 0.8493 | 0.082 | 0.636 | 0.5331 | 0.1725 | 0.1960 | +0.0003 [-0.0019, +0.0026] | 0/5/0 |
| Averaging-R *(baseline)* | 0.8647 | 0.9119 | 0.9038 | 0.7783 | 0.8078 | 0.8476 | 0.087 | 0.691 | 0.4732 | 0.1574 | 0.1724 | -0.0021 [-0.0049, +0.0007] | 0/2/0 |
| Dempster-R *(baseline)* | 0.8668 | 0.9101 | 0.9092 | 0.7809 | 0.8108 | 0.8491 | 0.067 | 0.651 | 0.5470 | 0.1741 | 0.1845 | +0.0000 [+0.0000, +0.0000] | 0/5/0 |
| T+OM (PCR6+) | 0.8671 | 0.9099 | 0.9083 | 0.7832 | 0.8101 | 0.8493 | 0.063 | 0.621 | 0.5479 | 0.1750 | 0.1926 | +0.0004 [-0.0010, +0.0017] | 0/5/0 |
| T+APP (PCR6+) | 0.8651 | 0.9104 | 0.9051 | 0.7799 | 0.8084 | 0.8470 | 0.073 | 0.633 | 0.5478 | 0.1738 | 0.1869 | -0.0016 [-0.0040, +0.0007] | 0/0/5 |
| APP+OM (PCR6+) | 0.8057 | 0.8929 | 0.7610 | 0.7633 | 0.7478 | 0.7460 | 0.133 | 0.668 | 0.6404 | 0.2095 | 0.2194 | -0.0611 [-0.0777, -0.0445] | 2/0/0 |
| T only | 0.8094 | 0.8059 | 0.9135 | 0.7088 | 0.7596 | 0.8026 | 0.137 | 0.644 | 0.6092 | 0.1997 | 0.1742 | -0.0573 [-0.0697, -0.0452] | 0/5/0 |

### Uncertainty scores, configuration `armB42`

| score | error-detection AUROC | E-AURC | risk@90 | risk@80 | Δ error-AUROC vs conflict-by-source (95% CI) |
|---|---:|---:|---:|---:|---|
| vacuity | 0.7190 | 0.10091 | 0.2497 | 0.2257 | -0.0101 [-0.0285, +0.0088] (1/5 splits above the floor) |
| conflict_K | 0.7107 | 0.09662 | 0.2501 | 0.2298 | -0.0016 [-0.0255, +0.0220] (2/5 splits above the floor) |
| entropy | 0.7794 | 0.07052 | 0.2401 | 0.2027 | -0.0704 [-0.0887, -0.0529] (1/5 splits above the floor) |
| margin | 0.7794 | 0.07052 | 0.2401 | 0.2027 | -0.0704 [-0.0887, -0.0529] (1/5 splits above the floor) |
| **conflict_by_source** | 0.7089 | 0.10463 | 0.2531 | 0.2404 | — |

Conflict-by-source coefficients (standardised inputs, mean over folds): vacuity +0.897, K_TS +0.701, K_AO +0.000, K_all +0.000, margin -0.537.


### Conflict case analysis, configuration `armB42`

| case | videos | K_TS | K_AO | K_all | PCR6+-R correct |
|---|---:|---:|---:|---:|---:|
| temporal right, both spatial wrong | 446 | 0.3800 | 0.0000 | 0.0000 | 0.5869 |
| temporal wrong, both spatial right | 501 | 0.3078 | 0.0000 | 0.0000 | 0.4440 |
| all three agree and right | 1581 | 0.0369 | 0.0000 | 0.0000 | 0.9896 |
| all three wrong | 305 | 0.0500 | 0.0000 | 0.0000 | 0.0180 |

## Configuration `armB1337`

| arm | v3.1 clean | swap | reenact | in-the-wild | JPEG q40 | resize 0.60 | FPR | TPR | NLL | Brier | ECE | Δ v3.1 vs Dempster-R (95% CI) | folds with T/APP/OM weight 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **PCR6P-R** | 0.8751 | 0.9202 | 0.8969 | 0.8081 | 0.8284 | 0.8500 | 0.049 | 0.585 | 0.5306 | 0.1658 | 0.1790 | -0.0008 [-0.0025, +0.0009] | 0/0/0 |
| PCR6-R | 0.8747 | 0.9204 | 0.8966 | 0.8072 | 0.8279 | 0.8495 | 0.048 | 0.575 | 0.5205 | 0.1643 | 0.1799 | -0.0012 [-0.0031, +0.0007] | 0/1/0 |
| PCR6-pairwise-R | 0.8752 | 0.9209 | 0.8946 | 0.8102 | 0.8288 | 0.8504 | 0.049 | 0.586 | 0.5535 | 0.1694 | 0.1827 | -0.0007 [-0.0024, +0.0010] | 0/0/0 |
| Yager-R | 0.8746 | 0.9192 | 0.8985 | 0.8060 | 0.8289 | 0.8486 | 0.054 | 0.591 | 0.4858 | 0.1597 | 0.1798 | -0.0013 [-0.0039, +0.0012] | 0/4/0 |
| Averaging-R *(baseline)* | 0.8739 | 0.9190 | 0.8977 | 0.8051 | 0.8270 | 0.8500 | 0.057 | 0.620 | 0.4540 | 0.1485 | 0.1651 | -0.0020 [-0.0042, +0.0003] | 0/0/0 |
| Dempster-R *(baseline)* | 0.8759 | 0.9212 | 0.8955 | 0.8110 | 0.8292 | 0.8513 | 0.053 | 0.600 | 0.5506 | 0.1686 | 0.1745 | +0.0000 [+0.0000, +0.0000] | 0/0/0 |
| T+OM (PCR6+) | 0.8750 | 0.9174 | 0.9024 | 0.8052 | 0.8303 | 0.8482 | 0.059 | 0.605 | 0.4890 | 0.1602 | 0.1737 | -0.0009 [-0.0036, +0.0018] | 0/5/0 |
| T+APP (PCR6+) | 0.8741 | 0.9176 | 0.8997 | 0.8049 | 0.8283 | 0.8491 | 0.059 | 0.599 | 0.4730 | 0.1543 | 0.1584 | -0.0018 [-0.0041, +0.0004] | 0/0/5 |
| APP+OM (PCR6+) | 0.8245 | 0.9076 | 0.7726 | 0.7933 | 0.7678 | 0.7653 | 0.103 | 0.632 | 0.5573 | 0.1899 | 0.1841 | -0.0515 [-0.0657, -0.0374] | 0/3/1 |
| T only | 0.8173 | 0.8191 | 0.9062 | 0.7265 | 0.7829 | 0.7995 | 0.097 | 0.612 | 0.5148 | 0.1698 | 0.1340 | -0.0585 [-0.0721, -0.0452] | 0/5/0 |

### Uncertainty scores, configuration `armB1337`

| score | error-detection AUROC | E-AURC | risk@90 | risk@80 | Δ error-AUROC vs conflict-by-source (95% CI) |
|---|---:|---:|---:|---:|---|
| vacuity | 0.7398 | 0.09959 | 0.2618 | 0.2353 | +0.1011 [+0.0780, +0.1247] (4/5 splits above the floor) |
| conflict_K | 0.7535 | 0.08787 | 0.2562 | 0.2268 | +0.0874 [+0.0717, +0.1036] (5/5 splits above the floor) |
| entropy | 0.8311 | 0.05692 | 0.2505 | 0.1903 | +0.0098 [+0.0049, +0.0150] (5/5 splits above the floor) |
| margin | 0.8311 | 0.05692 | 0.2505 | 0.1903 | +0.0098 [+0.0049, +0.0150] (5/5 splits above the floor) |
| **conflict_by_source** | 0.8409 | 0.05283 | 0.2518 | 0.1961 | — |

Conflict-by-source coefficients (standardised inputs, mean over folds): vacuity +0.646, K_TS +0.022, K_AO -0.306, K_all +0.053, margin +1.118.


### Conflict case analysis, configuration `armB1337`

| case | videos | K_TS | K_AO | K_all | PCR6+-R correct |
|---|---:|---:|---:|---:|---:|
| temporal right, both spatial wrong | 423 | 0.4683 | 0.0117 | 0.0344 | 0.4199 |
| temporal wrong, both spatial right | 468 | 0.2922 | 0.0261 | 0.0276 | 0.5153 |
| all three agree and right | 1574 | 0.0481 | 0.0070 | 0.0230 | 0.9981 |
| all three wrong | 361 | 0.0809 | 0.0213 | 0.0236 | 0.0049 |

## END SOURCE: `tbiom/reports/EXP_PCR6P_DEVELOPMENT.md`


--------------------------------------------------------------------------------------------------------------
## BEGIN SOURCE: `tbiom/reports/EXP_PCR6P_EXPOSED.md`

*Role in this handoff:* Ten test sets, processing conditions and all 41 DF40 entries, per arm and per configuration, with both required DF40 means.

--------------------------------------------------------------------------------------------------------------

# PCR6+ on the exposed suites

Ten test sets, the processed conditions, and all 41 DF40 generator sets, for every arm of the fixed matrix and every
detector configuration. Labelled exposed benchmark evaluation: CDFv2 and DeepfakeEval2024 informed development and
DF40 has informed operator design in three studies now, so none of this confirms anything.

Both DF40 means are reported with their weights: family-balanced gives each of the four family x source cells weight
1/4; generator-macro gives each of the 41 sets weight 1/41, which favours the Celeb-DF-sourced cells because they
contain more generators.

Wall clock 1.3 minutes, CPU.

## Configuration `42`

### Ten test sets, video AUROC

| arm | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | DFEval24 | FFIW | KoDF | IDForge | DFDM | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **PCR6P-R** | 0.9583 | 0.9639 | 0.9307 | 0.8530 | 0.8597 | 0.7235 | 0.9217 | 0.9600 | 0.9631 | 0.9954 | 0.9129 |
| PCR6-R | 0.9596 | 0.9618 | 0.9318 | 0.8553 | 0.8606 | 0.7235 | 0.9245 | 0.9583 | 0.9636 | 0.9954 | 0.9134 |
| PCR6-pairwise-R | 0.9607 | 0.9613 | 0.9334 | 0.8594 | 0.8633 | 0.7241 | 0.9272 | 0.9577 | 0.9648 | 0.9954 | 0.9147 |
| Yager-R | 0.9602 | 0.9635 | 0.9310 | 0.8507 | 0.8576 | 0.7223 | 0.9210 | 0.9600 | 0.9624 | 0.9953 | 0.9124 |
| Averaging-R | 0.9542 | 0.9632 | 0.9330 | 0.8578 | 0.8642 | 0.7236 | 0.9241 | 0.9602 | 0.9650 | 0.9960 | 0.9141 |
| Dempster-R | 0.9598 | 0.9616 | 0.9338 | 0.8626 | 0.8656 | 0.7252 | 0.9282 | 0.9574 | 0.9656 | 0.9956 | 0.9155 |
| T+OM (PCR6+) | 0.9575 | 0.9669 | 0.9294 | 0.8489 | 0.8576 | 0.7231 | 0.9174 | 0.9624 | 0.9621 | 0.9955 | 0.9121 |
| T+APP (PCR6+) | 0.9348 | 0.9515 | 0.9116 | 0.8346 | 0.8450 | 0.7174 | 0.9112 | 0.9540 | 0.9505 | 0.9923 | 0.9003 |
| APP+OM (PCR6+) | 0.9733 | 0.8810 | 0.9310 | 0.8828 | 0.8752 | 0.6694 | 0.9357 | 0.8547 | 0.9639 | 0.9921 | 0.8959 |
| T only | 0.7384 | 0.9263 | 0.7967 | 0.7009 | 0.7409 | 0.6733 | 0.8011 | 0.9656 | 0.8899 | 0.9798 | 0.8213 |

### PCR6+-R per test set, at the frozen threshold

| test set | AUROC | FPR | TPR | ECE | NLL | Brier | E-AURC (vacuity) | err-AUROC vacuity | err-AUROC K | err-AUROC entropy | risk@90 | risk@80 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CDFv2 | 0.9583 | 0.0404 | 0.6853 | 0.2544 | 0.5159 | 0.1776 | 0.1119 | 0.6409 | 0.6646 | 0.6668 | 0.2163 | 0.2082 |
| CDFv3 | 0.9639 | 0.0404 | 0.7960 | 0.2830 | 0.5174 | 0.1705 | 0.0547 | 0.7683 | 0.5420 | 0.6418 | 0.1741 | 0.1454 |
| DFD | 0.9307 | 0.1234 | 0.7984 | 0.2202 | 0.4558 | 0.1536 | 0.0470 | 0.7902 | 0.7154 | 0.7499 | 0.1692 | 0.1394 |
| DFDC | 0.8530 | 0.1906 | 0.7425 | 0.1053 | 0.5298 | 0.1671 | 0.0983 | 0.6722 | 0.6673 | 0.7082 | 0.2082 | 0.1982 |
| DFDCP | 0.8597 | 0.1791 | 0.7193 | 0.1805 | 0.5695 | 0.1881 | 0.1046 | 0.6746 | 0.6426 | 0.6684 | 0.2363 | 0.2221 |
| DFEval24 | 0.7235 | 0.1907 | 0.4819 | 0.1880 | 0.7598 | 0.2506 | 0.1855 | 0.6268 | 0.5676 | 0.6135 | 0.3315 | 0.3107 |
| FFIW | 0.9217 | 0.1150 | 0.7531 | 0.0987 | 0.4049 | 0.1323 | 0.0749 | 0.6994 | 0.7540 | 0.7851 | 0.1616 | 0.1490 |
| KoDF | 0.9600 | 0.0263 | 0.8796 | 0.1284 | 0.3452 | 0.0965 | 0.0199 | 0.8208 | 0.6551 | 0.7177 | 0.0597 | 0.0522 |
| IDForge | 0.9631 | 0.0670 | 0.8206 | 0.0784 | 0.2855 | 0.0916 | 0.0314 | 0.7935 | 0.7819 | 0.8459 | 0.0991 | 0.0815 |
| DFDM | 0.9954 | 0.0371 | 0.9710 | 0.0602 | 0.1080 | 0.0307 | 0.0034 | 0.8973 | 0.8695 | 0.9077 | 0.0165 | 0.0062 |

### DF40, both required means

| arm | swap CDF | swap FF++ | reenact CDF | reenact FF++ | family-balanced (1/4 each) | generator-macro (1/41 each) |
|---|---:|---:|---:|---:|---:|---:|
| **PCR6P-R** | 0.9713 | 0.9834 | 0.9245 | 0.9616 | 0.9602 | 0.9588 |
| PCR6-R | 0.9716 | 0.9845 | 0.9176 | 0.9593 | 0.9582 | 0.9566 |
| PCR6-pairwise-R | 0.9725 | 0.9851 | 0.9166 | 0.9602 | 0.9586 | 0.9569 |
| Yager-R | 0.9708 | 0.9831 | 0.9215 | 0.9585 | 0.9585 | 0.9570 |
| Averaging-R *(baseline)* | 0.9729 | 0.9846 | 0.9267 | 0.9628 | 0.9618 | 0.9604 |
| Dempster-R *(baseline)* | 0.9731 | 0.9854 | 0.9174 | 0.9607 | 0.9591 | 0.9575 |
| T+OM (PCR6+) | 0.9707 | 0.9819 | 0.9337 | 0.9633 | 0.9624 | 0.9613 |
| T+APP (PCR6+) | 0.9602 | 0.9845 | 0.9036 | 0.9605 | 0.9522 | 0.9504 |
| APP+OM (PCR6+) | 0.9674 | 0.9709 | 0.7481 | 0.8353 | 0.8804 | 0.8739 |
| T only | 0.8857 | 0.9019 | 0.9355 | 0.9550 | 0.9195 | 0.9212 |

### DF40 per generator, PCR6+-R against the baselines

| generator | family | source | **PCR6+-R** | PCR6-R | Dempster-R | Averaging-R |
|---|---|---|---:|---:|---:|---:|
| blendface | face swap | cdf | **0.9790** | 0.9787 | 0.9801 | 0.9806 |
| deepfacelab | face swap | cdf | **0.9577** | 0.9589 | 0.9610 | 0.9640 |
| e4s | face swap | cdf | **0.9841** | 0.9851 | 0.9892 | 0.9869 |
| facedancer | face swap | cdf | **0.9710** | 0.9709 | 0.9714 | 0.9721 |
| faceswap | face swap | cdf | **0.9993** | 0.9994 | 0.9995 | 0.9996 |
| fsgan | face swap | cdf | **0.9912** | 0.9911 | 0.9906 | 0.9919 |
| inswap | face swap | cdf | **0.9523** | 0.9526 | 0.9551 | 0.9544 |
| mobileswap | face swap | cdf | **0.9571** | 0.9582 | 0.9584 | 0.9564 |
| simswap | face swap | cdf | **0.9396** | 0.9401 | 0.9416 | 0.9399 |
| uniface | face swap | cdf | **0.9813** | 0.9813 | 0.9837 | 0.9833 |
| blendface | face swap | ff | **0.9846** | 0.9844 | 0.9858 | 0.9865 |
| e4s | face swap | ff | **0.9910** | 0.9925 | 0.9927 | 0.9907 |
| facedancer | face swap | ff | **0.9690** | 0.9713 | 0.9713 | 0.9699 |
| faceswap | face swap | ff | **0.9949** | 0.9948 | 0.9948 | 0.9952 |
| fsgan | face swap | ff | **0.9934** | 0.9939 | 0.9935 | 0.9939 |
| inswap | face swap | ff | **0.9739** | 0.9757 | 0.9783 | 0.9763 |
| mobileswap | face swap | ff | **0.9824** | 0.9831 | 0.9848 | 0.9845 |
| simswap | face swap | ff | **0.9708** | 0.9722 | 0.9749 | 0.9738 |
| uniface | face swap | ff | **0.9911** | 0.9922 | 0.9928 | 0.9909 |
| MRAA | reenactment | cdf | **0.9927** | 0.9925 | 0.9934 | 0.9937 |
| danet | reenactment | cdf | **0.8852** | 0.8688 | 0.8645 | 0.8834 |
| facevid2vid | reenactment | cdf | **0.8512** | 0.8390 | 0.8403 | 0.8581 |
| fomm | reenactment | cdf | **0.9337** | 0.9308 | 0.9341 | 0.9408 |
| hyperreenact | reenactment | cdf | **0.9759** | 0.9723 | 0.9708 | 0.9755 |
| lia | reenactment | cdf | **0.9974** | 0.9975 | 0.9977 | 0.9976 |
| mcnet | reenactment | cdf | **0.8819** | 0.8673 | 0.8662 | 0.8846 |
| one_shot_free | reenactment | cdf | **0.9096** | 0.9059 | 0.9102 | 0.9177 |
| pirender | reenactment | cdf | **0.8652** | 0.8618 | 0.8626 | 0.8707 |
| sadtalker | reenactment | cdf | **0.9020** | 0.8862 | 0.8857 | 0.8982 |
| tpsm | reenactment | cdf | **0.9751** | 0.9712 | 0.9661 | 0.9734 |
| MRAA | reenactment | ff | **0.9987** | 0.9988 | 0.9993 | 0.9992 |
| danet | reenactment | ff | **0.9592** | 0.9546 | 0.9546 | 0.9580 |
| facevid2vid | reenactment | ff | **0.9071** | 0.9017 | 0.9058 | 0.9103 |
| fomm | reenactment | ff | **0.9756** | 0.9766 | 0.9792 | 0.9783 |
| hyperreenact | reenactment | ff | **0.9839** | 0.9825 | 0.9827 | 0.9835 |
| lia | reenactment | ff | **0.9998** | 0.9998 | 0.9999 | 0.9998 |
| mcnet | reenactment | ff | **0.9602** | 0.9562 | 0.9566 | 0.9598 |
| one_shot_free | reenactment | ff | **0.9438** | 0.9414 | 0.9442 | 0.9486 |
| pirender | reenactment | ff | **0.9222** | 0.9207 | 0.9274 | 0.9314 |
| sadtalker | reenactment | ff | **0.9570** | 0.9525 | 0.9503 | 0.9526 |
| tpsm | reenactment | ff | **0.9705** | 0.9677 | 0.9677 | 0.9691 |

## Configuration `T1337`

### Ten test sets, video AUROC

| arm | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | DFEval24 | FFIW | KoDF | IDForge | DFDM | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **PCR6P-R** | 0.9512 | 0.9592 | 0.9315 | 0.8566 | 0.8749 | 0.7243 | 0.9209 | 0.9660 | 0.9542 | 0.9944 | 0.9133 |
| PCR6-R | 0.9526 | 0.9589 | 0.9323 | 0.8561 | 0.8744 | 0.7244 | 0.9213 | 0.9659 | 0.9546 | 0.9945 | 0.9135 |
| PCR6-pairwise-R | 0.9515 | 0.9565 | 0.9315 | 0.8578 | 0.8741 | 0.7242 | 0.9230 | 0.9642 | 0.9547 | 0.9941 | 0.9132 |
| Yager-R | 0.9536 | 0.9584 | 0.9330 | 0.8556 | 0.8733 | 0.7241 | 0.9216 | 0.9658 | 0.9550 | 0.9945 | 0.9135 |
| Averaging-R | 0.9502 | 0.9573 | 0.9323 | 0.8628 | 0.8806 | 0.7235 | 0.9228 | 0.9652 | 0.9551 | 0.9950 | 0.9145 |
| Dempster-R | 0.9506 | 0.9585 | 0.9307 | 0.8595 | 0.8774 | 0.7239 | 0.9209 | 0.9655 | 0.9540 | 0.9946 | 0.9136 |
| T+OM (PCR6+) | 0.9512 | 0.9592 | 0.9315 | 0.8566 | 0.8749 | 0.7243 | 0.9209 | 0.9660 | 0.9542 | 0.9944 | 0.9133 |
| T+APP (PCR6+) | 0.9254 | 0.9454 | 0.9129 | 0.8360 | 0.8596 | 0.7196 | 0.9112 | 0.9608 | 0.9372 | 0.9906 | 0.8999 |
| APP+OM (PCR6+) | 0.9733 | 0.8810 | 0.9310 | 0.8828 | 0.8752 | 0.6694 | 0.9357 | 0.8547 | 0.9639 | 0.9921 | 0.8959 |
| T only | 0.7646 | 0.9237 | 0.8119 | 0.7165 | 0.7831 | 0.6793 | 0.8114 | 0.9691 | 0.8712 | 0.9775 | 0.8308 |

### PCR6+-R per test set, at the frozen threshold

| test set | AUROC | FPR | TPR | ECE | NLL | Brier | E-AURC (vacuity) | err-AUROC vacuity | err-AUROC K | err-AUROC entropy | risk@90 | risk@80 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CDFv2 | 0.9512 | 0.0270 | 0.5400 | 0.2170 | 0.4696 | 0.1594 | 0.1792 | 0.6086 | 0.8146 | 0.8471 | 0.2955 | 0.2892 |
| CDFv3 | 0.9592 | 0.0270 | 0.6948 | 0.2722 | 0.4853 | 0.1616 | 0.0594 | 0.8253 | 0.6154 | 0.8156 | 0.2494 | 0.2084 |
| DFD | 0.9315 | 0.0512 | 0.6866 | 0.2319 | 0.4845 | 0.1647 | 0.0736 | 0.7750 | 0.8243 | 0.8589 | 0.2527 | 0.2216 |
| DFDC | 0.8566 | 0.1088 | 0.5974 | 0.1182 | 0.5243 | 0.1689 | 0.1398 | 0.6197 | 0.7253 | 0.7599 | 0.2465 | 0.2335 |
| DFDCP | 0.8749 | 0.1009 | 0.6005 | 0.1603 | 0.5269 | 0.1755 | 0.1418 | 0.6693 | 0.7159 | 0.7825 | 0.2683 | 0.2450 |
| DFEval24 | 0.7243 | 0.1103 | 0.3347 | 0.1817 | 0.7347 | 0.2469 | 0.2100 | 0.6105 | 0.6130 | 0.6643 | 0.3683 | 0.3494 |
| FFIW | 0.9209 | 0.0686 | 0.6578 | 0.0811 | 0.3904 | 0.1270 | 0.0818 | 0.7146 | 0.7503 | 0.8292 | 0.1737 | 0.1594 |
| KoDF | 0.9660 | 0.0154 | 0.8204 | 0.1058 | 0.2736 | 0.0787 | 0.0404 | 0.7692 | 0.7835 | 0.8305 | 0.1161 | 0.1027 |
| IDForge | 0.9542 | 0.0566 | 0.7328 | 0.0688 | 0.2983 | 0.0962 | 0.0496 | 0.7659 | 0.8229 | 0.8837 | 0.1339 | 0.1159 |
| DFDM | 0.9944 | 0.0247 | 0.9468 | 0.0502 | 0.0987 | 0.0272 | 0.0044 | 0.9247 | 0.8964 | 0.9575 | 0.0194 | 0.0099 |

### DF40, both required means

| arm | swap CDF | swap FF++ | reenact CDF | reenact FF++ | family-balanced (1/4 each) | generator-macro (1/41 each) |
|---|---:|---:|---:|---:|---:|---:|
| **PCR6P-R** | 0.9733 | 0.9814 | 0.9363 | 0.9693 | 0.9651 | 0.9641 |
| PCR6-R | 0.9735 | 0.9816 | 0.9344 | 0.9685 | 0.9645 | 0.9634 |
| PCR6-pairwise-R | 0.9730 | 0.9826 | 0.9288 | 0.9680 | 0.9631 | 0.9619 |
| Yager-R | 0.9735 | 0.9816 | 0.9326 | 0.9677 | 0.9639 | 0.9628 |
| Averaging-R *(baseline)* | 0.9741 | 0.9828 | 0.9325 | 0.9694 | 0.9647 | 0.9636 |
| Dempster-R *(baseline)* | 0.9735 | 0.9818 | 0.9339 | 0.9691 | 0.9646 | 0.9635 |
| T+OM (PCR6+) | 0.9733 | 0.9814 | 0.9363 | 0.9693 | 0.9651 | 0.9641 |
| T+APP (PCR6+) | 0.9626 | 0.9815 | 0.9146 | 0.9681 | 0.9567 | 0.9553 |
| APP+OM (PCR6+) | 0.9674 | 0.9709 | 0.7481 | 0.8353 | 0.8804 | 0.8739 |
| T only | 0.9001 | 0.9086 | 0.9427 | 0.9656 | 0.9292 | 0.9310 |

### DF40 per generator, PCR6+-R against the baselines

| generator | family | source | **PCR6+-R** | PCR6-R | Dempster-R | Averaging-R |
|---|---|---|---:|---:|---:|---:|
| blendface | face swap | cdf | **0.9825** | 0.9822 | 0.9827 | 0.9828 |
| deepfacelab | face swap | cdf | **0.9637** | 0.9633 | 0.9654 | 0.9676 |
| e4s | face swap | cdf | **0.9817** | 0.9820 | 0.9841 | 0.9844 |
| facedancer | face swap | cdf | **0.9736** | 0.9742 | 0.9729 | 0.9733 |
| faceswap | face swap | cdf | **0.9993** | 0.9993 | 0.9994 | 0.9994 |
| fsgan | face swap | cdf | **0.9923** | 0.9925 | 0.9914 | 0.9924 |
| inswap | face swap | cdf | **0.9530** | 0.9532 | 0.9537 | 0.9546 |
| mobileswap | face swap | cdf | **0.9558** | 0.9574 | 0.9547 | 0.9548 |
| simswap | face swap | cdf | **0.9474** | 0.9477 | 0.9470 | 0.9474 |
| uniface | face swap | cdf | **0.9833** | 0.9832 | 0.9837 | 0.9844 |
| blendface | face swap | ff | **0.9844** | 0.9845 | 0.9852 | 0.9855 |
| e4s | face swap | ff | **0.9887** | 0.9891 | 0.9891 | 0.9886 |
| facedancer | face swap | ff | **0.9630** | 0.9639 | 0.9627 | 0.9648 |
| faceswap | face swap | ff | **0.9943** | 0.9939 | 0.9944 | 0.9939 |
| fsgan | face swap | ff | **0.9927** | 0.9931 | 0.9925 | 0.9931 |
| inswap | face swap | ff | **0.9693** | 0.9692 | 0.9699 | 0.9728 |
| mobileswap | face swap | ff | **0.9813** | 0.9813 | 0.9820 | 0.9833 |
| simswap | face swap | ff | **0.9705** | 0.9702 | 0.9712 | 0.9737 |
| uniface | face swap | ff | **0.9886** | 0.9891 | 0.9891 | 0.9897 |
| MRAA | reenactment | cdf | **0.9936** | 0.9935 | 0.9937 | 0.9937 |
| danet | reenactment | cdf | **0.9049** | 0.9021 | 0.8994 | 0.8944 |
| facevid2vid | reenactment | cdf | **0.8877** | 0.8832 | 0.8840 | 0.8857 |
| fomm | reenactment | cdf | **0.9518** | 0.9505 | 0.9517 | 0.9537 |
| hyperreenact | reenactment | cdf | **0.9648** | 0.9635 | 0.9622 | 0.9597 |
| lia | reenactment | cdf | **0.9956** | 0.9956 | 0.9955 | 0.9954 |
| mcnet | reenactment | cdf | **0.8979** | 0.8936 | 0.8935 | 0.8917 |
| one_shot_free | reenactment | cdf | **0.9319** | 0.9305 | 0.9322 | 0.9350 |
| pirender | reenactment | cdf | **0.8922** | 0.8910 | 0.8912 | 0.8945 |
| sadtalker | reenactment | cdf | **0.9080** | 0.9044 | 0.9018 | 0.8896 |
| tpsm | reenactment | cdf | **0.9706** | 0.9708 | 0.9675 | 0.9640 |
| MRAA | reenactment | ff | **0.9967** | 0.9966 | 0.9972 | 0.9975 |
| danet | reenactment | ff | **0.9681** | 0.9676 | 0.9669 | 0.9661 |
| facevid2vid | reenactment | ff | **0.9328** | 0.9303 | 0.9329 | 0.9343 |
| fomm | reenactment | ff | **0.9795** | 0.9790 | 0.9802 | 0.9822 |
| hyperreenact | reenactment | ff | **0.9751** | 0.9742 | 0.9749 | 0.9747 |
| lia | reenactment | ff | **0.9991** | 0.9991 | 0.9993 | 0.9993 |
| mcnet | reenactment | ff | **0.9642** | 0.9630 | 0.9633 | 0.9632 |
| one_shot_free | reenactment | ff | **0.9609** | 0.9602 | 0.9612 | 0.9626 |
| pirender | reenactment | ff | **0.9407** | 0.9391 | 0.9419 | 0.9456 |
| sadtalker | reenactment | ff | **0.9687** | 0.9684 | 0.9660 | 0.9625 |
| tpsm | reenactment | ff | **0.9768** | 0.9763 | 0.9764 | 0.9757 |

## Configuration `armB42`


### DF40, both required means

| arm | swap CDF | swap FF++ | reenact CDF | reenact FF++ | family-balanced (1/4 each) | generator-macro (1/41 each) |
|---|---:|---:|---:|---:|---:|---:|
| **PCR6P-R** | 0.9577 | 0.9771 | 0.9468 | 0.9723 | 0.9635 | 0.9630 |
| PCR6-R | 0.9576 | 0.9776 | 0.9454 | 0.9712 | 0.9630 | 0.9624 |
| PCR6-pairwise-R | 0.9577 | 0.9771 | 0.9468 | 0.9723 | 0.9635 | 0.9630 |
| Yager-R | 0.9572 | 0.9775 | 0.9439 | 0.9701 | 0.9622 | 0.9616 |
| Averaging-R *(baseline)* | 0.9602 | 0.9784 | 0.9445 | 0.9722 | 0.9638 | 0.9632 |
| Dempster-R *(baseline)* | 0.9589 | 0.9768 | 0.9484 | 0.9730 | 0.9643 | 0.9638 |
| T+OM (PCR6+) | 0.9577 | 0.9771 | 0.9468 | 0.9723 | 0.9635 | 0.9630 |
| T+APP (PCR6+) | 0.9575 | 0.9789 | 0.9441 | 0.9726 | 0.9633 | 0.9626 |
| APP+OM (PCR6+) | 0.9477 | 0.9585 | 0.8377 | 0.8839 | 0.9070 | 0.9035 |
| T only | 0.8857 | 0.9019 | 0.9355 | 0.9550 | 0.9195 | 0.9212 |

### DF40 per generator, PCR6+-R against the baselines

| generator | family | source | **PCR6+-R** | PCR6-R | Dempster-R | Averaging-R |
|---|---|---|---:|---:|---:|---:|
| blendface | face swap | cdf | **0.9680** | 0.9674 | 0.9693 | 0.9702 |
| deepfacelab | face swap | cdf | **0.9600** | 0.9596 | 0.9616 | 0.9632 |
| e4s | face swap | cdf | **0.9724** | 0.9725 | 0.9755 | 0.9778 |
| facedancer | face swap | cdf | **0.9719** | 0.9720 | 0.9738 | 0.9763 |
| faceswap | face swap | cdf | **0.9992** | 0.9992 | 0.9993 | 0.9992 |
| fsgan | face swap | cdf | **0.9884** | 0.9884 | 0.9886 | 0.9903 |
| inswap | face swap | cdf | **0.9316** | 0.9313 | 0.9327 | 0.9346 |
| mobileswap | face swap | cdf | **0.9324** | 0.9333 | 0.9326 | 0.9328 |
| simswap | face swap | cdf | **0.8913** | 0.8911 | 0.8924 | 0.8930 |
| uniface | face swap | cdf | **0.9620** | 0.9616 | 0.9629 | 0.9642 |
| blendface | face swap | ff | **0.9837** | 0.9836 | 0.9844 | 0.9840 |
| e4s | face swap | ff | **0.9882** | 0.9887 | 0.9874 | 0.9877 |
| facedancer | face swap | ff | **0.9579** | 0.9593 | 0.9570 | 0.9596 |
| faceswap | face swap | ff | **0.9948** | 0.9951 | 0.9948 | 0.9942 |
| fsgan | face swap | ff | **0.9914** | 0.9923 | 0.9902 | 0.9918 |
| inswap | face swap | ff | **0.9632** | 0.9622 | 0.9634 | 0.9675 |
| mobileswap | face swap | ff | **0.9784** | 0.9784 | 0.9785 | 0.9812 |
| simswap | face swap | ff | **0.9534** | 0.9538 | 0.9523 | 0.9552 |
| uniface | face swap | ff | **0.9833** | 0.9847 | 0.9830 | 0.9844 |
| MRAA | reenactment | cdf | **0.9867** | 0.9866 | 0.9873 | 0.9877 |
| danet | reenactment | cdf | **0.9246** | 0.9213 | 0.9277 | 0.9206 |
| facevid2vid | reenactment | cdf | **0.9207** | 0.9193 | 0.9223 | 0.9224 |
| fomm | reenactment | cdf | **0.9544** | 0.9536 | 0.9555 | 0.9578 |
| hyperreenact | reenactment | cdf | **0.9813** | 0.9807 | 0.9824 | 0.9806 |
| lia | reenactment | cdf | **0.9896** | 0.9894 | 0.9901 | 0.9896 |
| mcnet | reenactment | cdf | **0.9306** | 0.9281 | 0.9334 | 0.9307 |
| one_shot_free | reenactment | cdf | **0.9451** | 0.9448 | 0.9463 | 0.9476 |
| pirender | reenactment | cdf | **0.8937** | 0.8930 | 0.8947 | 0.8962 |
| sadtalker | reenactment | cdf | **0.9124** | 0.9076 | 0.9155 | 0.8819 |
| tpsm | reenactment | cdf | **0.9755** | 0.9751 | 0.9776 | 0.9745 |
| MRAA | reenactment | ff | **0.9981** | 0.9980 | 0.9984 | 0.9986 |
| danet | reenactment | ff | **0.9718** | 0.9705 | 0.9728 | 0.9710 |
| facevid2vid | reenactment | ff | **0.9391** | 0.9377 | 0.9404 | 0.9419 |
| fomm | reenactment | ff | **0.9811** | 0.9809 | 0.9813 | 0.9830 |
| hyperreenact | reenactment | ff | **0.9917** | 0.9910 | 0.9921 | 0.9916 |
| lia | reenactment | ff | **0.9994** | 0.9994 | 0.9994 | 0.9994 |
| mcnet | reenactment | ff | **0.9746** | 0.9738 | 0.9761 | 0.9765 |
| one_shot_free | reenactment | ff | **0.9532** | 0.9518 | 0.9544 | 0.9572 |
| pirender | reenactment | ff | **0.9359** | 0.9332 | 0.9376 | 0.9409 |
| sadtalker | reenactment | ff | **0.9659** | 0.9640 | 0.9655 | 0.9514 |
| tpsm | reenactment | ff | **0.9841** | 0.9832 | 0.9851 | 0.9831 |

## Configuration `armB1337`


### DF40, both required means

| arm | swap CDF | swap FF++ | reenact CDF | reenact FF++ | family-balanced (1/4 each) | generator-macro (1/41 each) |
|---|---:|---:|---:|---:|---:|---:|
| **PCR6P-R** | 0.9687 | 0.9804 | 0.9503 | 0.9767 | 0.9690 | 0.9685 |
| PCR6-R | 0.9684 | 0.9803 | 0.9494 | 0.9764 | 0.9686 | 0.9681 |
| PCR6-pairwise-R | 0.9690 | 0.9809 | 0.9495 | 0.9765 | 0.9690 | 0.9684 |
| Yager-R | 0.9673 | 0.9790 | 0.9505 | 0.9759 | 0.9682 | 0.9677 |
| Averaging-R *(baseline)* | 0.9695 | 0.9807 | 0.9513 | 0.9775 | 0.9698 | 0.9692 |
| Dempster-R *(baseline)* | 0.9697 | 0.9816 | 0.9505 | 0.9775 | 0.9698 | 0.9692 |
| T+OM (PCR6+) | 0.9665 | 0.9780 | 0.9540 | 0.9770 | 0.9689 | 0.9685 |
| T+APP (PCR6+) | 0.9672 | 0.9795 | 0.9529 | 0.9774 | 0.9692 | 0.9688 |
| APP+OM (PCR6+) | 0.9583 | 0.9620 | 0.8451 | 0.8870 | 0.9131 | 0.9096 |
| T only | 0.9001 | 0.9086 | 0.9427 | 0.9656 | 0.9292 | 0.9310 |

### DF40 per generator, PCR6+-R against the baselines

| generator | family | source | **PCR6+-R** | PCR6-R | Dempster-R | Averaging-R |
|---|---|---|---:|---:|---:|---:|
| blendface | face swap | cdf | **0.9786** | 0.9782 | 0.9795 | 0.9795 |
| deepfacelab | face swap | cdf | **0.9627** | 0.9627 | 0.9653 | 0.9669 |
| e4s | face swap | cdf | **0.9848** | 0.9845 | 0.9868 | 0.9840 |
| facedancer | face swap | cdf | **0.9833** | 0.9831 | 0.9835 | 0.9827 |
| faceswap | face swap | cdf | **0.9992** | 0.9992 | 0.9993 | 0.9993 |
| fsgan | face swap | cdf | **0.9936** | 0.9935 | 0.9933 | 0.9934 |
| inswap | face swap | cdf | **0.9394** | 0.9389 | 0.9410 | 0.9422 |
| mobileswap | face swap | cdf | **0.9391** | 0.9389 | 0.9397 | 0.9378 |
| simswap | face swap | cdf | **0.9267** | 0.9261 | 0.9282 | 0.9285 |
| uniface | face swap | cdf | **0.9796** | 0.9794 | 0.9805 | 0.9808 |
| blendface | face swap | ff | **0.9843** | 0.9844 | 0.9858 | 0.9848 |
| e4s | face swap | ff | **0.9887** | 0.9886 | 0.9897 | 0.9864 |
| facedancer | face swap | ff | **0.9616** | 0.9610 | 0.9626 | 0.9621 |
| faceswap | face swap | ff | **0.9929** | 0.9928 | 0.9928 | 0.9925 |
| fsgan | face swap | ff | **0.9921** | 0.9920 | 0.9920 | 0.9920 |
| inswap | face swap | ff | **0.9640** | 0.9638 | 0.9659 | 0.9660 |
| mobileswap | face swap | ff | **0.9803** | 0.9802 | 0.9820 | 0.9816 |
| simswap | face swap | ff | **0.9713** | 0.9713 | 0.9736 | 0.9717 |
| uniface | face swap | ff | **0.9885** | 0.9885 | 0.9897 | 0.9888 |
| MRAA | reenactment | cdf | **0.9915** | 0.9913 | 0.9919 | 0.9915 |
| danet | reenactment | cdf | **0.9285** | 0.9272 | 0.9289 | 0.9316 |
| facevid2vid | reenactment | cdf | **0.9328** | 0.9316 | 0.9354 | 0.9370 |
| fomm | reenactment | cdf | **0.9627** | 0.9622 | 0.9641 | 0.9654 |
| hyperreenact | reenactment | cdf | **0.9799** | 0.9794 | 0.9800 | 0.9801 |
| lia | reenactment | cdf | **0.9908** | 0.9907 | 0.9912 | 0.9908 |
| mcnet | reenactment | cdf | **0.9308** | 0.9293 | 0.9325 | 0.9354 |
| one_shot_free | reenactment | cdf | **0.9500** | 0.9491 | 0.9516 | 0.9531 |
| pirender | reenactment | cdf | **0.9102** | 0.9091 | 0.9122 | 0.9144 |
| sadtalker | reenactment | cdf | **0.9006** | 0.8986 | 0.8955 | 0.8915 |
| tpsm | reenactment | cdf | **0.9753** | 0.9751 | 0.9720 | 0.9735 |
| MRAA | reenactment | ff | **0.9973** | 0.9972 | 0.9976 | 0.9975 |
| danet | reenactment | ff | **0.9738** | 0.9736 | 0.9745 | 0.9758 |
| facevid2vid | reenactment | ff | **0.9571** | 0.9568 | 0.9591 | 0.9585 |
| fomm | reenactment | ff | **0.9860** | 0.9856 | 0.9870 | 0.9871 |
| hyperreenact | reenactment | ff | **0.9844** | 0.9839 | 0.9857 | 0.9859 |
| lia | reenactment | ff | **0.9997** | 0.9997 | 0.9997 | 0.9996 |
| mcnet | reenactment | ff | **0.9753** | 0.9751 | 0.9765 | 0.9774 |
| one_shot_free | reenactment | ff | **0.9656** | 0.9651 | 0.9667 | 0.9666 |
| pirender | reenactment | ff | **0.9499** | 0.9489 | 0.9524 | 0.9530 |
| sadtalker | reenactment | ff | **0.9710** | 0.9709 | 0.9684 | 0.9670 |
| tpsm | reenactment | ff | **0.9835** | 0.9834 | 0.9847 | 0.9844 |

## END SOURCE: `tbiom/reports/EXP_PCR6P_EXPOSED.md`


--------------------------------------------------------------------------------------------------------------
## BEGIN SOURCE: `tbiom/reports/PCR6P_OUTCOME.md`

*Role in this handoff:* Pre-registered acceptance criteria applied; terminal outcome.

--------------------------------------------------------------------------------------------------------------

# PCR6+ outcome

Section 9 of `tbiom/docs/pcr6plus-fusion-brief.md`. Pre-registration: `tbiom/configs/pcr6p_fusion_preregister.yaml`, written before any candidate was scored.

## Terminal outcome: 4. **no demonstrated benefit**

AUROC noninferiority holds in **4 of 4** configurations; the calibrated operating-point safeguards hold in **1 of 4**; the conflict-by-source score beats every simple score in **1 of 4**. A benefit appearing in one configuration of four is reported as what it is — the configuration that happened to favour it — and not as a demonstrated benefit.

Outcome 6, independent confirmation pending, **also applies and applies unconditionally**: no untouched set remains. CDFv2 and DeepfakeEval2024 informed development, and DF40 has informed operator design in three consecutive studies.

## 1. Detection — the declared trade

| configuration | Δ v3.1 clean vs Dempster-R (95% CI) | noninferior (lb > −0.005) | Δ JPEG q40 | Δ resize 0.60 | Δ reenactment | FPR rise | TPR loss |
|---|---|---|---:|---:|---:|---:|---:|
| `42` | -0.0014 [lb -0.0040] | **yes** | +0.0008 | -0.0008 | +0.0025 | -0.003 ok | +0.002 ok |
| `T1337` | +0.0002 [lb -0.0016] | **yes** | +0.0014 | -0.0000 | +0.0010 | +0.013 **breach** | -0.034 ok |
| `armB42` | +0.0004 [lb -0.0010] | **yes** | -0.0008 | +0.0002 | -0.0009 | -0.004 ok | +0.029 **breach** |
| `armB1337` | -0.0008 [lb -0.0025] | **yes** | -0.0008 | -0.0013 | +0.0014 | -0.004 ok | +0.015 **breach** |

## 2. Reliability — the requirement for any uncertainty claim

The conflict-by-source score had to beat **each** of fused vacuity, total conflict K, predictive entropy and margin on development error-detection AUROC, with the paired interval excluding zero.

| configuration | conflict-by-source | vacuity | K | entropy | margin | beats all four | E-AURC vs vacuity |
|---|---:|---:|---:|---:|---:|---|---|
| `42` | **0.7432** | 0.7343 | 0.6727 | 0.7508 | 0.7508 | **no** | 0.06662 vs 0.07190 (improves) |
| `T1337` | **0.8047** | 0.7502 | 0.7062 | 0.8087 | 0.8087 | **no** | 0.05134 vs 0.07807 (improves) |
| `armB42` | **0.7089** | 0.7190 | 0.7107 | 0.7794 | 0.7794 | **no** | 0.10463 vs 0.10091 (worse) |
| `armB1337` | **0.8409** | 0.7398 | 0.7535 | 0.8311 | 0.8311 | yes | 0.05283 vs 0.09959 (improves) |

Entropy and margin are numerically identical as error rankers here, and necessarily so: on a binary frame both are strictly monotone functions of |p − 0.5|, so they induce the same ordering. They are reported separately because the pre-registration named both.

**The requirement is not met.** The conflict-by-source model is beaten by entropy in three of the four configurations. It does improve selective prediction over fused vacuity (lower E-AURC in three of four), but the pre-registration required it to beat every simple score, not the weakest one, precisely so that this kind of partial result could not be presented as a benefit.

## 2b. Why it fails — the research question, answered

Section 1 of the brief asked whether conflict attributed to its source views predicts fused errors better than vacuity, entropy or margin. The conflict case analysis on development answers it directly, and the answer is no, for a reason that is visible rather than statistical.

| case (configuration `42`) | videos | K_TS | K_AO | K_all | PCR6+-R correct |
|---|---:|---:|---:|---:|---:|
| temporal right, both spatial wrong | 362 | 0.4194 | 0.0072 | 0.0142 | 0.6552 |
| temporal wrong, both spatial right | 520 | 0.3740 | 0.0076 | 0.0106 | 0.6472 |
| all three agree and right | 1512 | 0.0324 | 0.0038 | 0.0095 | 1.0000 |
| all three wrong | 214 | 0.0439 | 0.0092 | 0.0135 | 0.0018 |

Temporal-versus-spatial conflict is high in **both** disagreement cases — 0.4194 when the temporal view is the one that is right, and 0.3740 when it is the one that is wrong. The quantity marks that the views disagree; it carries almost no information about *which* of them to believe, and a score built on it inherits that. Predictive entropy already captures the part that does predict errors — how confidently the fusion committed — which is why entropy wins in three of the four configurations.

K_AO and K_all are an order of magnitude smaller than K_TS in every case, so the decomposition by source pair adds little beyond the total: on this system essentially all conflict is temporal-versus-spatial.

## 3. What the PCR6+ correction changes

| configuration | Δ v3.1 (PCR6+ − PCR6) | Δ DF40 family-balanced | Δ DF40 reenactment, Celeb-DF |
|---|---:|---:|---:|
| `42` | +0.0001 | +0.0020 | +0.0070 |
| `T1337` | -0.0000 | +0.0006 | +0.0019 |
| `armB42` | +0.0001 | +0.0005 | +0.0014 |
| `armB1337` | +0.0004 | +0.0004 | +0.0009 |

The correction is not cosmetic on this data. It is worth stating why it can matter here at all: the reliability discount routinely drives a view's weight to zero, and a zero-weighted view is exactly the vacuous source whose neutrality PCR6 violates and PCR6+ restores. The largest effect is on the reenactment half, where the appearance view is the one being discounted away.

## 4. Decisions

1. **Dempster-R remains a baseline row only.** It is the DS standard and the DiCoME-family reference. It is not the proposed method and is not reinstated as one by this study's outcome.

2. **No uncertainty novelty claim is made.** Conflict attribution by source pair failed its pre-registered test. It would in any case have been a narrow application claim: Discounted Belief Fusion (AISTATS 2025) already decomposes conflict per source pair, and uses it for more — to discount the fusion itself, not merely to rank errors (`PCR6P_PRIOR_ART_AND_CLAIMS.md`).

3. **No fusion-operator novelty claim is made.** PCR6+ is Dezert, Dezert and Smarandache's rule and is credited as theirs throughout. What this study contributes is the application and the measurement.

4. A further honest caution for the write-up: our `Averaging-R` baseline is essentially the conflictive opinion aggregation of Xu et al. (AAAI 2024), which is a published method rather than a naive control, and it is the strongest arm on several cells.

## END SOURCE: `tbiom/reports/PCR6P_OUTCOME.md`


--------------------------------------------------------------------------------------------------------------
## BEGIN SOURCE: `tbiom/reports/PCR6P_PRIOR_ART_AND_CLAIMS.md`

*Role in this handoff:* Prior art with primary sources read, and the claim-to-evidence map separating established operators from our implementation choices and measured findings.

--------------------------------------------------------------------------------------------------------------

# PCR6+ prior art and claim-to-evidence map

Section 8 of `tbiom/docs/pcr6plus-fusion-brief.md`.

## 1. Availability statement

Unlike the grouped-fusion study, **primary sources were obtained and read for this one.** Each row below says which.
Papers downloaded during this task, with the sha256 of the exact file read:

| work | file | sha256 (32) |
|---|---|---|
| Dezert, Dezert & Smarandache 2021 (PCR5+/PCR6+) | https://isif.org/files/isif/2024-01/last.pdf | `d87ae493e7c409d1b194ed701b6fe746` |
| Bezirganyan et al., Discounted Belief Fusion, AISTATS 2025 | https://arxiv.org/pdf/2412.18024 | `0d6ab4fd233747b9dceb05f6872b12a0` |
| Xu et al., Reliable Conflictive Multi-View Learning, AAAI 2024 | https://ojs.aaai.org/index.php/AAAI/article/download/29546/30911 | read, text extracted |

## 2. Comparison

| work | what it establishes | relation to this study | status |
|---|---|---|---|
| **PCR5/PCR6** (Smarandache & Dezert; Martin & Osswald) | conflicting mass redistributed proportionally to the sources that produced it; PCR6 = PCR5 for two sources; neither is associative | `PCR6-R` is the direct S-source rule, eq. (15). The historical rows in earlier reports were a pairwise fold and are relabelled `PCR6-pairwise-R`; the two differ by up to 1.8e-01 | **read** (eq. 9-15 of the 2021 paper) |
| **PCR6+** (T. Dezert, J. Dezert, Smarandache, JAIF 16(1) 2021, 48-73) | binary keeping indexes discard focal elements that contain all others in a partial conflict, restoring neutrality of the vacuous BBA | **This is the operator we evaluate.** It is established prior work and is credited as such; no novelty is claimed for it. Our contribution is limited to applying it to this three-view detector and reporting what it does | **read**, eq. (15), (22)-(24), (26) implemented directly |
| **Dempster's rule** | conjunctive, conflict normalised away; associative | mandatory baseline row, labelled as the DS standard and the DiCoME-family reference. Verified associative in our implementation (direct vs fold: 7.8e-16) | **read** (via the 2021 paper's restatement) |
| **Yager's rule** | conflict assigned to Theta rather than redistributed | comparator arm. Not associative — implemented as a direct S-source rule, not a fold (direct vs fold: 5.3e-01) | **read** (restatement) |
| **Reliable Conflictive Multi-View Learning / ECML** (Xu et al., AAAI 2024) | conflictive opinion aggregation implemented as an **average pooling layer** over view opinions, with a proof that fused uncertainty increases for conflictive instances | **Our `Averaging-R` arm is essentially this rule.** It is therefore not merely a "simple control" but a published method, and it is the strongest arm on several of our cells. Any claim for PCR6+-R must clear it, which the pre-registration already requires | **read** |
| **Discounted Belief Fusion** (Bezirganyan et al., AISTATS 2025) | pairwise degree of conflict DC = PD x CC (projected-probability distance times conjunctive certainty), collected in an agreement matrix, turned into a **per-modality discount** with strictness parameter lambda, then order-invariant fusion | **Closest prior art to the conflict-by-source idea.** They decompose conflict *per pair of sources* and use it to discount. The difference in this study is only where the quantity is used: DBF changes the fusion; we leave the decision untouched and use conflict shares purely as an error-ranking score. That is a narrower use, and a weaker claim | **read** |
| **DiCoME** | two-view evidential detector combined with Dempster's rule | baseline row; our own reproduction is in the repo | partially verified (our training; not the published description) |
| **DISCERN confidence-modulated fusion, IBDC** | this project's earlier fusion work | not used here; no auxiliary loss or extra branch is added | in-repo |

## 3. Claim-to-evidence map

| category | item | evidence |
|---|---|---|
| **Established operator, credited** | PCR6+ and its neutrality property | Dezert et al. 2021, eq. (26). Reproduced on our frame: `EXP_PCR6P_AUDIT.md` §3 |
| **Our implementation choice** | applying PCR6+ on a binary frame, where the keeping indexes reduce to kappa(Theta)=0 | derived from eq. (22)-(24), verified in 17/17 unit tests |
| **Our implementation choice** | direct S-source Yager rather than a pairwise fold | associativity check in this study |
| **Measured finding** | under evidence scaling, a view discounted to r=0 leaves PCR6+ exactly unchanged (0.00e+00) but shifts PCR6 by up to 2.8e-01 | `EXP_PCR6P_AUDIT.md` §3 |
| **Measured finding** | historical `PCR6-R` rows were a non-associative fold, not the direct rule | `EXP_PCR6P_AUDIT.md` §5 |
| **Measured finding** | detection and reliability outcomes | `EXP_PCR6P_DEVELOPMENT.md`, `EXP_PCR6P_EXPOSED.md`, summarised in `PCR6P_OUTCOME.md` |
| **Unverified hypothesis** | that conflict attributed by source pair is a better error ranker than vacuity, entropy or margin | tested and **not supported** — see the outcome report. DBF already uses pairwise conflict for discounting, so even a positive result would have been a narrow application claim |

## 4. Provenance

No prior report was deleted, renamed or relabelled. The historical `PCR6-R` rows are relabelled only in tables
produced by this study; `EXP_FUSION_OPERATORS.md`, `EXP_DF40_OPERATORS.md` and the grouped-fusion reports are
unchanged.

## END SOURCE: `tbiom/reports/PCR6P_PRIOR_ART_AND_CLAIMS.md`


--------------------------------------------------------------------------------------------------------------
## BEGIN SOURCE: `tbiom/reports/EXP_DF40_CROSS_MANIPULATION.md`

*Role in this handoff:* Per-generator cross-manipulation against our trained DiCoME.

--------------------------------------------------------------------------------------------------------------

# Cross-manipulation results — DF40 face-swap generators

OMP_NUM_THREADS=8 python analysis/tbiom/df40_analysis.py

Each generator is scored against the genuine pool DF40 ships for its source. Our system fuses the three reference
views with DS-R, the weights fitted per split on the development folds exactly as everywhere else, so the numbers are
comparable with every other table; the logistic comparator is reported beside it. DiCoME is our own trained
checkpoint (the same one in the baseline matrix), replayed through its own two-view DS fusion, so both sides are
scored on identical videos with identical aggregation.

Two generators DiCoME's published cross-manipulation table does not list — the classical graphics **FaceSwap** and
**DeepFaceLab** — are included and marked, together with the FF++-sourced version of every generator.

Every number below uses the reference manifest `tbiom/reports/REFERENCE_MANIFEST.md` (three resolved checkpoints, DS-R definition, fold splits). Script `analysis/tbiom/temporal/finish_line.py`.

## Face swap — Celeb-DF-sourced generators

| generator | videos (real / fake) | **ours, DS-R** | ours, logistic | DiCoME (our training) | Δ ours − DiCoME | temporal view | appearance | off-manifold |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BlendFace | 178 / 640 | **0.9801** | 0.9757 | 0.9502 | +0.0299 | 0.9371 | 0.9444 | 0.9516 |
| DeepFaceLab ★ | 49 / 100 | **0.9610** | 0.9302 | 0.9390 | +0.0220 | 0.9161 | 0.9349 | 0.9378 |
| e4s | 178 / 292 | **0.9892** | 0.9920 | 0.9949 | -0.0057 | 0.8055 | 0.9931 | 0.9939 |
| FaceDancer | 178 / 648 | **0.9714** | 0.9809 | 0.9657 | +0.0058 | 0.8598 | 0.9553 | 0.9744 |
| FaceSwap ★ | 178 / 637 | **0.9995** | 0.9928 | 0.9991 | +0.0004 | 0.9981 | 0.9997 | 0.9974 |
| FSGAN | 178 / 616 | **0.9906** | 0.9817 | 0.9892 | +0.0014 | 0.9393 | 0.9782 | 0.9938 |
| InSwapper | 178 / 424 | **0.9551** | 0.9526 | 0.9433 | +0.0117 | 0.8872 | 0.9398 | 0.9429 |
| MobileFaceSwap | 178 / 637 | **0.9584** | 0.9570 | 0.9571 | +0.0013 | 0.7988 | 0.9454 | 0.9641 |
| SimSwap | 178 / 641 | **0.9416** | 0.9463 | 0.9339 | +0.0077 | 0.8006 | 0.9247 | 0.9380 |
| UniFace | 178 / 626 | **0.9837** | 0.9806 | 0.9758 | +0.0079 | 0.9137 | 0.9727 | 0.9747 |
| **mean over 10 generators** | | **0.9731** | 0.9690 | 0.9648 | +0.0082 | 0.8856 | 0.9588 | 0.9668 |

## Face swap — FaceForensics++-sourced generators

| generator | videos (real / fake) | **ours, DS-R** | ours, logistic | DiCoME (our training) | Δ ours − DiCoME | temporal view | appearance | off-manifold |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BlendFace | 140 / 140 | **0.9858** | 0.9795 | 0.9505 | +0.0353 | 0.9490 | 0.9499 | 0.9470 |
| e4s | 140 / 115 | **0.9927** | 0.9902 | 0.9891 | +0.0037 | 0.8081 | 0.9892 | 0.9881 |
| FaceDancer | 140 / 140 | **0.9713** | 0.9673 | 0.9632 | +0.0081 | 0.8330 | 0.9629 | 0.9616 |
| FaceSwap ★ | 140 / 140 | **0.9948** | 0.9923 | 0.9795 | +0.0153 | 0.9856 | 0.9800 | 0.9782 |
| FSGAN | 140 / 140 | **0.9935** | 0.9915 | 0.9879 | +0.0056 | 0.9453 | 0.9868 | 0.9879 |
| InSwapper | 140 / 118 | **0.9783** | 0.9685 | 0.9696 | +0.0087 | 0.8993 | 0.9706 | 0.9654 |
| MobileFaceSwap | 140 / 858 | **0.9848** | 0.9790 | 0.9745 | +0.0103 | 0.9165 | 0.9734 | 0.9740 |
| SimSwap | 140 / 140 | **0.9749** | 0.9648 | 0.9539 | +0.0210 | 0.8577 | 0.9551 | 0.9500 |
| UniFace | 140 / 140 | **0.9928** | 0.9908 | 0.9803 | +0.0125 | 0.9228 | 0.9804 | 0.9783 |
| **mean over 9 generators** | | **0.9854** | 0.9804 | 0.9720 | +0.0134 | 0.9019 | 0.9720 | 0.9701 |

## Reenactment — Celeb-DF-sourced generators

| generator | videos (real / fake) | **ours, DS-R** | ours, logistic | DiCoME (our training) | Δ ours − DiCoME | temporal view | appearance | off-manifold |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MRAA | 178 / 649 | **0.9934** | 0.9930 | 0.9770 | +0.0164 | 0.9724 | 0.9738 | 0.9780 |
| DaNet | 178 / 647 | **0.8645** | 0.9083 | 0.4821 | +0.3823 | 0.9522 | 0.4599 | 0.5289 |
| FaceVid2Vid | 178 / 637 | **0.8403** | 0.8966 | 0.6748 | +0.1655 | 0.8899 | 0.6505 | 0.7141 |
| FOMM | 178 / 647 | **0.9341** | 0.9260 | 0.8793 | +0.0547 | 0.9104 | 0.8673 | 0.8870 |
| HyperReenact | 178 / 641 | **0.9708** | 0.9786 | 0.7421 | +0.2287 | 0.9865 | 0.7253 | 0.7674 |
| LIA | 178 / 646 | **0.9977** | 0.9981 | 0.9854 | +0.0123 | 0.9818 | 0.9793 | 0.9873 |
| MCNet | 178 / 647 | **0.8662** | 0.9077 | 0.5594 | +0.3068 | 0.9411 | 0.5371 | 0.5992 |
| One-Shot Free-View | 178 / 647 | **0.9102** | 0.9031 | 0.8506 | +0.0596 | 0.8878 | 0.8410 | 0.8570 |
| PIRender | 178 / 636 | **0.8626** | 0.8766 | 0.8066 | +0.0559 | 0.8133 | 0.7855 | 0.8287 |
| SadTalker | 178 / 660 | **0.8857** | 0.8801 | 0.5402 | +0.3455 | 0.9629 | 0.5531 | 0.5159 |
| TPSM | 178 / 647 | **0.9661** | 0.9690 | 0.5097 | +0.4564 | 0.9915 | 0.5080 | 0.5191 |
| **mean over 11 generators** | | **0.9174** | 0.9306 | 0.7280 | +0.1895 | 0.9354 | 0.7164 | 0.7439 |

## Reenactment — FaceForensics++-sourced generators

| generator | videos (real / fake) | **ours, DS-R** | ours, logistic | DiCoME (our training) | Δ ours − DiCoME | temporal view | appearance | off-manifold |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MRAA | 140 / 140 | **0.9993** | 0.9984 | 0.9937 | +0.0056 | 0.9807 | 0.9938 | 0.9930 |
| DaNet | 140 / 140 | **0.9546** | 0.9496 | 0.6764 | +0.2782 | 0.9726 | 0.6761 | 0.6855 |
| FaceVid2Vid | 140 / 140 | **0.9058** | 0.9121 | 0.7776 | +0.1282 | 0.9119 | 0.7778 | 0.7818 |
| FOMM | 140 / 140 | **0.9792** | 0.9730 | 0.9609 | +0.0183 | 0.9363 | 0.9624 | 0.9552 |
| HyperReenact | 140 / 140 | **0.9827** | 0.9835 | 0.8483 | +0.1343 | 0.9818 | 0.8499 | 0.8462 |
| LIA | 140 / 140 | **0.9999** | 1.0000 | 0.9921 | +0.0079 | 0.9929 | 0.9923 | 0.9913 |
| MCNet | 140 / 140 | **0.9566** | 0.9540 | 0.7680 | +0.1886 | 0.9677 | 0.7703 | 0.7706 |
| One-Shot Free-View | 140 / 140 | **0.9442** | 0.9354 | 0.8815 | +0.0627 | 0.9274 | 0.8843 | 0.8733 |
| PIRender | 140 / 140 | **0.9274** | 0.9089 | 0.8833 | +0.0441 | 0.8840 | 0.8861 | 0.8734 |
| SadTalker | 140 / 138 | **0.9503** | 0.9445 | 0.6071 | +0.3431 | 0.9731 | 0.6169 | 0.5798 |
| TPSM | 140 / 140 | **0.9677** | 0.9705 | 0.7458 | +0.2219 | 0.9767 | 0.7467 | 0.7491 |
| **mean over 11 generators** | | **0.9607** | 0.9573 | 0.8304 | +0.1303 | 0.9550 | 0.8324 | 0.8272 |

★ = not listed in DiCoME's published cross-manipulation table. Every generator here is scored against the genuine pool DF40 ships for its source (`DF40-cdf-real`, `DF40-ff-real`, and DeepFaceLab's own pool), so a generator's AUROC is never computed against another source's genuine videos.

Our rows are means over the five fitted splits (the fusion weights differ per split); DiCoME is a single checkpoint, so its rows carry no spread. Both sides use per-frame opinions averaged per video and then fused — the same aggregation as every other table. DF40 was never trained on by either system; its generators are unseen manipulations built on FF++ and Celeb-DF source videos, which both systems have seen as training or development content.

One detector seed and one temporal-branch seed unless stated. CDFv2 and DeepfakeEval2024 informed development and are not untouched confirmation data. FaceShifter is sealed (`tbiom/docs/SEALED_BENCHMARKS.md`).

## END SOURCE: `tbiom/reports/EXP_DF40_CROSS_MANIPULATION.md`


--------------------------------------------------------------------------------------------------------------
## BEGIN SOURCE: `tbiom/reports/BASELINE_MATRIX.md`

*Role in this handoff:* Comparator provenance and per-benchmark values, all reproduced locally.

--------------------------------------------------------------------------------------------------------------

# Baseline matrix — frame- and video-level AUROC

    python analysis/tbiom/baselines/build_baseline_matrix.py --out tbiom/reports/BASELINE_MATRIX.md

Every measured row scores the SAME frames per dataset: CDFv2/CDFv3/DFD/DFDC/DFDCP/DFEval24 use our crops (the
baselines read PNGs dumped from our h5 files); FFIW/KoDF/IDForge/DFDM use the GenD-release crops our h5 files were
built from. Scope filters as protocol v3.1: CDFv3 without IP_LAP; KoDF without audio-driven and adversarial subsets.
Our models: DS replay of their own views (frame = per-frame opinion; video = mean opinion per video, then fused —
the corrected-table rule). Baselines: prob_fake per frame; video = mean over frames (GenD's own reduction).
A video is the directory containing the frame (full path). Published numbers are listed separately, never mixed in.

All detectors trained on FF++ c23. One seed per row. — = not yet available.

## Video-level AUROC

| method | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | DFEval24 | FFIW | KoDF | IDForge | DFDM | mean (complete rows) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 (v3.1, ep 6) | 0.9509 | 0.8954 | 0.9431 | 0.8879 | 0.9064 | 0.6869 | 0.9036 | 0.8198 | 0.9723 | 0.9929 | 0.8959 |
| A (v3.1, ep 5) | 0.9388 | 0.8770 | 0.9461 | 0.8889 | 0.9013 | 0.6852 | 0.9011 | 0.8242 | 0.9697 | 0.9915 | 0.8924 |
| B (v3.1, ep 8) | 0.9225 | 0.8801 | 0.9387 | 0.8889 | 0.9057 | 0.6943 | 0.8906 | 0.8497 | 0.9553 | 0.9944 | 0.8920 |
| C (v3.1, ep 2) | 0.9183 | 0.6648 | 0.8802 | 0.8763 | 0.8565 | 0.6989 | 0.9279 | 0.8548 | 0.9492 | 0.9868 | 0.8614 |
| DiCoME (our training, v3.1, ep 5) | 0.9693 | 0.8691 | 0.9343 | 0.8818 | 0.8755 | 0.6769 | 0.9284 | 0.8509 | 0.9598 | 0.9909 | 0.8937 |
| GenD-CLIP | 0.9534 | 0.8458 | 0.9350 | 0.8708 | 0.9162 | 0.6831 | 0.9278 | 0.8329 | 0.9839 | 0.9959 | 0.8945 |
| GenD-PE | 0.9548 | 0.8952 | 0.9203 | 0.8158 | 0.8860 | 0.6563 | 0.9330 | 0.8189 | 0.9808 | 0.9815 | 0.8842 |
| GenD-DINO | 0.9177 | 0.8172 | 0.9165 | 0.8470 | 0.9114 | 0.6536 | 0.9450 | 0.8984 | 0.9860 | 0.9903 | 0.8883 |
| Effort | 0.9312 | 0.7975 | 0.9276 | 0.8536 | 0.8617 | 0.6769 | 0.9252 | 0.8725 | 0.9583 | 0.9729 | 0.8777 |
| ForAda | 0.9544 | 0.7494 | 0.9395 | 0.8726 | 0.9289 | 0.6227 | 0.9055 | 0.8739 | 0.9101 | 0.9713 | 0.8728 |
| FSFM | 0.9554 | 0.8751 | 0.9219 | 0.8844 | 0.9061 | 0.6817 | 0.9168 | 0.8467 | 0.9566 | 0.9790 | 0.8924 |
| FSFM-Adapter | 0.8689 | 0.8381 | 0.7516 | 0.8318 | 0.8486 | 0.6280 | 0.8565 | 0.7405 | 0.9077 | 0.8452 | 0.8117 |

## Frame-level AUROC

| method | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | DFEval24 | FFIW | KoDF | IDForge | DFDM | mean (complete rows) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 (v3.1, ep 6) | 0.9019 | 0.8545 | 0.9032 | 0.8693 | 0.8680 | 0.6606 | 0.8585 | 0.8109 | 0.9360 | 0.9797 | 0.8643 |
| A (v3.1, ep 5) | 0.8854 | 0.8321 | 0.9052 | 0.8703 | 0.8665 | 0.6564 | 0.8582 | 0.8184 | 0.9315 | 0.9756 | 0.8600 |
| B (v3.1, ep 8) | 0.8746 | 0.8428 | 0.8986 | 0.8695 | 0.8712 | 0.6637 | 0.8450 | 0.8483 | 0.9261 | 0.9854 | 0.8625 |
| C (v3.1, ep 2) | 0.8437 | 0.6638 | 0.8401 | 0.8578 | 0.8243 | 0.6506 | 0.8800 | 0.8373 | 0.9049 | 0.9700 | 0.8273 |
| DiCoME (our training, v3.1, ep 5) | 0.9183 | 0.8177 | 0.9025 | 0.8538 | 0.8482 | 0.6377 | 0.8841 | 0.8422 | 0.9120 | 0.9743 | 0.8591 |
| GenD-CLIP | 0.8942 | 0.7944 | 0.8931 | 0.8332 | 0.8637 | 0.6369 | 0.8706 | 0.8201 | 0.9494 | 0.9744 | 0.8530 |
| GenD-PE | 0.8986 | 0.8500 | 0.8801 | 0.7990 | 0.8418 | 0.6264 | 0.8906 | 0.8276 | 0.9543 | 0.9563 | 0.8524 |
| GenD-DINO | 0.8457 | 0.7653 | 0.8677 | 0.8093 | 0.8469 | 0.6206 | 0.8878 | 0.8869 | 0.9528 | 0.9595 | 0.8442 |
| Effort | 0.8874 | 0.7625 | 0.8915 | 0.8276 | 0.8405 | 0.6352 | 0.8866 | 0.8629 | 0.9204 | 0.9445 | 0.8459 |
| ForAda | 0.8955 | 0.7134 | 0.9011 | 0.8429 | 0.8901 | 0.5883 | 0.8532 | 0.8650 | 0.8507 | 0.9361 | 0.8336 |
| FSFM | 0.8998 | 0.8322 | 0.8810 | 0.8574 | 0.8761 | 0.6521 | 0.8742 | 0.8388 | 0.9184 | 0.9518 | 0.8582 |
| FSFM-Adapter | 0.8311 | 0.8118 | 0.7287 | 0.8159 | 0.8338 | 0.6042 | 0.8238 | 0.7371 | 0.8809 | 0.8114 | 0.7879 |

## Video counts (must agree across rows)

| dataset | R0 (v3.1, ep 6) | A (v3.1, ep 5) | B (v3.1, ep 8) | C (v3.1, ep 2) | DiCoME (our training, v3.1, ep 5) | GenD-CLIP | GenD-PE | GenD-DINO | Effort | ForAda | FSFM | FSFM-Adapter |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CDFv2 | 518 | 518 | 518 | 518 | 518 | 518 | 518 | 518 | 518 | 518 | 518 | 518 |
| CDFv3 | 5118 | 5118 | 5118 | 5118 | 5118 | 5118 | 5118 | 5118 | 5118 | 5118 | 5118 | 5118 |
| DFD | 3431 | 3431 | 3431 | 3431 | 3431 | 3431 | 3431 | 3431 | 3431 | 3431 | 3431 | 3431 |
| DFDC | 4704 | 4704 | 4704 | 4704 | 4704 | 4704 | 4704 | 4704 | 4704 | 4704 | 4704 | 4704 |
| DFDCP | 654 | 654 | 654 | 654 | 654 | 654 | 654 | 654 | 654 | 654 | 654 | 654 |
| DFEval24 | 814 | 814 | 814 | 814 | 814 | 814 | 814 | 814 | 814 | 814 | 814 | 814 |
| FFIW | 3470 | 3470 | 3470 | 3470 | 3470 | 3470 | 3470 | 3470 | 3470 | 3470 | 3470 | 3470 |
| KoDF | 1395 | 1395 | 1395 | 1395 | 1395 | 1395 | 1395 | 1395 | 1395 | 1395 | 1395 | 1395 |
| IDForge | 5974 | 5974 | 5974 | 5974 | 5974 | 5974 | 5974 | 5974 | 5974 | 5974 | 5974 | 5974 |
| DFDM | 1896 | 1896 | 1896 | 1896 | 1896 | 1896 | 1896 | 1896 | 1896 | 1896 | 1896 | 1896 |

**Count mismatches:** none

## Published numbers (as printed; not re-measured)

- **DiCoME** (video, https://arxiv.org/html/2606.01885v1 (Table 1)): CDFv2 0.977, DFD 0.982, DFDC 0.882, DFo 0.993, WDF 0.911, CDFv3 0.886
- **Effort** (video, https://arxiv.org/html/2606.01885v1 (Table 1)): CDFv2 0.956, DFD 0.965, DFDC 0.843, DFo 0.977, WDF 0.848, CDFv3 0.850
- **GenD (backbone unstated)** (video, https://arxiv.org/html/2606.01885v1 (Table 1)): CDFv2 0.960, DFD 0.970, DFDC 0.871, DFo 0.989, WDF 0.890, CDFv3 0.859
- **Effort** (video, https://arxiv.org/html/2411.15633v3 (Table 1, Protocol-1)): CDFv2 0.956, DFD 0.965, DFDC 0.843, DFDCP 0.909, DFo 0.977, WDF 0.848, FFIW 0.921
- **ForAda** (video, https://arxiv.org/html/2508.06248 (v4, Table 4, 'recalculated by us')): UADFV 0.994, DFD 0.972, DFDC 0.873, FSh 0.820, CDFv2 0.957, FFIW 0.906, KoDF 0.882, FAVC 0.931, DFDM 0.971, PGF 0.866, IDForge 0.908, DSv1.1 0.818, DSv2 0.728, CDFv3 0.756
- **Effort** (video, https://arxiv.org/html/2508.06248 (v4, Table 4, 'recalculated by us')): UADFV 0.974, DFD 0.952, DFDC 0.854, FSh 0.912, CDFv2 0.932, FFIW 0.925, KoDF 0.881, FAVC 0.924, DFDM 0.982, PGF 0.849, IDForge 0.960, DSv1.1 0.821, DSv2 0.644, CDFv3 0.787
- **GenD-CLIP** (video, https://arxiv.org/html/2508.06248 (v4, Table 4, 'recalculated by us')): UADFV 0.992, DFD 0.964, DFDC 0.864, FSh 0.866, CDFv2 0.946, FFIW 0.915, KoDF 0.849, FAVC 0.960, DFDM 0.996, PGF 0.896, IDForge 0.978, DSv1.1 0.901, DSv2 0.777, CDFv3 0.859
- **GenD-PE** (video, https://arxiv.org/html/2508.06248 (v4, Table 4, 'recalculated by us')): UADFV 0.977, DFD 0.968, DFDC 0.822, FSh 0.876, CDFv2 0.950, FFIW 0.937, KoDF 0.851, FAVC 0.973, DFDM 0.983, PGF 0.923, IDForge 0.979, DSv1.1 0.878, DSv2 0.786, CDFv3 0.895
- **GenD-DINO** (video, https://arxiv.org/html/2508.06248 (v4, Table 4, 'recalculated by us')): UADFV 0.986, DFD 0.962, DFDC 0.856, FSh 0.888, CDFv2 0.925, FFIW 0.929, KoDF 0.897, FAVC 0.984, DFDM 0.998, PGF 0.924, IDForge 0.982, DSv1.1 0.869, DSv2 0.794, CDFv3 0.835
- **ForAda** (frame, /data/umar/Repos/ForensicsAdapter/README.md (frame table + video table)): CDFv1 0.914, CDFv2 0.900, DFDC 0.843, DFDCP 0.890, DFD 0.933
- **ForAda** (video, /data/umar/Repos/ForensicsAdapter/README.md (frame table + video table)): CDFv2 0.957, DFDC 0.872, DFDCP 0.929
- **FS-VFM ViT-L** (frame, https://huggingface.co/Wolowolo/fsfm-3c FS-VFM ViT-L FT_on_FF++_c23_32frames experiments_test/test_results.txt): DFD 0.933, CDFv1 0.845, CDFv2 0.876, DFDC 0.836, DFDCP 0.883, WDF 0.903, CDFv3 0.864
- **FS-VFM ViT-L** (video, https://huggingface.co/Wolowolo/fsfm-3c FS-VFM ViT-L FT_on_FF++_c23_32frames experiments_test/test_results.txt): DFD 0.972, CDFv1 0.937, CDFv2 0.952, DFDC 0.877, DFDCP 0.933, WDF 0.916, CDFv3 0.911

## END SOURCE: `tbiom/reports/BASELINE_MATRIX.md`


==============================================================================================================
# HISTORICAL EVIDENCE
==============================================================================================================


--------------------------------------------------------------------------------------------------------------
## BEGIN SOURCE: `tbiom/reports/REFERENCE_MANIFEST.md`

*Role in this handoff:* SUPERSEDED as the method of record (it documents the DS-R operator). Retained because it contains the numerical comparison of evidence scaling against belief-mass discounting, which is the definition the current method still uses.

--------------------------------------------------------------------------------------------------------------

# Reference manifest — executable brief, Stage 0 audit

Resolved from the live tree on 2026-09-20, commit `baf16b6`, by `analysis/tbiom/stage0_audit.py`; every number below is recomputed, not retyped. Every later experiment in `tbiom/docs/discern-v2-executable-brief.md` cites this file. Raw numbers: `STAGE0_AUDIT.json`.

## 1. The three reference branches

| view | checkpoint | sha256 (first 16) |
|---|---|---|
| T (V-JEPA 2 LoRA, B-augmented) | `logs/tbiom/temporal/branches/vjepa2_lora_B_seed42/best.pt` | `6f0d9c63a6c052de` |
| APP + OM (warm+gate, epoch 19) | `/data/umar/Repos/DISCERN_Ext/runs/discern_ext_train/warm_dicome_gate_seed42/checkpoints/discern_ext-best-epoch=19-val_auroc_video=0.9956.ckpt` | `fdc924a7cc9e1579` |

APP and OM are the Appearance and Off-manifold views of one checkpoint (shared CLIP ViT-L/14 + LoRA r8 α16 path; β-VAE latent 32, β_KLD 2.0, λ_align 1.0, λ_vae 0.7). The face prior inside that checkpoint is outside the reference. The hashes match the previous manifest (yes).

**Preprocessing and aggregation order.**
- T: one clip per video, 16 frames at `linspace(0, n−1, 16)` over the video's stored crops, in temporal order, V-JEPA 2's own 256-px processor; the branch emits one Dirichlet per clip, so there is no frame aggregation.
- APP/OM: every stored crop scored separately at 224 px (CLIP preprocessing); per-frame (p, u) are **averaged per video first**, then the three views are discounted and fused.
- Order: per-frame spatial opinions → per-video mean → reconstruct α from (p, u) → discount each view → Dempster combination T ⊕ APP ⊕ OM (associative).

Temporal-branch training record:

```json
{
 "model": "vjepa2",
 "seed": 42,
 "best_epoch": 6,
 "best_dev_v31": 0.8091709420774639,
 "budget": {
  "augment": "B",
  "arm": "lora",
  "lora": {
   "r": 8,
   "alpha": 16,
   "dropout": 0.1,
   "lr": 0.0002
  },
  "lr_blocks": 0.0002,
  "lr_head": 0.002,
  "epochs": 10,
  "unfrozen_blocks": 2,
  "trainable_block_params": 786432,
  "head_params": 133506,
  "batch": 32,
  "frames": 16
 }
}
```

## 2. The discount rule, read from code

`analysis/tbiom/temporal/finish_line.py::dsr` (identical in `dsr_system.py`):

```python
S = K / u                                   # K = 2 classes
alpha = [(1 - p) * S, p * S]                # reconstruct the Dirichlet from (p, u)
alpha' = 1 + w * (alpha - 1)                # discount
```

That is **evidence scaling**: e' = w·e, S' = w·Σe + K, b' = w·e / (w·Σe + K), u' = K / (w·Σe + K). It is **not** belief-mass (Shafer) discounting, b' = w·b, u' = 1 − w·(1 − u). The two coincide only at w = 1. The previous manifest called this rule "Shafer discounting"; that name was wrong and is corrected here — the equations it printed were right.

Numerical check on 2,000 opinions (p ∈ [0.05, 0.95], u ∈ [0.02, 0.9]):

| w | max \|Δp\| code vs belief-mass | max \|Δu\| | mean \|Δu\| |
|---:|---:|---:|---:|
| 0.25 | 0.2886 | 0.6788 | 0.1646 |
| 0.5 | 0.2053 | 0.4705 | 0.1453 |
| 0.75 | 0.1055 | 0.2384 | 0.0830 |

The combination after discounting is Dempster's rule on the discounted opinions, identical to DISCERN's in-model `DS_Combin` / `_combine_two_opinions` (max |Δp| 2.2e-16 at w = 1): with b_i = e_i / S_i and u_i = K / S_i, conflict C = Σ_{j≠k} b_1j b_2k, b = (b_1 b_2 + b_1 u_2 + b_2 u_1)/(1 − C), u = u_1 u_2 / (1 − C), α = b·K/u + 1. DISCERN's own face-prior gate uses the same evidence-scaling form (`1 + g·(α − 1)`).

**Consequence for Stage 1A:** every controller predicts w per view and applies α' = 1 + w(α − 1), unchanged. Changing to belief-mass discounting would be a separate experiment, never combined with a new predictor.

Fitted weights on the five source-grouped splits (from the previous manifest): (0.25, 0.25, 1.0); (0.25, 0.25, 1.0); (0.25, 0.25, 1.0); (0.25, 0.25, 1.0); (0.25, 0.25, 1.0) for (T, APP, OM).

## 3. Cache inventory and segment granularity

**What a 'frame' is here.** The stored crops are a sparse, even sample of each source video, not its frames:

| development source file | videos | stored crops per video (min / median / max) | median raw-frame step between stored crops |
|---|---:|---|---:|
| VALmix | 1350 | 1 / 32 / 32 | 11.0 |
| VALmixV2 | 4852 | 1 / 16 / 16 | 21.0 |
| FFpp_val | 700 | 31 / 32 / 32 | 14.0 |

So a video is 16–32 stills roughly 10–15 source frames apart (≈ 0.3–0.5 s at 30 fps). `/data/umar/Datasets/tbiom_frames` (29 GB) holds the same 32-still sampling for CDFv2, CDFv3, DFD, DFDC, DFDCP and DFEval24 — it is not denser.

**What exists per (view, granularity, condition):**

| view | granularity | clean development | JPEG q40 / resize 0.60 development | tests |
|---|---|---|---|---|
| APP, OM | per stored crop | yes (`warmgate_e19_*`) | yes (`p_dev_warmgate_e19_*`) | clean all ten; processed CDFv2, DFDCP |
| T | one opinion per video (full span) | yes | yes | clean all ten; processed CDFv2, DFDCP |
| T | per segment (4 contiguous quarters + full) | yes (`logs/tbiom/acquisition/`) | **no** | **no** |
| quality descriptors (blur, blockiness, contrast, motion, luma) | per segment | yes, same cache | **no** | **no** |

Segment cache detail (clean development only):

| source | videos | segments/video | distinct stored crops per segment (min / median / max) | segments whose 16-frame clip repeats crops | degenerate (empty) segments |
|---|---:|---:|---|---:|---:|
| VALmix | 1350 | 4 | 0 / 8 / 8 | 5388 of 5400 | 12 in 6 videos |
| VALmixV2 | 3764 | 4 | 0 / 4 / 4 | 15035 of 15056 | 21 in 10 videos |
| FFpp_val | 700 | 4 | 7 / 8 / 8 | 2800 of 2800 | 0 in 0 videos |

Timestamps: the stored-crop file names carry the source frame index (`000.png`, `014.png`, …; `frame_0010.png`), so segment order and approximate time are recoverable; source fps is not stored.

## 4. Gate

Aligned segment-level opinions from all three views **do not exist** in the form Stage 1A needs:

1. **Processed conditions have no segment-level temporal opinions and no quality descriptors.** The controllers are trained on clean *and* processed examples, so these must be generated by local clip inference before any controller is fitted. This adds GPU cost (the clean pass took ≈13 GPU-minutes for 29k segment clips; two processed conditions ≈ 30 more).
2. **At the stored density a segment holds 4–8 distinct crops**, so every per-segment V-JEPA clip is those crops repeated to 16 frames. Each segment is still a genuinely different observation — its frames are distinct from the other segments' — so this is not the forbidden repetition of one video-level opinion across timestamps; but the temporal branch sees a stuttered clip at segment granularity, and that is recorded as a limitation of every segment-level result.
3. **Degenerate segments** (videos with fewer crops than segments) were filled with the full-span opinion in the acquisition cache. That *is* the forbidden simulation; those segments are excluded, not filled, in every Stage 1A fit.
4. **Spatial segment opinions need no inference**: they are exact means of the cached per-frame opinions over the same crops the temporal segment used.

**Processing-sensitive branch for Stage 1B** (from Experiment 3 on the same reference): development v3.1 drop clean → JPEG q40 is T 0.0498, APP 0.0994, OM 0.0813. The spatial path is the more processing-sensitive one, and APP and OM share the adapted CLIP path, so 1B treats them as one coupled spatial experiment.

**Pass condition.** Three checkpoints named and hashed; the discount rule read from code and verified (evidence scaling); segment granularity recorded (4 segments of 4–8 distinct stored crops, clean development only). Stage 1A proceeds only after the missing processed segment opinions and descriptors are generated.

## END SOURCE: `tbiom/reports/REFERENCE_MANIFEST.md`


--------------------------------------------------------------------------------------------------------------
## BEGIN SOURCE: `tbiom/reports/GROUPED_FUSION_OUTCOME.md`

*Role in this handoff:* SUPERSEDED / closed line of work. Retained because it explains why a partitioned (grouped) t-norm operator is NOT the proposed method, and because its zero-weight finding is why the current operator was chosen for its vacuous-view behaviour.

--------------------------------------------------------------------------------------------------------------

# Grouped (dependency-partitioned) evidential fusion — outcome

Stage 10 of `tbiom/docs/grouped-fusion-brief.md`. Pre-registration: `tbiom/configs/grouped_fusion_preregister.yaml`,
written before any candidate was scored.

## Terminal outcome

**No demonstrated advantage.** grouped-R fails its Stage 3 confirmation gate. The requirement that it do more than a
plain two-view fusion fails at every one of the four detector configurations, for a reason that is algebraic rather
than statistical, and the falsification controls that passed on the deployed system fail once the detector changes.

All four seed configurations are now complete. The full detector-seed swap did **not** reproduce the effect, and it
exposed a stronger version of the same conclusion.

## The gate, requirement by requirement

Four configurations: the deployed warm+gate spatial views with V-JEPA 2 seed 42 and seed 1337, and SBI arm B — the
one three-branch family trained at several seeds — at seeds 42 and 1337 with the matching temporal branch. Each
system's reliability weights are fitted on **its own** development exports.

| requirement | warm+gate / T42 | warm+gate / T1337 | armB42 | armB1337 |
|---|---|---|---|---|
| DF40 reenactment gain over Dempster-R | +0.0114 | +0.0025 | **+0.0000** | +0.0020 |
| control grouped-TA-R loses | +0.0033 | +0.0012 | **+0.0000** | **−0.0003** |
| control grouped-TO-R loses | +0.0049 | +0.0051 | **+0.0010 (CI spans 0)** | **−0.0005** |
| does more than two-view T+OM | **+0.0000** | **−0.0001** | **+0.0000** | **+0.0000** |

Bold = fails. 2,000 paired source-group draws per contrast, reused real videos grouped across generator sets, and a
materiality floor of 1e-4 AUROC so that two arms differing only by floating-point noise cannot produce a "significant"
interval by repeating the same tiny difference in every draw.

**The T+OM requirement fails at all four seeds. The falsification controls, which passed under both warm+gate
configurations, fail under both arm-B seeds.** The partition effect does not survive a change of detector.

## Why the fourth requirement fails

The operator's own fitted discount sets the **appearance weight to zero** in 4 of 5 folds on the deployed system,
2 of 5 on armB42 and 3 of 5 on armB1337. An evidence-scaled opinion with r = 0 carries no evidence; its canonical
weights are both 1; the minimum of 1 and w_OM is w_OM. The cautious inner step therefore returns the off-manifold
opinion unchanged, and what remains is the outer Dempster step between two views. In those folds grouped-R **is** the
two-view fusion of temporal and off-manifold, bit for bit.

The arm-B seeds sharpen this considerably. There the **baseline's** fit discards the appearance view too —
Dempster-R zeroes it in 5 of 5 folds at armB42, as does grouped-TA-R — so Dempster-R, grouped-TA-R, grouped-R and
T+OM all collapse onto the same function and score identically (0.9607 reenactment, 0.9674 swap, 0.9638 overall).
That is why three requirements read exactly +0.0000 at that seed.

The general statement is therefore stronger than "grouped-R equals T+OM". It is: **once the reliability weight is
free to be fitted, the appearance view is discarded, and the redundancy the partition was designed to handle has
already been removed by the discount.** The partition has nothing left to do. A fused operator cannot demonstrate a
dependence correction on a pair of views when the fit has already deleted one of them.

## What the controls do and do not establish

Under the two warm+gate configurations the controls pass: pairing the temporal view with either spatial view is worse
than pairing the two CLIP heads. Read alone that looks like support for the dependence argument.

Under both arm-B seeds they do not. grouped-TA-R matches grouped-R exactly at armB42 and beats it by +0.0003 at
armB1337; grouped-TO-R beats it by +0.0005 at armB1337. So the control result is a property of one detector, not of
the architecture — which is exactly what a second detector seed is for, and exactly why the brief required one.

Even where the controls pass, they would not have rescued the operator: a grouping that beats two wrong groupings is
not evidence for a mechanism when a simpler arm matches the winner exactly.

## Development, for completeness

No arm clears the pre-registered detection bar. grouped-R is −0.0016 against Dempster-R with the CI spanning zero in
5 of 5 splits, and it fails the load-bearing vacuity guard at −0.0055 (`EXP_GROUPED_DEVELOPMENT.md` §3–4). The PCR6-
outer arms fail that guard heavily, as the audit predicted from the algebra (GCP-AO-R −0.0347, GCY-AO-R −0.0537).

The mechanism inspection describes a trade, not an advantage:

| case on the selection folds | videos | grouped-R | Dempster-R |
|---|---:|---:|---:|
| both CLIP heads wrong together, temporal right | 362 | **0.7428** | 0.5558 |
| temporal wrong, CLIP heads agree and right | 520 | 0.5796 | **0.7173** |

## The swap-family cost, stated plainly

Every configuration that gains on reenactment loses on face swap. grouped-R against Dempster-R is −0.0033 [−0.0052,
−0.0016] on the 19 swap sets at seed 42, and −0.0007 [−0.0011, −0.0004] at seed T1337. Both intervals exclude zero.
This is reported here and in every table rather than left to the favourable family.

## Decisions

1. **Dempster-R stays the deployed rule**, and stays a labelled baseline row — the standard Dempster–Shafer operator
   and the DiCoME-family reference. It is not presented as the proposed method, and it is not quietly reinstated as
   one either: nothing in this study proposes a replacement.
2. **No fusion-operator novelty claim is made.** `GROUPED_PRIOR_ART_AND_CLAIMS.md` records why, including that no
   primary source could be retrieved in this session and that Quost, Masson and Denœux (IJAR 2011) on optimised
   t-norm combination is the work most likely to subsume the construction and must be obtained first.
3. The reportable finding is diagnostic: on reenactment generators where both CLIP heads sit at chance they are wrong
   *together*, because they share a feature path, and a rule that reads their agreement as two independent
   confirmations is misled by it. The cleanest evidence for that is the fitted discount itself.
4. Prior results are preserved. `EXP_FUSION_NOVEL.md` (commit `2b3f752`), the single-seed interval-free version, is
   retained unchanged and superseded rather than deleted or relabelled.

## Deliverables

| file | stage |
|---|---|
| `GROUPED_REFERENCE_MANIFEST.md` | 4 — audit, algebra of the caches, exposure history |
| `tbiom/configs/grouped_fusion_preregister.yaml` | 8 — frozen before scoring |
| `EXP_GROUPED_AUDIT.md` / `.json` | 7 — 28/28 algebraic and mechanism checks |
| `EXP_GROUPED_DEVELOPMENT.md` / `.json` | 6 — the fixed matrix on development, both discount settings |
| `EXP_GROUPED_EXPOSED.md` / `.json` | 9 — ten test sets and DF40, both required means |
| `EXP_GROUPED_CONFIRM.md` / `.json` | 3 — the gate |
| `GROUPED_PRIOR_ART_AND_CLAIMS.md` | 9 — prior art and claim-to-evidence map |
| `GROUPED_FUSION_OUTCOME.md` | 10 — this file |

## Deficits, stated rather than papered over

- **No untouched confirmation set exists for this study.** DF40 informed the choice of this operator before the task
  began. The gate tests reproducibility across a seed and against falsification controls; neither is a
  generalisation claim to unseen generators.
- The five folds are not five detector seeds, and the bootstrap quantifies sample uncertainty conditional on the
  trained encoders, not encoder-training variability.
- The deployed warm+gate recipe exists at one seed and this task forbids training another, so the full detector swap
  uses SBI arm B, a different training recipe. Within arm B the seed is the only difference, but arm B is not the
  deployed recipe, so the four configurations differ in more than a seed.
- The reenactment effect size ranges over an order of magnitude across configurations where it is positive at all:
  +0.0114, +0.0025, +0.0020, and +0.0000.
- An earlier run of this stage fitted the arm-B weights on warm+gate development exports. That was wrong — the
  weights must come from the system being scored — and the numbers here are the corrected ones. The error is
  recorded rather than silently overwritten; it changed the arm-B conclusions.

## END SOURCE: `tbiom/reports/GROUPED_FUSION_OUTCOME.md`


--------------------------------------------------------------------------------------------------------------
## BEGIN SOURCE: `tbiom/reports/EXP_THREEVIEW_ABLATION.md`

*Role in this handoff:* SUPERSEDED for the view ablation: its numbers are DS-R, not PCR6+-R. The current view ladder is in EXP_PCR6P_DEVELOPMENT/EXPOSED. Retained for the paired bootstrap methodology and the original temporal-view effect size.

--------------------------------------------------------------------------------------------------------------

# Experiment 2 — three-view ablation of the reference triple (T = V-JEPA 2 LoRA B-augmented, APP, OM)

The full triple against each of its three pairs, every combination refitted with the **same** procedure under both fusion rules: DS-R (primary) and logistic stacking on matched inputs (comparator). The face prior does not appear. **These are ablations of the three-view claim, not proposals to drop a view.** Δ = triple − pair on the clean-development v3.1 objective, with a paired source-grouped bootstrap (1000 draws per split, one group resample per draw scoring both arms).

Every number below uses the reference manifest `tbiom/reports/REFERENCE_MANIFEST.md` (three resolved checkpoints, DS-R definition, fold splits). Script `analysis/tbiom/temporal/finish_line.py`.

## 1. Development (selection folds, mean ± sd over 5 splits)

| combination | rule | clean v3.1 | swap | reenactment | in-the-wild | JPEG q40 | resize 0.60 |
|---|---|---:|---:|---:|---:|---:|---:|
| T+APP+OM | DS-R | 0.8808 ±0.0048 | 0.9345 | 0.9085 | 0.7993 | 0.8096 | 0.8589 |
| T+APP+OM | LR | 0.8736 ±0.0068 | 0.9245 | 0.8997 | 0.7965 | 0.7564 | 0.8542 |
| T+APP | DS-R | 0.8706 ±0.0051 | 0.9219 | 0.9084 | 0.7816 | 0.7947 | 0.8533 |
| T+APP | LR | 0.8687 ±0.0052 | 0.9241 | 0.8832 | 0.7988 | 0.7780 | 0.8467 |
| T+OM | DS-R | 0.8795 ±0.0046 | 0.9281 | 0.9182 | 0.7921 | 0.8125 | 0.8600 |
| T+OM | LR | 0.8786 ±0.0056 | 0.9318 | 0.8984 | 0.8057 | 0.8142 | 0.8551 |
| APP+OM | DS-R | 0.8238 ±0.0067 | 0.9282 | 0.7600 | 0.7833 | 0.7409 | 0.7847 |
| APP+OM | LR | 0.8179 ±0.0085 | 0.9065 | 0.7732 | 0.7740 | 0.6499 | 0.7857 |

## 2. Does each view earn its place? (Δ = triple − pair, clean development)

| removed view | remaining pair | rule | mean Δ | 95% CI (split 0) | splits with CI lower bound > 0 | triple beats pair in |
|---|---|---|---:|---|---:|---:|
| OM | T+APP | DS-R | +0.0102 | [+0.0070, +0.0149] | 5/5 | 5/5 splits |
| OM | T+APP | LR | +0.0047 | [-0.0080, +0.0120] | 0/5 | 5/5 splits |
| APP | T+OM | DS-R | +0.0012 | [-0.0038, +0.0041] | 0/5 | 5/5 splits |
| APP | T+OM | LR | -0.0050 | [-0.0148, -0.0008] | 0/5 | 0/5 splits |
| T | APP+OM | DS-R | +0.0569 | [+0.0458, +0.0743] | 5/5 | 5/5 splits |
| T | APP+OM | LR | +0.0560 | [+0.0445, +0.0735] | 5/5 | 5/5 splits |

## 3. Processed development — does the pair catch up under processing?

| combination | rule | clean | JPEG q40 | resize 0.60 | mean processed Δ vs triple |
|---|---|---:|---:|---:|---:|
| T+APP+OM | DS-R | 0.8808 | 0.8096 | 0.8589 | +0.0000 |
| T+APP+OM | LR | 0.8736 | 0.7564 | 0.8542 | +0.0000 |
| T+APP | DS-R | 0.8706 | 0.7947 | 0.8533 | +0.0103 |
| T+APP | LR | 0.8687 | 0.7780 | 0.8467 | -0.0070 |
| T+OM | DS-R | 0.8795 | 0.8125 | 0.8600 | -0.0020 |
| T+OM | LR | 0.8786 | 0.8142 | 0.8551 | -0.0294 |
| APP+OM | DS-R | 0.8238 | 0.7409 | 0.7847 | +0.0714 |
| APP+OM | LR | 0.8179 | 0.6499 | 0.7857 | +0.0875 |

## 4. Test sets — video AUROC (read after fitting; mean over the 5 fitted splits)

| combination | rule | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | DFDM | DFEval24 | FFIW | IDForge | KoDF | CDFv2 jpeg40 | CDFv2 rs06 | CDFv3 reenactment | DFDCP jpeg40 | DFDCP rs06 | KoDF first-order motion | 10-set mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T+APP+OM | DS-R | 0.9598 | 0.9616 | 0.9338 | 0.8626 | 0.8656 | 0.9956 | 0.7252 | 0.9282 | 0.9656 | 0.9574 | 0.8695 | 0.9418 | 0.9642 | 0.8198 | 0.8084 | 0.9814 | **0.9155** |
| T+APP+OM | LR | 0.9697 | 0.9675 | 0.9300 | 0.8602 | 0.8557 | 0.9963 | 0.6865 | 0.9262 | 0.9622 | 0.9630 | 0.7199 | 0.9470 | 0.9709 | 0.7299 | 0.7935 | 0.9792 | **0.9117** |
| T+APP | DS-R | 0.9345 | 0.9530 | 0.9134 | 0.8373 | 0.8458 | 0.9930 | 0.7187 | 0.9128 | 0.9524 | 0.9544 | 0.8340 | 0.9260 | 0.9592 | 0.8039 | 0.7973 | 0.9776 | **0.9015** |
| T+APP | LR | 0.9562 | 0.9557 | 0.9125 | 0.8464 | 0.8551 | 0.9933 | 0.7029 | 0.9074 | 0.9546 | 0.9574 | 0.8530 | 0.9406 | 0.9586 | 0.7654 | 0.7925 | 0.9660 | **0.9041** |
| T+OM | DS-R | 0.9543 | 0.9676 | 0.9271 | 0.8487 | 0.8585 | 0.9957 | 0.7224 | 0.9142 | 0.9613 | 0.9628 | 0.8643 | 0.9338 | 0.9743 | 0.8134 | 0.8045 | 0.9870 | **0.9113** |
| T+OM | LR | 0.9675 | 0.9619 | 0.9352 | 0.8635 | 0.8658 | 0.9952 | 0.7199 | 0.9275 | 0.9640 | 0.9563 | 0.8876 | 0.9475 | 0.9630 | 0.8160 | 0.8025 | 0.9778 | **0.9157** |
| APP+OM | DS-R | 0.9732 | 0.8787 | 0.9341 | 0.8811 | 0.8733 | 0.9919 | 0.6701 | 0.9347 | 0.9633 | 0.8543 | 0.8848 | 0.9544 | 0.8397 | 0.8250 | 0.7976 | 0.7292 | **0.8955** |
| APP+OM | LR | 0.9699 | 0.9080 | 0.9123 | 0.8701 | 0.8541 | 0.9922 | 0.6272 | 0.9208 | 0.9521 | 0.8630 | 0.7139 | 0.9514 | 0.8850 | 0.7212 | 0.7778 | 0.7233 | **0.8870** |

DS-R weights per split are in the JSON; split-0 weights: T+APP+OM (0.25, 0.25, 1.0); T+APP (0.25, 1.0); T+OM (0.25, 1.0); APP+OM (0.0, 1.0).

One detector seed and one temporal-branch seed unless stated. CDFv2 and DeepfakeEval2024 informed development and are not untouched confirmation data. FaceShifter is sealed (`tbiom/docs/SEALED_BENCHMARKS.md`).

## END SOURCE: `tbiom/reports/EXP_THREEVIEW_ABLATION.md`


==============================================================================================================
# DISCERN-to-extension component map
==============================================================================================================

This map is maintained in the manuscript rather than in a report, because it is qualitative. Source: `tbiom/paper/tables/tab_extension.tex` (hand-written). It was compiled from the DISCERN paper (`tbiom/docs/DISCERN.pdf`, Secs. 1, 3.3, 3.4) and the live configs (`DISCERN_Ext/eval_adaptation/configs/warm_dicome_gate_seed42.yaml`, `analysis/tbiom/temporal/train_branch.py`).

```latex
% HAND-WRITTEN, not generated by build_tables.py (qualitative content).
% Sources: DISCERN paper (tbiom/docs/DISCERN.pdf, Secs. 1, 3.3, 3.4) and this system's configs
% (DISCERN_Ext warm_dicome_gate_seed42.yaml, analysis/tbiom/temporal/train_branch.py).
\begin{table*}[t]
\centering
\caption{Relation to the conference version~\cite{farooq2026discern}. The extension keeps the evidential treatment of
heterogeneous streams and replaces everything about how those streams are constituted and combined.}
\label{tab:extension}
\small
\begin{tabular}{p{0.20\textwidth}p{0.10\textwidth}p{0.30\textwidth}p{0.30\textwidth}}
\toprule
Conference component & Status & This paper & Why \\
\midrule
Visual stream: CLIP with layer-norm tuning & modified & appearance view \app{}: CLIP ViT-L with LoRA adapters on the attention projections & adapters leave the backbone untouched, so the view can be frozen and reused as a fixed body of evidence \\
Semantic stream: differentiable predicates over facial signals (expression--action-unit alignment, pose--gaze consistency, geometric symmetry) & removed & --- & the predicates are per-frame and appearance-derived; they do not address the reenactment failure this paper targets \\
Structural stream: class-dependent dependency modelling, retained as interpretable structure & replaced & off-manifold view \om{}: a $\beta$-VAE head on the same CLIP path, scoring distance from the genuine-face manifold & a single scalar deviation score proved to be the more reliable of the two spatial signals, contributing on every evaluation group (Table~\ref{tab:viewrole}) \\
--- & new & temporal view \tmp{}: V-JEPA\,2 ViT-L with LoRA adapters and an evidential head on ordered clips & the conference system had no access to motion, which is the only channel in which a cleanly rendered reenactment differs from a real video \\
Confidence-modulated evidence fusion & replaced & reliability as evidence scaling, $e'=r\,e$, fitted per fold on source-grouped development folds & a fitted per-view reliability is auditable and can be reported per fold; it also makes zero weights explicit rather than implicit \\
Inter-branch disagreement calibration & replaced & direct multi-source \pcrp{} with retained conflict mass $K$ & conflict is redistributed to the views that produced it and reported, instead of being converted into a calibration penalty during training \\
Per-stream auxiliary supervision & retained & per-view evidential supervision in the spatial checkpoint & unchanged from the conference recipe \\
Evaluation: five cross-dataset benchmarks & extended & ten cross-dataset benchmarks, $41$ DF40 generator--source entries, and two evaluation-time processing conditions & the swap/reenactment asymmetry is invisible at five swap-dominated benchmarks \\
\bottomrule
\end{tabular}
\end{table*}
```

## Face-prior view decision (why the fusion is three views, not four)

Source: `tbiom/reports/EXP_FACEPRIOR_VIEW.json`, produced by `analysis/tbiom/paper/faceprior_test.py`. Every arm fitted and scored by the established procedure on cached opinions.

| view set | dev v3.1 | dev reenactment | 10-set mean | DF40 family-balanced | zero-weight folds |
|---|---:|---:|---:|---:|---|
| T+A+O (deployed) | 0.8794 | 0.9111 | 0.9129 | 0.9602 | APP 3/5 |
| T+O+F | 0.8800 | 0.9157 | 0.9121 | 0.9624 | FP 5/5 |
| T+A+F | 0.8691 | 0.9035 | 0.9009 | 0.9505 | FP 5/5 |
| T+A+O+F | 0.8794 | 0.9111 | 0.9129 | 0.9602 | APP 3/5, FP 5/5 |
| T+F | 0.8383 | 0.8997 | 0.8801 | 0.9539 | none |
| F only | 0.7688 | 0.7605 | 0.8521 | 0.8855 | none |
| T+O | 0.8800 | 0.9157 | 0.9121 | 0.9624 | none |

The reliability fit drives the face-prior weight to zero in 5 of 5 folds both as a substitute for the appearance view and as a fourth view, leaving the fused system numerically identical. `T+O+F` is bit-for-bit `T+O`.


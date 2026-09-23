# DISCERN extension — three-view PCR6+ evidential fusion

## 1. Scope

This brief supersedes the grouped-fusion queue. Read the applicable AGENTS.md, locate the live DFB_NeSyNS / DISCERN_Ext tree, and preserve every prior report unchanged.

Decisions already made.
- The grouped (partitioned) rule is closed. Its outcome report stands as a diagnostic finding. Do not rebuild it.
- The proposed fusion is a three-view joint operator with fitted reliability. It is not Dempster's rule.
- Dempster-R stays in every table as the labelled baseline and the DiCoME-family reference. It is not the proposed method and is not reinstated if candidates underperform.
- A declared detection trade is acceptable. A lower AUROC is fine if a pre-registered reliability benefit is shown.
- Closed-form fusion on cached opinions only. No encoder training, no learned fusion head, no new detector runs.

Primary candidate, named in advance. **PCR6+-R**, direct three-source PCR6+ on opinions after the existing reliability discount. The reason for this choice is one property. A view discounted to zero becomes vacuous, and a vacuous view should act exactly as if it were removed. Standard multisource PCR6 violates that property. PCR6+ is designed to restore it. PCR6+ is established prior work and must be credited, never presented as a new operator.

Research question. Can the fused decision use redistributed conflict while conflict attributed to its source views predicts fused errors better than vacuity, total conflict, entropy or margin.

---

## 2. Audit first

Write `tbiom/reports/PCR6P_REFERENCE_MANIFEST.md` with the commit and dirty status, checkpoint and cache hashes, branch order by name, the four detector configurations and their own development exports, folds, the reliability grid, the frozen threshold rule and the exposure history. Reuse the grouped-fusion manifest where it is still accurate and say which parts were reused.

**Define R exactly.** The current discount is evidence scaling, e' = r e. State it, and state that it differs from classical belief-mass discounting at intermediate r while sharing the r = 0 endpoint. Keep evidence scaling throughout this study.

**Obtain the operator definitions from primary sources.** Standard PCR6 for n sources, and PCR6+ from its source paper. Record the exact formulas, the focal elements that receive each redistributed product, and the zero-denominator convention. If the PCR6+ primary definition cannot be obtained, stop and report IMPLEMENTATION_BLOCKED. Do not reconstruct it from memory or a secondary summary.

**Label historical results honestly.** Check whether the existing PCR6-R numbers came from direct three-source PCR6 or a pairwise fold. If pairwise, relabel that row PCR6-pairwise-R in every new table and keep the old report unchanged.

**Algebraic tests, before any dataset run.** Write them as unit tests with expected values.

| Test | Expected result |
| --- | --- |
| Valid output masses and probabilities | Non-negative, sum to one |
| Permutation of the three views | Identical output for direct PCR6 and PCR6+ |
| Views (real = 1), (fake = 1), (Θ = 1) under direct PCR6 | real, fake and Θ each 1/3 |
| Same three views under PCR6+ | Equal to two-source fusion of the first two views |
| One view with r = 0 | PCR6+ equals fusion of the other two views, for random admissible inputs |
| Two sources only | PCR6 and PCR6+ reduce to PCR5 |
| Fully conflicting dogmatic views | Stated, finite, documented output |
| Very small vacuity | Stable arithmetic, no NaN |
| Evidence to opinion and back | Exact round trip for u > 0 |

If any test fails, fix the implementation, record the failure, and rerun before scoring.

---

## 3. Conflict attribution

For each sample record, before redistribution, the total conflicting mass K and its share by source pair. PCR6 redistributes each conflicting product term back to the views that produced it, so this attribution comes from the terms themselves.

- K_TS, conflict between the temporal view and either spatial view.
- K_AO, conflict between Appearance and Off-manifold.
- K_all, three-way conflicting products, reported separately rather than split arbitrarily.

State the attribution rule explicitly. Check that the shares sum to K on every sample.

---

## 4. Fixed comparison matrix

Finite. Do not expand into a further operator sweep.

| ID | Construction | Role |
| --- | --- | --- |
| PCR6+-R | Direct three-source PCR6+ with fitted discount | Primary candidate |
| PCR6-R | Direct three-source PCR6 with fitted discount | Isolates the PCR6+ correction |
| PCR6-pairwise-R | Historical pairwise fold, if that is what exists | Labelled historical row |
| Yager-R | Conflict assigned to Θ | Redistribution versus ignorance |
| Averaging-R | Evidence averaging, prior added once | Simple non-Dempster control |
| Dempster-R | Standard three-source rule | Mandatory baseline |
| T+OM under PCR6+-R | Two views | Does Appearance add value |
| T+APP under PCR6+-R | Two views | Does Off-manifold add value |

Every arm uses the identical existing reliability grid, fitting folds, objective and candidate cap, fitted separately for each detector configuration on that configuration's own development exports.

Run all four detector configurations: warm+gate spatial views with V-JEPA seed 42, with V-JEPA seed 1337, SBI arm B seed 42 and arm B seed 1337. State that arm B is a different training recipe, so the four differ in more than a seed.

Report, per operator and configuration, how many of five folds set each view's weight to zero. If PCR6+-R zeroes Appearance in most folds, the T+OM row decides whether the paper stays three-view.

---

## 5. Uncertainty scores, frozen before scoring

The decision is always the fused PCR6+-R probability. Uncertainty scores do not change the decision, so they cannot change detection AUROC. Evaluate them only as error rankers and for selective prediction.

Scores for PCR6+-R.
1. Fused vacuity.
2. Total conflict K.
3. Predictive entropy.
4. Decision margin, |p_fake − 0.5|.
5. Conflict-by-source score. A logistic model with at most five inputs (vacuity, K_TS, K_AO, K_all, margin), fitted on the fitting folds only, with its coefficients reported.

Also report vacuity and margin for Dempster-R and averaging-R, and margin for every arm, so every operator has an uncertainty baseline.

Errors are defined at the frozen threshold rule already in the protocol.

---

## 6. Pre-registration and acceptance

Write `tbiom/configs/pcr6p_fusion_preregister.yaml` before scoring any candidate. Reuse the existing five source-grouped folds and v3.1 objective. Keep linked sources and processed versions together.

**Detection, declared trade.**
- Noninferiority of PCR6+-R against Dempster-R on development clean macro AUROC, paired lower bound above −0.005.
- The same noninferiority on development JPEG q40, resize 0.60 and reenactment.
- Calibrated FPR rise and TPR loss each at most 0.01 absolute.
- Failing these is reported as a trade, not hidden. It does not reinstate Dempster-R as the method.

**Reliability benefit, required for any uncertainty claim.**
- The conflict-by-source score must beat each of fused vacuity, total conflict, entropy and margin for PCR6+-R on development error-detection AUROC, paired interval excluding zero.
- It must also improve development macro E-AURC over PCR6+-R vacuity, paired interval excluding zero.
- Report the same comparisons within each development dataset and processing condition, not only pooled.
- Report against Dempster-R's vacuity as context, without making a match of its value the goal.

**PCR6+ correction.** Report PCR6+-R against PCR6-R on detection and on the zero-weight test from Section 2. A correction that changes nothing on data is itself reportable.

**Statistics and budget.** 2,000 source-grouped paired bootstrap draws per contrast, reused real videos grouped across generator sets, materiality floor of 1e-4 AUROC as in the grouped study. Cached CPU only, OMP_NUM_THREADS=8, at most 8 CPU wall-clock hours. Report unfinished analyses rather than extending scope.

If a requirement cannot be assessed reliably, report INCONCLUSIVE.

---

## 7. Evaluation after the recipe is fixed

Freeze PCR6+-R and its score definition. Then score the exposed ten test sets, processed conditions and DF40 per generator, labelled as exposed benchmark evaluation.

Per test set and per operator, report video AUROC, FPR and TPR at the frozen threshold, ECE, NLL, Brier, E-AURC, error-detection AUROC for every score, risk at 80% and 90% coverage, and coverage at the risk Dempster-R reaches at full coverage. Report DF40 overall, swap, reenactment and every generator, with explicit family and generator-macro weights.

Include a conflict case analysis on development, with counts. Split samples into temporal right and spatial views wrong, temporal wrong and spatial views right, and all agree. Show K_TS, K_AO, the fused decision and correctness in each group. This tests whether conflict attribution separates useful temporal disagreement from disagreement that predicts an error.

---

## 8. Prior-art gate

Compare equations, assumptions and uncertainty outputs with these works.
- Standard PCR5 and PCR6, including the two-source equivalence.
- The PCR6+ source paper.
- Yager's rule and Dempster's rule.
- Discounted Belief Fusion (AISTATS 2025).
- Reliable Conflictive Multi-View Learning (AAAI 2024).
- DiCoME's two-view Dempster fusion.
- DISCERN's confidence-modulated fusion and IBDC.

If a primary source is unavailable, mark that comparison pending. Do not invent a distinction. The claim-to-evidence map must separate established operators, our implementation choices, measured findings and unverified hypotheses. Conflict attribution by source is a candidate contribution only if it passes Section 6 and survives this comparison.

---

## 9. Deliverables and outcomes

New versioned files, nothing overwritten.
- `tbiom/reports/PCR6P_REFERENCE_MANIFEST.md`
- `tbiom/configs/pcr6p_fusion_preregister.yaml`
- `tbiom/reports/EXP_PCR6P_AUDIT.md` with the unit tests and exact formulas
- `tbiom/reports/EXP_PCR6P_DEVELOPMENT.md` and machine-readable metrics
- `tbiom/reports/EXP_PCR6P_EXPOSED.md` and machine-readable metrics
- `tbiom/reports/PCR6P_PRIOR_ART_AND_CLAIMS.md`
- `tbiom/reports/PCR6P_OUTCOME.md`

Return one or more outcomes.
1. PCR6+-R noninferior and the conflict-by-source score shows a reliability benefit, confirmed across configurations.
2. PCR6+-R noninferior with no reliability benefit beyond simple scores. PCR6+-R stays as a principled non-Dempster choice with no uncertainty claim.
3. PCR6+-R trades detection for a demonstrated reliability benefit.
4. No demonstrated benefit.
5. Implementation blocked.
6. Independent confirmation pending, which applies in every case because no untouched set remains.

The final recommendation keeps Dempster-R as a baseline row only, reports every predefined outcome including losses, and states plainly whether any fusion or uncertainty novelty claim is supported.

Begin with Section 2. Do not score any candidate until the unit tests pass and the pre-registration file is written.
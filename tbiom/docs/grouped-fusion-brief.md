# DISCERN extension — grouped (partitioned t-norm) evidential fusion

## 1. This brief supersedes the previous instructions

Do not execute the prior non-Dempster fusion queue or the GCP-first queue. This document replaces them. Read applicable AGENTS.md, locate the live DFB_NeSyNS / DISCERN_Ext tree, preserve existing work and historical results, and never overwrite or rename unfavorable prior reports.

User constraints, revised.
- Dempster-R is not the proposed method and is not the working reference. It is, however, a mandatory baseline row in every comparison table, as the standard Dempster-Shafer operator and the DiCoME-family reference. Excluding the standard baseline from the numbers would read as concealment to a reviewer, so it stays in the tables, labeled as baseline, never as our method.
- Keep the evidential representation and closed-form fusion. No new MLP, regression controller, GRU, Mamba, encoder retraining, or auxiliary-loss sweep in this task. The existing logistic stacker may be reused as a comparator only.
- Preserve prior experiment records for provenance. Do not delete, rename, or relabel unfavorable results, and do not claim the new method beats every operator investigated.
- Continue to cite and fairly describe DiCoME and relevant prior methods. Excluding Dempster-R as our method does not remove relevant prior literature.

Primary candidate, already promising, not yet confirmed. **grouped-R**, a partitioned t-norm on Denoeux canonical weights:
```
w_S = min(w_APP, w_OM)      cautious combination inside the dependent CLIP pair
w_F = w_S * w_T             Dempster combination across the independent temporal view
```
Minimum within the two heads that share one CLIP ViT-L path, product across the separately-encoded V-JEPA view. The outer product is the correct operation for genuinely independent sources, and it is not equivalent to applying Dempster to all three views, which double-counts the shared pair. That partitioned application, cautious within the dependent family and Dempster across the independent partition, is the candidate contribution. The method is neither Dempster nor cautious alone, so describe it as a dependency-partitioned fusion informed by the architecture, and check its novelty against prior art in Stage 9 rather than asserting it.

Prior single-seed result to confirm, not to assume: grouped-R tied Dempster-R on development (about -0.0016, CI spanning zero in 5/5 splits) and separated only on DF40 cross-manipulation (all-41 about 0.9621, reenactment about 0.9505), with the largest per-generator gains on danet, sadtalker, mcnet, facevid2vid, the reenactment cases where both CLIP heads sit at chance. It is a trade, ahead on 16 of 22 reenactment sets (about +0.0115) and slightly behind on 18 of 19 swap sets (about -0.0033). These numbers are one detector seed with no confidence interval and must be confirmed in Stage 3 before any claim.

---

## 2. Research question and limits

Hypothesis: consolidating the shared APP/OM feature path with a cautious (idempotent) rule before resolving spatial versus video conflict improves the detection-reliability trade-off compared with treating all three opinions as independent under one rule.

Cautious fusion addresses possible overlapping evidence. It does not identify which branch is correct. Different encoders do not establish independence, and shared encoders do not quantify dependence. Combining a cautious inner step with a Dempster outer step is an architecture-specific design whose novelty must be checked, not assumed from the name.

---

## 3. Confirm grouped-R first (before the comparator matrix)

The prior grouped-R number is single-seed and interval-free. Confirm it before building a matrix on top of it, because stacking an operator search on an unconfirmed result yields two exploratory numbers and no solid claim.

- Per-generator paired source-group bootstrap, 2,000 draws with fixed seeds, on the DF40 reenactment half, to put a 95% interval on the +0.0115 grouped-R over Dempster-R. Resample each source with its linked clean, processed, and manipulated copies together, and do not treat reused real videos across generator sets as independent.
- A second detector seed for the reference three-branch system, re-scored under grouped-R and Dempster-R, so the cross-manipulation effect is shown to hold under a different initialization. The five folds and controller-level resampling do not substitute for a detector seed.
- Re-run the falsification controls under both seeds: grouped-TA-R (assume T+APP dependent) and grouped-TO-R (assume T+OM dependent) must lose to grouped-R. If a wrong grouping matches grouped-R, the mechanism claim fails and the effect is generic min-of-weights, not dependency correction.

Gate. If the reenactment gain's interval excludes zero and holds sign across both seeds, and the control groupings lose under both seeds, grouped-R is a confirmed candidate and the comparator matrix in Stage 6 proceeds. If the interval spans zero or the controls do not clearly lose, report grouped-R as an exploratory trade-off, run the matrix as a diagnostic only, and make no confirmed novelty claim.

Write `tbiom/reports/EXP_GROUPED_CONFIRM.md`.

---

## 4. Audit first (run alongside Stage 3 setup)

Read the current reference manifest, the fusion operator implementation, EXP_FUSION_OPERATORS and EXP_FUSION_NOVEL outputs (reported commit 2b3f752), EXP_DF40_OPERATORS, and the exact imported functions. Resolve paths from the live tree, do not assume pasted descriptions match code. Numbers come from machine-readable outputs.

Write `tbiom/reports/GROUPED_REFERENCE_MANIFEST.md` with:
- Repository commit and dirty status, frozen checkpoint and cache hashes, actual branch order, the shared APP/OM modules, preprocessing and aggregation order.
- Evidence, concentration, base rate, probability, and vacuity definitions, the evidence-scaling discount, the fitting and search objective, existing folds and thresholds.
- Sample counts, exclusions, source and identity grouping, clean/processed links, and shared sources across DF40 generators.
- Exposure history: CDFv2 and DeepfakeEval2024 informed development, DF40 has informed operator design, FaceShifter and any other exposed sets documented. Re-partitioning exposed examples does not create untouched confirmation.

For binary uniform-base-rate opinions verify alpha_k = e_k + 1, u = 2/sum(alpha), b_k = e_k/sum(alpha), p_fake = b_fake + u/2, sum(b) + u = 1, and if reconstructing from p and u require u/2 <= p <= 1 - u/2. Confirm p is the Dirichlet mean. Prefer original evidence caches. Do not silently project invalid opinions into the admissible region, and separate roundoff from substantive inconsistency with counts.

Reuse evidence scaling e'_v = r_v e_v, and record every weight by branch name. Do not switch to belief-mass discounting in this experiment. Verify the normalized cautious operation from its original definition, including canonical-weight indexing, reconstruction, and domain conditions. Do not mistake minimum of probabilities or commonalities for minimum of canonical weights. Test the chosen normalized binary variant explicitly.

---

## 5. Exact operator definitions

For each source apply the audited evidence scaling and convert to an opinion.

Primary, grouped-R (Dempster outer):
```
omega_S = CautiousNormalized(omega'_APP, omega'_OM)
omega_F = Dempster_two_source(omega_S, omega'_T)
```
Dempster_two_source on the binary frame, singleton masses x for S and y for T, vacuities u_S, u_T, j = 1-k, conflict K = x_fake*y_real + x_real*y_fake:
```
b_F,k = (x_k*y_k + x_k*u_T + u_S*y_k) / (1 - K)
u_F   = (u_S*u_T) / (1 - K)
p_F,k = b_F,k + u_F/2
```
The outer Dempster step here combines two views that are independent by construction (consolidated spatial S versus temporal T), which is its valid regime. Handle K approaching 1 (near-total conflict) with a documented numerical policy and report its frequency.

Comparators using a non-product outer rule, for the matrix:
- GCP-AO-R: CautiousNormalized(APP, OM) inner, then two-source PCR6 outer with T. PCR6 binary two-source rule:
```
b_F,k = x_k*y_k + x_k*u_T + u_S*y_k + (x_k^2*y_j)/(x_k+y_j) + (y_k^2*x_j)/(y_k+x_j)
u_F   = u_S*u_T
```
Zero-denominator redistribution terms contribute zero, validate against the original formulation. For two sources PCR6 coincides with PCR5, say so. CRITICAL UNCERTAINTY LIMIT for this arm: u_F = u_S*u_T with conflict redistributed to classes, so opposite confident opinions can produce a near-0.5 prediction with very low vacuity. This arm can be confidently wrong on the conflict cases where your reliability story needs high vacuity, so measure its vacuity error-ranking explicitly and expect it to be the arm most at risk on the uncertainty endpoint.
- GCY-AO-R: CautiousNormalized(APP, OM) inner, two-source Yager outer with T (conflict to vacuity). Prespecified conflict-to-vacuity comparator.
- GAP-AO-R: evidence-average(APP, OM) inner, Dempster or PCR6 outer with T. Tests whether averaging the shared pair matches the cautious inner. Average discounted evidence then add the Dirichlet prior once, not averaging probabilities.

When u_F > 0 map back by e_F,k = 2 b_F,k / u_F, alpha_F,k = e_F,k + 1, with stable arithmetic. If u_F = 0 retain the limiting dogmatic opinion and report undefined concentration, do not introduce an unreported uncertainty floor. Any existing inference safeguard must be documented and applied identically to all arms, and raw-fusion results reported alongside.

Missingness: if one spatial view is absent pass through the other, if S or T is absent pass through the available family, all absent gives the vacuous opinion. Absence and an observed uncertain view are different. No missingness claim without a prespecified mask manifest.

---

## 6. Fixed experiment matrix

Finite, do not expand into another operator search.

| ID | construction | role |
|---|---|---|
| grouped-R | Cautious(APP,OM) then Dempster with T | primary candidate |
| grouped-TA-R | Cautious(T,APP) then Dempster with OM | falsification control |
| grouped-TO-R | Cautious(T,OM) then Dempster with APP | falsification control |
| Dempster-R | standard three-source Dempster with fitted scaling | mandatory baseline, DiCoME-family reference |
| AVG-R | evidence-averaging-R | strong simple comparator |
| CAUT-R | global normalized cautious-R | overlap-tolerant comparator |
| PCR6-R | direct three-source PCR6 with fitted scaling | flat conflict-redistribution comparator |
| GCP-AO-R | Cautious(APP,OM) then PCR6 with T | non-product outer comparator |
| GCY-AO-R | Cautious(APP,OM) then Yager with T | conflict-to-vacuity comparator |
| GAP-AO-R | Average(APP,OM) then outer with T | consolidation-versus-cautious comparator |
| T only | video opinion alone | is fusion needed |
| T+APP | two-source | does OM add value |
| T+OM | two-source | does APP add value |

For grouped and GCP/GCY/GAP arms, run both an all-one-discount version (isolating the grouping intervention) and a separately-fitted-discount version using the identical existing grid, folds, objective, and candidate cap. Optionally apply the PCR6-R fitting-fold optimum unchanged to all groupings as a common-weight diagnostic. Never import historical Dempster-R weights as the reference for the new candidate.

One candidate is nominated in advance: grouped-R. If it fails Stage 3, report the failure and trade-off, do not silently promote GCP or a different grouping and call it confirmed.

---

## 7. Algebraic and mechanism checks (before dataset runs)

Small numerical tests for: nonnegative normalized output masses, valid probabilities, evidence/opinion round trips; cautious idempotence on admissible binary opinions with a checked worked example; APP/OM exchange invariance; outer two-source symmetry where applicable; vacuous and missing-view cases; matching strong opinions, opposite strong opinions, one strong one weak, very small u. Confirm that repeating an identically discounted opinion within the same provenance group leaves its cautious aggregate unchanged, and do not claim invariance if duplication also changes fitted discounts. PCR6 is not associative, so preserve source-family metadata and do not claim arbitrary regrouping invariance.

On development data (fitting and selection folds only, fitting-calibrated branch thresholds), inspect APP+OM wrong / T correct, T wrong / spatial agreement correct, all correct, all wrong. Report counts, continuous scores, confident joint errors, and which rule rescues or reverses the useful view. This is where the grouped-R mechanism, letting the temporal view decide when both CLIP heads agree wrongly, is shown or not. Labels and generator identity are analysis-only, never inference inputs.

---

## 8. Freeze protocol, budget, and decisions before scoring

Write `tbiom/configs/grouped_fusion_preregister.yaml` before new candidate evaluation. Use the existing five source-grouped folds and v3.1 objective, keep every linked source and its transformed versions together, reuse frozen project tolerances if applicable. Otherwise use these planning defaults, practical choices not literature guarantees:
- Detection advancement: clean macro-AUROC gain >= 0.002 over both Dempster-R and AVG-R, paired 95% interval lower bound above zero for each.
- Safeguards: clean, processed, and development reenactment AUROC declines each no worse than 0.005 against either comparator, calibrated FPR rise and TPR loss each <= 0.01 absolute.
- Uncertainty guard, load-bearing. Any arm that degrades raw fused vacuity error-ranking below Dempster-R by more than 0.005 is disqualified as the reliability contribution regardless of its detection number, since the uncertainty signal is a headline claim. The PCR6-outer arm is the one most likely to fail this, by construction.
- Alternative reliability advancement: if detection is not superior, require AUROC noninferiority (paired lower bound > -0.005) against both comparators on clean and the prespecified processed and reenactment summaries, plus a development macro E-AURC improvement with its paired interval excluding zero. Use relative E-AURC reduction >= 5% only when the comparator's E-AURC is positive, otherwise declare the criterion undefined and do not claim it passed. Freeze the E-AURC uncertainty score to raw fused vacuity; entropy and margin remain diagnostics, not substitute endpoints chosen afterward.
- Cached CPU only, OMP_NUM_THREADS=8, same existing discount grid, no new continuous optimization or sweep, at most 2,000 paired source bootstrap draws per contrast, at most 8 total CPU wall-clock hours, report unfinished analyses rather than extending scope.

If noninferiority cannot be assessed reliably, report INCONCLUSIVE. Passing point-estimate safeguards alone does not establish noninferiority. Fold selection results are exploratory, independent confirmation is still required, and the headline detector claim needs the second seed from Stage 3.

---

## 9. Evaluation, statistics, prior art

Report raw values and paired differences against Dempster-R and AVG-R and all controls: clean development objective and components, JPEG q40, resize 0.60, reenactment, calibrated and historical-threshold operating points, NLL, Brier, existing ECE, vacuity error AUROC, predictive-margin/entropy error AUROC, AURC/E-AURC with exact conventions, within-dataset and within-processing uncertainty analysis, joint error and rescue diagnostics, conflict masses, vacuity and strength distributions, fitted weights in all folds, CPU latency. For logistic use its own predictive uncertainty score, not a blank.

Freeze the recipe before re-scoring the exposed ten-set suite and DF40. Report DF40 overall, swaps, reenactment, every generator, and both family-balanced and generator-macro means with explicit weights. Do not report only the favorable family. Compare groupings directly rather than inferring superiority from separate tests against a third method. Bootstrap source groups with linked manipulations together, synchronize resampling across overlapping folds, do not present five fitted folds as five detector seeds. Bootstraps quantify sample uncertainty conditional on the trained encoders, not encoder-training variability, and cannot supply unseen-generator generalization. Use a genuinely reserved confirmation set only if verified unused, otherwise label findings exploratory and identify the exact confirmation deficit.

Prior-art gate. Compare actual equations, assumptions, partition definition, and output uncertainty with: original PCR5/PCR6 (acknowledge two-source equivalence); Denoeux cautious fusion and extensions; Quost, Masson, Denoeux, optimized t-norm combination (2011), DOI 10.1016/j.ijar.2010.11.008; Chebbah et al., combining partially independent belief functions (2015), DOI 10.1016/j.dss.2015.02.017; Kallel and Le Hegarat-Mascle, cautious-adaptive rule (2009); DISCERN's confidence-modulated fusion, IBDC, and per-stream supervision; the actual DiCoME method; relevant conflict-aware evidential learning. If a primary source is unavailable, mark the comparison incomplete, do not invent a distinction. A new name or a successful combination of two established rules does not prove a new operator. Distinguish an architecture-informed application, a new empirical finding, and a new general method. Do not implement conflict-based dependence interpolation in this queue, and do not append IBDC or extra losses.

---

## 10. Deliverables and terminal outcomes

Versioned outputs under a new grouped experiment directory, never overwriting prior fusion results:
- `GROUPED_REFERENCE_MANIFEST.md`, `grouped_fusion_preregister.yaml`
- `tbiom/reports/EXP_GROUPED_CONFIRM.md` (Stage 3, the confirmation gate)
- `tbiom/reports/EXP_GROUPED_AUDIT.md` with algebraic tests and exact equations
- `tbiom/reports/EXP_GROUPED_DEVELOPMENT.md` and machine-readable metrics
- `tbiom/reports/EXP_GROUPED_EXPOSED.md` and machine-readable ten-set and DF40 metrics
- `tbiom/reports/GROUPED_PRIOR_ART_AND_CLAIMS.md` with a claim-to-evidence map

Return one or more of: grouped-R confirmed (CI excludes zero, controls lose, both seeds) / exploratory trade-off / no demonstrated advantage / implementation blocked / independent confirmation pending. The final recommendation keeps Dempster-R out of the proposed method but present as a baseline row, honestly reports every predefined candidate outcome including the swap-family cost, and states plainly whether a fusion-method novelty claim is supported. If grouped-R fails its confirmation gate, report that, do not promise novelty, and do not quietly switch the proposed method back to Dempster-R.

First run Stage 3 confirmation and the Stage 4 audit, then the finite matrix and analysis, with no further architecture-planning round.
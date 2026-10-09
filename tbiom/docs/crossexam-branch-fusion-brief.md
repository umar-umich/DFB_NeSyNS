# CROSS-EXAM: branch selection, spatio-temporal modeling, fusion and uncertainty

**Claude Code implementation brief, revised October 8, 2026**

Implement this as an extension of the current experiment repository. Preserve existing jobs, checkpoints, prediction exports and the previous fusion and causal-test records. This brief authorizes implementation and a bounded experiment campaign on available resources. It does not authorize replacing historical results or changing the dissertation's claimed configuration.

## 0. Purpose and what must exist at the end

The central question is whether a small set of branches adds complementary, transferable evidence, and whether evidential training or fusion improves reliability at an acceptable detection and compute cost.

The lab asked for six things, and each maps to a section.

| Request | Section |
|---|---|
| A PE-Core plus transformer branch, so the system has spatial, spatio-temporal and video branches | 2.2 and 7 |
| Try five strong branches and select the best three | 2 and 4 |
| Test results with and without uncertainty | 5 and 6 |
| Compare simple concatenation, weighted fusion and evidential fusion operators | 5 |
| LoRA ablations of rank, scaling and dropout | 8 |
| Individual and combined branch ablations in one combined table across datasets | 4 and 10 |

A PE-Core sequence transformer is a candidate branch, not an established contribution. V-JEPA 2 is a video encoder that carries spatial as well as temporal information, so it is never labelled "pure temporal".

Deliver runnable code, explicit configuration files, complete prediction provenance, the combined result tables and a decision report. Reuse working infrastructure rather than building another training framework. Discover existing entry points before writing new ones, and document the commands actually implemented.

**Interim report.** At 24 hours after launch, write `reports/branch_fusion/INTERIM.md` with the audit, the B0 reproduction, the branch configuration table and every cached-feature result completed so far. The final report follows at the end of the campaign.

## 1. Configuration and provenance audit

Read the repository instructions and the current run ledger. Locate the real encoder wrappers, head definitions, fusion operators, checkpoint selectors, dataset manifests and feature and prediction caches. Record their paths and the git commit in `reports/branch_fusion/audit.md`.

Resolve these items from code and artifacts before training.

**B0, the reference.** Identify one immutable current best verified reference. The expected candidate is the measured PE-Core-L/336 appearance view with LoRA plus LayerNorm tuning (development fused AUROC 0.9364, ten-set mean 0.9240), the existing CLIP off-manifold export and V-JEPA 2 with LoRA, fused with PCR6+-R. Confirm this from the run artifacts. Record:
- encoders and readouts
- projection widths
- adaptation recipes
- encoder input resolution and crop resolution
- clip sampling
- whether nuisance suppression is present
- calibration and fusion
- training data and checkpoint selection rule
- checkpoint hashes

Do not assemble B0 from the best cells of different runs.

**B0 reproduction tolerance.** Reproduce B0 from its caches. Accept the reproduction only if all of these hold:
- zero decision flips at the frozen threshold
- absolute AUROC difference of at most 0.0005 on every evaluation set
- Spearman correlation of at least 0.9999 between reproduced and recorded scores

Otherwise stop and report.

**B1.** Preserve B1, the global reliability-discount baseline, using its verified implementation and fitting protocol. Record whether it uses evidence scaling (`e' = r e`) or belief discounting. These are not equivalent.

**IBDC status.** Determine whether the heads-only one-sided IBDC experiment passed its recorded four-part rule. If that is not verified, label one-sided IBDC a candidate loss, not part of B0.

**Naming.** Resolve the legacy OM, LFS and facial-structure names from the actual computation. Keep `legacy_om` as an explicit identifier.

**Mixed systems.** Verify that upgrade screens replaced the appearance view while reusing old OM predictions (`upgrade/screen_eval.py:19`). Such a mixed system is a valid named baseline, but it is not a newly trained spatial pair.

**Caches.** Log preprocessing, output dimensions, prefix-token counts, token pooling, feature normalization, frozen and trainable modules, and all learned projections. Assert sample-ID and label alignment across every cache. Detect stale caches by model and configuration hashes and export provenance, not by expecting a particular AUROC value.

**PE-Core configuration.** The proposal text currently says PE-Core at 448 with LayerNorm-only tuning. That was a placeholder written before the screen ran. The measured best is PE-Core-L/14 at 336 with LoRA plus LayerNorm tuning.
- The primary appearance configuration in this campaign is the measured 336 configuration.
- Check which PE-Core checkpoints exist at 448. If the only 448 PE-Core checkpoint is the giant G/14 model, a 448 run is not a like-for-like change. Leave it out of this campaign and record that in the audit.
- Record crop resolution separately from encoder input resolution. A 256-pixel crop enlarged to 336 does not restore detail.

**Nuisance suppression.** Record from the artifacts whether B0 includes nuisance suppression. The feature-space nuisance diagnostic did not clear its permutation null, and the intervention-guided subspace study did not improve fusion.
- If B0 has no suppression, do not add it.
- If B0 has it, keep it fixed throughout and run one suppression-off ablation on the selected subset (Section 6).

**Correct these assumptions in inherited notes.**
- Shared features do not imply that fusion cannot help, because distinct heads can preserve different predictive information.
- LayerNorm adaptation changes activations and can change attention.
- LoRA does not force the original pretraining objective.
- Different recipes do not guarantee independent errors.
- Orthogonality is not causal identification.

## 2. Candidate branches

Five core candidates, plus one cheap side diagnostic.

| ID | Branch | Implementation | What it tests |
|---|---|---|---|
| A | PE spatial | Framewise PE-Core-L/336 (B0 recipe), global readout, order-invariant mean over frames, head | Facial appearance without sequence modeling |
| S | PE sequence | The same PE frame features as A, then a small temporal transformer and head | Added value of relations across frames |
| V | V-JEPA video | The verified V-JEPA 2 branch, its documented readout and head | Video-pretrained representation complementary to PE |
| O | Existing off-manifold | The verified CLIP/β-VAE residual computation and head (`legacy_om`) | Continuity with the existing system |
| M | VideoMAE video | The VideoMAE branch from the earlier encoder panel, reusing its checkpoint if it is on disk | A second video pretraining family. It led Deepfake-Eval-2024 in the panel (0.7279 against 0.7235 for V-JEPA 2) |
| R | PE patch relations (side diagnostic) | Intermediate PE patch tokens, covariance summary, frame mean, head | Local structure discarded by a pooled feature |

**Candidate rules.**
- If the VideoMAE checkpoint is missing, train it once under the recorded panel recipe. If that would exceed 8 GPU-hours, drop M and report it.
- R enters the subset search only if it beats its mean-pooled control on development with a source-grouped interval above zero (Section 2.3). Otherwise it is reported as a side result.
- Do not add DINO, I-JEPA, text projection or other backbones automatically. No branch is promised inclusion in the final three, including S.

### 2.1 Shared inputs and ordinary heads

Use the same face tracks, frame timestamps, video labels and processing draws across candidates.
- A, S and R use the same decoded frames.
- V and M keep their required clip protocols. Document every difference.
- Do not force unsupported input lengths.
- Report the number of unique observed frames and total decoded frames per branch.

For the controlled cache screen, freeze the representation producers and use their genuine exported features.
- Each branch gets a 256-dimensional head representation and a two-output prediction layer.
- Record native-to-256 projection parameters separately, because equal width is not equal parameter count.
- Start with dropout 0.1.
- Use the existing verified EDL loss and KL ramp, versioned and unchanged.
- Binary EDL uses nonnegative evidence, `alpha = evidence + 1`, `p_fake = alpha_fake / sum(alpha)`, and vacuity `u = 2 / sum(alpha)`.

For A, average the same per-frame representations used by S, then apply the video head. Keep the historical score-averaging appearance view separately if that is B0's implementation. This avoids attributing feature-versus-score pooling differences to temporal attention.

### 2.2 S: PE-Core plus a temporal transformer

**Architecture.** PE frame embeddings, then a linear projection to 256, then a timestamp-based temporal positional encoding, then two pre-norm transformer encoder blocks (4 heads, feed-forward width 1024, dropout 0.1), then a masked temporal mean, then the two-output head.
- Use valid-frame masks for padded clips.
- The time encoding must use real timestamps after subsampling, not an assumed constant frame rate.

S is a parallel branch with its own prediction. Reuse the V-JEPA branch's data loader where helpful. Do not insert PE vectors into the V-JEPA network or average them with V-JEPA features inside one branch, because that would confound the A, S, V comparison.

**Training data saturation.** The adapted PE features are near-perfect on FF++ training data: the adapted appearance view reaches about 0.995 on FF++ validation. A temporal module trained on top of them may see an already solved problem and learn nothing across frames.

Run two input variants:
- **S-adapt.** Built on A's adapted frame embeddings. Before training it, report the fraction of FF++ training videos that A already classifies with `p > 0.99` on the correct side.
- **S-frozen.** Built on frozen PE-Core-L block-16 patch-mean features. This was the strongest frozen PE readout (development 0.8019), and it is not fitted to FF++ labels.

Both are trained on FF++ train with identical recipes. The better of the two on development enters the subset search as S, and both are reported. If S-adapt shows saturation and S-frozen beats it, two-fold cross-fitted adaptation is a queued follow-up, not part of this campaign.

**Frame spacing.** Temporal relations need frames close enough to show motion. Earlier, shuffling 16 sparse frames of the video view changed AUROC by under 0.004.
- S runs first on the deployed frame set.
- Then run a dense-clip pilot, **S-dense**: 16 consecutive frames at stride 1 or 2 from the same interval. Use FF++ train and the development pool only.
- Report the dense pilot's extraction cost.
- S-dense is extended to the test sets only if it beats S on development with a source-grouped interval above zero.

**Cost accounting.** A and S share PE extraction cost only when weights, preprocessing and frames are identical. Measure the cost of any separate evaluation.

### 2.3 R: patch-relations side diagnostic

1. Read block 16 of the verified 24-block PE-L encoder.
2. Strip every prefix token, using the count from model metadata.
3. Project patch channels with a fixed, seeded orthonormal 32-dimensional projection.
4. For each frame, center the patches and compute `C_t = Z_tᵀ Z_t / max(N_t − 1, 1)`.
5. Vectorize the symmetric covariance with √2-scaled off-diagonal entries, giving 528 values.
6. Average over frames, standardize with training statistics, and fit the ordinary head.
7. Keep a log-norm feature so that magnitude is not discarded.

**Control.** Compare against mean-pooled features from the same block. This separates covariance from depth. Call R a patch-statistics branch, not a localization method.

## 3. Data access and fair training comparisons

Publish a data-access table before launching.
- All encoder adaptation, β-VAE fitting, S temporal-module training, R preprocessing statistics and representation projections use FF++ c23 training data only.
- Keep a genuine source, all its fake descendants and every corruption variant in one partition.
- Confirm duplicate identities and sources across datasets where metadata permits, and disclose unresolved grouping.

**Development folds.** Use the established five source-grouped development folds as the outer folds, so results stay comparable with B0.
- Inside each outer training part, create three inner source-grouped folds, stratified by dataset and class where grouping allows.
- Freeze all manifests.
- Each outer validation group must be absent from every fit or hyperparameter choice used for its prediction.
- Fit standardizers, projections, fusion weights and temperatures inside the relevant training fold.
- Source-trained representations may be cached once, because their training data are separate from development.

**Reserved evaluation groups.**
- Deepfake-Eval-2024 is the primary difficult-dataset audit, and DFDC is secondary. Both have been evaluated before, so do not call them untouched.
- If any of their test groups, or any other test-set groups, are inside the development pool, remove them and regenerate the manifests.
- If the pool changes, refit B0 and B1 on the new pool so every row shares it, and keep the historical B0 numbers as a separately labelled row.
- If source separation cannot be established, report the affected results as exploratory.

**Two protocol tracks.**

| Track | Neural heads and concatenation | Trust weights and calibration | Interpretation |
|---|---|---|---|
| P: source-trained reference | Fit on FF++ train, with checkpoint selection by B0's recorded rule (VALmix-v2 if the audit confirms it) | Only the declared low-dimensional trust fitting uses development. Temperature uses FF++ validation | Source-trained detector with development-fitted trust |
| D: learned-fusion comparison | Freeze source-trained representations. Fit small CE or EDL heads and the concatenation head on inner development training groups | Fit within the same inner procedure | Fusion and heads trained on development data |

Track D is needed to test learned fusion, because source-trained predictions are saturated.
- Label it in every table as "fusion/heads trained on development data".
- Give B1, probability averaging and weighting, concatenation and EDL operators the same development pool and folds.
- Report B1 with the original fixed heads and with Track-D refitted EDL heads.
- Never compare Track D with FF++-only literature baselines as if training access were equal.

**Selection.** Use inner validation for every architecture, subset, rank, loss, operator and stopping choice. For an unbiased estimate, select the subset and operator inside each outer fold and score that selected system on its outer validation group. The best outer-fold candidate mean is an exploratory statistic, not an unbiased estimate. After the selection rules are locked, fit on eligible development data and evaluate the locked model once on the test partitions.

## 4. Individual branches, subsets and selection

**Rows.**
1. Evaluate A, S, V, O and M individually.
2. Evaluate all 10 pairs and all 10 triples, using cached outputs and features. These 25 configurations are the core comparison.
3. Evaluate the 5 four-branch subsets and the five-branch ensemble as cheap diagnostic rows under the locked screen rule only.
4. Add one non-deployable reference row: the best single branch chosen per dataset (oracle).

**Screen rule.** Use B1 as the initial evidential screen rule, with uniform probability averaging as its inexpensive control. This separates branch selection from operator search.

**Shortlist for Section 5.** Compare the fusion families in Section 5 on:
- the triple A+S+V
- B0's triple
- the development-selected best pair and best triple

**Selection rule.**
- Do not choose the three highest standalone AUROCs.
- Select subsets by fused development macro AUROC, averaged equally across the fixed development datasets. Reenactment is secondary.
- Within 0.002 macro AUROC, prefer fewer encoder evaluations, then fewer trainable parameters.
- Keep the best pair as a deployment alternative if the third branch has no supported incremental benefit.
- Report whether each subset beats the oracle row on more than 4 of 10 test sets. Fusion that never beats the best single branch is selecting a view, not combining evidence.

**Branch contribution.** Refit fusion for the reduced subset with the same inner-fold budget. Removing a branch while keeping incompatible fitted weights is a separate sensitivity analysis.
- Report paired differences with 95% source-group bootstrap intervals, 2,000 resamples for final comparisons.
- Keep class membership and paired predictions consistent, and never treat frames as independent samples.
- Report seed variation separately from test-sample bootstrap intervals.

**Explanatory diagnostics.** Error correlation, rescue rate, harmful-overrule rate and zero trust weights explain results. They do not decide them.
- Fix classification thresholds on allowed validation data, and report class-conditional correlations.
- Do not promote a branch because its errors decorrelate, or reject it only because a fitted weight reaches zero in a fold.

## 5. Fusion, and "with versus without uncertainty"

Separate three questions:
- (a) CE versus evidential training of the heads.
- (b) Fusion of class probabilities versus fusion of evidence strength.
- (c) Uncertainty-based abstention at deployment.

Hiding an uncertainty output while leaving probabilities unchanged cannot change AUROC, and must not be reported as a new detector.

On the shortlisted subsets, run this matrix with identical samples, frozen representations and a declared track.

| ID | Comparator | Detail |
|---|---|---|
| F0 | Best single branch | Selected on development, not per test dataset |
| F1 | Uniform probability mean | On both CE and EDL-derived probabilities |
| F2 | Weighted probability mean | Nonnegative simplex weights, fitted on inner development only |
| F3 | Feature concatenation + CE head | Normalize each branch with training statistics, then concatenate. Linear and small-MLP versions |
| F4 | Feature concatenation + EDL head | F3's inputs and hidden architecture, EDL objective |
| F5 | Cumulative evidence | `alpha_f = 1 + Σ_v e_v`, prior counted once, with a duplicate-view diagnostic |
| F6 | Reliability-discounted cumulative evidence | B1's discount convention, equal weight-fitting budget |
| F7 | Reliability-discounted Dempster combination | Log normalization conflict and numerical fallback counts |
| F8 | Verified direct multisource PCR6+ | The existing validated implementation, named by variant and source |

**What each comparison isolates.**
- F1 on EDL probabilities against F6 to F8 with the same EDL heads isolates the use of evidence strength.
- F3 against F4 isolates the head loss under a fixed representation. Label it a frozen-representation head comparison, not a full training-objective comparison.
- For the final branch set, add matched source adaptation under CE against EDL if budget permits, with initialization, augmentation, epochs and schedule held fixed.

**Fitting rules.**
- Give the concatenation MLP the same hidden width as the standard head, and add a parameter-matched variant if counts differ materially.
- Never normalize with test statistics.
- Fit F2 weights with the same search budget as the evidential trust weights, starting from equal weights.
- Select by inner macro AUROC, with NLL as tie-breaker.

**Opinion algebra.** For binary opinions, `b_k = e_k / S` and `u = 2 / S`, with `S = 2 + Σ_k e_k`. Belief discounting with reliability ρ in [0, 1] gives `b'_k = ρ b_k` and `u' = 1 − ρ + ρ u`. Its equivalent evidence is

`e'_k = ρ u e_k / (1 − ρ + ρ u)`.

Evidence scaling `e'_k = ρ e_k` is a different operator. Test:
- endpoints at ρ = 0 and ρ = 1
- normalization and nonnegative evidence
- vacuous-opinion behavior
- high-conflict stability
- input-order behavior, including all six permutations of a triple for direct multisource PCR6+

Never relabel sequential pairwise PCR as a permutation-invariant multisource operator. If an exact implementation is unavailable, omit the row with a status note.

**Scope.** Do not add mixtures of experts, per-sample neural reliability controllers or interaction networks. The earlier learned-fusion gate stays closed unless this campaign produces a specific positive development result.

## 6. Stage 2 and the uncertainty–generalization tradeoff

**Nuisance suppression.** Follow the audit result from Section 1. Never change suppression while comparing fusion operators.

**IBDC schedules.** For the selected triple, compare matched heads-only schedules:
- EDL without a disagreement loss
- EDL with the verified symmetric IJCB objective
- EDL with one-sided IBDC

Freeze all backbones, adapters, S temporal blocks, the β-VAE, nuisance transforms and projections, so only prediction-head parameters update. Use identical data, epochs, optimizer and initialization per seed. In Track D these updates use development training folds. A source-only Stage 2 run, if any, is a separately labelled Track-P experiment.

**Disagreement weight.** Keep the EDL regularizer fixed. Sweep the disagreement weight λ in {0, 0.1, 0.3, 1.0}, after documenting the exact loss reduction and scale. λ = 0 is the shared control. Use a detached conflict target unless the verified method specifies otherwise, and log the choice.

**Tradeoff reporting.** Report detection AUROC against NLL, Brier score, ECE and selective risk across the sweep.
- Use 15 equal-width bins for the primary ECE, and also report 10-bin ECE for comparison with DISCERN.
- Plot development Pareto points with intervals and compute cost.
- Lock the operating point before test evaluation.
- A calibration gain with a detection loss is a tradeoff, not a win. Do not assume a tradeoff must occur.

**Temperature control.** Fit a positive scalar temperature by NLL on FF++ validation for each final probability predictor. In Track D, also report a separately named temperature fitted on inner development calibration data. Never fit on test. Use `sigmoid(logit(p_fake) / T)` with fixed clipping. Temperature scaling calibrates probabilities, not Dirichlet strength or vacuity.

**Selective prediction.** Compare entropy, 1 − max probability, EDL vacuity and verified conflict scores on the same correctness labels.
- Report risk at 100%, 95%, 90% and 80% coverage, plus AURC and E-AURC, with the oracle definition stated.
- Also freeze a validation-chosen rejection threshold, and report its achieved test coverage and risk.
- Fix the classification threshold independently of the rejection threshold.

This is the "with and without uncertainty" test result the lab asked for.

## 7. What "spatio-temporal" contributes

Run these on S, S-dense if it exists, V and M, with A as an order-invariant negative control, all on the same decoded frames:
1. Original order.
2. Five seeded random permutations.
3. Reversed sequence, reported separately.
4. One valid center frame repeated to the clip length.
5. For S, remove the positional encoding and retrain under the same budget.
6. For S, replace the transformer with temporal mean pooling, and with a parameter-matched per-frame MLP plus mean pooling.

**Rules for these tests.**
- Keep frame identities and intervals fixed for the order tests.
- Make processing augmentation clip-consistent, so it does not create flicker.
- Report paired changes in AUROC, NLL, scores and uncertainty.
- Repeat-frame results mix missing visual diversity with missing motion, so read them together with the shuffle results.
- An inference-time shuffle drop alone does not prove useful temporal reasoning.

**Naming.** S may be called a spatio-temporal architecture. A claim that temporal dynamics improve detection needs a positive matched control. V and M stay "video" branches in tables until their temporal use is shown.

## 8. LoRA ablations: rank, scaling, dropout

**Verify the implementation first.** Confirm the implementation uses `ΔW = (α / r) B A`. Log r, α, α / r, dropout, target modules and exact trainable count.

**Center configuration.** The center is B0's appearance recipe: LoRA r = 8, α = 16 (α / r = 2), dropout 0.1 on PE-Core's fused QKV projection, with LayerNorm tuning on. Every sweep keeps LayerNorm tuning on and target modules fixed, so the center matches B0. Write "LayerNorm" in full, because "LN" was previously used for LoRA plus LayerNorm.

**One-factor sweeps,** 10 unique configurations, since the center appears in all three:
- Rank: r in {2, 4, 8, 16}, α = 2r.
- Scaling: α / r in {0.5, 1, 2, 4}, r = 8.
- Dropout: {0, 0.05, 0.1, 0.2}, r = 8, α / r = 2.

**Controls at the center:** frozen, LayerNorm only, and LoRA only. A Q/V-only against full QKV comparison may follow on PE. Masked updates must keep the K update exactly zero, checked numerically.

**Which branches.** Screen on A first. Run the same design on V only if the budget allows. O keeps its recipe. S shares A's PE in the primary experiment, so do not adapt a separate PE for each S cell.

**Budget.** A PE-Core-L adaptation run took 100 to 320 minutes before, so the A sweep is roughly 17 to 55 GPU-hours. Measure one pilot, and drop the lowest-value cells first (dropout 0.05, then α / r 0.5) if the cap would be exceeded.

**Seeds and fairness.** Use one seed for screening, then three adaptation seeds for the center and the selected recipe. A seed means a new adaptation run. Keep the data budget, effective batch size, augmentation, schedule and early stopping equal, and record peak memory, throughput and GPU-hours.

**Evaluation.** Evaluate each cell standalone, and by its fused contribution in one fixed reference subset: B0's triple with A swapped in. Do not re-run the subset and operator search per LoRA cell. One-factor sweeps are not an interaction study. At most two predeclared follow-up combinations may be tested on development data.

## 9. Metrics, generalization and explanations

**Detection metrics.** All detection tables use video-level AUROC in [0, 1], with full precision in machine-readable files and four decimals on display.
- Final datasets: CDF-v2, CDF-v3, DFD, DFDC, DFDCP, Deepfake-Eval-2024, FFIW, KoDF, IDForge, DFDM.
- DF40 is reported by family (face swap, reenactment, synthesis, editing), as mean ± standard deviation across generators. Note that DF40 informed earlier fusion design.
- Macro means use exactly the datasets displayed. Missing cells are marked and excluded from a clearly named common-set mean, never filled from another run.

**Further metrics.** Report fixed-threshold FPR and FNR on real and fake videos, balanced accuracy, NLL, Brier score, ECE, error-detection AUROC and selective risk.
- Fit thresholds on validation data only, with a recorded objective (initially maximum balanced accuracy).
- Bootstrap by video-source groups.
- Give sample counts and class prevalence for every table.

**Robustness.** Reuse the B processing mixture for training if B0 does. Keep JPEG q40 and resize 0.60 as held-out stress settings. Evaluate original and corrupted versions of the same videos. Report clean macro AUROC, corrupted macro AUROC and worst-dataset AUROC separately.

**Explanations.** Do not present branch diversity as explainability. Provide:
- per-branch probabilities and vacuity
- refitted subset gains
- temporal and region perturbation responses
- cases chosen by predefined bins and seeded sampling, including false positives and confident errors

If region explanations are added, compare deleting attributed regions with deleting area-matched random regions under the same replacement. Attention maps alone are not faithful explanations. No causal claim follows from a projection, an attribution map or a shuffled sequence.

## 10. Required tables and figures

Generate CSV and LaTeX from one canonical results file, and figures as PNG only. No hand-edited numbers. Each canonical record holds:
- model ID, protocol track, checkpoint and config hash
- seed, fold, subset, operator and calibration
- dataset, split and corruption
- sample counts and metric fields

**1. Branch configuration table.** For A, S, S-dense, V, O, M and R: encoder and readout, input and crop resolution, frame count and spacing, trainable and total parameters, adaptation, training access, latency and GPU memory.

**2. Combined branch-and-subset table.** This is the table the lab asked for.
- **Rows, grouped:** singles (5), pairs (10), triples (10), four-branch (5), all five (1), then the oracle reference row.
- **Columns:** the ten test sets, the ten-set macro, the four DF40 families, DF40 macro, and real-video FPR at the frozen threshold.
- Use the same layout for a development out-of-fold version, kept as a separate table.
- Test rows use locked settings only. Mark the preregistered selected subset with ★ even if another row scores higher. Bold the best value per column.
- Also produce a subset-by-dataset heatmap PNG of the same numbers.

**3. Fusion table.** Shortlisted subsets crossed with F0 to F8: macro AUROC, Deepfake-Eval-2024 AUROC, NLL, ECE, AURC and cost. Keep CE, EDL-probability and evidence-operator rows separate.

**4. Uncertainty table.** Raw and temperature-scaled predictions, with entropy, margin, vacuity and conflict error ranking and selective risk. Mark unsupported types "not applicable".

**5. LoRA table.** Every tested r, α, dropout and module configuration, with trainable count, standalone and fixed-subset fused metrics, cost and seed status.

**6. Temporal-control table.** A, S, S-dense, V and M under the order, repetition and pooling controls, with paired changes and intervals.

**7. Tradeoff figures.**
- Development AUROC against NLL and against selective risk, with nondominated configurations labelled.
- Clean against corrupted AUROC.

**8. Decision report.** The selected triple, the best pair, each branch's incremental contribution, whether fusion beats the oracle row, the uncertainty benefit or tradeoff, temporal evidence, compute cost, negative results, and the next experiment the results justify.

Use compact IDs with a separate legend. Report three-seed means and dispersion for final models, never the best seed. Keep one-seed screens visibly labelled.

## 11. Execution order, lanes, budget and stop rules

**Order.**
1. Audit, manifests, cache signatures, B0 reproduction and fusion unit tests.
2. Shared PE frame and patch extraction. Train A, S-adapt, S-frozen and R heads. Export O, V and M. Run the temporal controls on S and V.
3. Singles and all subsets under the screen rule, with nested selection on development. Write the interim report.
4. F0 to F8 and the CE against EDL head comparisons on the shortlist. Start the LoRA sweep on A, and the S-dense pilot.
5. IBDC, calibration and suppression comparisons on the selected subset, independent-seed confirmation, then locked test evaluation.

**GPU lanes.** Inspect available GPUs and existing reservations, and never stop or overwrite unrelated work. As a default:
- **Lane 1:** audit, cached-feature heads, subset search and fusion matrix, with the IBDC heads-only runs later.
- **Lane 2:** PE extraction, then the A LoRA sweep.
- **Lane 3:** VideoMAE export or training, S training and temporal controls, then the S-dense pilot.

**Budget.** The campaign has a 72-hour wall-clock cap from launch. Estimate costs from short pilots, keep a GPU-hour ledger, and launch a job only if it can finish within the cap. Prioritize the audit and cached comparisons over adaptation sweeps. At the cap, stop launching, save consistent checkpoints, and report completed, failed and deferred cells with a resumable queue ordered by expected value. Do not fabricate complete-looking tables.

**Hard stops.** Fix these before interpreting any gain:
- leakage or sample misalignment
- an unreproducible B0 under the Section 1 tolerance
- invalid operator normalization
- changed inputs hidden under the same cache ID

**Decision rules.**
- A weak S does not block the other subsets.
- If no triple improves on the best pair with a supported paired gain, keep the pair and report the three-branch hypothesis as unsupported.
- If EDL improves only calibration at a detection cost, report the Pareto tradeoff.
- If calibrated CE matches EDL's benefit, report that directly.

**Not in this campaign.** Do not expand fusion capacity or adapt a second PE trunk to force a triple. These stay in the queue:
- the off-manifold width and depth arms (1024-to-64 projection, β-VAE latent size, mid-depth readout)
- the text-anchored manifold
- two-fold cross-fitted S
- PE-Core-G/448

## 12. Acceptance checks and handoff

- A and S use identical frame caches in their controlled comparison, and R strips the correct prefix tokens.
- A is invariant to frame permutation within numerical tolerance. Order experiments change only the intended variables.
- No training, calibration or selection group overlaps an evaluation group.
- Every combined-table row traces to exact branch exports. No historical OM substitution is hidden.
- CE and EDL controls share representation and track whenever a head-level claim is made.
- Operators stay finite and normalized at vacuity and high conflict, and endpoint discount tests pass.
- All reported means match the displayed members, and uncertainty metrics use the declared definitions.
- Track D is labelled "fusion/heads trained on development data" everywhere.
- Test evaluation uses locked settings, and all selection claims come from development.
- The report includes the best pair, the selected triple, every completed ablation and an honest budget and status ledger.

Return the implemented commands, configuration locations, result paths, brief findings and a concrete decision. Update research notes with new measurements. Do not change dissertation numbers or architecture claims.

## References

- Perception Encoder, Bolya et al., arXiv 2504.13181. Intermediate layers motivate the depth and readout controls.
- V-JEPA 2, Assran et al., arXiv 2506.09985. Video pretraining motivates testing temporal contribution rather than assuming it.
- VideoMAE, Tong et al., NeurIPS 2022.
- LoRA, Hu et al., arXiv 2106.09685. Verify the installed scaling implementation.
- Evidential Deep Learning, Sensoy et al., arXiv 1806.01768.
- The repository implementations and their cited sources remain authoritative for the exact IBDC and PCR6+ variants. Do not build an operator from its name alone.
# DISCERN-v2 joint IBDC brief

## Goal

Test whether a corrected Inter-Branch Disagreement Calibration (IBDC) loss, applied during joint training of the evidential heads, improves how well fused evidential vacuity ranks errors without hurting detection.

The IJCB version of IBDC matched fused uncertainty to branch disagreement in both directions. Reviewers 2 and 3 pointed out two failures. When all branches are uncertain they agree, so the loss pushed uncertainty down (shared ignorance). The model could also satisfy the loss by rescaling evidence without identifying hard samples (self-reference). The corrected version is one-sided, computed from discounted belief masses and detached.

This study trains only the evidential heads on cached features. All backbones, LoRA adapters and the β-VAE stay frozen.

## Fixed parts of the system

- **Views.** Appearance, the deployed Off-manifold view and the V-JEPA 2 video view. If the I-JEPA study later keeps a row, this study is rerun with that view in the artifact slot.
- **Fallback.** Check the PCR6+ study report first. If its "video plus one spatial view" row is within −0.005 of the three-view system on the development pool and Appearance has zero trust weight in at least 4 of 5 folds, run everything below with Off-manifold and video only, and state this in the report. If the PCR6+ report does not exist yet, use three views.
- **Fusion.** PCR6+-R with the deployed implementation. Record its commit hash.
- **Trust weights.** Fixed at the deployed staged values during training. For evaluation they are refit out of fold on the 5 source-grouped development folds, exactly as deployed.
- **Threshold.** Refit per run at FF++ c23 validation EER, then frozen for all tests.
- **Protocol v3.1.** Train on FF++ c23 only. Ten test sets, DF40 (41 sets), processed tests.

Locate the deployed configuration and state which repository holds it. Put new code in `training/joint_ibdc/` and outputs in `analysis/joint_ibdc/`. Do not modify deployed configs or checkpoints.

## Rules

- Never fabricate a number. Unrun cells are `TODO(run)`.
- Decisions use FF++ validation, VALmix-v2 or the development folds. Test sets are evaluated once per final run and reported, never used to choose anything.
- Gates run in order. A failed gate stops the study and is written up as a finding.
- Call u = K/S "evidential vacuity" everywhere in code comments, tables and the report. Never call it epistemic uncertainty.
- Every number in the report cites the file or command that produced it.

## Notation

For branch b with evidence e_b over (real, fake), S_b = e_b(real) + e_b(fake) + 2.

- Belief masses m_b(k) = e_b(k) / S_b and vacuity u_b = 2 / S_b.
- Discounted masses with trust weight r_b are m̃_b(k) = r_b · m_b(k), and ũ_b = 1 − r_b (1 − u_b).
- Pairwise conflict κ_ij = m̃_i(real) · m̃_j(fake) + m̃_i(fake) · m̃_j(real).
- κ = stop_gradient(max over pairs i < j of κ_ij).
- u_f is the fused vacuity from PCR6+-R.

A branch with r_b = 0 contributes no conflict, which is consistent with fusion ignoring it.

## Task 1. Differentiable fusion and unit tests

Write a PyTorch implementation of the deployed discounting and PCR6+-R fusion. It must reproduce the deployed fused opinions on 1,000 cached development videos to within 1e-6. Stop if it does not.

Implement the one-sided IBDC loss.

L_IBDC = mean over x of max(0, κ(x) − u_f(x))

Port the original IJCB IBDC exactly from the DFB_NeSyNS repository, including its definition of disagreement d, as `ibdc_v1`. Record the file and commit it came from.

Add `training/joint_ibdc/test_ibdc.py` with pytest assertions.

- **Shared ignorance.** All branches with evidence (0.01, 0.01) give κ below 1e-3, loss 0 and zero gradient.
- **Confident disagreement.** Branch evidence (100, 0) against (0, 100) gives loss above 0, and a gradient step raises u_f.
- **Confident agreement.** Two branches with evidence (100, 0) give loss 0.
- **Discounted branch.** A branch with r = 0 contributes κ = 0 to every pair.
- **Stop-gradient.** Backpropagating L_IBDC gives gradients to the evidence only through u_f, never through κ.
- **Original IBDC.** Under shared ignorance, `ibdc_v1` produces a gradient that lowers u_f. This documents Reviewer 3's failure case.

Run the tests and include the full output in the report.

## Task 2. Feature cache

Cache the exact input of each view's evidential head.
- Per frame for Appearance and Off-manifold.
- Per clip for the video view.

Confirm from the deployed code how each view aggregates to video level (averaging evidence or averaging opinions) and reproduce it exactly.

For FF++ c23 training data, cache 3 independent augmentation draws per training sample, each following that view's deployed training augmentation (the B processing mixture for the video view). Training samples one draw per sample per epoch. Cache FF++ validation, VALmix-v2, the development pool, the ten test sets, DF40 and the processed tests clean or at their defined settings.

Check that the deployed heads applied to the cache reproduce the deployed per-view opinions on 1,000 development videos to within 1e-5. Stop if they do not.

Record cache size and extraction time.

## Gate 0. Premise and activity

**G0a, premise.** Use J0 (the deployed staged system) out-of-fold development opinions. The error label is whether the fused decision at the frozen threshold is wrong. With 5-fold source-grouped cross-validation inside the development pool, fit logistic regressions that predict the error from each of these feature sets.

1. u_f alone
2. κ alone
3. u_f and κ
4. u_f and the three κ_ij

Report the AUROC of each, plus the difference between sets 3 and 1 with a source-grouped bootstrap 95% interval (1,000 resamples). **Pass if the interval for set 3 minus set 1 lies above zero.** If the PCR6+ study already produced this comparison with the same definitions, reuse and cite it instead of recomputing.

**G0b, activity.** On the FF++ training cache, using J0 heads, report the distribution of κ and the fraction of training samples where κ > u_f (where the hinge would be active), per augmentation level. **Pass if the hinge is active on at least 1% of training samples.** Training loss saturates in this system, so a near-zero rate would mean IBDC has nothing to act on under this protocol. That is a finding, not a reason to change the protocol.

**Two-case diagnostic**, reported whatever the gates show. On the development pool, count videos where the video view is correct and both spatial views are wrong, and the reverse, at each view's own FF++ validation EER threshold.

## Runs

All runs start from the deployed head weights and train only the heads. Each run uses 3 seeds (0, 1, 2).

| Run | Loss |
|---|---|
| J0 | Deployed staged heads, evaluated through the Task 1 fusion code. No training |
| J1 | EDL(fused) + 0.5 × Σ_b EDL(branch b before discounting) |
| J2a | J1 + λ · ibdc_v1 |
| J2b | J1 + λ · L_IBDC |

- **EDL variant and KL ramp.** Use the deployed variant for every EDL term. Confirm it from the deployed config.
- **Branch terms.** These act on raw evidence before discounting, so a heavily discounted branch keeps learning.
- **Optimizer.** Same as the deployed heads. Epochs at most 20, with VALmix-v2 checkpoint selection as deployed.
- **λ.** For J2a and J2b, run λ in {0.1, 0.3, 1.0} with seed 0. Choose the λ with the highest fused vacuity error AUROC on VALmix-v2, among those whose VALmix-v2 score is within −0.005 of J1's. Then run the chosen λ with seeds 1 and 2. The development pool is not used to choose λ.

**Collapse check after J1.** If any view with nonzero trust weight has a leave-one-out contribution on the development pool that is at most zero in J1 but positive in J0, stop after J1 and report. Joint training has then reproduced the known complementarity collapse, and a remedy will be designed separately.

## Metrics

Report every metric per seed, as the mean ± standard deviation over seeds, and with source-grouped bootstrap 95% intervals. Compute on the development pool (out of fold) and on each test set.

**Detection.** Video AUROC for all ten test sets and their mean, the DF40 mean over 41 sets, the four processed-test cells, and FPR and TPR at the frozen threshold.

**Evidential vacuity.**
- Error AUROC of u_f, where the error is the fused decision at the frozen threshold.
- AURC and E-AURC.
- Selective risk at 10% abstention, meaning the error rate on the 90% of videos with the lowest u_f. State this definition in the report, since Reviewer 1 asked.

**Calibration of the fused projected probability.** Adaptive ECE with 15 equal-mass bins, 10-bin ECE for continuity with the IJCB paper, Brier score and NLL.

**Shared-ignorance analysis.** Set thresholds on J0 FF++ validation opinions and freeze them.
- Mean branch vacuity (before discounting), 20th and 80th percentiles.
- κ, 20th percentile.

Subsets on each evaluation set.
- Shared ignorance (SI) is mean branch vacuity at or above its 80th-percentile threshold with κ at or below its threshold.
- Informed agreement (IA) is mean branch vacuity at or below its 20th-percentile threshold with κ at or below its threshold.

Report the size of each subset, the mean u_f on SI, the error rate on SI, and the inversion rate. The inversion rate is the probability that a random SI video has lower u_f than a random IA video.

**Agreement-on-wrong rate.** Among fused errors, the fraction where every branch with nonzero trust weight predicts the same wrong class with vacuity below its FF++ validation median.

**Leave-one-out contribution** of each view on the development pool, with intervals.

**Uncertainty baselines** for J2b.
- Deep ensemble. Use the three J2b seeds, and report the mutual information of the ensemble's fused Dirichlet predictions.
- Monte Carlo dropout. Run 20 passes if the heads contain dropout. If they do not, write "not applicable, heads have no dropout".
- For each baseline, report its error AUROC next to u_f's, and the Spearman correlation between u_f and mutual information.

## Pass rule for J2b (pre-set)

J2b is the corrected IBDC result only if all four hold against J1, on the development pool, using the three-seed mean.

1. Fused vacuity error AUROC is higher, with the bootstrap interval of the difference above zero.
2. Development AUROC is within −0.005.
3. Every view with nonzero trust weight has a positive leave-one-out contribution.
4. Mean u_f on the SI subset is not lower than J1's.

Report separately whether J2a lowers mean u_f on the SI subset relative to J1, which is the failure Reviewer 3 predicted. Report either outcome.

The test-set success criterion, reported and not used for selection, is that J2b's ten-set mean AUROC is within −0.005 of J0's.

## Report `analysis/joint_ibdc/JOINT-IBDC-REPORT.md`

1. Verdicts for Task 1, Task 2, G0a, G0b, the collapse check and the J2b pass rule, each PASS or FAIL with its evidence file and one sentence.
2. G0a table, G0b distribution table and the two-case diagnostic counts.
3. Detection table in the same format as the main results table, with J0, J1, J2a and J2b.
4. Vacuity and calibration table.
5. Shared-ignorance and agreement-on-wrong table.
6. Leave-one-out table.
7. Uncertainty-baseline table.
8. The chosen λ values, all λ-sweep results, the EDL variant, the aggregation rule, commit hashes, cache paths and GPU time.
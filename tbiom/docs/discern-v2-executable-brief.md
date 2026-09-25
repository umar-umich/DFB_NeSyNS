# DISCERN-v2 — Final Executable Brief (audit, reliability pilot, augmentation pilot, bounded temporal)

For the coding agent in the live tree (DFB_NeSyNS, DISCERN_Ext). Deliberation is closed. Begin the audit immediately. No further architecture brainstorming before the audit numbers are in.

Scope note stated once. Stopping an experiment for this paper does not require proving the research direction cannot work. A stop means insufficient demonstrated benefit under the tested configurations within the allocated budget, and it is a valid, reportable outcome, not a claim that the signal is absent.

Reference architecture, frozen for every experiment. Three branches: V-JEPA-2 LoRA (B-augmented) temporal, CLIP ViT-L LoRA Appearance, β-VAE Off-manifold. Primary fusion DS-R, logistic stacking as the matched comparator. Face-prior is out of the reference and appears only in labelled historical comparisons.

---

## Stage 0 — Audit first (do before any fitting)

Resolve from the live code and reports, do not guess, and write `tbiom/reports/REFERENCE_MANIFEST.md`.

1. The three reference checkpoint paths and hashes for V-JEPA, Appearance, Off-manifold, their preprocessing, and the clip/frame aggregation order.
2. The actual DS-R discounting equations, read from code. Confirm whether the implementation uses evidence scaling or belief-mass discounting, because they are different operations and the first controller experiment must reuse whichever one is actually implemented, unchanged. Record the exact equations.
3. Cache inventory: do aligned segment-level opinions from all three views exist, with timestamps and per-segment quality descriptors (blur, blockiness, contrast, motion, luma), or only frame-level spatial opinions plus one video-level V-JEPA opinion. Record segment granularity explicitly.

Gate. If aligned segment-level opinions and quality descriptors do not exist at segment granularity, the reliability pilot is not cached-only, it needs local clip inference to generate them. Generate those missing inputs before fitting any segment controller, and record that this adds GPU cost. Do not simulate per-segment evidence by repeating a single video-level opinion across timestamps, that creates no temporal signal.

Pass condition. The manifest names the three checkpoints, the exact DS-R discount rule from code, and the true segment granularity of opinions and descriptors. Every later experiment cites this manifest.

---

## Stage 1 — Two controlled pilots, in parallel

Both use the frozen reference branches. They do not share checkpoints with each other, and the reliability controller in 1A must use the original frozen reference, never outputs from the changing checkpoints of 1B.

### 1A — Reliability controller ladder

Predict per-view participation with the branches frozen, changing only the weight predictor, and apply the exact DS-R discount rule confirmed in Stage 0. Do not change the discount equation and the predictor at the same time.

Controllers, identical inputs, fitting splits, discounting, aggregation, and tuning budget across all four:
- Constant DS-R (the current fitted weights), reference.
- Segment-wise MLP on current-segment features.
- Small GRU over segments.
- Small Mamba over segments.

Controller inputs: per-view branch probabilities and vacuities, cross-view disagreement, the per-segment quality descriptors, and availability. Include branch probabilities, not vacuity alone, since vacuity alone may not flag a confidently wrong branch. Train on the fused prediction loss over clean and processed examples, branches frozen, and initialize near the current constant weights so the model earns departures. Do not assign low reliability merely because a segment was heavily compressed, some branches stay correct under compression. Aggregate acquired segments within each branch before cross-view fusion, and deduplicate overlapping clips so the same frames are not counted as independent evidence.

Two success questions, kept separate. Adaptive reliability succeeds if the MLP or a sequence controller beats constant DS-R. Mamba earns inclusion only if it beats the simpler adaptive controllers on performance or measured efficiency. The reliability mechanism does not fail because the MLP wins.

Controls: remove quality descriptors, shuffle segment chronology, reassign quality trajectories across videos without labels, and compare all controllers against the same clean-plus-processed logistic stacker. A bounded stress condition where quality changes between contiguous segments, identically distributed across classes, tests whether history helps, uniform degradation throughout every video does not.

Report to `tbiom/reports/EXP_RELIABILITY_CONTROLLER.md`: development macro AUROC clean and processed, reenactment, FPR/TPR at the frozen tau, error-detection AUROC and risk-coverage on the new fused output, and compute.

### 1B — Matched augmentation extension

Test whether composed processing helps beyond the mutually-exclusive B mixture. Matched initialization, training budget, sample count, class balance, and optimizer against the existing B control.

| processing | existing B | extended |
|---|---:|---:|
| unchanged | 40% | 40% |
| JPEG only, existing training severities | 30% | 20% |
| resize only, existing training scales | 30% | 20% |
| resize then JPEG | 0% | 20% |

Rules: draw transformations independently of label and manipulation type, apply consistent parameters across all frames of a clip (independent per-frame transforms manufacture flicker), give all three branches the same transformed video, keep every clean and processed version of a source in the same fitting or selection group, sample on the fly for training and use fixed manifests for repeatable development evaluation, keep q40 and resize 0.60 out of training. Adapt only the processing-sensitive branch identified by Experiment 3's DFDCP diagnosis, and treat Appearance and Off-manifold as one coupled spatial experiment if they share the adapted CLIP path, preserving the reference checkpoints and the β-VAE configuration. After any backbone change, regenerate opinions and refit fusion, old caches are not representative.

This tests whether composition helps beyond individual transforms. It does not test the ceiling of FF++ generalization, and one unsuccessful extension does not establish such a ceiling. Report to `tbiom/reports/EXP_AUGMENTATION_EXTENSION.md`.

---

## Stage 2 — Bounded temporal experiment

A predefined sparse/dense by pooling/small-temporal-head comparison on a bounded development subset with raw video. Fix the configurations and the maximum training budget before launch, and run no additional architecture sweep afterward.

| sampling | existing pooling | small temporal head (GRU or temporal conv) |
|---|---|---|
| current sparse | reference | head effect at sparse |
| dense | sampling effect | sampling-plus-head effect |

Define physical windows explicitly and include a same-window dense-versus-subsampled comparison, since changing stride at fixed frame count also changes duration, report frame count and compute for each. Evaluate ordered, reversed, locally permuted, and fully shuffled inputs as diagnostics with paired score changes and AUROC intervals. Linear probes, reversal, and shuffling are diagnostics, not success requirements, no arbitrary shuffle-gap threshold gates anything. If a linear order probe is run it is optional and diagnostic, a failed probe does not stop temporal work and a passed one does not authorize a full build.

Mamba enters as a temporal head only if this comparison shows a small temporal head is useful, and then only within the same bounded budget.

### The stopping rule, precise, set before launch

Continue temporal-architecture development only if the selected temporal configuration improves fused development performance by a predefined practically meaningful margin, its paired source-grouped confidence interval excludes zero, and it meets the predefined robustness, reenactment, and compute constraints. Select the candidate on the fitting and selection data, and assess this rule on a separate development confirmation partition, because picking the best head and computing its interval on the same examples makes the evidence optimistic. Record the margin, the constraints, and the partition split before running.

If the temporal pilot fails this criterion within its allocated budget, stop temporal-architecture development for this paper and report insufficient demonstrated benefit under the tested configurations. Do not translate that into "temporal information is absent." Report to `tbiom/reports/EXP_TEMPORAL_SAMPLING.md`.

---

## Stage 3 — Combine and confirm only successful changes

Take only the changes that passed their pilots. Regenerate predictions from the improved branch recipe and refit the selected reliability controller on the same source-grouped folds, then measure whether the gains survive together, since branch adaptation and dynamic discounting can interact. Confirm the selected combined recipe across training seeds. Keep test results out of selection. FaceShifter is already inspected and cannot serve as a newly sealed confirmation set for these changes.

Report to `tbiom/reports/EXP_COMBINE_CONFIRM.md`.

---

## Acceptance, metrics, and standing rules

- Freeze acceptance tolerances on development before each pilot launches. Report clean and processed discrimination, reenactment, FPR/TPR at the prescribed frozen threshold, uncertainty quality, and compute. Use predefined regression tolerances. Do not use CDFv3 test performance to accept or reject an arm.
- Threshold discipline. Keeping the old numeric threshold tests drop-in compatibility, refitting each model's threshold by the same calibration procedure tests comparative operation, report both, refit neither per test condition.
- Report vacuity error-ranking within datasets and processing conditions, not only pooled, since pooled ranking can reflect between-dataset differences.
- Data claim, stated plainly in the paper. FF++-only encoder training is not an FF++-only complete system if the fusion learns from labelled external development data. Disclose the fitting access, and give competing controllers the same access.
- Numbers from result files, not memory. Unrun cells `TODO(run)`.
- Do not, in this queue: run an encoder sweep, add human-in-the-loop or RL, sweep the augmentation or FreqBlender fraction, or change the DS-R discount equation and the weight predictor in the same run.

Begin Stage 0 now. Stages 1A and 1B run in parallel after the audit. Stage 2 runs on its bounded budget with the stopping rule fixed before launch. Stage 3 combines only what passed. The next report should be the audit numbers, not another plan.
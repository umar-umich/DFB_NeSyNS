# Claude Code instructions: prepare the complete T-BIOM evidence handoff

## 1. Objective and stopping point

Prepare a comprehensive, code-grounded handoff package for writing our DISCERN journal extension and later integrating it into my dissertation proposal.

This is an evidence consolidation, documentation, and export task. We are moving into writing. Do not initiate another research cycle. Give us as much usable material as the existing implementation, reports, configurations, and prediction caches support.

Work through the task and produce the actual files. Do not respond with only a plan, another operator recommendation, or an offer to export later. When information is missing, record a precise placeholder and continue with everything else.

Read the applicable AGENTS.md and locate the live DFB_NeSyNS / DISCERN_Ext project using the working directory and known project locations. Respect local instructions. Do not search unrelated personal directories or modify unrelated work.

## 2. Settled decisions

- Target: IEEE T-BIOM extension of our IJCB 2026 DISCERN paper, “Uncertainty-Aware Deepfake Detection via Multi-View Structural Learning.”
- Working framework: appearance (APP), off-manifold (OM), and adapted V-JEPA temporal evidence, with the actual final configuration established from code and current reports.
- We are proceeding with a selected PCR6-based variant or combination. Document the exact selected implementation; do not replace it with Dempster-R because of a small AUROC advantage.
- Dempster-R is not our proposed method and is not a mandatory manuscript comparison row or selection target. Preserve historical results unchanged. DiCoME remains an ordinary SOTA method in comparative results.
- The earlier grouped-fusion construction is closed as a candidate central contribution. Archive its findings accurately; do not restart it.
- We accept competitive rather than universally best AUROC when the framework provides a demonstrated methodological or reliability benefit.
- A logistic error predictor is a diagnostic probe unless explicitly included in the selected method. Do not silently make it the proposed fusion or deferral mechanism.
- Use a clear, affirmative account of demonstrated findings. Put material qualifications in protocol notes and an organized evidence map instead of repeating warnings in every summary.
- This package is for the writers. Do not draft a full journal paper or reorganize the dissertation here.

## 3. Task boundaries

Allowed:
- Read code, configs, logs, reports, existing statistical outputs, and prediction-cache schemas.
- Export existing metrics and predictions, create readable summaries, and document the actual implementation.
- Recompute an existing metric using the established evaluation script and existing cached predictions when necessary to complete an export or resolve an inconsistency.
- Produce inexpensive descriptive summaries from existing data, such as sample counts or reliability-weight frequencies, with their provenance.
- Run existing fast algebraic/format checks relevant to the selected implementation.

Not part of this task:
- Encoder training, new checkpoint inference, new datasets, new backbones, or fresh operator sweeps.
- New hyperparameter searches, reliability-weight fitting, threshold tuning, uncertainty-score fitting, or changing the selected recipe.
- Replacing the current evaluation protocol, creating a new acceptance gate, or selecting the method on exposed test results.
- Overwriting historical reports, removing failed studies, resetting the repository, or modifying existing jobs.
- Downloading raw media, uploading checkpoints, or creating huge cache copies merely to prepare the handoff.

Use existing trained/fitted artifacts. If the current final-variant experiment is already running, inspect its existing status and outputs without stopping, restarting, or changing it. Mark its unfinished results as pending. This brief does not grant new experimental authorization.

For necessary metric recomputation, use CPU only, at most eight threads and a total two-hour CPU wall-clock budget. Do not rerun expensive bootstraps just to regenerate confidence intervals that already exist. Report unfinished export items rather than expanding the budget or substituting estimates.

## 4. Establish the final-method identity first

Identify the user-selected PCR6 configuration from the latest explicit decision and its implementation/configuration. Record:

- Exact operator: PCR6, PCR6+, sequential pairwise, direct multisource, or a specified composition.
- Branches and checkpoints.
- Reliability-discount definition and fitted weights.
- Temporal/spatial training recipe and seed configuration.
- Aggregation, probability conversion, uncertainty outputs, and decision threshold.
- Current method/config identifiers and code commit.

Do not infer that the variant with the largest test AUROC is the final choice. Do not assume that old PCR6-pairwise results belong to direct PCR6 or PCR6+.

If no final configuration is actually selected, set `final_selection_status: unresolved`. Export the completed PCR6 candidates separately, identify the minimum selection information needed, and continue preparing the package. If the selected variant is still being evaluated, set `final_selection_status: selected_results_pending`. Missing final results should not block documentation of the implemented method.

Use the title **“Complementary Spatial and Temporal Evidence for Reliable Deepfake Video Detection”** only as a working paper label. Do not rename methods, checkpoints, or directories to imply a finalized publication identity.

## 5. Output location and required files

Create a new, clearly versioned handoff directory inside the project's documentation/report area, for example `tbiom/paper_handoff/<date>_<short-commit>/`. If that directory already exists, create a numbered successor. Preserve earlier handoffs and reports.

Produce:

1. `00_START_HERE.md`
2. `01_FINAL_METHOD_SPEC.md`
3. `02_EXPERIMENT_PROTOCOL.md`
4. `03_RESULTS_MASTER.csv` and `03_RESULTS_TABLES.md`
5. `04_CLAIM_EVIDENCE_MAP.md`
6. `05_DISCERN_EXTENSION_MAP.md`
7. `06_FIGURE_DATA_AND_CASES.md` plus available small figure-data exports
8. `07_LIMITATIONS_AND_MISSING_ITEMS.md`
9. `08_ARTIFACT_MANIFEST.json`
10. `09_DISSERTATION_FACTS.md`
11. `PREDICTIONS/` with appropriately sized existing video-level exports and a schema

If feasible, package the handoff as a ZIP, excluding raw videos, model weights, secrets, large frame-level caches, and unrelated files. Include all text summaries and aggregated numerical data. If predictions are too large, provide a separate prediction archive or a local-file manifest with exact reproduction/export instructions.

## 6. Document the method completely

In `01_FINAL_METHOD_SPEC.md`, describe the actual implemented system at a level sufficient for writing equations, architecture diagrams, algorithm pseudocode, and reproducibility details.

### Inputs and representations

- Video/frame/clip inputs, frame count, sampling policy and temporal interval, frame rate assumptions, face detection/alignment, crop size and margin, resizing, normalization, and missing-face behavior.
- APP backbone, feature extraction/pooling, feature dimensions, and evidence head.
- OM construction, training objective, inputs, relation to APP, shared parameters, and any manifold/reconstruction/projection mechanism actually used.
- V-JEPA exact version and model size, pretrained checkpoint, temporal input structure, pooling/head, and LoRA or other adaptation settings.
- Evidence/feature dimensions and tensor shapes through the forward path.

### Training

- Joint versus separate training, stages, trainable/frozen parameters, losses and coefficients, batch construction, optimizers, learning rates, schedules, epochs, early stopping, and checkpoint selection.
- Exact augmentations and their probabilities, whether applied per frame or clip and consistently across labels/views.
- Explicitly state whether FreqBlender, SBI, processing augmentation, IBDC, semantic constraints, SEMs, or other previously discussed components were used in each relevant configuration. Do not infer use from an old plan or a directory name.
- Expand warm+gate, arm B, and other internal labels from code/configs. If ambiguous, retain the label and document the ambiguity.

### Fusion and inference

- Evidence-to-opinion mapping, base rates/prior convention, probability extraction, and vacuity definition.
- Exact reliability scaling R. Distinguish evidence scaling from belief-mass discounting; specify the fitting objective, search grid, fitting data, stored weights, and checkpoint/config association.
- Selected PCR6-family formulas, source reference, direct versus sequential application, retained focal elements, combination order where relevant, and zero/near-zero numerical handling.
- Any conflict channels: their equations, timing relative to discounting/fusion, and implemented attribution convention.
- Video-level aggregation order. Explain separately how predictions and uncertainty/conflict scores are aggregated.
- Final threshold procedure and inference-time inputs. Labels must not be listed as an inference input.
- Distinguish any trained uncertainty probe from the actual deployed prediction/deferral pipeline.

Include existing relevant algebraic audit results and concise pseudocode. Link each major implementation description to its source file/function/config. State known properties at the implemented domain and do not invent proofs.

## 7. Document the complete experimental protocol

In `02_EXPERIMENT_PROTOCOL.md`, supply:

- Dataset names, versions/subsets, real/fake counts, source-group counts, manipulation families/generators, and preprocessing exclusions.
- Train, validation, reliability-fitting, score-fitting, calibration, and evaluation roles. Explain which datasets influenced development and which were only evaluated.
- Source-group linkage across real/manipulated videos, generator sets, and processed variants.
- The exact DF40 evaluated entries and grouping: distinguish generator, domain, and generator–domain combination. Explain the reported 41 entries rather than implying a different full benchmark definition.
- Processing conditions and whether each appeared in training, fitting, or evaluation.
- All detector configurations with spatial seed, temporal seed, recipe, checkpoint, and fitting exports. Clearly distinguish independent seeds, repeated folds, and changes in training recipe.
- Video versus frame metrics, units, macro/pooled averaging, generator/family weighting, and confidence-interval methods.
- ECE binning, NLL/Brier definitions, AURC/E-AURC convention, error labels, uncertainty-score direction, threshold fitting, FPR/TPR operating points, and handling of absent/degenerate metrics.
- Existing bootstrap settings, paired source-group resampling, and the uncertainty quantified by each reported interval.
- Comparator provenance: local reproduction, author checkpoint, published number, or other explicitly described source; training/pretraining access and evaluation differences where known.
- Hardware/software versions and measured efficiency conditions.

Do not call five fitting folds five detector seeds. Do not call warm+gate plus arm B a four-seed evaluation of one fixed training recipe. Do not merge their values into an unlabeled seed mean.

## 8. Consolidate the results we need for writing

Export all available results in long form, with one row per metric/configuration/dataset-condition or other explicitly defined unit. Suggested columns:

`method_id, operator_id, configuration_id, branches, spatial_seed, temporal_seed, fitting_split, dataset, generator, manipulation_family, processing_condition, evaluation_unit, metric, value, ci_lower, ci_upper, ci_level, interval_method, n_real, n_fake, n_source_groups, aggregation, result_status, source_report, source_artifact`.

Use blank/null for unavailable values and explain why. Never substitute zero. Include a data dictionary. Report the units consistently and avoid mixing [0,1] AUROC with percentages.

In `03_RESULTS_TABLES.md`, produce readable tables organized into these blocks:

### A. Cross-dataset detection

The final method on all available benchmark sets, including available CDFv2, CDFv3, DFD, DFDC, DFDCP, DFEval24, FFIW, KoDF, IDForge, and DFDM results. Include DISCERN, DiCoME, and relevant SOTA baselines already available. Identify actual scored datasets rather than filling a prescribed list with unsupported values. Separate frame/video and local/published comparisons.

### B. Cross-manipulation results

Per evaluated DF40 generator/domain entry, manipulation-family summaries, and overall means with explicit weights. Include reenactment/expression-related subgroup definitions and counts, as well as swap results. Supply existing paired differences and intervals where available.

### C. Temporal evidence and adaptation

Available temporal-only, spatial-only, and combined results; frozen versus adapted temporal models; ordered/shuffled/reversed/local-permutation controls; relevant backbone comparisons; and any same-frame order-invariant control already run. State exactly which controls exist, their input matching, and their measured effects. Do not manufacture a four-backbone study from discussion alone.

### D. Component necessity and fusion

Existing APP, OM, T, APP+OM, T+APP, T+OM, and full-system ablations; final fusion with/without R; relevant PCR6 versus PCR6+ comparisons; and non-Dempster controls such as averaging-R or Yager-R where available. State which ablations reuse frozen opinions and which involve separate training.

Export per-fold reliability weights and zero-weight frequencies. Distinguish the configured three-branch model from the effective branch set after weighting. Document any exact reduction to a two-view method.

### E. Reliability and selective prediction

Available ECE, NLL, Brier, error-detection AUROC, AURC/E-AURC, risk at fixed coverage, coverage at existing predefined risk targets, and FPR/TPR at documented operating points. Include vacuity, margin/entropy, retained conflict, and any combined score already evaluated.

If source-conflict attribution was studied, include the comparison of the attributed predictor with a matched basic predictor using vacuity, total conflict, and margin. Identify folds where a zero branch weight removes attribution information. Do not imply this analysis exists if it has not been run.

### F. Processing robustness

Clean and all already-tested JPEG/resize/other conditions, with absolute detection/reliability performance and degradation. Distinguish training augmentation from evaluation corruption. Separate detector recipes so an augmentation comparison is not mistaken for a seed comparison.

### G. Efficiency and cases

Existing parameter counts, trainable parameters, FLOPs if measured, runtime/throughput, memory, and training cost. Specify batch size, hardware, input length, precision, and whether preprocessing is included. Mark unavailable measurements instead of running new model inference.

Supply representative existing successes and failures with anonymous IDs, branch predictions, fused decision, uncertainty, and selection rationale. Cases illustrating temporal rescue, useful spatial evidence, and unresolved disagreement are useful. Do not cherry-pick cases into an unsupported claim of general behavior.

For each result block, provide a short factual summary of its strongest demonstrated finding and an exact source pointer. Avoid declaring a new operator novel or deciding publication acceptance.

## 9. Export usable prediction and figure data

Prefer existing video-level Parquet/CSV exports. Include anonymous video/source-group IDs, dataset/generator/family/condition, label, config/seeds/fold, branch outputs when available, reliability weights, final probability, uncertainty/conflict outputs, threshold, and predicted class.

Provide a schema and document aggregation. Do not reconstruct evidence from averaged probabilities: probabilities alone do not determine evidence strength. Do not claim video-level metrics can be reproduced from exports if required frame-level aggregation information is missing; explain the additional source needed.

Separate fold-specific predictions from any aggregated prediction. Do not present repeated predictions for the same video as independent samples.

In `06_FIGURE_DATA_AND_CASES.md`, map data to:

- Architecture diagram: components, dimensions, shared paths, training/inference stages.
- Temporal-control plot: ordered versus perturbed results and adaptation settings.
- Cross-manipulation plot: generator/family comparisons.
- Risk–coverage and reliability plots.
- Branch-evidence case studies.

Export available numerical curve data and existing figure sources where small. Do not generate fictional curves, saliency maps, or causal/semantic explanations. New plotting from already exported data is optional; numerical handoff completeness takes priority over figure styling.

## 10. Build the claim and conference-extension maps

In `04_CLAIM_EVIDENCE_MAP.md`, use columns:

`proposed statement | implemented mechanism | evidence/report | comparison | effect size | configurations | status | intended paper section`.

Focus on spatial–temporal complementarity, temporal adaptation, cross-manipulation generalization, reliability, robustness, and component necessity. Separate supported findings from preliminary observations and untested hypotheses. Do not promote an old proposed mechanism into a completed result.

In `05_DISCERN_EXTENSION_MAP.md`, compare the actual original DISCERN source with the current implementation. Use columns:

`DISCERN component/result | retained/modified/replaced/removed/new | current counterpart | scientific purpose | supporting implementation/result`.

Cover visual/CLIP adaptation, semantic/FACS constraints, structural/SEM streams, confidence-modulated fusion, IBDC, auxiliary supervision, training, temporal modeling, inference, and experimental coverage. Describe factual differences, without inventing an exact percentage of novelty or calling all evidential learning IBDC-derived.

Keep DiCoME comparison factual and compact. It does not need a dedicated handoff chapter or repeated mentions throughout the package.

## 11. Missing items and interpretation notes

In `07_LIMITATIONS_AND_MISSING_ITEMS.md`, consolidate material issues once:

- Unresolved final-method identity, unfinished results, inaccessible artifacts, or conflicting reports.
- Missing implementation details that prevent an accurate method section.
- Missing evidence for a proposed headline claim.
- Optional enrichments that do not block writing.
- Known exposure history, incomplete seed replication, comparator protocol mismatches, and corrected historical errors where relevant.

For each gap provide the precise missing item, the affected statement/table, whether it is recoverable from existing files, and a useful placeholder. Recommend narrowing an unsupported claim where that permits writing to proceed. Do not turn every gap into a request for another experiment.

## 12. Manifest, provenance, and validation

In `08_ARTIFACT_MANIFEST.json`, record the current commit/dirty state, handoff version, selected-method status, exact source paths, configuration/checkpoint/cache identifiers, reusable hashes when present, exported filenames/sizes, evaluation-script versions, and commands needed to reproduce the exports.

Do not hash huge media/checkpoint collections from scratch solely for this handoff. Reuse known identifiers and mark missing hashes. Exclude credentials, tokens, signed links, and unrelated personal content.

Perform targeted checks before delivery:

- Exported headline values agree with their current authoritative reports.
- No superseded result is attributed to the selected operator.
- Metric units, sample counts, branch/config IDs, and dataset labels are consistent.
- Missing values are explicit.
- Frame, clip, and video units are not silently interchanged.
- Fold/seed means and intervals have their aggregation stated.
- Machine-readable files open correctly and the archive, if produced, contains the promised documents.

Do not fix a numerical disagreement by choosing the more favorable value. Trace it to the generating configuration/script, record the resolution, or leave it explicitly unresolved.

## 13. Dissertation facts only

In `09_DISSERTATION_FACTS.md`, provide factual material useful for the future proposal:

- The extension's implemented research objective and method.
- Completed experiments and strongest findings.
- What has changed relative to DISCERN.
- Which technical questions remain unresolved.
- Existing manuscript-ready tables, figures, and implementation descriptions.
- Status of each item: completed, running, planned, or unavailable.

Do not invent a dissertation chapter structure, rename the thesis, add video partial-forgery/audio projects, or assert certification/symbolic attribution capabilities. The writing stage will integrate these facts with the latest proposal and advisor guidance.

## 14. Final response to the user

In `00_START_HERE.md` and the final response, give:

1. The final-method identity/status.
2. The exact handoff directory and archive path, if produced.
3. A compact inventory of delivered files.
4. The strongest existing findings, with numbers and report pointers where available.
5. A short prioritized list of missing items that actually matter for writing.
6. Confirmation that no new training/inference/operator search was initiated by this task.

Use readable summaries rather than dumping every CSV row into the chat. Distinguish a usable writing package from a fully submission-ready experimental record.

**Begin with the repository and current-artifact audit, establish the selected method, and produce the handoff. Finish all export/documentation work possible from existing evidence without reopening the research search.**

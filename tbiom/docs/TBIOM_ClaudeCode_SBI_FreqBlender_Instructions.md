# Claude Code brief: integrate official SBI and FreqBlender

Date: 2026-09-12  
Project: DFB_NESyNS / T-BIOM experiments

## Objective

Use the authors’ official FreqBlender repository and the existing SBI implementation to evaluate whether synthetic training improves the current detector’s generalization and processing robustness. Reuse upstream code rather than independently reimplementing FPNet or its losses.

Keep the current Stage 2/3 investigation intact. Add a separate, controlled training experiment; after selecting its configuration on dev data, evaluate whether M1/M2/M3 still add value beyond stronger training.

The user has authorized this implementation work. Respect repository instructions, preserve ongoing jobs, and schedule new runs around existing resources. Do not cancel Stage 2, overwrite existing checkpoints, or revise a frozen protocol based on held-out results.

## 1. Read the current project state

Reported context, to verify locally:

- Stage 2 has 20 processing passes.
- Relevant reported commits: `9cb11ae` and `6e54077`.
- Existing design: `tbiom/docs/stage3-design.md`.
- Arm B is the working detector; preserve its canonical checkpoint as an immutable historical reference.
- SBI is already present through DeepfakeBench. Locate its actual dataset/generator integration, not just its detector registration.

Read project instructions, current train/dev/test manifests, training recipe, allowed trainable parameters, checkpoint-selection rule, and definitions of M1/M2/M3. Private project files were not inspected in preparing this brief; resolve paths from the actual checkout.

Train new detector copies for this experiment. Preserve the original recipe’s load-bearing auxiliary losses and its allowed trainable parameters initially. The earlier choice to freeze the detector applied to the evidence-interpretation diagnostic, not to this new training comparison.

## 2. Use the official repositories

**FreqBlender:** https://github.com/OUC-VAS/FreqBlender  
**SBI:** https://github.com/mapooon/SelfBlendedImages  
**DeepfakeBench:** https://github.com/SCLBD/DeepfakeBench

FreqBlender source was inspected at commit:

```text
7c8bc7f5a653a068d6dadcd2e64996f9e2805fc7
```

Use an isolated dependency checkout outside the main source tree, or a properly pinned submodule if that is the project convention. If the user has already cloned it, reuse that checkout without discarding local edits. Example for a new checkout, with the destination adapted to local conventions:

```bash
git clone https://github.com/OUC-VAS/FreqBlender.git <dependency-directory>/FreqBlender
git -C <dependency-directory>/FreqBlender checkout 7c8bc7f5a653a068d6dadcd2e64996f9e2805fc7
```

The destination placeholder must be replaced before execution. Record the exact revision. If another revision is needed, audit its changes and record why. Preserve upstream attribution and applicable license notices when importing code.

Use upstream:

- `net/aeNet/conv_autoencoder_pixelshuffle.py` for FPNet;
- `utils/loss_function.py` for the public-source training objectives, if retraining FPNet;
- `utils/face_preprocess.py` for reference DCT conventions;
- `config/base.json` and `main.py` for the public-source recipe;
- README instructions for integrating FPNet with SBI.

**Do not blindly run upstream `main.py` against the whole dataset.** Replace data enumeration with explicit project training manifests and audit its dependencies first. Integration patches should alter paths, manifests, device handling, and interfaces as necessary, without silently changing the algorithm.

## 3. Obtain the actual integration dependencies

The official README links to external Baidu downloads for FPNet/ResNet weights and modified SBI files. It describes replacing SBI’s `src/configs/sbi/base.json` and `src/utils/sbi.py`, then adding FPNet code and weights to SBI’s utilities.

Retrieve those files through normal available access and compare them with the installed SBI implementation. They are not all supplied simply by cloning the GitHub repository. Record download source, checksum, expected architecture, and training-data provenance for each checkpoint.

The externally hosted modified SBI files and released checkpoint contents were not inspected when preparing this brief. Do not assume they match every detail of the public FPNet training code. If unavailable, continue SBI and processing controls, and report the specific blocked FreqBlender dependency. Do not substitute a new generator and call it an exact official reproduction.

Inspect teacher and parser training exposure before using released weights in a clean OOD experiment. If exposure is unknown, use those weights only for a labeled implementation diagnostic. Train split-safe auxiliary models when necessary for the formal comparison.

## 4. Paper versus source: required audit notes

Use the attached NeurIPS 2024 paper as the conceptual reference, but choose an explicit implementation target. Recommended target: **official-code-based, split-safe adaptation**. Do not silently mix paper losses with public code defaults.

| Component | Paper | Inspected public code |
|---|---|---|
| Frequency masks | Described as maps in [0,1] | No final sigmoid or softmax; outputs are unconstrained |
| Facial fidelity | Squared identity-embedding distance | Clipped cosine distance to facebank embeddings |
| Authenticity objective | Cross-entropy over reconstructed examples | Linear penalties on softmax class-1 probabilities |
| Quality objective | Image versus its semantic+structural reconstruction | Weighted MSE involving real, real-semantic, and mixed real-semantic/fake-structural reconstructions |
| Prior objective | Band-prior and integrity penalties | Additional region-area weighting |
| FPNet schedule | Batch 8; 200 epochs; Adam, LR 1e-4 | Batch 16; 100 epochs; Adam, LR 1e-4; StepLR every 5 epochs, gamma 0.5 |
| FPNet data | Final detector experiment uses SBI donors | Active FPNet loader reads actual FF++ Deepfakes; SBI loading is commented out |

The public FPNet loader defaults to numeric IDs up to 1000 rather than a project train/dev split. Facebank and auxiliary teacher inputs also require split control. Account for all auxiliary training data; do not claim “real-only training” for the entire pipeline solely because the final detector uses SBI donors.

The public checkpoint-saving logic ranks by the final training batch’s loss. If replacing it with an aggregate source-dev criterion, predeclare and document that adaptation. Verify the authenticity teacher’s class order: the public loss treats probability index 1 as fake. Map labels explicitly to the project’s convention.

Do not add mask normalization to official checkpoints. Do not convert all loss computations to a new image scale while retaining the old weights. The source computes orthonormal DCT on RGB images scaled to [0,1], but reconstructs/clips images to [0,255] for several FPNet training losses.

When training FPNet, freeze teacher weights while retaining gradients through their inputs. Do not wrap teacher forwards in `no_grad()`. During final detector training, frozen FPNet image generation can run without gradients.

## 5. Intended integration behavior

FreqBlender is a training-data generator in this experiment. Its three frequency masks are not three additional classification branches, and FPNet should not add inference cost unless separately proposed.

The paper’s operation uses a genuine host `r` and fake donor `p`:

```text
R = orthonormal_DCT(r)
P = orthonormal_DCT(p)
semantic_mask, structural_mask, noise_mask = FPNet(P)
mixed = R * semantic_mask + P * structural_mask + R * noise_mask
pseudo_fake = inverse_DCT(mixed)
```

In the main detector experiment, the donor is an SBI pseudo-fake. The paper applies frequency blending with probability **0.2**. This is the application probability, not a coefficient multiplying the structural component. When the operation is skipped, retain the SBI pseudo-fake.

Use the official modified SBI generator’s exact crop, interpolation, clipping, and quantization behavior after retrieving it. The public FPNet operates on a 400×400 canvas. Verify how that maps to the detector’s input resolution; do not resize coefficient masks arbitrarily or feed CLIP-normalized pixels into FPNet.

Pipeline:

1. Select a permitted source video/frame and common parent crop.
2. Generate SBI’s paired genuine host and pseudo-fake donor.
3. In the FreqBlender arm, optionally frequency-blend the donor.
4. Apply final processing to genuine and fake using the same sampled settings and order.
5. Apply shared detector resize/normalization and compute the inputs used by each branch.
6. Train with the existing detector losses and optimizer recipe.

SBI’s internal mismatch transforms generate the manipulation; they are distinct from the final shared processing. Use dedicated RNG streams so adding FreqBlender does not change source selection or all other augmentation draws.

**Multiview consistency is mandatory:** recompute image-dependent features/descriptors for synthesized images. Do not pair a synthetic CLIP input with stale genuine-parent cached features in another branch and call that ordinary multiview training. If the pipeline cannot support recomputation, report the limitation and run an explicitly labeled single-stream pilot first.

For temporal models, independent frame-wise synthetic masks can create artificial flicker. Use a documented temporally coherent adaptation or restrict the initial pilot to frame-based streams. Do not silently claim image SBI reproduces a realistic video manipulation process.

## 6. Controlled training comparison

Keep the historical Arm B result as `R0`. Prepare these matched training arms:

| Arm | Change | Purpose |
|---|---|---|
| A | Existing recipe under the common integration pipeline | Matched training baseline |
| B | A plus predeclared additional class-balanced processing | Processing exposure alone |
| C | B plus SBI replacement of some fake slots | Spatial synthesis benefit |
| D | C plus official FreqBlender on SBI donors | Additional frequency synthesis benefit |

If B duplicates existing augmentation, merge it or specify its incremental distribution before running. If the generator introduces an extra crop/resize path, provide the corresponding benign resampling path in the controls. A may then differ from R0; state that instead of attributing a preprocessing difference to frequency blending.

Initial proposed project setting, **not a paper hyperparameter**: replace 50% of fake slots with synthetic examples in C/D, retaining actual manipulated examples in the other fake slots. Apply FreqBlender to 20% of the SBI donors in D. Keep class balance, detector update count, batch size, and example budget comparable. If the project requires another replacement fraction, justify and freeze it before target evaluation.

C and D must share parent draws, SBI seeds, replacement fraction, and final processing settings. Match source/identity exposure across the other arms where possible, and report unavoidable differences. Use a common eligible source pool if landmark requirements would otherwise filter difficult examples only from synthetic arms. Count generator failures.

Screen with a common seed using permitted dev data, then confirm surviving comparisons under the existing seed protocol; if none exists, predeclare three seeds. Match starting checkpoint/training history and checkpoint-selection criteria. Do not give synthetic arms additional optimization steps without a matched-budget control.

## 7. Focused verification before expensive runs

- DCT/IDCT round-trip, channel order, value ranges, and official-checkpoint strict loading.
- Artificial-mask checks: structural-only reconstruction returns the donor; semantic-only returns the host; skipping frequency blending returns SBI. Identical host/donor reconstructs identically only when masks sum to one. Log the learned mask-sum error rather than assuming exact partitioning.
- Inspect genuine, SBI, frequency-blended, and processed examples; log out-of-range/clipping rates.
- If training FPNet, verify finite gradients from its losses into FPNet and unchanged teacher parameters.
- Verify every derived example, host, donor, auxiliary-teacher sample, and facebank reference respects the split rules. Keep all processing variants of an original video together; account for all contributing identities where donor pairs cross videos.
- Verify consistent inputs across detector branches and no stale feature caches.
- Check landmark-failure handling, sample counts, label mapping, throughput, and a short optimization smoke run.

Do not interpret a smoke test as evidence of generalization. These checks are required because transformation errors, label reversal, stale features, and auxiliary-data leakage could invalidate the comparison.

## 8. Evaluation and connection to M1/M2/M3

Use the existing original-video grouping and aggregation protocol. Report:

- Video AUROC and group sample counts.
- FPR/TPR at each arm’s threshold selected using the same baseline-dev rule, then fixed across processing conditions.
- Retrospective FPR at matched TPR / TPR at matched FPR, clearly separated from deployable threshold results.
- Worst evaluated group FPR, without claiming control over arbitrary future domains.
- Brier/NLL and existing calibration measures when outputs support them.
- Clean, held-out processing, and genuine-source-shift performance separately.
- Video-group uncertainty intervals, seed variation, and training/inference costs.

A stable AUROC does not imply stable operating-point performance. Pixel-based quality indicators are not automatically class-independent; retain within-class and cross-source checks. Do not use known test severity in the main deployable comparison.

Select the improved training configuration using dev only. Freeze that detector, recompute its measurements, and run M1/M2/M3 with identical deployment-available inputs. New training invalidates old feature caches, reliability curves, and calibration parameters; characterize the new detector rather than reusing Arm B’s fitted relationships.

Respect the existing out-of-fold reliability-estimation contract. If target results have already guided new method choices, mark those targets development-exposed and reserve fresh evidence for the final claim.

Do not add the previously discussed paired-ranking loss, new symbolic rules, or a new severity estimator to C/D by default. Those require separate same-data ablations after the ordinary augmentation comparison.

## 9. Deliverables and execution order

1. Commit a new training design and source audit without altering the existing frozen Stage 3 design.
2. Record upstream revision, checkpoint hashes/provenance, dependency status, actual integration paths, data counts, sample budgets, and deviations from upstream.
3. Implement the SBI integration and focused checks first; progress FreqBlender dependency integration independently where possible.
4. Queue matched controls and SBI runs around Stage 2; queue FreqBlender after its dependency and data-exposure checks pass.
5. Export per-video predictions and a report distinguishing processing augmentation, spatial synthesis, frequency synthesis, and subsequent reasoning effects.

Suggested documentation names: `tbiom/docs/sbi-freqblender-design.md` and `tbiom/docs/freqblender-source-audit.md`. Reuse project conventions for actual modules and configurations.

Continue authorized implementation and routine reversible fixes without repeated permission requests. If a required external weight/file is inaccessible, identify the exact blocked component and continue independent SBI/control work.

Success means a controlled answer about whether these training methods help. SBI/FreqBlender gains alone are baseline improvements; a reasoning contribution requires a measured advantage over the same-input neural control. None of these experiments guarantees novelty, publication, or robustness to all manipulation types.

## Audit status

The attached paper and public source at the pinned revision were inspected. The private project, released checkpoint contents, and externally hosted modified SBI files were not inspected, and no training was executed in preparing this brief. This document intentionally directs reuse of official code instead of supplying an independently reconstructed implementation.

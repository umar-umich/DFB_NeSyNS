# DISCERN-v2 independent views brief

This brief supersedes the pending parts of the I-JEPA brief and its amendments. The joint IBDC brief and its Option B addition stay in force.

## Status this brief starts from

- **I-JEPA residual branch.** Failed G2 (R3 0.6622 and R4 0.6658 against the R2 plain probe at 0.6812 on dev v3.1, and residual plus backbone at 0.6664). Under the brief's stop rule this is a finding.
- **Rows cancelled.** R5, T0 to T3, R6 and R7 are cancelled unless Part 2 reopens the residual line.
- **Joint IBDC.** Task 1, Task 2, G0a and G0b all passed, but only 2 of about 11 runs exist.
- **New measurements.** Appearance is discounted to zero in 3 of 5 folds. Off-manifold contributes +0.065 without Appearance and +0.013 with it. Appearance and Off-manifold share one CLIP trunk.

## Rules (apply to every part)

- Never fabricate a number. Unrun cells are `TODO(run)`.
- Decisions use FF++ c23 validation, VALmix-v2, dev v3.1 or the development folds only. Test sets are evaluated once per final row and reported, never used to choose anything.
- Every pre-set rule below is fixed before its run starts. A failed rule is reported as a finding.
- "dev v3.1" means the same cross-dataset development objective used in `analysis/ijepa/`.
- Every number cites the file or command that produced it.
- Fused rows always use PCR6+-R, with trust weights refit per seed on the 5 source-grouped development folds and the threshold at FF++ c23 validation EER, exactly as deployed.

## GPU plan

GPUs 1, 2 and 3 are free.
- **Lane A (GPU 1).** Parts 1 and 2.
- **Lane B (GPU 2).** Part 3 extraction for CLIP, DINOv3 and I-JEPA, then Parts 5 and 6.
- **Lane C (GPU 3).** Part 3 extraction for Perception Encoder, then Part 4.

Change the allocation if throughput measurements suggest a better split, and report why.

---

## Part 1. Finish the joint IBDC study (Lane A, top priority)

Run everything remaining in `discern-v2-joint-ibdc-brief.md` and its Option B addition.
- J1 seeds 1 and 2.
- J1s, all 3 seeds, if not already done.
- J2a and J2b λ sweeps, then the chosen λ with seeds 1 and 2.
- Test evaluation, all metric tables and the J2b four-part verdict.

Additional checks before the sweeps.
1. Confirm that G0b was reported with J1s heads as the Option B addition requires. If it was measured with J0 heads only, rerun it with J1s heads.
2. G0b showed the hinge active on 65% of training samples. Report the distribution of κ − u_f over active samples (5th, 25th, 50th, 75th and 95th percentiles) per augmentation level. At that activity rate the loss acts as a floor on fused vacuity across most of the training set, and the report must show how large that floor is.

## Part 2. Close or reopen the residual line (Lane A, minutes)

All inputs are cached, so no extraction is needed.

**Fix A variants.** Orthogonalize each block residual r̄_k before the head.
- **A1.** Project off the pooled I-JEPA feature g, as DiCoME does.
- **A2.** Project off the deployed Appearance feature. If its dimension differs, first map it with a ridge regression fitted on FF++ c23 train and use the mapped vector as g.
- **A3.** Project off the subspace spanned by the top 64 principal components of the pooled I-JEPA feature, with the PCA fitted on FF++ c23 train.

A1 and A2 remove a single direction out of 1280, so A3 is the strong version.

For each variant, train the R4 head on the backbone feature plus the orthogonalized residual (the same "both" configuration as in the report), 3 seeds.

**Pre-set rule.** The residual line reopens only if the best variant beats the R2 backbone-alone probe on dev v3.1 by a margin whose source-grouped bootstrap 95% interval lies above zero. Otherwise the residual line is closed, and the report states it as a negative result.

**R1, the DiCoME projection ablation.** Run the manifold-feature extraction pass needed for R0h and R1, then train both heads as specified in the I-JEPA brief. This is a paper ablation of DiCoME's orthogonal projection inside our system, and it runs whatever Fix A shows.

## Part 3. Frozen backbone screen for an independent Appearance view (Lanes B and C)

**Candidates.**

| Candidate | Checkpoint | Input size |
|---|---|---|
| Reference | CLIP ViT-L/14, the base of the deployed Appearance view, without LoRA | 224 |
| DINOv3 | ViT-L/16 | 224 |
| Perception Encoder | PE-Core at 448 resolution | 448 |
| I-JEPA | ViT-H/14 (reuse the R2 features if the frames match) | 224 |

Confirm and record the exact Perception Encoder checkpoint name. If only the largest model exists at 448, record its parameter count.

Record the license of every checkpoint. The DINOv3 and Perception Encoder licenses may restrict commercial use, and I-JEPA is CC BY-NC 4.0.

**Inputs.**
- Cut face crops at each model's native resolution from the original frames, using the deployed face boxes. Do not upsample the existing 224 crops.
- Report the fraction of faces whose box is smaller than the model's input size, since those are upsampled at any resolution.
- Use the same frames per video as the deployed Appearance view.

**Features.** Use each model's standard global embedding.
- CLIP: the pre-projection class token.
- DINOv3: the class token.
- Perception Encoder: its pooled output.
- I-JEPA: the mean of patch tokens, since it has no class token.

Record the choice for each.

**Head and training.**
- Linear to 256, GELU, dropout 0.1, linear to 2, softplus evidence.
- The deployed EDL variant and KL ramp.
- FF++ c23 train with the deployed Appearance view's augmentation (no B mixture), VALmix-v2 checkpoint selection, 3 seeds.
- Frame-averaged to video level as the Appearance view does.

**Measurements per candidate** (three-seed mean ± standard deviation).
1. FF++ c23 validation AUROC, VALmix-v2 score and dev v3.1.
2. Error correlation on the development pool with Off-manifold and with the video view. Use the phi coefficient between error indicators, each view at its own FF++ validation EER threshold, out of fold.
3. Fused system with the candidate in the Appearance slot, plus Off-manifold and the video view.
   - Out-of-fold development AUROC and development real-video FPR at the threshold.
   - The candidate's leave-one-out contribution with interval, and how many folds set its weight to zero.
   - Off-manifold's leave-one-out contribution with this candidate present. The deployed value is +0.013.

**Pre-set winner rule.**
- The winner has the highest fused out-of-fold development AUROC and must beat the fused system with frozen CLIP in the Appearance slot.
- If two candidates are within 0.002, the one with lower error correlation with Off-manifold wins.
- If the top two are within 0.005 of each other, both go to Part 4.
- If no candidate beats frozen CLIP, stop Parts 4 and 5, and report the result as a finding.
- If Perception Encoder wins, add a size and resolution control, the PE-Core-L14-336 checkpoint, to the screen before Part 4. This separates backbone from scale.

## Part 4. Adapt the winner as the Appearance view (Lane C)

Two recipes, one seed each first.
- **Recipe L.** LoRA rank 8, alpha 16, dropout 0.1 on query and value, matching the deployed views.
- **Recipe N.** Layer Normalization parameters only, with L2-normalized features, following GenD (Yermakov et al., WACV 2026). Include GenD's metric-learning loss if their released code provides it. Credit GenD in the report.

**Training.** FF++ c23 train, source-paired as deployed, the deployed Appearance augmentation, EDL head as in Part 3, VALmix-v2 checkpoint selection.

**Budget.** Estimate on 100 videos first. Cap each training run at 24 GPU hours. If a run would exceed that, stop and report the estimate.

**Selection.** Pick the recipe by dev v3.1, then run seeds 1 and 2 for it.

**Report.**
- Standalone dev v3.1 against the deployed Appearance view's 0.8174.
- The fused system (new Appearance view, Off-manifold and video view) against the deployed fused system, including development real-video FPR.
- The new view's weight-zeroing count, and Off-manifold's leave-one-out contribution.

## Part 5. Cross-backbone feature mapping (Lane B, frozen features)

**Idea.** Learn to predict one frozen backbone's face embedding from another's on real faces. Use the prediction error as a residual view. It is defined between two independent models rather than inside one.

**Pair.** The Part 3 winner W and frozen CLIP C, in both directions (W to C, C to W).

**Fitting.** Use only real frames of FF++ c23 train.
- **Ridge.** A closed-form linear map, with its penalty chosen by 5-fold cross-validation on real FF++ train.
- **MLP.** Two layers with hidden width 1024, early-stopped on mapping loss over real FF++ c23 validation frames.

**Residual.** Take observed minus predicted, standardized per dimension by the residual standard deviation on real FF++ train. The head receives the log residual norm plus the residual vector through a linear projection to 64 dimensions, then uses the Part 3 head. Train with 3 seeds.

**Controls, pre-set.** The best mapping variant must beat all three on dev v3.1, with source-grouped bootstrap 95% intervals above zero.
1. A probe on W alone.
2. A probe on C alone.
3. A probe on the concatenation of W and C, with capacity matched to the residual head.

**Fusion.** If it passes, evaluate it in the artifact slot, as W-Appearance, the mapping residual and the video view, against W-Appearance, Off-manifold and the video view. Report development real-video FPR per source dataset, since a map fitted on FF++ real faces may flag any non-FF++ real face.

**Related work to cite.** No image or video deepfake work mapping between two visual foundation models was found in a preliminary search. The nearest are:
- Real-only masked modeling (RFFR).
- Recover-and-map methods (Recap, Mover).
- Audio-visual cross-modal prediction trained on real videos (AVFF; Boldisor et al., CVPR 2026).
- Crossmodal feature mapping for industrial anomaly detection (Costanzino et al., CVPR 2024).

## Part 6. Perturbation stability, training-free (Lane B)

For each Part 3 backbone, compute the cosine distance between the embedding of a face crop and that of a perturbed copy. This follows RIGID (He et al., 2024) and its follow-up study on faces (Tsai et al., 2024).

**Perturbations.**
- Gaussian noise with standard deviation 0.01 and 0.05 (pixels in [0, 1]).
- Gaussian blur with sigma 0.5 and 1.0.
- The minimum distance over the four, as in the follow-up study's minimum-distance detector.

Choose the perturbation on FF++ c23 validation. Report zero-shot video AUROC (frame-averaged distance) on FF++ validation and dev v3.1.

**Compression check**, as in G1. On FF++ validation real videos, the shift in distance from clean to JPEG q50 must be smaller than the real-versus-fake gap on clean videos.

**Pre-set rule.** Add the score to fusion as an extra view only if dev v3.1 AUROC is at least 0.65 and the compression check passes. Otherwise report it as a finding.

## Deferred, not in this brief

- LoRA ablations of rank, target modules and depth. They apply only to the final views.
- Predictor adaptation (R6). It returns only if Part 2 reopens the residual line, with non-FF++ real faces as training data.
- Sequence heads on whichever view survives.
- UCF-Net (Zou et al., 2026, arXiv 2609.07670) as an external reference. It fuses CLIP and DINO with uncertainty weighting for deepfake images, trained on a different, much larger dataset. Check whether its checkpoint is released and report the check.

## Reports

- **Part 1.** Its own report, as specified in the joint IBDC brief.
- **Parts 2 to 6.** `analysis/independent_views/INDEPENDENT-VIEWS-REPORT.md`. Each part gets verdicts for its pre-set rules with evidence files, the tables specified above, checkpoints and licenses, GPU time used and cache paths.
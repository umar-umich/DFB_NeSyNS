# DISCERN-v2 I-JEPA branch brief

## Goal

Test whether a regional prediction-residual view built on frozen I-JEPA can replace the DiCoME-style Off-manifold view in the reference system without losing detection performance.

The idea is simple. Hide a block of the face, let the pretrained I-JEPA predictor guess its features from the rest of the face, and measure how wrong the guess is. Manipulated regions may not fit their unchanged surroundings. Compression and blur also raise the error, so the error is standardized against what real faces of the same quality produce.

Everything in I-JEPA stays frozen. Only a small evidential head is trained.

## Fixed parts of the system

- Appearance view, V-JEPA 2 video view, fusion operator (PCR6+-R), trust-weight fitting on the 5 source-grouped development folds, and the threshold at FF++ validation EER all stay exactly as deployed.
- Use the same PCR6+-R implementation that produced the video-view contribution table and record its commit hash. If the PCR6+ audit later changes that implementation, the fusion rows here are rerun on cached opinions.
- Protocol v3.1. Train on FF++ c23 only. Ten test sets, DF40 (41 sets), processed tests.

Locate the deployed configuration and state which repository holds it (DFB_NeSyNS or DISCERN_Ext). Put all new code in `branches/ijepa/` and all outputs in `analysis/ijepa/`. Do not modify deployed configs or checkpoints.

## Rules

- Never fabricate a number. Unrun cells are `TODO(run)`.
- Every decision uses FF++ validation, VALmix-v2 or the development folds. Test sets are evaluated once per row and reported, never used to choose anything.
- Gates run in order. If a gate fails, stop, write the result into the report as a finding, and do not continue to the next gate.
- Every number in the report cites the file or command that produced it.

## Gate G0. Predictor weights

Run the supplied script `check_ijepa_predictor.py`.

```
git clone https://github.com/facebookresearch/ijepa.git
wget https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar
python check_ijepa_predictor.py <folder of 32 real FF++ c23 training face crops, 224x224>
```

Pass if all three weight sets load with `strict=True` and the trained predictor beats all three controls on both blocks. Paste the full output into the report.

The checkpoint is CC BY-NC 4.0. Note this in the report.

## Feature extraction

**Model.** I-JEPA ViT-H/14, 224x224 input, 16x16 patch grid, ImageNet mean and std normalization. Context uses the `encoder` weights. Targets use the `target_encoder` weights, followed by layer normalization over the feature dimension, as in I-JEPA training.

**Face crops.** Use the same 224x224 face crops and the same frames per video as the Appearance view. Confirm the frame count from the deployed config.

**Blocks.** The predictor was pretrained on hidden blocks covering 15 to 20% of the image with aspect ratio 0.75 to 1.5, so use large blocks. Rows and columns are patch indices, end exclusive.

| Block | Rows | Columns | Patches |
|---|---|---|---|
| Upper face | 2 to 9 | 4 to 12 | 56 |
| Lower face | 9 to 16 | 4 to 12 | 56 |
| Left cheek | 5 to 12 | 0 to 8 | 56 |
| Right cheek | 5 to 12 | 8 to 16 | 56 |
| Centre | 4 to 11 | 4 to 12 | 56 |

Before extraction, check whether the crops are aligned. Save block overlays on 20 random FF++ crops and 20 CDFv2 crops to `analysis/ijepa/block_overlays/`. If the blocks do not land on the intended face areas, stop and report.

**Per frame and per block k.**
- Hide block k from the context encoder. The hidden patches must never enter the context input.
- Predict the hidden patch features and compute per-patch residuals r_j = s_j − ŝ_j.
- Store the mean squared residual ε_k (averaged over patches and feature dimension) and the mean residual vector r̄_k (1280 dimensions).
- Also store the mean-pooled target-encoder feature of the whole face (1280 dimensions) for the R2 control.

**Quality descriptors q per frame.** Store variance of the Laplacian (blur), a blockiness score (compression), face box size in the original frame, and yaw if landmarks are cached. Use no labels.

**Budget.** Run on 100 videos first and report the projected total GPU time. Continue without asking if it is at most 24 GPU hours. Otherwise stop and report the estimate.

Cache everything as fp16 under `analysis/ijepa/cache/` with one file per dataset. Extract under the B processing mixture for FF++ training (one draw per clip, as for the video view), clean for FF++ validation and the test sets, and at the processed-test settings for CDFv2 and DFDCP.

## Quality standardization

Fit on **real** FF++ c23 training frames only, under the B processing mixture. For each block k, fit a linear regression of log ε_k on q (linear and squared terms). Let σ_k be the standard deviation of its residuals. Then

a_k = (log ε_k − μ_k(q)) / σ_k

Freeze μ_k and σ_k and apply them unchanged everywhere else. Save the fitted coefficients.

## Gate G1. Compression confound

On FF++ c23 validation, for each block, compute three quantities.
- d_comp. The shift in mean log ε_k of real videos from clean to JPEG q50. Repeat for resize 0.50.
- d_fake. The difference in mean log ε_k between fake and real videos, both clean.
- Real-versus-fake AUROC of a_k within each quality level (clean, JPEG q50, JPEG q30, resize 0.50, resize 0.35).

Pass if d_comp < d_fake for at least 3 of the 5 blocks, for both JPEG q50 and resize 0.50. Report all numbers whether it passes or not.

## Views to train

Train every head with the same EDL variant, KL ramp, optimizer, epochs and VALmix-v2 checkpoint selection as the deployed Off-manifold head. Confirm those settings from the deployed config and record them. Training uses FF++ c23 train under the B processing mixture. The video-level opinion is the frame average, as for the other spatial views.

| Row | Artifact slot | Head input |
|---|---|---|
| R0 | Deployed Off-manifold view, unchanged | As deployed |
| R0h | Off-manifold head retrained with CLIP LoRA and β-VAE frozen | r_⊥ |
| R1 | Off-manifold head retrained the same way | r (no orthogonal projection) |
| R2 | Plain I-JEPA probe | Mean-pooled I-JEPA feature |
| R3 | I-JEPA residual view without standardization | log ε_1..5 and r̄_1..5 |
| R4 | I-JEPA residual view with standardization | a_1..5 and r̄_1..5 |

R0h exists so that R1 is compared against a head retrained in the same way, not against the deployed head.

**Head for R2 to R4.** Linear to 256, GELU, dropout 0.1, linear to 2, softplus evidence. For R3 and R4, project each r̄_k with a shared linear layer to 64 dimensions before concatenation, so the input is 5 + 320 dimensions. For R2, use a linear layer from 1280 to 325 first, so all three heads have matched capacity.

## Gate G2. Single-view signal

Report, per row, single-view video AUROC on FF++ validation, the VALmix-v2 score and development-pool AUROC.

Pass if R4 reaches AUROC ≥ .80 on FF++ validation **and** R3 beats R2 on both the VALmix-v2 score and development-pool AUROC. If R3 does not beat R2, any gain would come from the backbone rather than from prediction, so stop and report.

## Fusion rows

For each row, build the three-view system of Appearance, that artifact slot and the video view. Refit the trust weights on the 5 development folds and the threshold at FF++ validation EER, exactly as deployed.

## Gate G3. Fusion

The decisions use development data only.
- The I-JEPA view gets a nonzero trust weight in at least 4 of 5 folds.
- Its leave-one-out contribution on the development pool has a source-grouped 95% interval above zero.
- R4 versus R3 is decided on FF++ validation under JPEG q30 and resize 0.35. R4 is kept only if it is better at both.

Then evaluate every fusion row once on the test sets and report. The pre-set success criterion, reported and not used for selection, is that the kept I-JEPA row has a ten-set mean AUROC within −0.005 of R0.

## Report `analysis/ijepa/IJEPA-REPORT.md`

1. Gate verdicts G0 to G3, each PASS or FAIL with its evidence file and one sentence.
2. The G1 table, per block and per quality level.
3. The single-view table for R0 to R4.
4. The fusion table in the same format as the video-view contribution table, with all ten test sets, the mean, FPR and TPR at the threshold, the DF40 mean over 41 sets, and the four processed-test cells.
5. Trust weights per fold for every row, and how often each view is set to zero.
6. Leave-one-out contribution of each view with intervals.
7. Settings actually used (frames per video, EDL variant, selection score, PCR6+ commit), extraction time, and cache paths.
8. Caveats. I-JEPA was pretrained on ImageNet-1K. The checkpoint license is non-commercial.
# Amendment 1 to `discern-v2-ijepa-branch-brief.md`

Received 2026-09-25. Everything not changed here stays as in the brief. Leakage audit Task 7 remains dropped.

> **INCOMPLETE AS RECEIVED.** The text stops mid-sentence in section C's report list ("- The"), and
> **section D, which the order of work refers to as R7, was never received**. R7 is therefore not
> specified and is not planned here. Step 5 of the order of work is blocked until the rest arrives.

## Order of work

1. Let the current extraction finish. Run G1, then R2, R3, R4 and G2 as briefed, with R2b added.
2. Run diagnostics D1 and D2 as soon as the cache is complete. They depend on no gate.
3. Run R5 only if G2 passes.
4. Run R6 if G0 passed, whatever the frozen rows scored at G1 and G2. Measure its budget first.
5. Run R7 only if both R5 and R6 are kept. *(blocked: R7 unspecified)*
6. Evaluate test sets once per final row (R0, R4, every kept row). Decisions use FF++ validation,
   VALmix-v2 or the development folds only.

## Keep rule (R5, R6, R7; pre-set)

A row is kept over its comparator only if **all three** hold. A tie on any one counts as not kept.

- Higher single-view VALmix-v2 score **and** higher single-view development-pool AUROC.
- In the fused system (Appearance + this slot + video view, trust weights and threshold refit as
  deployed), out-of-fold development AUROC at least equal to the comparator's.
- Development real-video FPR at the frozen threshold no more than 0.01 above the comparator's.

## A. R2b, regional feature control

Per frame and per block, s_bar_k is the mean over that block's 56 patches of the layer-normalised
target-encoder features (1280-d). Head input is 5 scalars (log of each ||s_bar_k||) plus the 5 s_bar_k
through the same shared 1280→64 projection; head otherwise identical to R3 and R4.

**G2 now passes only if** R4 reaches FF++ validation AUROC ≥ .80 **and** R3 beats **both** R2 and R2b
on the VALmix-v2 score and on development-pool AUROC.

*Implementation note.* The amendment permits a separate target-encoder-only pass. That was avoided:
`y = apply_masks(h, [hid])` inside the extraction loop already is the layer-normalised target feature
at the block's patches, so `s_bar_k = y.mean(1)` costs nothing and is guaranteed to share the frame's
processing draw. Extraction was restarted to cache it (`sbar`, fp16, 28,176 bytes/frame total).

## B. R5, cross-block cosine, and diagnostics D1/D2

Pairs are (upper, lower) and (left cheek, right cheek); their patch sets are disjoint, asserted in
`branches/ijepa/features.py:assert_pairs_disjoint`. The centre block overlaps all four and is excluded.

- c_ij = <r_bar_i, r_bar_j> / (||r_bar_i||·||r_bar_j|| + 1e-6).
- tau_k = 5th percentile of ||r_bar_k|| over real FF++ c23 train frames under the B mixture, frozen.
  If ||r_bar_i|| < tau_i or ||r_bar_j|| < tau_j then c_ij = 0 and v_ij = 1, else v_ij = 0.
- Video-level inputs are the frame means of c_ij and v_ij over both pairs: 4 inputs.
- R5 = the R4 head with these 4 extra inputs (329-d), nothing else changed. No signed differences of
  a_k. Shared projection kept. No per-block projections in this amendment.
- Comparator is R4. Also report R5 − R4 separately on the reenactment and swap parts of VALmix-v2.

**D1.** FF++ c23 validation, logistic regression on the 4 inputs alone, 5-fold source-grouped CV,
AUROC per manipulation method. Pre-registered expectation: Face2Face and NeuralTextures above
Deepfakes and FaceSwap. Reported either way.

**D2.** Development-pool real videos: mean, sd and 5th/50th/95th percentiles of each c_ij per source
dataset. Flag any source whose median differs from FF++ validation real videos by more than one FF++ sd.

## C. R6, predictor adapted on real faces

Budget gate: adapt and re-extract on 100 videos first, project the total, continue only if adaptation
plus full re-extraction is ≤ 30 GPU hours, else stop and report.

- **Data.** Real FF++ c23 train videos only, same frames as the Appearance view, B mixture per clip.
  Validation on real FF++ c23 val, clean. No fake video in adaptation or checkpoint selection.
- **Masks.** I-JEPA's own collator (`src/masks/multiblock.py`) with `configs/in1k_vith14_ep300.yaml`:
  4 target blocks, pred_mask_scale 0.15–0.2, aspect_ratio 0.75–1.5; 1 context block, enc_mask_scale
  0.85–1.0, min_keep 10, allow_overlap false. **Verified present in the cloned repo and config.**
  Never train on the five fixed evaluation blocks.
- **Model.** encoder and target_encoder frozen, eval mode, `no_grad`. Predictor base weights frozen.
  LoRA r8, alpha 16, dropout 0.1 on `attn.qkv` and `attn.proj` in all 12 predictor blocks. Predictor
  embedding, output projection and mask token frozen.
- **Loss.** As I-JEPA training: smooth L1 to layer-normalised target features at hidden patches.
- **Optimisation.** AdamW, lr 1e-4, wd 0.05, 1 warmup epoch then cosine, 10 epochs, bf16, largest
  batch that fits.
- **Selection.** Lowest prediction loss on the five fixed evaluation blocks, FF++ c23 val real.
- **Report before moving on.** Loss curves; mean fixed-block error for frozen vs adapted predictor on
  FF++ val real, FF++ val fake (evaluation only, never selection) and development-pool real per source
  dataset; *(remainder of this list was truncated in transmission)*.

## D. R7

**Not received.** Nothing is planned or run for R7 until the text arrives.

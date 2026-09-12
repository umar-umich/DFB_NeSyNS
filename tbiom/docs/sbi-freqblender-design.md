# SBI / FreqBlender training comparison — design and integration

Companion to `freqblender-source-audit.md`. This document is the predeclaration required before
any arm runs. It does not alter `stage3-design.md`, which stays frozen.

## What is already true in this checkout, verified

| claim | verified |
|---|---|
| Stage 2 has 20 processing passes | yes — 20 CSVs in `logs/tbiom/stage2/` |
| commits `9cb11ae`, `6e54077` | both present |
| `tbiom/docs/stage3-design.md` | present, untouched |
| arm B canonical checkpoint | `run1_auxedl_seed42/.../epoch=05-val_auroc_video=0.9955.ckpt`, preserved as `R0` |
| SBI generator integration | `training/dataset/sbi_api.py`, `sbi_dataset.py` — a real generator, not only detector registration |

## The integration problem, and why it is tractable

Two chassis are in play, as they were for Branch 3.

- **SBI** lives in `DFB_NeSyNS/training/` and generates blends from **source frames plus
  81-point landmarks on disk**.
- **Arm B** trains in `DISCERN_Ext` from **pre-extracted 256x256 crops in h5 files**
  (`src/dataset/deepfake.py`, `H5DeepfakeDataset`). The h5 carries pixels only: no landmarks,
  no attrs.

So SBI cannot be dropped into arm B's loader as it stands. The blocker that would have ended
this, though, does not exist:

**81-point landmarks are complete.** `sbi_api` slices `landmark[:68]` and indexes 68..80, so it
needs dlib's 81 points, not our 5-point RetinaFace set. `landmarks81/` exists for 1000 FF++
youtube videos at shape `(81, 2)`, and **all 720 videos of the FF++ train split are covered**.
SBI is feasible on our actual training data.

### Multiview consistency is satisfied for free

The brief makes recomputation mandatory: a synthetic image must not be paired with stale cached
features from another branch. In arm B this cannot happen by construction. All three branches
derive from the same input image inside one forward pass — `semantic` and `artifact` from the
CLIP trunk (`artifact` is the orthogonal residual `f_s - f_c`), `fsvfm` from the frozen face
encoder. There is no per-branch feature cache to go stale. Feeding a synthesised image through
the model recomputes every branch by definition.

This is worth stating rather than assuming, because it is the one architectural property that
makes a single-stream pilot unnecessary here.

### Route chosen

A new dataset path in `DISCERN_Ext` that reads FF++ frames plus `landmarks81` from disk and
generates host/donor pairs on the fly, rather than pre-generating an h5. On-the-fly is required
by the brief's step ordering: processing must be applied *after* synthesis with the same sampled
settings for both classes, which a pre-baked h5 cannot express.

## Arms

`R0` is the historical arm B result, preserved and not re-run.

| arm | change | purpose | status |
|---|---|---|---|
| A | existing recipe under the common integration pipeline | matched training baseline | runnable |
| B | A + predeclared class-balanced processing | processing exposure alone | runnable |
| C | B + SBI replacing 50% of fake slots | spatial synthesis benefit | runnable |
| D | C + FreqBlender on 20% of SBI donors | frequency synthesis benefit | **blocked** |

D is blocked on `net/resnet/checkpoints/resnet34.tar`, the authenticity teacher. Both routes to
D pass through it — the released FPNet weights are behind the same Baidu share, and training
FPNet requires it at `main.py:50`. See the audit.

**A may differ from R0**, and that must be reported as a preprocessing difference rather than
attributed to any arm's treatment: A runs through the new on-the-fly loader while R0 read the
h5. Quantifying A - R0 is the first thing the comparison must publish about itself.

### Predeclared settings, frozen before any target evaluation

- replacement fraction 50% of fake slots in C and D; FreqBlender applied to 20% of SBI donors
  in D. Both are project settings, **not paper hyperparameters**.
- C and D share parent draws, SBI seeds, replacement fraction and final processing settings.
- dedicated RNG streams for synthesis, so adding FreqBlender does not perturb source selection
  or any other augmentation draw.
- class balance, update count, batch size and example budget held comparable; no extra
  optimisation steps for synthetic arms without a matched-budget control.
- a common eligible source pool, so landmark failures do not filter difficult examples out of
  the synthetic arms only. Generator failures counted and reported.
- screening on a common seed using dev only, then three seeds for anything that survives,
  matching the existing seed protocol (42 / 1337 / 7).

## Verification gates before any expensive run

Ordered, and all must pass:

1. DCT/IDCT round-trip, channel order, value ranges; strict loading of any official checkpoint.
2. Artificial-mask checks — structural-only returns the donor, semantic-only returns the host,
   skipping the blend returns the SBI image unchanged. Log the learned mask-sum error rather
   than assuming the masks partition exactly.
3. Visual inspection of genuine / SBI / blended / processed examples; log clipping rates.
4. Split safety for every derived example, host, donor, teacher sample and facebank reference.
   All processing variants of one original video stay together; identities counted where donor
   pairs cross videos.
5. Label mapping asserted explicitly — upstream index 1 is fake, ours is too, but asserted.
6. Landmark-failure handling, sample counts, throughput, short optimisation smoke run.

A passing smoke run is not evidence of generalisation and will not be reported as such.

## Evaluation

Existing video grouping and aggregation. Video AUROC with group counts; FPR/TPR at each arm's
own baseline-dev threshold then fixed across processing conditions; retrospective matched-TPR
figures reported separately from deployable ones; worst evaluated group FPR; calibration where
outputs support it; clean, held-out-processing and genuine-source-shift results kept apart.

Stage 2 already established that **a stable AUROC does not imply a stable operating point** —
jpeg q30 and resize 0.35 move the decision in opposite directions from one frozen threshold
(FPR 0.089 vs 0.584). Any arm claiming robustness must show it at the operating point, not only
in AUROC.

## Connection to M1/M2/M3

Selection on dev only. The winning configuration is then frozen, its measurements recomputed,
and M1/M2/M3 re-run on identical deployment-available inputs.

**New training invalidates the fitted relationships, not just the checkpoint.** Arm B's
reliability curves, feature caches and calibration parameters describe arm B. A retrained
detector needs its own Stage 1 export and its own Stage 2 curves before M1/M2/M3 mean anything
about it. Reusing them would be the same error as mixing fusion conventions.

Not added to C/D by default: the paired-ranking loss, new symbolic rules, a severity estimator.
Each needs its own same-data ablation after the ordinary augmentation comparison returns.

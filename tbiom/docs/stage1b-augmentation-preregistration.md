# Stage 1B — matched augmentation extension: frozen before launch

Written 2026-09-20 before the extended arm was trained or any of its outputs scored. Governs
`tbiom/reports/EXP_AUGMENTATION_EXTENSION.md`. Cites `tbiom/reports/REFERENCE_MANIFEST.md`.

## Question

Does **composed** processing (resize then JPEG) help beyond the mutually exclusive arm-B mixture? It does not test the
ceiling of FF++ generalisation, and one unsuccessful extension does not establish one.

## Arms

| processing | existing B (`sbi_armB_seed42`) | extended (`sbi_armE_seed42`) |
|---|---:|---:|
| unchanged | 40% | 40% |
| JPEG only (q70 / q50 / q30, equal shares) | 30% | 20% |
| resize only (0.75 / 0.50 / 0.35, equal shares) | 30% | 20% |
| resize then JPEG (scale and quality drawn independently from the same sets) | 0% | 20% |

* **Branch adapted:** the spatial path only. Experiment 3 on the reference measured the development drop clean → JPEG
  q40 as T 0.0498, APP 0.0994, OM 0.0813, so the spatial views are the processing-sensitive branch; Appearance and
  Off-manifold share the adapted CLIP path and are one coupled experiment. The temporal branch stays frozen at the
  reference checkpoint; the reference spatial checkpoint is not touched; the β-VAE configuration is unchanged.
* **Matched:** the extended arm's config is `sbi_armB.yaml` with `training_arm: E` and nothing else — same
  initialisation, seed, budget (20 epochs), sample count, class balance, optimiser, losses. The control is the existing
  arm-B run at its development-selected epoch 8 (the same checkpoint and exports used as the control in Experiment 5).
* **Draws** are per training sample and independent of label and manipulation type. The spatial detector trains on
  single frames, so "consistent parameters across a clip" has no frame sequence to apply to on this branch.
* JPEG q40 and resize 0.60 never occur in training; they remain evaluation-only severities.
* Checkpoint selection: the unchanged v3.1 development rule over each arm's saved candidates.
* After training, all three views are fused with the frozen temporal branch; DS-R and the logistic comparator are
  refitted per arm on the same source-grouped folds, where every clean and processed version of a video shares its
  group. Processed development uses the fixed manifests of every earlier experiment.

## Acceptance, fused three-view level, development only, DS-R primary (logistic reported beside it)

Δ = extended − B on the v3.1 objective over the 5 selection folds, paired source-group bootstrap, 1000 draws per split.

1. **Primary gain (the purpose of the extension):** mean Δ over the two processed development conditions ≥ **+0.005**,
   with the bootstrap CI lower bound > 0 in **≥ 3 of 5** splits.
2. **Regression tolerances**, any one of which rejects:
   * clean development Δ < **−0.005**;
   * reenactment Δ negative with the CI upper bound < 0 in ≥ 3 of 5 splits (the arm-C precedent);
   * clean-development FPR at the fusion's own τ up by more than **+0.02** (the Experiment 5 limit, unchanged).
3. **Threshold discipline:** FPR/TPR are reported both at the control's numeric τ (drop-in compatibility) and at each
   arm's own τ refitted by the same EER-on-FF++-val procedure (comparative operation); neither is refitted per test
   condition. The decision uses the refitted τ, as in Experiment 5.
4. Test sets, CDFv3 included, are read only after the verdict and never enter it.

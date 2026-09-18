# Void results — kept for the record, never cited

## EXP_FREQBLENDER_frozen_views (2026-09-18)

The first completed FreqBlender comparison. Both arms returned **identical numbers in every cell**, Δ = 0.0000 with
zero-width bootstrap intervals, which is a design error and not a null result.

Cause: the arms were built on the warm-start + gated recipe (`warm_dicome_gate_*`), and that recipe sets
`freeze_views: true` — everything except the face-prior branch and its gate is frozen and runs in eval mode. Comparing
the two checkpoints directly: of 1,015 tensors, **12 differ, all of them in `fsvfm_branch` and the gate**. The
Appearance and Off-manifold views are bit-identical between the arms, so a training-data intervention could not reach
the views the study is about, and the face prior — the only thing that could move — is excluded from the reference
fusion by design.

The FreqBlender generator itself was fine here (v2 geometry, 30-epoch FPNet, 23,006 pseudo-fakes); only the arms were
wrong. The corrected study keeps the frozen acceptance criteria and replaces the arms with the `sbi_arm*` family,
which trains the Appearance and Off-manifold views: control `sbi_armB_seed42` (arm B processing, development-selected
epoch 8), treatment `sbi_armD_seed42` (identical config, arm D at 10% of fake slots).

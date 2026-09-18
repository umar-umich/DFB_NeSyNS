# Experiment 7 — acquisition pilot, Step 1 (controls and the go/no-go)

OMP_NUM_THREADS=4 python analysis/tbiom/acquisition/step1_controls.py

THE DECISION. The frozen reference detector already scores a video from one full-span observation. Given a budget of
at most two ADDITIONAL clip-window inspections, which window should be inspected? Step 1 answers whether that choice
is worth learning at all, using only cached outputs: candidate windows were precomputed by `cache_windows.py`, and a
window's Appearance / Off-manifold opinion is the mean of that window's cached per-frame opinions, so no new inference
happens here.

STATE AND RULES (brief §Experiment 7).
  * State available at inference: the full-span observation's per-branch opinions, their disagreement, the candidate
    windows' label-free quality descriptors, which windows have been inspected, and the remaining budget. Labels exist
    only while a controller is being fitted, never at inference.
  * Unselected windows' detector outputs are HIDDEN from every policy until that window is chosen. A policy that could
    see them would be choosing with the answer in hand.
  * Acquired observations are aggregated WITHIN each branch (mean opinion over the observed windows) and only then
    fused, so overlapping clips are never treated as independent sources.
  * Fusion and thresholds are the deployed ones: DS-R weights fitted on the fitting fold, tau = EER of the
    no-acquisition score on the fold's FF++ val videos.
  * Policy training uses the FITTING folds; everything is reported on the SELECTION folds. Same 5 source-grouped
    splits as every other experiment.

GO/NO-GO, frozen in tbiom/docs/decision-2026-09-18.md before these numbers were produced. PPO is justified only if
  1. headroom:      an oracle with the same budget improves selection-fold AUROC by >= 0.005 over no acquisition;
  2. learnability:  the supervised one-step selector beats the better of fixed-uniform and random by >= 0.002 AUROC;
  3. variation:     the oracle's chosen window is the modal window on < 60% of videos.
A bounded two-step assessment is reported beside the one-step numbers so that a lack of one-step benefit does not by
itself rule out sequential value.

Every number below uses the reference manifest `tbiom/reports/REFERENCE_MANIFEST.md` (three resolved checkpoints, DS-R definition, fold splits). Script `analysis/tbiom/temporal/finish_line.py`.

## 1. Policies at equal budget (selection folds, mean of 5 splits)

| policy | additional inspections | AUROC | Δ vs no acquisition | log loss | FPR | TPR |
|---|---:|---:|---:|---:|---:|---:|
| no acquisition (k=0) | 0 | 0.9068 | +0.0000 | 0.4706 | 0.107 | 0.737 |
| fixed window w0 (k=1) | 1 | 0.8896 | -0.0172 | 0.4431 | 0.145 | 0.749 |
| fixed best-on-fitting w3 (k=1) | 1 | 0.8932 | -0.0135 | 0.4310 | 0.150 | 0.762 |
| random window (k=1) | 1 | 0.8917 | -0.0151 | 0.4354 | 0.149 | 0.758 |
| quality-based (k=1) | 1 | 0.8938 | -0.0130 | 0.4320 | 0.145 | 0.758 |
| supervised one-step (k=1) | 1 | 0.8923 | -0.0144 | 0.4330 | 0.147 | 0.759 |
| oracle (k=1) | 1 | 0.9378 | +0.0310 | 0.3297 | 0.107 | 0.812 |
| oracle on permuted windows (null, k=1) | 1 | 0.9855 | +0.0787 | 0.1536 | 0.105 | 0.963 |
| fixed windows w0+w2 (k=2) | 2 | 0.8873 | -0.0194 | 0.4326 | 0.161 | 0.758 |
| supervised greedy (k=2) | 2 | 0.8873 | -0.0195 | 0.4315 | 0.165 | 0.761 |
| oracle (k=2) | 2 | 0.9290 | +0.0223 | 0.3416 | 0.125 | 0.813 |

## 2. The frozen go/no-go

1. **Headroom** — oracle gain +0.0310 at k=1 and +0.0223 at k=2 (criterion ≥ +0.005) → **met** as written. It should not be read as real headroom: the NULL oracle, which picks among candidate scores borrowed from other videos and therefore cannot know anything about this one, still gains +0.0787. Most of the oracle's apparent room is selection on labels among four correlated scores, which is exactly what no deployable policy can use.
2. **Learnability** — supervised one-step 0.8923 vs the best non-learned control 0.8932, difference -0.0009 (criterion ≥ +0.002) → **NOT met**.
3. **Variation** — the oracle's chosen window is the modal window on 26% of videos (criterion < 60%); pooled choice histogram over windows w0…w3: [3306, 3711, 3702, 3716] → **met**.

**Verdict: DO NOT proceed to PPO** — at least one condition fails, and the brief makes PPO conditional on all three.

## 3. Cost

One additional inspection costs one temporal clip forward plus that window's spatial frames, on top of decoding the window and computing its quality descriptors; the controller itself is a gradient-boosted regressor whose inference cost is negligible beside a ViT-L forward. Measured on the cache run: 29,070 window-clips took 13.2 GPU-minutes, i.e. ≈27 ms per window inspection for the temporal branch. A k=2 policy therefore triples the detector cost per video for the gains in the table above.

## 4. Reading

These are full-population numbers: every video is scored, and no video is rejected. An improvement that appeared only after discarding hard videos would be a selective-prediction result, not an AUROC improvement, and is not claimed here. Novelty would need comparison against existing frame-selection methods (AdaFrame, AdaFocus) before any claim is made.

One detector seed and one temporal-branch seed unless stated. CDFv2 and DeepfakeEval2024 informed development and are not untouched confirmation data. FaceShifter is sealed (`tbiom/docs/SEALED_BENCHMARKS.md`).

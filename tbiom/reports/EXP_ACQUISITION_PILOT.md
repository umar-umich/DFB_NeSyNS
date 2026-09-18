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

## 0. The aggregation rule for acquired observations

The first pass averaged an acquired window into the branch's opinion, which is the obvious rule and also a way to lose: a denser look at one segment gets averaged into an estimate that already spans the video. Two other within-branch rules are tried before concluding anything (evidence summation is not among them — the brief forbids treating overlapping clips as independent sources).

| aggregation | no acquisition | supervised one-step (k=1) | best non-learned control | oracle (k=1) |
|---|---:|---:|---:|---:|
| mean | 0.9068 | 0.8923 | 0.8938 | 0.9378 |
| conf **(reported below)** | 0.9068 | 0.8952 | 0.8973 | 0.9382 |
| max | 0.9068 | 0.8825 | 0.8859 | 0.9410 |

The rest of this report uses **conf**, the aggregation whose learned selector scores highest.

## 1. Policies at equal budget (selection folds, mean of 5 splits)

| policy | additional inspections | AUROC | Δ vs no acquisition | log loss | FPR | TPR |
|---|---:|---:|---:|---:|---:|---:|
| no acquisition (k=0) | 0 | 0.9068 | +0.0000 | 0.4706 | 0.107 | 0.737 |
| fixed window w0 (k=1) | 1 | 0.8939 | -0.0128 | 0.4430 | 0.152 | 0.755 |
| fixed best-on-fitting (k=1) | 1 | 0.8937 | -0.0131 | 0.4417 | 0.161 | 0.768 |
| random window (k=1) | 1 | 0.8946 | -0.0122 | 0.4402 | 0.157 | 0.764 |
| quality-based (k=1) | 1 | 0.8973 | -0.0095 | 0.4350 | 0.151 | 0.764 |
| supervised one-step (k=1) | 1 | 0.8952 | -0.0115 | 0.4378 | 0.155 | 0.766 |
| oracle (k=1) | 1 | 0.9382 | +0.0315 | 0.3301 | 0.115 | 0.816 |
| oracle on permuted windows (null, k=1) | 1 | 0.9855 | +0.0788 | 0.1494 | 0.110 | 0.964 |
| fixed windows w0+w2 (k=2) | 2 | 0.8921 | -0.0147 | 0.4284 | 0.172 | 0.772 |
| supervised greedy (k=2) | 2 | 0.8919 | -0.0149 | 0.4287 | 0.174 | 0.777 |
| oracle (k=2) | 2 | 0.9302 | +0.0234 | 0.3397 | 0.130 | 0.817 |

## 2. The frozen go/no-go

1. **Headroom** — oracle gain +0.0315 at k=1 and +0.0234 at k=2 (criterion ≥ +0.005) → **met** as written. It should not be read as real headroom: the NULL oracle, which picks among candidate scores borrowed from other videos and therefore cannot know anything about this one, still gains +0.0788. Most of the oracle's apparent room is selection on labels among four correlated scores, which is exactly what no deployable policy can use.
2. **Learnability** — supervised one-step 0.8952 vs the best non-learned control 0.8973, difference -0.0020 (criterion ≥ +0.002) → **NOT met**.
3. **Variation** — the oracle's chosen window is the modal window on 27% of videos (criterion < 60%); pooled choice histogram over windows w0…w3: [3330, 3582, 3829, 3694] → **met**.

**Verdict: DO NOT proceed to PPO** — at least one condition fails, and the brief makes PPO conditional on all three.

## 3. Cost

One additional inspection costs one temporal clip forward plus that window's spatial frames, on top of decoding the window and computing its quality descriptors; the controller itself is a gradient-boosted regressor whose inference cost is negligible beside a ViT-L forward. Measured on the cache run: 29,070 window-clips took 13.2 GPU-minutes, i.e. ≈27 ms per window inspection for the temporal branch. A k=2 policy therefore triples the detector cost per video for the gains in the table above.

## 4. Reading

These are full-population numbers: every video is scored, and no video is rejected. An improvement that appeared only after discarding hard videos would be a selective-prediction result, not an AUROC improvement, and is not claimed here. Novelty would need comparison against existing frame-selection methods (AdaFrame, AdaFocus) before any claim is made.

One detector seed and one temporal-branch seed unless stated. CDFv2 and DeepfakeEval2024 informed development and are not untouched confirmation data. FaceShifter is sealed (`tbiom/docs/SEALED_BENCHMARKS.md`).

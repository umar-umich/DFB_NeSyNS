# Phase 3 Handoff — NeSy-DeFake

Four independent tasks. Complete each as a **separate self-contained change**
with its own verification command. Do not entangle them in one commit.

## Global rules (from prior phases — still apply)
- Never delete files. Move dead code to `attic/` preserving relative paths.
- Never modify `configs/retained_predicates.yaml` or the v8 rule logic.
- Every existing config must remain runnable after your changes.
- CRITICAL: no code path may read OOD test datasets during training,
  threshold selection, or hyperparameter choice. Add a guard/assert in any
  new calibration/threshold code that its input dataset == FF++.
- After each task: print files changed + a verification command.

## Task risk levels
- **T1, T2 touch training/eval code — review-critical.** Generate clean diffs.
- **T3, T4 are new standalone files — touch no existing code.**
- **T4 must NOT be wired into any training path** — module + unit test only.

---

## T1 — Feature-space augmentation for symbolic inputs
Add config flag `training.feature_augment: {enabled: false, gauss_std: 0.05,
feature_dropout: 0.1}`. When enabled, **during TRAINING only**: add zero-mean
Gaussian noise (std relative to each feature's FF++-train std) and random
per-dimension dropout to `precomputed_attrs` (58-d) and `forensic_features`
(83-d) **before** they enter the concept/CCV branches. Deterministic
passthrough at eval. Compute the per-feature train std once on the FF++ train
split and freeze it; fall back to per-batch std only if that's unavailable.
- Enable in `f2_full`. Create `f2_noaug` = exact copy of `f2_full` with
  `feature_augment.enabled: false` and nothing else different.
- Verify: both configs load and construct the model; a training step runs
  with augment on and off.

## T2 — Risk-head diagnostics
In `risk_head.py`, additionally: (a) dump the pooled LOMO+OOD design matrix
to `results/risk_design.npz` (arrays `V, C, Q, T, err`, err=1 if the video
was misclassified); (b) compute and append to `results/risk_report.md` the
V/C/Q/T Pearson correlation matrix and the **standardized univariate**
error-prediction AUROC for each of V, C, Q, T. Purpose: explain the negative
learned weight on C despite FNs carrying higher C (multicollinearity check).
- Verify: `risk_report.md` shows a 4×4 correlation table and 4 univariate
  AUROC values; `risk_design.npz` exists.

## T3 — Shift-aware feature audit (offline, cached features only)
New `scripts/feature_shift_audit.py`. Load cached `fast_semantic` (58-d) and
`forensic_features` (83-d) for FF++ train + all six eval sets, **reals only**
(measure pipeline shift uncontaminated by manipulation signal). For each
feature group — semantic: expressions / AUs / pose / gaze / geometry /
quality-symmetry (use the true group boundaries from
`refined_attributes.FAST_FEATURE_NAMES`, not guessed ranges); forensic: the 5
(or 3-parent) `FORENSIC_GROUPS` — compute per-feature Wasserstein-1 distance
FF++ vs each OOD set (features standardized by FF++ train stats), aggregate
per group (mean + max). Repeat everything under **per-video z-normalization**
(needs the video_id column; if the cache lacks it, use per-column z as a proxy
and flag it in the report). Output `results/feature_shift_report.md`:
(a) group × dataset shift matrix, raw vs per-video-normalized;
(b) ranked list of the highest-shift individual features with names;
(c) overlap: load the p4b checkpoint, read first-layer |weight| per input dim
    for the concept MLP and CCV anomaly/constraint heads, and report whether
    the high-shift dims are the high-weight dims.
Pure analysis — no training code touched.
- Verify: report file written with all three sections populated.

## T4 — Differentiable forensic bank (spec + test only, NOT integrated)
New `networks/nesy_defake/forensic_gpu.py`: torch implementations of the 83
forensic features grouped per `FORENSIC_GROUPS` (Sobel boundary gradients,
regional Laplacian blur, bilateral symmetry, color-hist chi-square, DCT/FFT
statistics, SRM 5-filter residual stats, patch / cross-channel / multi-scale
noise), operating on `(B,3,224,224)` in `[0,1]`. Unit test: cosine similarity
> 0.95 **per group** against the cached CPU features on 100 clean FF++ frames;
document any feature where parity is impossible and why (e.g. patch-noise
requiring the exact CPU patch grid). Do NOT wire into training — integration
is a separate decision gated on the T3 report and the f2 vs f2_noaug result.
- Verify: unit test runs and prints per-group cosine similarity.

---

## After completion, report back
- T1: confirmation both configs run; the augment module path.
- T2: the correlation table + univariate AUROCs.
- T3: `feature_shift_report.md`.
- T4: per-group parity numbers + list of documented gaps.

## Do NOT do in this phase
- Do not launch the full training runs (f1/f2/f3) — that's a separate manual
  step after T1's configs are reviewed.
- Do not integrate the forensic bank.
- Do not touch the frozen predicate set or retrain the predicate selection.
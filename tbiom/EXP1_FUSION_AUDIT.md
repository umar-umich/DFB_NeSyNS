# Experiment 1 — cached fusion audit (warm+gate, epoch 19)

Replay only: no detector parameter was trained. All arms read the same saved per-frame opinions of the selected warm+gate checkpoint (Appearance and Off-manifold residual = frozen DiCoME views; Bona-fide face prior raw and gated). Script: `analysis/tbiom/exp1_fusion_audit.py`; raw numbers: `tbiom/EXP1_FUSION_AUDIT.json`.

## Protocol

- Development data: the v3.1 selection data (VALmix, VALmix-v2 in-scope subsets, FF++ val) — 5814 videos in 2728 source-video groups (largest 25 videos). Groups join a fake with the genuine video it was made from and with every fake sharing that source (Celeb-DF target video, DFDCP target clip, DeepSpeak clip ids, FF++/DF40 identity ids); DFEval24 and FFIW names carry no source linkage and are grouped per name.
- 5 random grouped 50/50 splits into a fusion-fitting fold and a fusion-selection fold; no group in both.
- Fitted on the fitting fold only: constant g (grid 0..1 step 0.05, v3.1 objective); logistic (C by grouped inner 3-fold CV on log-loss, balanced classes); MLP (one hidden layer, size and L2 by grouped inner CV, class-balanced resampling, early stopping, 5 random initialisations — initialisations, not detector seeds).
- Input parity: logistic and MLP get per-view p, vacuity u, log-evidence and logit for all three views (raw prior), cross-view std and range of p, and the three pairwise DS conflict masses — a superset of the gate's inputs (p, u of each view), video means of per-frame values.
- Compared on the selection fold with the v3.1 objective (macro AUROC over swap / reenactment / in-the-wild cells). Group bootstrap (1000 resamples) for Δ = gate − arm. Test sets were read only after, never to choose.
- τ per arm and split: EER threshold on the selection fold's FF++ val videos; FPR/TPR at τ are deployable rates.
- Replay check: DS over (Appearance, Off-manifold, gated prior) reproduces the exported fused output to 5.4e-07.

Decision rule, fixed before the run: CANDIDATE CONTRIBUTION if mean Δ(gate − MLP) ≥ +0.005 and the 95% CI lower bound > 0 in ≥ 4/5 splits; STACKING MATCHES/BEATS if mean Δ < +0.005 and the CI upper bound < +0.005 in ≥ 4/5 splits; otherwise UNRESOLVED.

## Verdict: **UNRESOLVED: the result is close; not a win and not proof that fusion is exhausted**

## 1. Selection fold (v3.1 objective) — mean ± sd over the 5 grouped splits

| arm | v3.1 score | swap | reenactment | in-the-wild |
|---|---:|---:|---:|---:|
| gate | 0.8204 ±0.0090 | 0.9092 ±0.0109 | 0.7769 ±0.0172 | 0.7751 ±0.0078 |
| g=1 | 0.8179 ±0.0094 | 0.9054 ±0.0112 | 0.7764 ±0.0173 | 0.7720 ±0.0082 |
| const g | 0.8255 ±0.0075 | 0.9235 ±0.0106 | 0.7712 ±0.0203 | 0.7817 ±0.0066 |
| logistic | 0.8304 ±0.0111 | 0.9073 ±0.0121 | 0.7956 ±0.0177 | 0.7884 ±0.0134 |
| MLP | 0.8212 ±0.0096 | 0.9079 ±0.0126 | 0.7853 ±0.0170 | 0.7703 ±0.0127 |
| two-view | 0.8226 ±0.0060 | 0.9286 ±0.0086 | 0.7583 ±0.0179 | 0.7810 ±0.0083 |

MLP rows average the initialisations within each split. Spread across initialisations (selection-fold v3.1 score, per split): split 0 0.8029–0.8093; split 1 0.8214–0.8281; split 2 0.8335–0.8369; split 3 0.8127–0.8238; split 4 0.8085–0.8275

## 2. Gate vs each alternative — Δ = gate − arm on the selection fold, 95% group-bootstrap CI

| split | fitted g | logistic C | MLP (hidden, L2) | vs MLP (median init) | vs logistic | vs g=1 | vs const g | vs two-view |
|---|---:|---:|---|---|---|---|---|---|
| 0 | 0.05 | 10.0 | (64, 0.01) | +0.0031 [-0.0104, +0.0183] | -0.0056 [-0.0214, +0.0107] | +0.0023 [+0.0009, +0.0038] | -0.0065 [-0.0109, -0.0019] | -0.0033 [-0.0128, +0.0061] |
| 1 | 0.05 | 10.0 | (64, 0.01) | +0.0007 [-0.0098, +0.0112] | -0.0077 [-0.0218, +0.0073] | +0.0022 [+0.0010, +0.0035] | -0.0061 [-0.0109, -0.0018] | -0.0016 [-0.0112, +0.0077] |
| 2 | 0.00 | 10.0 | (64, 0.0001) | -0.0099 [-0.0179, -0.0017] | -0.0219 [-0.0363, -0.0075] | +0.0017 [+0.0006, +0.0029] | +0.0026 [-0.0074, +0.0121] | +0.0026 [-0.0074, +0.0121] |
| 3 | 0.05 | 10.0 | (16, 0.01) | -0.0077 [-0.0231, +0.0074] | -0.0131 [-0.0283, +0.0041] | +0.0037 [+0.0025, +0.0052] | -0.0102 [-0.0146, -0.0061] | -0.0101 [-0.0190, -0.0016] |
| 4 | 0.05 | 10.0 | (16, 0.0001) | +0.0074 [-0.0066, +0.0214] | -0.0018 [-0.0173, +0.0141] | +0.0025 [+0.0012, +0.0037] | -0.0050 [-0.0097, -0.0006] | +0.0012 [-0.0080, +0.0106] |

## 3. Test sets (read after selection; fitted arms averaged over the 5 splits)

### 3a. Video AUROC

| test | gate | g=1 | const g | logistic | MLP | two-view |
|---|---:|---:|---:|---:|---:|---:|
| CDFv2 | 0.9539 | 0.9515 | 0.9650 | 0.9599 | 0.9627 | 0.9693 |
| CDFv3 | 0.8881 | 0.8879 | 0.8807 | 0.9227 | 0.9083 | 0.8691 |
| DFD | 0.9462 | 0.9459 | 0.9430 | 0.9078 | 0.9224 | 0.9343 |
| DFDC | 0.8887 | 0.8869 | 0.8894 | 0.8889 | 0.8842 | 0.8818 |
| DFDCP | 0.8986 | 0.8975 | 0.8940 | 0.8667 | 0.8689 | 0.8755 |
| DFEval24 | 0.6907 | 0.6902 | 0.6879 | 0.6446 | 0.6436 | 0.6769 |
| FFIW | 0.9093 | 0.9062 | 0.9231 | 0.9055 | 0.9001 | 0.9284 |
| KoDF | 0.8434 | 0.8421 | 0.8494 | 0.8596 | 0.8546 | 0.8509 |
| IDForge | 0.9660 | 0.9656 | 0.9645 | 0.9489 | 0.9581 | 0.9598 |
| DFDM | 0.9918 | 0.9911 | 0.9921 | 0.9938 | 0.9921 | 0.9909 |
| CDFv3 reenactment | 0.8612 | 0.8621 | 0.8457 | 0.9110 | 0.8876 | 0.8271 |
| CDFv3 swap | 0.9376 | 0.9354 | 0.9451 | 0.9443 | 0.9462 | 0.9463 |
| KoDF first-order motion | 0.7068 | 0.7031 | 0.7201 | 0.7241 | 0.7176 | 0.7206 |
| CDFv2 base | 0.9539 | 0.9515 | 0.9650 | 0.9599 | 0.9627 | 0.9693 |
| CDFv2 jpeg40 | 0.8426 | 0.8397 | 0.8573 | 0.7710 | 0.7835 | 0.8625 |
| CDFv2 rs06 | 0.9386 | 0.9363 | 0.9478 | 0.9295 | 0.9396 | 0.9507 |
| DFDCP base | 0.8986 | 0.8975 | 0.8940 | 0.8667 | 0.8689 | 0.8755 |
| DFDCP jpeg40 | 0.8751 | 0.8732 | 0.8611 | 0.7941 | 0.8039 | 0.8185 |
| DFDCP rs06 | 0.8344 | 0.8349 | 0.8226 | 0.7972 | 0.8012 | 0.7986 |
| **OOD mean (10 sets)** | **0.8977** | **0.8965** | **0.8989** | **0.8898** | **0.8895** | **0.8937** |

KoDF here is FSGAN, DFL, FaceSwap and first-order motion (the brief lists KoDF as reenactment; only its first-order-motion subset is, so that subset is reported separately).

### 3b. Real-side FPR at frozen τ (and TPR)

| test | gate | g=1 | const g | logistic | MLP | two-view |
|---|---|---|---|---|---|---|
| CDFv2 | 0.084 / 0.835 | 0.076 / 0.816 | 0.121 / 0.900 | 0.039 / 0.765 | 0.064 / 0.839 | 0.157 / 0.939 |
| CDFv3 | 0.084 / 0.700 | 0.076 / 0.691 | 0.121 / 0.735 | 0.039 / 0.656 | 0.064 / 0.706 | 0.157 / 0.754 |
| DFD | 0.072 / 0.804 | 0.067 / 0.800 | 0.106 / 0.830 | 0.036 / 0.769 | 0.076 / 0.806 | 0.120 / 0.849 |
| DFDC | 0.227 / 0.836 | 0.222 / 0.832 | 0.260 / 0.850 | 0.164 / 0.789 | 0.215 / 0.814 | 0.287 / 0.849 |
| DFDCP | 0.357 / 0.945 | 0.360 / 0.947 | 0.369 / 0.937 | 0.241 / 0.803 | 0.269 / 0.834 | 0.371 / 0.902 |
| DFEval24 | 0.277 / 0.554 | 0.279 / 0.556 | 0.276 / 0.554 | 0.065 / 0.271 | 0.126 / 0.339 | 0.286 / 0.562 |
| FFIW | 0.071 / 0.710 | 0.070 / 0.706 | 0.076 / 0.745 | 0.014 / 0.580 | 0.029 / 0.646 | 0.082 / 0.764 |
| KoDF | 0.010 / 0.593 | 0.008 / 0.588 | 0.026 / 0.632 | 0.005 / 0.591 | 0.017 / 0.623 | 0.054 / 0.678 |
| IDForge | 0.094 / 0.874 | 0.092 / 0.873 | 0.109 / 0.882 | 0.023 / 0.775 | 0.041 / 0.828 | 0.121 / 0.884 |
| DFDM | 0.074 / 0.978 | 0.065 / 0.977 | 0.108 / 0.983 | 0.037 / 0.960 | 0.056 / 0.969 | 0.146 / 0.987 |
| CDFv3 reenactment | 0.084 / 0.643 | 0.076 / 0.637 | 0.121 / 0.668 | 0.039 / 0.603 | 0.064 / 0.649 | 0.157 / 0.679 |
| CDFv3 swap | 0.084 / 0.803 | 0.076 / 0.789 | 0.121 / 0.858 | 0.039 / 0.755 | 0.064 / 0.811 | 0.157 / 0.890 |
| KoDF first-order motion | 0.010 / 0.087 | 0.008 / 0.081 | 0.026 / 0.187 | 0.005 / 0.086 | 0.017 / 0.167 | 0.054 / 0.314 |
| CDFv2 base | 0.084 / 0.835 | 0.076 / 0.816 | 0.121 / 0.900 | 0.039 / 0.765 | 0.064 / 0.839 | 0.157 / 0.939 |
| CDFv2 jpeg40 | 0.021 / 0.314 | 0.018 / 0.289 | 0.056 / 0.494 | 0.000 / 0.155 | 0.006 / 0.271 | 0.097 / 0.646 |
| CDFv2 rs06 | 0.071 / 0.794 | 0.065 / 0.778 | 0.107 / 0.855 | 0.072 / 0.750 | 0.092 / 0.818 | 0.125 / 0.881 |
| DFDCP base | 0.357 / 0.945 | 0.360 / 0.947 | 0.369 / 0.937 | 0.241 / 0.803 | 0.269 / 0.834 | 0.371 / 0.902 |
| DFDCP jpeg40 | 0.101 / 0.583 | 0.103 / 0.585 | 0.104 / 0.555 | 0.002 / 0.302 | 0.009 / 0.382 | 0.103 / 0.532 |
| DFDCP rs06 | 0.445 / 0.906 | 0.443 / 0.903 | 0.458 / 0.893 | 0.360 / 0.789 | 0.391 / 0.821 | 0.456 / 0.835 |

## 4. Limitations

- **Replay cannot establish the best fusion-training method.** The prior head was trained with the gate present, so these arms measure how alternative fusions behave on branches shaped by that gate. A prior head trained under g = 1, a constant, or a stacker could differ.
- **One warm+gate seed is one seed.** MLP initialisations and fold splits vary the fusion fit and the selection sample; they are not independent detector seeds.
- Groups for DFEval24 and FFIW validation videos are per name (no source linkage recoverable from names).
- The OOD mean is reported for context only; the verdict uses the selection fold alone.

## 5. Notes on the brief (for the human decision)

- **Sealed benchmark.** Neither candidate named in the brief can be sealed: DFEval24 and FFIW test sets are already
  in CORRECTED_V3 and BASELINE_MATRIX, and their validation halves are in VALmix / VALmix-v2 (used for selection
  here). Never-inspected face-swap candidates on this machine: FaceShifter (`GEND/extracted/test/FSh`) and UADFV
  (`GEND/extracted/test/UADFV`). No DISCERN or DiCoME model has been run on either; GenD-repo baselines were run on
  a DeepfakeBench FaceShifter split in May. Choose one to seal; it will not be built or read until final confirmation.
- **KoDF is mostly face swaps** (FSGAN, DFL, FaceSwap); only first-order motion is reenactment, reported separately.
- **Experiment 2 recipe.** The existing B recipe is not "JPEG q40, resize 0.60" — those are the held-out TEST
  severities. B trains on a label-independent mixture: none 40%, JPEG q70/q50/q30 10% each, down-up resize
  0.75/0.50/0.35 10% each (`DISCERN_Ext/src/dataset/training_arms.py`, SharedProcessing). Training on q40/0.60 would
  put the test severities into training.

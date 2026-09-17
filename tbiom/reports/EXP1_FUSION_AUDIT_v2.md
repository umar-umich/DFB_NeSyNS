# Cached fusion audit v2 — run `warmgate`

**Conditional on the development-selected checkpoint:** `warmgate_e19` (epoch 19, v3.1 score 0.8196, sha256 `fdc924a7cc9e1579`), chosen earlier by the unchanged v3.1 rule. Replay only: no detector parameter is trained. The audit asks how alternative fusions behave on this checkpoint's branches. Script `analysis/tbiom/exp1_fusion_audit.py` (v2); raw numbers in the matching `.json`.

## Protocol

- Development data: v3.1 selection data (FF++ val, VALmix, VALmix-v2 in-scope subsets), 5814 videos in 2728 source-video groups (largest 25).
- 5 random grouped 50/50 splits into a fitting fold and a selection fold (no group in both). Fusion parameters — constant g, logistic C, MLP size/L2 and weights — are fitted on the CLEAN fitting fold only.
- Arms: learned gate · g = 1 · fitted constant g · logistic and MLP stacking on RICH inputs (per-view p, vacuity, log-evidence, logit; cross-view std/range; pairwise DS conflict) · logistic and MLP stacking on MATCHED inputs (exactly the gate's: p and vacuity of the three views) · two-view (prior removed).
- Comparison on the selection fold with the v3.1 objective. **Paired, source-grouped bootstrap** (1000 draws per split): each draw resamples source groups with replacement (a group's videos move together) and scores every arm on that same draw; a gate-vs-gate self-test returned Δ = 0 on every draw.
- MLP: 5 random initialisations per fit, reported both as the initialisation average and as the median initialisation (the v1-registered comparator). Initialisations are not detector seeds.
- τ per arm and split: EER threshold on the CLEAN selection-fold FF++ val videos, frozen for everything else.
- Processed development (jpeg40, rs06): the same selection-fold videos under JPEG q40 / resize 0.60, scored with the clean-fitted arms. Reported only; never used to fit, select or decide.
- Replay check: DS(Appearance, Off-manifold, gated prior) reproduces the exported fused output to 5.4e-07.

**Decision rule (unchanged from v1).** Registered comparison: gate vs MLP stacking, rich inputs, median initialisation. CANDIDATE CONTRIBUTION if mean Δ ≥ +0.005 and 95% CI lower bound > 0 in ≥ 4/5 splits; STACKING MATCHES OR BEATS if mean Δ < +0.005 and CI upper bound < +0.005 in ≥ 4/5 splits; otherwise UNRESOLVED. The same rule applied to the other comparators is reported as secondary information, not as a new rule.

## Verdicts

| comparison (Δ = gate − comparator) | mean Δ | CI>0 splits | CI upper<+0.005 splits | verdict |
|---|---:|---:|---:|---|
| MLP (rich) median init **(registered)** | -0.0007 | 0/5 | 1/5 | UNRESOLVED |
| MLP (rich) init mean | -0.0008 | 0/5 | 1/5 | UNRESOLVED |
| MLP (matched) init mean | -0.0020 | 0/5 | 2/5 | UNRESOLVED |
| logistic (rich) | -0.0100 | 0/5 | 2/5 | UNRESOLVED |
| logistic (matched) | +0.0029 | 0/5 | 0/5 | UNRESOLVED |

## 1. Clean development — selection fold, v3.1 objective (mean ± sd over 5 splits)

| arm | v3.1 score | swap | reenactment | in-the-wild |
|---|---:|---:|---:|---:|
| gate | 0.8204 ±0.0090 | 0.9092 ±0.0109 | 0.7769 ±0.0172 | 0.7751 ±0.0078 |
| g=1 | 0.8179 ±0.0094 | 0.9054 ±0.0112 | 0.7764 ±0.0173 | 0.7720 ±0.0082 |
| const g | 0.8255 ±0.0075 | 0.9235 ±0.0106 | 0.7712 ±0.0203 | 0.7817 ±0.0066 |
| logistic (rich) | 0.8304 ±0.0111 | 0.9073 ±0.0121 | 0.7956 ±0.0177 | 0.7884 ±0.0134 |
| MLP (rich) | 0.8213 ±0.0099 | 0.9081 ±0.0125 | 0.7859 ±0.0173 | 0.7698 ±0.0135 |
| logistic (matched) | 0.8176 ±0.0094 | 0.8989 ±0.0122 | 0.7786 ±0.0164 | 0.7752 ±0.0118 |
| MLP (matched) | 0.8224 ±0.0089 | 0.9165 ±0.0109 | 0.7674 ±0.0187 | 0.7832 ±0.0111 |
| two-view | 0.8226 ±0.0060 | 0.9286 ±0.0086 | 0.7583 ±0.0179 | 0.7810 ±0.0083 |

MLP rows average the initialisations. Median-initialisation scores per split: rich 0.8065, 0.8259, 0.8357, 0.8161, 0.8212; matched 0.8096, 0.8276, 0.8329, 0.8149, 0.8258.

### Arithmetic reconciliation (v1)

v1's table row "MLP" averaged the five initialisations, while v1's Δ and verdict used the median initialisation; the two differ because the median initialisation is not the average. Both are reported below with their own paired bootstrap, so every Δ equals the difference of two numbers shown in this report.

## 2. Δ = gate − comparator on the clean selection fold, 95% paired group-bootstrap CI

| split | g | MLP (rich) median init | MLP (rich) init mean | logistic (rich) | MLP (matched) median init | MLP (matched) init mean | logistic (matched) | g=1 | const g | two-view |
|---|---:|---|---|---|---|---|---|---|---|---|
| 0 | 0.05 | +0.0031 [-0.0104, +0.0183] | +0.0037 [-0.0089, +0.0173] | -0.0056 [-0.0214, +0.0107] | +0.0001 [-0.0092, +0.0099] | -0.0013 [-0.0089, +0.0070] | +0.0072 [-0.0086, +0.0239] | +0.0023 [+0.0009, +0.0038] | -0.0065 [-0.0109, -0.0019] | -0.0033 [-0.0128, +0.0061] |
| 1 | 0.05 | +0.0007 [-0.0098, +0.0112] | +0.0011 [-0.0110, +0.0137] | -0.0077 [-0.0218, +0.0073] | -0.0010 [-0.0099, +0.0079] | -0.0018 [-0.0092, +0.0056] | +0.0013 [-0.0132, +0.0165] | +0.0022 [+0.0010, +0.0035] | -0.0061 [-0.0109, -0.0018] | -0.0016 [-0.0112, +0.0077] |
| 2 | 0.00 | -0.0099 [-0.0179, -0.0016] | -0.0102 [-0.0204, +0.0003] | -0.0219 [-0.0363, -0.0075] | -0.0070 [-0.0137, -0.0007] | -0.0088 [-0.0149, -0.0031] | -0.0027 [-0.0179, +0.0125] | +0.0017 [+0.0006, +0.0029] | +0.0026 [-0.0074, +0.0121] | +0.0026 [-0.0074, +0.0121] |
| 3 | 0.05 | -0.0066 [-0.0180, +0.0052] | -0.0073 [-0.0196, +0.0051] | -0.0131 [-0.0283, +0.0041] | -0.0054 [-0.0147, +0.0035] | -0.0044 [-0.0114, +0.0026] | -0.0026 [-0.0172, +0.0143] | +0.0037 [+0.0025, +0.0052] | -0.0102 [-0.0146, -0.0061] | -0.0101 [-0.0190, -0.0016] |
| 4 | 0.05 | +0.0093 [-0.0062, +0.0249] | +0.0084 [-0.0055, +0.0227] | -0.0018 [-0.0173, +0.0141] | +0.0048 [-0.0053, +0.0156] | +0.0065 [-0.0029, +0.0165] | +0.0111 [-0.0049, +0.0272] | +0.0025 [+0.0012, +0.0037] | -0.0050 [-0.0097, -0.0006] | +0.0012 [-0.0080, +0.0106] |

## 3. Processed development — same selection-fold videos, clean-fitted arms (reported, not used)

| arm | jpeg40 v3.1 | jpeg40 FF++ val real FPR@τ | rs06 v3.1 | rs06 FF++ val real FPR@τ |
|---|---:|---:|---:|---:|
| gate | 0.7270 ±0.0097 | 0.000 | 0.7906 ±0.0124 | 0.079 |
| g=1 | 0.7248 ±0.0096 | 0.000 | 0.7889 ±0.0125 | 0.079 |
| const g | 0.7327 ±0.0082 | 0.007 | 0.7916 ±0.0110 | 0.080 |
| logistic (rich) | 0.6970 ±0.0108 | 0.000 | 0.7923 ±0.0133 | 0.084 |
| MLP (rich) | 0.6910 ±0.0113 | 0.000 | 0.7818 ±0.0117 | 0.112 |
| logistic (matched) | 0.6421 ±0.0122 | 0.000 | 0.7886 ±0.0122 | 0.099 |
| MLP (matched) | 0.7157 ±0.0099 | 0.007 | 0.7826 ±0.0131 | 0.081 |
| two-view | 0.7293 ±0.0066 | 0.013 | 0.7864 ±0.0114 | 0.082 |

## 4. Test sets — read after the development analysis, never used to choose (fitted arms: mean over splits)

### 4a. Video AUROC

| test | gate | g=1 | const g | logistic (rich) | MLP (rich) | logistic (matched) | MLP (matched) | two-view |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CDFv2 | 0.9539 | 0.9515 | 0.9650 | 0.9599 | 0.9617 | 0.9647 | 0.9672 | 0.9693 |
| CDFv3 | 0.8881 | 0.8879 | 0.8807 | 0.9227 | 0.9097 | 0.9152 | 0.8931 | 0.8691 |
| DFD | 0.9462 | 0.9459 | 0.9430 | 0.9078 | 0.9174 | 0.9174 | 0.9310 | 0.9343 |
| DFDC | 0.8887 | 0.8869 | 0.8894 | 0.8889 | 0.8837 | 0.8752 | 0.8853 | 0.8818 |
| DFDCP | 0.8986 | 0.8975 | 0.8940 | 0.8667 | 0.8675 | 0.8631 | 0.8820 | 0.8755 |
| DFEval24 | 0.6907 | 0.6902 | 0.6879 | 0.6446 | 0.6403 | 0.6362 | 0.6783 | 0.6769 |
| FFIW | 0.9093 | 0.9062 | 0.9231 | 0.9055 | 0.8975 | 0.9137 | 0.9083 | 0.9284 |
| KoDF | 0.8434 | 0.8421 | 0.8494 | 0.8596 | 0.8551 | 0.8554 | 0.8426 | 0.8509 |
| IDForge | 0.9660 | 0.9656 | 0.9645 | 0.9489 | 0.9571 | 0.9540 | 0.9643 | 0.9598 |
| DFDM | 0.9918 | 0.9911 | 0.9921 | 0.9938 | 0.9919 | 0.9924 | 0.9917 | 0.9909 |
| CDFv3 reenactment | 0.8612 | 0.8621 | 0.8457 | 0.9110 | 0.8901 | 0.8985 | 0.8640 | 0.8271 |
| CDFv3 swap | 0.9376 | 0.9354 | 0.9451 | 0.9443 | 0.9457 | 0.9460 | 0.9468 | 0.9463 |
| KoDF first-order motion | 0.7068 | 0.7031 | 0.7201 | 0.7241 | 0.7216 | 0.6998 | 0.6901 | 0.7206 |
| CDFv2 base | 0.9539 | 0.9515 | 0.9650 | 0.9599 | 0.9617 | 0.9647 | 0.9672 | 0.9693 |
| CDFv2 jpeg40 | 0.8426 | 0.8397 | 0.8573 | 0.7710 | 0.7865 | 0.6784 | 0.8520 | 0.8625 |
| CDFv2 rs06 | 0.9386 | 0.9363 | 0.9478 | 0.9295 | 0.9383 | 0.9470 | 0.9452 | 0.9507 |
| DFDCP base | 0.8986 | 0.8975 | 0.8940 | 0.8667 | 0.8675 | 0.8631 | 0.8820 | 0.8755 |
| DFDCP jpeg40 | 0.8751 | 0.8732 | 0.8611 | 0.7941 | 0.7923 | 0.7482 | 0.8325 | 0.8185 |
| DFDCP rs06 | 0.8344 | 0.8349 | 0.8226 | 0.7972 | 0.8007 | 0.7922 | 0.8037 | 0.7986 |
| **OOD mean (10 sets)** | **0.8977** | **0.8965** | **0.8989** | **0.8898** | **0.8882** | **0.8887** | **0.8944** | **0.8937** |

### 4b. Real-side FPR / TPR at frozen τ

| test | gate | g=1 | const g | logistic (rich) | MLP (rich) | logistic (matched) | MLP (matched) | two-view |
|---|---|---|---|---|---|---|---|---|
| CDFv2 | 0.084 / 0.835 | 0.076 / 0.816 | 0.121 / 0.900 | 0.039 / 0.765 | 0.064 / 0.835 | 0.047 / 0.834 | 0.098 / 0.888 | 0.157 / 0.939 |
| CDFv3 | 0.084 / 0.700 | 0.076 / 0.691 | 0.121 / 0.735 | 0.039 / 0.656 | 0.064 / 0.709 | 0.047 / 0.709 | 0.098 / 0.715 | 0.157 / 0.754 |
| DFD | 0.072 / 0.804 | 0.067 / 0.800 | 0.106 / 0.830 | 0.036 / 0.769 | 0.075 / 0.802 | 0.080 / 0.810 | 0.105 / 0.837 | 0.120 / 0.849 |
| DFDC | 0.227 / 0.836 | 0.222 / 0.832 | 0.260 / 0.850 | 0.164 / 0.789 | 0.215 / 0.813 | 0.219 / 0.822 | 0.239 / 0.833 | 0.287 / 0.849 |
| DFDCP | 0.357 / 0.945 | 0.360 / 0.947 | 0.369 / 0.937 | 0.241 / 0.803 | 0.267 / 0.830 | 0.280 / 0.846 | 0.311 / 0.884 | 0.371 / 0.902 |
| DFEval24 | 0.277 / 0.554 | 0.279 / 0.556 | 0.276 / 0.554 | 0.065 / 0.271 | 0.123 / 0.332 | 0.128 / 0.352 | 0.202 / 0.468 | 0.286 / 0.562 |
| FFIW | 0.071 / 0.710 | 0.070 / 0.706 | 0.076 / 0.745 | 0.014 / 0.580 | 0.029 / 0.641 | 0.027 / 0.664 | 0.049 / 0.697 | 0.082 / 0.764 |
| KoDF | 0.010 / 0.593 | 0.008 / 0.588 | 0.026 / 0.632 | 0.005 / 0.591 | 0.018 / 0.625 | 0.009 / 0.614 | 0.029 / 0.639 | 0.054 / 0.678 |
| IDForge | 0.094 / 0.874 | 0.092 / 0.873 | 0.109 / 0.882 | 0.023 / 0.775 | 0.039 / 0.825 | 0.041 / 0.841 | 0.071 / 0.866 | 0.121 / 0.884 |
| DFDM | 0.074 / 0.978 | 0.065 / 0.977 | 0.108 / 0.983 | 0.037 / 0.960 | 0.054 / 0.967 | 0.047 / 0.975 | 0.086 / 0.977 | 0.146 / 0.987 |
| CDFv3 reenactment | 0.084 / 0.643 | 0.076 / 0.637 | 0.121 / 0.668 | 0.039 / 0.603 | 0.064 / 0.655 | 0.047 / 0.653 | 0.098 / 0.645 | 0.157 / 0.679 |
| CDFv3 swap | 0.084 / 0.803 | 0.076 / 0.789 | 0.121 / 0.858 | 0.039 / 0.755 | 0.064 / 0.809 | 0.047 / 0.812 | 0.098 / 0.844 | 0.157 / 0.890 |
| KoDF first-order motion | 0.010 / 0.087 | 0.008 / 0.081 | 0.026 / 0.187 | 0.005 / 0.086 | 0.018 / 0.172 | 0.009 / 0.143 | 0.029 / 0.207 | 0.054 / 0.314 |
| CDFv2 base | 0.084 / 0.835 | 0.076 / 0.816 | 0.121 / 0.900 | 0.039 / 0.765 | 0.064 / 0.835 | 0.047 / 0.834 | 0.098 / 0.888 | 0.157 / 0.939 |
| CDFv2 jpeg40 | 0.021 / 0.314 | 0.018 / 0.289 | 0.056 / 0.494 | 0.000 / 0.155 | 0.006 / 0.267 | 0.000 / 0.236 | 0.031 / 0.443 | 0.097 / 0.646 |
| CDFv2 rs06 | 0.071 / 0.794 | 0.065 / 0.778 | 0.107 / 0.855 | 0.072 / 0.750 | 0.094 / 0.820 | 0.088 / 0.816 | 0.088 / 0.836 | 0.125 / 0.881 |
| DFDCP base | 0.357 / 0.945 | 0.360 / 0.947 | 0.369 / 0.937 | 0.241 / 0.803 | 0.267 / 0.830 | 0.280 / 0.846 | 0.311 / 0.884 | 0.371 / 0.902 |
| DFDCP jpeg40 | 0.101 / 0.583 | 0.103 / 0.585 | 0.104 / 0.555 | 0.002 / 0.302 | 0.008 / 0.375 | 0.009 / 0.371 | 0.066 / 0.485 | 0.103 / 0.532 |
| DFDCP rs06 | 0.445 / 0.906 | 0.443 / 0.903 | 0.458 / 0.893 | 0.360 / 0.789 | 0.389 / 0.819 | 0.409 / 0.832 | 0.412 / 0.836 | 0.456 / 0.835 |

CDFv2 and DeepfakeEval2024 (DFEval24) have informed development (their validation halves are in the selection data and their test halves appear in earlier tables); no test set here is untouched confirmation data. UADFV and FaceShifter remain uninspected.

## Limitations

- Conditional on the development-selected checkpoint; replay cannot establish the best fusion-TRAINING method (the prior head was trained with the gate present).
- One detector seed per run. MLP initialisations and fold splits are not independent detector seeds.
- DFEval24 and FFIW validation videos are grouped per name (no source linkage recoverable).
- KoDF is mostly face swaps; its first-order-motion subset is the reenactment part and is shown separately.

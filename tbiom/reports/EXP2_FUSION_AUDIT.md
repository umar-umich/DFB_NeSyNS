# Cached fusion audit v2 — run `warmgateB`

**Conditional on the development-selected checkpoint:** `warmgateB_last` (epoch 19, v3.1 score 0.8143, sha256 `35c560adc8f335ec`), chosen earlier by the unchanged v3.1 rule. Replay only: no detector parameter is trained. The audit asks how alternative fusions behave on this checkpoint's branches. Script `analysis/tbiom/exp1_fusion_audit.py` (v2); raw numbers in the matching `.json`.

## Protocol

- Development data: v3.1 selection data (FF++ val, VALmix, VALmix-v2 in-scope subsets), 5814 videos in 2728 source-video groups (largest 25).
- 5 random grouped 50/50 splits into a fitting fold and a selection fold (no group in both). Fusion parameters — constant g, logistic C, MLP size/L2 and weights — are fitted on the CLEAN fitting fold only.
- Arms: learned gate · g = 1 · fitted constant g · logistic and MLP stacking on RICH inputs (per-view p, vacuity, log-evidence, logit; cross-view std/range; pairwise DS conflict) · logistic and MLP stacking on MATCHED inputs (exactly the gate's: p and vacuity of the three views) · two-view (prior removed).
- Comparison on the selection fold with the v3.1 objective. **Paired, source-grouped bootstrap** (1000 draws per split): each draw resamples source groups with replacement (a group's videos move together) and scores every arm on that same draw; a gate-vs-gate self-test returned Δ = 0 on every draw.
- MLP: 5 random initialisations per fit, reported both as the initialisation average and as the median initialisation (the v1-registered comparator). Initialisations are not detector seeds.
- τ per arm and split: EER threshold on the CLEAN selection-fold FF++ val videos, frozen for everything else.
- Processed development (jpeg40, rs06): the same selection-fold videos under JPEG q40 / resize 0.60, scored with the clean-fitted arms. Reported only; never used to fit, select or decide.
- Replay check: DS(Appearance, Off-manifold, gated prior) reproduces the exported fused output to 2.1e-07.

**Decision rule (unchanged from v1).** Registered comparison: gate vs MLP stacking, rich inputs, median initialisation. CANDIDATE CONTRIBUTION if mean Δ ≥ +0.005 and 95% CI lower bound > 0 in ≥ 4/5 splits; STACKING MATCHES OR BEATS if mean Δ < +0.005 and CI upper bound < +0.005 in ≥ 4/5 splits; otherwise UNRESOLVED. The same rule applied to the other comparators is reported as secondary information, not as a new rule.

## Verdicts

| comparison (Δ = gate − comparator) | mean Δ | CI>0 splits | CI upper<+0.005 splits | verdict |
|---|---:|---:|---:|---|
| MLP (rich) median init **(registered)** | -0.0014 | 0/5 | 1/5 | UNRESOLVED |
| MLP (rich) init mean | -0.0025 | 0/5 | 2/5 | UNRESOLVED |
| MLP (matched) init mean | -0.0038 | 0/5 | 2/5 | UNRESOLVED |
| logistic (rich) | -0.0128 | 0/5 | 2/5 | UNRESOLVED |
| logistic (matched) | -0.0014 | 0/5 | 0/5 | UNRESOLVED |

## 1. Clean development — selection fold, v3.1 objective (mean ± sd over 5 splits)

| arm | v3.1 score | swap | reenactment | in-the-wild |
|---|---:|---:|---:|---:|
| gate | 0.8149 ±0.0084 | 0.9065 ±0.0099 | 0.7633 ±0.0187 | 0.7750 ±0.0081 |
| g=1 | 0.8149 ±0.0084 | 0.9065 ±0.0099 | 0.7633 ±0.0187 | 0.7749 ±0.0081 |
| const g | 0.8246 ±0.0073 | 0.9251 ±0.0096 | 0.7647 ±0.0183 | 0.7839 ±0.0074 |
| logistic (rich) | 0.8277 ±0.0096 | 0.9069 ±0.0106 | 0.7879 ±0.0174 | 0.7883 ±0.0137 |
| MLP (rich) | 0.8174 ±0.0084 | 0.9068 ±0.0119 | 0.7753 ±0.0201 | 0.7702 ±0.0144 |
| logistic (matched) | 0.8164 ±0.0087 | 0.8978 ±0.0118 | 0.7794 ±0.0162 | 0.7719 ±0.0114 |
| MLP (matched) | 0.8188 ±0.0083 | 0.9170 ±0.0095 | 0.7607 ±0.0200 | 0.7786 ±0.0126 |
| two-view | 0.8226 ±0.0060 | 0.9286 ±0.0086 | 0.7583 ±0.0179 | 0.7810 ±0.0083 |

MLP rows average the initialisations. Median-initialisation scores per split: rich 0.8026, 0.8259, 0.8272, 0.8092, 0.8167; matched 0.8072, 0.8231, 0.8312, 0.8131, 0.8255.

### Arithmetic reconciliation (v1)

v1's table row "MLP" averaged the five initialisations, while v1's Δ and verdict used the median initialisation; the two differ because the median initialisation is not the average. Both are reported below with their own paired bootstrap, so every Δ equals the difference of two numbers shown in this report.

## 2. Δ = gate − comparator on the clean selection fold, 95% paired group-bootstrap CI

| split | g | MLP (rich) median init | MLP (rich) init mean | logistic (rich) | MLP (matched) median init | MLP (matched) init mean | logistic (matched) | g=1 | const g | two-view |
|---|---:|---|---|---|---|---|---|---|---|---|
| 0 | 0.05 | +0.0016 [-0.0119, +0.0146] | +0.0001 [-0.0129, +0.0133] | -0.0105 [-0.0271, +0.0062] | -0.0030 [-0.0126, +0.0065] | -0.0031 [-0.0117, +0.0054] | +0.0014 [-0.0144, +0.0188] | -0.0000 [-0.0000, +0.0000] | -0.0097 [-0.0164, -0.0033] | -0.0087 [-0.0179, +0.0013] |
| 1 | 0.05 | -0.0032 [-0.0170, +0.0112] | -0.0015 [-0.0145, +0.0120] | -0.0092 [-0.0241, +0.0060] | -0.0004 [-0.0113, +0.0103] | +0.0034 [-0.0067, +0.0131] | -0.0014 [-0.0175, +0.0132] | +0.0000 [-0.0000, +0.0001] | -0.0094 [-0.0161, -0.0028] | -0.0055 [-0.0154, +0.0040] |
| 2 | 0.05 | -0.0093 [-0.0207, +0.0024] | -0.0101 [-0.0202, -0.0005] | -0.0243 [-0.0382, -0.0091] | -0.0133 [-0.0210, -0.0058] | -0.0123 [-0.0201, -0.0051] | -0.0082 [-0.0231, +0.0069] | +0.0000 [-0.0000, +0.0001] | -0.0084 [-0.0148, -0.0021] | -0.0054 [-0.0145, +0.0039] |
| 3 | 0.10 | -0.0034 [-0.0157, +0.0087] | -0.0077 [-0.0209, +0.0049] | -0.0143 [-0.0313, +0.0031] | -0.0073 [-0.0176, +0.0020] | -0.0064 [-0.0156, +0.0022] | -0.0045 [-0.0203, +0.0115] | +0.0000 [-0.0000, +0.0000] | -0.0126 [-0.0176, -0.0077] | -0.0138 [-0.0225, -0.0048] |
| 4 | 0.05 | +0.0073 [-0.0082, +0.0220] | +0.0068 [-0.0067, +0.0201] | -0.0055 [-0.0224, +0.0121] | -0.0014 [-0.0135, +0.0110] | -0.0007 [-0.0087, +0.0076] | +0.0054 [-0.0116, +0.0226] | +0.0000 [-0.0000, +0.0001] | -0.0081 [-0.0145, -0.0017] | -0.0053 [-0.0152, +0.0043] |

## 3. Processed development — same selection-fold videos, clean-fitted arms (reported, not used)

| arm | jpeg40 v3.1 | jpeg40 FF++ val real FPR@τ | rs06 v3.1 | rs06 FF++ val real FPR@τ |
|---|---:|---:|---:|---:|
| gate | 0.7380 ±0.0075 | 0.013 | 0.7849 ±0.0118 | 0.046 |
| g=1 | 0.7381 ±0.0075 | 0.013 | 0.7849 ±0.0118 | 0.046 |
| const g | 0.7354 ±0.0067 | 0.010 | 0.7904 ±0.0116 | 0.074 |
| logistic (rich) | 0.6639 ±0.0073 | 0.003 | 0.7909 ±0.0123 | 0.124 |
| MLP (rich) | 0.6389 ±0.0116 | 0.047 | 0.7821 ±0.0122 | 0.112 |
| logistic (matched) | 0.6423 ±0.0116 | 0.000 | 0.7856 ±0.0122 | 0.088 |
| MLP (matched) | 0.6929 ±0.0141 | 0.053 | 0.7797 ±0.0129 | 0.087 |
| two-view | 0.7293 ±0.0066 | 0.013 | 0.7864 ±0.0114 | 0.082 |

## 4. Test sets — read after the development analysis, never used to choose (fitted arms: mean over splits)

### 4a. Video AUROC

| test | gate | g=1 | const g | logistic (rich) | MLP (rich) | logistic (matched) | MLP (matched) | two-view |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CDFv2 | 0.9521 | 0.9521 | 0.9665 | 0.9640 | 0.9659 | 0.9655 | 0.9681 | 0.9693 |
| CDFv3 | 0.8770 | 0.8770 | 0.8750 | 0.9230 | 0.9074 | 0.9137 | 0.8886 | 0.8691 |
| DFD | 0.9440 | 0.9440 | 0.9414 | 0.9102 | 0.9279 | 0.9179 | 0.9329 | 0.9343 |
| DFDC | 0.8952 | 0.8952 | 0.8918 | 0.8872 | 0.8814 | 0.8772 | 0.8848 | 0.8818 |
| DFDCP | 0.9012 | 0.9012 | 0.8942 | 0.8663 | 0.8662 | 0.8562 | 0.8729 | 0.8755 |
| DFEval24 | 0.6870 | 0.6870 | 0.6864 | 0.6267 | 0.6270 | 0.6259 | 0.6573 | 0.6769 |
| FFIW | 0.9140 | 0.9140 | 0.9284 | 0.9178 | 0.9092 | 0.9139 | 0.9137 | 0.9284 |
| KoDF | 0.8538 | 0.8538 | 0.8542 | 0.8693 | 0.8631 | 0.8581 | 0.8522 | 0.8509 |
| IDForge | 0.9657 | 0.9657 | 0.9664 | 0.9512 | 0.9594 | 0.9547 | 0.9635 | 0.9598 |
| DFDM | 0.9950 | 0.9950 | 0.9932 | 0.9936 | 0.9923 | 0.9941 | 0.9925 | 0.9909 |
| CDFv3 reenactment | 0.8444 | 0.8444 | 0.8364 | 0.9100 | 0.8839 | 0.8951 | 0.8558 | 0.8271 |
| CDFv3 swap | 0.9369 | 0.9368 | 0.9462 | 0.9470 | 0.9505 | 0.9479 | 0.9488 | 0.9463 |
| KoDF first-order motion | 0.7245 | 0.7245 | 0.7273 | 0.7397 | 0.7296 | 0.7073 | 0.7105 | 0.7206 |
| CDFv2 base | 0.9521 | 0.9521 | 0.9665 | 0.9640 | 0.9659 | 0.9655 | 0.9681 | 0.9693 |
| CDFv2 jpeg40 | 0.8617 | 0.8616 | 0.8653 | 0.7254 | 0.7174 | 0.6894 | 0.8270 | 0.8625 |
| CDFv2 rs06 | 0.9375 | 0.9375 | 0.9481 | 0.9364 | 0.9452 | 0.9470 | 0.9469 | 0.9507 |
| DFDCP base | 0.9012 | 0.9012 | 0.8942 | 0.8663 | 0.8662 | 0.8562 | 0.8729 | 0.8755 |
| DFDCP jpeg40 | 0.8941 | 0.8941 | 0.8656 | 0.7352 | 0.7150 | 0.7206 | 0.7532 | 0.8185 |
| DFDCP rs06 | 0.8398 | 0.8398 | 0.8232 | 0.7989 | 0.7985 | 0.7806 | 0.7967 | 0.7986 |
| **OOD mean (10 sets)** | **0.8985** | **0.8985** | **0.8998** | **0.8909** | **0.8900** | **0.8877** | **0.8926** | **0.8937** |

### 4b. Real-side FPR / TPR at frozen τ

| test | gate | g=1 | const g | logistic (rich) | MLP (rich) | logistic (matched) | MLP (matched) | two-view |
|---|---|---|---|---|---|---|---|---|
| CDFv2 | 0.067 / 0.793 | 0.067 / 0.793 | 0.117 / 0.903 | 0.064 / 0.835 | 0.071 / 0.863 | 0.047 / 0.838 | 0.100 / 0.904 | 0.157 / 0.939 |
| CDFv3 | 0.067 / 0.643 | 0.067 / 0.643 | 0.117 / 0.720 | 0.064 / 0.736 | 0.071 / 0.729 | 0.047 / 0.700 | 0.100 / 0.723 | 0.157 / 0.754 |
| DFD | 0.056 / 0.791 | 0.056 / 0.791 | 0.093 / 0.829 | 0.071 / 0.809 | 0.094 / 0.825 | 0.071 / 0.804 | 0.109 / 0.839 | 0.120 / 0.849 |
| DFDC | 0.227 / 0.854 | 0.227 / 0.854 | 0.256 / 0.853 | 0.244 / 0.835 | 0.253 / 0.832 | 0.212 / 0.819 | 0.251 / 0.834 | 0.287 / 0.849 |
| DFDCP | 0.385 / 0.951 | 0.385 / 0.951 | 0.373 / 0.933 | 0.315 / 0.855 | 0.304 / 0.854 | 0.280 / 0.842 | 0.320 / 0.877 | 0.371 / 0.902 |
| DFEval24 | 0.252 / 0.532 | 0.252 / 0.532 | 0.262 / 0.540 | 0.119 / 0.325 | 0.157 / 0.360 | 0.121 / 0.324 | 0.227 / 0.465 | 0.286 / 0.562 |
| FFIW | 0.057 / 0.696 | 0.057 / 0.696 | 0.067 / 0.739 | 0.029 / 0.680 | 0.041 / 0.682 | 0.027 / 0.661 | 0.061 / 0.718 | 0.082 / 0.764 |
| KoDF | 0.013 / 0.602 | 0.013 / 0.602 | 0.031 / 0.647 | 0.011 / 0.622 | 0.021 / 0.632 | 0.010 / 0.612 | 0.034 / 0.648 | 0.054 / 0.678 |
| IDForge | 0.076 / 0.874 | 0.076 / 0.874 | 0.101 / 0.883 | 0.043 / 0.830 | 0.052 / 0.849 | 0.038 / 0.834 | 0.079 / 0.869 | 0.121 / 0.884 |
| DFDM | 0.063 / 0.983 | 0.063 / 0.983 | 0.103 / 0.984 | 0.057 / 0.978 | 0.067 / 0.977 | 0.044 / 0.978 | 0.091 / 0.979 | 0.146 / 0.987 |
| CDFv3 reenactment | 0.067 / 0.579 | 0.067 / 0.578 | 0.117 / 0.645 | 0.064 / 0.693 | 0.071 / 0.673 | 0.047 / 0.639 | 0.100 / 0.651 | 0.157 / 0.679 |
| CDFv3 swap | 0.067 / 0.763 | 0.067 / 0.763 | 0.117 / 0.858 | 0.064 / 0.814 | 0.071 / 0.831 | 0.047 / 0.814 | 0.100 / 0.854 | 0.157 / 0.890 |
| KoDF first-order motion | 0.013 / 0.110 | 0.013 / 0.110 | 0.031 / 0.228 | 0.011 / 0.168 | 0.021 / 0.192 | 0.010 / 0.139 | 0.034 / 0.232 | 0.054 / 0.314 |
| CDFv2 base | 0.067 / 0.793 | 0.067 / 0.793 | 0.117 / 0.903 | 0.064 / 0.835 | 0.071 / 0.863 | 0.047 / 0.838 | 0.100 / 0.904 | 0.157 / 0.939 |
| CDFv2 jpeg40 | 0.144 / 0.656 | 0.144 / 0.656 | 0.101 / 0.638 | 0.006 / 0.210 | 0.038 / 0.320 | 0.004 / 0.284 | 0.102 / 0.547 | 0.097 / 0.646 |
| CDFv2 rs06 | 0.048 / 0.652 | 0.048 / 0.652 | 0.081 / 0.831 | 0.100 / 0.818 | 0.103 / 0.839 | 0.075 / 0.805 | 0.098 / 0.839 | 0.125 / 0.881 |
| DFDCP base | 0.385 / 0.951 | 0.385 / 0.951 | 0.373 / 0.933 | 0.315 / 0.855 | 0.304 / 0.854 | 0.280 / 0.842 | 0.320 / 0.877 | 0.371 / 0.902 |
| DFDCP jpeg40 | 0.216 / 0.834 | 0.216 / 0.834 | 0.119 / 0.601 | 0.003 / 0.367 | 0.038 / 0.419 | 0.011 / 0.373 | 0.097 / 0.507 | 0.103 / 0.532 |
| DFDCP rs06 | 0.419 / 0.895 | 0.419 / 0.895 | 0.427 / 0.875 | 0.427 / 0.848 | 0.423 / 0.832 | 0.397 / 0.811 | 0.416 / 0.826 | 0.456 / 0.835 |

CDFv2 and DeepfakeEval2024 (DFEval24) have informed development (their validation halves are in the selection data and their test halves appear in earlier tables); no test set here is untouched confirmation data. UADFV and FaceShifter remain uninspected.

## Limitations

- Conditional on the development-selected checkpoint; replay cannot establish the best fusion-TRAINING method (the prior head was trained with the gate present).
- One detector seed per run. MLP initialisations and fold splits are not independent detector seeds.
- DFEval24 and FFIW validation videos are grouped per name (no source linkage recoverable).
- KoDF is mostly face swaps; its first-order-motion subset is the reenactment part and is shown separately.

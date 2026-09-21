# Stage 1B — matched augmentation extension (composed resize → JPEG)

Pre-registration: `tbiom/docs/stage1b-augmentation-preregistration.md`. Manifest: `tbiom/reports/REFERENCE_MANIFEST.md`. Arm B = `sbi_armB_seed42` at its development-selected epoch 8; arm E = `sbi_armE_seed42` (config identical except `training_arm: E`) at its development-selected checkpoint `sbiarmE_e02`. Both spatial detectors are fused with the frozen reference temporal branch; DS-R and the logistic comparator are refitted per arm on the same source-grouped folds. This tests whether composition helps beyond individual transforms; it does not test the ceiling of FF++ generalisation.

## 1. Development (selection folds, mean of 5 splits)

| arm | rule | clean v3.1 | swap | reenactment | in-the-wild | JPEG q40 | resize 0.60 | FPR own τ | TPR own τ | FPR drop-in τ | TPR drop-in τ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B (existing mixture) | DS-R | 0.8668 | 0.9101 | 0.9092 | 0.7809 | 0.8108 | 0.8491 | 0.067 | 0.651 | 0.067 | 0.651 |
| B (existing mixture) | LR | 0.8625 | 0.9101 | 0.8893 | 0.7880 | 0.8046 | 0.8444 | 0.059 | 0.598 | 0.059 | 0.598 |
| E (adds resize→JPEG) | DS-R | 0.8757 | 0.9232 | 0.9000 | 0.8040 | 0.8223 | 0.8539 | 0.056 | 0.607 | 0.110 | 0.730 |
| E (adds resize→JPEG) | LR | 0.8690 | 0.9173 | 0.8877 | 0.8022 | 0.8089 | 0.8449 | 0.042 | 0.553 | 0.064 | 0.592 |

## 2. Verdict against the frozen criteria (Δ = E − B, paired source-group bootstrap)

**DS-R** — processed Δ +0.0081 (CI > 0 in 5/5; need ≥ +0.005 and ≥ 3/5); clean Δ +0.0090 (need ≥ −0.005); reenactment Δ -0.0093 (CI upper < 0 in 1/5; ≥ 3/5 rejects); clean FPR Δ -0.011 at own τ (need ≤ +0.02) → **ACCEPT**.

**LR** — processed Δ +0.0023 (CI > 0 in 0/5; need ≥ +0.005 and ≥ 3/5); clean Δ +0.0066 (need ≥ −0.005); reenactment Δ -0.0018 (CI upper < 0 in 0/5; ≥ 3/5 rejects); clean FPR Δ -0.017 at own τ (need ≤ +0.02) → **NOT ACCEPTED**.

## 3. Test sets — video AUROC, read after the verdict

| test | B (existing mixture) DS-R | B (existing mixture) LR | E (adds resize→JPEG) DS-R | E (adds resize→JPEG) LR |
|---|---:|---:|---:|---:|
| CDFv2 | 0.9168 | 0.9244 | 0.9343 | 0.9294 |
| CDFv3 | 0.9546 | 0.9470 | 0.9557 | 0.9493 |
| DFD | 0.9144 | 0.9203 | 0.9341 | 0.9203 |
| DFDC | 0.8430 | 0.8593 | 0.8583 | 0.8502 |
| DFDCP | 0.8692 | 0.8815 | 0.8709 | 0.8644 |
| DFEval24 | 0.7252 | 0.7341 | 0.7280 | 0.7054 |
| FFIW | 0.8957 | 0.9018 | 0.9334 | 0.9301 |
| KoDF | 0.9625 | 0.9556 | 0.9544 | 0.9475 |
| IDForge | 0.9580 | 0.9493 | 0.9728 | 0.9586 |
| DFDM | 0.9952 | 0.9953 | 0.9963 | 0.9941 |
| CDFv3 reenactment | 0.9682 | 0.9563 | 0.9642 | 0.9569 |
| CDFv2 jpeg40 | 0.8692 | 0.8838 | 0.8727 | 0.8637 |
| CDFv2 rs06 | 0.8898 | 0.8974 | 0.9023 | 0.9008 |
| DFDCP jpeg40 | 0.8542 | 0.8639 | 0.8706 | 0.8492 |
| DFDCP rs06 | 0.8587 | 0.8724 | 0.8561 | 0.8496 |

No test set, CDFv3 included, entered the verdict. One seed per arm.

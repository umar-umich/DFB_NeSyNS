# Full results — every configuration on every test dataset

## What the numbers are

**AUROC** — frames averaged within a video, then ROC-AUC over videos. Threshold-free: it measures ranking only. Higher is better.

**FPR_real** — of the REAL videos, the fraction scored at or above `τ`; i.e. **false alarms**. Lower is better. `τ` is frozen **once per model** at its own EER on FF++ val and applied unchanged to every test set, so this is a fixed operating point, never re-tuned per dataset.

Six OOD sets, never seen in training or selection. Two references reported but excluded from every mean: **FF++ val** (in-domain, where `τ` is frozen) and **VALmix** (the development split used for checkpoint selection).

Multi-seed rows are the **mean over 3 seeds** (42 / 1337 / 7); the range is in the seed-spread table at the end.

## AUROC  (higher is better)

| configuration | seeds | Celeb-DF-v2 | Celeb-DF-v3 | DFD | DFDC | DFDCP | Deepfake-Eval-2024 | **OOD mean** | VALmix *(dev)* | FF++ val *(in-domain)* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P0-DS  (2 views, baseline) | 1 | 0.9646 | 0.8409 | 0.9421 | 0.8828 | 0.8573 | 0.6922 | **0.8633** | 0.8852 | 0.9961 |
| A  fused-only, 3 views | 1 | 0.8638 | 0.8626 | 0.8812 | 0.8559 | 0.8381 | 0.6789 | **0.8301** | 0.8276 | 0.9920 |
| B  student Br3, 3 views | 3 | 0.9505 | 0.9008 | 0.9486 | 0.8816 | 0.9002 | 0.6875 | **0.8782** | 0.8875 | 0.9959 |
| B  student Br3, 2 views | 3 | 0.9378 | 0.8969 | 0.9408 | 0.8772 | 0.8907 | 0.6903 | **0.8723** | 0.8783 | 0.9954 |
| C  FSFM Br3, 3 views | 3 | 0.9643 | 0.8996 | 0.9426 | 0.8831 | 0.8979 | 0.6824 | **0.8783** | 0.8940 | 0.9956 |
| C  FSFM Br3, 2 views | 3 | 0.9685 | 0.9092 | 0.9437 | 0.8892 | 0.9040 | 0.6877 | **0.8837** | 0.8992 | 0.9942 |
| D  simple CE, 3 views | 1 | 0.9376 | 0.8648 | 0.9449 | 0.8791 | 0.8724 | 0.6850 | **0.8639** | 0.8701 | 0.9945 |
| E  trained 2-view | 1 | 0.9583 | 0.8910 | 0.9385 | 0.8919 | 0.8909 | 0.6925 | **0.8772** | 0.8888 | 0.9949 |
|   branch: semantic (CLIP) | 3 | 0.9437 | 0.8692 | 0.9267 | 0.8588 | 0.8715 | 0.6614 | **0.8552** | 0.8738 | 0.9959 |
|   branch: artifact (β-VAE) | 3 | 0.9488 | 0.8765 | 0.9256 | 0.8583 | 0.8732 | 0.6570 | **0.8566** | 0.8741 | 0.9960 |
|   branch: FS-VFM | 3 | 0.9352 | 0.8926 | 0.8775 | 0.8546 | 0.8781 | 0.6856 | **0.8539** | 0.8669 | 0.9778 |

## FPR_real @ frozen τ  (lower is better)

| configuration | seeds | Celeb-DF-v2 | Celeb-DF-v3 | DFD | DFDC | DFDCP | Deepfake-Eval-2024 | **OOD mean** | VALmix *(dev)* | FF++ val *(in-domain)* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P0-DS  (2 views, baseline) | 1 | 0.1292 | 0.1292 | 0.1736 | 0.3175 | 0.3913 | 0.3411 | **0.2470** | 0.2133 | 0.0214 |
| A  fused-only, 3 views | 1 | 0.0449 | 0.0449 | 0.0826 | 0.1844 | 0.3565 | 0.2967 | **0.1684** | 0.1985 | 0.0500 |
| B  student Br3, 3 views | 3 | 0.0543 | 0.0543 | 0.0248 | 0.1878 | 0.2870 | 0.2017 | **0.1350** | 0.1358 | 0.0262 |
| B  student Br3, 2 views | 3 | 0.0337 | 0.0337 | 0.0184 | 0.1872 | 0.3101 | 0.2079 | **0.1318** | 0.1442 | 0.0238 |
| C  FSFM Br3, 3 views | 3 | 0.0899 | 0.0899 | 0.0826 | 0.2190 | 0.2594 | 0.3115 | **0.1754** | 0.1738 | 0.0286 |
| C  FSFM Br3, 2 views | 3 | 0.0655 | 0.0655 | 0.0845 | 0.2050 | 0.2478 | 0.3372 | **0.1676** | 0.1783 | 0.0286 |
| D  simple CE, 3 views | 1 | 0.1966 | 0.1966 | 0.3691 | 0.3201 | 0.3696 | 0.4416 | **0.3156** | 0.2978 | 0.0286 |
| E  trained 2-view | 1 | 0.1124 | 0.1124 | 0.0634 | 0.2022 | 0.2739 | 0.3107 | **0.1792** | 0.1881 | 0.0286 |
|   branch: semantic (CLIP) | 3 | 0.1667 | 0.1667 | 0.1185 | 0.2888 | 0.3594 | 0.3107 | **0.2351** | 0.2138 | 0.0286 |
|   branch: artifact (β-VAE) | 3 | 0.1348 | 0.1348 | 0.0927 | 0.2883 | 0.3420 | 0.2453 | **0.2063** | 0.1807 | 0.0262 |
|   branch: FS-VFM | 3 | 0.1030 | 0.1030 | 0.1846 | 0.2118 | 0.2957 | 0.5436 | **0.2403** | 0.2844 | 0.0667 |

## Seed spread (max − min of the OOD mean, 3 seeds)

| configuration | range |
|---|---:|
| B  student Br3, 3 views | 0.0025 |
| B  student Br3, 2 views | 0.0007 |
| C  FSFM Br3, 3 views | 0.0166 |
| C  FSFM Br3, 2 views | 0.0111 |
|   branch: semantic (CLIP) | 0.0222 |
|   branch: artifact (β-VAE) | 0.0089 |
|   branch: FS-VFM | 0.0008 |

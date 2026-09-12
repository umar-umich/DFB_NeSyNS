# Stage 2 — reliability against processing

Both classes receive identical processing, so nothing here can learn "compressed means fake". Severities are **ordinal within a dataset**: processing is additive on top of each dataset's existing encoding, not a clean intervention.

## Fit — dev, both classes

**FFpp_val** — AUROC by cue and severity

| cue | none | jpeg q70 | jpeg q50 | jpeg q30 | resize 0.75 | resize 0.50 | resize 0.35 |
|---|---|---|---|---|---|---|---|
| Appearance | 0.9951 | 0.9674 | 0.9377 | 0.8831 | 0.9925 | 0.9859 | 0.9612 |
| Off-manifold residual | 0.9949 | 0.9680 | 0.9439 | 0.8929 | 0.9923 | 0.9858 | 0.9607 |
| Bona-fide face prior | 0.9917 | 0.9678 | 0.9520 | 0.9414 | 0.9888 | 0.9841 | 0.9755 |
| **Fused** | 0.9955 | 0.9745 | 0.9539 | 0.9332 | 0.9936 | 0.9888 | 0.9772 |

FPR / TPR at a threshold **frozen on baseline dev**, never refitted

| cue | none | jpeg q70 | jpeg q50 | jpeg q30 | resize 0.75 | resize 0.50 | resize 0.35 |
|---|---|---|---|---|---|---|---|
| Appearance | 0.029 / 0.975 | 0.043 / 0.886 | 0.064 / 0.838 | 0.050 / 0.695 | 0.043 / 0.979 | 0.329 / 0.988 | 0.579 / 0.989 |
| Off-manifold residual | 0.029 / 0.975 | 0.029 / 0.879 | 0.036 / 0.818 | 0.014 / 0.637 | 0.050 / 0.979 | 0.336 / 0.988 | 0.579 / 0.989 |
| Bona-fide face prior | 0.043 / 0.955 | 0.014 / 0.800 | 0.021 / 0.727 | 0.021 / 0.670 | 0.057 / 0.955 | 0.121 / 0.975 | 0.436 / 0.988 |
| **Fused** | 0.029 / 0.973 | 0.014 / 0.875 | 0.021 / 0.798 | 0.014 / 0.682 | 0.043 / 0.973 | 0.229 / 0.988 | 0.550 / 0.993 |

Value drift (standardised, vs this dataset's own baseline)

| cue | measurement | jpeg q70 | jpeg q50 | jpeg q30 | resize 0.75 | resize 0.50 | resize 0.35 |
|---|---|---|---|---|---|---|---|
| Appearance | `norm_semantic` | -0.49 | -0.71 | -1.40 | +0.06 | +0.32 | +0.48 |
| Off-manifold residual | `recon_err` | -0.62 | -0.94 | -1.70 | +0.09 | +0.43 | +0.65 |

**VALmix** — AUROC by cue and severity

| cue | none | jpeg q70 | jpeg q50 | jpeg q30 | resize 0.75 | resize 0.50 | resize 0.35 |
|---|---|---|---|---|---|---|---|
| Appearance | 0.8722 | 0.8415 | 0.7944 | 0.7435 | 0.8721 | 0.8525 | 0.8073 |
| Off-manifold residual | 0.8752 | 0.8708 | 0.8470 | 0.8046 | 0.8718 | 0.8521 | 0.8060 |
| Bona-fide face prior | 0.8304 | 0.7888 | 0.7619 | 0.7353 | 0.8081 | 0.7991 | 0.7957 |
| **Fused** | 0.8823 | 0.8607 | 0.8285 | 0.7949 | 0.8718 | 0.8578 | 0.8288 |

FPR / TPR at a threshold **frozen on baseline dev**, never refitted

| cue | none | jpeg q70 | jpeg q50 | jpeg q30 | resize 0.75 | resize 0.50 | resize 0.35 |
|---|---|---|---|---|---|---|---|
| Appearance | 0.204 / 0.796 | 0.273 / 0.787 | 0.406 / 0.809 | 0.302 / 0.661 | 0.148 / 0.736 | 0.299 / 0.827 | 0.550 / 0.889 |
| Off-manifold residual | 0.210 / 0.788 | 0.104 / 0.711 | 0.116 / 0.650 | 0.044 / 0.381 | 0.167 / 0.759 | 0.364 / 0.847 | 0.603 / 0.913 |
| Bona-fide face prior | 0.249 / 0.748 | 0.160 / 0.541 | 0.142 / 0.464 | 0.124 / 0.404 | 0.247 / 0.711 | 0.317 / 0.787 | 0.597 / 0.908 |
| **Fused** | 0.200 / 0.800 | 0.127 / 0.677 | 0.136 / 0.603 | 0.089 / 0.452 | 0.167 / 0.739 | 0.290 / 0.830 | 0.584 / 0.927 |

Value drift (standardised, vs this dataset's own baseline)

| cue | measurement | jpeg q70 | jpeg q50 | jpeg q30 | resize 0.75 | resize 0.50 | resize 0.35 |
|---|---|---|---|---|---|---|---|
| Appearance | `norm_semantic` | -0.53 | -0.68 | -1.30 | +0.20 | +0.63 | +0.89 |
| Off-manifold residual | `recon_err` | -0.61 | -0.79 | -1.32 | +0.27 | +0.75 | +1.06 |

## Held-out severities

**CDFv2** — AUROC by cue and severity

| cue | none | jpeg q40 | resize 0.60 |
|---|---|---|---|
| Appearance | 0.9506 | 0.7451 | 0.9360 |
| Off-manifold residual | 0.9607 | 0.8611 | 0.9396 |
| Bona-fide face prior | 0.8666 | 0.7519 | 0.8471 |
| **Fused** | 0.9523 | 0.8404 | 0.9351 |

FPR / TPR at a threshold **frozen on baseline dev**, never refitted

| cue | none | jpeg q40 | resize 0.60 |
|---|---|---|---|
| Appearance | 0.135 / 0.868 | 0.298 / 0.615 | 0.101 / 0.821 |
| Off-manifold residual | 0.118 / 0.879 | 0.039 / 0.400 | 0.112 / 0.835 |
| Bona-fide face prior | 0.219 / 0.794 | 0.056 / 0.321 | 0.309 / 0.806 |
| **Fused** | 0.129 / 0.882 | 0.034 / 0.403 | 0.135 / 0.841 |

Value drift (standardised, vs this dataset's own baseline)

| cue | measurement | jpeg q40 | resize 0.60 |
|---|---|---|---|
| Appearance | `norm_semantic` | -1.20 | +0.31 |
| Off-manifold residual | `recon_err` | -1.33 | +0.41 |

**DFDCP** — AUROC by cue and severity

| cue | none | jpeg q40 | resize 0.60 |
|---|---|---|---|
| Appearance | 0.8892 | 0.7995 | 0.8018 |
| Off-manifold residual | 0.8898 | 0.8070 | 0.8012 |
| Bona-fide face prior | 0.8368 | 0.8120 | 0.7916 |
| **Fused** | 0.8991 | 0.8631 | 0.8344 |

FPR / TPR at a threshold **frozen on baseline dev**, never refitted

| cue | none | jpeg q40 | resize 0.60 |
|---|---|---|---|
| Appearance | 0.191 / 0.807 | 0.091 / 0.502 | 0.291 / 0.726 |
| Off-manifold residual | 0.187 / 0.811 | 0.013 / 0.396 | 0.326 / 0.755 |
| Bona-fide face prior | 0.257 / 0.743 | 0.087 / 0.436 | 0.348 / 0.748 |
| **Fused** | 0.196 / 0.807 | 0.052 / 0.465 | 0.287 / 0.790 |

Value drift (standardised, vs this dataset's own baseline)

| cue | measurement | jpeg q40 | resize 0.60 |
|---|---|---|---|
| Appearance | `norm_semantic` | -1.67 | +0.28 |
| Off-manifold residual | `recon_err` | -1.61 | +0.38 |

## Reading the curves

| cue | worst AUROC drop under processing | at |
|---|---:|---|
| Appearance | **-0.1287** | jpeg q30 |
| Off-manifold residual | **-0.1020** | jpeg q30 |
| Bona-fide face prior | **-0.0951** | jpeg q30 |
| Fused | **-0.0874** | jpeg q30 |

A cue whose AUROC survives a severity is reliable there whatever its value did; a cue whose AUROC collapses is not, and the fusion has no way to know that without this table. That is precisely the reliability field Stage 1 left null, and it is what Stage 3's M2 and M3 are given that M1 has to infer for itself.

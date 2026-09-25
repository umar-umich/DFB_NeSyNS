# Task 8 — refit control

Development pool: 5814 videos, of which **2991** share a Celeb-DF source video or identity with an evaluation set and are removed, leaving **2823**. Trust weights and the threshold are refitted on the clean subset with the same grid, folds, objective and threshold rule; the evaluation sets and the cached opinions are untouched.

| dataset | AUROC orig | AUROC refit | Δ | FPR orig | FPR refit | Δ | TPR orig | TPR refit | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CDFv2 | 0.9583 | 0.9578 | -0.0005 | 0.040 | 0.000 | -0.040 | 0.685 | 0.007 | -0.678 |
| CDFv3 | 0.9639 | 0.9639 | -0.0000 | 0.040 | 0.000 | -0.040 | 0.796 | 0.031 | -0.765 |
| DFD | 0.9307 | 0.9293 | -0.0014 | 0.123 | 0.000 | -0.123 | 0.798 | 0.055 | -0.743 |
| DFDC | 0.8530 | 0.8511 | -0.0019 | 0.191 | 0.001 | -0.190 | 0.742 | 0.069 | -0.674 |
| DFDCP | 0.8597 | 0.8588 | -0.0009 | 0.179 | 0.000 | -0.179 | 0.719 | 0.029 | -0.691 |
| DFEval24 | 0.7235 | 0.7226 | -0.0009 | 0.191 | 0.000 | -0.191 | 0.482 | 0.002 | -0.480 |
| FFIW | 0.9217 | 0.9198 | -0.0020 | 0.115 | 0.000 | -0.115 | 0.753 | 0.026 | -0.727 |
| KoDF | 0.9600 | 0.9601 | +0.0001 | 0.026 | 0.000 | -0.026 | 0.880 | 0.202 | -0.678 |
| IDForge | 0.9631 | 0.9620 | -0.0010 | 0.067 | 0.000 | -0.067 | 0.821 | 0.048 | -0.773 |
| DFDM | 0.9954 | 0.9953 | -0.0002 | 0.037 | 0.000 | -0.037 | 0.971 | 0.172 | -0.799 |

Mean AUROC change **-0.0009**, largest single move **-0.0020** (FFIW).


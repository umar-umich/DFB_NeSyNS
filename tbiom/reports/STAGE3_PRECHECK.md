# Stage 3 pre-check — arm E spatial views against the deployed spatial views

OMP_NUM_THREADS=8 python analysis/tbiom/stage3_precheck.py

Stage 1B compared arm E with arm B, both from the sbi_arm family. The deployed reference takes its Appearance and
Off-manifold views from the warm+gate detector (epoch 19). Same frozen temporal branch, same 5 source-grouped folds,
DS-R and the logistic comparator refitted per arm, paired source-group bootstrap. Development only.

| arm | rule | clean | swap | reenactment | in-the-wild | JPEG q40 | resize 0.60 | FPR own τ | TPR own τ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deployed (warm+gate e19) | DS-R | 0.8808 | 0.9345 | 0.9085 | 0.7993 | 0.8096 | 0.8589 | 0.107 | 0.737 |
| deployed (warm+gate e19) | LR | 0.8736 | 0.9245 | 0.8997 | 0.7965 | 0.7564 | 0.8542 | 0.057 | 0.648 |
| arm E (sbiarmE_e02) | DS-R | 0.8757 | 0.9232 | 0.9000 | 0.8040 | 0.8223 | 0.8539 | 0.056 | 0.607 |
| arm E (sbiarmE_e02) | LR | 0.8690 | 0.9173 | 0.8877 | 0.8022 | 0.8089 | 0.8449 | 0.042 | 0.553 |

| rule | quantity | mean Δ (E − deployed) | splits CI above 0 | splits CI below 0 |
|---|---|---:|---:|---:|
| DS-R | clean | -0.0051 | 0/5 | 0/5 |
| DS-R | processed | +0.0037 | 0/5 | 0/5 |
| DS-R | reenactment | -0.0087 | 0/5 | 2/5 |
| LR | clean | -0.0046 | 0/5 | 0/5 |
| LR | processed | +0.0215 | 5/5 | 0/5 |
| LR | reenactment | -0.0121 | 0/5 | 1/5 |

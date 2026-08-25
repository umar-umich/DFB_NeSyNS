# V1_RELIABILITY — V/C/A, risk and selective prediction (§17, §18, §27)

Calibrated on ['FaceForensics++:val', 'Celeb-DF-v2:val'] (25532 frames, epoch 7).

Policy provenance: `decision = EER on FaceForensics++:val + Celeb-DF-v2:val [NOT zero-shot for: Celeb-DF-v2:val] (out-of-fold q, epoch 7); defer = 10% abstention budget on FaceForensics++:val + Celeb-DF-v2:val [NOT zero-shot for: Celeb-DF-v2:val] (out-of-fold q, epoch 7); both frozen for all OOD sources`

## The risk model (§18)

Logistic regression on `[V, C, A, fused_margin]`, fit on the out-of-fold `q` from §12's cross-fitting. Five parameters, so the coefficients are readable and a good risk-coverage curve remains evidence about the reliability decomposition rather than about model capacity.

| feature | coefficient | reading |
|---|---:|---|
| `V` | +2.0456 | more fused vacuity → more risk ✓ |
| `C` | +3.6138 | more informative conflict → more risk ✓ |
| `A` | +0.9517 | less specialist support → more risk ✓ |
| `fused_margin` | -3.0717 | a more decided fusion → less risk ✓ |
| bias | -1.8385 | |

Error rate at full coverage: **0.1524**. Error-detection AUROC: **0.8118**.

## Selective prediction

At the 10% abstention budget: selective risk 0.1232 against 0.1524 at full coverage (reduction +0.0291).

| coverage | selective risk |
|---:|---:|
| 1.00 | 0.1524 |
| 0.90 | 0.1253 |
| 0.81 | 0.0896 |
| 0.71 | 0.0624 |
| 0.48 | 0.0310 |

## Coverage per source under the FROZEN policy

Coverage is expected to vary: the thresholds were frozen on the calibration sources, so a harder source defers more. Coverage pinned at the budget everywhere would be the signature of a threshold retuned per source.

| source | video coverage | selective accuracy | full-coverage accuracy |
|---|---:|---:|---:|
| FaceForensics++ | 0.943 | 0.9697 | TODO(run) |
| Celeb-DF-v2 | 0.942 | 0.8258 | TODO(run) |
| Celeb-DF-v3 | 0.956 | 0.7859 | TODO(run) |
| DFDC | 0.816 | 0.7981 | TODO(run) |
| DFDCP | 0.794 | 0.8764 | TODO(run) |
| UADFV | 0.929 | 0.9780 | TODO(run) |
| Deepfake-Eval-2024 | 0.910 | 0.6194 | TODO(run) |

## Gate behaviour (§13)

| gate | out-of-fold AUROC | accuracy | always-admit baseline | mean `q` |
|---|---:|---:|---:|---:|
| `q_ref` | 0.9082 | 0.8507 | 0.7214 | 0.722 |
| `q_proc` | 0.9224 | 0.8530 | 0.7069 | 0.708 |

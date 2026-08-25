# V1 contribution diagnostics (§20)

From `['logs/v1/eval/epoch_007_gated_diverse/per_sample_epoch_7.parquet']` — 387913 frames, 7 sources, epoch ?.

> **These are contribution diagnostics, not ablations.** Every configuration reuses heads that were trained with all three branches present, so a branch removed here was still present while the others learned. A small "+reference" number means *the reference adds little on top of heads trained alongside it*, not *a two-branch system would score this*. Publication-quality ablations require retraining (§20).

> ⚠️ **Not zero-shot:** Celeb-DF-v2:val — these sources' test videos were used to calibrate the gates and the defer policy. Label those rows calibrated, not OOD.

## Video AUROC by configuration

| source | `sem_only` | `ref_only` | `proc_only` | `sem+ref` | `sem+proc` | `full_plain_ds` | `full_applicability` | `direct_probe_only` |
|---|---|---|---|---|---|---|---|---|
| Celeb-DF-v2 ⚠️ | 0.9225 | 0.7448 | 0.6695 | 0.9259 | 0.9219 | 0.9246 | 0.9248 | 0.6725 |
| Celeb-DF-v3 | 0.9271 | 0.7439 | 0.6149 | 0.9229 | 0.9196 | 0.9144 | 0.9216 | 0.7063 |
| DFDC | 0.8477 | 0.7448 | 0.5959 | 0.8499 | 0.8469 | 0.8486 | 0.8468 | 0.7198 |
| DFDCP | 0.8913 | 0.7111 | 0.7113 | 0.9030 | 0.8958 | 0.9026 | 0.8970 | 0.6567 |
| Deepfake-Eval-2024 | 0.6357 | 0.5697 | 0.4914 | 0.6283 | 0.6337 | 0.6281 | 0.6298 | 0.5639 |
| FaceForensics++ | 0.9983 | 0.8963 | 0.6322 | 0.9973 | 0.9971 | 0.9960 | 0.9972 | 0.8880 |
| UADFV | 0.9958 | 0.8467 | 0.6426 | 0.9975 | 0.9971 | 0.9975 | 0.9971 | 0.9055 |

## The comparisons that decide things

### applicability_vs_plain_ds

*does the applicability layer beat plain DS? (§24)*

| source | delta | verdict |
|---|---:|---|
| Celeb-DF-v2 | +0.0001 | inside §22's noise floor — needs a second seed |
| Celeb-DF-v3 | +0.0072 | inside §22's noise floor — needs a second seed |
| DFDC | -0.0017 | inside §22's noise floor — needs a second seed |
| DFDCP | -0.0056 | inside §22's noise floor — needs a second seed |
| Deepfake-Eval-2024 | +0.0017 | inside §22's noise floor — needs a second seed |
| FaceForensics++ | +0.0012 | inside §22's noise floor — needs a second seed |
| UADFV | -0.0004 | inside §22's noise floor — needs a second seed |

7 of 7 sources are inside §22's noise floor — a second seed is required before any promotion or removal.

### full_vs_sem_only

*do both specialists together beat the anchor?*

| source | delta | verdict |
|---|---:|---|
| Celeb-DF-v2 | +0.0022 | inside §22's noise floor — needs a second seed |
| Celeb-DF-v3 | -0.0127 | sem_only better |
| DFDC | +0.0009 | inside §22's noise floor — needs a second seed |
| DFDCP | +0.0113 | full_plain_ds better |
| Deepfake-Eval-2024 | -0.0076 | inside §22's noise floor — needs a second seed |
| FaceForensics++ | -0.0023 | inside §22's noise floor — needs a second seed |
| UADFV | +0.0017 | inside §22's noise floor — needs a second seed |

5 of 7 sources are inside §22's noise floor — a second seed is required before any promotion or removal.

### process_contribution

*does the process branch add over the anchor alone?*

| source | delta | verdict |
|---|---:|---|
| Celeb-DF-v2 | -0.0006 | inside §22's noise floor — needs a second seed |
| Celeb-DF-v3 | -0.0075 | inside §22's noise floor — needs a second seed |
| DFDC | -0.0008 | inside §22's noise floor — needs a second seed |
| DFDCP | +0.0045 | inside §22's noise floor — needs a second seed |
| Deepfake-Eval-2024 | -0.0020 | inside §22's noise floor — needs a second seed |
| FaceForensics++ | -0.0012 | inside §22's noise floor — needs a second seed |
| UADFV | +0.0012 | inside §22's noise floor — needs a second seed |

7 of 7 sources are inside §22's noise floor — a second seed is required before any promotion or removal.

### reference_contribution

*does the reference add over the anchor alone?*

| source | delta | verdict |
|---|---:|---|
| Celeb-DF-v2 | +0.0035 | inside §22's noise floor — needs a second seed |
| Celeb-DF-v3 | -0.0042 | inside §22's noise floor — needs a second seed |
| DFDC | +0.0023 | inside §22's noise floor — needs a second seed |
| DFDCP | +0.0118 | sem+ref better |
| Deepfake-Eval-2024 | -0.0074 | inside §22's noise floor — needs a second seed |
| FaceForensics++ | -0.0010 | inside §22's noise floor — needs a second seed |
| UADFV | +0.0017 | inside §22's noise floor — needs a second seed |

6 of 7 sources are inside §22's noise floor — a second seed is required before any promotion or removal.

### reference_vs_direct_probe

*does P_R beat the capacity-matched direct probe? (§4.2)*

| source | delta | verdict |
|---|---:|---|
| Celeb-DF-v2 | +0.0723 | ref_only better |
| Celeb-DF-v3 | +0.0376 | ref_only better |
| DFDC | +0.0250 | ref_only better |
| DFDCP | +0.0543 | ref_only better |
| Deepfake-Eval-2024 | +0.0058 | inside §22's noise floor — needs a second seed |
| FaceForensics++ | +0.0083 | inside §22's noise floor — needs a second seed |
| UADFV | -0.0587 | direct_probe_only better |

2 of 7 sources are inside §22's noise floor — a second seed is required before any promotion or removal.

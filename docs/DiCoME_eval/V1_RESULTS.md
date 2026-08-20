# V1_RESULTS — DISCERN v2 V1 (§21, §27)

Checkpoint `logs/v1/stage_b_seed42/epoch_007.pth` (epoch 7). Video aggregation: mean frame p(fake) (§21).

> `fused_ungated` is plain DS over three ungated experts (q=1) — the baseline the applicability layer must beat, NOT the V1 system. `fused_gated` (present only with --stage-de) is the V1 system: frozen Stage-D gates and the frozen Stage-E defer policy, neither retuned on any evaluation source.

> ⚠️ **Not zero-shot:** Celeb-DF-v2:val. Those sources' test videos were used to calibrate the gates and the defer policy, a recorded deviation from §1/§11/§18. Label those rows calibrated, not OOD.

## Headline — video AUROC

| source | DISCERN-v2 V1 (gated) | plain DS (ungated) | anchor `sem` only | our reproduced DiCoME | DiCoME paper | DISCERN-v1 |
|---|---:|---:|---:|---:|---:|---:|
| FaceForensics++ | 0.9972 | 0.9960 | 0.9983 | TODO(run) | TODO(run) | TODO(run) |
| Celeb-DF-v2 ⚠️ | 0.9248 | 0.9246 | 0.9225 | 0.9550 | TODO(run) | TODO(run) |
| Celeb-DF-v3 | 0.9216 | 0.9144 | 0.9271 | 0.8440 | TODO(run) | TODO(run) |
| DFDC | 0.8468 | 0.8486 | 0.8477 | 0.8770 | TODO(run) | TODO(run) |
| DFDCP | 0.8970 | 0.9026 | 0.8913 | 0.8480 | TODO(run) | TODO(run) |
| UADFV | 0.9971 | 0.9975 | 0.9958 | TODO(run) | TODO(run) | TODO(run) |
| Deepfake-Eval-2024 | 0.6298 | 0.6281 | 0.6357 | 0.6850 | TODO(run) | TODO(run) |

Reproduced-DiCoME column: Phase-1 pilot P0-DS, MULTISOURCE_FINDINGS.md, video level, frozen FF++ threshold 0.5110, run 2026-08-15 — a different run from this one. The DiCoME-paper and DISCERN-v1 columns are `TODO(run)` — §21 requires all four rows to stay distinct, so they are shown unfilled rather than dropped.

## Per-branch and the §4.2 control

| source | `sem` | `ref` | `proc` | `e_direct` (control) | ref − direct |
|---|---:|---:|---:|---:|---:|
| FaceForensics++ | 0.9983 | 0.8963 | 0.6322 | 0.8880 | +0.0083 |
| Celeb-DF-v2 | 0.9225 | 0.7448 | 0.6695 | 0.6725 | +0.0723 |
| Celeb-DF-v3 | 0.9271 | 0.7439 | 0.6149 | 0.7063 | +0.0376 |
| DFDC | 0.8477 | 0.7448 | 0.5959 | 0.7198 | +0.0250 |
| DFDCP | 0.8913 | 0.7111 | 0.7113 | 0.6567 | +0.0543 |
| UADFV | 0.9958 | 0.8467 | 0.6426 | 0.9055 | -0.0587 |
| Deepfake-Eval-2024 | 0.6357 | 0.5697 | 0.4914 | 0.5639 | +0.0058 |

## §20 contribution diagnostics — how many sources clear §22's noise floor

*These are contribution diagnostics, not ablations: every configuration reuses heads trained with all three branches present.*

| question | sources outside the noise floor | range |
|---|---:|---|
| `applicability_vs_plain_ds` | 0 / 7 | -0.0056 … +0.0072 |
| `full_vs_sem_only` | 2 / 7 | -0.0127 … +0.0113 |
| `process_contribution` | 0 / 7 | -0.0075 … +0.0045 |
| `reference_contribution` | 1 / 7 | -0.0074 … +0.0118 |
| `reference_vs_direct_probe` | 5 / 7 | -0.0587 … +0.0723 |

## §22

`|ΔAUC| < 0.01`, or inconsistent family-level effects, requires a second seed before any promotion, removal or architecture change.

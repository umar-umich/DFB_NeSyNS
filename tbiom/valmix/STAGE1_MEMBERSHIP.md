# Stage 1 — branch membership gate

Anchor `clip` scored freshly in this harness (1350 DF40-Dev videos, 3 methods). Anchor operating threshold **0.3260**, EER on `logs/tbiom/score/clip_ffppval_seeded/per_sample_epoch_7.parquet`, frozen across every family.

> Each expert carries its OWN operating threshold, the EER on its own FF++ val export (listed below), also frozen across families. Both sides stay inside the firewall — no threshold sees DF40 — but they are not the same number. Scoring an expert at the anchor's threshold measures its calibration offset, not its complementarity: a branch whose probabilities all sit above the anchor's threshold reads as rescuing nearly every anchor error while harming nearly every anchor success, which is a constant predictor's signature. For the same reason the fused score blends MARGINS (`p − t`, zero at each source's own operating point), not raw probabilities.

| expert | operating threshold | val export |
|---|---:|---|
| `fsvfm_preserve` | 0.4239 | `logs/tbiom/score/fsvfm_preserve_ffppval/profile_epoch_009.parquet` |
| `fsvfm_ordinary` | 0.4405 | `logs/tbiom/score/fsvfm_ordinary_ffppval/profile_epoch_009.parquet` |
| `mrvae_rate` | 0.7940 | `logs/tbiom/score/fsvfm_preserve_ffppval/profile_epoch_009.parquet` |

Bars to enter: rescue margin > 0.05 on the pooled set AND on at least 2 families, AND a realizable gate recovering ≥ 25% of the accuracy headroom.

> Rescue rows are recomputed against THIS anchor. They are not inherited from a P0-DS, B1 or V1 table — that mistake was made twice in this project, and a rescue claim is only as valid as the baseline it is measured against.

## Decision

| expert | enters? | why |
|---|---|---|
| `fsvfm_preserve` | no | the realizable gate recovers only 7% of the headroom |
| `fsvfm_ordinary` | no | the realizable gate recovers only 10% of the headroom |
| `mrvae_rate` | no | rescue margin -0.131 (P(right|anchor wrong) 0.395 vs P(wrong|anchor right) 0.526) on 0 families; the realizable gate recovers only -2% of the headroom |

## The three numbers

| expert | P(right \| anchor wrong) | P(wrong \| anchor right) | margin | ceiling 1−P(both wrong) | anchor acc | realizable fused acc | recovered |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fsvfm_preserve` | 0.360 | 0.158 | +0.202 | 0.864 | 0.788 | 0.793 | 7% |
| `fsvfm_ordinary` | 0.367 | 0.161 | +0.206 | 0.866 | 0.788 | 0.796 | 10% |
| `mrvae_rate` | 0.395 | 0.526 | -0.131 | 0.872 | 0.788 | 0.787 | -2% |

The ceiling is an ACCURACY — the fraction at least one of anchor and expert gets right. A label-aware per-sample selector's AUROC would sit near 1 and make every realizable recovery look negligible, which is why the brief forbids it.

## Video AUROC, anchor vs realizable fusion

| expert | anchor AUROC | fused AUROC | gain | gate AUROC vs target | mean q |
|---|---:|---:|---:|---:|---:|
| `fsvfm_preserve` | 0.8719 | 0.8753 | +0.0034 | 0.6470 | 0.075 |
| `fsvfm_ordinary` | 0.8719 | 0.8773 | +0.0054 | 0.6665 | 0.079 |
| `mrvae_rate` | 0.8719 | 0.8564 | -0.0154 | 0.2850 | 0.087 |

## Per family (DF40-Dev method)

### `fsvfm_preserve`

| method | anchor acc | expert acc | P(right\|wrong) | P(wrong\|right) | margin | ceiling |
|---|---:|---:|---:|---:|---:|---:|
| VALmix:CDFv2val | 0.869 | 0.747 | 0.458 | 0.210 | +0.248 | 0.929 |
| VALmix:DFDCPval | 0.911 | 0.871 | 0.475 | 0.090 | +0.385 | 0.953 |
| VALmix:DFEval24val | 0.584 | 0.602 | 0.305 | 0.186 | +0.119 | 0.711 |

### `fsvfm_ordinary`

| method | anchor acc | expert acc | P(right\|wrong) | P(wrong\|right) | margin | ceiling |
|---|---:|---:|---:|---:|---:|---:|
| VALmix:CDFv2val | 0.869 | 0.740 | 0.441 | 0.215 | +0.226 | 0.927 |
| VALmix:DFDCPval | 0.911 | 0.878 | 0.500 | 0.085 | +0.415 | 0.956 |
| VALmix:DFEval24val | 0.584 | 0.600 | 0.316 | 0.198 | +0.118 | 0.716 |

### `mrvae_rate`

| method | anchor acc | expert acc | P(right\|wrong) | P(wrong\|right) | margin | ceiling |
|---|---:|---:|---:|---:|---:|---:|
| VALmix:CDFv2val | 0.869 | 0.402 | 0.390 | 0.596 | -0.206 | 0.920 |
| VALmix:DFDCPval | 0.911 | 0.498 | 0.125 | 0.466 | -0.341 | 0.922 |
| VALmix:DFEval24val | 0.584 | 0.471 | 0.455 | 0.517 | -0.063 | 0.773 |

## What happens next

**No expert passed.** Go to the reliability-centered fallback: do not build applicability machinery over experts that carry no recoverable signal. Note the fallback changes the reliability model's shape — with only the anchor there is no inter-branch conflict `C` and no `U_sup`, so the risk model becomes `g(V_sem, M_sem)`. Manufacturing C or U_sup from one branch would be wrong.

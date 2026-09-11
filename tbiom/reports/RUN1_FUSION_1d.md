# Run 1 — fusion replay (run1dce)

Three operators over the SAME saved branch opinions. No retraining: each view's Dirichlet opinion is recovered from its exported `(p, u)` via `S = K/u`, so the fusion can be swapped after the fact.

**F2's weights are fitted on VALmix only** and applied unchanged to every OOD set. No OOD export is opened during fitting.

| view | F2 weight |
|---|---:|
| semantic | 0.116 |
| artifact | 0.255 |
| fsvfm | 0.629 |

τ frozen per operator on FF++ val: F1 = 0.4385, F2 = 0.3806, F3 = 0.2565

## Complete-framework video AUROC

| dataset | F1 averaging | F2 learned | F3 DS | best branch | P0-DS |
|---|---:|---:|---:|---:|---:|
| CDFv2 | 0.9405 | **0.9530** | 0.9376 | 0.9340 | 0.9646 |
| CDFv3 | 0.8630 | **0.8873** | 0.8648 | 0.8808 | 0.8409 |
| DFD | **0.9467** | 0.9257 | 0.9449 | 0.9326 | 0.9421 |
| DFDC | 0.8772 | **0.8865** | 0.8791 | 0.8696 | 0.8828 |
| DFDCP | 0.8723 | **0.8889** | 0.8724 | 0.8867 | 0.8573 |
| DFEval24 | **0.6893** | 0.6800 | 0.6850 | 0.6657 | 0.6922 |
| VALmix | 0.8681 | **0.8793** | 0.8701 | 0.8624 | 0.8852 |

| operator | mean AUROC | mean FPR_real | vs P0-DS |
|---|---:|---:|---:|
| F1 | 0.8653 | 0.328 | -0.0012 |
| F2 | 0.8715 | 0.221 | +0.0051 |
| F3 | 0.8648 | 0.313 | -0.0016 |

## Real-side FPR at frozen τ

| dataset | F1 | F2 | F3 DS |
|---|---:|---:|---:|
| CDFv2 | 0.185 | 0.073 | 0.197 |
| CDFv3 | 0.185 | 0.073 | 0.197 |
| DFD | 0.410 | 0.193 | 0.369 |
| DFDC | 0.353 | 0.186 | 0.320 |
| DFDCP | 0.391 | 0.296 | 0.370 |
| DFEval24 | 0.453 | 0.481 | 0.442 |
| VALmix | 0.320 | 0.247 | 0.298 |

## Per-branch vacuity and standalone AUROC

| dataset | semantic AUROC / u | artifact AUROC / u | fsvfm AUROC / u |
|---|---|---|---|
| CDFv2 | 0.8487 / 0.036 | 0.8483 / 0.108 | 0.9340 / 0.045 |
| CDFv3 | 0.7574 / 0.032 | 0.7816 / 0.097 | 0.8808 / 0.051 |
| DFD | 0.9089 / 0.011 | 0.9326 / 0.082 | 0.8686 / 0.035 |
| DFDC | 0.8130 / 0.040 | 0.8035 / 0.079 | 0.8696 / 0.035 |
| DFDCP | 0.7915 / 0.039 | 0.7728 / 0.083 | 0.8867 / 0.043 |
| DFEval24 | 0.6657 / 0.062 | 0.6508 / 0.079 | 0.6579 / 0.048 |
| VALmix | 0.7980 / 0.046 | 0.7938 / 0.079 | 0.8624 / 0.038 |

## Verdict

**F2 beats DS** (0.8715 vs 0.8648, **+0.0067**). That isolates a second problem: DS combines these experts poorly even with branch withdrawal fixed. It is the measured motivation for a conflict-aware operator — subject to the real-side FPR column above not regressing.

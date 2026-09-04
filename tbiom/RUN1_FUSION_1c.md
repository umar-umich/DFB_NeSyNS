# Run 1 — fusion replay (run1cft)

Three operators over the SAME saved branch opinions. No retraining: each view's Dirichlet opinion is recovered from its exported `(p, u)` via `S = K/u`, so the fusion can be swapped after the fact.

**F2's weights are fitted on VALmix only** and applied unchanged to every OOD set. No OOD export is opened during fitting.

| view | F2 weight |
|---|---:|
| semantic | 0.000 |
| artifact | 0.536 |
| fsvfm | 0.464 |

τ frozen per operator on FF++ val: F1 = 0.5324, F2 = 0.5259, F3 = 0.6709

## Complete-framework video AUROC

| dataset | F1 averaging | F2 learned | F3 DS | best branch | P0-DS |
|---|---:|---:|---:|---:|---:|
| CDFv2 | 0.9596 | **0.9660** | 0.9582 | 0.9433 | 0.9646 |
| CDFv3 | 0.8876 | **0.9026** | 0.8843 | 0.8947 | 0.8409 |
| DFD | 0.9378 | **0.9394** | 0.9376 | 0.9213 | 0.9421 |
| DFDC | 0.8803 | **0.8837** | 0.8806 | 0.8623 | 0.8828 |
| DFDCP | 0.8798 | **0.8871** | 0.8806 | 0.8729 | 0.8573 |
| DFEval24 | 0.6796 | **0.6880** | 0.6792 | 0.6870 | 0.6922 |
| VALmix | 0.8830 | **0.8906** | 0.8834 | 0.8712 | 0.8852 |

| operator | mean AUROC | mean FPR_real | vs P0-DS |
|---|---:|---:|---:|
| F1 | 0.8725 | 0.260 | +0.0061 |
| F2 | 0.8796 | 0.222 | +0.0132 |
| F3 | 0.8720 | 0.226 | +0.0055 |

## Real-side FPR at frozen τ

| dataset | F1 | F2 | F3 DS |
|---|---:|---:|---:|
| CDFv2 | 0.157 | 0.107 | 0.124 |
| CDFv3 | 0.157 | 0.107 | 0.124 |
| DFD | 0.138 | 0.121 | 0.121 |
| DFDC | 0.297 | 0.263 | 0.264 |
| DFDCP | 0.387 | 0.330 | 0.343 |
| DFEval24 | 0.404 | 0.390 | 0.371 |
| VALmix | 0.281 | 0.236 | 0.234 |

## Per-branch vacuity and standalone AUROC

| dataset | semantic AUROC / u | artifact AUROC / u | fsvfm AUROC / u |
|---|---|---|---|
| CDFv2 | 0.9310 / 0.209 | 0.9433 / 0.226 | 0.9350 / 0.378 |
| CDFv3 | 0.8416 / 0.210 | 0.8611 / 0.222 | 0.8947 / 0.375 |
| DFD | 0.9213 / 0.127 | 0.9186 / 0.140 | 0.8814 / 0.297 |
| DFDC | 0.8590 / 0.201 | 0.8623 / 0.219 | 0.8525 / 0.325 |
| DFDCP | 0.8471 / 0.195 | 0.8583 / 0.230 | 0.8729 / 0.347 |
| DFEval24 | 0.6528 / 0.229 | 0.6616 / 0.284 | 0.6870 / 0.391 |
| VALmix | 0.8555 / 0.200 | 0.8659 / 0.235 | 0.8712 / 0.355 |

## Verdict

**F2 beats DS** (0.8796 vs 0.8720, **+0.0076**). That isolates a second problem: DS combines these experts poorly even with branch withdrawal fixed. It is the measured motivation for a conflict-aware operator — subject to the real-side FPR column above not regressing.

# Run 1 — fusion replay (run1fusedonly)

Three operators over the SAME saved branch opinions. No retraining: each view's Dirichlet opinion is recovered from its exported `(p, u)` via `S = K/u`, so the fusion can be swapped after the fact.

**F2's weights are fitted on VALmix only** and applied unchanged to every OOD set. No OOD export is opened during fitting.

| view | F2 weight |
|---|---:|
| semantic | 0.218 |
| artifact | 0.000 |
| fsvfm | 0.782 |

τ frozen per operator on FF++ val: F1 = 0.5422, F2 = 0.5982, F3 = 0.6261

## Complete-framework video AUROC

| dataset | F1 averaging | F2 learned | F3 DS | best branch | P0-DS |
|---|---:|---:|---:|---:|---:|
| CDFv2 | 0.8637 | **0.8638** | 0.8638 | 0.8638 | 0.9646 |
| CDFv3 | 0.8619 | **0.8626** | 0.8626 | 0.8626 | 0.8409 |
| DFD | **0.8812** | 0.8812 | 0.8812 | 0.8812 | 0.9421 |
| DFDC | 0.8554 | 0.8557 | **0.8559** | 0.8557 | 0.8828 |
| DFDCP | **0.8381** | 0.8381 | 0.8381 | 0.8381 | 0.8573 |
| DFEval24 | 0.6788 | **0.6790** | 0.6789 | 0.6790 | 0.6922 |
| VALmix | 0.8274 | **0.8276** | 0.8276 | 0.8276 | 0.8852 |

| operator | mean AUROC | mean FPR_real | vs P0-DS |
|---|---:|---:|---:|
| F1 | 0.8295 | 0.173 | -0.0370 |
| F2 | 0.8297 | 0.173 | -0.0367 |
| F3 | 0.8297 | 0.173 | -0.0367 |

## Real-side FPR at frozen τ

| dataset | F1 | F2 | F3 DS |
|---|---:|---:|---:|
| CDFv2 | 0.045 | 0.045 | 0.045 |
| CDFv3 | 0.045 | 0.045 | 0.045 |
| DFD | 0.085 | 0.083 | 0.083 |
| DFDC | 0.184 | 0.184 | 0.184 |
| DFDCP | 0.357 | 0.357 | 0.357 |
| DFEval24 | 0.299 | 0.297 | 0.297 |
| VALmix | 0.199 | 0.199 | 0.199 |

## Per-branch vacuity and standalone AUROC

| dataset | semantic AUROC / u | artifact AUROC / u | fsvfm AUROC / u |
|---|---|---|---|
| CDFv2 | 0.5562 / 1.000 | 0.4277 / 0.997 | 0.8638 / 0.132 |
| CDFv3 | 0.5333 / 1.000 | 0.1889 / 0.998 | 0.8626 / 0.142 |
| DFD | 0.6416 / 1.000 | 0.3965 / 0.997 | 0.8812 / 0.094 |
| DFDC | 0.5121 / 1.000 | 0.4784 / 0.993 | 0.8557 / 0.092 |
| DFDCP | 0.5459 / 1.000 | 0.4111 / 0.997 | 0.8381 / 0.132 |
| DFEval24 | 0.5346 / 1.000 | 0.4411 / 0.996 | 0.6790 / 0.164 |
| VALmix | 0.5525 / 1.000 | 0.4342 / 0.997 | 0.8276 / 0.121 |

## Verdict

**No second defect isolated.** Best is F3 at 0.8297 against DS's 0.8297 — a gap of +0.0000, which is inside the noise band and NOT evidence that DS combines these experts poorly. F1 0.8295, F2 0.8297, F3 0.8297. A difference has to clear ~0.005 before it says anything; anything smaller is operator-choice noise.

# Run 1 — fusion replay (run1auxedl)

Three operators over the SAME saved branch opinions. No retraining: each view's Dirichlet opinion is recovered from its exported `(p, u)` via `S = K/u`, so the fusion can be swapped after the fact.

**F2's weights are fitted on VALmix only** and applied unchanged to every OOD set. No OOD export is opened during fitting.

| view | F2 weight |
|---|---:|
| semantic | 0.000 |
| artifact | 0.809 |
| fsvfm | 0.191 |

τ frozen per operator on FF++ val: F1 = 0.4421, F2 = 0.4279, F3 = 0.3971

## Complete-framework video AUROC

| dataset | F1 averaging | F2 learned | F3 DS | best branch | P0-DS |
|---|---:|---:|---:|---:|---:|
| CDFv2 | 0.9517 | **0.9597** | 0.9474 | 0.9607 | 0.9646 |
| CDFv3 | 0.8860 | **0.8877** | 0.8871 | 0.8750 | 0.8409 |
| DFD | **0.9499** | 0.9474 | 0.9454 | 0.9418 | 0.9421 |
| DFDC | 0.8904 | 0.8885 | **0.8906** | 0.8851 | 0.8828 |
| DFDCP | 0.8971 | 0.8975 | **0.9038** | 0.8898 | 0.8573 |
| DFEval24 | **0.6915** | 0.6838 | 0.6866 | 0.6826 | 0.6922 |
| VALmix | 0.8808 | 0.8826 | **0.8831** | 0.8752 | 0.8852 |

| operator | mean AUROC | mean FPR_real | vs P0-DS |
|---|---:|---:|---:|
| F1 | 0.8782 | 0.237 | +0.0118 |
| F2 | 0.8782 | 0.226 | +0.0117 |
| F3 | 0.8777 | 0.215 | +0.0113 |

## Real-side FPR at frozen τ

| dataset | F1 | F2 | F3 DS |
|---|---:|---:|---:|
| CDFv2 | 0.135 | 0.152 | 0.118 |
| CDFv3 | 0.135 | 0.152 | 0.118 |
| DFD | 0.072 | 0.058 | 0.058 |
| DFDC | 0.273 | 0.283 | 0.249 |
| DFDCP | 0.430 | 0.391 | 0.396 |
| DFEval24 | 0.357 | 0.308 | 0.334 |
| VALmix | 0.253 | 0.239 | 0.231 |

## Per-branch vacuity and standalone AUROC

| dataset | semantic AUROC / u | artifact AUROC / u | fsvfm AUROC / u |
|---|---|---|---|
| CDFv2 | 0.9506 / 0.210 | 0.9607 / 0.216 | 0.8666 / 0.135 |
| CDFv3 | 0.8561 / 0.241 | 0.8750 / 0.240 | 0.8630 / 0.146 |
| DFD | 0.9418 / 0.152 | 0.9331 / 0.156 | 0.8831 / 0.109 |
| DFDC | 0.8851 / 0.214 | 0.8826 / 0.221 | 0.8564 / 0.094 |
| DFDCP | 0.8892 / 0.218 | 0.8898 / 0.232 | 0.8368 / 0.134 |
| DFEval24 | 0.6716 / 0.268 | 0.6704 / 0.302 | 0.6826 / 0.185 |
| VALmix | 0.8722 / 0.219 | 0.8752 / 0.234 | 0.8304 / 0.130 |

## Verdict

**No second defect isolated.** Best is F1 at 0.8782 against DS's 0.8777 — a gap of +0.0005, which is inside the noise band and NOT evidence that DS combines these experts poorly. F1 0.8782, F2 0.8782, F3 0.8777. A difference has to clear ~0.005 before it says anything; anything smaller is operator-choice noise.

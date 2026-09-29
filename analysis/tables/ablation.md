### Branch ablation — DF40 video-level AUROC, PCR6+-R throughout

Every row is the same operator, the same reliability fitting on the same five development
folds and the same 65 sets; only the views available to the fusion change.

| Views | Face swap | Reenactment | Synthesis | Editing | Macro |
|---|---:|---:|---:|---:|---:|
| APP | 0.968 ± 0.020 | 0.786 ± 0.158 | 0.976 ± 0.036 | 0.912 ± 0.095 | 0.910 |
| OM | 0.971 ± 0.018 | 0.799 ± 0.148 | 0.983 ± 0.022 | 0.936 ± 0.070 | 0.922 |
| T | 0.892 ± 0.062 | 0.918 ± 0.100 | 0.835 ± 0.102 | 0.830 ± 0.177 | 0.869 |
| APP + OM | 0.971 ± 0.018 | 0.799 ± 0.148 | 0.983 ± 0.022 | 0.936 ± 0.070 | 0.922 |
| T + APP | 0.972 ± 0.021 | 0.925 ± 0.061 | 0.957 ± 0.044 | 0.925 ± 0.103 | 0.945 |
| T + OM | 0.977 ± 0.016 | 0.939 ± 0.051 | 0.969 ± 0.036 | 0.926 ± 0.120 | 0.953 |
| T + APP + OM (ours) | 0.978 ± 0.016 | 0.935 ± 0.052 | 0.971 ± 0.035 | 0.929 ± 0.114 | 0.953 |

**What the rows say.**

- **`APP + OM` is identical to `OM` in every family.** The reliability fit drives the other view's weight to zero, so the pair is not a fusion at all -- it is `OM` with a second view that the fitting discards.
- **Adding APP to `T + OM` moves the macro by +0.0002** and reenactment by -0.0043. On this benchmark the appearance view does not earn its place once the temporal and off-manifold views are both present; it is carried on the strength of the other evaluations, not this one.

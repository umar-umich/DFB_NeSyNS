# Stage 1 — branch membership gate

Anchor `clip` scored freshly in this harness (19441 DF40-Dev videos, 36 methods). Anchor operating threshold **0.3260**, EER on `logs/tbiom/score/clip_ffppval_seeded/per_sample_epoch_7.parquet`, frozen across every family.

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
| `fsvfm_preserve` | no | the realizable gate recovers only 6% of the headroom |
| `fsvfm_ordinary` | no | the realizable gate recovers only 6% of the headroom |
| `mrvae_rate` | no | the realizable gate recovers only 1% of the headroom |

## The three numbers

| expert | P(right \| anchor wrong) | P(wrong \| anchor right) | margin | ceiling 1−P(both wrong) | anchor acc | realizable fused acc | recovered |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fsvfm_preserve` | 0.275 | 0.190 | +0.085 | 0.788 | 0.708 | 0.713 | 6% |
| `fsvfm_ordinary` | 0.254 | 0.196 | +0.058 | 0.782 | 0.708 | 0.712 | 6% |
| `mrvae_rate` | 0.734 | 0.439 | +0.295 | 0.922 | 0.708 | 0.711 | 1% |

The ceiling is an ACCURACY — the fraction at least one of anchor and expert gets right. A label-aware per-sample selector's AUROC would sit near 1 and make every realizable recovery look negligible, which is why the brief forbids it.

## Video AUROC, anchor vs realizable fusion

| expert | anchor AUROC | fused AUROC | gain | gate AUROC vs target | mean q |
|---|---:|---:|---:|---:|---:|
| `fsvfm_preserve` | 0.8501 | 0.8597 | +0.0096 | 0.7394 | 0.081 |
| `fsvfm_ordinary` | 0.8501 | 0.8589 | +0.0089 | 0.7217 | 0.075 |
| `mrvae_rate` | 0.8501 | 0.8264 | -0.0237 | 0.5302 | 0.215 |

## Per family (DF40-Dev method)

### `fsvfm_preserve`

| method | anchor acc | expert acc | P(right\|wrong) | P(wrong\|right) | margin | ceiling |
|---|---:|---:|---:|---:|---:|---:|
| CollabDiff | 0.583 | 0.591 | 0.139 | 0.084 | +0.054 | 0.641 |
| DiT_cdf | 0.530 | 0.551 | 0.523 | 0.423 | +0.099 | 0.776 |
| DiT_ff | 0.669 | 0.697 | 0.381 | 0.147 | +0.234 | 0.795 |
| SiT_cdf | 0.622 | 0.564 | 0.539 | 0.421 | +0.117 | 0.826 |
| SiT_ff | 0.768 | 0.764 | 0.508 | 0.159 | +0.350 | 0.886 |
| StyleGAN2_cdf | 0.979 | 0.977 | 0.636 | 0.016 | +0.621 | 0.992 |
| StyleGAN2_ff | 0.972 | 0.988 | 0.714 | 0.004 | +0.710 | 0.992 |
| VQGAN_cdf | 0.979 | 0.992 | 1.000 | 0.008 | +0.992 | 1.000 |
| VQGAN_ff | 0.969 | 0.980 | 0.875 | 0.016 | +0.859 | 0.996 |
| blendface_cdf | 0.771 | 0.736 | 0.466 | 0.184 | +0.281 | 0.878 |
| blendface_ff | 0.815 | 0.807 | 0.431 | 0.107 | +0.324 | 0.895 |
| danet_cdf | 0.425 | 0.403 | 0.204 | 0.329 | -0.125 | 0.543 |
| danet_ff | 0.612 | 0.612 | 0.215 | 0.136 | +0.079 | 0.696 |
| deepfacelab | 0.845 | 0.865 | 0.435 | 0.056 | +0.379 | 0.912 |
| faceswap_cdf | 0.981 | 0.970 | 0.900 | 0.029 | +0.871 | 0.998 |
| faceswap_ff | 0.943 | 0.939 | 0.625 | 0.042 | +0.583 | 0.978 |
| facevid2vid_cdf | 0.641 | 0.608 | 0.378 | 0.264 | +0.115 | 0.777 |
| facevid2vid_ff | 0.633 | 0.651 | 0.228 | 0.103 | +0.124 | 0.716 |
| heygen | 0.604 | 0.624 | 0.100 | 0.033 | +0.067 | 0.644 |
| mcnet_cdf | 0.501 | 0.470 | 0.217 | 0.278 | -0.062 | 0.609 |
| mcnet_ff | 0.662 | 0.655 | 0.213 | 0.120 | +0.093 | 0.734 |
| rddm_cdf | 0.842 | 0.307 | 0.182 | 0.669 | -0.487 | 0.871 |
| rddm_ff | 0.949 | 0.539 | 0.462 | 0.456 | +0.005 | 0.972 |
| sadtalker_cdf | 0.362 | 0.322 | 0.143 | 0.362 | -0.219 | 0.454 |
| sadtalker_ff | 0.567 | 0.571 | 0.168 | 0.122 | +0.046 | 0.640 |
| simswap | 0.722 | 0.612 | 0.248 | 0.248 | -0.001 | 0.791 |
| simswap_cdf | 0.669 | 0.518 | 0.319 | 0.384 | -0.065 | 0.775 |
| simswap_ff | 0.827 | 0.687 | 0.188 | 0.209 | -0.021 | 0.860 |
| stargan | 0.919 | 0.843 | 0.303 | 0.109 | +0.194 | 0.944 |
| starganv2 | 0.624 | 0.617 | 0.135 | 0.093 | +0.042 | 0.675 |
| styleclip | 0.646 | 0.753 | 0.425 | 0.067 | +0.357 | 0.797 |
| tpsm_cdf | 0.447 | 0.316 | 0.100 | 0.417 | -0.317 | 0.502 |
| tpsm_ff | 0.665 | 0.607 | 0.228 | 0.202 | +0.026 | 0.742 |
| uniface | 0.834 | 0.723 | 0.463 | 0.226 | +0.237 | 0.911 |
| uniface_cdf | 0.790 | 0.669 | 0.382 | 0.254 | +0.128 | 0.870 |
| uniface_ff | 0.863 | 0.820 | 0.421 | 0.117 | +0.304 | 0.921 |

### `fsvfm_ordinary`

| method | anchor acc | expert acc | P(right\|wrong) | P(wrong\|right) | margin | ceiling |
|---|---:|---:|---:|---:|---:|---:|
| CollabDiff | 0.583 | 0.581 | 0.124 | 0.091 | +0.032 | 0.634 |
| DiT_cdf | 0.530 | 0.549 | 0.502 | 0.409 | +0.093 | 0.766 |
| DiT_ff | 0.669 | 0.681 | 0.369 | 0.165 | +0.204 | 0.791 |
| SiT_cdf | 0.622 | 0.544 | 0.508 | 0.434 | +0.074 | 0.814 |
| SiT_ff | 0.768 | 0.740 | 0.424 | 0.164 | +0.260 | 0.866 |
| StyleGAN2_cdf | 0.979 | 0.990 | 0.727 | 0.004 | +0.723 | 0.994 |
| StyleGAN2_ff | 0.972 | 0.984 | 0.714 | 0.008 | +0.706 | 0.992 |
| VQGAN_cdf | 0.979 | 0.996 | 1.000 | 0.004 | +0.996 | 1.000 |
| VQGAN_ff | 0.969 | 0.980 | 0.875 | 0.016 | +0.859 | 0.996 |
| blendface_cdf | 0.771 | 0.728 | 0.491 | 0.202 | +0.289 | 0.884 |
| blendface_ff | 0.815 | 0.807 | 0.471 | 0.116 | +0.355 | 0.902 |
| danet_cdf | 0.425 | 0.378 | 0.181 | 0.356 | -0.175 | 0.529 |
| danet_ff | 0.612 | 0.616 | 0.196 | 0.118 | +0.078 | 0.688 |
| deepfacelab | 0.845 | 0.851 | 0.522 | 0.088 | +0.434 | 0.926 |
| faceswap_cdf | 0.981 | 0.968 | 0.900 | 0.031 | +0.869 | 0.998 |
| faceswap_ff | 0.943 | 0.932 | 0.500 | 0.042 | +0.458 | 0.971 |
| facevid2vid_cdf | 0.641 | 0.577 | 0.341 | 0.291 | +0.050 | 0.763 |
| facevid2vid_ff | 0.633 | 0.651 | 0.228 | 0.103 | +0.124 | 0.716 |
| heygen | 0.604 | 0.604 | 0.000 | 0.000 | +0.000 | 0.604 |
| mcnet_cdf | 0.501 | 0.432 | 0.209 | 0.345 | -0.136 | 0.605 |
| mcnet_ff | 0.662 | 0.615 | 0.202 | 0.174 | +0.028 | 0.730 |
| rddm_cdf | 0.842 | 0.316 | 0.182 | 0.659 | -0.478 | 0.871 |
| rddm_ff | 0.949 | 0.555 | 0.462 | 0.440 | +0.022 | 0.972 |
| sadtalker_cdf | 0.362 | 0.259 | 0.071 | 0.412 | -0.341 | 0.408 |
| sadtalker_ff | 0.567 | 0.535 | 0.092 | 0.128 | -0.036 | 0.607 |
| simswap | 0.722 | 0.614 | 0.238 | 0.240 | -0.003 | 0.788 |
| simswap_cdf | 0.669 | 0.514 | 0.295 | 0.378 | -0.083 | 0.767 |
| simswap_ff | 0.827 | 0.709 | 0.188 | 0.183 | +0.005 | 0.860 |
| stargan | 0.919 | 0.806 | 0.242 | 0.144 | +0.098 | 0.939 |
| starganv2 | 0.624 | 0.614 | 0.135 | 0.097 | +0.038 | 0.675 |
| styleclip | 0.646 | 0.729 | 0.384 | 0.082 | +0.301 | 0.782 |
| tpsm_cdf | 0.447 | 0.297 | 0.086 | 0.443 | -0.357 | 0.494 |
| tpsm_ff | 0.665 | 0.600 | 0.196 | 0.197 | -0.001 | 0.731 |
| uniface | 0.834 | 0.739 | 0.475 | 0.208 | +0.267 | 0.913 |
| uniface_cdf | 0.790 | 0.677 | 0.409 | 0.252 | +0.157 | 0.876 |
| uniface_ff | 0.863 | 0.831 | 0.474 | 0.113 | +0.361 | 0.928 |

### `mrvae_rate`

| method | anchor acc | expert acc | P(right\|wrong) | P(wrong\|right) | margin | ceiling |
|---|---:|---:|---:|---:|---:|---:|
| CollabDiff | 0.583 | 0.493 | 0.936 | 0.824 | +0.112 | 0.973 |
| DiT_cdf | 0.530 | 0.733 | 0.798 | 0.325 | +0.474 | 0.905 |
| DiT_ff | 0.669 | 0.563 | 0.702 | 0.506 | +0.196 | 0.902 |
| SiT_cdf | 0.622 | 0.706 | 0.741 | 0.314 | +0.426 | 0.902 |
| SiT_ff | 0.768 | 0.614 | 0.797 | 0.441 | +0.356 | 0.953 |
| StyleGAN2_cdf | 0.979 | 0.504 | 0.273 | 0.491 | -0.219 | 0.985 |
| StyleGAN2_ff | 0.972 | 0.664 | 0.143 | 0.321 | -0.178 | 0.976 |
| VQGAN_cdf | 0.979 | 0.899 | 0.455 | 0.091 | +0.363 | 0.989 |
| VQGAN_ff | 0.969 | 0.684 | 0.125 | 0.298 | -0.173 | 0.973 |
| blendface_cdf | 0.771 | 0.540 | 0.474 | 0.440 | +0.034 | 0.880 |
| blendface_ff | 0.815 | 0.531 | 0.529 | 0.469 | +0.061 | 0.913 |
| danet_cdf | 0.425 | 0.775 | 0.862 | 0.342 | +0.520 | 0.921 |
| danet_ff | 0.612 | 0.645 | 0.766 | 0.432 | +0.334 | 0.909 |
| deepfacelab | 0.845 | 0.466 | 0.304 | 0.504 | -0.200 | 0.892 |
| faceswap_cdf | 0.981 | 0.693 | 0.100 | 0.296 | -0.196 | 0.983 |
| faceswap_ff | 0.943 | 0.563 | 0.375 | 0.426 | -0.051 | 0.964 |
| facevid2vid_cdf | 0.641 | 0.763 | 0.843 | 0.282 | +0.561 | 0.944 |
| facevid2vid_ff | 0.633 | 0.633 | 0.772 | 0.448 | +0.324 | 0.916 |
| heygen | 0.604 | 0.465 | 0.950 | 0.852 | +0.098 | 0.980 |
| mcnet_cdf | 0.501 | 0.768 | 0.870 | 0.333 | +0.537 | 0.935 |
| mcnet_ff | 0.662 | 0.644 | 0.830 | 0.451 | +0.379 | 0.942 |
| rddm_cdf | 0.842 | 0.781 | 0.896 | 0.241 | +0.655 | 0.984 |
| rddm_ff | 0.949 | 0.681 | 0.692 | 0.320 | +0.373 | 0.984 |
| sadtalker_cdf | 0.362 | 0.501 | 0.523 | 0.538 | -0.015 | 0.696 |
| sadtalker_ff | 0.567 | 0.535 | 0.588 | 0.506 | +0.082 | 0.822 |
| simswap | 0.722 | 0.441 | 0.505 | 0.584 | -0.079 | 0.862 |
| simswap_cdf | 0.669 | 0.524 | 0.488 | 0.458 | +0.030 | 0.831 |
| simswap_ff | 0.827 | 0.550 | 0.562 | 0.452 | +0.110 | 0.924 |
| stargan | 0.919 | 0.525 | 0.848 | 0.504 | +0.344 | 0.988 |
| starganv2 | 0.624 | 0.446 | 0.782 | 0.757 | +0.025 | 0.918 |
| styleclip | 0.646 | 0.700 | 0.911 | 0.416 | +0.495 | 0.969 |
| tpsm_cdf | 0.447 | 0.629 | 0.711 | 0.472 | +0.239 | 0.840 |
| tpsm_ff | 0.665 | 0.629 | 0.815 | 0.464 | +0.351 | 0.938 |
| uniface | 0.834 | 0.484 | 0.438 | 0.506 | -0.069 | 0.907 |
| uniface_cdf | 0.790 | 0.530 | 0.509 | 0.465 | +0.044 | 0.897 |
| uniface_ff | 0.863 | 0.550 | 0.605 | 0.458 | +0.147 | 0.946 |

## What happens next

**No expert passed.** Go to the reliability-centered fallback: do not build applicability machinery over experts that carry no recoverable signal. Note the fallback changes the reliability model's shape — with only the anchor there is no inter-branch conflict `C` and no `U_sup`, so the risk model becomes `g(V_sem, M_sem)`. Manufacturing C or U_sup from one branch would be wrong.

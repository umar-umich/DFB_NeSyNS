# Stage 4 — processing discrimination (the central test)

**Detector: arm B, the frozen configuration.** The detector is named because this property is a property OF a detector, and a table that does not say which one invites being quoted about the wrong model.

`tau` = 0.3971, the EER on **baseline** FF++ val, applied unchanged everywhere below. Held-out severities (jpeg q40, resize 0.60) appear in no fitted Stage 2 curve.

## Deployable: frozen threshold, never refitted

| dataset | condition | FPR_real | TPR | ΔFPR | ΔTPR |
|---|---|---:|---:|---:|---:|
| CDFv2 | baseline | 0.118 | 0.844 | — | — |
| CDFv2 | jpeg q40 | 0.034 | 0.391 | -0.084 | -0.453 |
| CDFv2 | resize 0.60 | 0.118 | 0.824 | +0.000 | -0.021 |
| DFDCP | baseline | 0.396 | 0.965 | — | — |
| DFDCP | jpeg q40 | 0.122 | 0.660 | -0.274 | -0.304 |
| DFDCP | resize 0.60 | 0.496 | 0.915 | +0.100 | -0.050 |

## Retrospective: FPR at matched TPR (not deployable)

Matching TPR after the fact needs the labels, so this reads discrimination separately from calibration and is not an operating point anyone could set.

| dataset | condition | FPR @ TPR=0.80 | FPR @ TPR=0.90 |
|---|---|---:|---:|
| CDFv2 | baseline | 0.079 | 0.157 |
| CDFv2 | jpeg q40 | 0.376 | 0.478 |
| CDFv2 | resize 0.60 | 0.101 | 0.185 |
| DFDCP | baseline | 0.191 | 0.265 |
| DFDCP | jpeg q40 | 0.252 | 0.374 |
| DFDCP | resize 0.60 | 0.317 | 0.452 |

## Does fusion resist processing better than any single view?

The claim that would justify a multi-view design is that combining views is more robust than the best view alone, not merely more accurate on clean data.

| dataset | condition | Appearance | Off-manifold residual | Bona-fide face prior | Fused |
|---|---|---:|---:|---:|---:|
| CDFv2 | jpeg q40 | +0.163 | -0.124 | -0.062 | **-0.084** |
| CDFv2 | resize 0.60 | -0.022 | +0.028 | +0.028 | **+0.000** |
| DFDCP | jpeg q40 | -0.104 | -0.322 | -0.278 | **-0.274** |
| DFDCP | resize 0.60 | +0.057 | +0.113 | +0.113 | **+0.100** |

Cells are the CHANGE in FPR_real from that dataset's own baseline, so a smaller number is a view that resists the processing better.

## Verdict

**Mean ΔFPR_real -0.065, mean ΔTPR -0.207** across 4 held-out conditions.

**False positives are controlled, but by missing fakes.** FPR moves -0.065 while TPR falls -0.207. Lowering false positives by losing sensitivity is not an improvement, and reporting the FPR alone would have looked like one.

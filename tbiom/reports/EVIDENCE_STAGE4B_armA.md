# Stage 4 — processing discrimination (the central test)

**Detector: SBI arm A.** The detector is named because this property is a property OF a detector, and a table that does not say which one invites being quoted about the wrong model.

`tau` = 0.6133, the EER on **baseline** FF++ val, applied unchanged everywhere below. Held-out severities (jpeg q40, resize 0.60) appear in no fitted Stage 2 curve.

## Deployable: frozen threshold, never refitted

| dataset | condition | FPR_real | TPR | ΔFPR | ΔTPR |
|---|---|---:|---:|---:|---:|
| CDFv2 | baseline | 0.101 | 0.788 | — | — |
| CDFv2 | jpeg q40 | 0.028 | 0.379 | -0.073 | -0.409 |
| CDFv2 | resize 0.60 | 0.129 | 0.809 | +0.028 | +0.021 |
| DFDCP | baseline | 0.348 | 0.932 | — | — |
| DFDCP | jpeg q40 | 0.104 | 0.587 | -0.243 | -0.344 |
| DFDCP | resize 0.60 | 0.457 | 0.882 | +0.109 | -0.050 |

## Retrospective: FPR at matched TPR (not deployable)

Matching TPR after the fact needs the labels, so this reads discrimination separately from calibration and is not an operating point anyone could set.

| dataset | condition | FPR @ TPR=0.80 | FPR @ TPR=0.90 |
|---|---|---:|---:|
| CDFv2 | baseline | 0.112 | 0.202 |
| CDFv2 | jpeg q40 | 0.416 | 0.567 |
| CDFv2 | resize 0.60 | 0.129 | 0.236 |
| DFDCP | baseline | 0.230 | 0.296 |
| DFDCP | jpeg q40 | 0.235 | 0.309 |
| DFDCP | resize 0.60 | 0.352 | 0.478 |

## Does fusion resist processing better than any single view?

The claim that would justify a multi-view design is that combining views is more robust than the best view alone, not merely more accurate on clean data.

| dataset | condition | Appearance | Off-manifold residual | Bona-fide face prior | Fused |
|---|---|---:|---:|---:|---:|
| CDFv2 | jpeg q40 | +0.096 | -0.045 | -0.028 | **-0.073** |
| CDFv2 | resize 0.60 | +0.028 | +0.039 | +0.011 | **+0.028** |
| DFDCP | jpeg q40 | -0.139 | -0.230 | -0.222 | **-0.243** |
| DFDCP | resize 0.60 | +0.078 | +0.096 | +0.083 | **+0.109** |

Cells are the CHANGE in FPR_real from that dataset's own baseline, so a smaller number is a view that resists the processing better.

## Verdict

**Mean ΔFPR_real -0.045, mean ΔTPR -0.196** across 4 held-out conditions.

**False positives are controlled, but by missing fakes.** FPR moves -0.045 while TPR falls -0.196. Lowering false positives by losing sensitivity is not an improvement, and reporting the FPR alone would have looked like one.

# Stage 4 — processing discrimination (the central test)

> **Superseded (2026-09-14).** These numbers used an undeclared newest-checkpoint rule and averaged
> processing effects across conditions. Use `tbiom/reports/CORRECTED_V2.md` and the corrections in
> `tbiom/docs/decision-2026-09-13.md`.


**Detector: SBI arm B.** The detector is named because this property is a property OF a detector, and a table that does not say which one invites being quoted about the wrong model.

`tau` = 0.7710, the EER on **baseline** FF++ val, applied unchanged everywhere below. Held-out severities (jpeg q40, resize 0.60) appear in no fitted Stage 2 curve.

## Deployable: frozen threshold, never refitted

| dataset | condition | FPR_real | TPR | ΔFPR | ΔTPR |
|---|---|---:|---:|---:|---:|
| CDFv2 | baseline | 0.045 | 0.597 | — | — |
| CDFv2 | jpeg q40 | 0.135 | 0.668 | +0.090 | +0.071 |
| CDFv2 | resize 0.60 | 0.034 | 0.406 | -0.011 | -0.191 |
| DFDCP | baseline | 0.196 | 0.830 | — | — |
| DFDCP | jpeg q40 | 0.252 | 0.894 | +0.057 | +0.064 |
| DFDCP | resize 0.60 | 0.187 | 0.764 | -0.009 | -0.066 |

## Retrospective: FPR at matched TPR (not deployable)

Matching TPR after the fact needs the labels, so this reads discrimination separately from calibration and is not an operating point anyone could set.

| dataset | condition | FPR @ TPR=0.80 | FPR @ TPR=0.90 |
|---|---|---:|---:|
| CDFv2 | baseline | 0.140 | 0.225 |
| CDFv2 | jpeg q40 | 0.242 | 0.281 |
| CDFv2 | resize 0.60 | 0.163 | 0.270 |
| DFDCP | baseline | 0.178 | 0.257 |
| DFDCP | jpeg q40 | 0.170 | 0.261 |
| DFDCP | resize 0.60 | 0.213 | 0.304 |

## Does fusion resist processing better than any single view?

The claim that would justify a multi-view design is that combining views is more robust than the best view alone, not merely more accurate on clean data.

| dataset | condition | Appearance | Off-manifold residual | Bona-fide face prior | Fused |
|---|---|---:|---:|---:|---:|
| CDFv2 | jpeg q40 | +0.051 | +0.051 | +0.011 | **+0.090** |
| CDFv2 | resize 0.60 | -0.011 | -0.011 | -0.006 | **-0.011** |
| DFDCP | jpeg q40 | -0.004 | -0.017 | +0.043 | **+0.057** |
| DFDCP | resize 0.60 | -0.030 | -0.035 | +0.013 | **-0.009** |

Cells are the CHANGE in FPR_real from that dataset's own baseline, so a smaller number is a view that resists the processing better.

## Verdict

**Mean ΔFPR_real +0.032, mean ΔTPR -0.031** across 4 held-out conditions.

**The system does not resist held-out processing.** False alarms on genuine video rise +0.032 at a threshold that was never refitted, on 2 of 4 conditions. That is the nuisance-induced false positive the whole direction exists to fix, and the frozen detector does not fix it — which is the finding, and it says improve the measurements before building a reasoning layer on top of them.

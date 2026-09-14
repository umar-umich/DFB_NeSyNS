# Stage 4 — processing discrimination (the central test)

> **Superseded (2026-09-14).** These numbers used an undeclared newest-checkpoint rule and averaged
> processing effects across conditions. Use `tbiom/reports/CORRECTED_V2.md` and the corrections in
> `tbiom/docs/decision-2026-09-13.md`.


**Detector: SBI arm C.** The detector is named because this property is a property OF a detector, and a table that does not say which one invites being quoted about the wrong model.

`tau` = 0.9997, the EER on **baseline** FF++ val, applied unchanged everywhere below. Held-out severities (jpeg q40, resize 0.60) appear in no fitted Stage 2 curve.

## Deployable: frozen threshold, never refitted

| dataset | condition | FPR_real | TPR | ΔFPR | ΔTPR |
|---|---|---:|---:|---:|---:|
| CDFv2 | baseline | 0.045 | 0.632 | — | — |
| CDFv2 | jpeg q40 | 0.000 | 0.044 | -0.045 | -0.588 |
| CDFv2 | resize 0.60 | 0.006 | 0.188 | -0.039 | -0.444 |
| DFDCP | baseline | 0.248 | 0.764 | — | — |
| DFDCP | jpeg q40 | 0.000 | 0.217 | -0.248 | -0.547 |
| DFDCP | resize 0.60 | 0.061 | 0.370 | -0.187 | -0.394 |

## Retrospective: FPR at matched TPR (not deployable)

Matching TPR after the fact needs the labels, so this reads discrimination separately from calibration and is not an operating point anyone could set.

| dataset | condition | FPR @ TPR=0.80 | FPR @ TPR=0.90 |
|---|---|---:|---:|
| CDFv2 | baseline | 0.101 | 0.197 |
| CDFv2 | jpeg q40 | 0.129 | 0.287 |
| CDFv2 | resize 0.60 | 0.090 | 0.191 |
| DFDCP | baseline | 0.278 | 0.465 |
| DFDCP | jpeg q40 | 0.204 | 0.317 |
| DFDCP | resize 0.60 | 0.430 | 0.530 |

## Does fusion resist processing better than any single view?

The claim that would justify a multi-view design is that combining views is more robust than the best view alone, not merely more accurate on clean data.

| dataset | condition | Appearance | Off-manifold residual | Bona-fide face prior | Fused |
|---|---|---:|---:|---:|---:|
| CDFv2 | jpeg q40 | +0.000 | +0.000 | +0.000 | **-0.045** |
| CDFv2 | resize 0.60 | +0.000 | +0.000 | +0.000 | **-0.039** |
| DFDCP | jpeg q40 | +0.000 | +0.000 | +0.000 | **-0.248** |
| DFDCP | resize 0.60 | +0.000 | +0.000 | +0.000 | **-0.187** |

Cells are the CHANGE in FPR_real from that dataset's own baseline, so a smaller number is a view that resists the processing better.

## Verdict

**Mean ΔFPR_real -0.130, mean ΔTPR -0.493** across 4 held-out conditions.

**False positives are controlled, but by missing fakes.** FPR moves -0.130 while TPR falls -0.493. Lowering false positives by losing sensitivity is not an improvement, and reporting the FPR alone would have looked like one.

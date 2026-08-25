# READOFF — the six questions §27 allows (and nothing else)

Epoch 7; sources ['FaceForensics++', 'Celeb-DF-v2', 'Celeb-DF-v3', 'DFDC', 'DFDCP', 'UADFV', 'Deepfake-Eval-2024'].

## 1. Is V1 competitive enough to continue?

V1 video AUROC ranges 0.6298–0.9972.

| source | V1 | reproduced DiCoME | delta |
|---|---:|---:|---:|
| Celeb-DF-v2 | 0.9248 | 0.9550 | -0.0302 |
| Celeb-DF-v3 | 0.9216 | 0.8440 | +0.0776 |
| DFDC | 0.8468 | 0.8770 | -0.0302 |
| DFDCP | 0.8970 | 0.8480 | +0.0490 |
| Deepfake-Eval-2024 | 0.6298 | 0.6850 | -0.0552 |

(Reproduced-DiCoME provenance: Phase-1 pilot P0-DS, MULTISOURCE_FINDINGS.md, video level, frozen FF++ threshold 0.5110, run 2026-08-15 — a different run from this one.)

## 2. Which branch contributes the most unique rescue?

- `ref`: 1/7 sources outside §22's noise floor (range -0.0074…+0.0118)
- `proc`: 0/7 sources outside the noise floor (range -0.0075…+0.0045)

Neither specialist clears the noise floor on a majority of sources, so **no branch shows nontrivial rescue over the anchor** on this run. §23 lists that as a V1 success criterion, so it is not met.

## 3. Which branch is the largest source of harm or domain shift?

- `ref`: flagged on 6/6 sources — Celeb-DF-v2, Celeb-DF-v3, DFDC, DFDCP, Deepfake-Eval-2024, UADFV
- `proc`: flagged on 5/6 sources — Celeb-DF-v2, Celeb-DF-v3, DFDC, DFDCP, Deepfake-Eval-2024
- `sem`: flagged on 5/6 sources — Celeb-DF-v2, Celeb-DF-v3, DFDC, DFDCP, Deepfake-Eval-2024

The §20 audit compares each branch's residual on FF++ reals vs OOD reals against reals vs fakes within a source; a flagged branch responds to provenance at least as much as to manipulation.

## 4. Does applicability improve over plain DS?

**0 of 7 sources** clear §22's noise floor (range -0.0056…+0.0072).

§13's named approximation says three-way Shapley-style marginal utility is pulled in **only if** this diagnostic shows the applicability layer failing to beat plain DS. On this run it fails, so that escalation is now earned.

## 5. Do V/C/A improve error detection and selective prediction?

Error-detection AUROC **0.8118**; at a 10% budget selective risk falls from 0.1524 to 0.1232 (+0.0291).

Coefficients: `V` +2.046, `C` +3.614, `A` +0.952, `fused_margin` -3.072, `bias` -1.838

See V1_RELIABILITY.md for the sign check against §17's predictions.

## 6. The top two changes to run next

Derived from the answers above, and deliberately two (§27: do not propose ten modules).

1. **Address the reference branch's domain sensitivity.** It is flagged by the §20 audit on nearly every source, and the mechanism is visible in the design: `P_R` is fit on FF++ reals only, so its residual measures "unlike FF++ authentic" as much as "unlike authentic". §24's iterate order names additional diverse real-face data for exactly this. Note this does NOT condemn the reference transformation itself — §4.2's control shows `P_R` beating the capacity-matched direct probe on most sources.
2. **Decide the process slot on evidence.** `proc` contributes nothing outside the noise floor on any source and is flagged by the audit on nearly all of them. §5's gate says a specialist that shows neither conditional information nor audit compliance should be removed rather than kept for richness — so either replace it (§24 lists diffusion-noise consistency or an earned frequency specialist) or drop the slot.

Both are subject to §22: any promotion or removal needs a second seed first.

# SBI training comparison — arms A / B / C

> **Superseded (2026-09-14).** These numbers used an undeclared newest-checkpoint rule and averaged
> processing effects across conditions. Use `tbiom/reports/CORRECTED_V2.md` and the corrections in
> `tbiom/docs/decision-2026-09-13.md`.


Matched arms, one variable each, all reading the same eligible source pool. `tau` frozen per run at its own FF++ val EER; probabilities are the DS replay, matching every other table in the writeup.

| arm | mean OOD AUROC | mean FPR_real |
|---|---:|---:|
| R0  (historical arm B) | **0.8768** | 0.212 |
| A   common pipeline | **0.8667** | 0.184 |
| B   + processing | **0.8669** | 0.094 |
| C   + SBI 50% | **0.8138** | 0.098 |

## The differences, in the order they should be read

| comparison | ΔAUROC | ΔFPR_real | what it measures |
|---|---:|---:|---|
| A − R0 | **-0.0101** | -0.028 | **loader, not treatment** |
| B − A | +0.0002 | -0.090 | processing exposure alone |
| C − B | **-0.0531** | +0.003 | spatial synthesis alone |
| C − A | **-0.0529** | -0.087 | both together (confounded) |

Bold marks |ΔAUROC| ≥ 0.01, the band a single seed can produce in this project. Smaller differences are reported and not claimed; the seed protocol (42/1337/7) applies to anything that survives.

## Per dataset

| dataset | R0 | A | B | C |
|---|---|---|---|---|
| CDFv2 | 0.9474 | 0.9337 | 0.9146 | 0.9300 |
| CDFv3 | 0.8871 | 0.8811 | 0.8728 | 0.6272 |
| DFD | 0.9454 | 0.9373 | 0.9388 | 0.9027 |
| DFDC | 0.8906 | 0.8794 | 0.8872 | 0.8667 |
| DFDCP | 0.9038 | 0.8869 | 0.9063 | 0.8479 |
| DFEval24 | 0.6866 | 0.6818 | 0.6817 | 0.7085 |

## Verdict

**A − R0 = -0.0101.** **Outside the seed band. The wrapper changes the result on its own**, so every later comparison is against A and never against R0, and the gap is a preprocessing difference rather than evidence about any treatment.

**C − B = -0.0531** is the SBI result proper. It clears the seed band, so spatial synthesis is doing something beyond processing exposure and deserves the three-seed confirmation.

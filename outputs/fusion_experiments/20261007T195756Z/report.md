# CROSS-EXAM fusion experiments — run 20261007T195756Z

Brief: `tbiom/docs/CROSS_EXAM_Fusion_Experiments_Claude_Code_Instructions_2026-10-07.md (revision 3)`.

Clock start 2026-10-07T19:57:56+00:00, deadline 2026-10-10T19:57:56+00:00, elapsed **7.0 h** at writing.

## 1. Baseline reproduction

| item | value |
|---|---|
| B0 identity | PE-Core-L/336, LayerNorm-only appearance adaptation; OM and T as deployed |
| recorded development macro 0.9329 | **observed 0.9329**, delta 0.0000 |
| recorded ten-set mean 0.9212 | **observed 0.9212**, delta 0.0000 |
| identity resolved | **True** |
| J2b verdict | **None** → accepted baseline component: **False** |

The J2b artifact is absent, so joint IBDC is not an accepted baseline component and B0 stands.

## 2. Phase 0 algebra audit — it invalidated prior project results

A binary opinion needs the three masses to sum to one and stay non-negative, so holding the
probability fixed while multiplying vacuity is valid only while `u <= 2*min(p, 1-p)`. The
project's earlier vacuity scaling clipped at 1.0 and never checked that bound:

| view | multiplier | invalid videos | worst belief mass |
|---|---:|---:|---:|
| APP | 2.0 | 2795 / 5814 | -0.1010 |
| APP | 4.0 | 4549 / 5814 | -0.3439 |
| OM | 2.0 | 5272 / 5814 | -0.2456 |
| OM | 4.0 | 5812 / 5814 | -0.3740 |
| T | 2.0 | 5814 / 5814 | -0.2115 |
| T | 4.0 | 5814 / 5814 | -0.3663 |

Evidence discounting is clean: identity at r=1 and vacuity at r=0 are exact, and the closed
form matches the implementation to 3.3e-16. Three earlier results that used
vacuity scaling are withdrawn; everything using discounting stands.

## 3. Data access and reservation

- development sources 9; reserved from all fitting: **['DFDCPval', 'DFEval24val']**
- development videos 5814 → **4914** after reservation
- all ten benchmarks and DF40 are **previously exposed**; DeepfakeEval24 and DFDC are targeted
  stress tests, not untouched confirmation

## 4. Phase 1 — global discount

| gamma | development macro, out of fold |
|---:|---:|
| 1.0 | 0.9513 |
| 0.5 | 0.9516 |
| 0.25 | 0.9520 |
| 0.125 | 0.9522 |

B0-matched 0.9513; **B1 best gamma 0.125 → 0.9522**, delta **+0.0010**. Monotone in gamma: the smaller the
global discount the better, which is the region the `max(w)=1` constraint forbids.

## 5. Phase 2 — the hard gate FAILED

| head | development macro, out of fold |
|---|---:|
| L | 0.9401 |
| S | 0.9461 |
| F | 0.8580 |
| B0_matched | 0.9513 |
| B1_refit | 0.9522 |

- **F − S = -0.0881**, 97.5% lower bound -0.1761
- **F − B1-refit = -0.0942**

**GATE: FAILED.** Phases 3, 4 and 5 are SKIPPED. B1 is retained as the cheap intervention. The brief forbids tuning F after inspecting this gate.

Capacity was matched to 0.8% — F holds 23,194 trainable parameters against S's 23,386 — so this
is not a bigger head winning or losing. F beat S on INNER validation (0.9625 against 0.9381) and
lost by 0.088 out of fold. That is overfitting a 23k-parameter head to 4,914 videos across 52
source groups, which the brief named in advance as the expectation to test.

## 6. Final evaluation — DeepfakeEval24 first, DFDC second, then the panel

Frozen development threshold 0.4329; gamma 0.125. All configurations were locked
before this ran; nothing was refitted in response to it.

| dataset | B0-matched | B1-refit | delta | genuine FPR at the dev threshold |
|---|---:|---:|---:|---:|
| **DFEval24** | 0.7331 | 0.7345 | +0.0014 | 0.3551 |
| **DFDC** | 0.8605 | 0.8621 | +0.0015 | 0.1806 |
| CDFv2 | 0.9502 | 0.9438 | -0.0064 | 0.1798 |
| CDFv3 | 0.9543 | 0.9569 | +0.0026 | 0.1798 |
| DFD | 0.9405 | 0.9399 | -0.0005 | 0.1543 |
| DFDCP | 0.8991 | 0.8980 | -0.0011 | 0.1391 |
| FFIW | 0.9665 | 0.9690 | +0.0025 | 0.0807 |
| KoDF | 0.9399 | 0.9426 | +0.0026 | 0.0397 |
| IDForge | 0.9885 | 0.9907 | +0.0022 | 0.0713 |
| DFDM | 0.9858 | 0.9854 | -0.0004 | 0.2079 |

**DeepfakeEval24 remains the weak case at 0.734**, and B1 moves it by +0.0014 — nothing. DFDC
0.862, +0.0016. The global discount is real but tiny and does not address the difficult datasets.

DeepfakeEval24 and DFDC are PREVIOUSLY EXPOSED. This is targeted evaluation on known-difficult datasets, not untouched confirmation. L, S and F are reported on development out-of-fold only: their heads require per-view FEATURES, and the off-manifold cache does not cover DFDC, CDFv2, CDFv3 or KoDF, so a panel row for them would be computed on a different set of datasets than B0/B1.

## 7. The six-output bottleneck hypothesis: **UNSUPPORTED**

Retaining per-view features and giving the fusion head access to them made out-of-fold
performance decisively **worse** at matched capacity. Bounds on that claim: one feature
dimension (d=8), one pooling rule, one backbone, and a development pool of 4,914 videos across
52 source groups — small for a 23k-parameter head. This is evidence against the hypothesis as
tested, not proof that no retained-information scheme could work.

## 8. Negative results and skipped work

| item | status | reason |
|---|---|---|
| Phase 3 (A/I interactions) | **skipped** | Phase 2 gate failed; the brief forbids proceeding |
| Phase 4 (intervention subspaces) | **skipped** | same |
| Phase 5 (group-aware loss) | **skipped** | same |
| L/S/F on the panel | **not run** | the off-manifold cache lacks DFDC, CDFv2, CDFv3 and KoDF, so a panel row would cover different datasets than B0/B1 |
| prior vacuity-calibration results | **withdrawn** | invalid opinion masses, section 2 |

## 9. Recommended next experiment

Not more fusion capacity. Two facts survived: a global discount helps slightly in a region the
weight constraint forbids, and DeepfakeEval24 sits at 0.73 while every other panel set is above
0.86. The next experiment should drop `max(w)=1` and fit an unconstrained per-view scale on the
development pool — three parameters, no new heads — and report whether the DeepfakeEval24 gap
moves at all. If three free scalars cannot move it, the problem is not in the fusion
parameterisation and further fusion work is unwarranted.


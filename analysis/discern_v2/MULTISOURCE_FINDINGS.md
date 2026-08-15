# Cross-source response matrix — and what it does to the P1b recommendation

All eight operators across all seven OOD sources, video level, frozen FFpp threshold
(0.5110). Run 2026-08-15 once the export matrix reached 8 pilots × 8 sources.

Reproduce: `python analysis/discern_v2/response_matrix.py` (after `a1_complementarity.py`).

## Standalone AUROC

| operator | DF40 | CDFv3 | CDFv2 | DFEval24 | DFDC | DFDCP | DFD | **mean** |
|---|---|---|---|---|---|---|---|---|
| P0-DS | 0.837 | 0.844 | 0.955 | 0.685 | 0.877 | 0.848 | 0.936 | 0.8545 |
| P1a | 0.842 | **0.912** | 0.931 | 0.647 | 0.870 | 0.875 | 0.923 | 0.8570 |
| P1b | **0.858** | 0.892 | 0.939 | 0.648 | 0.858 | 0.873 | 0.927 | 0.8565 |
| P1c | 0.823 | 0.889 | 0.936 | 0.653 | 0.867 | 0.860 | 0.924 | 0.8503 |
| P1d | 0.835 | 0.884 | 0.945 | 0.661 | 0.875 | **0.881** | 0.932 | 0.8589 |
| P2a | 0.844 | 0.897 | 0.949 | 0.678 | **0.877** | 0.874 | 0.928 | **0.8639** |
| P3a | 0.837 | 0.865 | **0.959** | 0.659 | 0.872 | 0.857 | **0.944** | 0.8562 |
| P4 | 0.831 | 0.883 | 0.937 | 0.653 | 0.866 | 0.860 | 0.920 | 0.8501 |

## Error overlap with P0 (Jaccard — lower means more complementary)

| operator | mean overlap |
|---|---|
| **P1c** | **0.500** |
| P1b | 0.554 |
| P1a | 0.556 |
| P4 | 0.566 |
| P2a | 0.577 |
| P1d | 0.600 |
| P3a | 0.655 |

## This weakens the P1b recommendation, and the weakening should be recorded

P1b was selected on DF40 evidence: it wins all eight DF40 inverted rows, has the best DF40
standalone AUROC (0.858 vs P0's 0.837), and the largest oracle headroom. CDFv3's single
inverted row agreed. On seven sources that picture does not hold up:

- **P1b is 4th of 8 on mean AUROC** (0.8565), inside a 0.004 band with P1a, P3a and P0
  itself. Its DF40 lead is DF40-specific — on CDFv3 P1a is better (0.912 vs 0.892), on DFDC
  and DFD P1b is *below* the P0 baseline.
- **P2a has the best mean AUROC** (0.8639) and the lowest harm on most sources.
- **P1c is the most complementary operator by error overlap** (mean 0.500 vs P1b's 0.554)
  and posts the highest rescue on five of seven sources — while having the *worst* mean
  AUROC (0.8503). Exactly the "mediocre standalone, complementary where it counts" profile
  the north star says to look for.
- **P1d is the most gateable** (A2b) but the second-most redundant with P0 by error overlap.

**No projector dominates.** Which one "wins" depends entirely on the criterion:

| criterion | winner |
|---|---|
| inverted rows (DF40 + CDFv3) | **P1b** |
| mean standalone AUROC | P2a, then P1d |
| error complementarity with P0 | **P1c** |
| deployment-valid gate recovery (A2b) | **P1d** |

## What I would not do

I would not quietly keep the P1b recommendation as if the multi-source evidence supported
it. It supports P1b only on the inversion criterion — which is the criterion the north star
privileges, but which is also **only measurable on two of the eight sources** (the rest label
every fake as one method). So the strongest argument for P1b rests on the narrowest evidence
base. That is a defensible choice, but it must be stated as such.

## Two items for Umar

1. **Confirm P1b, or switch on a stated criterion.** P1b remains my recommendation, because
   the phase's north star is regions of expertise rather than average AUC, and the inverted
   rows are the only place that is directly observable. But if the paper's claim is going to
   be evaluated on mean cross-source AUROC, P2a or P1d is the better pick and P1b should not
   be defended on numbers that do not support it.

2. **P1c (WAE) is worth reconsidering.** The Phase-1 spec left it unported "unless Track A
   unexpectedly selects it". Track A has now produced exactly that unexpected result on the
   complementarity criterion: lowest error overlap with P0 on every source, highest rescue
   on most, worst standalone AUROC. Porting it is a small job (the interface already exists
   in `projectors.py`; only the WAE objective is missing). Flagging rather than doing it,
   since the spec makes porting conditional on your call.

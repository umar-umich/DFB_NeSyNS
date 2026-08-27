# Anchor decision — P0-DS epoch 1, retrainable

**Decided by UMAR 2026-08-27: retrainability is required.** This closes Step 2's pass condition,
which asked for a selected checkpoint and left the frozen/retrainable trade open.

## What it costs, stated plainly

| | best readout | OOD mean (6 datasets) |
|---|---|---:|
| DiCoME released (frozen) | artifact | 0.8742 |
| **P0-DS epoch 1 (selected)** | **fused** | **0.8633** |

**−0.0109.** That exceeds the spec's §22 significance band, so it is a real cost and not noise —
an earlier version of this comparison put it at 0.0035, but that mean was computed while the
released checkpoint was missing Celeb-DF-v3, the hardest column and the one where it is
strongest. The corrected figure is the one above.

What the cost buys: the frozen checkpoint cannot be retrained, which forecloses the corpus
ablation (Step 9), any recipe work on the remaining unexplained half of the Step-1 gap, and any
later component replacement that touches training. Every one of those stays open with P0-DS.

## Which readout

`fused` at 0.8633 OOD mean, over `artifact` (0.8618) and `semantic` (0.8572). Close, and the
choice is not purely AUROC:

| readout | OOD mean | zero-shot FPR_real | Step-4 selective-risk reduction |
|---|---:|---:|---:|
| fused | **0.8633** | 0.247 | **+14.0%**, 0 of 6 domains worse |
| artifact | 0.8618 | **0.169** | +4.4%, 1 of 6 worse |
| semantic | 0.8572 | 0.321 | +11.9%, 1 of 6 worse |

`fused` wins AUROC and is the only readout whose deferral never makes a domain worse. `artifact`
has markedly better real-side FPR. If the paper's headline is selective prediction, `fused` is
right; if it is operating-point robustness, `artifact` has the better case. **Recorded as an open
sub-choice rather than settled**, because the two disagree and the decision depends on which
claim leads.

Checkpoint:
`/data/umar/Repos/DiCoME/runs/dicome_train/ffpp_reproduce_seed42/checkpoints/dicome-best-epoch=01-val_auroc_video=0.9960.ckpt`
reachable from the fork as `runs/_baseline_dicome/dicome_train/ffpp_reproduce_seed42/...`.

Note epoch 1 was the original selection on saturated FF++ val AUROC. Step 2 re-checked all four
available checkpoints on the health dashboard and found them clustered (FPR_real 0.196–0.246), so
the original pick stands — not because it was well-chosen, but because no better one exists.

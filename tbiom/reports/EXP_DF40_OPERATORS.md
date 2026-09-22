# Combination rules on every DF40 generator

`EXP_FUSION_OPERATORS.md` compares the seven rules on development and the ten test sets. This applies the same
fourteen arms to the 41 DF40 generator sets, so an alternative to Dempster's rule can be chosen on cross-manipulation
evidence too, and not only on the sets the deployed rule was already selected against.

Nothing is trained here: an operator is a closed-form combination of three cached view opinions. The only fitted
quantity is the three-view reliability weight, a grid search on the fitting fold of each split, exactly as in the
report this extends. Every arm sees identical opinions, so the comparison is decided by the operator alone.

Wall clock for the whole study: **0.4 minutes** on CPU, cached opinions, no GPU.

| operator | swap, CDF | swap, FF++ | reenactment, CDF | reenactment, FF++ | **all 41** | Δ vs Dempster-R |
|---|---:|---:|---:|---:|---:|---:|
| Dempster | 0.9679 | 0.9846 | 0.9149 | 0.9609 | **0.9555** | -0.0021 |
| Dempster-R | 0.9731 | 0.9854 | 0.9174 | 0.9607 | **0.9575** | +0.0000 |
| Yager | 0.9728 | 0.9849 | 0.8767 | 0.9385 | **0.9405** | -0.0171 |
| Yager-R | 0.9714 | 0.9832 | 0.9217 | 0.9582 | **0.9571** | -0.0004 |
| PCR6 | 0.9686 | 0.9856 | 0.8936 | 0.9529 | **0.9480** | -0.0095 |
| PCR6-R | 0.9725 | 0.9851 | 0.9166 | 0.9602 | **0.9569** | -0.0006 |
| Murphy | 0.9688 | 0.9847 | 0.8834 | 0.9479 | **0.9437** | -0.0138 |
| Murphy-R | 0.9719 | 0.9847 | 0.9062 | 0.9558 | **0.9527** | -0.0048 |
| cautious | 0.9588 | 0.9770 | 0.9352 | 0.9649 | **0.9581** | +0.0006 |
| cautious-R | 0.9707 | 0.9812 | 0.9296 | 0.9615 | **0.9595** | +0.0020 |
| cumulative | 0.9650 | 0.9826 | 0.9254 | 0.9622 | **0.9575** | -0.0000 |
| cumulative-R | 0.9727 | 0.9839 | 0.9263 | 0.9610 | **0.9596** | +0.0020 |
| averaging | 0.9637 | 0.9806 | 0.9334 | 0.9653 | **0.9597** | +0.0022 |
| averaging-R | 0.9729 | 0.9846 | 0.9267 | 0.9628 | **0.9604** | +0.0028 |

Best alternative to the deployed rule over all 41 generator sets: **averaging-R** (0.9604, +0.0028). On the 22 reenactment sets alone it is **cautious** (0.9501 against Dempster-R's 0.9391).

DF40 played no part in fitting: the reliability weights come from the development folds, and every arm is applied to these sets unchanged. The ten-set and development view of the same fourteen arms is in `EXP_FUSION_OPERATORS.md`; sections 6 and 6a of `DETAILED_RESULTS.md` carry it per test set.

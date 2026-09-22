# Combination rules for the reference triple

OMP_NUM_THREADS=8 python analysis/tbiom/fusion_operators_v2.py

Cached opinions, CPU, the same five source-grouped folds, the same evidence-scaling discount, the same v3.1 objective
as every other report. Only the operator that combines the three views changes.

WHY THIS IS WORTH ASKING. Dempster's rule assumes the sources are independent and normalises the conflicting mass
away. Neither assumption fits this system: Appearance and Off-manifold are two heads on ONE CLIP feature path, so
their evidence overlaps, and the conflict between a temporal view and a spatial view is exactly the signal the
uncertainty safeguard uses. The rules below relax one assumption each.

The operators, all on binomial opinions (b1 = p − u/2 fake mass, b0 = 1 − p − u/2 genuine mass, u vacuity):
  Dempster            conjunctive, conflict normalised away (the deployed operator)
  Yager               conjunctive, conflict returned to vacuity instead of normalised (Dubois–Prade coincides with
                      Yager on a binary frame, so it is not listed twice)
  PCR6                conflicting mass redistributed to the views that caused it, in proportion to their own masses
  Murphy              average the three opinions, then combine the average with itself three times
  cautious (Denœux)   canonical weights w_A = q(Θ)/q(A) per view, combined by MIN — the rule for sources that are NOT
                      independent, which is what a shared feature path makes them; renormalised like Dempster
  cumulative (SL)     evidence summed across views (subjective-logic cumulative belief fusion)
  averaging           evidence averaged across views
  logistic            not an operator: the matched stacking comparator, for reference

Each operator is run twice: with equal trust, and with the same per-view reliability weights fitted by the same grid
search on the same fitting folds (`-R`). Nothing else differs, so a difference is the operator's.

Every number below uses the reference manifest `tbiom/reports/REFERENCE_MANIFEST.md` (three resolved checkpoints, DS-R definition, fold splits). Script `analysis/tbiom/temporal/finish_line.py`.

## 1. Development (selection folds, mean of 5 splits)

| operator | clean v3.1 | swap | reenactment | in-the-wild | JPEG q40 | resize 0.60 | FPR | TPR | error AUROC (fused vacuity) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dempster | 0.8771 | 0.9290 | 0.9099 | 0.7925 | 0.8043 | 0.8576 | 0.110 | 0.729 | 0.789 |
| **Dempster-R** | 0.8808 | 0.9345 | 0.9085 | 0.7993 | 0.8096 | 0.8589 | 0.107 | 0.737 | 0.781 |
| Yager | 0.8727 | 0.9370 | 0.8794 | 0.8017 | 0.7980 | 0.8473 | 0.100 | 0.707 | 0.784 |
| Yager-R | 0.8791 | 0.9309 | 0.9075 | 0.7989 | 0.8139 | 0.8576 | 0.102 | 0.725 | 0.734 |
| PCR6 | 0.8753 | 0.9317 | 0.8953 | 0.7989 | 0.7979 | 0.8532 | 0.091 | 0.680 | 0.695 |
| PCR6-R | 0.8802 | 0.9333 | 0.9067 | 0.8006 | 0.8091 | 0.8574 | 0.078 | 0.675 | 0.718 |
| Murphy | 0.8735 | 0.9327 | 0.8889 | 0.7990 | 0.7953 | 0.8507 | 0.082 | 0.667 | 0.736 |
| Murphy-R | 0.8790 | 0.9333 | 0.9008 | 0.8029 | 0.8067 | 0.8562 | 0.096 | 0.714 | 0.733 |
| cautious | 0.8662 | 0.9112 | 0.9188 | 0.7687 | 0.8006 | 0.8520 | 0.113 | 0.684 | 0.680 |
| cautious-R | 0.8751 | 0.9267 | 0.9136 | 0.7849 | 0.7994 | 0.8554 | 0.113 | 0.753 | 0.730 |
| cumulative | 0.8726 | 0.9215 | 0.9156 | 0.7807 | 0.8026 | 0.8561 | 0.116 | 0.710 | 0.713 |
| cumulative-R | 0.8789 | 0.9315 | 0.9127 | 0.7924 | 0.8095 | 0.8590 | 0.117 | 0.759 | 0.765 |
| averaging | 0.8720 | 0.9190 | 0.9193 | 0.7776 | 0.8042 | 0.8563 | 0.112 | 0.708 | 0.715 |
| averaging-R | 0.8799 | 0.9322 | 0.9133 | 0.7942 | 0.8088 | 0.8592 | 0.109 | 0.748 | 0.763 |
| logistic | 0.8736 | 0.9245 | 0.8997 | 0.7965 | 0.7564 | 0.8542 | 0.057 | 0.648 | nan |

`-R` = the same operator with per-view reliability weights fitted by the same grid search on the same fitting folds. **Dempster-R is the deployed system.**

## 2. Against the deployed operator (paired source-grouped bootstrap, 1000 draws per split)

| operator | mean Δ vs Dempster-R | splits with CI above 0 | splits with CI below 0 |
|---|---:|---:|---:|
| Dempster | -0.0036 | 0/5 | 5/5 |
| Dempster-R | +0.0000 | 0/5 | 0/5 |
| Yager | -0.0082 | 0/5 | 5/5 |
| Yager-R | -0.0017 | 0/5 | 1/5 |
| PCR6 | -0.0055 | 0/5 | 5/5 |
| PCR6-R | -0.0006 | 0/5 | 0/5 |
| Murphy | -0.0072 | 0/5 | 5/5 |
| Murphy-R | -0.0018 | 0/5 | 0/5 |
| cautious | -0.0145 | 0/5 | 5/5 |
| cautious-R | -0.0057 | 0/5 | 5/5 |
| cumulative | -0.0082 | 0/5 | 5/5 |
| cumulative-R | -0.0020 | 0/5 | 0/5 |
| averaging | -0.0088 | 0/5 | 5/5 |
| averaging-R | -0.0009 | 0/5 | 0/5 |
| logistic | -0.0072 | 0/5 | 2/5 |

## 3. Test sets — video AUROC (read after fitting; mean over the 5 fitted splits)

| operator | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | DFEval24 | FFIW | KoDF | IDForge | DFDM | CDFv3 reenactment | KoDF first-order motion | CDFv2 jpeg40 | CDFv2 rs06 | DFDCP jpeg40 | DFDCP rs06 | 10-set mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dempster | 0.9515 | 0.9596 | 0.9249 | 0.8521 | 0.8588 | 0.7223 | 0.9189 | 0.9558 | 0.9593 | 0.9944 | 0.9640 | 0.9799 | 0.8608 | 0.9372 | 0.8128 | 0.8032 | 0.9098 |
| Dempster-R | 0.9598 | 0.9616 | 0.9338 | 0.8626 | 0.8656 | 0.7252 | 0.9282 | 0.9574 | 0.9656 | 0.9956 | 0.9642 | 0.9814 | 0.8695 | 0.9418 | 0.8198 | 0.8084 | 0.9155 |
| Yager | 0.9667 | 0.9458 | 0.9390 | 0.8716 | 0.8683 | 0.7172 | 0.9356 | 0.9448 | 0.9661 | 0.9946 | 0.9401 | 0.9575 | 0.8787 | 0.9480 | 0.8168 | 0.8019 | 0.9150 |
| Yager-R | 0.9616 | 0.9635 | 0.9324 | 0.8529 | 0.8591 | 0.7225 | 0.9223 | 0.9599 | 0.9631 | 0.9954 | 0.9667 | 0.9826 | 0.8807 | 0.9428 | 0.8160 | 0.8009 | 0.9133 |
| PCR6 | 0.9576 | 0.9521 | 0.9299 | 0.8605 | 0.8639 | 0.7189 | 0.9273 | 0.9510 | 0.9613 | 0.9941 | 0.9521 | 0.9694 | 0.8618 | 0.9416 | 0.8133 | 0.8009 | 0.9117 |
| PCR6-R | 0.9607 | 0.9613 | 0.9334 | 0.8594 | 0.8633 | 0.7241 | 0.9272 | 0.9577 | 0.9648 | 0.9954 | 0.9638 | 0.9802 | 0.8700 | 0.9427 | 0.8191 | 0.8052 | 0.9147 |
| Murphy | 0.9594 | 0.9481 | 0.9306 | 0.8643 | 0.8642 | 0.7195 | 0.9297 | 0.9473 | 0.9617 | 0.9937 | 0.9462 | 0.9619 | 0.8624 | 0.9427 | 0.8129 | 0.8010 | 0.9118 |
| Murphy-R | 0.9626 | 0.9577 | 0.9329 | 0.8622 | 0.8641 | 0.7230 | 0.9286 | 0.9545 | 0.9644 | 0.9950 | 0.9582 | 0.9753 | 0.8723 | 0.9436 | 0.8181 | 0.8028 | 0.9145 |
| cautious | 0.9236 | 0.9607 | 0.8972 | 0.8152 | 0.8406 | 0.7104 | 0.8925 | 0.9612 | 0.9459 | 0.9930 | 0.9733 | 0.9872 | 0.8355 | 0.9098 | 0.8025 | 0.7947 | 0.8940 |
| cautious-R | 0.9481 | 0.9622 | 0.9317 | 0.8497 | 0.8574 | 0.7184 | 0.9187 | 0.9630 | 0.9627 | 0.9960 | 0.9690 | 0.9843 | 0.8310 | 0.9254 | 0.8037 | 0.8060 | 0.9108 |
| cumulative | 0.9399 | 0.9608 | 0.9159 | 0.8335 | 0.8493 | 0.7176 | 0.9077 | 0.9583 | 0.9540 | 0.9941 | 0.9686 | 0.9843 | 0.8514 | 0.9272 | 0.8101 | 0.7991 | 0.9031 |
| cumulative-R | 0.9567 | 0.9639 | 0.9340 | 0.8546 | 0.8612 | 0.7233 | 0.9224 | 0.9607 | 0.9643 | 0.9959 | 0.9685 | 0.9845 | 0.8644 | 0.9367 | 0.8160 | 0.8072 | 0.9137 |
| averaging | 0.9356 | 0.9623 | 0.9132 | 0.8293 | 0.8480 | 0.7170 | 0.9033 | 0.9615 | 0.9528 | 0.9941 | 0.9722 | 0.9864 | 0.8435 | 0.9203 | 0.8054 | 0.7985 | 0.9017 |
| averaging-R | 0.9542 | 0.9632 | 0.9330 | 0.8578 | 0.8642 | 0.7236 | 0.9241 | 0.9602 | 0.9650 | 0.9960 | 0.9680 | 0.9842 | 0.8581 | 0.9340 | 0.8153 | 0.8102 | 0.9141 |
| logistic | 0.9697 | 0.9675 | 0.9300 | 0.8602 | 0.8557 | 0.6865 | 0.9262 | 0.9630 | 0.9622 | 0.9963 | 0.9709 | 0.9792 | 0.7199 | 0.9470 | 0.7299 | 0.7935 | 0.9117 |

Reliability weights per operator (split 0): Dempster-R (0.25, 0.25, 1.0); Yager-R (0.25, 0.25, 1.0); PCR6-R (0.25, 0.25, 1.0); Murphy-R (0.25, 0.25, 1.0); cautious-R (0.25, 0.0, 1.0); cumulative-R (0.25, 0.25, 1.0); averaging-R (0.25, 0.5, 1.0).

One detector seed and one temporal-branch seed. CDFv2 and DeepfakeEval2024 informed development. FaceShifter is not scored here.

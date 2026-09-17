# Experiment 6 — uncertainty safeguard for the deployed logistic fusion (`vjepa2_lora_seed42` + OM + FP)

The deployed fusion is logistic stacking, so DS vacuity is **not** its uncertainty and is not attached to its output. The signals below are compared as ERROR RANKERS for the logistic decision (score ≥ τ, τ = EER on selection-fold FF++ val). Higher error-AUROC and lower AURC / E-AURC are better; risk@c = error rate on the c most confident fraction. Clean and processed conditions are separate; processed development uses the same selection-fold videos. Mean over 5 grouped splits. Goal: useful error identification — not making vacuity beat margin.

## clean dev

accuracy 0.732 · FPR 0.063 · FNR 0.364 · videos 2903 · confident false negatives (FN with margin above the median correct-genuine margin): 0.12 of FNs

| signal | error AUROC | AURC | E-AURC | risk@90% | risk@80% | FN-vs-TN AUROC (genuine calls) |
|---|---:|---:|---:|---:|---:|---:|
| margin | 0.744 | 0.1388 | 0.0986 | 0.241 | 0.210 | 0.796 |
| u_T | 0.658 | 0.1654 | 0.1253 | 0.253 | 0.236 | 0.583 |
| u_OM | 0.707 | 0.1337 | 0.0936 | 0.251 | 0.230 | 0.611 |
| u_FP | 0.669 | 0.1563 | 0.1162 | 0.250 | 0.231 | 0.627 |
| u_max | 0.694 | 0.1397 | 0.0996 | 0.252 | 0.233 | 0.634 |
| u_mean | 0.726 | 0.1285 | 0.0883 | 0.245 | 0.220 | 0.662 |
| disagree_std | 0.767 | 0.1185 | 0.0784 | 0.241 | 0.204 | 0.771 |
| disagree_range | 0.769 | 0.1176 | 0.0775 | 0.244 | 0.202 | 0.772 |
| **conflict_max** | 0.769 | 0.1160 | 0.0759 | 0.243 | 0.204 | 0.782 |

## jpeg40 dev

accuracy 0.452 · FPR 0.004 · FNR 0.804 · videos 2903 · confident false negatives (FN with margin above the median correct-genuine margin): 0.15 of FNs

| signal | error AUROC | AURC | E-AURC | risk@90% | risk@80% | FN-vs-TN AUROC (genuine calls) |
|---|---:|---:|---:|---:|---:|---:|
| margin | 0.663 | 0.4074 | 0.2173 | 0.545 | 0.524 | 0.778 |
| u_T | 0.675 | 0.3902 | 0.2001 | 0.532 | 0.512 | 0.738 |
| **u_OM** | 0.711 | 0.3631 | 0.1730 | 0.523 | 0.497 | 0.690 |
| u_FP | 0.615 | 0.4231 | 0.2330 | 0.543 | 0.529 | 0.612 |
| u_max | 0.690 | 0.3739 | 0.1838 | 0.531 | 0.509 | 0.691 |
| u_mean | 0.698 | 0.3619 | 0.1718 | 0.531 | 0.513 | 0.707 |
| disagree_std | 0.657 | 0.4104 | 0.2203 | 0.542 | 0.517 | 0.729 |
| disagree_range | 0.662 | 0.4046 | 0.2145 | 0.540 | 0.516 | 0.734 |
| conflict_max | 0.639 | 0.4242 | 0.2341 | 0.540 | 0.515 | 0.716 |

## rs06 dev

accuracy 0.729 · FPR 0.167 · FNR 0.319 · videos 2903 · confident false negatives (FN with margin above the median correct-genuine margin): 0.09 of FNs

| signal | error AUROC | AURC | E-AURC | risk@90% | risk@80% | FN-vs-TN AUROC (genuine calls) |
|---|---:|---:|---:|---:|---:|---:|
| margin | 0.774 | 0.1162 | 0.0751 | 0.248 | 0.216 | 0.763 |
| u_T | 0.560 | 0.2081 | 0.1669 | 0.271 | 0.269 | 0.277 |
| u_OM | 0.671 | 0.1520 | 0.1109 | 0.255 | 0.241 | 0.603 |
| u_FP | 0.667 | 0.1598 | 0.1186 | 0.253 | 0.236 | 0.614 |
| u_max | 0.656 | 0.1592 | 0.1180 | 0.258 | 0.244 | 0.574 |
| u_mean | 0.672 | 0.1531 | 0.1119 | 0.257 | 0.237 | 0.547 |
| disagree_std | 0.799 | 0.1096 | 0.0684 | 0.240 | 0.197 | 0.734 |
| disagree_range | 0.800 | 0.1089 | 0.0677 | 0.242 | 0.197 | 0.738 |
| **conflict_max** | 0.801 | 0.1053 | 0.0642 | 0.242 | 0.199 | 0.739 |

## clean tests (10 sets pooled)

accuracy 0.771 · FPR 0.052 · FNR 0.313 · videos 27974 · confident false negatives (FN with margin above the median correct-genuine margin): 0.08 of FNs

| signal | error AUROC | AURC | E-AURC | risk@90% | risk@80% | FN-vs-TN AUROC (genuine calls) |
|---|---:|---:|---:|---:|---:|---:|
| **margin** | 0.773 | 0.1009 | 0.0717 | 0.202 | 0.167 | 0.835 |
| u_T | 0.645 | 0.1432 | 0.1140 | 0.221 | 0.207 | 0.636 |
| u_OM | 0.761 | 0.0944 | 0.0652 | 0.204 | 0.177 | 0.629 |
| u_FP | 0.675 | 0.1323 | 0.1032 | 0.208 | 0.191 | 0.646 |
| u_max | 0.737 | 0.1017 | 0.0726 | 0.210 | 0.186 | 0.673 |
| u_mean | 0.751 | 0.0980 | 0.0689 | 0.204 | 0.179 | 0.686 |
| disagree_std | 0.767 | 0.0917 | 0.0625 | 0.214 | 0.178 | 0.801 |
| disagree_range | 0.770 | 0.0909 | 0.0618 | 0.212 | 0.177 | 0.802 |
| conflict_max | 0.762 | 0.0963 | 0.0671 | 0.213 | 0.178 | 0.798 |

## CDFv2 clean

accuracy 0.714 · FPR 0.024 · FNR 0.424 · videos 518 · confident false negatives (FN with margin above the median correct-genuine margin): 0.01 of FNs

| signal | error AUROC | AURC | E-AURC | risk@90% | risk@80% | FN-vs-TN AUROC (genuine calls) |
|---|---:|---:|---:|---:|---:|---:|
| **margin** | 0.774 | 0.1211 | 0.0730 | 0.266 | 0.240 | 0.931 |
| u_T | 0.557 | 0.2249 | 0.1768 | 0.294 | 0.295 | 0.693 |
| u_OM | 0.734 | 0.1472 | 0.0991 | 0.260 | 0.234 | 0.543 |
| u_FP | 0.648 | 0.1821 | 0.1340 | 0.276 | 0.263 | 0.753 |
| u_max | 0.659 | 0.1729 | 0.1248 | 0.287 | 0.265 | 0.607 |
| u_mean | 0.708 | 0.1456 | 0.0975 | 0.278 | 0.269 | 0.708 |
| disagree_std | 0.683 | 0.1463 | 0.0982 | 0.302 | 0.293 | 0.905 |
| disagree_range | 0.688 | 0.1447 | 0.0966 | 0.297 | 0.291 | 0.905 |
| conflict_max | 0.691 | 0.1404 | 0.0923 | 0.296 | 0.293 | 0.920 |

## CDFv2 jpeg40

accuracy 0.412 · FPR 0.002 · FNR 0.895 · videos 518 · confident false negatives (FN with margin above the median correct-genuine margin): 0.08 of FNs

| signal | error AUROC | AURC | E-AURC | risk@90% | risk@80% | FN-vs-TN AUROC (genuine calls) |
|---|---:|---:|---:|---:|---:|---:|
| margin | 0.748 | 0.3767 | 0.1530 | 0.587 | 0.560 | 0.869 |
| u_T | 0.638 | 0.4634 | 0.2397 | 0.585 | 0.566 | 0.704 |
| **u_OM** | 0.764 | 0.3889 | 0.1652 | 0.553 | 0.519 | 0.735 |
| u_FP | 0.658 | 0.4632 | 0.2395 | 0.571 | 0.547 | 0.714 |
| u_max | 0.745 | 0.3941 | 0.1704 | 0.566 | 0.529 | 0.752 |
| u_mean | 0.762 | 0.3647 | 0.1410 | 0.573 | 0.532 | 0.795 |
| disagree_std | 0.724 | 0.4098 | 0.1861 | 0.605 | 0.567 | 0.850 |
| disagree_range | 0.728 | 0.3999 | 0.1762 | 0.605 | 0.567 | 0.854 |
| conflict_max | 0.734 | 0.3741 | 0.1504 | 0.607 | 0.567 | 0.862 |

## CDFv2 rs06

accuracy 0.807 · FPR 0.076 · FNR 0.254 · videos 518 · confident false negatives (FN with margin above the median correct-genuine margin): 0.11 of FNs

| signal | error AUROC | AURC | E-AURC | risk@90% | risk@80% | FN-vs-TN AUROC (genuine calls) |
|---|---:|---:|---:|---:|---:|---:|
| **margin** | 0.774 | 0.0839 | 0.0635 | 0.159 | 0.121 | 0.797 |
| u_T | 0.576 | 0.1482 | 0.1278 | 0.182 | 0.186 | 0.488 |
| u_OM | 0.759 | 0.0776 | 0.0572 | 0.169 | 0.156 | 0.694 |
| u_FP | 0.594 | 0.1495 | 0.1291 | 0.192 | 0.178 | 0.698 |
| u_max | 0.688 | 0.1002 | 0.0798 | 0.177 | 0.165 | 0.640 |
| u_mean | 0.699 | 0.0984 | 0.0780 | 0.171 | 0.164 | 0.665 |
| disagree_std | 0.648 | 0.1162 | 0.0958 | 0.188 | 0.183 | 0.645 |
| disagree_range | 0.652 | 0.1151 | 0.0947 | 0.184 | 0.182 | 0.654 |
| conflict_max | 0.659 | 0.1074 | 0.0870 | 0.185 | 0.184 | 0.687 |

## DFDCP clean

accuracy 0.724 · FPR 0.103 · FNR 0.369 · videos 654 · confident false negatives (FN with margin above the median correct-genuine margin): 0.06 of FNs

| signal | error AUROC | AURC | E-AURC | risk@90% | risk@80% | FN-vs-TN AUROC (genuine calls) |
|---|---:|---:|---:|---:|---:|---:|
| margin | 0.718 | 0.1432 | 0.1004 | 0.249 | 0.231 | 0.804 |
| u_T | 0.530 | 0.2336 | 0.1907 | 0.287 | 0.282 | 0.687 |
| u_OM | 0.716 | 0.1404 | 0.0976 | 0.243 | 0.221 | 0.547 |
| u_FP | 0.616 | 0.2068 | 0.1639 | 0.258 | 0.240 | 0.476 |
| u_max | 0.682 | 0.1524 | 0.1095 | 0.265 | 0.240 | 0.612 |
| u_mean | 0.662 | 0.1615 | 0.1187 | 0.258 | 0.244 | 0.543 |
| disagree_std | 0.719 | 0.1341 | 0.0912 | 0.278 | 0.248 | 0.816 |
| disagree_range | 0.736 | 0.1317 | 0.0889 | 0.256 | 0.239 | 0.817 |
| **conflict_max** | 0.740 | 0.1282 | 0.0853 | 0.254 | 0.239 | 0.811 |

## DFDCP jpeg40

accuracy 0.485 · FPR 0.001 · FNR 0.793 · videos 654 · confident false negatives (FN with margin above the median correct-genuine margin): 0.21 of FNs

| signal | error AUROC | AURC | E-AURC | risk@90% | risk@80% | FN-vs-TN AUROC (genuine calls) |
|---|---:|---:|---:|---:|---:|---:|
| margin | 0.611 | 0.4341 | 0.2692 | 0.521 | 0.511 | 0.752 |
| u_T | 0.570 | 0.4106 | 0.2457 | 0.535 | 0.535 | 0.685 |
| **u_OM** | 0.749 | 0.3010 | 0.1360 | 0.495 | 0.461 | 0.676 |
| u_FP | 0.638 | 0.4170 | 0.2521 | 0.494 | 0.486 | 0.607 |
| u_max | 0.718 | 0.3168 | 0.1519 | 0.493 | 0.470 | 0.669 |
| u_mean | 0.711 | 0.3282 | 0.1632 | 0.496 | 0.468 | 0.663 |
| disagree_std | 0.675 | 0.3404 | 0.1755 | 0.516 | 0.504 | 0.815 |
| disagree_range | 0.687 | 0.3326 | 0.1677 | 0.505 | 0.497 | 0.817 |
| conflict_max | 0.681 | 0.3379 | 0.1729 | 0.502 | 0.493 | 0.807 |

## DFDCP rs06

accuracy 0.729 · FPR 0.242 · FNR 0.287 · videos 654 · confident false negatives (FN with margin above the median correct-genuine margin): 0.16 of FNs

| signal | error AUROC | AURC | E-AURC | risk@90% | risk@80% | FN-vs-TN AUROC (genuine calls) |
|---|---:|---:|---:|---:|---:|---:|
| margin | 0.705 | 0.1497 | 0.1088 | 0.249 | 0.226 | 0.706 |
| u_T | 0.637 | 0.1759 | 0.1350 | 0.260 | 0.242 | 0.613 |
| u_OM | 0.680 | 0.1513 | 0.1104 | 0.252 | 0.231 | 0.578 |
| u_FP | 0.619 | 0.2047 | 0.1638 | 0.254 | 0.247 | 0.480 |
| u_max | 0.672 | 0.1546 | 0.1137 | 0.249 | 0.230 | 0.603 |
| u_mean | 0.683 | 0.1510 | 0.1101 | 0.252 | 0.225 | 0.558 |
| disagree_std | 0.729 | 0.1382 | 0.0973 | 0.242 | 0.219 | 0.688 |
| disagree_range | 0.733 | 0.1374 | 0.0965 | 0.242 | 0.219 | 0.689 |
| **conflict_max** | 0.737 | 0.1349 | 0.0940 | 0.241 | 0.221 | 0.689 |

Error rate at full coverage equals risk@100%; a signal no better than chance has error AUROC ≈ 0.5. Test rows are read after all fitting and use each split's fitted stacker and τ. One detector seed, one temporal-branch seed.

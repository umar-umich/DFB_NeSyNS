# Experiment 6b — uncertainty safeguard for the deployed DS-R system (`vjepa2_lora_seed42` + APP + OM)

Deployed fusion = conflict-aware Dempster–Shafer with reliability discounting, so the fused Dirichlet vacuity **is** the system's uncertainty. Signals are scored as error rankers for the decision p_f ≥ τ (τ = EER on selection-fold FF++ val). DS implementation checked against DISCERN's operator (max |Δp| 0.0e+00). Mean over 5 source-grouped splits; clean and processed separate; test rows read after fitting. Script `analysis/tbiom/temporal/dsr_system.py`.

Reliability weights per split (T, APP, OM): (0.25, 0.25, 1.0); (0.25, 0.25, 1.0); (0.25, 0.25, 1.0); (0.25, 0.25, 1.0); (0.25, 0.25, 1.0)

## clean dev

AUROC 0.905 · accuracy 0.779 · FPR 0.099 · FNR 0.278 · confident FN share (margin) 0.13 · (fused vacuity) 0.14

| signal | error AUROC | AURC | E-AURC | risk@90% | risk@80% | FN-vs-TN AUROC |
|---|---:|---:|---:|---:|---:|---:|
| margin | 0.737 | 0.1090 | 0.0821 | 0.194 | 0.167 | 0.776 |
| **u_fused** | 0.798 | 0.0790 | 0.0522 | 0.189 | 0.157 | 0.719 |
| conflict_fused | 0.745 | 0.1008 | 0.0739 | 0.188 | 0.160 | 0.737 |
| u_T | 0.708 | 0.1096 | 0.0827 | 0.204 | 0.183 | 0.649 |
| u_APP | 0.663 | 0.1186 | 0.0917 | 0.216 | 0.207 | 0.433 |
| u_OM | 0.657 | 0.1209 | 0.0940 | 0.220 | 0.207 | 0.566 |
| disagree_range | 0.710 | 0.1323 | 0.1054 | 0.195 | 0.166 | 0.686 |
| conflict_pair_max | 0.733 | 0.1039 | 0.0771 | 0.192 | 0.166 | 0.734 |

## jpeg40 dev

AUROC 0.832 · accuracy 0.483 · FPR 0.014 · FNR 0.754 · confident FN share (margin) 0.14 · (fused vacuity) 0.13

| signal | error AUROC | AURC | E-AURC | risk@90% | risk@80% | FN-vs-TN AUROC |
|---|---:|---:|---:|---:|---:|---:|
| margin | 0.662 | 0.3628 | 0.1953 | 0.514 | 0.498 | 0.789 |
| **u_fused** | 0.731 | 0.3185 | 0.1510 | 0.493 | 0.467 | 0.780 |
| conflict_fused | 0.664 | 0.3785 | 0.2110 | 0.500 | 0.472 | 0.716 |
| u_T | 0.657 | 0.3715 | 0.2040 | 0.505 | 0.487 | 0.738 |
| u_APP | 0.581 | 0.4091 | 0.2416 | 0.515 | 0.505 | 0.453 |
| u_OM | 0.668 | 0.3593 | 0.1918 | 0.503 | 0.482 | 0.679 |
| disagree_range | 0.646 | 0.3888 | 0.2213 | 0.505 | 0.478 | 0.669 |
| conflict_pair_max | 0.652 | 0.3831 | 0.2156 | 0.506 | 0.479 | 0.706 |

## rs06 dev

AUROC 0.878 · accuracy 0.796 · FPR 0.272 · FNR 0.171 · confident FN share (margin) 0.12 · (fused vacuity) 0.48

| signal | error AUROC | AURC | E-AURC | risk@90% | risk@80% | FN-vs-TN AUROC |
|---|---:|---:|---:|---:|---:|---:|
| **margin** | 0.773 | 0.0817 | 0.0590 | 0.176 | 0.143 | 0.766 |
| u_fused | 0.748 | 0.0853 | 0.0627 | 0.186 | 0.162 | 0.554 |
| conflict_fused | 0.759 | 0.0862 | 0.0636 | 0.170 | 0.146 | 0.761 |
| u_T | 0.659 | 0.1161 | 0.0935 | 0.196 | 0.184 | 0.405 |
| u_APP | 0.657 | 0.1144 | 0.0918 | 0.196 | 0.184 | 0.459 |
| u_OM | 0.606 | 0.1299 | 0.1073 | 0.205 | 0.201 | 0.511 |
| disagree_range | 0.721 | 0.1202 | 0.0976 | 0.170 | 0.149 | 0.730 |
| conflict_pair_max | 0.752 | 0.0881 | 0.0655 | 0.170 | 0.147 | 0.756 |

## clean tests (10 sets pooled)

AUROC 0.931 · accuracy 0.797 · FPR 0.079 · FNR 0.262 · confident FN share (margin) 0.07 · (fused vacuity) 0.09

| signal | error AUROC | AURC | E-AURC | risk@90% | risk@80% | FN-vs-TN AUROC |
|---|---:|---:|---:|---:|---:|---:|
| margin | 0.785 | 0.0795 | 0.0567 | 0.174 | 0.144 | 0.837 |
| **u_fused** | 0.822 | 0.0631 | 0.0403 | 0.171 | 0.137 | 0.756 |
| conflict_fused | 0.800 | 0.0705 | 0.0476 | 0.173 | 0.136 | 0.804 |
| u_T | 0.665 | 0.1126 | 0.0898 | 0.194 | 0.181 | 0.664 |
| u_APP | 0.717 | 0.0916 | 0.0688 | 0.192 | 0.176 | 0.477 |
| u_OM | 0.732 | 0.0890 | 0.0661 | 0.188 | 0.166 | 0.603 |
| disagree_range | 0.751 | 0.0982 | 0.0753 | 0.183 | 0.148 | 0.737 |
| conflict_pair_max | 0.787 | 0.0733 | 0.0504 | 0.179 | 0.143 | 0.798 |

## CDFv2 clean

AUROC 0.956 · accuracy 0.750 · FPR 0.036 · FNR 0.362 · confident FN share (margin) 0.01 · (fused vacuity) 0.03

| signal | error AUROC | AURC | E-AURC | risk@90% | risk@80% | FN-vs-TN AUROC |
|---|---:|---:|---:|---:|---:|---:|
| margin | 0.789 | 0.0987 | 0.0613 | 0.226 | 0.193 | 0.920 |
| **u_fused** | 0.809 | 0.0861 | 0.0488 | 0.232 | 0.199 | 0.766 |
| conflict_fused | 0.796 | 0.0874 | 0.0500 | 0.248 | 0.203 | 0.928 |
| u_T | 0.494 | 0.2182 | 0.1808 | 0.262 | 0.272 | 0.650 |
| u_APP | 0.602 | 0.1576 | 0.1202 | 0.266 | 0.270 | 0.228 |
| u_OM | 0.754 | 0.1155 | 0.0782 | 0.224 | 0.199 | 0.572 |
| disagree_range | 0.739 | 0.1151 | 0.0777 | 0.260 | 0.231 | 0.880 |
| conflict_pair_max | 0.772 | 0.0934 | 0.0561 | 0.255 | 0.224 | 0.923 |

## CDFv2 jpeg40

AUROC 0.880 · accuracy 0.419 · FPR 0.006 · FNR 0.882 · confident FN share (margin) 0.06 · (fused vacuity) 0.08

| signal | error AUROC | AURC | E-AURC | risk@90% | risk@80% | FN-vs-TN AUROC |
|---|---:|---:|---:|---:|---:|---:|
| margin | 0.753 | 0.3556 | 0.1378 | 0.577 | 0.547 | 0.868 |
| **u_fused** | 0.781 | 0.3436 | 0.1258 | 0.558 | 0.526 | 0.827 |
| conflict_fused | 0.761 | 0.3591 | 0.1413 | 0.568 | 0.530 | 0.842 |
| u_T | 0.610 | 0.4733 | 0.2554 | 0.586 | 0.571 | 0.696 |
| u_APP | 0.373 | 0.6077 | 0.3899 | 0.616 | 0.650 | 0.257 |
| u_OM | 0.756 | 0.3892 | 0.1714 | 0.546 | 0.512 | 0.735 |
| disagree_range | 0.716 | 0.3925 | 0.1747 | 0.577 | 0.539 | 0.781 |
| conflict_pair_max | 0.738 | 0.3700 | 0.1521 | 0.575 | 0.540 | 0.818 |

## CDFv2 rs06

AUROC 0.931 · accuracy 0.842 · FPR 0.137 · FNR 0.168 · confident FN share (margin) 0.14 · (fused vacuity) 0.31

| signal | error AUROC | AURC | E-AURC | risk@90% | risk@80% | FN-vs-TN AUROC |
|---|---:|---:|---:|---:|---:|---:|
| margin | 0.784 | 0.0576 | 0.0441 | 0.130 | 0.100 | 0.778 |
| u_fused | 0.779 | 0.0547 | 0.0412 | 0.132 | 0.120 | 0.648 |
| **conflict_fused** | 0.794 | 0.0516 | 0.0381 | 0.131 | 0.106 | 0.738 |
| u_T | 0.594 | 0.1062 | 0.0927 | 0.149 | 0.150 | 0.522 |
| u_APP | 0.719 | 0.0712 | 0.0578 | 0.145 | 0.128 | 0.438 |
| u_OM | 0.754 | 0.0614 | 0.0479 | 0.141 | 0.126 | 0.711 |
| disagree_range | 0.730 | 0.0754 | 0.0620 | 0.141 | 0.117 | 0.614 |
| conflict_pair_max | 0.778 | 0.0544 | 0.0410 | 0.143 | 0.111 | 0.713 |

## DFDCP clean

AUROC 0.867 · accuracy 0.726 · FPR 0.141 · FNR 0.346 · confident FN share (margin) 0.08 · (fused vacuity) 0.15

| signal | error AUROC | AURC | E-AURC | risk@90% | risk@80% | FN-vs-TN AUROC |
|---|---:|---:|---:|---:|---:|---:|
| margin | 0.723 | 0.1361 | 0.0935 | 0.251 | 0.232 | 0.814 |
| **u_fused** | 0.767 | 0.1222 | 0.0796 | 0.248 | 0.219 | 0.741 |
| conflict_fused | 0.718 | 0.1358 | 0.0932 | 0.268 | 0.237 | 0.778 |
| u_T | 0.526 | 0.2315 | 0.1889 | 0.288 | 0.287 | 0.696 |
| u_APP | 0.639 | 0.1607 | 0.1181 | 0.272 | 0.262 | 0.339 |
| u_OM | 0.712 | 0.1406 | 0.0980 | 0.246 | 0.225 | 0.545 |
| disagree_range | 0.666 | 0.1767 | 0.1341 | 0.275 | 0.255 | 0.732 |
| conflict_pair_max | 0.704 | 0.1387 | 0.0961 | 0.275 | 0.248 | 0.778 |

## DFDCP jpeg40

AUROC 0.824 · accuracy 0.483 · FPR 0.007 · FNR 0.794 · confident FN share (margin) 0.13 · (fused vacuity) 0.13

| signal | error AUROC | AURC | E-AURC | risk@90% | risk@80% | FN-vs-TN AUROC |
|---|---:|---:|---:|---:|---:|---:|
| margin | 0.656 | 0.3570 | 0.1906 | 0.518 | 0.501 | 0.784 |
| **u_fused** | 0.745 | 0.3050 | 0.1386 | 0.507 | 0.476 | 0.742 |
| conflict_fused | 0.659 | 0.3529 | 0.1865 | 0.525 | 0.494 | 0.773 |
| u_T | 0.555 | 0.4212 | 0.2548 | 0.541 | 0.541 | 0.681 |
| u_APP | 0.577 | 0.4029 | 0.2365 | 0.513 | 0.513 | 0.435 |
| u_OM | 0.729 | 0.3133 | 0.1469 | 0.501 | 0.470 | 0.666 |
| disagree_range | 0.628 | 0.3691 | 0.2027 | 0.538 | 0.515 | 0.743 |
| conflict_pair_max | 0.643 | 0.3586 | 0.1921 | 0.533 | 0.512 | 0.764 |

## DFDCP rs06

AUROC 0.811 · accuracy 0.728 · FPR 0.304 · FNR 0.255 · confident FN share (margin) 0.16 · (fused vacuity) 0.27

| signal | error AUROC | AURC | E-AURC | risk@90% | risk@80% | FN-vs-TN AUROC |
|---|---:|---:|---:|---:|---:|---:|
| margin | 0.721 | 0.1358 | 0.0943 | 0.249 | 0.225 | 0.723 |
| **u_fused** | 0.737 | 0.1317 | 0.0902 | 0.241 | 0.213 | 0.680 |
| conflict_fused | 0.715 | 0.1411 | 0.0995 | 0.254 | 0.234 | 0.659 |
| u_T | 0.648 | 0.1709 | 0.1294 | 0.257 | 0.242 | 0.637 |
| u_APP | 0.671 | 0.1538 | 0.1122 | 0.254 | 0.243 | 0.494 |
| u_OM | 0.680 | 0.1515 | 0.1099 | 0.256 | 0.236 | 0.583 |
| disagree_range | 0.651 | 0.2015 | 0.1599 | 0.262 | 0.237 | 0.582 |
| conflict_pair_max | 0.706 | 0.1435 | 0.1019 | 0.258 | 0.232 | 0.653 |

Confident FN share = false negatives whose signal marks them as MORE confident than the median correct genuine call. One detector seed, one temporal-branch seed; CDFv2 and DeepfakeEval2024 informed development; FaceShifter sealed.

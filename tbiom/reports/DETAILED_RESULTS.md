# Detailed results — DISCERN-v2

Nothing here is retyped: each table is assembled from a committed `tbiom/reports/*.json` and formatted the way that
report computes it. A missing input becomes `TODO(run)` rather than a silent gap. The ten test sets are shown in full
-- the seven-set main view stays in `FINAL_TABLE.md` -- together with the manipulation-family slices, the processed
conditions, the per-view decomposition, the fusion-operator study and the DF40 per-generator cross-manipulation
results for both face swap and face reenactment.

Test rows are means over the five fitted splits; the fusion weights are refitted per split on development, so a test number carries the spread of the weight fit but not of detector training (one detector seed). Development is never mixed into a test row.

## 0. Where the strengths are

| | |
|---|---|
| ten-set test mean | **0.9155** against the best baseline's 0.8959 (R0 (v3.1, ep 6)), +0.0196 |
| largest margins | CDFv3 +0.0661; KoDF +0.0591; DFEval24 +0.0263 |
| where a baseline leads | IDForge -0.0203; DFDC -0.0263; DFDCP -0.0632 |
| reenactment slices | CDFv3 reenactment 0.9642, KoDF first-order motion 0.9814 |
| view that earns most | temporal, +0.0569 on development (5/5 splits) |
| sealed benchmark | FaceShifter 0.9549, read once |
| fusion rule | Dempster-R deployed; the whole spread across seven rules is 0.0215 |

The pattern across sections 1-4: the system's margin comes from the sets and slices where motion carries the manipulation — CDFv3, KoDF, the two reenactment slices, DeepfakeEval2024 — and it is at or just below the field on the frame-artefact sets where a strong spatial baseline is already near its ceiling. That is the behaviour a temporal view added to two spatial views should produce, and it is what sections 3 and 7 measure directly.

## 1. Full test table — video AUROC on all ten sets

| method | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | DFEval24 | FFIW | KoDF | IDForge | DFDM | **10-set mean** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **DISCERN-v2 (DS-R)** | **0.9598** | **0.9616** | **0.9338** | **0.8626** | **0.8656** | **0.7252** | **0.9282** | **0.9574** | **0.9656** | **0.9956** | **0.9155** |
| DISCERN-v2 (logistic comparator) | 0.9697 | 0.9675 | 0.9300 | 0.8602 | 0.8557 | 0.6865 | 0.9262 | 0.9630 | 0.9622 | 0.9963 | 0.9117 |
| R0 (v3.1, ep 6) | 0.9509 | 0.8954 | 0.9431 | 0.8879 | 0.9064 | 0.6869 | 0.9036 | 0.8198 | 0.9723 | 0.9929 | 0.8959 |
| A (v3.1, ep 5) | 0.9388 | 0.8770 | 0.9461 | 0.8889 | 0.9013 | 0.6852 | 0.9011 | 0.8242 | 0.9697 | 0.9915 | 0.8924 |
| B (v3.1, ep 8) | 0.9225 | 0.8801 | 0.9387 | 0.8889 | 0.9057 | 0.6943 | 0.8906 | 0.8497 | 0.9553 | 0.9944 | 0.8920 |
| C (v3.1, ep 2) | 0.9183 | 0.6648 | 0.8802 | 0.8763 | 0.8565 | 0.6989 | 0.9279 | 0.8548 | 0.9492 | 0.9868 | 0.8614 |
| DiCoME (our training, v3.1, ep 5) | 0.9693 | 0.8691 | 0.9343 | 0.8818 | 0.8755 | 0.6769 | 0.9284 | 0.8509 | 0.9598 | 0.9909 | 0.8937 |
| GenD-CLIP | 0.9534 | 0.8458 | 0.9350 | 0.8708 | 0.9162 | 0.6831 | 0.9278 | 0.8329 | 0.9839 | 0.9959 | 0.8945 |
| GenD-PE | 0.9548 | 0.8952 | 0.9203 | 0.8158 | 0.8860 | 0.6563 | 0.9330 | 0.8189 | 0.9808 | 0.9815 | 0.8842 |
| GenD-DINO | 0.9177 | 0.8172 | 0.9165 | 0.8470 | 0.9114 | 0.6536 | 0.9450 | 0.8984 | 0.9860 | 0.9903 | 0.8883 |
| Effort | 0.9312 | 0.7975 | 0.9276 | 0.8536 | 0.8617 | 0.6769 | 0.9252 | 0.8725 | 0.9583 | 0.9729 | 0.8777 |
| ForAda | 0.9544 | 0.7494 | 0.9395 | 0.8726 | 0.9289 | 0.6227 | 0.9055 | 0.8739 | 0.9101 | 0.9713 | 0.8728 |
| FSFM | 0.9554 | 0.8751 | 0.9219 | 0.8844 | 0.9061 | 0.6817 | 0.9168 | 0.8467 | 0.9566 | 0.9790 | 0.8924 |
| FSFM-Adapter | 0.8689 | 0.8381 | 0.7516 | 0.8318 | 0.8486 | 0.6280 | 0.8565 | 0.7405 | 0.9077 | 0.8452 | 0.8117 |

The seven-set main view in `FINAL_TABLE.md` drops DFDC, DFDCP and DeepfakeEval2024; they are kept here. The ten-set mean is over all ten in every row, so the mean column is comparable with the main table.

## 2. Where the system is strong, set by set

| test set | **ours (DS-R)** | best baseline | that baseline | Δ ours − best | rank of ours among 13 |
|---|---:|---:|---|---:|---:|
| CDFv2 | **0.9598** | 0.9693 | DiCoME (our training, v3.1, ep 5) | -0.0095 | 2 / 13 |
| CDFv3 | **0.9616** | 0.8954 | R0 (v3.1, ep 6) | +0.0661 | 1 / 13 |
| DFD | **0.9338** | 0.9461 | A (v3.1, ep 5) | -0.0124 | 7 / 13 |
| DFDC | **0.8626** | 0.8889 | A (v3.1, ep 5) | -0.0263 | 9 / 13 |
| DFDCP | **0.8656** | 0.9289 | ForAda | -0.0632 | 10 / 13 |
| DFEval24 | **0.7252** | 0.6989 | C (v3.1, ep 2) | +0.0263 | 1 / 13 |
| FFIW | **0.9282** | 0.9450 | GenD-DINO | -0.0168 | 4 / 13 |
| KoDF | **0.9574** | 0.8984 | GenD-DINO | +0.0591 | 1 / 13 |
| IDForge | **0.9656** | 0.9860 | GenD-DINO | -0.0203 | 6 / 13 |
| DFDM | **0.9956** | 0.9959 | GenD-CLIP | -0.0003 | 2 / 13 |
| **10-set mean** | **0.9155** | 0.8959 | — | +0.0196 | 1 / 13 |

Ours is the best of the thirteen systems on 3 of the ten sets, and best on the ten-set mean. The sets where a baseline leads are named above rather than omitted.

## 3. What each view contributes on the test sets

| test set | temporal T | appearance APP | off-manifold OM | best single view | **fused DS-R** | Δ fusion − best view |
|---|---:|---:|---:|---:|---:|---:|
| CDFv2 | 0.7383 | 0.9612 | 0.9736 | off-manifold 0.9736 | **0.9598** | -0.0138 |
| CDFv3 | 0.9263 | 0.8576 | 0.8801 | temporal 0.9263 | **0.9616** | +0.0353 |
| DFD | 0.7967 | 0.9304 | 0.9338 | off-manifold 0.9338 | **0.9338** | -0.0001 |
| DFDC | 0.7009 | 0.8758 | 0.8808 | off-manifold 0.8808 | **0.8626** | -0.0182 |
| DFDCP | 0.7408 | 0.8698 | 0.8727 | off-manifold 0.8727 | **0.8656** | -0.0070 |
| DFEval24 | 0.6732 | 0.6741 | 0.6688 | appearance 0.6741 | **0.7252** | +0.0510 |
| FFIW | 0.8011 | 0.9186 | 0.9354 | off-manifold 0.9354 | **0.9282** | -0.0072 |
| KoDF | 0.9656 | 0.8494 | 0.8548 | temporal 0.9656 | **0.9574** | -0.0081 |
| IDForge | 0.8899 | 0.9510 | 0.9635 | off-manifold 0.9635 | **0.9656** | +0.0021 |
| DFDM | 0.9798 | 0.9886 | 0.9920 | off-manifold 0.9920 | **0.9956** | +0.0036 |
| **mean** | 0.8213 | 0.8877 | 0.8955 | — | **0.9155** | +0.0038 |

Fusion beats the best single view on 4 of the ten sets (mean +0.0038). Which view is best changes by set — the temporal view leads where motion carries the artefact, the off-manifold view where the frame does — which is the case for combining them rather than picking one.

## 4. Manipulation-family slices and the sealed benchmark

| slice | what it isolates | **ours (DS-R)** | ours (logistic) | temporal view | appearance | off-manifold |
|---|---|---:|---:|---:|---:|---:|
| CDFv3 reenactment | reenactment inside a swap-dominated set | **0.9642** | 0.9709 | 0.9741 | 0.8137 | 0.8416 |
| KoDF first-order motion | a single reenactment generator, Korean subjects | **0.9814** | 0.9792 | 0.9887 | 0.7164 | 0.7306 |
| FaceShifter (sealed, read once) | unseen swap on held-out FF++ ids | **0.9549** | 0.9480 | 0.9244 | 0.8659 | 0.8686 |

FaceShifter was read once, after the recipe was frozen, and never tuned against (`tbiom/docs/SEALED_BENCHMARKS.md`). Its 140 source ids are exactly the FF++ test split.

## 5. Processed conditions

| condition | **ours (DS-R)** | ours (logistic) | Δ vs clean (DS-R) |
|---|---:|---:|---:|
| CDFv2 jpeg40 | **0.8695** | 0.7199 | -0.0903 |
| CDFv2 rs06 | **0.9418** | 0.9470 | -0.0180 |
| DFDCP jpeg40 | **0.8198** | 0.7299 | -0.0458 |
| DFDCP rs06 | **0.8084** | 0.7935 | -0.0572 |

Both conditions are applied to the stored crops at evaluation time; no arm was trained on them.

## 6. Fusion operators — alternatives to Dempster's rule

| operator | dev clean | dev JPEG q40 | dev resize 0.60 | 10-set test mean | FPR @ own τ | TPR @ own τ | vacuity error-AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dempster | 0.8771 | 0.8043 | 0.8576 | 0.9098 | 0.110 | 0.729 | 0.7894 |
| Dempster-R | 0.8808 | 0.8096 | 0.8589 | 0.9155 | 0.107 | 0.737 | 0.7805 |
| Yager | 0.8727 | 0.7980 | 0.8473 | 0.9150 | 0.100 | 0.707 | 0.7837 |
| Yager-R | 0.8791 | 0.8139 | 0.8576 | 0.9133 | 0.102 | 0.725 | 0.7343 |
| PCR6 | 0.8753 | 0.7979 | 0.8532 | 0.9117 | 0.091 | 0.680 | 0.6954 |
| PCR6-R | 0.8802 | 0.8091 | 0.8574 | 0.9147 | 0.078 | 0.675 | 0.7180 |
| Murphy | 0.8735 | 0.7953 | 0.8507 | 0.9118 | 0.082 | 0.667 | 0.7361 |
| Murphy-R | 0.8790 | 0.8067 | 0.8562 | 0.9145 | 0.096 | 0.714 | 0.7332 |
| cautious | 0.8662 | 0.8006 | 0.8520 | 0.8940 | 0.113 | 0.684 | 0.6799 |
| cautious-R | 0.8751 | 0.7994 | 0.8554 | 0.9108 | 0.113 | 0.753 | 0.7301 |
| cumulative | 0.8726 | 0.8026 | 0.8561 | 0.9031 | 0.116 | 0.710 | 0.7128 |
| cumulative-R | 0.8789 | 0.8095 | 0.8590 | 0.9137 | 0.117 | 0.759 | 0.7654 |
| averaging | 0.8720 | 0.8042 | 0.8563 | 0.9017 | 0.112 | 0.708 | 0.7148 |
| averaging-R | 0.8799 | 0.8088 | 0.8592 | 0.9141 | 0.109 | 0.748 | 0.7630 |
| logistic | 0.8736 | 0.7564 | 0.8542 | 0.9117 | 0.057 | 0.648 | n/a |

Every operator is fused over the same three view opinions with the same fitted reliability weights; `-R` marks the reliability-discounted form. Deployment uses **Dempster-R** (0.9155 ten-set mean) because it was pre-registered before any of these were run.

The strongest alternative is **Yager** (0.9150, -0.0006 against Dempster-R). Operators within 0.002 of the deployed rule: Yager, PCR6-R, Murphy-R, averaging-R, cumulative-R. Reliability discounting helps on the ten-set mean for 6 of the seven rules (mean +0.0071), and helps on clean development for 7 of seven — a larger and more consistent effect than the choice of rule, whose whole spread is 0.0215.

### 6a. Every operator on every test set

| operator | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | DFEval24 | FFIW | KoDF | IDForge | DFDM | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Dempster | 0.9515 | 0.9596 | 0.9249 | 0.8521 | 0.8588 | 0.7223 | 0.9189 | 0.9558 | 0.9593 | 0.9944 | 0.9098 |
| Dempster-R | 0.9598 | 0.9616 | 0.9338 | 0.8626 | 0.8656 | 0.7252 | 0.9282 | 0.9574 | 0.9656 | 0.9956 | 0.9155 |
| Yager | 0.9667 | 0.9458 | 0.9390 | 0.8716 | 0.8683 | 0.7172 | 0.9356 | 0.9448 | 0.9661 | 0.9946 | 0.9150 |
| Yager-R | 0.9616 | 0.9635 | 0.9324 | 0.8529 | 0.8591 | 0.7225 | 0.9223 | 0.9599 | 0.9631 | 0.9954 | 0.9133 |
| PCR6 | 0.9576 | 0.9521 | 0.9299 | 0.8605 | 0.8639 | 0.7189 | 0.9273 | 0.9510 | 0.9613 | 0.9941 | 0.9117 |
| PCR6-R | 0.9607 | 0.9613 | 0.9334 | 0.8594 | 0.8633 | 0.7241 | 0.9272 | 0.9577 | 0.9648 | 0.9954 | 0.9147 |
| Murphy | 0.9594 | 0.9481 | 0.9306 | 0.8643 | 0.8642 | 0.7195 | 0.9297 | 0.9473 | 0.9617 | 0.9937 | 0.9118 |
| Murphy-R | 0.9626 | 0.9577 | 0.9329 | 0.8622 | 0.8641 | 0.7230 | 0.9286 | 0.9545 | 0.9644 | 0.9950 | 0.9145 |
| cautious | 0.9236 | 0.9607 | 0.8972 | 0.8152 | 0.8406 | 0.7104 | 0.8925 | 0.9612 | 0.9459 | 0.9930 | 0.8940 |
| cautious-R | 0.9481 | 0.9622 | 0.9317 | 0.8497 | 0.8574 | 0.7184 | 0.9187 | 0.9630 | 0.9627 | 0.9960 | 0.9108 |
| cumulative | 0.9399 | 0.9608 | 0.9159 | 0.8335 | 0.8493 | 0.7176 | 0.9077 | 0.9583 | 0.9540 | 0.9941 | 0.9031 |
| cumulative-R | 0.9567 | 0.9639 | 0.9340 | 0.8546 | 0.8612 | 0.7233 | 0.9224 | 0.9607 | 0.9643 | 0.9959 | 0.9137 |
| averaging | 0.9356 | 0.9623 | 0.9132 | 0.8293 | 0.8480 | 0.7170 | 0.9033 | 0.9615 | 0.9528 | 0.9941 | 0.9017 |
| averaging-R | 0.9542 | 0.9632 | 0.9330 | 0.8578 | 0.8642 | 0.7236 | 0.9241 | 0.9602 | 0.9650 | 0.9960 | 0.9141 |
| logistic | 0.9697 | 0.9675 | 0.9300 | 0.8602 | 0.8557 | 0.6865 | 0.9262 | 0.9630 | 0.9622 | 0.9963 | 0.9117 |

### 6b. Choosing an alternative rule

Four criteria, each already in the table above: clean development (what selection would have used), the two processed development conditions (robustness), the ten-set test mean (generalisation), and how well the rule's own fused vacuity ranks its errors (the uncertainty claim the paper makes). A rule is only a usable alternative if it is close on all four, so they are ranked on each separately rather than blended into one score.

| criterion | best | second | third | Dempster-R |
|---|---|---|---|---:|
| clean development | Dempster-R 0.8808 | PCR6-R 0.8802 | averaging-R 0.8799 | 0.8808 (rank 1) |
| JPEG q40 development | Yager-R 0.8139 | Dempster-R 0.8096 | cumulative-R 0.8095 | 0.8096 (rank 2) |
| resize 0.60 development | averaging-R 0.8592 | cumulative-R 0.8590 | Dempster-R 0.8589 | 0.8589 (rank 3) |
| ten-set test mean | Dempster-R 0.9155 | Yager 0.9150 | PCR6-R 0.9147 | 0.9155 (rank 1) |
| vacuity error-AUROC | Dempster 0.7894 | Yager 0.7837 | Dempster-R 0.7805 | 0.7805 (rank 3) |

The alternative whose **worst** rank across the five criteria is best is **averaging-R** (never below rank 5 of 14); it is -0.0014 against Dempster-R on the ten-set test mean. No alternative is better than Dempster-R on every criterion, so the deployed rule is kept and this table is reported as the operator study rather than as a change.

## 7. Three-view ablation — what each view is worth

Δ = the full triple minus the named pair, on the clean development selection folds, paired source-grouped bootstrap (1000 draws per split). The right-hand columns are the same subsets scored on the ten test sets.

| removed view | remaining pair | Δ DS-R (dev) | splits with CI above 0 | Δ logistic (dev) | pair, 10-set test mean | triple − pair (test) |
|---|---|---:|---:|---:|---:|---:|
| temporal | APP+OM | +0.0569 | 5/5 | +0.0560 | 0.8955 | +0.0201 |
| appearance | T+OM | +0.0012 | 0/5 | -0.0050 | 0.9113 | +0.0043 |
| off-manifold | T+APP | +0.0102 | 5/5 | +0.0047 | 0.9015 | +0.0140 |

The full triple reaches 0.9155 on the ten-set test mean. The temporal view is the one whose removal costs decisively on both development and test; off-manifold earns its place under DS-R; appearance is marginal and is reported as such rather than dropped.

## 8. Uncertainty — what ranks the system's own errors

| condition | best signal, DS-R | error AUROC | best signal, logistic | error AUROC |
|---|---|---:|---|---:|
| clean development | u_fused | 0.7805 | conflict_pair_max | 0.7453 |
| jpeg40 development | u_fused | 0.7152 | u_mean | 0.7172 |
| rs06 development | u_fused | 0.7555 | conflict_pair_max | 0.7345 |
| clean tests (10 sets pooled) | u_fused | 0.8072 | conflict_pair_max | 0.7794 |
| CDFv2 clean | u_fused | 0.7910 | u_OM | 0.7587 |
| CDFv2 jpeg40 | u_OM | 0.7736 | u_OM | 0.7772 |
| CDFv2 rs06 | u_OM | 0.8019 | u_OM | 0.7753 |
| DFDCP clean | u_OM | 0.7548 | u_OM | 0.7408 |
| DFDCP jpeg40 | u_OM | 0.7517 | u_OM | 0.7577 |
| DFDCP rs06 | u_fused | 0.7366 | u_OM | 0.7115 |

## 9. Seed stability of the temporal branch

| temporal seed | fused DS-R, 10-set mean | branch alone, 10-set mean |
|---|---:|---:|
| vjepa2_lora_B_seed42 | 0.9155 | 0.8213 |
| vjepa2_lora_B_seed1337 | 0.9136 | 0.8308 |
| vjepa2_lora_B_seed7 | 0.9155 | 0.8077 |

## 10. Cross-manipulation on DF40 — face swap

Every generator is scored against the genuine pool DF40 ships for its source, so a generator's AUROC is never computed against another source's real videos. DiCoME is our own trained checkpoint, replayed through its own two-view fusion, so both sides see identical videos with identical aggregation.

### 10a. Celeb-DF-sourced

TODO(run) `analysis/tbiom/df40_analysis.py` — exports still in flight.

### 10b. FaceForensics++-sourced

TODO(run) `analysis/tbiom/df40_analysis.py` — exports still in flight.

## 11. Cross-manipulation on DF40 — face reenactment

### 11a. Celeb-DF-sourced

TODO(run) `analysis/tbiom/df40_analysis.py` — exports still in flight.

### 11b. FaceForensics++-sourced

TODO(run) `analysis/tbiom/df40_analysis.py` — exports still in flight.

DiCoME's published cross-manipulation table lists UniFace, BlendFace, MobileFaceSwap, e4s, FaceDancer, FSGAN, InSwapper and SimSwap. The classical-graphics FaceSwap and DeepFaceLab generators, the FF++-sourced version of every generator, and the whole reenactment half are added here.

---

One detector seed and one temporal-branch seed unless stated. CDFv2 and DeepfakeEval2024 informed development and are not untouched confirmation data. Generated by `analysis/tbiom/detailed_report.py`.

# Proposal results — all tables, every dataset

Regenerated 2026-10-14T15:16:27-04:00. Framework: **App + TD + LFS + STT**, fused with PCR6+-R.

**Evaluation scope `nolipadv`** — lip-sync/audio-driven and adversarial ADMITTED; whole-face synthesis excluded.
The scope changes the video census of CDFv3 and KoDF only; the other ten sets are unaffected.
A number here is comparable to one in another scope's report only where the census matches.

| id | thesis name | what it is |
|---|---|---|
| A | **App** | Appearance — PE-Core-L/336, LoRA + LayerNorm, frame mean |
| V | **TD** | Temporal Dynamics — V-JEPA 2 LoRA, 16 frames |
| O | **LFS** | Latent Facial Structure — CLIP/β-VAE orthogonal residual |
| S16ctl | **STT** | Spatio-Temporal Texture — PE-Core block 16 + real relative timestamps |
| P16ctl | **STT-order** | the same with learned slot positions |

Twelve test sets. Every number is video level. Weights are fitted once on the
development pool (VALmix, VALmixV2, FF++ val) and applied unchanged to every test set.

## How to read the mean columns

Tables 1 and 2 carry three. A row's own `mean` is over that row's coverage and is NOT
comparable across rows — ours spans twelve sets, DiCoME five. `mean (all)` and
`mean (wide)` are matched supports and are the only columns a ranking may be read from.
Each section states which datasets each one covers.

Figure: `tbiom/reports/branch_fusion/scope_nolipadv/figures/uncertainty_riskcoverage.png`

| table | what it answers |
|---|---|
| A | what each branch and each hand-picked group contributes |
| A2 | the same question settled by exhaustion — all 31 non-empty combinations of the five |
| B | which decision rule to fuse with, the four-branch set held fixed |
| 1 | where the framework stands against the field, on detection |
| 2 | and on probability quality and uncertainty |

---

# Table A — branch ablation

Ten groups, every dataset, fused with **PCR6+-R**. No baseline rows: this table answers
what each branch contributes, and a SOTA column here would invite two questions to be
read off one row. The comparison against the field is Table 1.

`*in progress*` means the cell is not computable yet, not that it is zero or not applicable.
Every branch group is computed on all 12 test sets.

## AUROC (higher is better)

| branch group | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| App | 0.9460 | 0.8893 | 0.9505 | 0.8371 | 0.9000 | 0.9760 | 0.8117 | 0.9950 | 0.9653 | 0.9708 | 0.9048 | 0.8636 | 0.9175 |
| TD | 0.7383 | 0.9288 | 0.7967 | 0.7009 | 0.7408 | 0.8011 | 0.9298 | 0.8899 | 0.9798 | 0.9456 | 0.8499 | 0.8378 | 0.8450 |
| LFS | 0.9736 | 0.8790 | 0.9338 | 0.8808 | 0.8727 | 0.9354 | 0.7999 | 0.9635 | 0.9920 | 0.9908 | 0.8595 | 0.7678 | 0.9041 |
| STT-order | 0.9661 | 0.8942 | 0.9524 | 0.8360 | 0.8887 | 0.9640 | 0.9035 | 0.9935 | 0.9969 | 0.9825 | 0.9012 | 0.8589 | 0.9282 |
| STT | 0.9683 | 0.8884 | 0.9485 | 0.8176 | 0.8976 | 0.9655 | 0.9093 | 0.9968 | 0.9954 | 0.9731 | 0.9056 | 0.8663 | 0.9277 |
| App+TD+LFS *(deployed)* | 0.9645 | 0.9623 | 0.9513 | 0.8596 | 0.9058 | 0.9693 | 0.9021 | 0.9950 | 0.9927 | 0.9883 | 0.9268 | 0.9119 | 0.9441 |
| App+TD+LFS+STT-order | 0.9738 | 0.9621 | 0.9577 | 0.8593 | 0.9090 | 0.9701 | 0.9060 | 0.9961 | 0.9953 | 0.9867 | 0.9285 | 0.9210 | 0.9471 |
| App+TD+LFS+STT **(final)** | 0.9788 | 0.9614 | 0.9569 | 0.8622 | 0.9112 | 0.9743 | 0.9112 | 0.9961 | 0.9966 | 0.9875 | 0.9277 | 0.9144 | 0.9482 |
| App+TD+LFS+STT-order+STT | 0.9738 | 0.9621 | 0.9577 | 0.8593 | 0.9090 | 0.9701 | 0.9060 | 0.9961 | 0.9953 | 0.9867 | 0.9285 | 0.9210 | 0.9471 |
| App+LFS+STT-order+STT | 0.9847 | 0.9145 | 0.9528 | 0.8652 | 0.9224 | 0.9800 | 0.8631 | 0.9962 | 0.9939 | 0.9863 | 0.9146 | 0.8713 | 0.9371 |

## FPR at TPR = 90% (lower is better)

| branch group | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| App | 0.1910 | 0.3596 | 0.1708 | 0.4894 | 0.2565 | 0.0576 | 0.6960 | 0.0013 | 0.0730 | 0.0612 | 0.2542 | 0.4836 | 0.2579 |
| TD | 0.6180 | 0.2247 | 0.5868 | 0.7525 | 0.6304 | 0.4922 | 0.2419 | 0.3174 | 0.0393 | 0.1837 | 0.4922 | 0.4697 | 0.4207 |
| LFS | 0.0618 | 0.4213 | 0.2011 | 0.4289 | 0.3739 | 0.2242 | 0.7320 | 0.1145 | 0.0056 | 0.0000 | 0.3651 | 0.6554 | 0.2987 |
| STT-order | 0.1292 | 0.3708 | 0.1736 | 0.4592 | 0.3000 | 0.1101 | 0.3586 | 0.0057 | 0.0000 | 0.0000 | 0.2458 | 0.4799 | 0.2194 |
| STT | 0.0843 | 0.4101 | 0.2094 | 0.5110 | 0.3130 | 0.0974 | 0.3288 | 0.0020 | 0.0000 | 0.0408 | 0.2387 | 0.4842 | 0.2266 |
| App+TD+LFS *(deployed)* | 0.1011 | 0.0955 | 0.1653 | 0.3983 | 0.2870 | 0.0997 | 0.4342 | 0.0094 | 0.0112 | 0.0204 | 0.1773 | 0.2271 | 0.1689 |
| App+TD+LFS+STT-order | 0.0674 | 0.0674 | 0.1570 | 0.3978 | 0.2739 | 0.1032 | 0.4268 | 0.0037 | 0.0056 | 0.0204 | 0.1751 | 0.2281 | 0.1605 |
| App+TD+LFS+STT **(final)** | 0.0674 | 0.1124 | 0.1570 | 0.3948 | 0.2696 | 0.0646 | 0.3834 | 0.0067 | 0.0056 | 0.0204 | 0.1645 | 0.2222 | 0.1557 |
| App+TD+LFS+STT-order+STT | 0.0674 | 0.0674 | 0.1570 | 0.3978 | 0.2739 | 0.1032 | 0.4268 | 0.0037 | 0.0056 | 0.0204 | 0.1751 | 0.2281 | 0.1605 |
| App+LFS+STT-order+STT | 0.0449 | 0.3146 | 0.1653 | 0.4078 | 0.2348 | 0.0519 | 0.5633 | 0.0003 | 0.0000 | 0.0000 | 0.2147 | 0.4423 | 0.2033 |


---

# Table A2 — all 31 combinations of the five branches

The exhaustive version of Table A: every non-empty subset of {App, TD, LFS, STT-order,
STT}, each one fitted on development and scored on all twelve test sets. It is here so
the ten hand-picked groups in Table A cannot be read as a favourable selection — the
best cell in this grid IS the framework.

> **EXPLORATORY DIAGNOSTIC -- NOT A HELD-OUT RESULT. Every subset here was scored on every test set, so the best cell in this grid is a maximum over a large search on the test partitions and is biased upward. Use it to read structure and to choose table scope. The held-out numbers are the locked rows in test_eval.json, and the deployed subset must be chosen on development.**

## Dataset census

| set | videos | real | fake | prevalence |
|---|---:|---:|---:|---:|
| CDFv2 | 518 | 178 | 340 | 0.656 |
| CDFv3 | 5418 | 178 | 5240 | 0.967 |
| DFD | 3431 | 363 | 3068 | 0.894 |
| DFDC | 4704 | 2315 | 2389 | 0.508 |
| DFDCP | 654 | 230 | 424 | 0.648 |
| FFIW | 3470 | 1735 | 1735 | 0.500 |
| KoDF | 2239 | 806 | 1433 | 0.640 |
| IDForge | 5974 | 2987 | 2987 | 0.500 |
| DFDM | 1896 | 178 | 1718 | 0.906 |
| UADFV | 98 | 49 | 49 | 0.500 |
| DeepSpeak-1.1 | 2102 | 1416 | 686 | 0.326 |
| DeepSpeak-2.0 | 2572 | 1863 | 709 | 0.276 |

## AUROC by subset and test set

`worst` is the subset's weakest set, `spread` its best minus its worst -- a subset with a high macro and a wide spread is carried by a few sets.

| subset | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | macro | worst | spread |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| App+TD+LFS+STT | 0.9788 | 0.9614 | 0.9569 | 0.8622 | 0.9112 | 0.9743 | 0.9112 | 0.9961 | 0.9966 | 0.9875 | 0.9277 | 0.9144 | **0.9482** | 0.8622 (DFDC) | 0.1345 |
| App+TD+LFS+STT-order | 0.9738 | 0.9621 | 0.9577 | 0.8593 | 0.9090 | 0.9701 | 0.9060 | 0.9961 | 0.9953 | 0.9867 | 0.9285 | 0.9210 | **0.9471** | 0.8593 (DFDC) | 0.1368 |
| App+TD+LFS+STT-order+STT | 0.9738 | 0.9621 | 0.9577 | 0.8593 | 0.9090 | 0.9701 | 0.9060 | 0.9961 | 0.9953 | 0.9867 | 0.9285 | 0.9210 | **0.9471** | 0.8593 (DFDC) | 0.1368 |
| TD+LFS+STT-order+STT | 0.9782 | 0.9656 | 0.9543 | 0.8587 | 0.8895 | 0.9667 | 0.9346 | 0.9931 | 0.9981 | 0.9850 | 0.9243 | 0.9129 | **0.9467** | 0.8587 (DFDC) | 0.1394 |
| TD+LFS+STT | 0.9764 | 0.9680 | 0.9495 | 0.8566 | 0.8857 | 0.9638 | 0.9369 | 0.9904 | 0.9977 | 0.9846 | 0.9249 | 0.9078 | **0.9452** | 0.8566 (DFDC) | 0.1411 |
| App+TD+STT-order+STT | 0.9639 | 0.9577 | 0.9551 | 0.8365 | 0.9114 | 0.9713 | 0.9099 | 0.9970 | 0.9948 | 0.9825 | 0.9257 | 0.9252 | **0.9443** | 0.8365 (DFDC) | 0.1605 |
| App+TD+LFS | 0.9645 | 0.9623 | 0.9513 | 0.8596 | 0.9058 | 0.9693 | 0.9021 | 0.9950 | 0.9927 | 0.9883 | 0.9268 | 0.9119 | **0.9441** | 0.8596 (DFDC) | 0.1354 |
| App+TD+STT | 0.9592 | 0.9587 | 0.9530 | 0.8338 | 0.9099 | 0.9706 | 0.9119 | 0.9967 | 0.9939 | 0.9817 | 0.9262 | 0.9251 | **0.9434** | 0.8338 (DFDC) | 0.1628 |
| App+TD+STT-order | 0.9592 | 0.9598 | 0.9542 | 0.8350 | 0.9037 | 0.9658 | 0.9060 | 0.9964 | 0.9936 | 0.9850 | 0.9257 | 0.9282 | **0.9427** | 0.8350 (DFDC) | 0.1614 |
| TD+LFS+STT-order | 0.9739 | 0.9714 | 0.9486 | 0.8565 | 0.8774 | 0.9497 | 0.9331 | 0.9885 | 0.9976 | 0.9854 | 0.9172 | 0.9085 | **0.9423** | 0.8565 (DFDC) | 0.1412 |
| TD+STT-order+STT | 0.9603 | 0.9570 | 0.9383 | 0.8108 | 0.8858 | 0.9587 | 0.9399 | 0.9936 | 0.9970 | 0.9783 | 0.9188 | 0.9261 | **0.9387** | 0.8108 (DFDC) | 0.1861 |
| App+LFS+STT-order+STT | 0.9847 | 0.9145 | 0.9528 | 0.8652 | 0.9224 | 0.9800 | 0.8631 | 0.9962 | 0.9939 | 0.9863 | 0.9146 | 0.8713 | **0.9371** | 0.8631 (KoDF) | 0.1331 |
| App+LFS+STT-order | 0.9859 | 0.9156 | 0.9531 | 0.8732 | 0.9181 | 0.9819 | 0.8553 | 0.9953 | 0.9942 | 0.9888 | 0.9166 | 0.8643 | **0.9368** | 0.8553 (KoDF) | 0.1401 |
| LFS+STT-order+STT | 0.9834 | 0.9089 | 0.9532 | 0.8686 | 0.9064 | 0.9746 | 0.9018 | 0.9947 | 0.9972 | 0.9875 | 0.9103 | 0.8548 | **0.9368** | 0.8548 (DeepSpeak-2.0) | 0.1424 |
| App+LFS+STT | 0.9838 | 0.9134 | 0.9518 | 0.8656 | 0.9241 | 0.9807 | 0.8628 | 0.9963 | 0.9918 | 0.9854 | 0.9137 | 0.8699 | **0.9366** | 0.8628 (KoDF) | 0.1335 |
| LFS+STT | 0.9868 | 0.9090 | 0.9489 | 0.8703 | 0.9099 | 0.9742 | 0.9006 | 0.9932 | 0.9964 | 0.9850 | 0.9105 | 0.8480 | **0.9361** | 0.8480 (DeepSpeak-2.0) | 0.1485 |
| LFS+STT-order | 0.9826 | 0.9107 | 0.9533 | 0.8783 | 0.9026 | 0.9762 | 0.8950 | 0.9930 | 0.9978 | 0.9883 | 0.9058 | 0.8472 | **0.9359** | 0.8472 (DeepSpeak-2.0) | 0.1506 |
| App+TD | 0.9330 | 0.9607 | 0.9419 | 0.8250 | 0.8946 | 0.9591 | 0.9055 | 0.9935 | 0.9878 | 0.9808 | 0.9230 | 0.9236 | **0.9357** | 0.8250 (DFDC) | 0.1685 |
| TD+STT | 0.9515 | 0.9560 | 0.9329 | 0.8045 | 0.8836 | 0.9550 | 0.9430 | 0.9868 | 0.9942 | 0.9683 | 0.9178 | 0.9205 | **0.9345** | 0.8045 (DFDC) | 0.1897 |
| App+STT-order+STT | 0.9774 | 0.9077 | 0.9533 | 0.8436 | 0.9172 | 0.9766 | 0.8698 | 0.9967 | 0.9919 | 0.9813 | 0.9125 | 0.8782 | **0.9339** | 0.8436 (DFDC) | 0.1531 |
| App+STT | 0.9761 | 0.9063 | 0.9528 | 0.8418 | 0.9188 | 0.9772 | 0.8704 | 0.9969 | 0.9868 | 0.9746 | 0.9134 | 0.8788 | **0.9328** | 0.8418 (DFDC) | 0.1551 |
| App+STT-order | 0.9745 | 0.9073 | 0.9567 | 0.8489 | 0.9135 | 0.9777 | 0.8607 | 0.9962 | 0.9927 | 0.9838 | 0.9085 | 0.8736 | **0.9328** | 0.8489 (DFDC) | 0.1472 |
| STT-order+STT | 0.9693 | 0.8945 | 0.9516 | 0.8289 | 0.8977 | 0.9653 | 0.9069 | 0.9956 | 0.9965 | 0.9792 | 0.9044 | 0.8656 | **0.9296** | 0.8289 (DFDC) | 0.1675 |
| App+LFS | 0.9747 | 0.9059 | 0.9492 | 0.8678 | 0.9153 | 0.9806 | 0.8192 | 0.9951 | 0.9845 | 0.9863 | 0.9125 | 0.8568 | **0.9290** | 0.8192 (KoDF) | 0.1759 |
| STT-order | 0.9661 | 0.8942 | 0.9524 | 0.8360 | 0.8887 | 0.9640 | 0.9035 | 0.9935 | 0.9969 | 0.9825 | 0.9012 | 0.8589 | **0.9282** | 0.8360 (DFDC) | 0.1609 |
| STT | 0.9683 | 0.8884 | 0.9485 | 0.8176 | 0.8976 | 0.9655 | 0.9093 | 0.9968 | 0.9954 | 0.9731 | 0.9056 | 0.8663 | **0.9277** | 0.8176 (DFDC) | 0.1792 |
| TD+LFS | 0.9575 | 0.9670 | 0.9294 | 0.8489 | 0.8576 | 0.9174 | 0.9270 | 0.9621 | 0.9955 | 0.9863 | 0.9037 | 0.8746 | **0.9273** | 0.8489 (DFDC) | 0.1466 |
| TD+STT-order | 0.9438 | 0.9684 | 0.9125 | 0.7808 | 0.8501 | 0.9204 | 0.9407 | 0.9854 | 0.9972 | 0.9804 | 0.9082 | 0.9160 | **0.9253** | 0.7808 (DFDC) | 0.2163 |
| App | 0.9460 | 0.8893 | 0.9505 | 0.8371 | 0.9000 | 0.9760 | 0.8117 | 0.9950 | 0.9653 | 0.9708 | 0.9048 | 0.8636 | **0.9175** | 0.8117 (KoDF) | 0.1833 |
| LFS | 0.9736 | 0.8790 | 0.9338 | 0.8808 | 0.8727 | 0.9354 | 0.7999 | 0.9635 | 0.9920 | 0.9908 | 0.8595 | 0.7678 | **0.9041** | 0.7678 (DeepSpeak-2.0) | 0.2242 |
| TD | 0.7383 | 0.9288 | 0.7967 | 0.7009 | 0.7408 | 0.8011 | 0.9298 | 0.8899 | 0.9798 | 0.9456 | 0.8499 | 0.8378 | **0.8450** | 0.7009 (DFDC) | 0.2789 |
| ORACLE (best single per set, uses test labels) | 0.9736 | 0.9288 | 0.9524 | 0.8808 | 0.9000 | 0.9760 | 0.9298 | 0.9968 | 0.9969 | 0.9908 | 0.9056 | 0.8663 | **0.9415** | 0.8663 (DeepSpeak-2.0) | 0.1306 |

## Best single branch per set (the oracle's choice)

| set | best branch | AUROC | App | TD | LFS | STT-order | STT |
|---|---|---:|---:|---:|---:|---:|---:|
| CDFv2 | **LFS** | 0.9736 | 0.9460 | 0.7383 | 0.9736 | 0.9661 | 0.9683 |
| CDFv3 | **TD** | 0.9288 | 0.8893 | 0.9288 | 0.8790 | 0.8942 | 0.8884 |
| DFD | **STT-order** | 0.9524 | 0.9505 | 0.7967 | 0.9338 | 0.9524 | 0.9485 |
| DFDC | **LFS** | 0.8808 | 0.8371 | 0.7009 | 0.8808 | 0.8360 | 0.8176 |
| DFDCP | **App** | 0.9000 | 0.9000 | 0.7408 | 0.8727 | 0.8887 | 0.8976 |
| FFIW | **App** | 0.9760 | 0.9760 | 0.8011 | 0.9354 | 0.9640 | 0.9655 |
| KoDF | **TD** | 0.9298 | 0.8117 | 0.9298 | 0.7999 | 0.9035 | 0.9093 |
| IDForge | **STT** | 0.9968 | 0.9950 | 0.8899 | 0.9635 | 0.9935 | 0.9968 |
| DFDM | **STT-order** | 0.9969 | 0.9653 | 0.9798 | 0.9920 | 0.9969 | 0.9954 |
| UADFV | **LFS** | 0.9908 | 0.9708 | 0.9456 | 0.9908 | 0.9825 | 0.9731 |
| DeepSpeak-1.1 | **STT** | 0.9056 | 0.9048 | 0.8499 | 0.8595 | 0.9012 | 0.9056 |
| DeepSpeak-2.0 | **STT** | 0.8663 | 0.8636 | 0.8378 | 0.7678 | 0.8589 | 0.8663 |

---

# Table B — fusion operator on the final combination (App + TD + LFS + STT)

`App + TD + LFS + STT` under each decision rule. The cross-entropy row is fused
by mean-of-logits and NOT by DS or PCR6+: a CE branch has no Dirichlet, so its
uncertainty is a decision margin, and combining margins as belief masses would invent
evidence that was never measured. Mean-of-logits is what DiCoME's row (D) does.

`*in progress*` in the CE row is coverage, not failure: the CE counterparts of the four
branches were trained before the three corpora packed on 2026-10-13 and have not been
scored on them. The other rows cover every set.

**Mean appears twice on purpose.** F1 is a uniform mean and reads no reliability at all,
so comparing it against two fitted operators would confound the decision rule with the
weight fit. The second mean row is the same average on the fitted simplex, which is the
matched comparator for DS-C and PCR6+-R.

## AUROC (higher is better)

| fusion rule | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cross-entropy (mean of logits) | 0.9465 | 0.9397 | 0.9570 | 0.8377 | 0.8922 | 0.9671 | 0.9401 | 0.9907 | 0.9886 | *in progress* | *in progress* | *in progress* | 0.9400 |
| Mean (unweighted) | 0.9805 | 0.9589 | 0.9552 | 0.8583 | 0.9144 | 0.9739 | 0.9208 | 0.9951 | 0.9957 | 0.9850 | 0.9308 | 0.9158 | 0.9487 |
| Mean (fitted weights) | 0.9795 | 0.9601 | 0.9552 | 0.8602 | 0.9156 | 0.9747 | 0.9156 | 0.9957 | 0.9947 | 0.9863 | 0.9312 | 0.9147 | 0.9486 |
| DS-C | 0.9795 | 0.9690 | 0.9512 | 0.8578 | 0.9116 | 0.9748 | 0.9225 | 0.9966 | 0.9970 | 0.9867 | 0.9321 | 0.9222 | 0.9501 |
| PCR6+-R (ours) | 0.9788 | 0.9614 | 0.9569 | 0.8622 | 0.9112 | 0.9743 | 0.9112 | 0.9961 | 0.9966 | 0.9875 | 0.9277 | 0.9144 | 0.9482 |

## FPR at TPR = 90% (lower is better)

| fusion rule | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cross-entropy (mean of logits) | 0.1011 | 0.1966 | 0.1460 | 0.5136 | 0.3304 | 0.0778 | 0.1216 | 0.0295 | 0.0225 | *in progress* | *in progress* | *in progress* | 0.1710 |
| Mean (unweighted) | 0.0506 | 0.1404 | 0.1543 | 0.4086 | 0.2826 | 0.0657 | 0.2233 | 0.0117 | 0.0000 | 0.0204 | 0.1603 | 0.2405 | 0.1465 |
| Mean (fitted weights) | 0.0506 | 0.1124 | 0.1570 | 0.4039 | 0.2696 | 0.0646 | 0.2903 | 0.0094 | 0.0000 | 0.0204 | 0.1596 | 0.2480 | 0.1488 |
| DS-C | 0.0674 | 0.0899 | 0.1928 | 0.3991 | 0.2826 | 0.0715 | 0.2469 | 0.0054 | 0.0000 | 0.0204 | 0.1702 | 0.2163 | 0.1469 |
| PCR6+-R (ours) | 0.0674 | 0.1124 | 0.1570 | 0.3948 | 0.2696 | 0.0646 | 0.3834 | 0.0067 | 0.0056 | 0.0204 | 0.1645 | 0.2222 | 0.1557 |


---

# Table 1 — detection performance

Per dataset. **AUROC higher is better, FPR lower is better.** FPR is read at a MATCHED
true-positive rate, so every method is compared at the same sensitivity and no
calibration difference enters: of the genuine videos, what fraction is flagged when the
method is tuned to catch 90% (resp. 95%) of the fakes.

This is an ROC property, like AUROC. Whether a threshold fixed in advance survives the
domain shift is a different and far less flattering question, answered in
`SOTA_OPERATING_POINTS.md`. The two must not be quoted as the same measurement.

`*in progress*`: not scored on that corpus yet, not zero and not inapplicable. Our row covers
every set. The baselines were never run on the three corpora packed on 2026-10-13, and
DiCoME's public checkpoints cover only the five sets its paper reports.

A ROW'S `mean` IS OVER ITS OWN COVERAGE, which is not a comparison. Our row spans twelve
sets, most baselines nine, DiCoME five -- and the three corpora only we cover are not the
easy ones, so our own mean is read off a harder average than anyone else's. The two
matched columns are the comparable figures:

- `mean (all)` -- every set every row in the table resolves. DiCoME's five-set coverage
  sets that width, so it is honest for all nine methods but narrow.
- `mean (wide)` -- every set that every method covering nine or more sets resolves. The
  narrow rows say in progress there rather than shrinking the column for everyone.

Rank methods on a matched column and say which one. The per-row mean belongs in the
per-dataset discussion, not in a ranking.

**Cells not on our video census.** These rows were scored on a different set of
videos for that dataset, so the difference against our cell is not only a
difference in method:

- DiCoME (trained, their repo), CDFv3: 1918 videos against our 5418
- GenD-CLIP, CDFv3: 5118 videos against our 5418
- GenD-CLIP, KoDF: 1509 videos against our 2239
- GenD-PE, CDFv3: 5118 videos against our 5418
- GenD-PE, KoDF: 1509 videos against our 2239
- GenD-DINO, CDFv3: 5118 videos against our 5418
- GenD-DINO, KoDF: 1509 videos against our 2239
- FSFM, CDFv3: 5118 videos against our 5418
- FSFM, KoDF: 1509 videos against our 2239
- FSFM-Adapter, CDFv3: 5118 videos against our 5418
- FSFM-Adapter, KoDF: 1509 videos against our 2239
- Effort, CDFv3: 5118 videos against our 5418
- Effort, KoDF: 1509 videos against our 2239
- ForAda, CDFv3: 5118 videos against our 5418
- ForAda, KoDF: 1509 videos against our 2239

## AUROC (higher is better)

`mean (all)`: 5 sets (CDFv2, CDFv3, DFD, DFDC, DFDCP).  
`mean (wide)`: 9 sets (CDFv2, CDFv3, DFD, DFDC, DFDCP, FFIW, KoDF, IDForge, DFDM).

| method | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean | mean (all) | mean (wide) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours (App+TD+LFS+STT, PCR6+-R)** | 0.9788 | 0.9614 | 0.9569 | 0.8622 | 0.9112 | 0.9743 | 0.9112 | 0.9961 | 0.9966 | 0.9875 | 0.9277 | 0.9144 | 0.9482 | 0.9341 | 0.9499 |
| DiCoME (trained, their repo) | 0.9644 | 0.9516 | 0.9419 | 0.8825 | 0.8576 | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | 0.9196 | 0.9196 | *in progress* |
| GenD-CLIP | 0.9534 | 0.8458 | 0.9350 | 0.8708 | 0.9162 | 0.9278 | 0.8439 | 0.9839 | 0.9959 | *in progress* | *in progress* | *in progress* | 0.9192 | 0.9042 | 0.9192 |
| GenD-PE | 0.9548 | 0.8952 | 0.9203 | 0.8158 | 0.8860 | 0.9330 | 0.8344 | 0.9808 | 0.9815 | *in progress* | *in progress* | *in progress* | 0.9113 | 0.8944 | 0.9113 |
| GenD-DINO | 0.9177 | 0.8172 | 0.9165 | 0.8470 | 0.9114 | 0.9450 | 0.9057 | 0.9860 | 0.9903 | *in progress* | *in progress* | *in progress* | 0.9152 | 0.8820 | 0.9152 |
| FSFM | 0.9554 | 0.8751 | 0.9219 | 0.8844 | 0.9061 | 0.9168 | 0.8578 | 0.9566 | 0.9790 | *in progress* | *in progress* | *in progress* | 0.9170 | 0.9086 | 0.9170 |
| FSFM-Adapter | 0.8689 | 0.8381 | 0.7516 | 0.8318 | 0.8486 | 0.8565 | 0.7413 | 0.9077 | 0.8452 | *in progress* | *in progress* | *in progress* | 0.8322 | 0.8278 | 0.8322 |
| Effort | 0.9312 | 0.7975 | 0.9276 | 0.8536 | 0.8617 | 0.9252 | 0.8815 | 0.9583 | 0.9729 | *in progress* | *in progress* | *in progress* | 0.9011 | 0.8743 | 0.9011 |
| ForAda | 0.9544 | 0.7494 | 0.9395 | 0.8726 | 0.9289 | 0.9055 | 0.8823 | 0.9101 | 0.9713 | *in progress* | *in progress* | *in progress* | 0.9015 | 0.8890 | 0.9015 |

## FPR at TPR = 90% (lower is better)

`mean (all)`: 5 sets (CDFv2, CDFv3, DFD, DFDC, DFDCP).  
`mean (wide)`: 9 sets (CDFv2, CDFv3, DFD, DFDC, DFDCP, FFIW, KoDF, IDForge, DFDM).

| method | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean | mean (all) | mean (wide) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours (App+TD+LFS+STT, PCR6+-R)** | 0.0674 | 0.1124 | 0.1570 | 0.3948 | 0.2696 | 0.0646 | 0.3834 | 0.0067 | 0.0056 | 0.0204 | 0.1645 | 0.2222 | 0.1557 | 0.2002 | 0.1624 |
| DiCoME (trained, their repo) | 0.1011 | 0.1292 | 0.1901 | 0.3970 | 0.4783 | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | 0.2591 | 0.2591 | *in progress* |
| GenD-CLIP | 0.1685 | 0.5000 | 0.2507 | 0.4082 | 0.2217 | 0.2196 | 0.6377 | 0.0305 | 0.0056 | *in progress* | *in progress* | *in progress* | 0.2714 | 0.3098 | 0.2714 |
| GenD-PE | 0.1404 | 0.3539 | 0.3003 | 0.5296 | 0.3348 | 0.1712 | 0.7122 | 0.0495 | 0.0449 | *in progress* | *in progress* | *in progress* | 0.2930 | 0.3318 | 0.2930 |
| GenD-DINO | 0.2135 | 0.6348 | 0.3416 | 0.4985 | 0.2174 | 0.1481 | 0.4342 | 0.0171 | 0.0225 | *in progress* | *in progress* | *in progress* | 0.2809 | 0.3812 | 0.2809 |
| FSFM | 0.1236 | 0.4494 | 0.3251 | 0.3434 | 0.3348 | 0.2565 | 0.5682 | 0.1570 | 0.0449 | *in progress* | *in progress* | *in progress* | 0.2892 | 0.3153 | 0.2892 |
| FSFM-Adapter | 0.3427 | 0.4775 | 0.6556 | 0.5240 | 0.4391 | 0.4259 | 0.7767 | 0.3190 | 0.4326 | *in progress* | *in progress* | *in progress* | 0.4881 | 0.4878 | 0.4881 |
| Effort | 0.2640 | 0.6236 | 0.2507 | 0.5132 | 0.4348 | 0.2582 | 0.5211 | 0.1336 | 0.0562 | *in progress* | *in progress* | *in progress* | 0.3395 | 0.4173 | 0.3395 |
| ForAda | 0.1236 | 0.6854 | 0.2259 | 0.4095 | 0.2087 | 0.2859 | 0.4715 | 0.2929 | 0.0562 | *in progress* | *in progress* | *in progress* | 0.3066 | 0.3306 | 0.3066 |

## FPR at TPR = 95% (lower is better)

`mean (all)`: 5 sets (CDFv2, CDFv3, DFD, DFDC, DFDCP).  
`mean (wide)`: 9 sets (CDFv2, CDFv3, DFD, DFDC, DFDCP, FFIW, KoDF, IDForge, DFDM).

| method | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean | mean (all) | mean (wide) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours (App+TD+LFS+STT, PCR6+-R)** | 0.1067 | 0.2303 | 0.2397 | 0.5892 | 0.3609 | 0.1487 | 0.6576 | 0.0144 | 0.0056 | 0.0204 | 0.2712 | 0.3548 | 0.2500 | 0.3054 | 0.2615 |
| DiCoME (trained, their repo) | 0.1629 | 0.2584 | 0.4353 | 0.5944 | 0.7130 | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | 0.4328 | 0.4328 | *in progress* |
| GenD-CLIP | 0.2809 | 0.6966 | 0.3747 | 0.5646 | 0.3478 | 0.3746 | 0.8213 | 0.0870 | 0.0112 | *in progress* | *in progress* | *in progress* | 0.3954 | 0.4529 | 0.3954 |
| GenD-PE | 0.1910 | 0.6011 | 0.5289 | 0.7158 | 0.4522 | 0.3078 | 0.8536 | 0.1058 | 0.1011 | *in progress* | *in progress* | *in progress* | 0.4286 | 0.4978 | 0.4286 |
| GenD-DINO | 0.3483 | 0.7865 | 0.5234 | 0.6873 | 0.3522 | 0.2565 | 0.5980 | 0.0807 | 0.0225 | *in progress* | *in progress* | *in progress* | 0.4061 | 0.5395 | 0.4061 |
| FSFM | 0.1629 | 0.6180 | 0.4683 | 0.5123 | 0.4348 | 0.4081 | 0.7593 | 0.2266 | 0.1067 | *in progress* | *in progress* | *in progress* | 0.4108 | 0.4393 | 0.4108 |
| FSFM-Adapter | 0.4775 | 0.6404 | 0.7741 | 0.6540 | 0.5435 | 0.5458 | 0.8834 | 0.4372 | 0.5281 | *in progress* | *in progress* | *in progress* | 0.6093 | 0.6179 | 0.6093 |
| Effort | 0.3596 | 0.7978 | 0.5041 | 0.7201 | 0.5217 | 0.4063 | 0.7295 | 0.2417 | 0.1461 | *in progress* | *in progress* | *in progress* | 0.4919 | 0.5807 | 0.4919 |
| ForAda | 0.2360 | 0.8090 | 0.3857 | 0.5862 | 0.3217 | 0.4323 | 0.7146 | 0.4054 | 0.1517 | *in progress* | *in progress* | *in progress* | 0.4492 | 0.4677 | 0.4492 |


---

# Table 2 — probability quality and uncertainty

Per dataset. **All three lower is better.** ECE and Brier ask whether a reported
probability means what it says; E-AURC asks whether the model knows which of its own
answers to distrust, which the other two cannot see.

E-AURC subtracts the AURC an oracle ranking of the same predictions would reach, so the
base error rate drops out and what remains is the quality of the uncertainty ranking
alone — the only way to compare it across systems of different accuracy.

Ours ranks by the fused **vacuity**; the baselines rank by **1 − max p**, which is the
only uncertainty a cross-entropy detector emits. That asymmetry is the point of the
column, not a handicap we imposed.

`*in progress*`: not computable yet, and not zero or not applicable. Our row covers every
set. The baselines were never run on the three corpora packed on 2026-10-13 (UADFV,
DeepSpeak-1.1, DeepSpeak-2.0), and DiCoME's public checkpoints cover only the five
sets its paper reports; closing those cells needs a scoring pass, not a re-run.

A ROW'S `mean` IS OVER ITS OWN COVERAGE, which is not a comparison. Our row spans twelve
sets, most baselines nine, DiCoME five -- and the three corpora only we cover are not the
easy ones, so our own mean is read off a harder average than anyone else's. The two
matched columns are the comparable figures:

- `mean (all)` -- every set every row in the table resolves. DiCoME's five-set coverage
  sets that width, so it is honest for all nine methods but narrow.
- `mean (wide)` -- every set that every method covering nine or more sets resolves. The
  narrow rows say in progress there rather than shrinking the column for everyone.

Rank methods on a matched column and say which one. The per-row mean belongs in the
per-dataset discussion, not in a ranking.

## ECE-15 (lower is better)

`mean (all)`: 5 sets (CDFv2, CDFv3, DFD, DFDC, DFDCP).  
`mean (wide)`: 9 sets (CDFv2, CDFv3, DFD, DFDC, DFDCP, FFIW, KoDF, IDForge, DFDM).

| method | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean | mean (all) | mean (wide) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours (App+TD+LFS+STT, PCR6+-R)** | 0.1194 | 0.1732 | 0.1940 | 0.1296 | 0.1796 | 0.0623 | 0.1825 | 0.0505 | 0.0281 | 0.0988 | 0.0990 | 0.0933 | 0.1175 | 0.1592 | 0.1244 |
| DiCoME (trained, their repo) | 0.1119 | 0.1520 | 0.1191 | 0.0916 | 0.0480 | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | 0.1045 | 0.1045 | *in progress* |
| GenD-CLIP | 0.1180 | 0.1949 | 0.1162 | 0.0588 | 0.0641 | 0.0759 | 0.0823 | 0.1251 | 0.0597 | *in progress* | *in progress* | *in progress* | 0.0994 | 0.1104 | 0.0994 |
| GenD-PE | 0.0654 | 0.1986 | 0.1929 | 0.1095 | 0.0979 | 0.0732 | 0.2283 | 0.0681 | 0.0385 | *in progress* | *in progress* | *in progress* | 0.1191 | 0.1329 | 0.1191 |
| GenD-DINO | 0.0849 | 0.2693 | 0.2266 | 0.0681 | 0.0846 | 0.1054 | 0.2176 | 0.1150 | 0.0609 | *in progress* | *in progress* | *in progress* | 0.1369 | 0.1467 | 0.1369 |
| FSFM | 0.1373 | 0.2876 | 0.2697 | 0.0850 | 0.1334 | 0.1498 | 0.2652 | 0.1159 | 0.0984 | *in progress* | *in progress* | *in progress* | 0.1714 | 0.1826 | 0.1714 |
| FSFM-Adapter | 0.2054 | 0.3750 | 0.3975 | 0.1142 | 0.1470 | 0.1433 | 0.2761 | 0.1585 | 0.3383 | *in progress* | *in progress* | *in progress* | 0.2395 | 0.2478 | 0.2395 |
| Effort | 0.1242 | 0.2435 | 0.0939 | 0.0870 | 0.1168 | 0.1400 | 0.0564 | 0.1826 | 0.0677 | *in progress* | *in progress* | *in progress* | 0.1236 | 0.1331 | 0.1236 |
| ForAda | 0.0969 | 0.3643 | 0.1586 | 0.0944 | 0.1883 | 0.0848 | 0.2617 | 0.0658 | 0.0607 | *in progress* | *in progress* | *in progress* | 0.1528 | 0.1805 | 0.1528 |

## Brier (lower is better)

`mean (all)`: 5 sets (CDFv2, CDFv3, DFD, DFDC, DFDCP).  
`mean (wide)`: 9 sets (CDFv2, CDFv3, DFD, DFDC, DFDCP, FFIW, KoDF, IDForge, DFDM).

| method | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean | mean (all) | mean (wide) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours (App+TD+LFS+STT, PCR6+-R)** | 0.0730 | 0.0889 | 0.1406 | 0.1688 | 0.1626 | 0.0711 | 0.1584 | 0.0257 | 0.0150 | 0.0562 | 0.1162 | 0.1166 | 0.0994 | 0.1268 | 0.1005 |
| DiCoME (trained, their repo) | 0.0829 | 0.0838 | 0.0828 | 0.1498 | 0.1477 | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | 0.1094 | 0.1094 | *in progress* |
| GenD-CLIP | 0.1070 | 0.1002 | 0.0815 | 0.1503 | 0.1140 | 0.1151 | 0.1567 | 0.0722 | 0.0258 | *in progress* | *in progress* | *in progress* | 0.1025 | 0.1106 | 0.1025 |
| GenD-PE | 0.0795 | 0.1183 | 0.1377 | 0.1878 | 0.1421 | 0.1115 | 0.2347 | 0.0637 | 0.0319 | *in progress* | *in progress* | *in progress* | 0.1230 | 0.1331 | 0.1230 |
| GenD-DINO | 0.1101 | 0.1566 | 0.1546 | 0.1636 | 0.1155 | 0.1117 | 0.1877 | 0.0627 | 0.0223 | *in progress* | *in progress* | *in progress* | 0.1205 | 0.1401 | 0.1205 |
| FSFM | 0.0914 | 0.1465 | 0.1539 | 0.1476 | 0.1394 | 0.1490 | 0.2343 | 0.0995 | 0.0399 | *in progress* | *in progress* | *in progress* | 0.1335 | 0.1358 | 0.1335 |
| FSFM-Adapter | 0.1817 | 0.1819 | 0.2551 | 0.1830 | 0.1721 | 0.1813 | 0.2534 | 0.1548 | 0.1801 | *in progress* | *in progress* | *in progress* | 0.1937 | 0.1948 | 0.1937 |
| Effort | 0.1234 | 0.1215 | 0.0763 | 0.1615 | 0.1567 | 0.1329 | 0.1291 | 0.1298 | 0.0421 | *in progress* | *in progress* | *in progress* | 0.1192 | 0.1279 | 0.1192 |
| ForAda | 0.0915 | 0.2276 | 0.1060 | 0.1567 | 0.1612 | 0.1355 | 0.2302 | 0.1276 | 0.0440 | *in progress* | *in progress* | *in progress* | 0.1423 | 0.1486 | 0.1423 |

## E-AURC (lower is better)

`mean (all)`: 5 sets (CDFv2, CDFv3, DFD, DFDC, DFDCP).  
`mean (wide)`: 9 sets (CDFv2, CDFv3, DFD, DFDC, DFDCP, FFIW, KoDF, IDForge, DFDM).

| method | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean | mean (all) | mean (wide) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours (App+TD+LFS+STT, PCR6+-R)** | 0.0167 | 0.0109 | 0.0278 | 0.0886 | 0.0571 | 0.0197 | 0.0362 | 0.0047 | 0.0016 | 0.0448 | 0.1297 | 0.1005 | 0.0449 | 0.0402 | 0.0293 |
| DiCoME (trained, their repo) | 0.0164 | 0.0172 | 0.0141 | 0.0670 | 0.0634 | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | 0.0356 | 0.0356 | *in progress* |
| GenD-CLIP | 0.0342 | 0.0278 | 0.0134 | 0.0720 | 0.0439 | 0.0353 | 0.0564 | 0.0124 | 0.0015 | *in progress* | *in progress* | *in progress* | 0.0330 | 0.0382 | 0.0330 |
| GenD-PE | 0.0217 | 0.0462 | 0.0484 | 0.1177 | 0.0599 | 0.0417 | 0.0680 | 0.0107 | 0.0024 | *in progress* | *in progress* | *in progress* | 0.0463 | 0.0588 | 0.0463 |
| GenD-DINO | 0.0427 | 0.0838 | 0.0773 | 0.0913 | 0.0420 | 0.0402 | 0.1409 | 0.0076 | 0.0020 | *in progress* | *in progress* | *in progress* | 0.0586 | 0.0674 | 0.0586 |
| FSFM | 0.0227 | 0.0672 | 0.0630 | 0.0637 | 0.0507 | 0.0638 | 0.0854 | 0.0176 | 0.0026 | *in progress* | *in progress* | *in progress* | 0.0485 | 0.0535 | 0.0485 |
| FSFM-Adapter | 0.0758 | 0.0855 | 0.2099 | 0.0968 | 0.0678 | 0.0879 | 0.2186 | 0.0423 | 0.0671 | *in progress* | *in progress* | *in progress* | 0.1057 | 0.1072 | 0.1057 |
| Effort | 0.0416 | 0.0346 | 0.0120 | 0.0778 | 0.0627 | 0.0394 | 0.0421 | 0.0438 | 0.0037 | *in progress* | *in progress* | *in progress* | 0.0397 | 0.0457 | 0.0397 |
| ForAda | 0.0203 | 0.1552 | 0.0262 | 0.0772 | 0.0634 | 0.0593 | 0.0879 | 0.0436 | 0.0038 | *in progress* | *in progress* | *in progress* | 0.0597 | 0.0685 | 0.0597 |



# Proposal results — all tables, every dataset

Regenerated 2026-10-13T16:43:28-04:00. Framework: **App + TD + LFS + STT**, fused with PCR6+-R.

| id | thesis name | what it is |
|---|---|---|
| A | **App** | Appearance — PE-Core-L/336, LoRA + LayerNorm, frame mean |
| V | **TD** | Temporal Dynamics — V-JEPA 2 LoRA, 16 frames |
| O | **LFS** | Latent Facial Structure — CLIP/β-VAE orthogonal residual |
| S16ctl | **STT** | Spatio-Temporal Texture — PE-Core block 16 + real relative timestamps |
| P16ctl | **STT-order** | the same with learned slot positions |

Figure: `reports/branch_fusion/figures/uncertainty_riskcoverage.png`

---

# Table A — branch ablation

Ten groups, every dataset, fused with **PCR6+-R**. No baseline rows: this table answers
what each branch contributes, and a SOTA column here would invite two questions to be
read off one row. The comparison against the field is Table C.

`*in progress*` means the cell is not computable yet, not that it is zero or not applicable.
The three newly packed sets (UADFV, DeepSpeak-1.1, DeepSpeak-2.0) have no branch scores yet.

## AUROC (higher is better)

| branch group | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| App | 0.9460 | 0.8935 | 0.9505 | 0.8371 | 0.9000 | 0.9760 | 0.8563 | 0.9950 | 0.9653 | 0.9708 | 0.9048 | 0.8636 | 0.9216 |
| TD | 0.7383 | 0.9263 | 0.7967 | 0.7009 | 0.7408 | 0.8011 | 0.9656 | 0.8899 | 0.9798 | 0.9456 | 0.8499 | 0.8378 | 0.8477 |
| LFS | 0.9736 | 0.8801 | 0.9338 | 0.8808 | 0.8727 | 0.9354 | 0.8548 | 0.9635 | 0.9920 | 0.9908 | 0.8595 | 0.7678 | 0.9087 |
| STT-order | 0.9661 | 0.8942 | 0.9524 | 0.8360 | 0.8887 | 0.9640 | 0.9157 | 0.9935 | 0.9969 | 0.9825 | 0.9012 | 0.8589 | 0.9292 |
| STT | 0.9683 | 0.8864 | 0.9485 | 0.8176 | 0.8976 | 0.9655 | 0.9268 | 0.9968 | 0.9954 | 0.9731 | 0.9056 | 0.8663 | 0.9290 |
| App+TD+LFS *(deployed)* | 0.9645 | 0.9627 | 0.9513 | 0.8596 | 0.9058 | 0.9693 | 0.9430 | 0.9950 | 0.9927 | 0.9883 | 0.9268 | 0.9119 | 0.9476 |
| App+TD+LFS+STT-order | 0.9738 | 0.9625 | 0.9577 | 0.8593 | 0.9090 | 0.9701 | 0.9426 | 0.9961 | 0.9953 | 0.9867 | 0.9285 | 0.9210 | 0.9502 |
| App+TD+LFS+STT **(final)** | 0.9788 | 0.9612 | 0.9569 | 0.8622 | 0.9112 | 0.9743 | 0.9436 | 0.9961 | 0.9966 | 0.9875 | 0.9277 | 0.9144 | 0.9509 |
| App+TD+LFS+STT-order+STT | 0.9738 | 0.9625 | 0.9577 | 0.8593 | 0.9090 | 0.9701 | 0.9426 | 0.9961 | 0.9953 | 0.9867 | 0.9285 | 0.9210 | 0.9502 |
| App+LFS+STT-order+STT | 0.9847 | 0.9146 | 0.9528 | 0.8652 | 0.9224 | 0.9800 | 0.8974 | 0.9962 | 0.9939 | 0.9863 | 0.9146 | 0.8713 | 0.9399 |

## FPR at TPR = 90% (lower is better)

| branch group | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| App | 0.1910 | 0.3539 | 0.1708 | 0.4894 | 0.2565 | 0.0576 | 0.6055 | 0.0013 | 0.0730 | 0.0612 | 0.2542 | 0.4836 | 0.2499 |
| TD | 0.6180 | 0.2360 | 0.5868 | 0.7525 | 0.6304 | 0.4922 | 0.0397 | 0.3174 | 0.0393 | 0.1837 | 0.4922 | 0.4697 | 0.4048 |
| LFS | 0.0618 | 0.4213 | 0.2011 | 0.4289 | 0.3739 | 0.2242 | 0.6476 | 0.1145 | 0.0056 | 0.0000 | 0.3651 | 0.6554 | 0.2916 |
| STT-order | 0.1292 | 0.3876 | 0.1736 | 0.4592 | 0.3000 | 0.1101 | 0.2605 | 0.0057 | 0.0000 | 0.0000 | 0.2458 | 0.4799 | 0.2126 |
| STT | 0.0843 | 0.4213 | 0.2094 | 0.5110 | 0.3130 | 0.0974 | 0.2208 | 0.0020 | 0.0000 | 0.0408 | 0.2387 | 0.4842 | 0.2186 |
| App+TD+LFS *(deployed)* | 0.1011 | 0.0955 | 0.1653 | 0.3983 | 0.2870 | 0.0997 | 0.0471 | 0.0094 | 0.0112 | 0.0204 | 0.1773 | 0.2271 | 0.1366 |
| App+TD+LFS+STT-order | 0.0674 | 0.0674 | 0.1570 | 0.3978 | 0.2739 | 0.1032 | 0.0422 | 0.0037 | 0.0056 | 0.0204 | 0.1751 | 0.2281 | 0.1285 |
| App+TD+LFS+STT **(final)** | 0.0674 | 0.1124 | 0.1570 | 0.3948 | 0.2696 | 0.0646 | 0.0397 | 0.0067 | 0.0056 | 0.0204 | 0.1645 | 0.2222 | 0.1271 |
| App+TD+LFS+STT-order+STT | 0.0674 | 0.0674 | 0.1570 | 0.3978 | 0.2739 | 0.1032 | 0.0422 | 0.0037 | 0.0056 | 0.0204 | 0.1751 | 0.2281 | 0.1285 |
| App+LFS+STT-order+STT | 0.0449 | 0.3146 | 0.1653 | 0.4078 | 0.2348 | 0.0519 | 0.3995 | 0.0003 | 0.0000 | 0.0000 | 0.2147 | 0.4423 | 0.1897 |


---

# Table B — fusion operator on the final combination (App + TD + LFS + STT)

`App + TD + LFS + STT` under each decision rule. The cross-entropy row is fused
by mean-of-logits and NOT by DS or PCR6+: a CE branch has no Dirichlet, so its
uncertainty is a decision margin, and combining margins as belief masses would invent
evidence that was never measured. Mean-of-logits is what DiCoME's row (D) does.

## AUROC (higher is better)

| fusion rule | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cross-entropy (mean of logits) | 0.9465 | 0.9397 | 0.9570 | 0.8377 | 0.8922 | 0.9671 | 0.9401 | 0.9907 | 0.9886 | *in progress* | *in progress* | *in progress* | 0.9400 |
| Mean | 0.9805 | 0.9586 | 0.9552 | 0.8583 | 0.9144 | 0.9739 | 0.9449 | 0.9951 | 0.9957 | 0.9850 | 0.9308 | 0.9158 | 0.9507 |
| DS-C | 0.9795 | 0.9688 | 0.9512 | 0.8578 | 0.9116 | 0.9748 | 0.9459 | 0.9966 | 0.9970 | 0.9867 | 0.9321 | 0.9222 | 0.9520 |
| PCR6+-R (ours) | 0.9788 | 0.9612 | 0.9569 | 0.8622 | 0.9112 | 0.9743 | 0.9436 | 0.9961 | 0.9966 | 0.9875 | 0.9277 | 0.9144 | 0.9509 |

## FPR at TPR = 90% (lower is better)

| fusion rule | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cross-entropy (mean of logits) | 0.1011 | 0.1966 | 0.1460 | 0.5136 | 0.3304 | 0.0778 | 0.1216 | 0.0295 | 0.0225 | *in progress* | *in progress* | *in progress* | 0.1710 |
| Mean | 0.0506 | 0.1404 | 0.1543 | 0.4086 | 0.2826 | 0.0657 | 0.0471 | 0.0117 | 0.0000 | 0.0204 | 0.1603 | 0.2405 | 0.1319 |
| DS-C | 0.0674 | 0.0899 | 0.1928 | 0.3991 | 0.2826 | 0.0715 | 0.0447 | 0.0054 | 0.0000 | 0.0204 | 0.1702 | 0.2163 | 0.1300 |
| PCR6+-R (ours) | 0.0674 | 0.1124 | 0.1570 | 0.3948 | 0.2696 | 0.0646 | 0.0397 | 0.0067 | 0.0056 | 0.0204 | 0.1645 | 0.2222 | 0.1271 |


---

# Table 1 — detection performance

Per dataset. **AUROC higher is better, FPR lower is better.** FPR is read at a MATCHED
true-positive rate, so every method is compared at the same sensitivity and no
calibration difference enters: of the genuine videos, what fraction is flagged when the
method is tuned to catch 90% (resp. 95%) of the fakes.

This is an ROC property, like AUROC. Whether a threshold fixed in advance survives the
domain shift is a different and far less flattering question, answered in
`SOTA_OPERATING_POINTS.md`. The two must not be quoted as the same measurement.

## AUROC (higher is better)

| method | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours (App+TD+LFS+STT, PCR6+-R)** | 0.9788 | 0.9612 | 0.9569 | 0.8622 | 0.9112 | 0.9743 | 0.9436 | 0.9961 | 0.9966 | 0.9875 | 0.9277 | 0.9144 | 0.9509 |
| DiCoME (trained, their repo) | 0.9644 | 0.9516 | 0.9419 | 0.8825 | 0.8576 | — | — | — | — | — | — | — | 0.9196 |
| GenD-CLIP | 0.9534 | 0.8458 | 0.9350 | 0.8708 | 0.9162 | 0.9278 | 0.8329 | 0.9839 | 0.9959 | — | — | — | 0.9180 |
| GenD-PE | 0.9548 | 0.8952 | 0.9203 | 0.8158 | 0.8860 | 0.9330 | 0.8189 | 0.9808 | 0.9815 | — | — | — | 0.9096 |
| GenD-DINO | 0.9177 | 0.8172 | 0.9165 | 0.8470 | 0.9114 | 0.9450 | 0.8984 | 0.9860 | 0.9903 | — | — | — | 0.9144 |
| FSFM | 0.9554 | 0.8751 | 0.9219 | 0.8844 | 0.9061 | 0.9168 | 0.8467 | 0.9566 | 0.9790 | — | — | — | 0.9158 |
| FSFM-Adapter | 0.8689 | 0.8381 | 0.7516 | 0.8318 | 0.8486 | 0.8565 | 0.7405 | 0.9077 | 0.8452 | — | — | — | 0.8321 |
| Effort | 0.9312 | 0.7975 | 0.9276 | 0.8536 | 0.8617 | 0.9252 | 0.8725 | 0.9583 | 0.9729 | — | — | — | 0.9001 |
| ForAda | 0.9544 | 0.7494 | 0.9395 | 0.8726 | 0.9289 | 0.9055 | 0.8739 | 0.9101 | 0.9713 | — | — | — | 0.9006 |

## FPR at TPR = 90% (lower is better)

| method | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours (App+TD+LFS+STT, PCR6+-R)** | 0.0674 | 0.1124 | 0.1570 | 0.3948 | 0.2696 | 0.0646 | 0.0397 | 0.0067 | 0.0056 | 0.0204 | 0.1645 | 0.2222 | 0.1271 |
| DiCoME (trained, their repo) | 0.1011 | 0.1292 | 0.1901 | 0.3970 | 0.4783 | — | — | — | — | — | — | — | 0.2591 |
| GenD-CLIP | 0.1685 | 0.5000 | 0.2507 | 0.4082 | 0.2217 | 0.2196 | 0.6551 | 0.0305 | 0.0056 | — | — | — | 0.2733 |
| GenD-PE | 0.1404 | 0.3539 | 0.3003 | 0.5296 | 0.3348 | 0.1712 | 0.7345 | 0.0495 | 0.0449 | — | — | — | 0.2955 |
| GenD-DINO | 0.2135 | 0.6348 | 0.3416 | 0.4985 | 0.2174 | 0.1481 | 0.4417 | 0.0171 | 0.0225 | — | — | — | 0.2817 |
| FSFM | 0.1236 | 0.4494 | 0.3251 | 0.3434 | 0.3348 | 0.2565 | 0.6104 | 0.1570 | 0.0449 | — | — | — | 0.2939 |
| FSFM-Adapter | 0.3427 | 0.4775 | 0.6556 | 0.5240 | 0.4391 | 0.4259 | 0.8065 | 0.3190 | 0.4326 | — | — | — | 0.4914 |
| Effort | 0.2640 | 0.6236 | 0.2507 | 0.5132 | 0.4348 | 0.2582 | 0.5558 | 0.1336 | 0.0562 | — | — | — | 0.3433 |
| ForAda | 0.1236 | 0.6854 | 0.2259 | 0.4095 | 0.2087 | 0.2859 | 0.5161 | 0.2929 | 0.0562 | — | — | — | 0.3116 |

## FPR at TPR = 95% (lower is better)

| method | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours (App+TD+LFS+STT, PCR6+-R)** | 0.1067 | 0.2303 | 0.2397 | 0.5892 | 0.3609 | 0.1487 | 0.5931 | 0.0144 | 0.0056 | 0.0204 | 0.2712 | 0.3548 | 0.2446 |
| DiCoME (trained, their repo) | 0.1629 | 0.2584 | 0.4353 | 0.5944 | 0.7130 | — | — | — | — | — | — | — | 0.4328 |
| GenD-CLIP | 0.2809 | 0.6966 | 0.3747 | 0.5646 | 0.3478 | 0.3746 | 0.8412 | 0.0870 | 0.0112 | — | — | — | 0.3976 |
| GenD-PE | 0.1910 | 0.6011 | 0.5289 | 0.7158 | 0.4522 | 0.3078 | 0.8734 | 0.1058 | 0.1011 | — | — | — | 0.4308 |
| GenD-DINO | 0.3483 | 0.7865 | 0.5234 | 0.6873 | 0.3522 | 0.2565 | 0.6253 | 0.0807 | 0.0225 | — | — | — | 0.4092 |
| FSFM | 0.1629 | 0.6180 | 0.4683 | 0.5123 | 0.4348 | 0.4081 | 0.7742 | 0.2266 | 0.1067 | — | — | — | 0.4124 |
| FSFM-Adapter | 0.4775 | 0.6404 | 0.7741 | 0.6540 | 0.5435 | 0.5458 | 0.8933 | 0.4372 | 0.5281 | — | — | — | 0.6104 |
| Effort | 0.3596 | 0.7978 | 0.5041 | 0.7201 | 0.5217 | 0.4063 | 0.7444 | 0.2417 | 0.1461 | — | — | — | 0.4935 |
| ForAda | 0.2360 | 0.8090 | 0.3857 | 0.5862 | 0.3217 | 0.4323 | 0.7246 | 0.4054 | 0.1517 | — | — | — | 0.4503 |


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

`*in progress*`: not computable yet. The three newly packed sets have no branch scores.

## ECE-15 (lower is better)

| method | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours (App+TD+LFS+STT, PCR6+-R)** | 0.1194 | 0.1721 | 0.1940 | 0.1296 | 0.1796 | 0.0623 | 0.1504 | 0.0505 | 0.0281 | 0.0988 | 0.0990 | 0.0933 | 0.0988 | 0.0990 | 0.0933 | 0.1112 |
| DiCoME (trained, their repo) | 0.1119 | 0.1520 | 0.1191 | 0.0916 | 0.0480 | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | 0.1045 |
| GenD-CLIP | 0.1180 | 0.1949 | 0.1162 | 0.0588 | 0.0641 | 0.0759 | 0.0823 | 0.1251 | 0.0597 | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | 0.0994 |
| GenD-PE | 0.0654 | 0.1986 | 0.1929 | 0.1095 | 0.0979 | 0.0732 | 0.2283 | 0.0681 | 0.0385 | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | 0.1191 |
| GenD-DINO | 0.0849 | 0.2693 | 0.2266 | 0.0681 | 0.0846 | 0.1054 | 0.2176 | 0.1150 | 0.0609 | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | 0.1369 |
| FSFM | 0.1373 | 0.2876 | 0.2697 | 0.0850 | 0.1334 | 0.1498 | 0.2652 | 0.1159 | 0.0984 | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | 0.1714 |
| FSFM-Adapter | 0.2054 | 0.3750 | 0.3975 | 0.1142 | 0.1470 | 0.1433 | 0.2761 | 0.1585 | 0.3383 | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | 0.2395 |
| Effort | 0.1242 | 0.2435 | 0.0939 | 0.0870 | 0.1168 | 0.1400 | 0.0564 | 0.1826 | 0.0677 | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | 0.1236 |
| ForAda | 0.0969 | 0.3643 | 0.1586 | 0.0944 | 0.1883 | 0.0848 | 0.2617 | 0.0658 | 0.0607 | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | 0.1528 |

## Brier (lower is better)

| method | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours (App+TD+LFS+STT, PCR6+-R)** | 0.0730 | 0.0891 | 0.1406 | 0.1688 | 0.1626 | 0.0711 | 0.1129 | 0.0257 | 0.0150 | 0.0562 | 0.1162 | 0.1166 | 0.0562 | 0.1162 | 0.1166 | 0.0958 |
| DiCoME (trained, their repo) | 0.0829 | 0.0838 | 0.0828 | 0.1498 | 0.1477 | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | 0.1094 |
| GenD-CLIP | 0.1070 | 0.1002 | 0.0815 | 0.1503 | 0.1140 | 0.1151 | 0.1567 | 0.0722 | 0.0258 | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | 0.1025 |
| GenD-PE | 0.0795 | 0.1183 | 0.1377 | 0.1878 | 0.1421 | 0.1115 | 0.2347 | 0.0637 | 0.0319 | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | 0.1230 |
| GenD-DINO | 0.1101 | 0.1566 | 0.1546 | 0.1636 | 0.1155 | 0.1117 | 0.1877 | 0.0627 | 0.0223 | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | 0.1205 |
| FSFM | 0.0914 | 0.1465 | 0.1539 | 0.1476 | 0.1394 | 0.1490 | 0.2343 | 0.0995 | 0.0399 | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | 0.1335 |
| FSFM-Adapter | 0.1817 | 0.1819 | 0.2551 | 0.1830 | 0.1721 | 0.1813 | 0.2534 | 0.1548 | 0.1801 | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | 0.1937 |
| Effort | 0.1234 | 0.1215 | 0.0763 | 0.1615 | 0.1567 | 0.1329 | 0.1291 | 0.1298 | 0.0421 | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | 0.1192 |
| ForAda | 0.0915 | 0.2276 | 0.1060 | 0.1567 | 0.1612 | 0.1355 | 0.2302 | 0.1276 | 0.0440 | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | 0.1423 |

## E-AURC (lower is better)

| method | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours (App+TD+LFS+STT, PCR6+-R)** | 0.0167 | 0.0108 | 0.0278 | 0.0886 | 0.0571 | 0.0197 | 0.0289 | 0.0047 | 0.0016 | 0.0448 | 0.1297 | 0.1005 | 0.0448 | 0.1297 | 0.1005 | 0.0537 |
| DiCoME (trained, their repo) | 0.0164 | 0.0172 | 0.0141 | 0.0670 | 0.0634 | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | 0.0356 |
| GenD-CLIP | 0.0342 | 0.0278 | 0.0134 | 0.0720 | 0.0439 | 0.0353 | 0.0564 | 0.0124 | 0.0015 | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | 0.0330 |
| GenD-PE | 0.0217 | 0.0462 | 0.0484 | 0.1177 | 0.0599 | 0.0417 | 0.0680 | 0.0107 | 0.0024 | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | 0.0463 |
| GenD-DINO | 0.0427 | 0.0838 | 0.0773 | 0.0913 | 0.0420 | 0.0402 | 0.1409 | 0.0076 | 0.0020 | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | 0.0586 |
| FSFM | 0.0227 | 0.0672 | 0.0630 | 0.0637 | 0.0507 | 0.0638 | 0.0854 | 0.0176 | 0.0026 | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | 0.0485 |
| FSFM-Adapter | 0.0758 | 0.0855 | 0.2099 | 0.0968 | 0.0678 | 0.0879 | 0.2186 | 0.0423 | 0.0671 | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | 0.1057 |
| Effort | 0.0416 | 0.0346 | 0.0120 | 0.0778 | 0.0627 | 0.0394 | 0.0421 | 0.0438 | 0.0037 | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | 0.0397 |
| ForAda | 0.0203 | 0.1552 | 0.0262 | 0.0772 | 0.0634 | 0.0593 | 0.0879 | 0.0436 | 0.0038 | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | 0.0597 |



# Proposal results — all tables, every dataset

Generated 2026-10-12T23:51:13-04:00. Nine scored test sets; UADFV, DeepSpeak-1.1 and DeepSpeak-2.0 are
packed and configured but not yet scored, so their cells read *in progress*.

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
| A | 0.9460 | 0.8935 | 0.9505 | 0.8371 | 0.9000 | 0.9760 | 0.8563 | 0.9950 | 0.9653 | *in progress* | *in progress* | *in progress* | 0.9244 |
| V | 0.7383 | 0.9263 | 0.7967 | 0.7009 | 0.7408 | 0.8011 | 0.9656 | 0.8899 | 0.9798 | *in progress* | *in progress* | *in progress* | 0.8377 |
| O | 0.9736 | 0.8801 | 0.9338 | 0.8808 | 0.8727 | 0.9354 | 0.8548 | 0.9635 | 0.9920 | *in progress* | *in progress* | *in progress* | 0.9207 |
| P16ctl | 0.9661 | 0.8942 | 0.9524 | 0.8360 | 0.8887 | 0.9640 | 0.9157 | 0.9935 | 0.9969 | *in progress* | *in progress* | *in progress* | 0.9342 |
| S16ctl | 0.9683 | 0.8864 | 0.9485 | 0.8176 | 0.8976 | 0.9655 | 0.9268 | 0.9968 | 0.9954 | *in progress* | *in progress* | *in progress* | 0.9337 |
| A+V+O *(deployed)* | 0.9645 | 0.9627 | 0.9513 | 0.8596 | 0.9058 | 0.9693 | 0.9430 | 0.9950 | 0.9927 | *in progress* | *in progress* | *in progress* | 0.9493 |
| A+V+O+P16ctl | 0.9738 | 0.9625 | 0.9577 | 0.8593 | 0.9090 | 0.9701 | 0.9426 | 0.9961 | 0.9953 | *in progress* | *in progress* | *in progress* | 0.9518 |
| A+V+O+S16ctl | 0.9788 | 0.9612 | 0.9569 | 0.8622 | 0.9112 | 0.9743 | 0.9436 | 0.9961 | 0.9966 | *in progress* | *in progress* | *in progress* | 0.9534 |
| A+V+O+P16ctl+S16ctl **(final)** | 0.9738 | 0.9625 | 0.9577 | 0.8593 | 0.9090 | 0.9701 | 0.9426 | 0.9961 | 0.9953 | *in progress* | *in progress* | *in progress* | 0.9518 |
| A+O+P16ctl+S16ctl | 0.9847 | 0.9146 | 0.9528 | 0.8652 | 0.9224 | 0.9800 | 0.8974 | 0.9962 | 0.9939 | *in progress* | *in progress* | *in progress* | 0.9452 |

## FPR at TPR = 90% (lower is better)

| branch group | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.1910 | 0.3539 | 0.1708 | 0.4894 | 0.2565 | 0.0576 | 0.6055 | 0.0013 | 0.0730 | *in progress* | *in progress* | *in progress* | 0.2444 |
| V | 0.6180 | 0.2360 | 0.5868 | 0.7525 | 0.6304 | 0.4922 | 0.0397 | 0.3174 | 0.0393 | *in progress* | *in progress* | *in progress* | 0.4125 |
| O | 0.0618 | 0.4213 | 0.2011 | 0.4289 | 0.3739 | 0.2242 | 0.6476 | 0.1145 | 0.0056 | *in progress* | *in progress* | *in progress* | 0.2755 |
| P16ctl | 0.1292 | 0.3876 | 0.1736 | 0.4592 | 0.3000 | 0.1101 | 0.2605 | 0.0057 | 0.0000 | *in progress* | *in progress* | *in progress* | 0.2029 |
| S16ctl | 0.0843 | 0.4213 | 0.2094 | 0.5110 | 0.3130 | 0.0974 | 0.2208 | 0.0020 | 0.0000 | *in progress* | *in progress* | *in progress* | 0.2066 |
| A+V+O *(deployed)* | 0.1011 | 0.0955 | 0.1653 | 0.3983 | 0.2870 | 0.0997 | 0.0471 | 0.0094 | 0.0112 | *in progress* | *in progress* | *in progress* | 0.1350 |
| A+V+O+P16ctl | 0.0674 | 0.0674 | 0.1570 | 0.3978 | 0.2739 | 0.1032 | 0.0422 | 0.0037 | 0.0056 | *in progress* | *in progress* | *in progress* | 0.1243 |
| A+V+O+S16ctl | 0.0674 | 0.1124 | 0.1570 | 0.3948 | 0.2696 | 0.0646 | 0.0397 | 0.0067 | 0.0056 | *in progress* | *in progress* | *in progress* | 0.1242 |
| A+V+O+P16ctl+S16ctl **(final)** | 0.0674 | 0.0674 | 0.1570 | 0.3978 | 0.2739 | 0.1032 | 0.0422 | 0.0037 | 0.0056 | *in progress* | *in progress* | *in progress* | 0.1243 |
| A+O+P16ctl+S16ctl | 0.0449 | 0.3146 | 0.1653 | 0.4078 | 0.2348 | 0.0519 | 0.3995 | 0.0003 | 0.0000 | *in progress* | *in progress* | *in progress* | 0.1799 |


---

# Table B — fusion operator on the final five-branch combination

`A + V + O + P16ctl + S16ctl` under each decision rule. The cross-entropy row is fused
by mean-of-logits and NOT by DS or PCR6+: a CE branch has no Dirichlet, so its
uncertainty is a decision margin, and combining margins as belief masses would invent
evidence that was never measured. Mean-of-logits is what DiCoME's row (D) does.

## AUROC (higher is better)

| fusion rule | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cross-entropy (mean of logits) | 0.9715 | 0.9418 | 0.9504 | 0.8517 | 0.9099 | 0.9718 | 0.9375 | 0.9890 | 0.9958 | *in progress* | *in progress* | *in progress* | 0.9466 |
| Mean | 0.9823 | 0.9557 | 0.9572 | 0.8585 | 0.9149 | 0.9756 | 0.9427 | 0.9964 | 0.9970 | *in progress* | *in progress* | *in progress* | 0.9534 |
| DS-C | 0.9795 | 0.9688 | 0.9512 | 0.8578 | 0.9116 | 0.9748 | 0.9459 | 0.9966 | 0.9970 | *in progress* | *in progress* | *in progress* | 0.9537 |
| PCR6+-R (ours) | 0.9738 | 0.9625 | 0.9577 | 0.8593 | 0.9090 | 0.9701 | 0.9426 | 0.9961 | 0.9953 | *in progress* | *in progress* | *in progress* | 0.9518 |

## FPR at TPR = 90% (lower is better)

| fusion rule | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cross-entropy (mean of logits) | 0.0562 | 0.1910 | 0.1901 | 0.4488 | 0.2696 | 0.0622 | 0.1365 | 0.0345 | 0.0000 | *in progress* | *in progress* | *in progress* | 0.1543 |
| Mean | 0.0506 | 0.1517 | 0.1543 | 0.4048 | 0.2652 | 0.0686 | 0.0471 | 0.0054 | 0.0000 | *in progress* | *in progress* | *in progress* | 0.1275 |
| DS-C | 0.0674 | 0.0899 | 0.1928 | 0.3991 | 0.2826 | 0.0715 | 0.0447 | 0.0054 | 0.0000 | *in progress* | *in progress* | *in progress* | 0.1282 |
| PCR6+-R (ours) | 0.0674 | 0.0674 | 0.1570 | 0.3978 | 0.2739 | 0.1032 | 0.0422 | 0.0037 | 0.0056 | *in progress* | *in progress* | *in progress* | 0.1243 |


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

| method | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours (PCR6+-R, 5 branches)** | 0.9738 | 0.9625 | 0.9577 | 0.8593 | 0.9090 | 0.9701 | 0.9426 | 0.9961 | 0.9953 | 0.9518 |
| DiCoME (trained, their repo) | 0.9644 | 0.9516 | 0.9419 | 0.8825 | 0.8576 | — | — | — | — | 0.9196 |
| GenD-CLIP | 0.9534 | 0.8458 | 0.9350 | 0.8708 | 0.9162 | 0.9278 | 0.8329 | 0.9839 | 0.9959 | 0.9180 |
| GenD-PE | 0.9548 | 0.8952 | 0.9203 | 0.8158 | 0.8860 | 0.9330 | 0.8189 | 0.9808 | 0.9815 | 0.9096 |
| GenD-DINO | 0.9177 | 0.8172 | 0.9165 | 0.8470 | 0.9114 | 0.9450 | 0.8984 | 0.9860 | 0.9903 | 0.9144 |
| FSFM | 0.9554 | 0.8751 | 0.9219 | 0.8844 | 0.9061 | 0.9168 | 0.8467 | 0.9566 | 0.9790 | 0.9158 |
| FSFM-Adapter | 0.8689 | 0.8381 | 0.7516 | 0.8318 | 0.8486 | 0.8565 | 0.7405 | 0.9077 | 0.8452 | 0.8321 |
| Effort | 0.9312 | 0.7975 | 0.9276 | 0.8536 | 0.8617 | 0.9252 | 0.8725 | 0.9583 | 0.9729 | 0.9001 |
| ForAda | 0.9544 | 0.7494 | 0.9395 | 0.8726 | 0.9289 | 0.9055 | 0.8739 | 0.9101 | 0.9713 | 0.9006 |

## FPR at TPR = 90% (lower is better)

| method | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours (PCR6+-R, 5 branches)** | 0.0674 | 0.0674 | 0.1570 | 0.3978 | 0.2739 | 0.1032 | 0.0422 | 0.0037 | 0.0056 | 0.1243 |
| DiCoME (trained, their repo) | 0.1011 | 0.1292 | 0.1901 | 0.3970 | 0.4783 | — | — | — | — | 0.2591 |
| GenD-CLIP | 0.1685 | 0.5000 | 0.2507 | 0.4082 | 0.2217 | 0.2196 | 0.6551 | 0.0305 | 0.0056 | 0.2733 |
| GenD-PE | 0.1404 | 0.3539 | 0.3003 | 0.5296 | 0.3348 | 0.1712 | 0.7345 | 0.0495 | 0.0449 | 0.2955 |
| GenD-DINO | 0.2135 | 0.6348 | 0.3416 | 0.4985 | 0.2174 | 0.1481 | 0.4417 | 0.0171 | 0.0225 | 0.2817 |
| FSFM | 0.1236 | 0.4494 | 0.3251 | 0.3434 | 0.3348 | 0.2565 | 0.6104 | 0.1570 | 0.0449 | 0.2939 |
| FSFM-Adapter | 0.3427 | 0.4775 | 0.6556 | 0.5240 | 0.4391 | 0.4259 | 0.8065 | 0.3190 | 0.4326 | 0.4914 |
| Effort | 0.2640 | 0.6236 | 0.2507 | 0.5132 | 0.4348 | 0.2582 | 0.5558 | 0.1336 | 0.0562 | 0.3433 |
| ForAda | 0.1236 | 0.6854 | 0.2259 | 0.4095 | 0.2087 | 0.2859 | 0.5161 | 0.2929 | 0.0562 | 0.3116 |

## FPR at TPR = 95% (lower is better)

| method | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours (PCR6+-R, 5 branches)** | 0.1292 | 0.2079 | 0.2397 | 0.5862 | 0.3696 | 0.1660 | 0.5633 | 0.0157 | 0.0112 | 0.2543 |
| DiCoME (trained, their repo) | 0.1629 | 0.2584 | 0.4353 | 0.5944 | 0.7130 | — | — | — | — | 0.4328 |
| GenD-CLIP | 0.2809 | 0.6966 | 0.3747 | 0.5646 | 0.3478 | 0.3746 | 0.8412 | 0.0870 | 0.0112 | 0.3976 |
| GenD-PE | 0.1910 | 0.6011 | 0.5289 | 0.7158 | 0.4522 | 0.3078 | 0.8734 | 0.1058 | 0.1011 | 0.4308 |
| GenD-DINO | 0.3483 | 0.7865 | 0.5234 | 0.6873 | 0.3522 | 0.2565 | 0.6253 | 0.0807 | 0.0225 | 0.4092 |
| FSFM | 0.1629 | 0.6180 | 0.4683 | 0.5123 | 0.4348 | 0.4081 | 0.7742 | 0.2266 | 0.1067 | 0.4124 |
| FSFM-Adapter | 0.4775 | 0.6404 | 0.7741 | 0.6540 | 0.5435 | 0.5458 | 0.8933 | 0.4372 | 0.5281 | 0.6104 |
| Effort | 0.3596 | 0.7978 | 0.5041 | 0.7201 | 0.5217 | 0.4063 | 0.7444 | 0.2417 | 0.1461 | 0.4935 |
| ForAda | 0.2360 | 0.8090 | 0.3857 | 0.5862 | 0.3217 | 0.4323 | 0.7246 | 0.4054 | 0.1517 | 0.4503 |


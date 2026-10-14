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

Every baseline cell is scored on the same videos as our row.

## AUROC (higher is better)

`mean (all)`: 12 sets (CDFv2, CDFv3, DFD, DFDC, DFDCP, FFIW, KoDF, IDForge, DFDM, UADFV, DeepSpeak-1.1, DeepSpeak-2.0).  
`mean (wide)`: 12 sets (CDFv2, CDFv3, DFD, DFDC, DFDCP, FFIW, KoDF, IDForge, DFDM, UADFV, DeepSpeak-1.1, DeepSpeak-2.0).

| method | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean | mean (all) | mean (wide) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours (App+TD+LFS+STT, PCR6+-R)** | 0.9788 | 0.9614 | 0.9569 | 0.8622 | 0.9112 | 0.9743 | 0.9112 | 0.9961 | 0.9966 | 0.9875 | 0.9277 | 0.9144 | 0.9482 | 0.9482 | 0.9482 |
| DiCoME (trained, their repo) | 0.9644 | 0.8435 | 0.9419 | 0.8825 | 0.8576 | 0.9511 | 0.7835 | 0.9701 | 0.9928 | 0.9842 | 0.8491 | 0.7630 | 0.8986 | 0.8986 | 0.8986 |
| GenD-CLIP | 0.9534 | 0.8484 | 0.9350 | 0.8708 | 0.9162 | 0.9278 | 0.8011 | 0.9839 | 0.9959 | 0.9921 | 0.8580 | 0.7445 | 0.9022 | 0.9022 | 0.9022 |
| GenD-PE | 0.9548 | 0.8953 | 0.9203 | 0.8158 | 0.8860 | 0.9330 | 0.8031 | 0.9808 | 0.9815 | 0.9758 | 0.8613 | 0.7865 | 0.8995 | 0.8995 | 0.8995 |
| GenD-DINO | 0.9177 | 0.8218 | 0.9165 | 0.8470 | 0.9114 | 0.9450 | 0.8675 | 0.9860 | 0.9903 | 0.9871 | 0.8403 | 0.7701 | 0.9001 | 0.9001 | 0.9001 |
| FSFM | 0.9554 | 0.8744 | 0.9219 | 0.8844 | 0.9061 | 0.9168 | 0.7731 | 0.9566 | 0.9790 | 0.9754 | 0.8601 | 0.7452 | 0.8957 | 0.8957 | 0.8957 |
| FSFM-Adapter | 0.8689 | 0.8342 | 0.7516 | 0.8318 | 0.8486 | 0.8565 | 0.7245 | 0.9077 | 0.8452 | 0.9933 | 0.7741 | 0.6550 | 0.8243 | 0.8243 | 0.8243 |
| Effort | 0.9312 | 0.7915 | 0.9276 | 0.8536 | 0.8617 | 0.9252 | 0.8180 | 0.9583 | 0.9729 | 0.9742 | 0.8396 | 0.6622 | 0.8763 | 0.8763 | 0.8763 |
| ForAda | 0.9544 | 0.7505 | 0.9395 | 0.8726 | 0.9289 | 0.9055 | 0.8133 | 0.9101 | 0.9713 | 0.9942 | 0.7597 | 0.6947 | 0.8746 | 0.8746 | 0.8746 |

## FPR at TPR = 90% (lower is better)

`mean (all)`: 12 sets (CDFv2, CDFv3, DFD, DFDC, DFDCP, FFIW, KoDF, IDForge, DFDM, UADFV, DeepSpeak-1.1, DeepSpeak-2.0).  
`mean (wide)`: 12 sets (CDFv2, CDFv3, DFD, DFDC, DFDCP, FFIW, KoDF, IDForge, DFDM, UADFV, DeepSpeak-1.1, DeepSpeak-2.0).

| method | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean | mean (all) | mean (wide) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours (App+TD+LFS+STT, PCR6+-R)** | 0.0674 | 0.1124 | 0.1570 | 0.3948 | 0.2696 | 0.0646 | 0.3834 | 0.0067 | 0.0056 | 0.0204 | 0.1645 | 0.2222 | 0.1557 | 0.1557 | 0.1557 |
| DiCoME (trained, their repo) | 0.1011 | 0.5955 | 0.1901 | 0.3970 | 0.4783 | 0.1573 | 0.7457 | 0.0947 | 0.0112 | 0.0000 | 0.4449 | 0.6973 | 0.3261 | 0.3261 | 0.3261 |
| GenD-CLIP | 0.1685 | 0.4775 | 0.2507 | 0.4082 | 0.2217 | 0.2196 | 0.7084 | 0.0305 | 0.0056 | 0.0000 | 0.4541 | 0.7558 | 0.3084 | 0.3084 | 0.3084 |
| GenD-PE | 0.1404 | 0.3539 | 0.3003 | 0.5296 | 0.3348 | 0.1712 | 0.7022 | 0.0495 | 0.0449 | 0.0612 | 0.4011 | 0.6634 | 0.3127 | 0.3127 | 0.3127 |
| GenD-DINO | 0.2135 | 0.5955 | 0.3416 | 0.4985 | 0.2174 | 0.1481 | 0.5471 | 0.0171 | 0.0225 | 0.0204 | 0.4781 | 0.6715 | 0.3143 | 0.3143 | 0.3143 |
| FSFM | 0.1236 | 0.4494 | 0.3251 | 0.3434 | 0.3348 | 0.2565 | 0.7047 | 0.1570 | 0.0449 | 0.0204 | 0.4845 | 0.6575 | 0.3252 | 0.3252 | 0.3252 |
| FSFM-Adapter | 0.3427 | 0.4831 | 0.6556 | 0.5240 | 0.4391 | 0.4259 | 0.7978 | 0.3190 | 0.4326 | 0.0204 | 0.6250 | 0.7719 | 0.4864 | 0.4864 | 0.4864 |
| Effort | 0.2640 | 0.6292 | 0.2507 | 0.5132 | 0.4348 | 0.2582 | 0.6675 | 0.1336 | 0.0562 | 0.0000 | 0.5395 | 0.7950 | 0.3785 | 0.3785 | 0.3785 |
| ForAda | 0.1236 | 0.6854 | 0.2259 | 0.4095 | 0.2087 | 0.2859 | 0.6216 | 0.2929 | 0.0562 | 0.0000 | 0.7274 | 0.8159 | 0.3711 | 0.3711 | 0.3711 |

## FPR at TPR = 95% (lower is better)

`mean (all)`: 12 sets (CDFv2, CDFv3, DFD, DFDC, DFDCP, FFIW, KoDF, IDForge, DFDM, UADFV, DeepSpeak-1.1, DeepSpeak-2.0).  
`mean (wide)`: 12 sets (CDFv2, CDFv3, DFD, DFDC, DFDCP, FFIW, KoDF, IDForge, DFDM, UADFV, DeepSpeak-1.1, DeepSpeak-2.0).

| method | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean | mean (all) | mean (wide) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours (App+TD+LFS+STT, PCR6+-R)** | 0.1067 | 0.2303 | 0.2397 | 0.5892 | 0.3609 | 0.1487 | 0.6576 | 0.0144 | 0.0056 | 0.0204 | 0.2712 | 0.3548 | 0.2500 | 0.2500 | 0.2500 |
| DiCoME (trained, their repo) | 0.1629 | 0.8202 | 0.4353 | 0.5944 | 0.7130 | 0.2761 | 0.8697 | 0.1758 | 0.0225 | 0.0000 | 0.6970 | 0.8631 | 0.4692 | 0.4692 | 0.4692 |
| GenD-CLIP | 0.2809 | 0.6798 | 0.3747 | 0.5646 | 0.3478 | 0.3746 | 0.8486 | 0.0870 | 0.0112 | 0.0000 | 0.6893 | 0.8900 | 0.4290 | 0.4290 | 0.4290 |
| GenD-PE | 0.1910 | 0.5955 | 0.5289 | 0.7158 | 0.4522 | 0.3078 | 0.8300 | 0.1058 | 0.1011 | 0.1633 | 0.5360 | 0.8019 | 0.4441 | 0.4441 | 0.4441 |
| GenD-DINO | 0.3483 | 0.7865 | 0.5234 | 0.6873 | 0.3522 | 0.2565 | 0.7246 | 0.0807 | 0.0225 | 0.0408 | 0.6730 | 0.7837 | 0.4400 | 0.4400 | 0.4400 |
| FSFM | 0.1629 | 0.6067 | 0.4683 | 0.5123 | 0.4348 | 0.4081 | 0.8350 | 0.2266 | 0.1067 | 0.0612 | 0.6179 | 0.8164 | 0.4381 | 0.4381 | 0.4381 |
| FSFM-Adapter | 0.4775 | 0.6517 | 0.7741 | 0.6540 | 0.5435 | 0.5458 | 0.8908 | 0.4372 | 0.5281 | 0.0612 | 0.7938 | 0.8583 | 0.6013 | 0.6013 | 0.6013 |
| Effort | 0.3596 | 0.7978 | 0.5041 | 0.7201 | 0.5217 | 0.4063 | 0.8164 | 0.2417 | 0.1461 | 0.0816 | 0.7415 | 0.8959 | 0.5194 | 0.5194 | 0.5194 |
| ForAda | 0.2360 | 0.7921 | 0.3857 | 0.5862 | 0.3217 | 0.4323 | 0.7928 | 0.4054 | 0.1517 | 0.0408 | 0.8651 | 0.9034 | 0.4928 | 0.4928 | 0.4928 |


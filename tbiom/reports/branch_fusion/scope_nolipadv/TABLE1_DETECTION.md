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
- GenD-CLIP, KoDF: 1509 videos against our 2239
- GenD-PE, KoDF: 1509 videos against our 2239
- GenD-DINO, KoDF: 1509 videos against our 2239
- FSFM, KoDF: 1509 videos against our 2239
- FSFM-Adapter, KoDF: 1509 videos against our 2239
- Effort, KoDF: 1509 videos against our 2239
- ForAda, KoDF: 1509 videos against our 2239

## AUROC (higher is better)

`mean (all)`: 11 sets (CDFv2, DFD, DFDC, DFDCP, FFIW, KoDF, IDForge, DFDM, UADFV, DeepSpeak-1.1, DeepSpeak-2.0).  
`mean (wide)`: 11 sets (CDFv2, DFD, DFDC, DFDCP, FFIW, KoDF, IDForge, DFDM, UADFV, DeepSpeak-1.1, DeepSpeak-2.0).

| method | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean | mean (all) | mean (wide) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours (App+TD+LFS+STT, PCR6+-R)** | 0.9788 | 0.9614 | 0.9569 | 0.8622 | 0.9112 | 0.9743 | 0.9112 | 0.9961 | 0.9966 | 0.9875 | 0.9277 | 0.9144 | 0.9482 | 0.9470 | 0.9470 |
| DiCoME (trained, their repo) | 0.9644 | 0.9516 | 0.9419 | 0.8825 | 0.8576 | 0.9511 | 0.7835 | 0.9701 | 0.9928 | 0.9842 | 0.8491 | 0.7630 | 0.9077 | 0.9037 | 0.9037 |
| GenD-CLIP | 0.9534 | *in progress* | 0.9350 | 0.8708 | 0.9162 | 0.9278 | 0.8439 | 0.9839 | 0.9959 | 0.9921 | 0.8580 | 0.7445 | 0.9110 | 0.9110 | 0.9110 |
| GenD-PE | 0.9548 | *in progress* | 0.9203 | 0.8158 | 0.8860 | 0.9330 | 0.8344 | 0.9808 | 0.9815 | 0.9758 | 0.8613 | 0.7865 | 0.9027 | 0.9027 | 0.9027 |
| GenD-DINO | 0.9177 | *in progress* | 0.9165 | 0.8470 | 0.9114 | 0.9450 | 0.9057 | 0.9860 | 0.9903 | 0.9871 | 0.8403 | 0.7701 | 0.9106 | 0.9106 | 0.9106 |
| FSFM | 0.9554 | *in progress* | 0.9219 | 0.8844 | 0.9061 | 0.9168 | 0.8578 | 0.9566 | 0.9790 | 0.9754 | 0.8601 | 0.7452 | 0.9053 | 0.9053 | 0.9053 |
| FSFM-Adapter | 0.8689 | *in progress* | 0.7516 | 0.8318 | 0.8486 | 0.8565 | 0.7413 | 0.9077 | 0.8452 | 0.9933 | 0.7741 | 0.6550 | 0.8249 | 0.8249 | 0.8249 |
| Effort | 0.9312 | *in progress* | 0.9276 | 0.8536 | 0.8617 | 0.9252 | 0.8815 | 0.9583 | 0.9729 | 0.9742 | 0.8396 | 0.6622 | 0.8898 | 0.8898 | 0.8898 |
| ForAda | 0.9544 | *in progress* | 0.9395 | 0.8726 | 0.9289 | 0.9055 | 0.8823 | 0.9101 | 0.9713 | 0.9942 | 0.7597 | 0.6947 | 0.8921 | 0.8921 | 0.8921 |

## FPR at TPR = 90% (lower is better)

`mean (all)`: 11 sets (CDFv2, DFD, DFDC, DFDCP, FFIW, KoDF, IDForge, DFDM, UADFV, DeepSpeak-1.1, DeepSpeak-2.0).  
`mean (wide)`: 11 sets (CDFv2, DFD, DFDC, DFDCP, FFIW, KoDF, IDForge, DFDM, UADFV, DeepSpeak-1.1, DeepSpeak-2.0).

| method | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean | mean (all) | mean (wide) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours (App+TD+LFS+STT, PCR6+-R)** | 0.0674 | 0.1124 | 0.1570 | 0.3948 | 0.2696 | 0.0646 | 0.3834 | 0.0067 | 0.0056 | 0.0204 | 0.1645 | 0.2222 | 0.1557 | 0.1597 | 0.1597 |
| DiCoME (trained, their repo) | 0.1011 | 0.1292 | 0.1901 | 0.3970 | 0.4783 | 0.1573 | 0.7457 | 0.0947 | 0.0112 | 0.0000 | 0.4449 | 0.6973 | 0.2872 | 0.3016 | 0.3016 |
| GenD-CLIP | 0.1685 | *in progress* | 0.2507 | 0.4082 | 0.2217 | 0.2196 | 0.6377 | 0.0305 | 0.0056 | 0.0000 | 0.4541 | 0.7558 | 0.2866 | 0.2866 | 0.2866 |
| GenD-PE | 0.1404 | *in progress* | 0.3003 | 0.5296 | 0.3348 | 0.1712 | 0.7122 | 0.0495 | 0.0449 | 0.0612 | 0.4011 | 0.6634 | 0.3099 | 0.3099 | 0.3099 |
| GenD-DINO | 0.2135 | *in progress* | 0.3416 | 0.4985 | 0.2174 | 0.1481 | 0.4342 | 0.0171 | 0.0225 | 0.0204 | 0.4781 | 0.6715 | 0.2784 | 0.2784 | 0.2784 |
| FSFM | 0.1236 | *in progress* | 0.3251 | 0.3434 | 0.3348 | 0.2565 | 0.5682 | 0.1570 | 0.0449 | 0.0204 | 0.4845 | 0.6575 | 0.3015 | 0.3015 | 0.3015 |
| FSFM-Adapter | 0.3427 | *in progress* | 0.6556 | 0.5240 | 0.4391 | 0.4259 | 0.7767 | 0.3190 | 0.4326 | 0.0204 | 0.6250 | 0.7719 | 0.4848 | 0.4848 | 0.4848 |
| Effort | 0.2640 | *in progress* | 0.2507 | 0.5132 | 0.4348 | 0.2582 | 0.5211 | 0.1336 | 0.0562 | 0.0000 | 0.5395 | 0.7950 | 0.3424 | 0.3424 | 0.3424 |
| ForAda | 0.1236 | *in progress* | 0.2259 | 0.4095 | 0.2087 | 0.2859 | 0.4715 | 0.2929 | 0.0562 | 0.0000 | 0.7274 | 0.8159 | 0.3289 | 0.3289 | 0.3289 |

## FPR at TPR = 95% (lower is better)

`mean (all)`: 11 sets (CDFv2, DFD, DFDC, DFDCP, FFIW, KoDF, IDForge, DFDM, UADFV, DeepSpeak-1.1, DeepSpeak-2.0).  
`mean (wide)`: 11 sets (CDFv2, DFD, DFDC, DFDCP, FFIW, KoDF, IDForge, DFDM, UADFV, DeepSpeak-1.1, DeepSpeak-2.0).

| method | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean | mean (all) | mean (wide) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours (App+TD+LFS+STT, PCR6+-R)** | 0.1067 | 0.2303 | 0.2397 | 0.5892 | 0.3609 | 0.1487 | 0.6576 | 0.0144 | 0.0056 | 0.0204 | 0.2712 | 0.3548 | 0.2500 | 0.2517 | 0.2517 |
| DiCoME (trained, their repo) | 0.1629 | 0.2584 | 0.4353 | 0.5944 | 0.7130 | 0.2761 | 0.8697 | 0.1758 | 0.0225 | 0.0000 | 0.6970 | 0.8631 | 0.4224 | 0.4373 | 0.4373 |
| GenD-CLIP | 0.2809 | *in progress* | 0.3747 | 0.5646 | 0.3478 | 0.3746 | 0.8213 | 0.0870 | 0.0112 | 0.0000 | 0.6893 | 0.8900 | 0.4038 | 0.4038 | 0.4038 |
| GenD-PE | 0.1910 | *in progress* | 0.5289 | 0.7158 | 0.4522 | 0.3078 | 0.8536 | 0.1058 | 0.1011 | 0.1633 | 0.5360 | 0.8019 | 0.4325 | 0.4325 | 0.4325 |
| GenD-DINO | 0.3483 | *in progress* | 0.5234 | 0.6873 | 0.3522 | 0.2565 | 0.5980 | 0.0807 | 0.0225 | 0.0408 | 0.6730 | 0.7837 | 0.3969 | 0.3969 | 0.3969 |
| FSFM | 0.1629 | *in progress* | 0.4683 | 0.5123 | 0.4348 | 0.4081 | 0.7593 | 0.2266 | 0.1067 | 0.0612 | 0.6179 | 0.8164 | 0.4159 | 0.4159 | 0.4159 |
| FSFM-Adapter | 0.4775 | *in progress* | 0.7741 | 0.6540 | 0.5435 | 0.5458 | 0.8834 | 0.4372 | 0.5281 | 0.0612 | 0.7938 | 0.8583 | 0.5961 | 0.5961 | 0.5961 |
| Effort | 0.3596 | *in progress* | 0.5041 | 0.7201 | 0.5217 | 0.4063 | 0.7295 | 0.2417 | 0.1461 | 0.0816 | 0.7415 | 0.8959 | 0.4862 | 0.4862 | 0.4862 |
| ForAda | 0.2360 | *in progress* | 0.3857 | 0.5862 | 0.3217 | 0.4323 | 0.7146 | 0.4054 | 0.1517 | 0.0408 | 0.8651 | 0.9034 | 0.4584 | 0.4584 | 0.4584 |


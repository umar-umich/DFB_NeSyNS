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
- GenD-PE, CDFv3: 5118 videos against our 5418
- GenD-DINO, CDFv3: 5118 videos against our 5418
- FSFM, CDFv3: 5118 videos against our 5418
- FSFM-Adapter, CDFv3: 5118 videos against our 5418
- Effort, CDFv3: 5118 videos against our 5418
- ForAda, CDFv3: 5118 videos against our 5418

## AUROC (higher is better)

`mean (all)`: 5 sets (CDFv2, CDFv3, DFD, DFDC, DFDCP).  
`mean (wide)`: 9 sets (CDFv2, CDFv3, DFD, DFDC, DFDCP, FFIW, KoDF, IDForge, DFDM).

| method | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean | mean (all) | mean (wide) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours (App+TD+LFS+STT, PCR6+-R)** | 0.9788 | 0.9614 | 0.9569 | 0.8622 | 0.9112 | 0.9743 | 0.9494 | 0.9961 | 0.9966 | 0.9875 | 0.9277 | 0.9144 | 0.9514 | 0.9341 | 0.9541 |
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
| **Ours (App+TD+LFS+STT, PCR6+-R)** | 0.0674 | 0.1124 | 0.1570 | 0.3948 | 0.2696 | 0.0646 | 0.0323 | 0.0067 | 0.0056 | 0.0204 | 0.1645 | 0.2222 | 0.1265 | 0.2002 | 0.1234 |
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
| **Ours (App+TD+LFS+STT, PCR6+-R)** | 0.1067 | 0.2303 | 0.2397 | 0.5892 | 0.3609 | 0.1487 | 0.4591 | 0.0144 | 0.0056 | 0.0204 | 0.2712 | 0.3548 | 0.2334 | 0.3054 | 0.2394 |
| DiCoME (trained, their repo) | 0.1629 | 0.2584 | 0.4353 | 0.5944 | 0.7130 | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | 0.4328 | 0.4328 | *in progress* |
| GenD-CLIP | 0.2809 | 0.6966 | 0.3747 | 0.5646 | 0.3478 | 0.3746 | 0.8213 | 0.0870 | 0.0112 | *in progress* | *in progress* | *in progress* | 0.3954 | 0.4529 | 0.3954 |
| GenD-PE | 0.1910 | 0.6011 | 0.5289 | 0.7158 | 0.4522 | 0.3078 | 0.8536 | 0.1058 | 0.1011 | *in progress* | *in progress* | *in progress* | 0.4286 | 0.4978 | 0.4286 |
| GenD-DINO | 0.3483 | 0.7865 | 0.5234 | 0.6873 | 0.3522 | 0.2565 | 0.5980 | 0.0807 | 0.0225 | *in progress* | *in progress* | *in progress* | 0.4061 | 0.5395 | 0.4061 |
| FSFM | 0.1629 | 0.6180 | 0.4683 | 0.5123 | 0.4348 | 0.4081 | 0.7593 | 0.2266 | 0.1067 | *in progress* | *in progress* | *in progress* | 0.4108 | 0.4393 | 0.4108 |
| FSFM-Adapter | 0.4775 | 0.6404 | 0.7741 | 0.6540 | 0.5435 | 0.5458 | 0.8834 | 0.4372 | 0.5281 | *in progress* | *in progress* | *in progress* | 0.6093 | 0.6179 | 0.6093 |
| Effort | 0.3596 | 0.7978 | 0.5041 | 0.7201 | 0.5217 | 0.4063 | 0.7295 | 0.2417 | 0.1461 | *in progress* | *in progress* | *in progress* | 0.4919 | 0.5807 | 0.4919 |
| ForAda | 0.2360 | 0.8090 | 0.3857 | 0.5862 | 0.3217 | 0.4323 | 0.7146 | 0.4054 | 0.1517 | *in progress* | *in progress* | *in progress* | 0.4492 | 0.4677 | 0.4492 |


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
| **Ours (App+TD+LFS+STT, PCR6+-R)** | 0.1194 | 0.1732 | 0.1940 | 0.1296 | 0.1796 | 0.0623 | 0.1443 | 0.0505 | 0.0281 | 0.0988 | 0.0990 | 0.0933 | 0.1143 | 0.1592 | 0.1201 |
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
| **Ours (App+TD+LFS+STT, PCR6+-R)** | 0.0730 | 0.0889 | 0.1406 | 0.1688 | 0.1626 | 0.0711 | 0.1063 | 0.0257 | 0.0150 | 0.0562 | 0.1162 | 0.1166 | 0.0951 | 0.1268 | 0.0947 |
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
| **Ours (App+TD+LFS+STT, PCR6+-R)** | 0.0167 | 0.0109 | 0.0278 | 0.0886 | 0.0571 | 0.0197 | 0.0250 | 0.0047 | 0.0016 | 0.0448 | 0.1297 | 0.1005 | 0.0439 | 0.0402 | 0.0280 |
| DiCoME (trained, their repo) | 0.0164 | 0.0172 | 0.0141 | 0.0670 | 0.0634 | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | 0.0356 | 0.0356 | *in progress* |
| GenD-CLIP | 0.0342 | 0.0278 | 0.0134 | 0.0720 | 0.0439 | 0.0353 | 0.0564 | 0.0124 | 0.0015 | *in progress* | *in progress* | *in progress* | 0.0330 | 0.0382 | 0.0330 |
| GenD-PE | 0.0217 | 0.0462 | 0.0484 | 0.1177 | 0.0599 | 0.0417 | 0.0680 | 0.0107 | 0.0024 | *in progress* | *in progress* | *in progress* | 0.0463 | 0.0588 | 0.0463 |
| GenD-DINO | 0.0427 | 0.0838 | 0.0773 | 0.0913 | 0.0420 | 0.0402 | 0.1409 | 0.0076 | 0.0020 | *in progress* | *in progress* | *in progress* | 0.0586 | 0.0674 | 0.0586 |
| FSFM | 0.0227 | 0.0672 | 0.0630 | 0.0637 | 0.0507 | 0.0638 | 0.0854 | 0.0176 | 0.0026 | *in progress* | *in progress* | *in progress* | 0.0485 | 0.0535 | 0.0485 |
| FSFM-Adapter | 0.0758 | 0.0855 | 0.2099 | 0.0968 | 0.0678 | 0.0879 | 0.2186 | 0.0423 | 0.0671 | *in progress* | *in progress* | *in progress* | 0.1057 | 0.1072 | 0.1057 |
| Effort | 0.0416 | 0.0346 | 0.0120 | 0.0778 | 0.0627 | 0.0394 | 0.0421 | 0.0438 | 0.0037 | *in progress* | *in progress* | *in progress* | 0.0397 | 0.0457 | 0.0397 |
| ForAda | 0.0203 | 0.1552 | 0.0262 | 0.0772 | 0.0634 | 0.0593 | 0.0879 | 0.0436 | 0.0038 | *in progress* | *in progress* | *in progress* | 0.0597 | 0.0685 | 0.0597 |


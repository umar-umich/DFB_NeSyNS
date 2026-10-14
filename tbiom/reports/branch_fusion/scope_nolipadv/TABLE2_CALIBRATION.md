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

`mean (all)`: 11 sets (CDFv2, DFD, DFDC, DFDCP, FFIW, KoDF, IDForge, DFDM, UADFV, DeepSpeak-1.1, DeepSpeak-2.0).  
`mean (wide)`: 11 sets (CDFv2, DFD, DFDC, DFDCP, FFIW, KoDF, IDForge, DFDM, UADFV, DeepSpeak-1.1, DeepSpeak-2.0).

| method | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean | mean (all) | mean (wide) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours (App+TD+LFS+STT, PCR6+-R)** | 0.1194 | 0.1732 | 0.1940 | 0.1296 | 0.1796 | 0.0623 | 0.1825 | 0.0505 | 0.0281 | 0.0988 | 0.0990 | 0.0933 | 0.1175 | 0.1125 | 0.1125 |
| DiCoME (trained, their repo) | 0.1119 | 0.1520 | 0.1191 | 0.0916 | 0.0480 | 0.0748 | 0.1163 | 0.0944 | 0.0450 | 0.1235 | 0.2111 | 0.2634 | 0.1209 | 0.1181 | 0.1181 |
| GenD-CLIP | 0.1180 | 0.1943 | 0.1162 | 0.0588 | 0.0641 | 0.0759 | 0.0814 | 0.1251 | 0.0597 | 0.1867 | 0.2112 | 0.2316 | 0.1269 | 0.1208 | 0.1208 |
| GenD-PE | 0.0654 | *in progress* | 0.1929 | 0.1095 | 0.0979 | 0.0732 | 0.2186 | 0.0681 | 0.0385 | 0.1893 | 0.1399 | 0.1095 | 0.1184 | 0.1184 | 0.1184 |
| GenD-DINO | 0.0849 | *in progress* | 0.2266 | 0.0681 | 0.0846 | 0.1054 | 0.2118 | 0.1150 | 0.0609 | 0.1407 | 0.1344 | 0.1387 | 0.1247 | 0.1247 | 0.1247 |
| FSFM | 0.1373 | *in progress* | 0.2697 | 0.0850 | 0.1334 | 0.1498 | 0.2697 | 0.1159 | 0.0984 | 0.1627 | 0.0700 | 0.0585 | 0.1410 | 0.1410 | 0.1410 |
| FSFM-Adapter | 0.2054 | *in progress* | 0.3975 | 0.1142 | 0.1470 | 0.1433 | 0.3012 | 0.1585 | 0.3383 | 0.2664 | 0.1589 | 0.1316 | 0.2147 | 0.2147 | 0.2147 |
| Effort | 0.1242 | *in progress* | 0.0939 | 0.0870 | 0.1168 | 0.1400 | 0.0552 | 0.1826 | 0.0677 | 0.2045 | 0.2760 | 0.3393 | 0.1534 | 0.1534 | 0.1534 |
| ForAda | 0.0969 | *in progress* | 0.1586 | 0.0944 | 0.1883 | 0.0848 | 0.2567 | 0.0658 | 0.0607 | 0.1403 | 0.0505 | 0.1013 | 0.1180 | 0.1180 | 0.1180 |

## Brier (lower is better)

`mean (all)`: 11 sets (CDFv2, DFD, DFDC, DFDCP, FFIW, KoDF, IDForge, DFDM, UADFV, DeepSpeak-1.1, DeepSpeak-2.0).  
`mean (wide)`: 11 sets (CDFv2, DFD, DFDC, DFDCP, FFIW, KoDF, IDForge, DFDM, UADFV, DeepSpeak-1.1, DeepSpeak-2.0).

| method | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean | mean (all) | mean (wide) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours (App+TD+LFS+STT, PCR6+-R)** | 0.0730 | 0.0889 | 0.1406 | 0.1688 | 0.1626 | 0.0711 | 0.1584 | 0.0257 | 0.0150 | 0.0562 | 0.1162 | 0.1166 | 0.0994 | 0.1004 | 0.1004 |
| DiCoME (trained, their repo) | 0.0829 | 0.0838 | 0.0828 | 0.1498 | 0.1477 | 0.0939 | 0.1939 | 0.0830 | 0.0224 | 0.0634 | 0.1945 | 0.2498 | 0.1206 | 0.1240 | 0.1240 |
| GenD-CLIP | 0.1070 | 0.0980 | 0.0815 | 0.1503 | 0.1140 | 0.1151 | 0.1485 | 0.0722 | 0.0258 | 0.0817 | 0.1884 | 0.2294 | 0.1177 | 0.1194 | 0.1194 |
| GenD-PE | 0.0795 | *in progress* | 0.1377 | 0.1878 | 0.1421 | 0.1115 | 0.2231 | 0.0637 | 0.0319 | 0.1366 | 0.1715 | 0.1580 | 0.1312 | 0.1312 | 0.1312 |
| GenD-DINO | 0.1101 | *in progress* | 0.1546 | 0.1636 | 0.1155 | 0.1117 | 0.1801 | 0.0627 | 0.0223 | 0.0754 | 0.1700 | 0.1829 | 0.1226 | 0.1226 | 0.1226 |
| FSFM | 0.0914 | *in progress* | 0.1539 | 0.1476 | 0.1394 | 0.1490 | 0.2295 | 0.0995 | 0.0399 | 0.0702 | 0.1360 | 0.1692 | 0.1296 | 0.1296 | 0.1296 |
| FSFM-Adapter | 0.1817 | *in progress* | 0.2551 | 0.1830 | 0.1721 | 0.1813 | 0.2626 | 0.1548 | 0.1801 | 0.1147 | 0.1919 | 0.2073 | 0.1895 | 0.1895 | 0.1895 |
| Effort | 0.1234 | *in progress* | 0.0763 | 0.1615 | 0.1567 | 0.1329 | 0.1219 | 0.1298 | 0.0421 | 0.1167 | 0.2269 | 0.3130 | 0.1455 | 0.1455 | 0.1455 |
| ForAda | 0.0915 | *in progress* | 0.1060 | 0.1567 | 0.1612 | 0.1355 | 0.2219 | 0.1276 | 0.0440 | 0.0695 | 0.1707 | 0.2026 | 0.1352 | 0.1352 | 0.1352 |

## E-AURC (lower is better)

`mean (all)`: 11 sets (CDFv2, DFD, DFDC, DFDCP, FFIW, KoDF, IDForge, DFDM, UADFV, DeepSpeak-1.1, DeepSpeak-2.0).  
`mean (wide)`: 11 sets (CDFv2, DFD, DFDC, DFDCP, FFIW, KoDF, IDForge, DFDM, UADFV, DeepSpeak-1.1, DeepSpeak-2.0).

| method | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean | mean (all) | mean (wide) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours (App+TD+LFS+STT, PCR6+-R)** | 0.0167 | 0.0109 | 0.0278 | 0.0886 | 0.0571 | 0.0197 | 0.0362 | 0.0047 | 0.0016 | 0.0448 | 0.1297 | 0.1005 | 0.0449 | 0.0479 | 0.0479 |
| DiCoME (trained, their repo) | 0.0164 | 0.0172 | 0.0141 | 0.0670 | 0.0634 | 0.0262 | 0.0821 | 0.0154 | 0.0010 | 0.0123 | 0.1529 | 0.2099 | 0.0565 | 0.0601 | 0.0601 |
| GenD-CLIP | 0.0342 | 0.0266 | 0.0134 | 0.0720 | 0.0439 | 0.0353 | 0.0492 | 0.0124 | 0.0015 | 0.0131 | 0.1208 | 0.1970 | 0.0516 | 0.0539 | 0.0539 |
| GenD-PE | 0.0217 | *in progress* | 0.0484 | 0.1177 | 0.0599 | 0.0417 | 0.0608 | 0.0107 | 0.0024 | 0.0452 | 0.0981 | 0.0899 | 0.0542 | 0.0542 | 0.0542 |
| GenD-DINO | 0.0427 | *in progress* | 0.0773 | 0.0913 | 0.0420 | 0.0402 | 0.1358 | 0.0076 | 0.0020 | 0.0131 | 0.1110 | 0.1175 | 0.0619 | 0.0619 | 0.0619 |
| FSFM | 0.0227 | *in progress* | 0.0630 | 0.0637 | 0.0507 | 0.0638 | 0.0916 | 0.0176 | 0.0026 | 0.0140 | 0.0657 | 0.1106 | 0.0515 | 0.0515 | 0.0515 |
| FSFM-Adapter | 0.0758 | *in progress* | 0.2099 | 0.0968 | 0.0678 | 0.0879 | 0.2224 | 0.0423 | 0.0671 | 0.0046 | 0.1096 | 0.2008 | 0.1077 | 0.1077 | 0.1077 |
| Effort | 0.0416 | *in progress* | 0.0120 | 0.0778 | 0.0627 | 0.0394 | 0.0363 | 0.0438 | 0.0037 | 0.0329 | 0.1748 | 0.3342 | 0.0781 | 0.0781 | 0.0781 |
| ForAda | 0.0203 | *in progress* | 0.0262 | 0.0772 | 0.0634 | 0.0593 | 0.0864 | 0.0436 | 0.0038 | 0.0123 | 0.1232 | 0.2051 | 0.0655 | 0.0655 | 0.0655 |


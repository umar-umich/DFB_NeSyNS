# Operator sweep -- conflict routing on the cached four-branch opinions

App + TD + LFS + STT, reliability weights fitted ONCE on development and applied
unchanged. Every rule reads the same discounted masses, so the only thing that varies
is where conflicting mass goes.

`Dempster` is DiCoME eq. (7) -- the paper's own rule, written in subjective-logic form.
On a BINARY frame Yager is also Dubois-Prade, and PCR5 collapses onto PCR6, so several
operators the literature separates are one row here.

**This table cannot settle the question.** A frozen branch cannot adapt to the operator,
so a rule that pays off only when branches can respond to it looks flat. A win here is
strong evidence; a loss is weak evidence. The end-to-end arms decide.

## AUROC (higher is better)

| rule | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dempster (DiCoME eq. 7) | 0.9826 | 0.9654 | 0.9533 | 0.8630 | 0.9141 | 0.9776 | 0.9461 | 0.9969 | 0.9973 | 0.9875 | 0.9312 | 0.9172 | **0.9527** |
| PCR6+ | 0.9788 | 0.9612 | 0.9569 | 0.8622 | 0.9112 | 0.9743 | 0.9436 | 0.9961 | 0.9966 | 0.9875 | 0.9277 | 0.9144 | **0.9509** |
| Yager | 0.9712 | 0.9576 | 0.9574 | 0.8595 | 0.9082 | 0.9712 | 0.9402 | 0.9949 | 0.9962 | 0.9875 | 0.9273 | 0.9128 | **0.9487** |
| WBF (Josang) | 0.9755 | 0.9624 | 0.9549 | 0.8597 | 0.9109 | 0.9760 | 0.9429 | 0.9972 | 0.9969 | 0.9875 | 0.9276 | 0.9190 | **0.9509** |
| Cautious (Denoeux) | 0.9541 | 0.9555 | 0.9530 | 0.8494 | 0.9000 | 0.9704 | 0.9415 | 0.9971 | 0.9959 | 0.9846 | 0.9165 | 0.9151 | **0.9444** |

## FPR at TPR 90% (lower is better)

| rule | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dempster (DiCoME eq. 7) | 0.0674 | 0.1067 | 0.1818 | 0.3935 | 0.2826 | 0.0628 | 0.0496 | 0.0033 | 0.0000 | 0.0204 | 0.1667 | 0.2442 | **0.1316** |
| PCR6+ | 0.0674 | 0.1124 | 0.1570 | 0.3948 | 0.2696 | 0.0646 | 0.0397 | 0.0067 | 0.0056 | 0.0204 | 0.1645 | 0.2222 | **0.1271** |
| Yager | 0.0674 | 0.1067 | 0.1433 | 0.4082 | 0.2870 | 0.0876 | 0.0347 | 0.0141 | 0.0056 | 0.0204 | 0.1695 | 0.2308 | **0.1313** |
| WBF (Josang) | 0.0787 | 0.1180 | 0.1846 | 0.4069 | 0.2870 | 0.0767 | 0.0422 | 0.0023 | 0.0056 | 0.0204 | 0.1879 | 0.2501 | **0.1384** |
| Cautious (Denoeux) | 0.1236 | 0.1180 | 0.1680 | 0.4479 | 0.3174 | 0.0928 | 0.0571 | 0.0030 | 0.0056 | 0.0204 | 0.2253 | 0.2458 | **0.1521** |

## ECE-15 (lower is better)

| rule | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dempster (DiCoME eq. 7) | 0.0464 | 0.1306 | 0.1890 | 0.1538 | 0.1789 | 0.0472 | 0.1412 | 0.0123 | 0.0092 | 0.0896 | 0.1095 | 0.0947 | **0.1002** |
| PCR6+ | 0.1194 | 0.1721 | 0.1940 | 0.1296 | 0.1796 | 0.0623 | 0.1504 | 0.0505 | 0.0281 | 0.0988 | 0.0990 | 0.0933 | **0.1148** |
| Yager | 0.2232 | 0.2365 | 0.2142 | 0.1027 | 0.1800 | 0.0974 | 0.1520 | 0.1138 | 0.0705 | 0.1281 | 0.0992 | 0.1128 | **0.1442** |
| WBF (Josang) | 0.1483 | 0.1805 | 0.1892 | 0.0624 | 0.1606 | 0.1076 | 0.1151 | 0.1188 | 0.0751 | 0.1434 | 0.1419 | 0.1673 | **0.1342** |
| Cautious (Denoeux) | 0.1558 | 0.1772 | 0.1804 | 0.0563 | 0.1490 | 0.0957 | 0.1139 | 0.1189 | 0.0802 | 0.1650 | 0.1469 | 0.1866 | **0.1355** |

## Brier (lower is better)

| rule | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dempster (DiCoME eq. 7) | 0.0529 | 0.0870 | 0.1605 | 0.1802 | 0.1765 | 0.0672 | 0.1172 | 0.0183 | 0.0112 | 0.0670 | 0.1242 | 0.1208 | **0.0986** |
| PCR6+ | 0.0730 | 0.0891 | 0.1406 | 0.1688 | 0.1626 | 0.0711 | 0.1129 | 0.0257 | 0.0150 | 0.0562 | 0.1162 | 0.1166 | **0.0957** |
| Yager | 0.1182 | 0.1096 | 0.1309 | 0.1634 | 0.1612 | 0.0847 | 0.1131 | 0.0453 | 0.0266 | 0.0663 | 0.1154 | 0.1199 | **0.1046** |
| WBF (Josang) | 0.0816 | 0.0745 | 0.1134 | 0.1558 | 0.1471 | 0.0736 | 0.0944 | 0.0338 | 0.0198 | 0.0733 | 0.1240 | 0.1282 | **0.0933** |
| Cautious (Denoeux) | 0.1036 | 0.0708 | 0.1041 | 0.1606 | 0.1495 | 0.0782 | 0.0920 | 0.0360 | 0.0229 | 0.0868 | 0.1340 | 0.1400 | **0.0982** |

## mean fused vacuity (diagnostic, not a score)

| rule | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dempster (DiCoME eq. 7) | 0.0104 | 0.0128 | 0.0087 | 0.0155 | 0.0177 | 0.0148 | 0.0070 | 0.0098 | 0.0029 | 0.0062 | 0.0120 | 0.0177 | **0.0113** |
| PCR6+ | 0.0047 | 0.0064 | 0.0045 | 0.0085 | 0.0088 | 0.0090 | 0.0030 | 0.0063 | 0.0014 | 0.0030 | 0.0064 | 0.0093 | **0.0059** |
| Yager | 0.5323 | 0.3864 | 0.2936 | 0.3353 | 0.4056 | 0.2979 | 0.2177 | 0.2509 | 0.1633 | 0.2739 | 0.3145 | 0.3676 | **0.3199** |
| WBF (Josang) | 0.1603 | 0.1519 | 0.1489 | 0.1931 | 0.1961 | 0.1966 | 0.1397 | 0.1694 | 0.1045 | 0.1339 | 0.1924 | 0.2148 | **0.1668** |
| Cautious (Denoeux) | 0.0940 | 0.0946 | 0.1017 | 0.1261 | 0.1236 | 0.1328 | 0.0949 | 0.1145 | 0.0718 | 0.0846 | 0.1285 | 0.1402 | **0.1089** |

## Against DiCoME's own rule

Difference from the Dempster row, signed so that POSITIVE IS BETTER for every column.

| rule | AUROC | FPR@90 | ECE | Brier |
|---|---:|---:|---:|---:|
| Dempster (DiCoME eq. 7) | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| PCR6+ | -0.0018 | +0.0045 | -0.0146 | +0.0029 |
| Yager | -0.0040 | +0.0003 | -0.0440 | -0.0060 |
| WBF (Josang) | -0.0018 | -0.0068 | -0.0340 | +0.0053 |
| Cautious (Denoeux) | -0.0083 | -0.0205 | -0.0353 | +0.0004 |


## Verdict on the frozen branches

**DEMPSTER WINS THIS TABLE.** It is best on AUROC (0.9527) and best on ECE (0.1002), and no
alternative beats it on both. DiCoME's own rule is the strongest of the five in this regime.

**The cautious rule's prediction FAILED.** It was adopted on the argument that our branches are
dependent (score correlations 0.45 to 0.84), that Dempster therefore double-counts shared
evidence, and that the symptom would be over-confidence -- so an idempotent rule should improve
calibration, possibly at a cost in AUROC. What happened: AUROC -0.0083 as predicted, but ECE
-0.0353 and FPR@90 -0.0205, i.e. calibration got WORSE, and Brier moved +0.0004, which is
nothing. The stated falsifier was "if AUROC moves and calibration does not, the diagnosis was
wrong". Calibration moved the wrong way, which is worse than that.

So on this evidence the cautious rule does not earn the headline, and the independence argument
does not survive contact with the measurement. It stays in the code as a tested operator and an
ablation row.

Two things are genuinely open, and neither rescues the above:

1. The reliability weights were fitted ONCE under F8 (PCR6+) and reused for every rule, so the
   four alternatives run on weights tuned for a different operator. That confound works AGAINST
   Dempster and it still won, which strengthens its row rather than weakening it -- but a fair
   table would refit per rule.
2. Raw ECE is not scale-free. The cautious rule raises mean fused vacuity from 0.011 to 0.109,
   and p = b_fake + u/2, so every probability is pulled toward 0.5. Much of the ECE gap is that
   shift and a monotone recalibration would remove it. Whether it is better calibrated AFTER
   rescaling is a different question and is not answered here.

**And this table cannot settle the operator question anyway**, in the direction that matters: a
frozen branch cannot adapt to the rule it is fused by. A win here is strong evidence and a loss
is weak evidence. What it does settle is that there is no free win sitting in the cached
opinions -- the end-to-end arms have to earn it.

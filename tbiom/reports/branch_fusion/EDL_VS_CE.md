# EDL against cross-entropy, and PCR6+ against the alternatives

Rebuilt from DiCoME Table 2(b) and Table 3 on our system, **per dataset**. A mean over
ten corpora that differ by 0.25 in AUROC hides what a reader needs.

Fusion rules in order: Mean, evidential mean, DS-C, and **PCR6+-R**, ours and last in
each block. Every row uses the same fitted reliability weights, so what changes between
rows is the rule and nothing else.

## AUROC (higher is better)

| system | fusion rule | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | macro |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A+V+O | Mean | 0.9690 | 0.9658 | 0.9460 | 0.8590 | 0.8984 | 0.9603 | 0.9451 | 0.9904 | 0.9928 | 0.9875 | 0.9274 | 0.9067 | 0.9457 |
| A+V+O | Evidential mean | 0.9470 | 0.9685 | 0.9432 | 0.8380 | 0.8915 | 0.9520 | 0.9461 | 0.9918 | 0.9924 | 0.9850 | 0.9248 | 0.9216 | 0.9418 |
| A+V+O | DS-C | 0.9678 | 0.9691 | 0.9492 | 0.8602 | 0.9047 | 0.9701 | 0.9448 | 0.9949 | 0.9937 | 0.9883 | 0.9312 | 0.9156 | 0.9491 |
| **A+V+O** | **PCR6+-R (ours)** | 0.9645 | 0.9627 | 0.9513 | 0.8596 | 0.9058 | 0.9693 | 0.9430 | 0.9950 | 0.9927 | 0.9883 | 0.9268 | 0.9119 | 0.9476 |
| A+V+O+P16ctl+S16ctl | Mean | 0.9823 | 0.9557 | 0.9572 | 0.8585 | 0.9149 | 0.9756 | 0.9427 | 0.9964 | 0.9970 | 0.9842 | 0.9310 | 0.9178 | 0.9511 |
| A+V+O+P16ctl+S16ctl | Evidential mean | 0.9705 | 0.9647 | 0.9551 | 0.8466 | 0.9079 | 0.9698 | 0.9434 | 0.9967 | 0.9970 | 0.9842 | 0.9272 | 0.9280 | 0.9492 |
| A+V+O+P16ctl+S16ctl | DS-C | 0.9795 | 0.9688 | 0.9512 | 0.8578 | 0.9116 | 0.9748 | 0.9459 | 0.9966 | 0.9970 | 0.9867 | 0.9321 | 0.9222 | 0.9520 |
| **A+V+O+P16ctl+S16ctl** | **PCR6+-R (ours)** | 0.9738 | 0.9625 | 0.9577 | 0.8593 | 0.9090 | 0.9701 | 0.9426 | 0.9961 | 0.9953 | 0.9867 | 0.9285 | 0.9210 | 0.9502 |
| *CE baseline (DISCERN_Ext arm D, 3 other views)* | mean-of-logits | 0.9376 | 0.8648 | 0.9449 | 0.8791 | 0.8724 | -- | -- | -- | -- | -- | -- | -- | 0.8639\* |

## ECE-15 (lower is better)

| system | fusion rule | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | macro |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A+V+O | Mean | 0.2593 | 0.2802 | 0.2367 | 0.0663 | 0.2033 | 0.1439 | 0.1472 | 0.1507 | 0.1102 | 0.1393 | 0.1121 | 0.1393 | 0.1657 |
| A+V+O | Evidential mean | 0.2224 | 0.1872 | 0.2055 | 0.1277 | 0.2213 | 0.0839 | 0.1040 | 0.0771 | 0.0610 | 0.0919 | 0.1039 | 0.1473 | 0.1361 |
| A+V+O | DS-C | 0.1228 | 0.1553 | 0.1799 | 0.1372 | 0.1865 | 0.0772 | 0.1523 | 0.0404 | 0.0281 | 0.0554 | 0.0871 | 0.1156 | 0.1115 |
| **A+V+O** | **PCR6+-R (ours)** | 0.1717 | 0.1893 | 0.1890 | 0.1229 | 0.1880 | 0.0858 | 0.1548 | 0.0655 | 0.0399 | 0.0997 | 0.0817 | 0.1187 | 0.1256 |
| A+V+O+P16ctl+S16ctl | Mean | 0.2146 | 0.2742 | 0.2428 | 0.1088 | 0.2301 | 0.1489 | 0.1500 | 0.1441 | 0.0906 | 0.1359 | 0.0893 | 0.1013 | 0.1609 |
| A+V+O+P16ctl+S16ctl | Evidential mean | 0.1227 | 0.1551 | 0.2073 | 0.1449 | 0.2159 | 0.0511 | 0.1085 | 0.0462 | 0.0306 | 0.1012 | 0.1083 | 0.1068 | 0.1165 |
| A+V+O+P16ctl+S16ctl | DS-C | 0.0588 | 0.1270 | 0.1981 | 0.1704 | 0.2043 | 0.0569 | 0.1329 | 0.0073 | 0.0074 | 0.0777 | 0.1100 | 0.1018 | 0.1044 |
| **A+V+O+P16ctl+S16ctl** | **PCR6+-R (ours)** | 0.1739 | 0.2180 | 0.2128 | 0.1716 | 0.2438 | 0.1084 | 0.1626 | 0.0503 | 0.0369 | 0.0820 | 0.0578 | 0.0564 | 0.1312 |

## NLL (lower is better)

| system | fusion rule | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | macro |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A+V+O | Mean | 0.4453 | 0.4028 | 0.4072 | 0.4861 | 0.5122 | 0.3488 | 0.3594 | 0.2577 | 0.1671 | 0.2165 | 0.3715 | 0.4134 | 0.3657 |
| A+V+O | Evidential mean | 0.4445 | 0.2916 | 0.3919 | 0.5767 | 0.5707 | 0.3055 | 0.3245 | 0.1655 | 0.1143 | 0.1792 | 0.3823 | 0.4027 | 0.3458 |
| A+V+O | DS-C | 0.2948 | 0.2832 | 0.4244 | 0.6328 | 0.5624 | 0.2675 | 0.4909 | 0.1158 | 0.0768 | 0.1443 | 0.4089 | 0.4292 | 0.3443 |
| **A+V+O** | **PCR6+-R (ours)** | 0.3458 | 0.3024 | 0.3850 | 0.5706 | 0.5059 | 0.2679 | 0.4764 | 0.1394 | 0.0911 | 0.1756 | 0.3791 | 0.3972 | 0.3364 |
| A+V+O+P16ctl+S16ctl | Mean | 0.3616 | 0.4009 | 0.4278 | 0.5098 | 0.5193 | 0.3158 | 0.3692 | 0.2076 | 0.1290 | 0.2328 | 0.3558 | 0.3645 | 0.3495 |
| A+V+O+P16ctl+S16ctl | Evidential mean | 0.2765 | 0.2562 | 0.4308 | 0.6000 | 0.5533 | 0.2401 | 0.3529 | 0.0997 | 0.0578 | 0.2262 | 0.3990 | 0.3509 | 0.3203 |
| A+V+O+P16ctl+S16ctl | DS-C | 0.1980 | 0.2922 | 0.6338 | 0.8375 | 0.6837 | 0.2482 | 0.5936 | 0.0716 | 0.0399 | 0.2260 | 0.5630 | 0.4610 | 0.4040 |
| **A+V+O+P16ctl+S16ctl** | **PCR6+-R (ours)** | 0.3375 | 0.3693 | 0.4652 | 0.6478 | 0.6084 | 0.2958 | 0.5351 | 0.1151 | 0.0801 | 0.1616 | 0.3489 | 0.3248 | 0.3575 |

## Brier (lower is better)

| system | fusion rule | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | macro |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A+V+O | Mean | 0.1400 | 0.1273 | 0.1356 | 0.1606 | 0.1742 | 0.1068 | 0.1148 | 0.0697 | 0.0421 | 0.0566 | 0.1139 | 0.1298 | 0.1143 |
| A+V+O | Evidential mean | 0.1490 | 0.0910 | 0.1329 | 0.1843 | 0.1976 | 0.0984 | 0.0830 | 0.0449 | 0.0299 | 0.0464 | 0.1211 | 0.1290 | 0.1090 |
| A+V+O | DS-C | 0.0952 | 0.0900 | 0.1402 | 0.1757 | 0.1819 | 0.0857 | 0.1208 | 0.0333 | 0.0227 | 0.0377 | 0.1145 | 0.1292 | 0.1022 |
| **A+V+O** | **PCR6+-R (ours)** | 0.1069 | 0.0942 | 0.1294 | 0.1681 | 0.1689 | 0.0855 | 0.1168 | 0.0386 | 0.0254 | 0.0471 | 0.1111 | 0.1244 | 0.1014 |
| A+V+O+P16ctl+S16ctl | Mean | 0.1030 | 0.1272 | 0.1454 | 0.1712 | 0.1775 | 0.0916 | 0.1196 | 0.0483 | 0.0288 | 0.0668 | 0.1086 | 0.1098 | 0.1082 |
| A+V+O+P16ctl+S16ctl | Evidential mean | 0.0792 | 0.0790 | 0.1462 | 0.1857 | 0.1895 | 0.0742 | 0.0868 | 0.0227 | 0.0125 | 0.0686 | 0.1231 | 0.1098 | 0.0981 |
| A+V+O+P16ctl+S16ctl | DS-C | 0.0615 | 0.0872 | 0.1713 | 0.1918 | 0.1968 | 0.0734 | 0.1110 | 0.0194 | 0.0114 | 0.0586 | 0.1243 | 0.1207 | 0.1023 |
| **A+V+O+P16ctl+S16ctl** | **PCR6+-R (ours)** | 0.1069 | 0.1187 | 0.1529 | 0.1873 | 0.2025 | 0.0953 | 0.1225 | 0.0316 | 0.0228 | 0.0470 | 0.1039 | 0.1008 | 0.1077 |

## FPR_real at the frozen threshold (lower is better)

| system | fusion rule | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | macro |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A+V+O | Mean | 0.0337 | 0.0337 | 0.0716 | 0.1197 | 0.0739 | 0.0415 | 0.0099 | 0.0194 | 0.0337 | 0.0816 | 0.1660 | 0.1707 | 0.0713 |
| A+V+O | Evidential mean | 0.0562 | 0.0562 | 0.0909 | 0.1521 | 0.0957 | 0.1037 | 0.0223 | 0.0348 | 0.0506 | 0.0816 | 0.2196 | 0.2324 | 0.0997 |
| A+V+O | DS-C | 0.0449 | 0.0449 | 0.0551 | 0.1045 | 0.0609 | 0.0352 | 0.0050 | 0.0110 | 0.0449 | 0.0408 | 0.1511 | 0.1632 | 0.0635 |
| **A+V+O** | **PCR6+-R (ours)** | 0.0562 | 0.0562 | 0.0579 | 0.1110 | 0.0739 | 0.0357 | 0.0050 | 0.0121 | 0.0506 | 0.0408 | 0.1575 | 0.1723 | 0.0691 |
| A+V+O+P16ctl+S16ctl | Mean | 0.1517 | 0.1517 | 0.0551 | 0.1343 | 0.0783 | 0.0582 | 0.0199 | 0.0429 | 0.1742 | 0.2245 | 0.2133 | 0.1551 | 0.1216 |
| A+V+O+P16ctl+S16ctl | Evidential mean | 0.1348 | 0.1348 | 0.0771 | 0.1724 | 0.1217 | 0.1107 | 0.0273 | 0.0529 | 0.1685 | 0.2041 | 0.2669 | 0.2512 | 0.1435 |
| A+V+O+P16ctl+S16ctl | DS-C | 0.0730 | 0.0730 | 0.0551 | 0.1171 | 0.0652 | 0.0501 | 0.0149 | 0.0275 | 0.0899 | 0.1633 | 0.1836 | 0.1637 | 0.0897 |
| **A+V+O+P16ctl+S16ctl** | **PCR6+-R (ours)** | 0.0674 | 0.0674 | 0.0579 | 0.1261 | 0.0783 | 0.0553 | 0.0174 | 0.0238 | 0.0674 | 0.1633 | 0.1879 | 0.1986 | 0.0926 |
| *CE baseline (DISCERN_Ext arm D, 3 other views)* | mean-of-logits | 0.1966 | 0.1966 | 0.3691 | 0.3201 | 0.3696 | -- | -- | -- | -- | -- | -- | -- | 0.3156\* |

\* the CE row is DISCERN_Ext's arm D over a DIFFERENT three views and trunk, and its
macro is over its own six OOD sets, not these ten. It is the CE reference point this
project has today, not a CE version of the system above. Treat the columns as
indicative and the macro as not comparable.

## What the comparison actually shows

On ranking, CE does **not** collapse the way DiCoME's (D) does: our arm D reaches
0.8639 OOD AUROC against the evidential arm's 0.8633 --
a tie, +0.0006 in CE's favour. Any claim that the evidential machinery buys accuracy is
not supported here and must not be written.

Where it does separate is the **operating point**. At the frozen threshold arm D sits at
FPR_real 0.3156 against the evidential arm's 0.1350 -- more
than twice the false-alarm rate on genuine video at equal ranking quality. That is a
deployment difference, not a leaderboard one, and it is the honest form of the claim.

And on **referral**, which CE cannot do at all: vacuity ranks the model's own errors at
err-AUROC 0.771 where the decision margin -- the only uncertainty a CE fusion of the same
branches would hand you -- manages 0.568 (`tbiom/reports/REFERRAL.md`).

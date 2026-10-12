# EDL against cross-entropy, and PCR6+ against the alternatives

Rebuilt from DiCoME Table 2(b) and Table 3 on our system, **per dataset**. A mean over
ten corpora that differ by 0.25 in AUROC hides what a reader needs.

## AUROC (higher is better)

| system | fusion | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | DFEval24 | FFIW | KoDF | IDForge | DFDM | macro |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A+V+O | F1 | 0.9690 | 0.9658 | 0.9460 | 0.8590 | 0.8984 | 0.7098 | 0.9603 | 0.9451 | 0.9904 | 0.9928 | 0.9236 |
| A+V+O | F5 | 0.9470 | 0.9685 | 0.9432 | 0.8380 | 0.8915 | 0.7068 | 0.9520 | 0.9461 | 0.9918 | 0.9924 | 0.9177 |
| A+V+O | F7 | 0.9678 | 0.9691 | 0.9492 | 0.8602 | 0.9047 | 0.7097 | 0.9701 | 0.9448 | 0.9949 | 0.9937 | 0.9264 |
| **A+V+O** | F8 | 0.9645 | 0.9627 | 0.9513 | 0.8596 | 0.9058 | 0.7042 | 0.9693 | 0.9430 | 0.9950 | 0.9927 | 0.9248 |
| A+V+O+P16ctl+S16ctl | F1 | 0.9823 | 0.9557 | 0.9572 | 0.8585 | 0.9149 | 0.6911 | 0.9756 | 0.9427 | 0.9964 | 0.9970 | 0.9271 |
| A+V+O+P16ctl+S16ctl | F5 | 0.9705 | 0.9647 | 0.9551 | 0.8466 | 0.9079 | 0.6969 | 0.9698 | 0.9434 | 0.9967 | 0.9970 | 0.9249 |
| A+V+O+P16ctl+S16ctl | F7 | 0.9795 | 0.9688 | 0.9512 | 0.8578 | 0.9116 | 0.7061 | 0.9748 | 0.9459 | 0.9966 | 0.9970 | 0.9289 |
| **A+V+O+P16ctl+S16ctl** | F8 | 0.9738 | 0.9625 | 0.9577 | 0.8593 | 0.9090 | 0.6941 | 0.9701 | 0.9426 | 0.9961 | 0.9953 | 0.9261 |
| *CE baseline (DISCERN_Ext arm D, 3 other views)* | mean-of-logits | 0.9376 | 0.8648 | 0.9449 | 0.8791 | 0.8724 | 0.6850 | -- | -- | -- | -- | 0.8639\* |

## ECE-15 (lower is better)

| system | fusion | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | DFEval24 | FFIW | KoDF | IDForge | DFDM | macro |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A+V+O | F1 | 0.2593 | 0.2802 | 0.2367 | 0.0663 | 0.2033 | 0.0963 | 0.1439 | 0.1472 | 0.1507 | 0.1102 | 0.1694 |
| A+V+O | F5 | 0.2224 | 0.1872 | 0.2055 | 0.1277 | 0.2213 | 0.1349 | 0.0839 | 0.1040 | 0.0771 | 0.0610 | 0.1425 |
| A+V+O | F7 | 0.1228 | 0.1553 | 0.1799 | 0.1372 | 0.1865 | 0.1931 | 0.0772 | 0.1523 | 0.0404 | 0.0281 | 0.1273 |
| **A+V+O** | F8 | 0.1717 | 0.1893 | 0.1890 | 0.1229 | 0.1880 | 0.1555 | 0.0858 | 0.1548 | 0.0655 | 0.0399 | 0.1362 |
| A+V+O+P16ctl+S16ctl | F1 | 0.2146 | 0.2742 | 0.2428 | 0.1088 | 0.2301 | 0.1376 | 0.1489 | 0.1500 | 0.1441 | 0.0906 | 0.1742 |
| A+V+O+P16ctl+S16ctl | F5 | 0.1227 | 0.1551 | 0.2073 | 0.1449 | 0.2159 | 0.1980 | 0.0511 | 0.1085 | 0.0462 | 0.0306 | 0.1280 |
| A+V+O+P16ctl+S16ctl | F7 | 0.0588 | 0.1270 | 0.1981 | 0.1704 | 0.2043 | 0.2709 | 0.0569 | 0.1329 | 0.0073 | 0.0074 | 0.1234 |
| **A+V+O+P16ctl+S16ctl** | F8 | 0.1739 | 0.2180 | 0.2128 | 0.1716 | 0.2438 | 0.1993 | 0.1084 | 0.1626 | 0.0503 | 0.0369 | 0.1578 |

## NLL (lower is better)

| system | fusion | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | DFEval24 | FFIW | KoDF | IDForge | DFDM | macro |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A+V+O | F1 | 0.4453 | 0.4028 | 0.4072 | 0.4861 | 0.5122 | 0.6456 | 0.3488 | 0.3594 | 0.2577 | 0.1671 | 0.4032 |
| A+V+O | F5 | 0.4445 | 0.2916 | 0.3919 | 0.5767 | 0.5707 | 0.7037 | 0.3055 | 0.3245 | 0.1655 | 0.1143 | 0.3889 |
| A+V+O | F7 | 0.2948 | 0.2832 | 0.4244 | 0.6328 | 0.5624 | 0.8221 | 0.2675 | 0.4909 | 0.1158 | 0.0768 | 0.3971 |
| **A+V+O** | F8 | 0.3458 | 0.3024 | 0.3850 | 0.5706 | 0.5059 | 0.7584 | 0.2679 | 0.4764 | 0.1394 | 0.0911 | 0.3843 |
| A+V+O+P16ctl+S16ctl | F1 | 0.3616 | 0.4009 | 0.4278 | 0.5098 | 0.5193 | 0.6922 | 0.3158 | 0.3692 | 0.2076 | 0.1290 | 0.3933 |
| A+V+O+P16ctl+S16ctl | F5 | 0.2765 | 0.2562 | 0.4308 | 0.6000 | 0.5533 | 0.8045 | 0.2401 | 0.3529 | 0.0997 | 0.0578 | 0.3672 |
| A+V+O+P16ctl+S16ctl | F7 | 0.1980 | 0.2922 | 0.6338 | 0.8375 | 0.6837 | 1.1650 | 0.2482 | 0.5936 | 0.0716 | 0.0399 | 0.4763 |
| **A+V+O+P16ctl+S16ctl** | F8 | 0.3375 | 0.3693 | 0.4652 | 0.6478 | 0.6084 | 0.8632 | 0.2958 | 0.5351 | 0.1151 | 0.0801 | 0.4317 |

## Brier (lower is better)

| system | fusion | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | DFEval24 | FFIW | KoDF | IDForge | DFDM | macro |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A+V+O | F1 | 0.1400 | 0.1273 | 0.1356 | 0.1606 | 0.1742 | 0.2251 | 0.1068 | 0.1148 | 0.0697 | 0.0421 | 0.1296 |
| A+V+O | F5 | 0.1490 | 0.0910 | 0.1329 | 0.1843 | 0.1976 | 0.2368 | 0.0984 | 0.0830 | 0.0449 | 0.0299 | 0.1248 |
| A+V+O | F7 | 0.0952 | 0.0900 | 0.1402 | 0.1757 | 0.1819 | 0.2549 | 0.0857 | 0.1208 | 0.0333 | 0.0227 | 0.1200 |
| **A+V+O** | F8 | 0.1069 | 0.0942 | 0.1294 | 0.1681 | 0.1689 | 0.2428 | 0.0855 | 0.1168 | 0.0386 | 0.0254 | 0.1177 |
| A+V+O+P16ctl+S16ctl | F1 | 0.1030 | 0.1272 | 0.1454 | 0.1712 | 0.1775 | 0.2455 | 0.0916 | 0.1196 | 0.0483 | 0.0288 | 0.1258 |
| A+V+O+P16ctl+S16ctl | F5 | 0.0792 | 0.0790 | 0.1462 | 0.1857 | 0.1895 | 0.2641 | 0.0742 | 0.0868 | 0.0227 | 0.0125 | 0.1140 |
| A+V+O+P16ctl+S16ctl | F7 | 0.0615 | 0.0872 | 0.1713 | 0.1918 | 0.1968 | 0.2991 | 0.0734 | 0.1110 | 0.0194 | 0.0114 | 0.1223 |
| **A+V+O+P16ctl+S16ctl** | F8 | 0.1069 | 0.1187 | 0.1529 | 0.1873 | 0.2025 | 0.2679 | 0.0953 | 0.1225 | 0.0316 | 0.0228 | 0.1308 |

## FPR_real at the frozen threshold (lower is better)

| system | fusion | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | DFEval24 | FFIW | KoDF | IDForge | DFDM | macro |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A+V+O | F1 | 0.0337 | 0.0337 | 0.0716 | 0.1197 | 0.0739 | 0.1612 | 0.0415 | 0.0099 | 0.0194 | 0.0337 | 0.0598 |
| A+V+O | F5 | 0.0562 | 0.0562 | 0.0909 | 0.1521 | 0.0957 | 0.2477 | 0.1037 | 0.0223 | 0.0348 | 0.0506 | 0.0910 |
| A+V+O | F7 | 0.0449 | 0.0449 | 0.0551 | 0.1045 | 0.0609 | 0.1729 | 0.0352 | 0.0050 | 0.0110 | 0.0449 | 0.0579 |
| **A+V+O** | F8 | 0.0562 | 0.0562 | 0.0579 | 0.1110 | 0.0739 | 0.1822 | 0.0357 | 0.0050 | 0.0121 | 0.0506 | 0.0641 |
| A+V+O+P16ctl+S16ctl | F1 | 0.1517 | 0.1517 | 0.0551 | 0.1343 | 0.0783 | 0.2430 | 0.0582 | 0.0199 | 0.0429 | 0.1742 | 0.1109 |
| A+V+O+P16ctl+S16ctl | F5 | 0.1348 | 0.1348 | 0.0771 | 0.1724 | 0.1217 | 0.2780 | 0.1107 | 0.0273 | 0.0529 | 0.1685 | 0.1278 |
| A+V+O+P16ctl+S16ctl | F7 | 0.0730 | 0.0730 | 0.0551 | 0.1171 | 0.0652 | 0.1939 | 0.0501 | 0.0149 | 0.0275 | 0.0899 | 0.0760 |
| **A+V+O+P16ctl+S16ctl** | F8 | 0.0674 | 0.0674 | 0.0579 | 0.1261 | 0.0783 | 0.2593 | 0.0553 | 0.0174 | 0.0238 | 0.0674 | 0.0820 |
| *CE baseline (DISCERN_Ext arm D, 3 other views)* | mean-of-logits | 0.1966 | 0.1966 | 0.3691 | 0.3201 | 0.3696 | 0.4416 | -- | -- | -- | -- | 0.3156\* |

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

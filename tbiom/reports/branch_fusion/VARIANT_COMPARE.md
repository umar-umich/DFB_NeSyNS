# TD+LFS+STT-order vs TD+LFS+STT

Both candidates keep LFS and STT and drop App, per UMAR 2026-10-14. They differ only in
STT's positional encoding: real relative timestamps (`STT`) against learned slot
positions (`STT-order`).

Weights fitted once per candidate on development and applied unchanged; `tau` is each
candidate's own EER on FF++-val. Same protocol as every other table here.

| candidate | weights | dev v3.1 |
|---|---|---:|
| TD+LFS+STT-order | [0.25, 0.75, 1.0] | 0.8986 |
| TD+LFS+STT | [0.25, 1.0, 0.25] | 0.9009 |

## AUROC (higher is better)

| candidate | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TD+LFS+STT-order | 0.9739 | 0.9716 | 0.9486 | 0.8565 | 0.8774 | 0.9497 | 0.9554 | 0.9885 | 0.9976 | 0.9854 | 0.9172 | 0.9085 | **0.9442** |
| TD+LFS+STT | 0.9764 | 0.9676 | 0.9495 | 0.8566 | 0.8857 | 0.9638 | 0.9570 | 0.9904 | 0.9977 | 0.9846 | 0.9249 | 0.9078 | **0.9468** |
| *TD+LFS+STT-order − TD+LFS+STT* | -0.0024 | +0.0040 | -0.0009 | -0.0002 | -0.0084 | -0.0141 | -0.0016 | -0.0019 | -0.0001 | +0.0008 | -0.0077 | +0.0007 | *-0.0026* |

## FPR at TPR = 90% (lower is better)

| candidate | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TD+LFS+STT-order | 0.0899 | 0.0899 | 0.1818 | 0.4328 | 0.3565 | 0.1625 | 0.0372 | 0.0281 | 0.0000 | 0.0204 | 0.2507 | 0.2228 | **0.1561** |
| TD+LFS+STT | 0.0506 | 0.0899 | 0.1791 | 0.4181 | 0.3478 | 0.1130 | 0.0347 | 0.0248 | 0.0000 | 0.0204 | 0.1928 | 0.2158 | **0.1406** |
| *TD+LFS+STT-order − TD+LFS+STT* | +0.0393 | +0.0000 | +0.0028 | +0.0147 | +0.0087 | +0.0496 | +0.0025 | +0.0033 | +0.0000 | +0.0000 | +0.0579 | +0.0070 | *+0.0155* |

## FPR at TPR = 95% (lower is better)

| candidate | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TD+LFS+STT-order | 0.1236 | 0.1517 | 0.2865 | 0.6199 | 0.5174 | 0.2121 | 0.3027 | 0.0696 | 0.0112 | 0.0204 | 0.4082 | 0.3994 | **0.2602** |
| TD+LFS+STT | 0.1236 | 0.1854 | 0.2645 | 0.6117 | 0.4826 | 0.1885 | 0.3697 | 0.0348 | 0.0169 | 0.0204 | 0.3432 | 0.3516 | **0.2494** |
| *TD+LFS+STT-order − TD+LFS+STT* | +0.0000 | -0.0337 | +0.0220 | +0.0082 | +0.0348 | +0.0236 | -0.0670 | +0.0348 | -0.0056 | +0.0000 | +0.0650 | +0.0478 | *+0.0108* |

## ECE-15 (lower is better)

| candidate | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TD+LFS+STT-order | 0.2552 | 0.2985 | 0.2484 | 0.1796 | 0.2628 | 0.1399 | 0.1507 | 0.0601 | 0.0499 | 0.0518 | 0.0635 | 0.0326 | **0.1494** |
| TD+LFS+STT | 0.1607 | 0.2135 | 0.2220 | 0.1373 | 0.1953 | 0.0612 | 0.1261 | 0.0533 | 0.0271 | 0.0815 | 0.1225 | 0.0753 | **0.1230** |
| *TD+LFS+STT-order − TD+LFS+STT* | +0.0945 | +0.0849 | +0.0264 | +0.0423 | +0.0674 | +0.0787 | +0.0246 | +0.0068 | +0.0228 | -0.0296 | -0.0590 | -0.0427 | *+0.0264* |

## Brier (lower is better)

| candidate | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TD+LFS+STT-order | 0.1745 | 0.1910 | 0.1857 | 0.1949 | 0.2311 | 0.1302 | 0.1093 | 0.0549 | 0.0278 | 0.0350 | 0.1082 | 0.1053 | **0.1290** |
| TD+LFS+STT | 0.0947 | 0.1209 | 0.1651 | 0.1750 | 0.1853 | 0.0835 | 0.0927 | 0.0382 | 0.0127 | 0.0453 | 0.1276 | 0.1162 | **0.1048** |
| *TD+LFS+STT-order − TD+LFS+STT* | +0.0797 | +0.0701 | +0.0206 | +0.0199 | +0.0458 | +0.0467 | +0.0166 | +0.0167 | +0.0151 | -0.0102 | -0.0194 | -0.0109 | *+0.0242* |

## E-AURC (lower is better)

| candidate | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TD+LFS+STT-order | 0.0280 | 0.0349 | 0.0425 | 0.0931 | 0.0875 | 0.0568 | 0.0271 | 0.0091 | 0.0010 | 0.0120 | 0.0938 | 0.0986 | **0.0487** |
| TD+LFS+STT | 0.0189 | 0.0168 | 0.0353 | 0.0945 | 0.0617 | 0.0269 | 0.0260 | 0.0063 | 0.0016 | 0.0411 | 0.1093 | 0.0985 | **0.0447** |
| *TD+LFS+STT-order − TD+LFS+STT* | +0.0091 | +0.0181 | +0.0072 | -0.0014 | +0.0258 | +0.0298 | +0.0011 | +0.0028 | -0.0005 | -0.0291 | -0.0154 | +0.0001 | *+0.0040* |

## DF40 — all 41 generator sets, by manipulation family

| candidate | face swap | reenactment | all 41 |
|---|---:|---:|---:|
| TD+LFS+STT-order | 0.9889 | 0.9548 | **0.9706** |
| TD+LFS+STT | 0.9894 | 0.9335 | **0.9594** |
| *TD+LFS+STT-order − TD+LFS+STT* | -0.0005 | +0.0213 | *+0.0112* |


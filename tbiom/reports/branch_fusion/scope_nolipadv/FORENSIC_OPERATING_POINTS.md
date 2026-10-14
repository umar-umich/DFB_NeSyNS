# Operating points for forensic use

Per dataset, never a macro. Thresholds come from the FF++ validation rows inside
development and are applied unchanged to every test set -- a threshold tuned per corpus
is not an operating point, it is a result fitted on the thing it is reporting.

**Precision is deliberately absent.** It depends on how many videos reaching the court
are fake, which is unknown and contested; FPR and FNR are properties of the detector
alone. **LR+ = TPR/FPR** is the prevalence-free quantity an expert witness can defend,
and it combines with the court's own prior under Bayes.

## Threshold: EER (research convention)  (tau = 0.3570)

| dataset | TPR (fake caught) | FNR (fake missed) | FPR (genuine flagged) | LR+ | false alarms per 100k genuine |
|---|---:|---:|---:|---:|---:|
| CDFv2 | 0.9176 | 0.0824 | 0.0674 | 13.6 | 6,742 |
| CDFv3 | 0.8973 | 0.1027 | 0.0674 | 13.3 | 6,742 |
| DFD | 0.8057 | 0.1943 | 0.0579 | 13.9 | 5,785 |
| DFDC | 0.6794 | 0.3206 | 0.1261 | 5.4 | 12,613 |
| DFDCP | 0.6816 | 0.3184 | 0.0783 | 8.7 | 7,826 |
| FFIW | 0.8334 | 0.1666 | 0.0553 | 15.1 | 5,533 |
| KoDF | 0.7446 | 0.2554 | 0.0186 | 40.0 | 1,861 |
| IDForge | 0.9615 | 0.0385 | 0.0238 | 40.5 | 2,377 |
| DFDM | 0.9854 | 0.0146 | 0.0674 | 14.6 | 6,742 |
| UADFV | 0.9796 | 0.0204 | 0.1633 | 6.0 | 16,327 |
| DeepSpeak-1.1 | 0.9111 | 0.0889 | 0.1879 | 4.8 | 18,785 |
| DeepSpeak-2.0 | 0.8872 | 0.1128 | 0.1986 | 4.5 | 19,860 |

## Threshold: FPR<=1.0% on development  (tau = 0.2490)

| dataset | TPR (fake caught) | FNR (fake missed) | FPR (genuine flagged) | LR+ | false alarms per 100k genuine |
|---|---:|---:|---:|---:|---:|
| CDFv2 | 0.9676 | 0.0324 | 0.1573 | 6.2 | 15,730 |
| CDFv3 | 0.9355 | 0.0645 | 0.1573 | 5.9 | 15,730 |
| DFD | 0.8585 | 0.1415 | 0.0937 | 9.2 | 9,366 |
| DFDC | 0.7505 | 0.2495 | 0.1849 | 4.1 | 18,488 |
| DFDCP | 0.7736 | 0.2264 | 0.1652 | 4.7 | 16,522 |
| FFIW | 0.8986 | 0.1014 | 0.1026 | 8.8 | 10,259 |
| KoDF | 0.7879 | 0.2121 | 0.0409 | 19.2 | 4,094 |
| IDForge | 0.9809 | 0.0191 | 0.0425 | 23.1 | 4,252 |
| DFDM | 0.9913 | 0.0087 | 0.1517 | 6.5 | 15,169 |
| UADFV | 0.9796 | 0.0204 | 0.2449 | 4.0 | 24,490 |
| DeepSpeak-1.1 | 0.9359 | 0.0641 | 0.2528 | 3.7 | 25,282 |
| DeepSpeak-2.0 | 0.9323 | 0.0677 | 0.2764 | 3.4 | 27,644 |

## Threshold: FPR<=0.1% on development  (tau = 0.4279)

| dataset | TPR (fake caught) | FNR (fake missed) | FPR (genuine flagged) | LR+ | false alarms per 100k genuine |
|---|---:|---:|---:|---:|---:|
| CDFv2 | 0.8588 | 0.1412 | 0.0674 | 12.7 | 6,742 |
| CDFv3 | 0.8643 | 0.1357 | 0.0674 | 12.8 | 6,742 |
| DFD | 0.7810 | 0.2190 | 0.0441 | 17.7 | 4,408 |
| DFDC | 0.6362 | 0.3638 | 0.1006 | 6.3 | 10,065 |
| DFDCP | 0.6179 | 0.3821 | 0.0522 | 11.8 | 5,217 |
| FFIW | 0.7988 | 0.2012 | 0.0352 | 22.7 | 3,516 |
| KoDF | 0.7034 | 0.2966 | 0.0062 | 113.4 | 620 |
| IDForge | 0.9471 | 0.0529 | 0.0151 | 62.9 | 1,507 |
| DFDM | 0.9802 | 0.0198 | 0.0449 | 21.8 | 4,494 |
| UADFV | 0.9592 | 0.0408 | 0.1429 | 6.7 | 14,286 |
| DeepSpeak-1.1 | 0.8819 | 0.1181 | 0.1582 | 5.6 | 15,819 |
| DeepSpeak-2.0 | 0.8293 | 0.1707 | 0.1439 | 5.8 | 14,385 |

## How to read these

A video offered as AUTHENTIC evidence is endangered by a false negative -- the FNR
column. A video challenged as FABRICATED is endangered by a false positive -- the FPR
column. The two are not interchangeable and the right one depends on which side is
relying on the video, so both are reported at every threshold.

For police triage the last column is the operative number: it is how many genuine
videos an analyst has to clear for every hundred thousand screened, and it is what
makes an EER threshold unusable at scale however good the AUROC above it looks.

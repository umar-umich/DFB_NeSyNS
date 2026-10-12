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
| CDFv3 | 0.8988 | 0.1012 | 0.0674 | 13.3 | 6,742 |
| DFD | 0.8057 | 0.1943 | 0.0579 | 13.9 | 5,785 |
| DFDC | 0.6794 | 0.3206 | 0.1261 | 5.4 | 12,613 |
| DFDCP | 0.6816 | 0.3184 | 0.0783 | 8.7 | 7,826 |
| DFEval24 | 0.4974 | 0.5026 | 0.2593 | 1.9 | 25,935 |
| FFIW | 0.8334 | 0.1666 | 0.0553 | 15.1 | 5,533 |
| KoDF | 0.8508 | 0.1492 | 0.0174 | 49.0 | 1,737 |
| IDForge | 0.9615 | 0.0385 | 0.0238 | 40.5 | 2,377 |
| DFDM | 0.9854 | 0.0146 | 0.0674 | 14.6 | 6,742 |

## Threshold: FPR<=1.0% on development  (tau = 0.2490)

| dataset | TPR (fake caught) | FNR (fake missed) | FPR (genuine flagged) | LR+ | false alarms per 100k genuine |
|---|---:|---:|---:|---:|---:|
| CDFv2 | 0.9676 | 0.0324 | 0.1573 | 6.2 | 15,730 |
| CDFv3 | 0.9362 | 0.0638 | 0.1573 | 6.0 | 15,730 |
| DFD | 0.8585 | 0.1415 | 0.0937 | 9.2 | 9,366 |
| DFDC | 0.7505 | 0.2495 | 0.1849 | 4.1 | 18,488 |
| DFDCP | 0.7736 | 0.2264 | 0.1652 | 4.7 | 16,522 |
| DFEval24 | 0.5933 | 0.4067 | 0.3435 | 1.7 | 34,346 |
| FFIW | 0.8986 | 0.1014 | 0.1026 | 8.8 | 10,259 |
| KoDF | 0.8942 | 0.1058 | 0.0347 | 25.7 | 3,474 |
| IDForge | 0.9809 | 0.0191 | 0.0425 | 23.1 | 4,252 |
| DFDM | 0.9913 | 0.0087 | 0.1517 | 6.5 | 15,169 |

## Threshold: FPR<=0.1% on development  (tau = 0.4279)

| dataset | TPR (fake caught) | FNR (fake missed) | FPR (genuine flagged) | LR+ | false alarms per 100k genuine |
|---|---:|---:|---:|---:|---:|
| CDFv2 | 0.8588 | 0.1412 | 0.0674 | 12.7 | 6,742 |
| CDFv3 | 0.8648 | 0.1352 | 0.0674 | 12.8 | 6,742 |
| DFD | 0.7810 | 0.2190 | 0.0441 | 17.7 | 4,408 |
| DFDC | 0.6362 | 0.3638 | 0.1006 | 6.3 | 10,065 |
| DFDCP | 0.6179 | 0.3821 | 0.0522 | 11.8 | 5,217 |
| DFEval24 | 0.4301 | 0.5699 | 0.1963 | 2.2 | 19,626 |
| FFIW | 0.7988 | 0.2012 | 0.0352 | 22.7 | 3,516 |
| KoDF | 0.8024 | 0.1976 | 0.0099 | 80.8 | 993 |
| IDForge | 0.9471 | 0.0529 | 0.0151 | 62.9 | 1,507 |
| DFDM | 0.9802 | 0.0198 | 0.0449 | 21.8 | 4,494 |

## How to read these

A video offered as AUTHENTIC evidence is endangered by a false negative -- the FNR
column. A video challenged as FABRICATED is endangered by a false positive -- the FPR
column. The two are not interchangeable and the right one depends on which side is
relying on the video, so both are reported at every threshold.

For police triage the last column is the operative number: it is how many genuine
videos an analyst has to clear for every hundred thousand screened, and it is what
makes an EER threshold unusable at scale however good the AUROC above it looks.

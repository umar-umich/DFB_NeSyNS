# Ours against DiCoME at matched false-alarm targets

Each system's threshold comes from its OWN FF++ scores at the same target rate, then is
applied unchanged to every test set. Neither side is tuned on what it reports, and
neither is given the other's threshold -- that would compare calibrations, not
detectors.

DiCoME publishes no false-alarm figures; their evaluation computes video-level AUROC and
no threshold appears anywhere in it. These rows are measured from their own per-video
prediction CSVs. Blank cells are sets they never ran.

## Target false-alarm rate 1% (set on FF++, applied unchanged)

| dataset | system | TPR | FPR | LR+ |
|---|---|---:|---:|---:|
| **CDFv2** | **ours (PCR6+-R, 5 branches)** | 0.9676 | 0.1573 | 6.2 |
| | DiCoME (trained) | 0.9294 | 0.1292 | 7.2 |
| | DiCoME (repro) | — | — | — |
| **CDFv3** | **ours (PCR6+-R, 5 branches)** | 0.9362 | 0.1573 | 6.0 |
| | DiCoME (trained) | 0.8977 | 0.1292 | 6.9 |
| | DiCoME (repro) | 0.7684 | 0.0337 | 22.8 |
| **DFD** | **ours (PCR6+-R, 5 branches)** | 0.8585 | 0.0937 | 9.2 |
| | DiCoME (trained) | 0.8833 | 0.1708 | 5.2 |
| | DiCoME (repro) | 0.7298 | 0.0083 | 88.3 |
| **DFDC** | **ours (PCR6+-R, 5 branches)** | 0.7505 | 0.1849 | 4.1 |
| | DiCoME (trained) | 0.8652 | 0.3162 | 2.7 |
| | DiCoME (repro) | 0.6815 | 0.1037 | 6.6 |
| **DFDCP** | **ours (PCR6+-R, 5 branches)** | 0.7736 | 0.1652 | 4.7 |
| | DiCoME (trained) | 0.8656 | 0.3826 | 2.3 |
| | DiCoME (repro) | 0.7028 | 0.1130 | 6.2 |
| **FFIW** | **ours (PCR6+-R, 5 branches)** | 0.8986 | 0.1026 | 8.8 |
| | DiCoME (trained) | — | — | — |
| | DiCoME (repro) | — | — | — |
| **KoDF** | **ours (PCR6+-R, 5 branches)** | 0.8942 | 0.0347 | 25.7 |
| | DiCoME (trained) | — | — | — |
| | DiCoME (repro) | — | — | — |
| **IDForge** | **ours (PCR6+-R, 5 branches)** | 0.9809 | 0.0425 | 23.1 |
| | DiCoME (trained) | — | — | — |
| | DiCoME (repro) | — | — | — |
| **DFDM** | **ours (PCR6+-R, 5 branches)** | 0.9913 | 0.1517 | 6.5 |
| | DiCoME (trained) | — | — | — |
| | DiCoME (repro) | — | — | — |

## Target false-alarm rate 5% (set on FF++, applied unchanged)

| dataset | system | TPR | FPR | LR+ |
|---|---|---:|---:|---:|
| **CDFv2** | **ours (PCR6+-R, 5 branches)** | 1.0000 | 0.4438 | 2.3 |
| | DiCoME (trained) | 0.9794 | 0.2640 | 3.7 |
| | DiCoME (repro) | — | — | — |
| **CDFv3** | **ours (PCR6+-R, 5 branches)** | 0.9791 | 0.4438 | 2.2 |
| | DiCoME (trained) | 0.9517 | 0.2640 | 3.6 |
| | DiCoME (repro) | 0.9316 | 0.2247 | 4.1 |
| **DFD** | **ours (PCR6+-R, 5 branches)** | 0.9361 | 0.2204 | 4.2 |
| | DiCoME (trained) | 0.9244 | 0.3140 | 2.9 |
| | DiCoME (repro) | 0.8664 | 0.1460 | 5.9 |
| **DFDC** | **ours (PCR6+-R, 5 branches)** | 0.8527 | 0.3002 | 2.8 |
| | DiCoME (trained) | 0.9184 | 0.4600 | 2.0 |
| | DiCoME (repro) | 0.8539 | 0.2786 | 3.1 |
| **DFDCP** | **ours (PCR6+-R, 5 branches)** | 0.9009 | 0.2739 | 3.3 |
| | DiCoME (trained) | 0.9127 | 0.5087 | 1.8 |
| | DiCoME (repro) | 0.8962 | 0.3696 | 2.4 |
| **FFIW** | **ours (PCR6+-R, 5 branches)** | 0.9545 | 0.1735 | 5.5 |
| | DiCoME (trained) | — | — | — |
| | DiCoME (repro) | — | — | — |
| **KoDF** | **ours (PCR6+-R, 5 branches)** | 0.9113 | 0.0695 | 13.1 |
| | DiCoME (trained) | — | — | — |
| | DiCoME (repro) | — | — | — |
| **IDForge** | **ours (PCR6+-R, 5 branches)** | 0.9926 | 0.1309 | 7.6 |
| | DiCoME (trained) | — | — | — |
| | DiCoME (repro) | — | — | — |
| **DFDM** | **ours (PCR6+-R, 5 branches)** | 0.9983 | 0.4551 | 2.2 |
| | DiCoME (trained) | — | — | — |
| | DiCoME (repro) | — | — | — |


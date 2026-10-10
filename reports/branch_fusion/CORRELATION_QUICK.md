# Branch correlation structure

> phi is computed PER VIDEO on binary errors, each branch thresholded at its own EER point from the FF++ validation rows inside development. An earlier version of this analysis correlated the ten dataset-level AUROCs instead; that statistic is dominated by how hard each dataset is and cannot see whether two branches fail on the same videos.

### Development pool  (n = 5814 videos)

| branch | AUROC | error rate |
|---|---:|---:|
| A | 0.9066 | 0.1689 |
| V | 0.8452 | 0.2902 |
| O | 0.8455 | 0.2387 |
| Y | 0.9082 | 0.1761 |

**phi — correlation of BINARY ERRORS (lower = more complementary)**

| | A | V | O | Y |
|---|---:|---:|---:|---:|
| **A** | — | +0.043 | +0.395 | +0.789 |
| **V** | +0.043 | — | +0.106 | +0.083 |
| **O** | +0.395 | +0.106 | — | +0.415 |
| **Y** | +0.789 | +0.083 | +0.415 | — |

**r_p — correlation of predicted probabilities**

| | A | V | O | Y |
|---|---:|---:|---:|---:|
| **A** | — | +0.449 | +0.745 | +0.973 |
| **V** | +0.449 | — | +0.399 | +0.443 |
| **O** | +0.745 | +0.399 | — | +0.743 |
| **Y** | +0.973 | +0.443 | +0.743 | — |

### All ten test sets pooled  (n = 27974 videos)

| branch | AUROC | error rate |
|---|---:|---:|
| A | 0.9392 | 0.1305 |
| V | 0.8353 | 0.3165 |
| O | 0.9071 | 0.1713 |
| Y | 0.9385 | 0.1455 |

**phi — correlation of BINARY ERRORS (lower = more complementary)**

| | A | V | O | Y |
|---|---:|---:|---:|---:|
| **A** | — | +0.166 | +0.415 | +0.786 |
| **V** | +0.166 | — | +0.108 | +0.205 |
| **O** | +0.415 | +0.108 | — | +0.403 |
| **Y** | +0.786 | +0.205 | +0.403 | — |

**r_p — correlation of predicted probabilities**

| | A | V | O | Y |
|---|---:|---:|---:|---:|
| **A** | — | +0.528 | +0.808 | +0.984 |
| **V** | +0.528 | — | +0.481 | +0.527 |
| **O** | +0.808 | +0.481 | — | +0.808 |
| **Y** | +0.984 | +0.527 | +0.808 | — |

### Every triple: strength, worst pairwise phi, and measured fusion

| triple | fused macro | gain over best single | worst phi | mean phi | weakest branch |
|---|---:|---:|---:|---:|---:|
| A+V+O | 0.9047 | +0.0311 | +0.395 | +0.181 | 0.8452 |
| V+O+Y | 0.9026 | +0.0330 | +0.415 | +0.201 | 0.8452 |
| A+V+Y | 0.9004 | +0.0268 | +0.789 | +0.305 | 0.8452 |
| A+O+Y | 0.8778 | +0.0042 | +0.789 | +0.533 | 0.8455 |

**Spearman(worst phi, gain over best single) = -0.738**  over 4 triples; against the fused macro -0.949.

A NEGATIVE spearman means more-decorrelated triples fuse better, which is the assumption behind picking branches by decorrelation. A value near zero means the correlation matrix does NOT predict fusion on this data and must be read as description only -- section 4 forbids promoting a branch because its errors decorrelate, and this is the check that keeps that rule honest.


# Branch correlation structure

> phi is computed PER VIDEO on binary errors, each branch thresholded at its own EER point from the FF++ validation rows inside development. An earlier version of this analysis correlated the ten dataset-level AUROCs instead; that statistic is dominated by how hard each dataset is and cannot see whether two branches fail on the same videos.

### Development pool  (n = 5814 videos)

| branch | AUROC | error rate |
|---|---:|---:|
| A | 0.9066 | 0.1689 |
| S | 0.9009 | 0.1844 |
| V | 0.8452 | 0.2902 |
| O | 0.8455 | 0.2387 |
| M | 0.8356 | 0.2919 |
| P | 0.8929 | 0.2788 |

**phi — correlation of BINARY ERRORS (lower = more complementary)**

| | A | S | V | O | M | P |
|---|---:|---:|---:|---:|---:|---:|
| **A** | — | +0.748 | +0.043 | +0.395 | +0.061 | +0.336 |
| **S** | +0.748 | — | +0.089 | +0.441 | +0.125 | +0.486 |
| **V** | +0.043 | +0.089 | — | +0.106 | +0.439 | +0.180 |
| **O** | +0.395 | +0.441 | +0.106 | — | +0.131 | +0.476 |
| **M** | +0.061 | +0.125 | +0.439 | +0.131 | — | +0.216 |
| **P** | +0.336 | +0.486 | +0.180 | +0.476 | +0.216 | — |

**r_p — correlation of predicted probabilities**

| | A | S | V | O | M | P |
|---|---:|---:|---:|---:|---:|---:|
| **A** | — | +0.840 | +0.449 | +0.745 | +0.441 | +0.844 |
| **S** | +0.840 | — | +0.363 | +0.540 | +0.353 | +0.674 |
| **V** | +0.449 | +0.363 | — | +0.399 | +0.607 | +0.382 |
| **O** | +0.745 | +0.540 | +0.399 | — | +0.362 | +0.714 |
| **M** | +0.441 | +0.353 | +0.607 | +0.362 | — | +0.407 |
| **P** | +0.844 | +0.674 | +0.382 | +0.714 | +0.407 | — |

### All ten test sets pooled  (n = 27974 videos)

| branch | AUROC | error rate |
|---|---:|---:|
| A | 0.9392 | 0.1305 |
| S | 0.9366 | 0.1494 |
| V | 0.8353 | 0.3165 |
| O | 0.9071 | 0.1713 |
| M | 0.7281 | 0.4341 |
| P | 0.9057 | 0.2197 |

**phi — correlation of BINARY ERRORS (lower = more complementary)**

| | A | S | V | O | M | P |
|---|---:|---:|---:|---:|---:|---:|
| **A** | — | +0.792 | +0.166 | +0.415 | +0.096 | +0.525 |
| **S** | +0.792 | — | +0.225 | +0.411 | +0.150 | +0.618 |
| **V** | +0.166 | +0.225 | — | +0.108 | +0.434 | +0.288 |
| **O** | +0.415 | +0.411 | +0.108 | — | +0.028 | +0.397 |
| **M** | +0.096 | +0.150 | +0.434 | +0.028 | — | +0.174 |
| **P** | +0.525 | +0.618 | +0.288 | +0.397 | +0.174 | — |

**r_p — correlation of predicted probabilities**

| | A | S | V | O | M | P |
|---|---:|---:|---:|---:|---:|---:|
| **A** | — | +0.883 | +0.528 | +0.808 | +0.325 | +0.880 |
| **S** | +0.883 | — | +0.427 | +0.685 | +0.291 | +0.769 |
| **V** | +0.528 | +0.427 | — | +0.481 | +0.509 | +0.477 |
| **O** | +0.808 | +0.685 | +0.481 | — | +0.279 | +0.755 |
| **M** | +0.325 | +0.291 | +0.509 | +0.279 | — | +0.288 |
| **P** | +0.880 | +0.769 | +0.477 | +0.755 | +0.288 | — |

### Every triple: strength, worst pairwise phi, and measured fusion

| triple | fused macro | gain over best single | worst phi | mean phi | weakest branch |
|---|---:|---:|---:|---:|---:|
| A+V+O | 0.9047 | +0.0311 | +0.395 | +0.181 | 0.8452 |
| A+V+P | 0.9041 | +0.0305 | +0.336 | +0.186 | 0.8452 |
| V+O+P | 0.9032 | +0.0519 | +0.476 | +0.254 | 0.8452 |
| A+O+M | 0.9030 | +0.0294 | +0.395 | +0.196 | 0.8356 |
| A+M+P | 0.8995 | +0.0259 | +0.336 | +0.204 | 0.8356 |
| A+S+V | 0.8983 | +0.0247 | +0.748 | +0.293 | 0.8452 |
| A+V+M | 0.8983 | +0.0247 | +0.439 | +0.181 | 0.8356 |
| O+M+P | 0.8953 | +0.0440 | +0.476 | +0.274 | 0.8356 |
| A+S+M | 0.8947 | +0.0211 | +0.748 | +0.311 | 0.8356 |
| V+M+P | 0.8933 | +0.0420 | +0.439 | +0.278 | 0.8356 |
| S+V+O | 0.8918 | +0.0276 | +0.441 | +0.212 | 0.8452 |
| S+M+P | 0.8875 | +0.0234 | +0.486 | +0.276 | 0.8356 |
| S+O+M | 0.8868 | +0.0227 | +0.441 | +0.232 | 0.8356 |
| S+V+P | 0.8867 | +0.0226 | +0.486 | +0.251 | 0.8452 |
| A+S+O | 0.8790 | +0.0054 | +0.748 | +0.528 | 0.8455 |
| A+O+P | 0.8787 | +0.0051 | +0.476 | +0.402 | 0.8455 |
| A+S+P | 0.8776 | +0.0040 | +0.748 | +0.523 | 0.8929 |
| V+O+M | 0.8761 | +0.0509 | +0.439 | +0.225 | 0.8356 |
| S+V+M | 0.8742 | +0.0101 | +0.439 | +0.218 | 0.8356 |
| S+O+P | 0.8694 | +0.0053 | +0.486 | +0.468 | 0.8455 |

**Spearman(worst phi, gain over best single) = -0.552**  over 20 triples; against the fused macro -0.444.

A NEGATIVE spearman means more-decorrelated triples fuse better, which is the assumption behind picking branches by decorrelation. A value near zero means the correlation matrix does NOT predict fusion on this data and must be read as description only -- section 4 forbids promoting a branch because its errors decorrelate, and this is the check that keeps that rule honest.


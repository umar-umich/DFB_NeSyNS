# Y16 against A under jpeg:40

Development pool. Thresholds are the CLEAN EER points, reused on the degraded data: a
branch rethresholded after the attack would be scored at an operating point it could not
have chosen in deployment.

| branch | AUROC clean (n=6902) | AUROC jpeg:40 (n=6902) | change |
|---|---:|---:|---:|
| A | 0.9090 | 0.8133 | -0.0958 |
| Y16 | 0.9047 | 0.7857 | -0.1190 |

**Per-video error phi, Y16 against A**

| condition | phi |
|---|---:|
| clean | +0.706 |
| jpeg:40 | +0.528 |
| change | -0.178 |

Y16 and A SEPARATE under compression: phi falls 0.178, so the two views stop failing on the same videos exactly where robustness is at stake. That is evidence for the row, not just a reason to hope for one.

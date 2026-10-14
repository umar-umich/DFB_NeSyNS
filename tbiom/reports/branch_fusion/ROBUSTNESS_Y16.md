# Y16 against A under degradation

Development pool. Thresholds are the CLEAN EER points, reused on every degraded pool: a
branch rethresholded after the attack would be scored at an operating point it could not
have chosen in deployment.

| condition | A AUROC | Y16 AUROC | dA | dY16 | phi(Y16, A) | d phi |
|---|---:|---:|---:|---:|---:|---:|
| clean (n=6902) | 0.9090 | 0.9047 | -- | -- | +0.706 | -- |
| jpeg:40 (n=6902) | 0.8133 | 0.7857 | -0.0958 | -0.1190 | +0.528 | -0.178 |
| resize:0.6 (n=6902) | 0.8796 | 0.8648 | -0.0294 | -0.0399 | +0.877 | +0.171 |

**How to read this.** A fall in phi is the result that matters: it means Y16 and A
stop failing on the same videos exactly where robustness is at stake, which is the
evidence for giving Y16 a row rather than treating it as a duplicate of A. A fall in
Y16's own AUROC larger than A's is NOT a contradiction and must not be written as a
robustness win: Y16 is differently wrong, not less wrong, and for a fusion row that
is the property that counts. The two claims are separate and the table reports both.

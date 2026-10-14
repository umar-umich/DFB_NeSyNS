# Figure — vacuity and risk-coverage

`figures/uncertainty_riskcoverage.png`, regenerated with the panels chosen by AUROC among the sets with at least forty errors: IDForge, FFIW, CDFv3, DFD.

## Top row — the separation holds

Vacuity split by whether the fused prediction is right. Misclassified videos carry
visibly more vacuity than correct ones on every panel, which is the claim DiCoME's
Figure 3 makes and it reproduces here:

| set | u(correct) | u(wrong) | ratio | errors |
|---|---:|---:|---:|---:|
| IDForge | 0.0059 | 0.0208 | 3.54x | 174 |
| FFIW | 0.0075 | 0.0254 | 3.40x | 295 |
| CDFv3 | 0.0056 | 0.0193 | 3.45x | 380 |
| DFD | 0.0034 | 0.0100 | 2.90x | 534 |

## Bottom row — is it actually a better rejector?

Separating errors in a histogram does not by itself make a quantity a better rejector
than the decision margin, which costs nothing and needs no evidential model. So the
verdict below is computed, not asserted. Area under the risk-coverage curve, lower is
better, both rejectors on the same coverage grid:

| set | vacuity (ours) | margin \|p-tau\| | oracle | vacuity better? |
|---|---:|---:|---:|---|
| IDForge | 0.0094 | 0.1085 | 0.0007 | yes |
| FFIW | 0.0433 | 0.1642 | 0.0047 | yes |
| CDFv3 | 0.0915 | 0.0938 | 0.0101 | yes |
| DFD | 0.0478 | 0.1201 | 0.0149 | yes |

Vacuity wins 4 of the 4 panels (IDForge, FFIW, CDFv3, DFD).

### What may be claimed from this

Both rows. Vacuity separates errors in the histogram AND orders them better
than the decision margin on every panel, so the evidential apparatus is paying
for itself as a rejector and not only as an interpretable quantity.

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
| CDFv3 | 0.0054 | 0.0193 | 3.56x | 362 |
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
| CDFv3 | 0.0956 | 0.0931 | 0.0105 | NO |
| DFD | 0.0478 | 0.1201 | 0.0149 | yes |

Vacuity wins 3 of the 4 panels (IDForge, FFIW, DFD), and loses on CDFv3.

### What may be claimed from this

The first row on every panel, the second on 3 of 4. Vacuity
is the better rejector where it wins and the margin is free, so the claim to make
is the one the table supports: vacuity dominates the margin on most sets, by a
wide margin on IDForge, and ties or
loses on CDFv3. State the exception. A reviewer who runs the margin
baseline will find it, and it costs nothing to report.

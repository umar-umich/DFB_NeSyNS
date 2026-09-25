# Conflict-conditioned reliability — REJECTED

Pre-registered in `tbiom/docs/ccr-preregistration.md` before any number below existed.
Code `analysis/tbiom/ccr/ccr.py`, output `analysis/ccr/ccr.json`.

## The question

The deployed system fits one reliability triple per fold and applies it to every video. CCR predicts
w per video from inter-view conflict, trained by back-propagating binary cross-entropy on the fused
probability **through** the PCR6+ fusion — possible because the Task 1 implementation is
differentiable and reproduces the deployed fusion exactly.

## Result

| arm | v3.1 mean | sd | delta vs const | folds with interval above zero |
|---|---:|---:|---:|---:|
| `const` (deployed, one triple per fold) | 0.8794 | 0.0051 | — | — |
| `ccr` (conflict features) | 0.8802 | 0.0053 | **+0.0008** | 3/5 |
| `ccr-shuffled` (control) | 0.8799 | 0.0055 | **+0.0005** | 3/5 |

Bar: +0.005 mean **and** interval above zero in at least 3 of 5 folds, with the shuffled control not
passing. CCR reaches +0.0008, a sixth of the bar. **REJECTED.**

## The control is the finding

`ccr-shuffled` permutes the conflict features across videos, destroying any association between a
video's conflict and its own fusion, and it scores **+0.0005 against CCR's +0.0008** — a difference of
0.0003 — while meeting the same 3-of-5 fold criterion.

Whatever CCR gains is therefore **not conflict**. It is the extra capacity nudging the reliability
triple in a globally useful direction, which a permuted feature vector supplies just as well. Without
this arm the result would read as a weak positive: "+0.0008, interval above zero in 3 of 5 folds".
With it, the mechanism is shown to be absent.

## Where this leaves the fusion

Four independent attempts, all negative:

| attempt | conditioning | outcome |
|---|---|---|
| Grouped partitioned t-norm | operator | grouped-R identical to two-view T+OM |
| PCR6+ direct three-source | operator | AUROC non-inferior, safeguards 1/4 |
| Reliability controllers (MLP/GRU/Mamba) | quality | −0.0020, interval above zero 0/5 |
| **CCR** | **conflict** | **+0.0008, control matches it** |

Two of those changed the combination rule and two changed the reliability. Neither route moves the
number, and the reason is visible in the opinions rather than in the operators: Appearance is
discounted to zero in three of five folds, Off-manifold's contribution is five times larger when
Appearance is absent, and the video view contributes +0.058 to +0.069 in every fold. The views are
redundant and one dominates. **No combination rule can manufacture complementarity that is not in the
opinions**, and no reliability schedule can either.

G0a remains true and is not contradicted: conflict genuinely predicts the fused system's own errors,
+0.0312 with an interval of [+0.0201, +0.0424]. That is a statement about **ranking errors after the
fact**, which is what evidential vacuity is for. It does not follow that conflict can be used
*prospectively* to reweight the views, and CCR is the measurement showing it cannot.

# Symbolic gate — the overlay as a selective-prediction rule

The frozen detector is untouched. The overlay decides only whether to answer. Abstain where the rules return Inconclusive (R4 conflict-with-insufficient-evidence, R0 fallback); answer everywhere else. `tau` is the frozen dev EER point.

Controls abstain at the **same coverage, matched per dataset**, using the model's own mean branch vacuity, the fused margin |p−τ|, or a coin. Matching coverage per dataset hands the controls information a deployed gate would not have; that is intentional.

## Per dataset — selective balanced error (lower is better)

| dataset | coverage | no gate | **symbolic** | vacuity | margin | random |
|---|---:|---:|---:|---:|---:|---:|
| CDFv2 | 0.842 | 0.1428 | **0.0951** | 0.1167 | 0.1008 | 0.1308 |
| CDFv3 | 0.812 | 0.1895 | **0.1733** | 0.1641 | 0.1613 | 0.1862 |
| DFD | 0.876 | 0.1293 | **0.1136** | 0.0914 | 0.1098 | 0.1311 |
| DFDC | 0.875 | 0.1866 | **0.1486** | 0.1579 | 0.1496 | 0.1869 |
| DFDCP | 0.806 | 0.2006 | **0.1362** | 0.1709 | 0.1464 | 0.2075 |
| DFEval24 | 0.713 | 0.3650 | **0.3475** | 0.3226 | 0.3447 | 0.3714 |

## Per dataset — selective AUROC (higher is better)

| dataset | coverage | no gate | **symbolic** | vacuity | margin | random |
|---|---:|---:|---:|---:|---:|---:|
| CDFv2 | 0.842 | 0.9474 | **0.9672** | 0.9669 | 0.9661 | 0.9565 |
| CDFv3 | 0.812 | 0.8871 | **0.8961** | 0.8953 | 0.8988 | 0.8889 |
| DFD | 0.876 | 0.9454 | **0.9501** | 0.9643 | 0.9518 | 0.9435 |
| DFDC | 0.875 | 0.8906 | **0.9088** | 0.9095 | 0.9083 | 0.8893 |
| DFDCP | 0.806 | 0.9038 | **0.9379** | 0.9224 | 0.9276 | 0.8970 |
| DFEval24 | 0.713 | 0.6866 | **0.6999** | 0.7224 | 0.7006 | 0.6852 |

## Pooled over the six OOD sets

| gate | coverage | balanced error | AUROC |
|---|---:|---:|---:|
| none | 1.000 | 0.2233 | 0.8507 |
| **symbolic** | 0.841 | 0.1835 | 0.8717 |
| vacuity | 0.841 | 0.1836 | 0.8767 |
| margin | 0.841 | 0.1835 | 0.8720 |
| random | 0.840 | 0.2214 | 0.8516 |

## Why

| dataset | J(symbolic, vacuity) | J(symbolic, margin) | vacuity ranks errors | symbolic flag ranks errors |
|---|---:|---:|---:|---:|
| CDFv2 | 0.813 | 0.892 | 0.778 | 0.672 |
| CDFv3 | 0.796 | 0.873 | 0.736 | 0.517 |
| DFD | 0.829 | 0.919 | 0.863 | 0.551 |
| DFDC | 0.842 | 0.922 | 0.740 | 0.609 |
| DFDCP | 0.833 | 0.886 | 0.756 | 0.671 |
| DFEval24 | 0.721 | 0.807 | 0.626 | 0.563 |

## Incremental value over the model's own uncertainty

A wrong/right predictor fitted on **dev only**, then read on OOD. If the symbolic predicates carry error information vacuity does not, the second column wins.

| dataset | vacuity alone | vacuity + symbolic predicates | Δ |
|---|---:|---:|---:|
| CDFv2 | 0.778 | 0.837 | +0.059 |
| CDFv3 | 0.736 | 0.799 | +0.063 |
| DFD | 0.863 | 0.874 | +0.010 |
| DFDC | 0.740 | 0.726 | -0.014 |
| DFDCP | 0.756 | 0.755 | -0.001 |
| DFEval24 | 0.626 | 0.626 | +0.001 |
| **mean** | **0.750** | **0.769** | **+0.020** |

## Verdict

**At its own operating point the symbolic gate is indistinguishable from a trivial one.** It abstains at a fixed 0.841 coverage, cutting pooled balanced error 0.2233 → 0.1835. At that same coverage the model's own vacuity gives 0.1836 and the fused margin 0.1835 — a total spread of 0.0001 across all three, which is nothing. All three clear the random floor (0.2214), so the rules are not noise; they are simply not distinguishable from free alternatives here.

**The comparison is coverage-bound, and that matters for how it is written up.** The rule set yields ONE operating point; vacuity and margin are dials. At 0.841 coverage only 16% of samples are withheld, so every gate is close to the no-gate error and they converge by construction. Turned down to ~0.60 coverage the graded signals separate clearly (see the risk-coverage rows above) — but the symbolic rules cannot be turned down, and a fixed abstention rate is a real limitation of a rule-based gate rather than an artifact of this measurement.

**As a standalone signal the Boolean flag remains weaker than graded vacuity**, and that has not changed: it ranks errors at 0.517–0.672 across the six sets against vacuity's 0.626–0.863. Collapsing a graded quantity into a rule firing throws information away, which is the mechanism.

**But as an ADDITION to vacuity the predicates now carry something.** A wrong/right predictor fitted on dev and read on OOD improves by **+0.020** when the symbolic predicates are supplied alongside vacuity, positive on 4 of 6 sets. **This reverses the earlier finding.** Under the hand-written rule set the same test came out at −0.026 — the predicates fit dev and did not transfer. Under the dev-searched rule set they do transfer. The change is not the search being cleverer; it is that the relaxed decisiveness threshold makes `strong_<branch>` fire on far more samples, so the predicates carry graded information instead of marking a rare extreme.

**How much to claim from that: not much yet.** +0.020 mean with 2 of 6 sets negative, on n=6, is not a result — it is a reason to test the idea properly rather than a finding. What it does do is retire the clean negative that was previously reported here.

**What survives regardless.** The symbolic layer remains a faithful *explanation* of the fused decision — that is Phase S, measured by intervention — and nothing here shows it is an independent source of evidence strong enough to replace the model's own uncertainty. Keeping the overlay post-hoc is still the position the measurements support. The honest summary changed from "the rules add no decision value" to "the rules add no decision value at their own operating point, and may add a little when combined with vacuity, on evidence too thin to bank".

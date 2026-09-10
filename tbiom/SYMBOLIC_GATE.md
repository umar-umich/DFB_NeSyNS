# Symbolic gate — the overlay as a selective-prediction rule

The frozen detector is untouched. The overlay decides only whether to answer. Abstain where the rules return Inconclusive (R4 conflict-with-insufficient-evidence, R0 fallback); answer everywhere else. `tau` is the frozen dev EER point.

Controls abstain at the **same coverage, matched per dataset**, using the model's own mean branch vacuity, the fused margin |p−τ|, or a coin. Matching coverage per dataset hands the controls information a deployed gate would not have; that is intentional.

## Per dataset — selective balanced error (lower is better)

| dataset | coverage | no gate | **symbolic** | vacuity | margin | random |
|---|---:|---:|---:|---:|---:|---:|
| CDFv2 | 0.598 | 0.1428 | **0.1154** | 0.0438 | 0.1142 | 0.1289 |
| CDFv3 | 0.518 | 0.1895 | **0.2206** | 0.0990 | 0.2102 | 0.1911 |
| DFD | 0.599 | 0.1293 | **0.1105** | 0.0252 | 0.1153 | 0.1359 |
| DFDC | 0.672 | 0.1866 | **0.1107** | 0.1232 | 0.1123 | 0.1893 |
| DFDCP | 0.557 | 0.2006 | **0.0613** | 0.1507 | 0.0664 | 0.2224 |
| DFEval24 | 0.506 | 0.3650 | **0.4148** | 0.2962 | 0.3428 | 0.3691 |

## Per dataset — selective AUROC (higher is better)

| dataset | coverage | no gate | **symbolic** | vacuity | margin | random |
|---|---:|---:|---:|---:|---:|---:|
| CDFv2 | 0.598 | 0.9474 | **0.9624** | 0.9902 | 0.9605 | 0.9506 |
| CDFv3 | 0.518 | 0.8871 | **0.8606** | 0.9304 | 0.8571 | 0.8796 |
| DFD | 0.599 | 0.9454 | **0.9411** | 0.9938 | 0.9424 | 0.9428 |
| DFDC | 0.672 | 0.8906 | **0.9287** | 0.9301 | 0.9282 | 0.8871 |
| DFDCP | 0.557 | 0.9038 | **0.9647** | 0.9248 | 0.9639 | 0.8865 |
| DFEval24 | 0.506 | 0.6866 | **0.6345** | 0.7602 | 0.6885 | 0.6830 |

## Pooled over the six OOD sets

| gate | coverage | balanced error | AUROC |
|---|---:|---:|---:|
| none | 1.000 | 0.2233 | 0.8507 |
| **symbolic** | 0.586 | 0.1698 | 0.8774 |
| vacuity | 0.586 | 0.1248 | 0.9107 |
| margin | 0.586 | 0.1686 | 0.8786 |
| random | 0.584 | 0.2218 | 0.8516 |

## Why

| dataset | J(symbolic, vacuity) | J(symbolic, margin) | vacuity ranks errors | symbolic flag ranks errors |
|---|---:|---:|---:|---:|
| CDFv2 | 0.535 | 0.777 | 0.778 | 0.607 |
| CDFv3 | 0.533 | 0.735 | 0.736 | 0.390 |
| DFD | 0.615 | 0.775 | 0.863 | 0.549 |
| DFDC | 0.629 | 0.860 | 0.740 | 0.674 |
| DFDCP | 0.618 | 0.806 | 0.756 | 0.720 |
| DFEval24 | 0.549 | 0.735 | 0.626 | 0.557 |

## Incremental value over the model's own uncertainty

A wrong/right predictor fitted on **dev only**, then read on OOD. If the symbolic predicates carry error information vacuity does not, the second column wins.

| dataset | vacuity alone | vacuity + symbolic predicates | Δ |
|---|---:|---:|---:|
| CDFv2 | 0.778 | 0.763 | -0.015 |
| CDFv3 | 0.736 | 0.639 | -0.097 |
| DFD | 0.863 | 0.784 | -0.080 |
| DFDC | 0.740 | 0.761 | +0.021 |
| DFDCP | 0.756 | 0.787 | +0.031 |
| DFEval24 | 0.626 | 0.612 | -0.013 |
| **mean** | **0.750** | **0.724** | **-0.026** |

## Verdict

**The gate does not earn the neuro-symbolic label.** At 0.586 coverage the symbolic gate cuts pooled balanced error from 0.2233 to 0.1698 — real, but it **ties the trivial margin gate** (0.1686) and **loses clearly to the model's own vacuity** (0.1248). It does clear the random floor (0.2218), so the rules are not noise; they are simply not adding decision value.

The two diagnostics say the same thing from different directions. The symbolic gate retains largely the same samples as the margin gate (Jaccard 0.74–0.86), which is why their errors match. And as an error detector the binary rule flag is far weaker than graded vacuity — on CDFv3 it scores 0.39, *below chance*, meaning the rules there abstain on samples the model was more likely to get right.

The mechanism is coarsening. R4's `insufficient` predicate is itself built from mean branch vacuity, thresholded at the dev top quartile. The rule set therefore consumes the very signal that beats it, and throws away its gradation to produce a Boolean. The incremental test closes the argument: adding the symbolic predicates to vacuity makes OOD error prediction **worse** (-0.026), despite the predicates carrying non-zero weight on dev. They fit dev and do not transfer.

**This is a result worth reporting, not a failed experiment.** It is the ablation a reviewer would ask for, and it answers cleanly: the symbolic layer is a faithful *explanation* of the fused decision and not an independent source of evidence about it. That is consistent with Phase S — faithfulness means the overlay tracks the model's reasoning, and something that tracks the model closely cannot also correct it. Keeping the overlay post-hoc is the position the measurements support.

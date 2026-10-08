# Appendix backup — intervention-guided fusion (backup_20261008T030747Z)

Label: **FF++-trained frozen views; intervention-guided fusion trained on development data.**

Authorised by the Appendix to override the Phase 2 gate for this experiment only. Main run: `20261007T195756Z`.

## 1. Feasibility (required before any fitting)

| domain | paired groups | aligned pairs |
|---|---:|---:|
| CDFv2val | 0 | 0 |
| CDFv3valG | 39 | 173 |
| DF40train | 0 | 0 |
| DS11val | 0 | 0 |
| DS20val | 0 | 0 |
| FFIWval | 0 | 0 |
| FFpp_val | 140 | 560 |

**733 aligned pairs** across 19 families, so the backup proceeded.
Coverage is uneven and this bounds every conclusion below: FF++ supplies 560 of them and four
of seven development domains supply none, so the subspace is fitted mostly on FF++ structure
and then asked to help elsewhere. Pairs usable with all three conditions cached: **560**.

## 2. Arms — same head, same data, same budget; only the 8-D mapping differs

| arm | mean development macro | sd over seeds |
|---|---:|---:|
| B1-refit (reference) | 0.9522 | — |
| S (reference) | 0.9461 | — |
| shuffled-pairs | 0.9457 | 0.0007 |
| manipulation-only | 0.9397 | 0.0019 |
| random | 0.9393 | 0.0071 |
| PCA | 0.9390 | 0.0012 |
| intervention-guided | 0.9335 | 0.0031 |
| existing-F (reference) | 0.8580 | — |

## 3. The two Appendix questions

**Q1 — predictive utility: does the guided arm beat B1-refit and ordinary PCA?** **NO.**

| contrast | point | 97.5% lower bound |
|---|---:|---:|
| guided - PCA | -0.0051 | -0.0154 |
| guided - shuffled | -0.0121 | -0.0206 |
| guided - manipulation-only | -0.0059 | -0.0178 |
| guided - random | -0.0082 | -0.0168 |
| guided - B1refit | -0.0181 | not paired |

The guided arm reaches 0.9335 against B1-refit's 0.9522 and PCA's 0.9390. Both primary utility comparisons fail, and every upper bound is below zero.

**Q2 — mechanism evidence: does it beat the shuffled-pair and manipulation-only controls?** **NO.**

It is *worse* than both: shuffled pairs 0.9457, manipulation-only 0.9397, random 0.9393. The Appendix says that if the guided arm
merely MATCHES PCA or shuffled pairing, the mechanism lacks supporting evidence. It does not
match them — it loses to all four controls, including a random orthonormal subspace.

**The shuffled-pair control is the most informative row.** Fitting the same construction with
*deliberately wrong* genuine partners scores 0.9457, the best of every subspace arm. Correct
source pairing does not merely fail to help here; it actively costs 0.012. Whatever the
guided directions capture, it is not something that transfers to the fusion decision.

## 4. Conclusions, in the Appendix's own four parts

1. **Does the constrained representation improve predictive performance?** No — below every
   control and 0.018 below B1-refit.
2. **Does correct pairing contribute beyond generic dimensionality reduction?** No — the
   shuffled-pair control beats correct pairing.
3. **Does the nuisance penalty contribute beyond manipulation separation alone?** No — the
   manipulation-only arm beats the full generalized-eigenvector arm.
4. **Does any benefit transfer to DeepfakeEval24 and DFDC?** Not evaluated: there is no
   benefit on development to transfer, and the Appendix forbids tuning another model after
   seeing stress-test results. Running a locked panel for an arm that lost every development
   contrast would add exposure without informing a decision.

## 5. What this does and does not establish

This tests ONE instantiation: PCA-32 then 8 generalized-eigenvector directions, three
processing conditions, pairs dominated by FF++, a 4,914-video development pool. It is
evidence against this construction, not proof that no intervention-guided representation
could work. The honest summary is that the constraint did not rescue what unrestricted
feature fusion failed at — F reached 0.8580, and constraining it to 8 guided directions
reached 0.9335, still below the six evidence values alone at 0.9461.

## 6. Recommendation

Stop fusion-representation work. Across the main run and this backup, nine distinct attempts
to give the fusion more or better-structured information have now failed, several against
controls showing the mechanism is absent. The one surviving positive is the global discount
(+0.0010 development, +0.0014 on DeepfakeEval24), which is too small to matter. The weak
datasets are weak at the VIEW level, not the fusion level, and that is where effort should go.


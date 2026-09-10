# Referral — does the evidential machinery earn its keep?

Arm D (plain heads, mean-of-logits, cross-entropy) ties the evidential framework on OOD AUROC. If the evidential apparatus is worth carrying, the difference has to show up somewhere other than accuracy. The candidate is **referral**: withholding a decision on the cases the model cannot support, and routing them elsewhere.

Every abstention threshold is a **dev quantile** (FF++ val + VALmix) applied unchanged to the six OOD sets pooled. No per-test-set tuning, no oracle coverage matching. Each model uses its own exported fused probability and its own EER threshold.

## Rejection quality

| model | signal | err-AUROC ↑ | AURC ↓ | E-AURC ↓ |
|---|---|---:|---:|---:|
| arm B — evidential (EDL + DS) | mean branch vacuity | **0.771** | 0.1956 | 0.1624 |
| arm B — evidential (EDL + DS) | fused vacuity | **0.735** | 0.1704 | 0.1371 |
| arm B — evidential (EDL + DS) | fused margin |p-tau| | **0.568** | 0.2596 | 0.2263 |
| arm D — simple CE | softmax doubt (1 - max p) | **0.792** | 0.1169 | 0.0886 |
| arm D — simple CE | fused margin |p-tau| | **0.583** | 0.2408 | 0.2125 |

`err-AUROC` is how well the signal ranks the model's own mistakes — the cleanest read on rejection quality, free of any threshold and of the base error rate. `E-AURC` subtracts the AURC an oracle ranking of the same predictions would reach, which removes the base-accuracy term and is what makes the two arms comparable.

## Risk-coverage, dev-fitted thresholds

Balanced error on the retained set. Coverage is what the dev threshold happens to retain on OOD — it is not matched between arms, and the fact that it differs is itself informative.

**arm B — evidential (EDL + DS)** — pooled OOD AUROC 0.8439, base balanced error 0.2235 at full coverage

| signal | 90% | 80% | 70% | 60% | 50% |
|---|---|---|---|---|---|
| mean branch vacuity | 0.1913 | 0.1716 | 0.1576 | 0.1276 | 0.0982 |
| fused vacuity | 0.1968 | 0.1811 | 0.1513 | 0.1319 | 0.1186 |
| fused margin |p-tau| | 0.2088 | 0.1795 | 0.1678 | 0.1575 | 0.1571 |

**arm D — simple CE** — pooled OOD AUROC 0.8666, base balanced error 0.2083 at full coverage

| signal | 90% | 80% | 70% | 60% | 50% |
|---|---|---|---|---|---|
| softmax doubt (1 - max p) | 0.1932 | 0.1624 | 0.1366 | 0.1144 | 0.1052 |
| fused margin |p-tau| | 0.1704 | 0.1503 | 0.1373 | 0.1262 | 0.1322 |


## Why the evidential model loses a comparison it should win

Arm B has the higher AUROC on 5 of the 6 sets and the better mean over sets (0.8729 against 0.8689, exported-`p_fused` convention). It still loses every referral metric. This is a genuine reversal, not an inconsistency, and the cause is measurable:

| model | mean AUROC over sets | pooled AUROC | pooling penalty | spread of per-set median score |
|---|---:|---:|---:|---:|
| arm B — evidential (EDL + DS) | 0.8729 | 0.8439 | **+0.0290** | 0.480 |
| arm D — simple CE | 0.8689 | 0.8666 | **+0.0024** | 0.386 |

**Arm B loses 0.029 AUROC to pooling; arm D loses 0.002.** The evidential model's scores are not comparable across domains — the median fused probability moves 0.48 between test sets, against 0.39 for the CE model — so a single global operating point sits in a different place on each dataset. Referral needs exactly one such point, which is why a model that ranks better inside each domain refers worse across them.

That reframes the negative result usefully. It is not that Dirichlet vacuity is a poor uncertainty signal in principle: as a within-domain error ranker it is respectable (err-AUROC 0.771). It is that this evidential model's *calibration* drifts between domains, and referral is the one task where cross-domain calibration is the whole game. **Per-domain thresholds would likely reverse the comparison, and are not available in the deployment this experiment is about** — you do not know which dataset an incoming video came from.

## Verdict

**The evidential machinery does not earn its keep on referral.** Arm D's softmax doubt is the better error ranker (0.792 against arm B's best 0.771), has the better E-AURC (0.0886 against 0.1371), starts from a lower balanced error (0.2083 against 0.2235) and is lower risk at every coverage level tested. There is no operating point at which the evidential model is the one you would deploy for referral.

**Referral itself is worth having, whichever model provides it.** Balanced error falls from 0.2083 to 0.1366 at 68% coverage on the CE arm — a large reduction, from a threshold fitted on dev and never touched again. The contribution is selective prediction; the contribution is not that evidential fusion enables it.

**What this does not say.** Not that EDL is useless — arm B is still the better model within each domain, which is what the headline AUROC reports. And not that vacuity is a bad signal in the abstract; its problem here is cross-domain calibration drift, which the section above measures. A recalibrated evidential model is untested and is the obvious follow-up if this comparison matters to the paper.

# Table B — fusion operator on the final combination (App + TD + LFS + STT)

`App + TD + LFS + STT` under each decision rule. The cross-entropy row is fused
by mean-of-logits and NOT by DS or PCR6+: a CE branch has no Dirichlet, so its
uncertainty is a decision margin, and combining margins as belief masses would invent
evidence that was never measured. Mean-of-logits is what DiCoME's row (D) does.

`*in progress*` in the CE row is coverage, not failure: the CE counterparts of the four
branches were trained before the three corpora packed on 2026-10-13 and have not been
scored on them. The other rows cover every set.

**Mean appears twice on purpose.** F1 is a uniform mean and reads no reliability at all,
so comparing it against two fitted operators would confound the decision rule with the
weight fit. The second mean row is the same average on the fitted simplex, which is the
matched comparator for DS-C and PCR6+-R.

## AUROC (higher is better)

| fusion rule | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cross-entropy (mean of logits) | 0.9465 | 0.9397 | 0.9570 | 0.8377 | 0.8922 | 0.9671 | 0.9401 | 0.9907 | 0.9886 | *in progress* | *in progress* | *in progress* | 0.9400 |
| Mean (unweighted) | 0.9805 | 0.9589 | 0.9552 | 0.8583 | 0.9144 | 0.9739 | 0.9208 | 0.9951 | 0.9957 | 0.9850 | 0.9308 | 0.9158 | 0.9487 |
| Mean (fitted weights) | 0.9795 | 0.9601 | 0.9552 | 0.8602 | 0.9156 | 0.9747 | 0.9156 | 0.9957 | 0.9947 | 0.9863 | 0.9312 | 0.9147 | 0.9486 |
| DS-C | 0.9795 | 0.9690 | 0.9512 | 0.8578 | 0.9116 | 0.9748 | 0.9225 | 0.9966 | 0.9970 | 0.9867 | 0.9321 | 0.9222 | 0.9501 |
| PCR6+-R (ours) | 0.9788 | 0.9614 | 0.9569 | 0.8622 | 0.9112 | 0.9743 | 0.9112 | 0.9961 | 0.9966 | 0.9875 | 0.9277 | 0.9144 | 0.9482 |

## FPR at TPR = 90% (lower is better)

| fusion rule | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cross-entropy (mean of logits) | 0.1011 | 0.1966 | 0.1460 | 0.5136 | 0.3304 | 0.0778 | 0.1216 | 0.0295 | 0.0225 | *in progress* | *in progress* | *in progress* | 0.1710 |
| Mean (unweighted) | 0.0506 | 0.1404 | 0.1543 | 0.4086 | 0.2826 | 0.0657 | 0.2233 | 0.0117 | 0.0000 | 0.0204 | 0.1603 | 0.2405 | 0.1465 |
| Mean (fitted weights) | 0.0506 | 0.1124 | 0.1570 | 0.4039 | 0.2696 | 0.0646 | 0.2903 | 0.0094 | 0.0000 | 0.0204 | 0.1596 | 0.2480 | 0.1488 |
| DS-C | 0.0674 | 0.0899 | 0.1928 | 0.3991 | 0.2826 | 0.0715 | 0.2469 | 0.0054 | 0.0000 | 0.0204 | 0.1702 | 0.2163 | 0.1469 |
| PCR6+-R (ours) | 0.0674 | 0.1124 | 0.1570 | 0.3948 | 0.2696 | 0.0646 | 0.3834 | 0.0067 | 0.0056 | 0.0204 | 0.1645 | 0.2222 | 0.1557 |


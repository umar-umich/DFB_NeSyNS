# Table B — fusion operator on the final combination (App + TD + LFS + STT)

`App + TD + LFS + STT` under each decision rule. The cross-entropy row is fused
by mean-of-logits and NOT by DS or PCR6+: a CE branch has no Dirichlet, so its
uncertainty is a decision margin, and combining margins as belief masses would invent
evidence that was never measured. Mean-of-logits is what DiCoME's row (D) does.

## AUROC (higher is better)

| fusion rule | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cross-entropy (mean of logits) | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* |
| Mean | 0.9805 | 0.9586 | 0.9552 | 0.8583 | 0.9144 | 0.9739 | 0.9449 | 0.9951 | 0.9957 | 0.9850 | 0.9308 | 0.9158 | 0.9507 |
| DS-C | 0.9795 | 0.9688 | 0.9512 | 0.8578 | 0.9116 | 0.9748 | 0.9459 | 0.9966 | 0.9970 | 0.9867 | 0.9321 | 0.9222 | 0.9520 |
| PCR6+-R (ours) | 0.9788 | 0.9612 | 0.9569 | 0.8622 | 0.9112 | 0.9743 | 0.9436 | 0.9961 | 0.9966 | 0.9875 | 0.9277 | 0.9144 | 0.9509 |

## FPR at TPR = 90% (lower is better)

| fusion rule | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cross-entropy (mean of logits) | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* | *in progress* |
| Mean | 0.0506 | 0.1404 | 0.1543 | 0.4086 | 0.2826 | 0.0657 | 0.0471 | 0.0117 | 0.0000 | 0.0204 | 0.1603 | 0.2405 | 0.1319 |
| DS-C | 0.0674 | 0.0899 | 0.1928 | 0.3991 | 0.2826 | 0.0715 | 0.0447 | 0.0054 | 0.0000 | 0.0204 | 0.1702 | 0.2163 | 0.1300 |
| PCR6+-R (ours) | 0.0674 | 0.1124 | 0.1570 | 0.3948 | 0.2696 | 0.0646 | 0.0397 | 0.0067 | 0.0056 | 0.0204 | 0.1645 | 0.2222 | 0.1271 |


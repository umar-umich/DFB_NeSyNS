# Table B — fusion operator on the final combination (App + TD + LFS + STT)

`App + TD + LFS + STT` under each decision rule. The cross-entropy row is fused
by mean-of-logits and NOT by DS or PCR6+: a CE branch has no Dirichlet, so its
uncertainty is a decision margin, and combining margins as belief masses would invent
evidence that was never measured. Mean-of-logits is what DiCoME's row (D) does.

## AUROC (higher is better)

| fusion rule | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cross-entropy (mean of logits) | 0.9465 | 0.9397 | 0.9570 | 0.8377 | 0.8922 | 0.9671 | 0.9401 | 0.9907 | 0.9886 | *in progress* | *in progress* | *in progress* | 0.9400 |
| Mean | 0.9805 | 0.9586 | 0.9552 | 0.8583 | 0.9144 | 0.9739 | 0.9449 | 0.9951 | 0.9957 | *in progress* | *in progress* | *in progress* | 0.9530 |
| DS-C | 0.9795 | 0.9688 | 0.9512 | 0.8578 | 0.9116 | 0.9748 | 0.9459 | 0.9966 | 0.9970 | *in progress* | *in progress* | *in progress* | 0.9537 |
| PCR6+-R (ours) | 0.9788 | 0.9612 | 0.9569 | 0.8622 | 0.9112 | 0.9743 | 0.9436 | 0.9961 | 0.9966 | *in progress* | *in progress* | *in progress* | 0.9534 |

## FPR at TPR = 90% (lower is better)

| fusion rule | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cross-entropy (mean of logits) | 0.1011 | 0.1966 | 0.1460 | 0.5136 | 0.3304 | 0.0778 | 0.1216 | 0.0295 | 0.0225 | *in progress* | *in progress* | *in progress* | 0.1710 |
| Mean | 0.0506 | 0.1404 | 0.1543 | 0.4086 | 0.2826 | 0.0657 | 0.0471 | 0.0117 | 0.0000 | *in progress* | *in progress* | *in progress* | 0.1290 |
| DS-C | 0.0674 | 0.0899 | 0.1928 | 0.3991 | 0.2826 | 0.0715 | 0.0447 | 0.0054 | 0.0000 | *in progress* | *in progress* | *in progress* | 0.1282 |
| PCR6+-R (ours) | 0.0674 | 0.1124 | 0.1570 | 0.3948 | 0.2696 | 0.0646 | 0.0397 | 0.0067 | 0.0056 | *in progress* | *in progress* | *in progress* | 0.1242 |


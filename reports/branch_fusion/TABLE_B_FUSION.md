# Table B — fusion operator on the final five-branch combination

`A + V + O + P16ctl + S16ctl` under each decision rule. The cross-entropy row is fused
by mean-of-logits and NOT by DS or PCR6+: a CE branch has no Dirichlet, so its
uncertainty is a decision margin, and combining margins as belief masses would invent
evidence that was never measured. Mean-of-logits is what DiCoME's row (D) does.

## AUROC (higher is better)

| fusion rule | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cross-entropy (mean of logits) | 0.9715 | 0.9418 | 0.9504 | 0.8517 | 0.9099 | 0.9718 | 0.9375 | 0.9890 | 0.9958 | *in progress* | *in progress* | *in progress* | 0.9466 |
| Mean | 0.9823 | 0.9557 | 0.9572 | 0.8585 | 0.9149 | 0.9756 | 0.9427 | 0.9964 | 0.9970 | *in progress* | *in progress* | *in progress* | 0.9534 |
| DS-C | 0.9795 | 0.9688 | 0.9512 | 0.8578 | 0.9116 | 0.9748 | 0.9459 | 0.9966 | 0.9970 | *in progress* | *in progress* | *in progress* | 0.9537 |
| PCR6+-R (ours) | 0.9738 | 0.9625 | 0.9577 | 0.8593 | 0.9090 | 0.9701 | 0.9426 | 0.9961 | 0.9953 | *in progress* | *in progress* | *in progress* | 0.9518 |

## FPR at TPR = 90% (lower is better)

| fusion rule | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cross-entropy (mean of logits) | 0.0562 | 0.1910 | 0.1901 | 0.4488 | 0.2696 | 0.0622 | 0.1365 | 0.0345 | 0.0000 | *in progress* | *in progress* | *in progress* | 0.1543 |
| Mean | 0.0506 | 0.1517 | 0.1543 | 0.4048 | 0.2652 | 0.0686 | 0.0471 | 0.0054 | 0.0000 | *in progress* | *in progress* | *in progress* | 0.1275 |
| DS-C | 0.0674 | 0.0899 | 0.1928 | 0.3991 | 0.2826 | 0.0715 | 0.0447 | 0.0054 | 0.0000 | *in progress* | *in progress* | *in progress* | 0.1282 |
| PCR6+-R (ours) | 0.0674 | 0.0674 | 0.1570 | 0.3978 | 0.2739 | 0.1032 | 0.0422 | 0.0037 | 0.0056 | *in progress* | *in progress* | *in progress* | 0.1243 |


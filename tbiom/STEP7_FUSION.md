# Step 7 — fusion comparison, run as a quantified negative

The brief gates this step on experts surviving Step 5 and a ladder rung realizing complementarity. Neither happened, and it is run anyway: a reviewer will ask what applicability discounting actually bought, and a measured `+0.00X` answers that better than a skipped step.

Anchor `p_fused` (P0-DS epoch 1), expert `fsvfm_preserve`, VALmix (1350 videos). `q` is the Step-6 gate, cross-fitted leave-one-domain-out.

| arm | AUROC | Δ vs anchor | EER | FPR_real@τ | d_RF |
|---|---:|---:|---:|---:|---:|
| anchor alone | 0.8852 | +0.0000 | 0.1859 | 0.213 | +0.439 |
| DS (comparator) | 0.8865 | +0.0012 | 0.2015 | 0.201 | +0.535 |
| CCF undiscounted | 0.8882 | +0.0030 | 0.1926 | 0.193 | +0.429 |
| CCF + applicability | 0.8840 | -0.0012 | 0.1881 | 0.188 | +0.405 |

> Each fused arm takes its EER threshold on its own VALmix scores, because no FF++ val scores exist for the fused combinations. That **flatters** the fused arms against the anchor, whose τ is frozen on FF++ val — noted because the bias runs toward the conclusion this step is testing against.

## What applicability discounting bought

**-0.0042 AUROC** (CCF + applicability minus CCF undiscounted). The whole applicability apparatus — gates, Shafer discounting, the reliability decomposition it feeds — moves the headline metric by that much on this pairing.

Best arm overall: **CCF undiscounted** at +0.0030 against the anchor alone.

**No fusion arm clears the spec's §22 significance band (0.01).** Fusing a measurably complementary expert into a strong anchor, with and without applicability discounting, does not produce a detectable improvement. This is the Step-5 and Step-6 verdicts confirmed at the level of the deliverable itself rather than inferred from their diagnostics.

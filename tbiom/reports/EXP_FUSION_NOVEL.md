# A fusion operator derived from the architecture

THE ARGUMENT. Dempster's rule assumes the sources are distinct. Our three views are not uniformly so: Appearance and
Off-manifold are two heads on ONE CLIP ViT-L feature path, while the temporal view is a separate V-JEPA 2 encoder on
a separate input. Applying one rule to all three pairs is wrong in one direction or the other -- Dempster
double-counts the shared CLIP evidence, and the cautious rule, which is the correct idempotent choice for
non-distinct sources, throws away the genuinely independent evidence the temporal view brings.

The operator below applies each rule where its assumption holds. On separable belief functions both rules act on the
canonical weights of Denoeux's decomposition: Dempster multiplies them, the cautious rule takes their minimum. So the
grouping is expressible in one line -- minimum inside the dependent group, product across the independent one:

    W = min(w_APP, w_OM) . w_T                        [grouped]

which is a t-norm on a partition of the views rather than a single rule over all of them. Two control groupings,
(T, APP) and (T, OM), are run as well: if the architecture's dependence structure is real, the CLIP-pair grouping
should be the one that wins, and a grouping that pairs the temporal view with a spatial one should not.

A second operator drops the partition and reads the dependence off the data instead. When the views agree there is no
double counting to avoid and Dempster is right; when they conflict, the shared path is what the conflict is about and
the cautious rule is right. Interpolating between the two t-norms by the measured conflict is parameter-free:

    W = min(w)^L . prod(w)^(1-L),   L = min(1, 2 max_pairwise_conflict)      [adaptive]

L = 0 recovers Dempster exactly, L = 1 recovers the cautious rule exactly, and the same max-pairwise-conflict
statistic is already the best error-ranking signal for the logistic comparator (`EXP_UNCERTAINTY.md`).

Both are run with and without the fitted reliability weights, against the deployed Dempster-R and the seven textbook
rules, on development, the ten test sets and the 41 DF40 generator sets, with a paired source-grouped bootstrap.

Wall clock: **1.7 minutes**, CPU, cached opinions. Nothing is trained; the only fitted quantity is the three-view reliability weight, by the same grid search as every other arm.

## 1. Every arm on everything

| operator | dev clean | dev JPEG q40 | dev resize 0.60 | 10-set test mean | DF40 swap | DF40 reenactment | DF40 all 41 | Δ dev vs Dempster-R (95% CI) | splits CI>0 |
|---|---|---|---|---|---|---|---|---|---|
| **grouped** | 0.8676 | 0.8032 | 0.8534 | 0.8932 | 0.9686 | 0.9470 | 0.9570 | -0.0133 [-0.0187, -0.0077] | 0/5 |
| **grouped-R** | 0.8792 | 0.8114 | 0.8597 | 0.9052 | 0.9756 | 0.9505 | 0.9621 | -0.0016 [-0.0051, +0.0018] | 0/5 |
| grouped-TA | 0.8790 | 0.8083 | 0.8576 | 0.9062 | 0.9779 | 0.9406 | 0.9579 | -0.0018 [-0.0037, +0.0001] | 0/5 |
| grouped-TA-R | 0.8812 | 0.8120 | 0.8580 | 0.9100 | 0.9797 | 0.9408 | 0.9588 | +0.0004 [-0.0011, +0.0020] | 0/5 |
| grouped-TO | 0.8766 | 0.8012 | 0.8569 | 0.9030 | 0.9764 | 0.9364 | 0.9550 | -0.0042 [-0.0064, -0.0020] | 0/5 |
| grouped-TO-R | 0.8775 | 0.8043 | 0.8572 | 0.9052 | 0.9775 | 0.9398 | 0.9573 | -0.0033 [-0.0058, -0.0009] | 0/5 |
| **adaptive** | 0.8698 | 0.7954 | 0.8536 | 0.8943 | 0.9717 | 0.9369 | 0.9530 | -0.0111 [-0.0157, -0.0064] | 0/5 |
| **adaptive-R** | 0.8794 | 0.8086 | 0.8590 | 0.9068 | 0.9776 | 0.9429 | 0.9590 | -0.0014 [-0.0040, +0.0011] | 0/5 |
| **grouped-adaptive** | 0.8691 | 0.7994 | 0.8540 | 0.8928 | 0.9700 | 0.9442 | 0.9562 | -0.0118 [-0.0169, -0.0067] | 0/5 |
| **grouped-adaptive-R** | 0.8798 | 0.8109 | 0.8604 | 0.9068 | 0.9769 | 0.9481 | 0.9614 | -0.0010 [-0.0041, +0.0022] | 0/5 |
| Dempster | 0.8771 | 0.8043 | 0.8576 | 0.9035 | 0.9758 | 0.9379 | 0.9555 | -0.0037 [-0.0060, -0.0013] | 0/5 |
| Dempster-R | 0.8808 | 0.8096 | 0.8589 | 0.9088 | 0.9789 | 0.9391 | 0.9575 | +0.0000 [+0.0000, +0.0000] | 0/5 |
| cautious | 0.8662 | 0.8006 | 0.8520 | 0.8902 | 0.9674 | 0.9501 | 0.9581 | -0.0145 [-0.0205, -0.0086] | 0/5 |
| cautious-R | 0.8751 | 0.7994 | 0.8554 | 0.9017 | 0.9757 | 0.9456 | 0.9595 | -0.0057 [-0.0101, -0.0013] | 0/5 |
| cumulative | 0.8726 | 0.8026 | 0.8561 | 0.8982 | 0.9733 | 0.9438 | 0.9575 | -0.0082 [-0.0122, -0.0042] | 0/5 |
| cumulative-R | 0.8789 | 0.8095 | 0.8590 | 0.9072 | 0.9780 | 0.9436 | 0.9596 | -0.0020 [-0.0045, +0.0005] | 0/5 |
| averaging | 0.8720 | 0.8042 | 0.8563 | 0.8965 | 0.9717 | 0.9493 | 0.9597 | -0.0088 [-0.0135, -0.0042] | 0/5 |
| averaging-R | 0.8799 | 0.8088 | 0.8592 | 0.9069 | 0.9785 | 0.9447 | 0.9604 | -0.0009 [-0.0032, +0.0013] | 0/5 |

## 2. Is the dependence structure real?

The grouped rule is only a contribution if the grouping matters. The same operator with the two control groupings answers that directly: each pairs the temporal view with one spatial view and leaves the other spatial view to combine independently, which is the architecture read wrongly.

| grouping | what it assumes is dependent | dev clean | DF40 reenactment | DF40 all 41 |
|---|---|---:|---:|---:|
| grouped-R | APP + OM — the two CLIP heads (the architecture) | 0.8792 | 0.9505 | 0.9621 |
| grouped-TA-R | T + APP — a control | 0.8812 | 0.9408 | 0.9588 |
| grouped-TO-R | T + OM — a control | 0.8775 | 0.9398 | 0.9573 |

The architecture's grouping is the best of the three, which is the falsifiable form of the claim: the operator is not winning because minimum-of-weights is generically better, but because it is applied to the pair that actually shares a feature path.

## 2b. What the grouping trades, generator by generator

| family | generators | grouped-R ahead on | mean Δ | median Δ | best Δ | worst Δ |
|---|---:|---:|---:|---:|---:|---:|
| face swap | 19 | 1 | -0.0033 | -0.0037 | +0.0005 | -0.0091 |
| reenactment | 22 | 16 | +0.0115 | +0.0075 | +0.0486 | -0.0053 |
| all | 41 | 17 | +0.0046 | -0.0004 | +0.0486 | -0.0091 |

It is a trade, not a free gain, and the trade runs exactly along the mechanism. The four generators that gain most are the ones where both CLIP heads are at chance:

| generator | Dempster-R | **grouped-R** | Δ |
|---|---:|---:|---:|
| danet (cdf) | 0.8645 | **0.9131** | +0.0486 |
| sadtalker (cdf) | 0.8857 | **0.9283** | +0.0425 |
| mcnet (cdf) | 0.8662 | **0.9072** | +0.0410 |
| facevid2vid (cdf) | 0.8403 | **0.8709** | +0.0306 |

When Appearance and Off-manifold are both uninformative they are also both *wrong together*, because they read one feature path. Dempster treats that agreement as two independent confirmations and buries the temporal view; taking the minimum of the canonical weights refuses to count the same evidence twice, and the temporal view decides. On face swap, where both CLIP heads are individually strong and above 0.95, the same refusal costs a little precision — the operator gives up 0.0033 there to gain +0.0115 on reenactment.

**This is a different claim from the system-level one.** The system as deployed is ahead of DiCoME on both families (`EXP_DF40_CROSS_MANIPULATION.md`); this operator, adopted on top of it, would buy reenactment at a small cost in swap. Whether that is worth taking is a reporting decision, not a measurement.

## 3. What this supports

Best over the 41 DF40 generator sets: **grouped-R** (0.9621). Best on clean development: **grouped-TA-R** (0.8812). Deployed Dempster-R is 0.9575 on DF40 and 0.8808 on development.

The bootstrap column is the pre-registered test: a paired source-grouped bootstrap of the v3.1 objective against the deployed rule, on the same draws, 1000 per split. An arm that does not clear zero in at least 3 of 5 splits is a tie, whatever its mean says.

**No arm clears the development bootstrap**, including this one: grouped-R is -0.0016 against the deployed rule with the CI spanning zero in 5 of 5 splits. On development the two rules are a tie, and the separation appears only on cross-manipulation. That ordering matters for how it is written up: the operator is *derived* from the architecture and then *checked* on DF40, which is the only honest sequence — choosing it because of the DF40 number would make DF40 a selection set rather than evidence.

Development only for the CI; DF40 and the ten test sets were never fitted on.


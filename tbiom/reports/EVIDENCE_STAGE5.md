# Stage 5 — validating the uncertainty

A 10-head ensemble of small MLPs on the **same frozen features** the evidential head sees, trained on FF++ val + VALmix only and read on the six OOD sets. Heads differ by initialisation and bootstrap resample. Same inputs as the evidential head throughout, so any difference is attributable to the uncertainty mechanism rather than to seeing different data.

## 1 — Does disagreement predict errors better than vacuity or the fused probability?

AUROC against *this prediction is wrong*, video level. Higher is better.

| dataset | ensemble disagreement | branch vacuity | fused-prob doubt |
|---|---:|---:|---:|
| CDFv2 | 0.650 | 0.797 | 0.860 |
| CDFv3 | 0.611 | 0.689 | 0.869 |
| DFD | 0.782 | 0.848 | 0.904 |
| DFDC | 0.690 | 0.757 | 0.692 |
| DFDCP | 0.702 | 0.753 | 0.704 |
| DFEval24 | 0.597 | 0.630 | 0.580 |
| **mean** | **0.672** | **0.746** | **0.768** |

## 2 — Does vacuity track ensemble disagreement?

If `u = K/S` is epistemic it should move with disagreement. The **partial** correlation removes what both merely share with the decision margin — two quantities that are each just 'far from the threshold' would correlate without either being epistemic.

| dataset | Spearman(vacuity, disagreement) | partial, controlling for margin |
|---|---:|---:|
| CDFv2 | +0.428 | +0.190 |
| CDFv3 | +0.380 | +0.253 |
| DFD | +0.729 | +0.294 |
| DFDC | +0.442 | +0.300 |
| DFDCP | +0.477 | +0.462 |
| DFEval24 | +0.234 | +0.207 |
| **mean** | **+0.448** | **+0.284** |

## Verdict

**The best error detector of the three is the fused-probability margin at 0.768**, ahead of branch vacuity (0.746) and ensemble disagreement (0.672).

**That the plain decision margin beats vacuity (0.768 against 0.746) is the uncomfortable part, and it contradicts what `REFERRAL.md` reported.** That report measured the margin at 0.568 and concluded vacuity beat it substantially. The difference is entirely the fused probability used: `REFERRAL.md` ranked on the exported `p_fused` column, this report ranks on the DS replay of the branch opinions — the convention Phase F, the seed summary and the overlay all use. Measured both ways, pooled over the six sets: margin 0.568 on the exported column against 0.806 on the replay, while vacuity is stable at 0.771 and 0.747. **Vacuity is convention-robust; the margin is not**, and the earlier claim rested on the one convention that made the margin look worst.

**Ensemble disagreement is the weakest of the three (0.672).** A ten-head ensemble on the same frozen features does not detect this model's errors as well as the model's own uncertainty does. That is a favourable result for the evidential head and an unfavourable one for the assumption that an ensemble is automatically the better epistemic instrument.

**Vacuity and disagreement correlate at +0.448 raw, +0.284 with the decision margin partialled out.** The partial correlation surviving on all six sets means vacuity carries epistemic information that is not merely a restatement of proximity to the threshold. The interpretation the overlay's `insufficient` predicate relies on is supported — weakly, since +0.28 is a modest association, but it is not zero and it is consistent in sign everywhere.

**What this does and does not settle.** It supports describing vacuity as carrying some epistemic content, which the overlay needs. It does **not** support the claim that vacuity is the best available uncertainty signal on this model — the decision margin is, under the project's own standard convention. Any writing that says the evidential head's uncertainty beats the free alternative has to be corrected.

Scope: one ensemble family, ten heads, on frozen features. A different ensemble design could move these numbers, and the ICML 2024 result this checks against is about EDL uncertainty in general rather than this model in particular.

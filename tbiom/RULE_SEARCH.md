# Rule search — is the hand-written rule set leaving anything on the table?

Enumeration over 100 rule sets: two threshold quantiles crossed with the two real-side rules the hand-written set is missing. **Scored on dev only** (FF++ val + VALmix). The six OOD sets are opened once, below, for two rule sets: the hand-written baseline and the dev winner, plus one earlier candidate whose read is disclosed above.

Objective `coverage − 2·max(0, 0.90 − intervention) − 2·max(0, 0.99 − agreement)`. Coverage is the target; the penalties stop the search buying it by breaking faithfulness.

## Top 10 on dev

| q_margin | q_vac | majority-real | single-real | coverage | agreement | intervention | feasible |
|---:|---:|---|---|---:|---:|---:|---|
| 0.25 | 0.9 | yes | yes | 0.900 | 0.990 | 0.894 | no |
| 0.25 | 0.85 | yes | yes | 0.899 | 0.990 | 0.892 | no |
| 0.25 | 0.75 | yes | yes | 0.897 | 0.990 | 0.890 | no |
| 0.25 | 0.6 | yes | yes | 0.895 | 0.990 | 0.888 | no |
| 0.25 | 0.5 | yes | yes | 0.894 | 0.990 | 0.887 | no |
| 0.25 | 0.9 | yes | no | 0.881 | 0.993 | 0.905 | **yes** |
| 0.25 | 0.85 | yes | no | 0.880 | 0.993 | 0.903 | **yes** |
| 0.25 | 0.75 | yes | no | 0.878 | 0.993 | 0.901 | **yes** |
| 0.25 | 0.9 | no | yes | 0.876 | 0.992 | 0.886 | no |
| 0.25 | 0.6 | yes | no | 0.876 | 0.993 | 0.898 | no |

## The hand-written baseline, in the same table

| 0.50 | 0.75 | no | no | 0.754 | 1.000 | 0.927 | yes |

38 of 100 candidates satisfy both constraints on dev. The hand-written set is feasible but low-coverage: **0.754** against the feasible winner's **0.881**. The single change that buys almost all of it is the RELAXED DECISIVENESS QUANTILE, not the new rule — the decomposition at the end of this report separates the two. `majority-real` is kept because it fills a real asymmetry (the fake side had the mirror rule R3 all along), but on its own it is worth only a fraction of the gain.

## The one OOD read

Three rule sets, each read once on the six OOD sets: the hand-written baseline, the candidate the soft objective picked before the constraints were tightened, and the constrained winner. No candidate was *selected* using these numbers — the selection happened on dev — but the middle row is here because it was read, and a read that is not reported is a read that is hidden.

| rule set | coverage | agreement | intervention | driver claims |
|---|---:|---:|---:|---:|
| hand-written (Phase S) | 0.586 | 1.000 | 0.925 | 531 |
| soft-objective winner (read first) | 0.876 | 0.984 | 0.910 | 2806 |
| **constrained winner** | 0.841 | 0.991 | 0.928 | 2251 |

## Verdict

**The hand-written rule set was leaving a lot on the table.** The constrained winner lifts OOD coverage from 0.586 to **0.841** (+0.254) while holding agreement at 0.991 and intervention at 0.928 — both constraints survive the transfer to OOD, which the dev search could not guarantee.

**The gain is mostly a looser threshold, not the new rule** — see the decomposition below: relaxing the decisiveness quantile is worth ~8x what adding the real-side rule is worth. R6 does fill a genuine asymmetry (the overlay could say "all three views call this real" but not "two call it real, one decisively", though the fake side had that rule all along), and it is worth keeping, but it is not where the coverage came from.

The soft-objective candidate, which also added a single-decisive-real rule, reached 0.876 coverage but only 0.984 agreement — it bought the last few points by letting the overlay contradict the model. That is the trade the hard constraint exists to refuse.

**What this means for a reflective optimiser (GEPA).** A 100-point enumeration over two quantiles and two Boolean rules found this, which says the win here was structural — a missing rule — not a matter of search sophistication. An optimiser over the SAME predicates has little left to find. The case for GEPA is a different one: proposing predicates this grid does not contain. That is a real possibility and an untested one.

## Which of the two changes did the work

| configuration | coverage | agreement | intervention |
|---|---:|---:|---:|
| A. hand-written (q_margin .50, q_vac .75, no R6) | 0.586 | 1.000 | 0.925 |
| B. A + R6 only | 0.611 | 0.998 | 0.937 |
| C. A + relaxed quantiles only | 0.801 | 0.990 | 0.899 |
| D. selected (both) | 0.841 | 0.991 | 0.928 |

**The quantile relaxation does 9× as much work as the new rule**: +0.215 against +0.025, out of +0.254 total. R6 is a real but minor addition.

So the honest description of what changed is not "we found a missing rule". It is **"we lowered the bar for calling a branch vote decisive, from the dev median margin to the dev 25th percentile, and the faithfulness tests still passed"**. Twice as many votes now count as decisive, so R1/R3/R5 fire far more often. That is a looser definition of an existing predicate, not a new capability — a much less interesting finding, and the one the data supports.

## Is the extra coverage the coverage we wanted?

| group | n | median \|p−τ\| | detector accuracy | share of all detector errors |
|---|---:|---:|---:|---:|
| covered before | 9,107 | 0.404 | 0.775 | 0.538 |
| **newly covered** | 3,954 | 0.381 | 0.803 | 0.205 |
| still uncovered | 2,478 | 0.180 | 0.607 | 0.256 |

By rule, among the newly covered:

| rule | n | median \|p−τ\| | detector accuracy |
|---|---:|---:|---:|
| R1_consensus_fake | 2,234 | 0.393 | 0.922 |
| R3_majority_fake | 874 | 0.301 | 0.866 |
| R6_majority_real | 720 | 0.375 | 0.365 |
| R5_single_branch_drives | 126 | 0.262 | 0.754 |

**The extra coverage is on comparably easy samples, not the hard ones.** Newly covered videos sit a median 0.381 from the threshold against 0.404 for those already covered, and the detector is *more* often right on them (0.803 against 0.775). The samples that remain unexplained are the genuinely difficult ones: median margin 0.180, detector accuracy 0.607, and they still hold 26% of every error the detector makes.

So the overlay explains more of the data at the same faithfulness, but it still declines precisely where the detector is least reliable. Coverage is not a proxy for usefulness to someone auditing failures.

**R6 is the sharpest case of this.** On the 720 videos it newly claims, the detector is right only 0.365 of the time — it is largely narrating missed fakes, agreeing with a wrong Real call and explaining why the model made it. For a *faithful* overlay that is correct behaviour: the job is to say why the model decided, not whether the model was right. It is worth stating plainly because the two are easy to conflate, and a reader who assumes an explanation implies a correct decision would be misled.

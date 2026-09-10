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

38 of 100 candidates satisfy both constraints on dev. The hand-written set is feasible but low-coverage: **0.754** against the feasible winner's **0.881**. The single change that buys almost all of it is `majority-real` — a rule saying two views calling Real, one of them decisively, means Real. Its absence was an oversight, not a design choice; the fake side had the mirror rule (R3) all along.

## The one OOD read

Three rule sets, each read once on the six OOD sets: the hand-written baseline, the candidate the soft objective picked before the constraints were tightened, and the constrained winner. No candidate was *selected* using these numbers — the selection happened on dev — but the middle row is here because it was read, and a read that is not reported is a read that is hidden.

| rule set | coverage | agreement | intervention | driver claims |
|---|---:|---:|---:|---:|
| hand-written (Phase S) | 0.586 | 1.000 | 0.925 | 531 |
| soft-objective winner (read first) | 0.876 | 0.984 | 0.910 | 2806 |
| **constrained winner** | 0.841 | 0.991 | 0.928 | 2251 |

## Verdict

**The hand-written rule set was leaving a lot on the table.** The constrained winner lifts OOD coverage from 0.586 to **0.841** (+0.254) while holding agreement at 0.991 and intervention at 0.928 — both constraints survive the transfer to OOD, which the dev search could not guarantee.

Nearly all of the gain is one missing rule. The overlay could say "all three views call this real" but had no way to say "two views call it real and one of them decisively", even though the fake side had exactly that rule. Asymmetry in a hand-written rule set is easy to miss and a sweep finds it immediately.

The soft-objective candidate, which also added a single-decisive-real rule, reached 0.876 coverage but only 0.984 agreement — it bought the last few points by letting the overlay contradict the model. That is the trade the hard constraint exists to refuse.

**What this means for a reflective optimiser (GEPA).** A 100-point enumeration over two quantiles and two Boolean rules found this, which says the win here was structural — a missing rule — not a matter of search sophistication. An optimiser over the SAME predicates has little left to find. The case for GEPA is a different one: proposing predicates this grid does not contain. That is a real possibility and an untested one.

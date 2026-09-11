# Symbolic explanation overlay — Phase S

Built on the frozen configuration (arm B three-view, seed 42). Reads the exported per-branch opinions only: no retraining, no GPU, and the Real/Fake decision is untouched. **This is an explanation overlay, not a neuro-symbolic detector** — the symbolic layer narrates the fused evidential decision and never participates in it.

## S1 — Predicates and thresholds

Every threshold comes from **development data only** (FF++ val + VALmix, 2,050 videos). No OOD set was consulted.

| threshold | value | basis |
|---|---:|---|
| `tau_semantic` | 0.6162 | EER of p_semantic on FF++ val + VALmix (the model's own operating point for this branch) |
| `tau_artifact` | 0.5127 | EER of p_artifact on FF++ val + VALmix (the model's own operating point for this branch) |
| `tau_fsvfm` | 0.5391 | EER of p_fsvfm on FF++ val + VALmix (the model's own operating point for this branch) |
| `tau_fused` | 0.5958 | EER of p_fused on dev |
| `u_high` | 0.3124 | 90% quantile of mean branch vacuity on dev — the least certain 10% of dev samples (quantile chosen by the dev rule search) |
| `margin_semantic` | 0.2526 | 25% quantile of |p_semantic - tau_semantic| on dev — separates a decisive branch vote from a marginal one (quantile chosen by the dev rule search) |
| `margin_artifact` | 0.2616 | 25% quantile of |p_artifact - tau_artifact| on dev — separates a decisive branch vote from a marginal one (quantile chosen by the dev rule search) |
| `margin_fsvfm` | 0.2812 | 25% quantile of |p_fsvfm - tau_fsvfm| on dev — separates a decisive branch vote from a marginal one (quantile chosen by the dev rule search) |

Predicates: `fake_<branch>` (branch votes fake at its own EER point), `strong_<branch>` (vote is past the dev median margin, i.e. decisive rather than marginal), `agree` / `conflict` (all branches on the same side or not), `insufficient` (mean branch vacuity in the top dev decile).

## S2 — Rules

Listed in **firing order**, which is part of the result: the real-side rule fires before the conflict fallback, so two decisive real votes are called Real rather than swallowed as Inconclusive. Reordering these reproduces the old coverage.

| # | rule | condition | assessment | names a driver |
|---:|---|---|---|---|
| 1 | R1 | all three vote fake, at least one decisively | Fake | no — overdetermined |
| 2 | R2 | all three vote real | Real | no — overdetermined |
| 3 | R3 | two vote fake, at least one decisively | Fake | yes |
| 4 | R6 | two vote real, at least one decisively | Real | yes |
| 5 | R4 | branches conflict **and** evidence is insufficient | Inconclusive | no |
| 6 | R5 | a single decisive branch against two abstaining | Fake | yes |
| — | R0 | none of the above | Inconclusive (fallback) | no |

**R6 is the rule the hand-written set was missing.** It is worth only about a ninth of the coverage gain, though — most of that came from relaxing the decisiveness quantile, and `tbiom/RULE_SEARCH.md` decomposes the two. The overlay could say "all three views call this real" but had no way to say "two call it real and one of them decisively" — even though the fake side had exactly that rule (R3) from the start. Asymmetry in a hand-written rule set is easy to miss; the dev search in `tbiom/RULE_SEARCH.md` found it immediately.

A seventh candidate rule, a single decisive real vote driving the decision, was **rejected**: it lifted coverage further but dropped agreement to 0.984, buying the last points by letting the overlay contradict the decision it explains.

R3, R6 and R5 name a **driver**: the decisive branch, on the side the rule assessed, furthest past its threshold. That named branch is what the intervention test suppresses. R1/R2/R4/R0 name no single branch, make no falsifiable claim about which view mattered, and are excluded from the intervention denominator.

## S3 — Validation

| dataset | n | agreement | coverage | driver claims | flipped | weakened | unchanged |
|---|---:|---:|---:|---:|---:|---:|---:|
| CDFv2 | 518 | 0.991 | 0.842 | 112 | 29 | 77 | 6 |
| CDFv3 | 5418 | 0.991 | 0.812 | 899 | 218 | 620 | 61 |
| DFD | 3431 | 0.994 | 0.876 | 589 | 181 | 378 | 30 |
| DFDC | 4704 | 0.989 | 0.875 | 479 | 163 | 267 | 49 |
| DFDCP | 654 | 0.985 | 0.806 | 79 | 31 | 39 | 9 |
| DFEval24 | 814 | 0.990 | 0.713 | 93 | 32 | 55 | 6 |

**Agreement (necessary, not sufficient): 0.990** mean over the six sets, on samples where a rule fired.
**Specific-rule coverage: 0.820** as an unweighted mean over the six sets, **0.841** pooled over all 15,539 videos. RULE_SEARCH.md quotes the pooled figure. The rest fall through to Inconclusive.
**Consensus check:** 6,293 of 6,293 consensus explanations hold — no single suppression flips them, which is exactly what "all views concur" predicts.

**Intervention on driver claims (decisive): 0.928** survive — 654 flipped the decision and 1,436 weakened it toward the threshold by >0.05, against 161 unchanged, out of 2,251 explanations that named a driver.

## S4 — Verdict

**Faithful.** Agreement 0.990 with 0.928 of explanations surviving intervention: suppressing the branch the overlay blamed does change or weaken the decision. It is a legitimate secondary contribution — a symbolic evidence overlay translating heterogeneous branch evidence into readable forensic traces. It remains an explanation overlay; the detector is not neuro-symbolic.

## Naming

The symbolic layer is post-hoc and does not enter inference, so this is a **symbolic explanation overlay**. Calling the detector neuro-symbolic would require the symbolic layer to participate in the Real/Fake decision and to be shown to hold or improve performance — a separate, later, measured claim.

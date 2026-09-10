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
| `u_high` | 0.2321 | 75th percentile of mean branch vacuity on dev — the quarter of dev samples the model is least certain about |
| `margin_semantic` | 0.3568 | median |p_semantic - tau_semantic| on dev — separates a decisive branch vote from a marginal one |
| `margin_artifact` | 0.4139 | median |p_artifact - tau_artifact| on dev — separates a decisive branch vote from a marginal one |
| `margin_fsvfm` | 0.4497 | median |p_fsvfm - tau_fsvfm| on dev — separates a decisive branch vote from a marginal one |

Predicates: `fake_<branch>` (branch votes fake at its own EER point), `strong_<branch>` (vote is past the dev median margin, i.e. decisive rather than marginal), `agree` / `conflict` (all branches on the same side or not), `insufficient` (mean branch vacuity in the top dev quartile).

## S2 — Rules

| rule | condition | assessment |
|---|---|---|
| R1 | all three vote fake, at least one decisively | Fake |
| R2 | all three vote real | Real |
| R3 | two vote fake, at least one decisively | Fake |
| R4 | branches conflict **and** evidence is insufficient | Inconclusive |
| R5 | a single decisive branch against two abstaining | Fake |
| R0 | none of the above | Inconclusive (fallback) |

R1, R3 and R5 name a **driver**: the decisive fake-voting branch furthest past its threshold. That named branch is what the intervention test suppresses. R2/R4/R0 name no single branch, make no falsifiable claim about which view mattered, and are excluded from the intervention denominator.

## S3 — Validation

| dataset | n | agreement | coverage | driver claims | flipped | weakened | unchanged |
|---|---:|---:|---:|---:|---:|---:|---:|
| CDFv2 | 518 | 1.000 | 0.598 | 40 | 3 | 35 | 2 |
| CDFv3 | 5418 | 1.000 | 0.518 | 263 | 26 | 216 | 21 |
| DFD | 3431 | 1.000 | 0.599 | 144 | 18 | 114 | 12 |
| DFDC | 4704 | 1.000 | 0.672 | 61 | 13 | 44 | 4 |
| DFDCP | 654 | 1.000 | 0.557 | 20 | 11 | 8 | 1 |
| DFEval24 | 814 | 1.000 | 0.506 | 3 | 1 | 2 | 0 |

**Agreement (necessary, not sufficient): 1.000** mean over the six sets, on samples where a rule fired.
**Specific-rule coverage: 0.575** — the rest fall through to Inconclusive.
**Consensus check:** 4,059 of 4,059 consensus explanations hold — no single suppression flips them, which is exactly what "all views concur" predicts.

**Intervention on driver claims (decisive): 0.925** survive — 72 flipped the decision and 419 weakened it toward the threshold by >0.05, against 40 unchanged, out of 531 explanations that named a driver.

## S4 — Verdict

**Faithful.** Agreement 1.000 with 0.925 of explanations surviving intervention: suppressing the branch the overlay blamed does change or weaken the decision. It is a legitimate secondary contribution — a symbolic evidence overlay translating heterogeneous branch evidence into readable forensic traces. It remains an explanation overlay; the detector is not neuro-symbolic.

## Naming

The symbolic layer is post-hoc and does not enter inference, so this is a **symbolic explanation overlay**. Calling the detector neuro-symbolic would require the symbolic layer to participate in the Real/Fake decision and to be shown to hold or improve performance — a separate, later, measured claim.

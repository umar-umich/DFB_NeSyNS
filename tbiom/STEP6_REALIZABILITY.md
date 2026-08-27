# Step 6 — realizability ladder

Anchor `p_fused` (P0-DS epoch 1), expert `fsvfm_preserve`, on VALmix (1350 videos). Gates are cross-fitted LEAVE-ONE-DOMAIN-OUT, so a rung cannot pass by memorising the domains it trained on.

Anchor error 0.1941, oracle error 0.1163, so the achievable reduction is 0.0778. `rho_realize` is the fraction of that a realizable gate delivers.

| rung | features | gate AUROC vs target | **rho_realize** | gated error | domain AUROC | audit |
|---|---:|---:|---:|---:|---:|---|
| G0 confidence only | 2 | 0.526 | **+0.000** | 0.1941 | 0.663 | pass |
| G1 cross-branch | 4 | 0.635 | **-0.010** | 0.1948 | 0.764 | pass |
| G2 relational | 6 | 0.827 | **-0.181** | 0.2081 | 0.765 | pass |
| G3 branch diagnostics | 23 | 0.784 | **-0.210** | 0.2104 | 0.844 | **FAIL** |
| G4 probe on frozen [h_A ‖ h_b] | — | TODO(run) | TODO(run) | | | |

G4 needs embedding exports that no current run writes; recorded rather than silently omitted.

## Audit clamp

Domain AUROC is how well the SAME gate inputs predict which corpus a sample came from. A rung that reads provenance can appear to realize complementarity while actually recognising the dataset — the recurring failure mode this project has hit before. Above 0.80 is reported as failing, whatever its rho.

## Target variants

On `G3 branch diagnostics`. `t_b = 1[phi_b > 0]` gives opposite labels to samples a hair either side of zero, which could by itself hold gate AUROC near chance.

| target | gate AUROC |
|---|---:|
| binary 1[phi>0] | 0.784 |
| margin-filtered (drop ties) | 0.519 |
| continuous regression of phi | 0.267 |

## Verdict

**No rung realizes complementarity.** Best was G0 confidence only at rho +0.000 against a 0.25 bar.

Every rung, including one with per-layer adaptation and rate statistics, leaves the gate unable to tell where the expert applies. Combined with Steps 1-5 this is a characterised negative rather than an absence of evidence: the complementarity is measurable and stable (rescue margin +0.22), the ceiling is real, and what is missing is any test-time-observable signal that locates it.

**The paper is therefore the single-anchor reliability result** from Step 4 — margin-and-vacuity selective prediction that degrades gracefully under shift — with this realizability gap reported as the negative result it is. Do NOT manufacture a fusion story from a rung that only passed by reading provenance.

### What the target variants reveal — the mechanism, not just the number

| target | gate AUROC | reading |
|---|---:|---|
| binary `1[phi > 0]` | 0.784 | looks like real skill |
| margin-filtered (ties dropped) | 0.519 | **chance** |
| continuous regression of `phi` | 0.267 | below chance |

This is the most informative row in the step. The binary target's apparent 0.784 does **not**
come from predicting where the expert helps. It comes from predicting the **ties** — the large
mass of videos where anchor and expert agree, so `phi = 0` and the label is 0 by construction.
Remove those and the gate falls to 0.519, indistinguishable from chance; ask it to regress the
signed quantity and it does worse than chance.

So the gate can tell *whether the two branches will agree*, and cannot tell *who is right when
they disagree* — which is precisely the quantity applicability discounting needs. That is a
sharper statement of the negative than "rho is low", and it is the sentence the paper should
carry: the failure is not that the signal is weak, it is that the observable structure encodes
agreement rather than correctness.

It also explains the negative rho values. A gate that fires on agreement routes to the expert on
exactly the samples where routing cannot help, and occasionally where it hurts.

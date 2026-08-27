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

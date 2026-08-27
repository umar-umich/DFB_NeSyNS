# Step 5 — complementarity against the chosen anchor (P0-DS epoch 1)

Every earlier complementarity number was measured against the weaker CLIP port. A stronger anchor makes rescue HARDER — it is wrong less often, so the denominator shrinks and the remaining errors are the hard ones. An expert that failed against the port cannot pass here by accident.

Basis: VALmix, 1,350 videos over three real-world domains. The anchor comes through DiCoME's h5 path and the experts through ours, so videos are joined on (domain, basename) and labels are asserted to agree across the join.

| anchor readout | expert | videos | rescue | harm | margin | ceiling | recovered | AUROC | enters? |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| P0-DS fused | `fsvfm_preserve` | 1350 | 0.401 | 0.178 | +0.222 | 0.884 | 2.9% | 0.8852 → 0.8866 | no |
| P0-DS fused | `fsvfm_ordinary` | 1350 | 0.401 | 0.179 | +0.222 | 0.884 | 5.7% | 0.8852 → 0.8864 | no |
| P0-DS fused | `mrvae_rate` | 1350 | 0.405 | 0.530 | -0.126 | 0.884 | 0.0% | 0.8852 → 0.8797 | no |
| P0-DS artifact | `fsvfm_preserve` | 1350 | 0.415 | 0.183 | +0.233 | 0.887 | -2.8% | 0.8789 → 0.8825 | no |
| P0-DS artifact | `fsvfm_ordinary` | 1350 | 0.385 | 0.176 | +0.208 | 0.881 | -4.0% | 0.8789 → 0.8820 | no |
| P0-DS artifact | `mrvae_rate` | 1350 | 0.600 | 0.577 | +0.023 | 0.923 | -1.3% | 0.8789 → 0.8591 | no |

Bars: rescue margin > 0.05 pooled AND on ≥ 2 domains, AND realizable-gate recovery ≥ 25%.

## Per domain

| anchor | expert | domain | anchor acc | expert acc | margin |
|---|---|---|---:|---:|---:|
| P0-DS fused | `fsvfm_preserve` | CDFv2val | 0.893 | 0.747 | +0.276 |
| P0-DS fused | `fsvfm_preserve` | DFDCPval | 0.904 | 0.871 | +0.483 |
| P0-DS fused | `fsvfm_preserve` | DFEval24val | 0.620 | 0.602 | +0.098 |
| P0-DS fused | `fsvfm_ordinary` | CDFv2val | 0.893 | 0.740 | +0.250 |
| P0-DS fused | `fsvfm_ordinary` | DFDCPval | 0.904 | 0.878 | +0.511 |
| P0-DS fused | `fsvfm_ordinary` | DFEval24val | 0.620 | 0.600 | +0.095 |
| P0-DS fused | `mrvae_rate` | CDFv2val | 0.893 | 0.402 | -0.238 |
| P0-DS fused | `mrvae_rate` | DFDCPval | 0.904 | 0.498 | -0.366 |
| P0-DS fused | `mrvae_rate` | DFEval24val | 0.620 | 0.471 | -0.048 |
| P0-DS artifact | `fsvfm_preserve` | CDFv2val | 0.911 | 0.747 | +0.090 |
| P0-DS artifact | `fsvfm_preserve` | DFDCPval | 0.924 | 0.871 | +0.455 |
| P0-DS artifact | `fsvfm_preserve` | DFEval24val | 0.587 | 0.602 | +0.149 |
| P0-DS artifact | `fsvfm_ordinary` | CDFv2val | 0.911 | 0.740 | +0.060 |
| P0-DS artifact | `fsvfm_ordinary` | DFDCPval | 0.924 | 0.878 | +0.463 |
| P0-DS artifact | `fsvfm_ordinary` | DFEval24val | 0.587 | 0.600 | +0.134 |
| P0-DS artifact | `mrvae_rate` | CDFv2val | 0.911 | 0.402 | +0.028 |
| P0-DS artifact | `mrvae_rate` | DFDCPval | 0.924 | 0.498 | -0.354 |
| P0-DS artifact | `mrvae_rate` | DFEval24val | 0.587 | 0.471 | +0.003 |

## Verdict

**No expert enters against the chosen anchor.** Best pairing was P0-DS fused | fsvfm_ordinary at 5.7% realizable recovery against a 25% bar. That is now THREE independent bases — DF40-Dev and VALmix against the CLIP port, and VALmix against the stronger P0-DS anchor — agreeing that no fusion path is live.

Note the rescue half is UNDIMINISHED against the stronger anchor: margins of +0.222 and +0.233 here against +0.202 and +0.206 against the port. The oracle complementarity is real and did not shrink when the anchor improved. What collapsed is the realizable half — recovery fell from 7-10% to between -4.0% and 5.7%, with several pairings now NEGATIVE, meaning a fitted gate does worse than not gating.

Step 6 becomes the decisive experiment rather than a follow-up: if the realizability ladder also fails its audit, the multi-expert architecture has no support and the paper is the single-anchor reliability result from Step 4, with the realizability gap reported as a characterised negative.

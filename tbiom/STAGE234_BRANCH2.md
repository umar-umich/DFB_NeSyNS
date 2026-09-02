# Stages 2-4 — Branch 2: the projector and the rate curve, taken apart

| arm | projector | artifact | what it isolates |
|---|---|---|---|
| **P0-DS** | β-VAE | 64-d | the incumbent Branch 2 |
| **P1d** | MR-VAE | 69-d (+rate) | both changes at once |
| **Stage 2** | MR-VAE | 64-d | the **projector** alone |
| **Stage 3** | β-VAE | 69-d (+rate) | the **rate curve** alone |

`τ` frozen per arm **and** per readout on that arm's own FF++ val. Arms are differently calibrated, so a shared threshold would not compare operating points.

## fused (DS)

| dataset | P0-DS (β-VAE) AUROC | P1d (MR-VAE+rate) AUROC | Stage 2 (MR-VAE proj only) AUROC | Stage 3 (β-VAE + rate) AUROC | P0-DS (β-VAE) FPR_real | P1d (MR-VAE+rate) FPR_real | Stage 2 (MR-VAE proj only) FPR_real | Stage 3 (β-VAE + rate) FPR_real |
|---|---|---|---|---|---|---|---|---|
| CDFv2 | 0.9646 | 0.9505 | 0.9201 | 0.9470 | 0.129 | 0.067 | 0.034 | 0.118 |
| CDFv3 | 0.8409 | 0.8849 | 0.9076 | 0.8804 | 0.129 | 0.067 | 0.034 | 0.118 |
| DFD | 0.9421 | 0.9349 | 0.9311 | 0.9461 | 0.174 | 0.058 | 0.039 | 0.140 |
| DFDC | 0.8828 | 0.8700 | 0.8566 | 0.8669 | 0.317 | 0.254 | 0.173 | 0.290 |
| DFDCP | 0.8573 | 0.8957 | 0.8734 | 0.8838 | 0.391 | 0.252 | 0.178 | 0.365 |
| DFEval24 | 0.6922 | 0.6635 | 0.6425 | 0.6790 | 0.341 | 0.280 | 0.166 | 0.400 |
| VALmix | 0.8852 | 0.8843 | 0.8624 | 0.8795 | 0.213 | 0.175 | 0.090 | 0.246 |

| arm | mean AUROC (7) | mean FPR_real (7) | Δ AUROC vs P0-DS | Δ FPR vs P0-DS |
|---|---:|---:|---:|---:|
| P0-DS (β-VAE) | 0.8664 | 0.242 | +0.0000 | +0.000 |
| P1d (MR-VAE+rate) | 0.8691 | 0.165 | +0.0027 | -0.077 |
| Stage 2 (MR-VAE proj only) | 0.8562 | 0.102 | -0.0102 | -0.140 |
| Stage 3 (β-VAE + rate) | 0.8690 | 0.240 | +0.0025 | -0.003 |

## artifact view

| dataset | P0-DS (β-VAE) AUROC | P1d (MR-VAE+rate) AUROC | Stage 2 (MR-VAE proj only) AUROC | Stage 3 (β-VAE + rate) AUROC | P0-DS (β-VAE) FPR_real | P1d (MR-VAE+rate) FPR_real | Stage 2 (MR-VAE proj only) FPR_real | Stage 3 (β-VAE + rate) FPR_real |
|---|---|---|---|---|---|---|---|---|
| CDFv2 | 0.9685 | 0.9513 | 0.9189 | 0.9495 | 0.073 | 0.051 | 0.034 | 0.101 |
| CDFv3 | 0.8490 | 0.8845 | 0.9091 | 0.8889 | 0.073 | 0.051 | 0.034 | 0.101 |
| DFD | 0.9487 | 0.9316 | 0.9433 | 0.9425 | 0.132 | 0.044 | 0.019 | 0.047 |
| DFDC | 0.8764 | 0.8594 | 0.8609 | 0.8608 | 0.261 | 0.206 | 0.177 | 0.216 |
| DFDCP | 0.8570 | 0.8812 | 0.8797 | 0.8825 | 0.291 | 0.213 | 0.183 | 0.213 |
| DFEval24 | 0.6712 | 0.6480 | 0.6590 | 0.6829 | 0.182 | 0.173 | 0.077 | 0.159 |
| VALmix | 0.8789 | 0.8742 | 0.8647 | 0.8842 | 0.127 | 0.107 | 0.064 | 0.132 |

| arm | mean AUROC (7) | mean FPR_real (7) | Δ AUROC vs P0-DS | Δ FPR vs P0-DS |
|---|---:|---:|---:|---:|
| P0-DS (β-VAE) | 0.8642 | 0.163 | +0.0000 | +0.000 |
| P1d (MR-VAE+rate) | 0.8615 | 0.120 | -0.0028 | -0.042 |
| Stage 2 (MR-VAE proj only) | 0.8622 | 0.084 | -0.0020 | -0.079 |
| Stage 3 (β-VAE + rate) | 0.8702 | 0.138 | +0.0060 | -0.025 |

## semantic (CLIP)

| dataset | P0-DS (β-VAE) AUROC | P1d (MR-VAE+rate) AUROC | Stage 2 (MR-VAE proj only) AUROC | Stage 3 (β-VAE + rate) AUROC | P0-DS (β-VAE) FPR_real | P1d (MR-VAE+rate) FPR_real | Stage 2 (MR-VAE proj only) FPR_real | Stage 3 (β-VAE + rate) FPR_real |
|---|---|---|---|---|---|---|---|---|
| CDFv2 | 0.9564 | 0.9431 | 0.9195 | 0.9443 | 0.225 | 0.101 | 0.034 | 0.152 |
| CDFv3 | 0.8199 | 0.8773 | 0.9081 | 0.8737 | 0.225 | 0.101 | 0.034 | 0.152 |
| DFD | 0.9399 | 0.9319 | 0.9297 | 0.9415 | 0.223 | 0.058 | 0.050 | 0.223 |
| DFDC | 0.8812 | 0.8695 | 0.8577 | 0.8652 | 0.368 | 0.265 | 0.173 | 0.347 |
| DFDCP | 0.8547 | 0.8990 | 0.8752 | 0.8773 | 0.448 | 0.291 | 0.183 | 0.457 |
| DFEval24 | 0.6909 | 0.6633 | 0.6403 | 0.6686 | 0.439 | 0.371 | 0.182 | 0.493 |
| VALmix | 0.8818 | 0.8810 | 0.8607 | 0.8634 | 0.314 | 0.225 | 0.105 | 0.332 |

| arm | mean AUROC (7) | mean FPR_real (7) | Δ AUROC vs P0-DS | Δ FPR vs P0-DS |
|---|---:|---:|---:|---:|
| P0-DS (β-VAE) | 0.8607 | 0.320 | +0.0000 | +0.000 |
| P1d (MR-VAE+rate) | 0.8664 | 0.202 | +0.0058 | -0.118 |
| Stage 2 (MR-VAE proj only) | 0.8559 | 0.109 | -0.0048 | -0.212 |
| Stage 3 (β-VAE + rate) | 0.8620 | 0.308 | +0.0013 | -0.012 |

## Attribution — which half of P1d moved DFDCP and CDFv3

| dataset | P0-DS | Stage 2 (projector) | Stage 3 (rate) | P1d (both) | Δ projector | Δ rate | additive? |
|---|---:|---:|---:|---:|---:|---:|---|
| DFDCP | 0.8573 | 0.8734 | 0.8838 | 0.8957 | +0.0161 | +0.0265 | yes |
| CDFv3 | 0.8409 | 0.9076 | 0.8804 | 0.8849 | +0.0667 | +0.0395 | no (-0.0621) |
| CDFv2 | 0.9646 | 0.9201 | 0.9470 | 0.9505 | -0.0445 | -0.0176 | no (+0.0480) |
| DFD | 0.9421 | 0.9311 | 0.9461 | 0.9349 | -0.0111 | +0.0040 | yes |
| DFDC | 0.8828 | 0.8566 | 0.8669 | 0.8700 | -0.0262 | -0.0159 | no (+0.0293) |
| DFEval24 | 0.6922 | 0.6425 | 0.6790 | 0.6635 | -0.0497 | -0.0132 | no (+0.0342) |
| VALmix | 0.8852 | 0.8624 | 0.8795 | 0.8843 | -0.0229 | -0.0057 | no (+0.0277) |

"Additive?" asks whether P1d's effect equals the sum of its two halves. Where it does not, the projector and the rate curve interact and neither can be read alone.

## Stage 4 — the Branch-2 decision

Standing instruction: the updated β-VAE (**Stage 3**) is the **default** Branch 2 and ships even if its gain is not significant, for differentiation and possible real-side robustness — subject to one guardrail. It must not degrade **CDFv3 AUROC** or **real-side FPR** beyond the noise band. *Neutral is acceptable, regression is not.*

- CDFv3 AUROC: **+0.0395** (outside the band)
- CDFv3 real-side FPR: **-0.011**
- real-side FPR across all seven: mean **-0.003**, worse by >0.02 on **2** dataset(s) (DFEval24, VALmix)

**SHIP Stage 3 as Branch 2.** The rate response is neutral-or-better on the strong axes, so the standing instruction applies and the updated β-VAE is adopted. Report its contribution honestly — if the AUROC gain is inside the band, say "comparable AUROC with a real-side benefit" and do not sell it as a driver of gains it did not produce.

Third clause of the rule — if Stage 2's projector beats Stage 3 on the full framework **and** does not hurt CDFv3, it may be Branch 2 instead. Stage 2 mean AUROC 0.8562 vs Stage 3 0.8690; Stage 2 CDFv3 +0.0667 vs P0-DS. **Stage 2 does not qualify.**

## Missing exports

- Stage 3 @e10 (sensitivity)/p_fused: no FF++ val export
- Stage 3 @e10 (sensitivity)/p_artifact: no FF++ val export
- Stage 3 @e10 (sensitivity)/p_semantic: no FF++ val export

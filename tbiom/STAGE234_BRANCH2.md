# Stages 2-4 — Branch 2: the projector and the rate curve, taken apart

| arm | projector | artifact | what it isolates |
|---|---|---|---|
| **P0-DS** | β-VAE | 64-d | the incumbent Branch 2 |
| **P1d** | MR-VAE | 69-d (+rate) | both changes at once |
| **Stage 2** | MR-VAE | 64-d | the **projector** alone |
| **Stage 3** | β-VAE | 69-d (+rate) | the **rate curve** alone |

`τ` frozen per arm **and** per readout on that arm's own FF++ val. Arms are differently calibrated, so a shared threshold would not compare operating points.

## fused (DS)

| dataset | P0-DS (β-VAE) AUROC | P1d (MR-VAE+rate) AUROC | P0-DS (β-VAE) FPR_real | P1d (MR-VAE+rate) FPR_real |
|---|---|---|---|---|
| CDFv2 | 0.9646 | 0.9505 | 0.129 | 0.067 |
| CDFv3 | 0.8409 | 0.8849 | 0.129 | 0.067 |
| DFD | 0.9421 | 0.9349 | 0.174 | 0.058 |
| DFDC | 0.8828 | 0.8700 | 0.317 | 0.254 |
| DFDCP | 0.8573 | 0.8957 | 0.391 | 0.252 |
| DFEval24 | 0.6922 | 0.6635 | 0.341 | 0.280 |
| VALmix | 0.8852 | 0.8843 | 0.213 | 0.175 |

| arm | mean AUROC (7) | mean FPR_real (7) | Δ AUROC vs P0-DS | Δ FPR vs P0-DS |
|---|---:|---:|---:|---:|
| P0-DS (β-VAE) | 0.8664 | 0.242 | +0.0000 | +0.000 |
| P1d (MR-VAE+rate) | 0.8691 | 0.165 | +0.0027 | -0.077 |

## artifact view

| dataset | P0-DS (β-VAE) AUROC | P1d (MR-VAE+rate) AUROC | P0-DS (β-VAE) FPR_real | P1d (MR-VAE+rate) FPR_real |
|---|---|---|---|---|
| CDFv2 | 0.9685 | 0.9513 | 0.073 | 0.051 |
| CDFv3 | 0.8490 | 0.8845 | 0.073 | 0.051 |
| DFD | 0.9487 | 0.9316 | 0.132 | 0.044 |
| DFDC | 0.8764 | 0.8594 | 0.261 | 0.206 |
| DFDCP | 0.8570 | 0.8812 | 0.291 | 0.213 |
| DFEval24 | 0.6712 | 0.6480 | 0.182 | 0.173 |
| VALmix | 0.8789 | 0.8742 | 0.127 | 0.107 |

| arm | mean AUROC (7) | mean FPR_real (7) | Δ AUROC vs P0-DS | Δ FPR vs P0-DS |
|---|---:|---:|---:|---:|
| P0-DS (β-VAE) | 0.8642 | 0.163 | +0.0000 | +0.000 |
| P1d (MR-VAE+rate) | 0.8615 | 0.120 | -0.0028 | -0.042 |

## semantic (CLIP)

| dataset | P0-DS (β-VAE) AUROC | P1d (MR-VAE+rate) AUROC | P0-DS (β-VAE) FPR_real | P1d (MR-VAE+rate) FPR_real |
|---|---|---|---|---|
| CDFv2 | 0.9564 | 0.9431 | 0.225 | 0.101 |
| CDFv3 | 0.8199 | 0.8773 | 0.225 | 0.101 |
| DFD | 0.9399 | 0.9319 | 0.223 | 0.058 |
| DFDC | 0.8812 | 0.8695 | 0.368 | 0.265 |
| DFDCP | 0.8547 | 0.8990 | 0.448 | 0.291 |
| DFEval24 | 0.6909 | 0.6633 | 0.439 | 0.371 |
| VALmix | 0.8818 | 0.8810 | 0.314 | 0.225 |

| arm | mean AUROC (7) | mean FPR_real (7) | Δ AUROC vs P0-DS | Δ FPR vs P0-DS |
|---|---:|---:|---:|---:|
| P0-DS (β-VAE) | 0.8607 | 0.320 | +0.0000 | +0.000 |
| P1d (MR-VAE+rate) | 0.8664 | 0.202 | +0.0058 | -0.118 |

## Attribution — which half of P1d moved DFDCP and CDFv3

| dataset | P0-DS | Stage 2 (projector) | Stage 3 (rate) | P1d (both) | Δ projector | Δ rate | additive? |
|---|---:|---:|---:|---:|---:|---:|---|
| DFDCP | 0.8573 | TODO(run) | TODO(run) | 0.8957 | | | |
| CDFv3 | 0.8409 | TODO(run) | TODO(run) | 0.8849 | | | |
| CDFv2 | 0.9646 | TODO(run) | TODO(run) | 0.9505 | | | |
| DFD | 0.9421 | TODO(run) | TODO(run) | 0.9349 | | | |
| DFDC | 0.8828 | TODO(run) | TODO(run) | 0.8700 | | | |
| DFEval24 | 0.6922 | TODO(run) | TODO(run) | 0.6635 | | | |
| VALmix | 0.8852 | TODO(run) | TODO(run) | 0.8843 | | | |

"Additive?" asks whether P1d's effect equals the sum of its two halves. Where it does not, the projector and the rate curve interact and neither can be read alone.

## Stage 4 — the Branch-2 decision

Standing instruction: the updated β-VAE (**Stage 3**) is the **default** Branch 2 and ships even if its gain is not significant, for differentiation and possible real-side robustness — subject to one guardrail. It must not degrade **CDFv3 AUROC** or **real-side FPR** beyond the noise band. *Neutral is acceptable, regression is not.*

TODO(run) — Stage 2/3 exports incomplete; decision deferred.

## Missing exports

- Stage 2 (MR-VAE proj only)/p_fused: no FF++ val export
- Stage 2 (MR-VAE proj only)/p_artifact: no FF++ val export
- Stage 2 (MR-VAE proj only)/p_semantic: no FF++ val export
- Stage 3 (β-VAE + rate)/p_fused: no FF++ val export
- Stage 3 (β-VAE + rate)/p_artifact: no FF++ val export
- Stage 3 (β-VAE + rate)/p_semantic: no FF++ val export

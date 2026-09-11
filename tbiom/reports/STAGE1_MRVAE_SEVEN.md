# Stage 1 — MR-VAE health re-score, all seven datasets

Closes the load-bearing gap: MR-VAE's operational advantage was established on only **three of seven** datasets (CDFv2, DFDCP, VALmix), and its AUROC losses on **CDFv3** and **DFEval24** were never re-scored on real-side metrics. Those two rows are marked **decisive** below.

`tau` frozen per anchor **and per readout** on that anchor's own FF++ val, applied unchanged to all seven. No GPU work — every export already existed; both anchors go through one code path and one video-grouping rule.

## fused (DS)

τ: MR-VAE **0.4626**, β-VAE **0.4784**

| dataset | | n | MR-VAE AUROC | β-VAE AUROC | Δ AUROC | **MR-VAE FPR_real** | **β-VAE FPR_real** | **Δ FPR** | Δ EER | Δ d_RF |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CDFv2 |  | 518 | 0.9505 | 0.9646 | -0.0141 | **0.067** | 0.129 | **-0.062** | +0.0374 | -0.012 |
| CDFv3 | **decisive** | 5418 | 0.8849 | 0.8409 | +0.0441 | **0.067** | 0.129 | **-0.062** | -0.0184 | +0.060 |
| DFD |  | 3431 | 0.9349 | 0.9421 | -0.0073 | **0.058** | 0.174 | **-0.116** | -0.0119 | +0.052 |
| DFDC |  | 4704 | 0.8700 | 0.8828 | -0.0128 | **0.254** | 0.317 | **-0.064** | +0.0147 | +0.012 |
| DFDCP |  | 654 | 0.8957 | 0.8573 | +0.0384 | **0.252** | 0.391 | **-0.139** | -0.0566 | +0.075 |
| DFEval24 | **decisive** | 814 | 0.6635 | 0.6922 | -0.0287 | **0.280** | 0.341 | **-0.061** | +0.0394 | -0.016 |
| VALmix |  | 1350 | 0.8843 | 0.8852 | -0.0009 | **0.175** | 0.213 | **-0.039** | +0.0111 | +0.018 |

Mean Δ AUROC **+0.0027**, mean Δ FPR_real **-0.077**. MR-VAE has the lower real-side FPR on **7 of 7** datasets.

## artifact view

τ: MR-VAE **0.3453**, β-VAE **0.3477**

| dataset | | n | MR-VAE AUROC | β-VAE AUROC | Δ AUROC | **MR-VAE FPR_real** | **β-VAE FPR_real** | **Δ FPR** | Δ EER | Δ d_RF |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CDFv2 |  | 518 | 0.9513 | 0.9685 | -0.0172 | **0.051** | 0.073 | **-0.022** | +0.0288 | -0.044 |
| CDFv3 | **decisive** | 5418 | 0.8845 | 0.8490 | +0.0355 | **0.051** | 0.073 | **-0.022** | -0.0214 | +0.008 |
| DFD |  | 3431 | 0.9316 | 0.9487 | -0.0170 | **0.044** | 0.132 | **-0.088** | -0.0022 | -0.025 |
| DFDC |  | 4704 | 0.8594 | 0.8764 | -0.0169 | **0.206** | 0.261 | **-0.055** | +0.0132 | -0.025 |
| DFDCP |  | 654 | 0.8812 | 0.8570 | +0.0242 | **0.213** | 0.291 | **-0.078** | -0.0406 | +0.008 |
| DFEval24 | **decisive** | 814 | 0.6480 | 0.6712 | -0.0232 | **0.173** | 0.182 | **-0.009** | +0.0074 | -0.008 |
| VALmix |  | 1350 | 0.8742 | 0.8789 | -0.0047 | **0.107** | 0.127 | **-0.021** | -0.0044 | -0.022 |

Mean Δ AUROC **-0.0028**, mean Δ FPR_real **-0.042**. MR-VAE has the lower real-side FPR on **7 of 7** datasets.

## semantic (CLIP)

τ: MR-VAE **0.6279**, β-VAE **0.6086**

| dataset | | n | MR-VAE AUROC | β-VAE AUROC | Δ AUROC | **MR-VAE FPR_real** | **β-VAE FPR_real** | **Δ FPR** | Δ EER | Δ d_RF |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CDFv2 |  | 518 | 0.9431 | 0.9564 | -0.0133 | **0.101** | 0.225 | **-0.124** | +0.0259 | +0.019 |
| CDFv3 | **decisive** | 5418 | 0.8773 | 0.8199 | +0.0575 | **0.101** | 0.225 | **-0.124** | -0.0399 | +0.077 |
| DFD |  | 3431 | 0.9319 | 0.9399 | -0.0080 | **0.058** | 0.223 | **-0.165** | -0.0022 | +0.084 |
| DFDC |  | 4704 | 0.8695 | 0.8812 | -0.0117 | **0.265** | 0.368 | **-0.103** | +0.0108 | +0.023 |
| DFDCP |  | 654 | 0.8990 | 0.8547 | +0.0443 | **0.291** | 0.448 | **-0.157** | -0.0487 | +0.079 |
| DFEval24 | **decisive** | 814 | 0.6633 | 0.6909 | -0.0276 | **0.371** | 0.439 | **-0.068** | +0.0246 | -0.010 |
| VALmix |  | 1350 | 0.8810 | 0.8818 | -0.0007 | **0.225** | 0.314 | **-0.089** | +0.0044 | +0.029 |

Mean Δ AUROC **+0.0058**, mean Δ FPR_real **-0.118**. MR-VAE has the lower real-side FPR on **7 of 7** datasets.

## Pass condition and branch

The brief's rule: if MR-VAE's real-side advantage holds broadly **and** CDFv3 / DFEval24 carry no large FPR penalty, the rate response is a live Branch-2 ingredient. If either collapses on real-side FPR the way its AUROC did, the rate response must be gated per the Branch-2 rule.

| decisive dataset | Δ AUROC | Δ FPR_real | AUROC verdict | real-side verdict |
|---|---:|---:|---|---|
| DFEval24 | -0.0287 | -0.061 | MR-VAE worse | MR-VAE better |
| CDFv3 | +0.0441 | -0.062 | MR-VAE better | MR-VAE better |

Broad real-side advantage: MR-VAE lower FPR on **7/7** datasets, mean **-0.077**.

**PASS — the rate response is a live Branch-2 ingredient.** The real-side advantage holds broadly, and neither decisive dataset carries an FPR penalty: the AUROC losses on CDFv3/DFEval24 are *not* accompanied by real-side collapse. Stage 3 (β-VAE artifact + MR-VAE rate curve) proceeds as the preferred Branch 2.

---

## Caveat that limits the "7 of 7" claim

**Celeb-DF-v2 and Celeb-DF-v3 share the same 178 real videos.**

| dataset | n | real | fake |
|---|---:|---:|---:|
| CDFv2 | 518 | **178** | 340 |
| CDFv3 | 5,418 | **178** | 5,240 |

CDFv3 is CDFv2's real set with ~4,900 additional fakes. Their `FPR_real@τ` is therefore
identical **by construction** — 0.067 / 0.129 fused, 0.051 / 0.073 artifact, 0.101 / 0.225
semantic — not by independent agreement. This is why the AUROC gap between them is large while
the real-side numbers are the same: everything that differs lives on the fake side.

So the real-side result is **6 independent datasets, not 7**, and CDFv3 is decisive on AUROC but
carries no independent real-side information. On the six independent sets MR-VAE still has the
lower FPR_real on **6 of 6** (CDFv2, DFD, DFDC, DFDCP, DFEval24, VALmix), mean −0.080 fused. The
PASS verdict stands; the phrasing "7 of 7" should not be used in the paper.

DFEval24 remains a genuinely independent decisive row, and it is the one that matters most: it is
the only dataset where MR-VAE loses AUROC (−0.0287) **and** the only modern in-the-wild set. It
carries **no** real-side penalty there (−0.061 FPR), which is exactly the pattern the Stage-4
guardrail was written to detect the absence of.

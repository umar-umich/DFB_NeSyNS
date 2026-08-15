# DISCERN v2 — complete pilot results

All numbers read directly from result files by `experiments/common/analysis/make_results_summary.py`. Baseline for every delta is **P0-DS** (self-trained FF++ c23 reproduction, seed 42).

## Pilots

| pilot | change from P0-DS | views |
|---|---|---|
| P1a | deterministic AE (no KL) | 2 |
| P1b | beta-TCVAE (drops MI term) | 2 |
| P1c | WAE | 2 |
| P1d | MR-VAE + rate-distortion response | 2 |
| P2a | AEROBLADE recon residual (frozen SDXL-VAE, 3-view) | 3 |
| P3a | LaRE2-inspired diffusion consistency (frozen SD 3.5 MMDiT, 3-view) | 3 |
| P4 | LGrad gradient image (frozen StyleGAN-D + ResNet-50, 3-view) | 3 |

Training: FF++ c23, seed 42, batch 128, LR 1e-4, 20 epochs (**P2a stopped at 13, P3a at 18** — selected epochs 5 and 1, well inside the range every pilot selected from).

---

## 1. Conventional cross-dataset + wild (video-level AUC)

| source | P0-DS | P1a | P1b | P1c | P1d | P2a | P3a | P4 |
|---|---|---|---|---|---|---|---|---|
| FFpp | 0.9929 | 0.9924 | 0.9902 | 0.9892 | 0.9924 | 0.9928 | 0.9914 | 0.9897 |
| CDFv2 | 0.9644 | 0.9396 | 0.9434 | 0.9421 | 0.9495 | 0.9597 | 0.9641 | 0.9458 |
| CDFv3 | 0.9516 | 0.9100 | 0.8950 | 0.8902 | 0.8852 | 0.8969 | 0.8664 | 0.8840 |
| DFD | 0.9419 | 0.9320 | 0.9255 | 0.9346 | 0.9358 | 0.9362 | 0.9482 | 0.9309 |
| DFDC | 0.8825 | 0.8613 | 0.8563 | 0.8627 | 0.8705 | 0.8660 | 0.8664 | 0.8615 |
| DFDCP | 0.8576 | 0.8879 | 0.8814 | 0.8713 | 0.8959 | 0.8854 | 0.8686 | 0.8708 |
| DFEval24 | 0.6918 | 0.6494 | 0.6514 | 0.6560 | 0.6645 | 0.6811 | 0.6685 | 0.6603 |
| **mean Δ** |  | **-0.0157** | **-0.0199** | **-0.0195** | **-0.0127** | **-0.0092** | **-0.0156** | **-0.0199** |

Deepfake-Eval-2024 is the in-the-wild deployment benchmark; FF++ is in-domain.

---

## 2. DF40 family aggregates — Δ vs P0-DS

| source | P0-DS | P1a | P1b | P1c | P1d | P2a | P3a | P4 |
|---|---|---|---|---|---|---|---|---|
| FSAll-cdf | 0.9709 | +0.0035 | -0.0154 | -0.0057 | +0.0041 | -0.0009 | +0.0052 | -0.0058 |
| FSAll-ff | 0.9820 | -0.0100 | -0.0135 | -0.0187 | -0.0041 | -0.0072 | -0.0037 | -0.0080 |
| FRAll-cdf | 0.6594 | +0.1272 | +0.1455 | +0.1343 | +0.1034 | +0.1231 | +0.0327 | +0.1135 |
| FRAll-ff | 0.8216 | +0.0130 | +0.0502 | +0.0216 | +0.0212 | +0.0297 | -0.0259 | +0.0253 |
| EFSAll-cdf | 0.9880 | +0.0023 | -0.0213 | -0.0146 | -0.0056 | -0.0096 | -0.0022 | -0.0109 |
| EFSAll-ff | 0.9976 | -0.0069 | -0.0116 | -0.0142 | -0.0037 | -0.0067 | +0.0005 | -0.0074 |
| FEAll-cdf | 0.9898 | -0.0034 | +0.0043 | -0.0072 | +0.0066 | -0.0142 | -0.0013 | -0.0199 |
| FEAll-ff | 0.9993 | -0.0059 | -0.0020 | -0.0134 | +0.0003 | -0.0084 | -0.0004 | -0.0065 |

`-cdf` rows use Celeb-DF source video, `-ff` use FF++. FS=face swap, FR=reenactment, EFS=entire-face synthesis, FE=face editing.

---

## 3. Key diagnostic DF40 rows (absolute AUC)

The rows that decide the verdicts. `danet/mcnet/tpsm-cdf` are the three where **P0-DS is anti-correlated** (worse than chance). `heygen` is the only *commercial* reenactment product in DF40 and the hardest row in the benchmark.

| source | P0-DS | P1a | P1b | P1c | P1d | P2a | P3a | P4 |
|---|---|---|---|---|---|---|---|---|
| danet-cdf | 0.3083 | 0.6089 | 0.6348 | 0.6314 | 0.5723 | 0.5680 | 0.4154 | 0.5510 |
| mcnet-cdf | 0.3824 | 0.6932 | 0.7201 | 0.7121 | 0.6506 | 0.6480 | 0.4851 | 0.6407 |
| tpsm-cdf | 0.4005 | 0.5910 | 0.6613 | 0.5790 | 0.5951 | 0.5595 | 0.3966 | 0.5514 |
| facevid2vid-cdf | 0.5287 | 0.7762 | 0.7876 | 0.7964 | 0.7162 | 0.7682 | 0.5843 | 0.7626 |
| heygen | 0.9439 | 0.8329 | 0.8294 | 0.7859 | 0.8745 | 0.8082 | 0.9431 | 0.7443 |
| deepfacelab | 0.9627 | 0.9671 | 0.9573 | 0.9514 | 0.9700 | 0.9553 | 0.9604 | 0.9500 |

---

## 4. Training saturation diagnostic

Why in-domain validation cannot rank these models.

| pilot | epochs | sel epoch | peak val | epochs to 99% of peak | final train | train−val gap |
|---|---|---|---|---|---|---|
| P0-DS | 20 | 1 | 0.9960 | 0 | 1.0000 | +0.0071 |
| P1a | 20 | 8 | 0.9954 | 0 | 1.0000 | +0.0053 |
| P1b | 20 | 4 | 0.9962 | 0 | 1.0000 | +0.0057 |
| P1c | 20 | 10 | 0.9957 | 0 | 1.0000 | +0.0052 |
| P1d | 20 | 1 | 0.9957 | 0 | 1.0000 | +0.0057 |
| P2a | 13 | 5 | 0.9960 | 0 | 1.0000 | +0.0049 |
| P3a | 20 | 1 | 0.9958 | 0 | 1.0000 | +0.0056 |
| P4 | 20 | 7 | 0.9952 | 0 | 1.0000 | +0.0054 |

**Every pilot reaches 99% of its peak validation AUROC in epoch 0, and every pilot ends at train AUROC 1.0000.** Between-pilot peak spread is 0.00093 while mean within-pilot epoch-to-epoch std is 0.00046 — a signal/noise ratio of **2.0x**. Checkpoint selection is therefore close to arbitrary, and the models are **not capacity-limited** (so LoRA-rank / LayerNorm / SVD-style adapter changes would be optimising something already at ceiling).

---

## 5. Caveats that must survive into the paper

- **P3a is "LaRE2-inspired", never "LaRE2".** Stability removed every SD 1.x/2.x repo from HuggingFace, so it uses **SD 3.5 medium**, whose MMDiT is a rectified-flow model predicting *velocity* on a 16-channel latent — LaRE2's epsilon-residual does not exist there.
- **P4 legitimately keeps the name "LGrad"**: both stages are the authors' original weights (StyleGAN-D + ProGAN-trained ResNet-50); only the executor changed, from TensorFlow to a PyTorch port of the discriminator.
- **P4's weights are off-distribution for faces** — trained on ProGAN *objects* (car/cat/chair/horse) with a *bedroom* StyleGAN. That is the likely cause of its conventional-axis harm, not a flaw in the gradient representation as such.
- **`heygen` is n=101 videos.** Treat P3a's 0.943 as "did not degrade", not an exact match.
- **Single seed throughout.** Per the pilot plan, |Δ| in 0.005–0.015 needs a second seed before any conclusion; the FR gains (+0.10–0.15) and heygen losses (0.07–0.20) are far outside that band.
- **`pixart` excluded** (incomplete download), so EFSAll is pixart-excluded.
- **DFD baseline is 0.9419**, not the paper's 0.982 — interpret DFD deltas against ours.
- **DF40 family taxonomy corrected** vs `stage_df40.py`: stargan/starganv2 -> FE, deepfacelab -> FS, heygen -> FR. Aggregate rows unaffected.
- **DF40 AUCs are fp32** (matching `eval_df40.py`); the 7 primary sources are bf16-mixed (matching Lightning's eval loop). Do not mix the two when quoting per-frame scores.

## 6. Assets built but NOT yet adopted

- **VALmix** (`eval_adaptation/data/h5/VALmix.h5`, 1,350 videos / 40,142 frames): leak-free out-of-domain validation from Celeb-DF-v2 non-test + DFDCP official-train + Deepfake-Eval-2024 finetuning-train. Adopting it requires retraining the suite for comparability, **and makes Deepfake-Eval-2024's test AUC no longer strictly zero-shot** (it becomes a held-out test with in-distribution model selection).
- **DF40 train split** (31 methods, pre-cropped, at `/data/saad/datasets/video/df40/train`): candidate *training* data for the process-diverse direction. Deliberately NOT used for validation — it shares all 31 generators with the DF40 test split.

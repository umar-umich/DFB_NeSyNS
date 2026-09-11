# MR-VAE rate operator — fit, and why the rate dimension is inert

Umar unblocked this on 2026-08-25: fit the mechanism-valid multi-rate operator and evaluate it
fresh against the CLIP anchor, with the rate-response analysis explicitly **diagnostic only** and
membership decided on complementarity. The operator is now fit
(`configs/discern_v2/rate/rate_operator_mrvae.pt`). This records what the fit shows, so the Stage 1
complementarity result can be read in context.

## The fit succeeded; the rate dimension did not

Trained on 23,039 authentic FF++ FS-VFM features, 100 epochs, β log-uniform over (0.1, 10.0), the
schedule the operator's readout grid assumes. β-ELBO converged 27.79 → 0.0062. The authentic rate
response:

```
beta_0.1  0.00616    beta_0.32 0.00616    beta_1.0 0.00616    beta_3.16 0.00616    beta_10.0 0.00615
```

Flat to five significant figures across two decades of β. `fit_rate_operator.py`'s monotonicity
guard fired, which is what it exists for.

## Why: the KL term never bites, and cosine has no range on this embedding

Measured on 4,000 held-out authentic features:

| β | recon | KL | β·KL |
|---:|---:|---:|---:|
| 0.10 | 0.006195 | 2e-06 | 0.000000 |
| 1.00 | 0.006189 | 1e-06 | 0.000001 |
| 10.0 | 0.006183 | 1e-06 | 0.000013 |

`β·KL` is negligible everywhere, so β cannot move the reconstruction — there is no rate–distortion
tradeoff to trace.

The deeper cause is the feature space, not the hyperparameters:

* the fitted operator's authentic reconstruction error is **0.00616**;
* simply predicting **the mean authentic embedding** gives **0.00606**;
* so the operator learned **nothing beyond the mean** — it is marginally worse than the constant
  predictor;
* random authentic *pairs* differ by only **0.0120** in cosine distance, i.e. FS-VFM face
  embeddings are nearly collinear, so cosine reconstruction has almost no dynamic range to work in;
* authentic vs fake distance-to-mean is 0.00606 vs 0.00632 — a **0.1σ** separation.

## It is not a sizing choice — five configurations, none work

Latent 8–128, hidden 64–256, cosine and MSE objectives, 15 epochs each:

| config | rate curve (β 0.1 → 10) | spread | real/fake AUROC |
|---|---|---:|---:|
| L128 h256 cos | 0.01142 0.01123 0.01106 0.01089 0.01073 | 6.1% | 0.5475 |
| L32 h256 cos | 0.00869 0.00855 0.00842 0.00830 0.00817 | 5.9% | 0.5397 |
| L8 h64 cos | 0.00845 0.00852 0.00860 0.00870 0.00882 | 4.1% | 0.5539 |
| L32 h256 mse | 0.31534 0.31187 0.30840 0.30483 0.30119 | 4.5% | 0.5012 |
| L8 h64 mse | 0.62043 0.62059 0.62051 0.62020 0.61965 | 0.2% | 0.5045 |

No configuration produces both a meaningful rate response and forgery signal. The best real/fake
AUROC anywhere is **0.554**, near chance, on the training distribution itself.

## The honest scope of this finding

This is **MR-VAE hosted on frozen FS-VFM features**, which was a Phase-2 hosting decision
(`phase2/REPO_MAP.md` item 5), taken because the operator's original DiCoME feature space — a 64-d
LoRA-CLIP `f_s` — does not exist for this architecture. So the finding is "MR-VAE is inert **on this
feature space**", not "MR-VAE is inert".

There is one alternative host that might work: the CLIP-LoRA anchor's own 64-d semantic feature,
which is the closest analogue to DiCoME's `f_s`. But hosting the rate expert on the anchor's
representation makes it a re-reading of the anchor rather than a **heterogeneous** prior, which is
the property the architecture is built around. That trade should be made deliberately if it is made
at all — it is not a free fix.

## What this does and does not decide

Per Umar's instruction the rate-response diagnostic **does not by itself decide membership**;
complementarity does. So MR-VAE still goes through the Stage 1 gate against the CLIP anchor on
DF40-Dev, and it costs nothing extra to do so — the rate response is computed from the same frozen
FS-VFM features the FS-VFM expert's scoring pass already produces.

The expectation should be calibrated, though: an operator at 0.554 AUROC on its own training
distribution has little signal to be complementary *with*. If Stage 1 confirms that, MR-VAE is
dropped and the architecture is the anchor plus one specialist, which reduces Stage 3's
applicability target to the brief's single-specialist form `phi_b = U(A+b) - U(A)`.

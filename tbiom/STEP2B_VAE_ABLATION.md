# Step 2b — controlled VAE/alignment ablation

Does the auxiliary objective explain why DiCoME's semantic branch beats a plain CLIP+LoRA port
by +0.045 AUROC on Celeb-DF-v2 (Step 1)?

## Correction to an earlier claim in this file

An earlier version said BOTH detachments were necessary — the projector's input and the
reconstruction target — and that detaching only the input would let gradient back into CLIP
through the target. **That is wrong.** `aligned_vae_loss_func` already does
`z_original.detach()` internally, so the target detach was redundant. Measured, gradient into the
CLIP encoder from the VAE/alignment term alone:

| what is detached | gradient into CLIP |
|---|---:|
| nothing (baseline) | 34,873 |
| target only | 34,873 — no effect |
| **projector input only** | **0** |
| both (what the ablation ran) | 0 |

The ablation itself is unaffected: it ran with both detaches and gradient was 0, so the -0.0211
result stands. Only the explanation of why was wrong.

## The manipulation

One variable. `detach_vae_from_encoder: true` feeds the Semantic Manifold branch a detached CLIP
feature and detaches the reconstruction target, so the VAE and alignment terms train the
projector but send NO gradient into the shared CLIP encoder. The evidential loss still reaches
CLIP through both heads.

Verified by measurement before training — gradient magnitude into the CLIP encoder from the
VAE/alignment term alone:

| arm | into CLIP encoder | into projector |
|---|---:|---:|
| baseline | 34,873 (98 tensors) | 613.635 |
| ablation | **0** (0 tensors) | 613.635 |

The training config is identical to the one that produced P0-DS across all 17 compared keys
(seed, batch size, LR, schedule, epochs, precision, backbone, feature dim, and the VAE
hyperparameters themselves), so **P0-DS is the baseline of this pair** — no baseline retrain.

## Result

Development data only: tau frozen per arm/readout on FF++ val, metrics on VALmix, macro-averaged
over its three domains.

| arm | readout | macro AUROC | EER | macro FPR_real | d_RF |
|---|---|---:|---:|---:|---:|
| baseline (P0-DS) | fused | 0.8647 | 0.1859 | 0.213 | +0.439 |
| baseline (P0-DS) | **semantic** | **0.8617** | 0.1933 | 0.314 | +0.314 |
| baseline (P0-DS) | artifact | 0.8559 | 0.2044 | 0.127 | +0.368 |
| ablation | fused | 0.8457 | 0.2000 | 0.267 | +0.487 |
| ablation | **semantic** | **0.8406** | 0.2037 | 0.258 | +0.407 |
| ablation | artifact | 0.8531 | 0.1963 | 0.194 | +0.420 |

**Removing the VAE/alignment gradient from the shared encoder costs the semantic branch
−0.0211 macro AUROC (0.8617 → 0.8406).** The fused head loses a similar −0.0190.

## Reading

The auxiliary objective **is** load-bearing for the CLIP representation. It is not a
hyperparameter detail: the same encoder, same data, same schedule, same seed, differing only in
whether the manifold loss can shape it, is measurably worse without it.

But it does not account for the whole Step-1 gap. The port sits ~0.045 below DiCoME's semantic
branch on Celeb-DF-v2; this ablation recovers about **half** that magnitude on VALmix's macro
AUROC. So the auxiliary objective is *a* cause, not *the* cause, and the remaining confounders —
batch size (128 vs 32), precision, augmentation, optimisation dynamics — still have room to
explain the rest. Those should be tested the same way, one at a time.

Caveats that limit how hard this can be pushed:

- Single seed. The spec's §22 rule treats |delta AUROC| < 0.01 as needing a confirming seed;
  −0.0211 clears that bar but only just, and a second seed would settle it.
- Measured on VALmix macro AUROC, which is not the Celeb-DF-v2 test number the +0.045 gap was
  quoted on. The two are not directly subtractable; only the direction and rough magnitude
  transfer.

## A side observation worth carrying

The artifact view is barely affected by the ablation (0.8559 → 0.8531) while the semantic view
drops. That is the expected direction — the projector still trains normally, only the encoder
stops being shaped by it — and it is a small consistency check that the manipulation did what
the gradient measurement said it did.

Note also the artifact view has by far the best real-side health in BOTH arms (FPR_real 0.127
baseline, 0.194 ablation, against the semantic branch's 0.314 / 0.258). Combined with Step 1's
finding that artifact-only beats fused on Celeb-DF-v2, DFD and DFDC, this strengthens the case
for not inheriting DiCoME's fused output as the eventual anchor.

# Why the I-JEPA residual branch is failing, and what DiCoME's artifact branch does differently

Self-contained. All numbers are from `analysis/ijepa/` on this repository.

---

## 1. What we measured

The branch: I-JEPA ViT-H/14, frozen. Hide one of five face blocks, let the pretrained predictor
guess the hidden block's features from the rest of the face, and use the prediction error as
evidence. Heads train on FF++ c23 only; selection and scoring are cross-dataset (the v3.1
development objective over swap, reenactment and in-the-wild).

| row | what the head sees | FF++ val AUROC (in domain) | dev v3.1 (cross-dataset) |
|---|---|---:|---:|
| R2 | mean-pooled I-JEPA feature, **no prediction at all** | 0.7842 | **0.6812** |
| R2b | per-block target features, **no prediction** | 0.8132 | 0.6701 |
| R3 | prediction residuals | 0.8955 | 0.6622 |
| R4 | prediction residuals, quality-standardised | 0.8942 | 0.6658 |
| — | **backbone feature + residuals together** | — | **0.6664** |
| — | residual magnitudes only (5 scalars, no directions) | — | **0.5024** |

Three facts follow, and they are the whole problem.

1. **The residual loses to a plain probe on its own backbone.** 0.6658 against 0.6812.
2. **It does not add to that backbone — it subtracts.** Given both, the head reaches 0.6664, worse
   than the backbone alone at 0.6812. They are redundant, and the residual is the noisier of the two.
3. **It is a pure generalisation failure, not a weak signal.** In domain the residual is far stronger
   (0.8955 against 0.7842). Its cross-dataset score peaks at epoch 2 and declines for eighteen more.

We tried to fix (3) as overfitting and could not. Cutting the shared projection eightfold moved the
turnover from epoch 2 to epoch 9 and reached the same 0.6655. Dropout 0.5 gave 0.6648. The peak is
the ceiling, not an artefact of capacity.

---

## 2. What DiCoME's artifact branch actually does

From `DISCERN_Ext/src/model/core_model.py:318-336` and `src/losses/aligned_vae_loss.py`:

```
f_s  = CLIP ViT-L/14 + LoRA  ->  semantic feature, 64-d
f_c  = BetaVAEWithAlignment(f_s)              # 32-d latent, reconstruction back to 64-d
f_r  = f_s - f_c                              # what the manifold cannot explain
f_par = (<f_r, f_s> / ||f_s||^2) * f_s        # the part of f_r pointing along f_s
f_a  = f_r - f_par                            # THE ARTIFACT FEATURE, orthogonal to f_s
```

The VAE objective is `lambda_align * (1 - cos(f_s.detach(), f_c)) + beta_kld * KLD`, with
`beta_kld = 2.0`, `lambda_align = 1.0`, latent 32.

Four properties matter, and the I-JEPA branch has none of them.

**(a) The reference is learned on the task.** The "manifold" f_c is a β-VAE trained jointly with the
detection objective on the training data. It learns what a normal face looks like *for this task*.
I-JEPA's reference is a predictor trained on ImageNet for masked feature prediction and then frozen.
Its notion of "normal" is ImageNet's, not faces'.

**(b) The residual is orthogonalised against the view it must complement.** DiCoME explicitly removes
the component of f_r that points along f_s. The artifact branch is therefore guaranteed by
construction to carry something the semantic branch does not. **I-JEPA has no such step**, and our
"both" experiment is exactly the consequence: the residual is redundant with the backbone feature and
degrades it when combined.

**(c) The residual is small and lives in the same space as the feature it complements.** f_a is 64-d,
in CLIP's space, next to a 64-d semantic feature. I-JEPA's residual is 5 x 1280 in ViT-H space,
squeezed through a shared 1280->64 projection. Our scalars-only result (0.5024, near chance) says the
transferable part is not in the magnitudes, so the head is forced to work in the high-dimensional
directions, which is where it memorises FF++.

**(d) The bottleneck is deliberate and tight.** 64 -> 32 -> 64 with beta_kld 2.0. The residual is
defined as what a 32-dimensional code cannot hold. I-JEPA's residual is whatever a 12-layer predictor
gets wrong, with no bottleneck controlling its information content.

---

## 3. Suggested fixes, in order of cost

**Fix A — orthogonalise the residual against the view it must complement. Cheapest, directly
targets the measured failure.**

Port DiCoME's step verbatim onto the cached I-JEPA residuals:

```
r_perp_k = r_bar_k - (<r_bar_k, g> / ||g||^2) * g
```

where `g` is the feature the branch must complement — either the pooled I-JEPA feature (to
decorrelate from its own backbone) or the deployed Appearance feature (to decorrelate from the view
it would sit beside in fusion). Requires no retraining of anything: the residuals and both candidate
`g` vectors are already cached. This is the single intervention most directly indicated by the
evidence, because our failure mode is *redundancy*, which is precisely what (b) exists to prevent.

**Fix B — adapt the predictor on real faces (R6, already specified in Amendment 1).**

LoRA r8/α16 on `attn.qkv` and `attn.proj` across all 12 predictor blocks, trained on real FF++ faces
only, encoder and target-encoder frozen, masks from I-JEPA's own multiblock collator, never the five
evaluation blocks. This is the analogue of DiCoME property (a): the reference distribution becomes
faces instead of ImageNet. It changes what the measurement *means* rather than how the head reads it,
which is the one lever the head-side variants could not touch.

**Fix C — bottleneck the residual, mirroring property (d).**

Replace the 1280->64 linear projection with a trained compressive bottleneck — for instance a small
VAE over the concatenated residual directions with a KL term — so the head sees "what a k-dimensional
code of the residual retains" rather than a linear read of 6400 raw dimensions. Directly attacks the
memorisation route our variants showed is being used.

**Fix D — the structural question worth asking.**

DiCoME's artifact branch is not an independent detector. It is defined *relative to* the semantic
branch (a residual of it, orthogonalised against it) and only has to be non-redundant, not strong.
The I-JEPA branch is being built as a standalone view and then asked to beat a probe. If the goal is
a fusion contribution, the branch should be defined relative to the view it must complement from the
start — which is Fix A taken seriously rather than applied as a post-hoc projection.

---

## 4. The honest caveat

Fix A addresses redundancy and Fix B addresses the prior, and they are independent. If after both the
orthogonalised, face-adapted residual still fails to add to the backbone, the conclusion is that
regional predictability under a JEPA prior does not carry face-forgery evidence that survives a
dataset shift — which is a publishable negative, and one nobody has established.

Also note what this branch was *good* at: it beat the plain probe on in-the-wild forgeries (0.608
against 0.555) while losing on reenactment (0.684 against 0.775). If it is retained at all, that
asymmetry is the reason, not its aggregate.

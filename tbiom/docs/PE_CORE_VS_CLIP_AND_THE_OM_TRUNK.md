# PE-Core vs CLIP, and why the Off-manifold view needs a trunk of its own

Companion to `LORA_AND_VIT_ARCHITECTURE.md`, written the same way: every shape and every number below
was read off the instantiated models or off a run JSON in `analysis/upgrade/`, not from a paper.

It answers a question UMAR put directly — *"if we have a good APP view with PE-Core using LoRA on
Q,V and LN updates, maybe we should do the similar encoder replacement for the OM branch"* — and
it ends in a plan. The short answer is **yes, and it matters more than you think, because the OM
branch does not currently have an encoder to replace.**

**§2 is the motivation layer** — what CLIP, PE-Core and DINOv3 were each pretrained to optimise, why
that makes a contrastively-trained trunk structurally unhelpful for an artifact view, and which five
OM-side transformations that reasoning supports. Read it before deciding which arm in §7 to fund.

---

## 1. The two backbones, measured

Read from the live models (`timm 'vit_pe_core_large_patch14_336'`, HF `openai/clip-vit-large-patch14`):

| | CLIP ViT-L/14 | PE-Core ViT-L/14-336 |
|---|---|---|
| params | 303.2 M | **316.1 M** |
| blocks / width / heads | 24 / 1024 / 16 | 24 / 1024 / 16 |
| patch | 14 | 14 |
| input | 224 | **336** |
| patch tokens | 256 | **576** (2.25x) |
| attention projections | **separate** `q_proj`, `k_proj`, `v_proj`, `out_proj` | **fused** `attn.qkv` (3072x1024), `attn.proj` |
| global readout | pre-projection **CLS token** | **`AttentionPoolLatent`** (`global_pool='map'`) |
| preprocessing mean/std | CLIP's (0.481/0.458/0.408) | **(0.5, 0.5, 0.5)** |
| LayerNorm parameters | 102,400 | 104,448 |

Four of those rows are the whole story of why this is not a drop-in swap:

1. **Capacity is matched.** 316 M against 303 M, same depth, same width, same patch size. A win here
   is the backbone, not the parameter count. That was the confound that sank PE-Core *base* (86 M)
   in the earlier screen and is why `pe_core_l` was added on Dr Kutub's suggestion.
2. **PE-Core sees 2.25x the tokens.** Our stored crops are 256 px, so PE-Core runs at a **1.31x
   upsample on 100 % of frames** while CLIP runs at a downsample. This is a real confound and it is
   recorded in `upgrade/extract_spatial.py:57-63`. It also explains the wall-clock: a PE-Core-L
   adaptation run takes 100-320 min against CLIP's 38-48 min.
3. **The readout is a different object.** CLIP's feature is one token that was never trained to
   summarise anything (the pre-projection CLS). PE-Core's is a learned attention pool over all 576
   tokens. That difference, not LoRA, is the main reason PE-Core's frozen features are better
   behaved at mid depth — see §2.2 and §3.
4. **`qkv` is fused, so LoRA on PE-Core adapts K too.** On CLIP we deliberately target `q_proj,
   v_proj` and skip K. On PE-Core there is one Linear carrying all three, so wrapping it adapts
   Q, K and V. The arithmetic is a coincidence worth knowing:

   ```
   CLIP     q_proj + v_proj : 2 x (8*1024 + 1024*8)  = 32,768 / block
   PE-Core  fused qkv       :      8*1024 + 3072*8   = 32,768 / block
   both x 24 blocks                                  = 786,432 trainable
   ```

   Identical budget, different coverage. `upgrade/adapt_backbone.py:79-91` logs this as
   `fused_qkv: true` in every run JSON so the deviation is never silent.

**Recipe parameter counts, measured from the run JSONs:**

| recipe | what trains | CLIP | PE-Core-L |
|---|---|---:|---:|
| **L** | LoRA r=8, alpha=16, drop=0.1 on Q/V (CLIP) or fused QKV (PE-Core) | 786,432 | 786,432 |
| **N** | LayerNorm affine parameters only, features L2-normalised (GenD recipe) | 102,400 | 104,448 |
| **LN** | both at once | — | **890,880** |

`LN` is 0.28 % of PE-Core-L. Nothing is unfrozen; the backbone is restorable exactly.

---

## 2. What each backbone was pretrained to do, and what that implies

This section is the motivation layer. Everything above and below it is measured; the pretraining
descriptions here come from the papers and releases, and are marked **[lit]** where that is the case
so nobody mistakes them for something we verified. Everything marked **[measured]** was read off our
own runs or the live models.

### 2.1 The three objectives

| | CLIP ViT-L/14 | PE-Core L/336 | DINOv3 ViT-L/16 |
|---|---|---|---|
| supervision | **image-text contrastive** (InfoNCE over caption pairs) **[lit]** | **image-text contrastive**, refined recipe: curated data engine, image **and video** pairs, progressive resolution, tuned augmentation **[lit]** | **self-supervised, no language**: image-level self-distillation with an EMA teacher and multi-crop, plus masked patch-level prediction, plus Gram anchoring to stop dense features degrading on long schedules **[lit]** |
| what the loss rewards | "which caption goes with this image" | same question, better data and more of it | "agree with yourself across crops and augmentations, and predict masked patches from context" |
| readout it optimises | the pooled/CLS embedding | the attention-pool embedding **[measured]** | CLS + 4 **register tokens**, `global_pool='avg'` **[measured]** |
| params / input / patch tokens | 303.2 M / 224 / 256 **[measured]** | 316.1 M / 336 / 576 **[measured]** | 303.1 M / 256 / 256 + 5 prefix **[measured]** |

The three sit in two camps, and the camp boundary is the one that matters for us:

**CLIP and PE-Core are both language-supervised.** Their training signal is "could this image be
described by this sentence". A representation is rewarded for carrying whatever a human caption would
mention and is *penalised for nothing else* — dimensions that encode resampling kernels, JPEG block
structure, blending seams, or per-generator high-frequency texture never help match a caption, so the
objective has no reason to preserve them and the finite width of the embedding gives it reason to
discard them. **The contrastive objective is, from a forensic point of view, a learned invariance to
exactly the signals we need.** PE-Core is a better-trained member of the same family, not a different
kind of model: better data, video pairs, higher resolution, a learned attention pool instead of a
CLS token that was never trained to summarise anything.

**DINOv3 is not language-supervised**, so it has no caption bottleneck and is free to keep appearance
and texture detail — which is why the DINO family is the standard choice for dense correspondence,
segmentation and depth. But its augmentation pipeline trains invariance to photometric nuisances
(blur, jitter, solarisation) **[lit]**, and those overlap the processing perturbations our robustness
caches apply. So it trades one blindness for another: it keeps texture that CLIP throws away, and it
is explicitly trained not to care about some of the degradations we test under.

### 2.2 The one place our measurements speak directly to a pretraining claim

PE's titular finding is that the best visual embeddings are **not** at the output of the network: the
contrastive objective concentrates language alignment at the final layers while intermediate layers
retain stronger general-purpose visual structure, and the aligned variants (PE-Lang, PE-Spatial) exist
to pull those intermediate features back out **[lit]**. Two independent confirmations sit in our own
numbers:

- **Our Part 3d result is that claim, on our task.** PE-Core-L's block-16 feature beats its own final
  attention-pooled feature by **+0.059 dev AUROC** frozen (0.8019 vs 0.7356), while CLIP shows no such
  gap (0.7311 vs 0.7478) **[measured, §3]**. We found this by sweeping depth for unrelated reasons and
  it lines up with the paper's central claim.
- **PE-Lang-L is physically 23 blocks deep, not 24** — `timm` loads
  `vit_pe_lang_large_patch14_448` with `len(m.blocks) == 23` against PE-Spatial-L's 24 **[measured]**.
  Meta shipped the language-aligned variant with the last block removed. That is the "not at the
  output" claim expressed in the checkpoint.

This is the strongest available motivation for **Arm 1** in §7: if the contrastive objective is what
strips low-level evidence, then the depth at which it has not yet done so is where an artifact view
should read — and for PE-Core that depth is measurably better, for free, in the same forward pass.

### 2.3 Why this also settles the LoRA-vs-LayerNorm choice for OM

§7's Arm 2 gives APP the `LN` recipe and OM `N` (LayerNorm only). The pretraining objective is the
reason, not just the fusion numbers:

**LoRA on a contrastively pretrained trunk pushes the representation further along the objective it
was already trained on.** It adapts the attention projections — *what gets retrieved* — under a
discriminative loss, which is directionally the same thing the caption loss was doing: make the
embedding more about the decision, less about everything else. For APP that is exactly right. For a
branch whose entire job is to carry what the semantic manifold does **not** explain, it is the wrong
direction, and it also moves the manifold the beta-VAE is trying to reconstruct while the beta-VAE is
trying to reconstruct it.

LayerNorm-only rescales each block's activation statistics and leaves the attention function intact,
so the subspace structure the orthogonal projection operates on survives training. The fusion numbers
agree from the other side: `N` has 1/7.7 the trainable parameters and still beats `L` on fused AUROC
and the 10-set mean while losing standalone **[measured, §3]** — a knob that trades standalone
strength for complementarity is the right knob for the complementary view.

### 2.4 Five OM-side transformations, ranked by how well the pretraining supports them

UMAR's framing — *apply OM-related transformations to expose artifacts using these big backbones* —
is the right one. Each candidate below is tied to a specific property of how the backbone was
trained, and each has an **availability** line, because three of the five need something we do not
currently have installed.

---

**(1) Text-anchored semantic manifold.** Replace the beta-VAE entirely. CLIP's contrastive objective
co-locates image and text embeddings in one space, so the span of the text embeddings of a set of
face-describing prompts **is** a semantic manifold, defined by the pretraining objective rather than
invented by a 60-line MLP:

```
f_c = P_text f_s ,   P_text = projection onto span{ E_text(prompt_i) }
f_a = f_s - f_c        "what cannot be named"
```

**Why the pretraining supports it:** this is the one construction that makes "off the semantic
manifold" mean something checkable. Today the manifold is whatever a 32-dimensional MLP bottleneck
happened to learn, and §4(b) shows the result adds no information. A text-span projection is
training-free, its rank is set by the number of prompts rather than an arbitrary latent width, and it
inherits the alignment the objective spent 400 M pairs establishing. We already have a 211-attribute
FaceBench vocabulary to build the prompt set from.

**Availability: runnable today on CLIP, not on PE-Core.** The text tower requires the joint embedding
space, so the readout must move from the pre-projection CLS we currently use to the projected
embedding — a known, small change. HF `CLIPModel` gives the text tower and is already a dependency
(`src/encoders/clip_encoder.py` loads the full `CLIPModel` and discards the text side today).
**`timm` ships PE vision towers only** — no `open_clip` in `discern_ext_timm` **[measured]** — so
PE-Core's text tower needs Meta's `perception_models` release installed first.

**This is the best idea in this section** and it is cheap. It should be Arm 5.

---

**(2) Predictor-residual OM.** Use a backbone that was pretrained *with* a predictor and take the
residual between what the predictor expects and what is actually there.

**Why the pretraining supports it:** this is what OM has always been trying to approximate. I-JEPA's
objective is latent-space prediction — predict the representation of masked target blocks from a
context block — so its predictor **is** a trained manifold model, and the prediction error on a real
face is the model's own notion of "on-manifold" **[lit]**. Our beta-VAE is a hand-rolled, 68 k-parameter
stand-in for that, fitted on 64-d features.

**Availability: immediate for I-JEPA.** `branches/ijepa/IN1K-vit.h.14-300e.pth.tar` contains a
`predictor` state dict and `extract_residuals.py:48` already loads it `strict=True` **[measured]** —
12 blocks at width 384, 22.4 M params. This is the R6 proposal in `LORA_AND_VIT_ARCHITECTURE.md` §3,
now with a reason to run it that is about OM rather than about appearance.
**Not immediate for DINOv3:** its masked-prediction head is part of the training setup, and `timm`
distributes the backbone only, so the iBOT-style head would have to come from Meta's full release.
Check before planning around it.

**Caveat that must be stated:** I-JEPA is the *weakest* backbone we have on this task when probed by
its global feature, and it is a frozen-head probe so the comparison is confounded. Predictor residual
is a different readout of it, so the prior is weak, not negative.

---

**(3) Patch-token second-order statistics (Gram / covariance).** Our OM lives on a single global
64-d vector. PE-Core-L has **576 patch tokens** and DINOv3 **256** **[measured]**, and a mean-pool
throws away all of their covariance structure.

**Why the pretraining supports it:** DINOv3 needed **Gram anchoring** as an explicit training
objective because the *similarity structure among patch features* degrades on long schedules **[lit]**
— the release treats patch-to-patch Gram structure as a first-class quantity worth protecting. Face
swaps and blends are local operations that disturb exactly that structure while leaving the global
pooled embedding, which the contrastive objective optimised, largely intact. A Gram or covariance
readout is a genuinely different *perspective* on an identical distribution: same backbone, same
forward pass, same preprocessing, no new weights.

**Availability: immediate.** `upgrade/extract_layers.py` already hooks patch tokens at chosen depths
and currently mean-pools them (`g[:, 1:].mean(1)`); emitting a covariance summary instead is a few
lines in the same pass.

**Trap, measured:** that slice assumes one prefix token. DINOv3 ViT-L has **5** (CLS + 4 register
tokens) **[measured]**, so `g[:, 1:]` would fold four register tokens into the patch mean. We only
ever screened DINOv3 at the `last` readout so no published number is affected, but any DINOv3 depth
work must fix the slice first.

---

**(4) Local-versus-global consistency.** Both camps' objectives enforce an invariance we can test for
violations of: contrastive training pulls crops of one image toward the same caption embedding, and
DINO's multi-crop objective makes a local crop's embedding agree with the global one **[lit]**. Real
faces satisfy that invariance by construction — it is what the model was trained on. A face that was
swapped or blended region-by-region has no reason to.

So: embed the full face crop and its sub-regions with the **same** backbone, and score the
disagreement. Same distribution by construction, and the signal is "this image violates an invariance
the pretraining established", which is a sharper story than "a second head disagreed".

**Availability: mostly built.** `upgrade/part6c_regional.py` already does regional extraction, and
`upgrade/extract_regional.py` exists. The missing piece is the consistency score rather than the
features.

---

**(5) PE-Spatial-L and PE-Lang-L, the variants we dismissed on a confounded result.** We screened
`pe_spatial_base_patch16_512`: 86 M against CLIP's 303 M, and a **2.0x upsample** of our 256 px
crops. It lost (0.6842 dev) and PE-Spatial was set aside. But the **large** variants exist and are
size-matched: `vit_pe_spatial_large_patch14_448` is **304.0 M, 24 blocks, 1024-d**, and
`vit_pe_lang_large_patch14_448` is **291.4 M, 23 blocks** **[measured — both load from `timm` with
pretrained weights in `discern_ext_timm` today]**.

**Why the pretraining supports it:** PE-Spatial is the variant whose *intermediate* features were
explicitly aligned for dense and spatial tasks **[lit]** — the same property §2.2 shows our own depth
sweep rediscovering. If the mid-depth advantage is the mechanism, PE-Spatial-L is the checkpoint built
to deliver it, and it is the natural trunk for the OM side of Arm 2 or Arm 3.

**Availability: immediate, but expensive.** 448 input from 256 crops is a **1.75x upsample** (against
PE-Core-L's 1.31x) and 448/14 gives **1024 patch tokens**, 4x CLIP's. Expect roughly 2x PE-Core-L's
already-slow adaptation runs. Screen it frozen first — that is cached-feature work and costs almost
nothing — before committing to an adaptation run.

---

### 2.5 What this motivation rules *out*

Worth writing down, because it saves GPU time:

- **Do not expect a bigger or better contrastive backbone to fix OM.** CLIP to PE-Core-L is worth
  ~0.01 dev AUROC; adaptation is worth ~0.12 **[measured, §3]**. Both are the same objective, and
  that objective is what discards the evidence. Swapping PE-Core-L for PE-Core-G would be more of the
  same.
- **Do not build OM as another discriminatively-adapted global readout of the same trunk.** That is
  what we have, and §4(b) explains why it cannot add information. Every transformation above changes
  either the *input* (regions), the *depth* (pre-collapse features), the *readout* (patch covariance),
  or the *manifold definition* (text span, trained predictor) — one of those four, or it is not a new
  view.
- **Do not reach for DINOv3 on the strength of its texture retention alone.** Its global feature is
  the worst of the four we screened (0.7013 dev, APP weight zeroed in 4/5 folds **[measured, §5]**).
  The interesting part of DINOv3 is its patch-level machinery, and the head that would make that
  usable is not in the `timm` weights.

---

## 3. What the swap actually bought

All numbers from `analysis/upgrade/screen_*.json` (5 source-grouped dev folds) and
`analysis/upgrade/final/*_s0.json` (10 video-level test sets).

### Frozen backbones, no adaptation

| backbone | dev v3.1 | VALmix-v2 | FF++ val | fused |
|---|---:|---:|---:|---:|
| CLIP ViT-L/14 | 0.7550 | 0.7723 | 0.9138 | 0.9130 |
| DINOv3 ViT-L/16 | 0.7013 | 0.7298 | 0.8146 | 0.9091 |
| PE-Core base/224 | 0.6761 | 0.6974 | 0.8419 | 0.9093 |
| PE-Spatial base/512 | 0.6842 | 0.7157 | 0.8721 | 0.9093 |
| **PE-Core L/336** | 0.7426 | **0.7893** | 0.9144 | 0.9135 |

Frozen, PE-Core-L is a wash with CLIP — slightly behind on dev v3.1, ahead on VALmix-v2.

### Adapted

| arm | dev v3.1 | VALmix-v2 | FF++ val | fused | 10-set mean |
|---|---:|---:|---:|---:|---:|
| CLIP + N | 0.8433 | 0.8602 | 0.9898 | 0.9144 | — |
| CLIP + L | 0.8605 | 0.8943 | 0.9945 | 0.9259 | 0.9130 |
| PE-Core-L + L | 0.8699 | 0.9134 | 0.9947 | 0.9302 | 0.9190 |
| PE-Core-L + N | 0.8679 | 0.9087 | 0.9929 | 0.9329 | 0.9212 |
| **PE-Core-L + LN** | **0.8735** | **0.9223** | **0.9953** | **0.9364** | **0.9240** |

Three things to take from this table, because they drive the plan in §7:

- **Adaptation is worth ~0.12 dev AUROC; the backbone swap is worth ~0.01.** Frozen CLIP 0.7550 to
  adapted CLIP 0.8605 is the big move. CLIP+L 0.8605 to PE-Core-L+L 0.8699 is the small one. The
  headline is the *recipe*, and PE-Core is a real but modest top-up.
- **N is not the poor relation.** LayerNorm-only has **1/7.7 the trainable parameters of LoRA** and
  beats LoRA on fused AUROC (0.9329 vs 0.9302) and on the 10-set mean (0.9212 vs 0.9190), while
  losing marginally standalone. LoRA buys standalone strength; LN buys *fusion* strength. Hold that
  thought.
- **LN (both) wins everything**, and it wins by less than the sum of its parts, which is what you
  expect from two knobs acting on the same trunk.

### Layer readout (Part 3d, frozen)

| readout | CLIP dev | PE-Core-L dev | PE-Core-L phi(APP,OM) |
|---|---:|---:|---:|
| `last` (model's own pool) | 0.7478 | 0.7356 | 0.2119 |
| `half` (block 12) | 0.7279 | 0.7704 | 0.2628 |
| `mid` (block 16, 2/3 depth) | 0.7311 | **0.8019** | 0.2206 |
| `last+mid` | 0.7690 | 0.7902 | 0.2404 |
| `last+mid+half` | **0.7843** | 0.8124 | 0.2882 |

**PE-Core-L's mid-depth feature beats its own final feature by +0.059 dev AUROC, frozen.** CLIP shows
no such gap (0.7311 vs 0.7478). That is a property of PE-Core, and it is the cheapest untapped axis
we have: `upgrade/extract_layers.py` already caches every depth **in one forward pass**, so reading a
second depth costs nothing.

### Nuisance profile (Part 2b)

| | CLIP | PE-Core-L |
|---|---:|---:|
| processing leakage `E_delta` | 0.0094 | 0.0049 |
| source leakage `E_delta` | 0.0250 | 0.0778 |
| manipulation-family energy | 0.886 | 0.614 |

Neither leakage term clears its permutation null in either backbone, so no claim either way. PE-Core
carries *less* of its class-separation direction in manipulation-family identity (0.614 vs 0.886),
which is mildly encouraging for generalisation and is not evidence of it.

---

## 4. The finding that reframes the question: OM has no encoder

UMAR's phrasing was "the OM branch encoder". There isn't one. Read
`DISCERN_Ext/src/model/core_model.py:194-235` and `src/modules/beta_vae_aligned.py` together and the
Off-manifold view is this, end to end:

```
image ──► CLIP ViT-L/14 (+LoRA q,v) ──► CLS ──► Linear(1024 -> 64) ──► f_s     [this IS the APP feature]
                                                                        │
                      beta-VAE: Linear(64->32) ReLU ──► mu (32)  ────────┤      z = mu at eval
                                Linear(32->32) ReLU Linear(32->64) ──► f_c
                                                                        │
            f_r = f_s - f_c ;  f_a = f_r - proj_{f_s}(f_r)  ────────────►┘ ──► artifact head ──► OM opinion
```

Three consequences, each checkable:

**(a) OM and APP are two readouts of one 64-dimensional vector.** There is no second forward pass, no
second backbone, no second preprocessing. `_geometric_view_purification` is six lines of linear
algebra on `f_s`.

**(b) At evaluation, OM carries no information that APP does not already have.**
`beta_vae_aligned.py:47-49` is `if not self.training: return mu` — the latent is deterministic at
eval. So `f_c = decoder(mu(f_s))` is a function of `f_s`, and

```
f_a  =  f_r - proj_{f_s}(f_r)  =  -( f_c - proj_{f_s}(f_c) )
```

is a deterministic function of `f_s` alone, factored through a **32-dimensional** bottleneck. Formally
the OM opinion is `g(f_s)` for a fixed `g`. It is a *different nonlinear readout* of the same vector,
which is why it can score differently — not a second source of evidence.

**(c) The feature both views live on is 94 % discarded before either sees it.**
`src/encoders/clip_encoder.py:52-55` projects CLIP's 1024 dims to `config.feature_dim = 64`. The
upgrade screens, by contrast, put their head on the **full 1024-d** backbone output
(`upgrade/adapt_backbone.py:71-77`). So the deployed OM is built on a 64-d squeeze of a feature the
upgrade's APP uses at full width. That mismatch has never been ablated.

---

## 5. Why this is exactly the fusion problem

Eleven fusion experiments failed this month. (b) above predicts all of them. The measurements agree.

**Leave-one-out contribution and error correlation, per screen arm** (mean over 5 dev folds;
`folds_weight_zero` counts folds where the fitted weight for that view was exactly 0):

| arm | fused | loo T | loo **APP** | loo OM | phi(APP,OM) | phi(APP,T) | APP zero-weight folds |
|---|---:|---:|---:|---:|---:|---:|---:|
| CLIP frozen | 0.9130 | 0.0479 | **0.0037** | 0.0491 | 0.251 | 0.240 | **2 / 5** |
| CLIP `-half` | 0.9093 | 0.0643 | **0.0000** | 0.0643 | 0.354 | 0.176 | **5 / 5** |
| PE-Core base | 0.9093 | 0.0643 | **0.0000** | 0.0643 | 0.291 | 0.308 | **5 / 5** |
| DINOv3 | 0.9091 | 0.0609 | **-0.0002** | 0.0626 | 0.184 | 0.184 | **4 / 5** |
| PE-Core-L frozen | 0.9135 | 0.0428 | 0.0042 | 0.0483 | 0.230 | 0.287 | 0 / 5 |
| CLIP + L | 0.9259 | 0.0310 | 0.0187 | **0.0020** | 0.508 | 0.142 | 0 / 5 |
| PE-Core-L + L | 0.9302 | 0.0262 | 0.0224 | **0.0062** | 0.439 | 0.154 | 0 / 5 |
| PE-Core-L + LN | 0.9364 | 0.0256 | **0.0308** | **0.0044** | 0.402 | **0.075** | 0 / 5 |

Read the APP and OM columns as a pair. **They never both pay.** On every frozen trunk the fitted
fusion sets APP's weight to zero in 2-5 of 5 folds and its leave-one-out value is 0.000 — OM, the
better-shaped readout of the same vector, has already taken the credit. Adapt the trunk and the
ledger flips: APP rises to 0.031 and **OM collapses from 0.048 to 0.004**. The sum stays near 0.035
throughout. That is not two views combining. That is one view's worth of evidence changing hands,
which is precisely what (b) says must happen.

Meanwhile `phi(APP,OM)` *rises* with adaptation, 0.230 -> 0.439 -> 0.402: the two readouts' errors
become **more** correlated, not less. The only genuinely decorrelated view is the temporal one, and
adaptation makes it more so — `phi(APP,T)` falls to **0.075** under LN.

**The same thing seen from the test sets.** Per-view standalone AUROC under the deployed rule:

| dataset | T | APP | OM | fused | vs best single view |
|---|---:|---:|---:|---:|---:|
| DFEval24 | 0.673 | 0.679 | 0.669 | 0.729 | **+0.050** |
| DFDCP | 0.741 | 0.867 | 0.873 | 0.890 | +0.017 |
| CDFv3 | 0.926 | 0.857 | 0.880 | 0.956 | +0.029 |
| DFD | 0.797 | 0.928 | 0.934 | 0.935 | +0.001 |
| IDForge | 0.890 | **0.987** | 0.964 | 0.983 | -0.004 |
| DFDM | 0.980 | 0.896 | **0.992** | 0.984 | -0.008 |
| FFIW | 0.801 | **0.966** | 0.935 | 0.953 | -0.013 |
| KoDF | **0.966** | 0.853 | 0.855 | 0.943 | -0.023 |
| DFDC | 0.701 | 0.823 | **0.881** | 0.851 | -0.030 |
| CDFv2 | 0.738 | 0.870 | **0.974** | 0.943 | -0.031 |
| **mean** | 0.821 | 0.873 | 0.896 | **0.9166** | oracle best-single **0.9177** |

**An oracle that simply picked the single best view per dataset scores 0.9177; our three-view fusion
scores 0.9166.** The fusion loses to the best single view on 6 of 10 sets. It is performing *view
selection*, not evidence combination — and no combination rule can do better than selection when two
of the three views are functions of one vector.

Two corollaries worth stating plainly:

- **Stop looking for a better operator.** PCR6+-R, Dempster-R, cumulative-R, cautious-R, vacuity
  scaling, evidence vectors, grouped fusion, intervention-guided subspaces: all bounded by the same
  ceiling. The views are the binding constraint.
- **DFEval24 is a separate problem.** All three views sit at 0.669-0.679 there. Nothing is working,
  so nothing can be fused; fusion already adds the most it adds anywhere (+0.050) on top of three
  broken views. That is a training-domain gap and no encoder swap in this document addresses it.

---

## 6. The design target, made precise

UMAR's formulation is the right one and worth keeping as the objective function:

> **"The views should be from the same distribution but a different perspective, so fusion beats the
> individual views."**

Unpacked into things we can measure:

| | what it means here | how we check it |
|---|---|---|
| **same distribution** | same pretraining corpus, same input size, same mean/std, same readout family, so the two opinions are commensurable and a single reliability weight means the same thing for both | one backbone family (PE-Core-L/336) for both spatial views; identical preprocessing; log it |
| **different perspective** | the two views are not functions of one vector — they see different depths, different adapted parameters, or different inputs | `phi` error correlation **down**; both `loo` values **positive in the same fold** |
| **fusion beats the views** | not just fused AUROC up, but fused above the *oracle best single view* | `fused > max_view` on more than 4/10 sets |

Note what the current system gets right and wrong against this. Same-distribution: **right in the
deployed system** (one CLIP trunk, trivially) and **wrong in every upgrade arm** — the screens
replaced only `p_APP`/`u_APP` and kept OM from the old CLIP export (`upgrade/screen_eval.py:19`),
which is why OM's standalone AUROC is frozen to four decimals across arms (CDFv2 0.97358 / 0.97366 /
0.97365 / 0.97346). Different-perspective: **wrong in both.**

So we are currently failing one of the two conditions in every configuration we have ever run, and
neither failure is the one we spent the month attacking.

---

## 7. The plan

Five arms, cheapest first. Each is a self-contained experiment with its own stop condition, and the
primary outcome for all of them is **diversity**, not AUROC — because §5 shows AUROC moves last.

Fixed across Arms 1-4, so a difference is the arm and nothing else: PE-Core-L/336 for both spatial
views, same preprocessing, same head (`Linear(d,256)` GELU drop 0.1 `Linear(256,2)` softplus), the
same EDL loss and KL ramp, 20 epochs, FF++ c23 source-paired train, VALmix-v2 selection, 5
source-grouped dev folds, the temporal view untouched. Arm 5 runs on CLIP for the availability reason
in §2.4(1), so it is compared against the CLIP+L reference (fused 0.9259) rather than PE-Core-L+LN.

---

### Arm 1 — OM at a different depth of the same frozen trunk (`depth-OM`)

**Cost: ~0 extra GPU.** `upgrade/extract_layers.py` already returns every depth from one forward pass.

APP reads `last` (the attention pool). OM gets its own `Linear(1024 -> d_om)` projection, its own
beta-VAE, its own orthogonal residual and its own head, all on the **block-16 (`mid`) patch-token
mean**. Nothing is shared but the frozen weights.

**Why this first.** It is the minimal intervention that breaks condition (b) of §4 — the two views
stop being functions of one vector — while holding the distribution *exactly* fixed by construction.
And PE-Core's mid-depth feature is independently its strongest frozen readout (0.8019 vs 0.7356), so
this is not a sacrifice arm. `phi(APP,OM)` is already lowest at `last` and `mid` (0.212, 0.221).

**Adaptation decision: none.** Frozen, both views. This arm is about whether depth alone supplies
perspective. Adding adaptation here would confound it.

---

### Arm 2 — different knob per view on one trunk (`split-recipe`)

**Cost: 2 adaptation runs, ~100-230 min each on GPU 1.**

- **APP: recipe `LN`** — LoRA r=8 alpha=16 on fused `qkv` **plus** LayerNorm, readout `last`.
  This is the measured best arm (dev 0.8735, fused 0.9364, 10-set 0.9240) and we keep it unchanged.
- **OM: recipe `N` — LayerNorm only, no LoRA** — readout `mid`, with its own beta-VAE and orthogonal
  residual.

**Why N and not LoRA for OM, specifically.** Three reasons, in order of how much I trust them:

1. **LoRA rewrites the geometry the orthogonal projection depends on.** OM is defined as a
   *decomposition* of a feature into a manifold part and a residual orthogonal to it. LoRA changes
   what attention retrieves at every depth, so the manifold the beta-VAE is trying to reconstruct
   moves under it. LayerNorm rescales each block's activations without altering the attention
   function, so the subspace structure survives and the beta-VAE has a stable target. This is an
   argument from the construction, and it is the reason to prefer N here even though it is weaker
   standalone.
2. **N is measurably the better fusion citizen.** PE-Core-L+N beats +L on fused AUROC (0.9329 vs
   0.9302) and on the 10-set mean (0.9212 vs 0.9190) with **1/7.7 the trainable parameters**, while
   losing standalone (0.8679 vs 0.8699). A view whose job is to complement rather than to win alone
   should be trained with the knob that demonstrably trades standalone strength for fusion value.
3. **Different knob = different function class = decorrelation by construction.** If both views used
   `LN` on the same trunk with the same objective they would converge toward the same function and
   `phi` would rise, which is exactly the failure mode in §5's table. Different parameter sets is the
   cheapest available guarantee that it cannot happen.

**And one thing not to do:** do **not** give OM LoRA on a *different* projection set (say `mlp.fc1`,
or `attn.proj`) as the diversity mechanism. It is tempting and it is untested, but PE-Core's `qkv` is
fused so the "skip K" lever we have on CLIP does not exist here, and an MLP adapter is a 3x larger
budget on the branch we want to keep small. If Arm 2 passes and we want more diversity, that is
Arm 3's job, not a hyper-parameter tweak.

---

### Arm 3 — a second, independently adapted trunk (`two-trunk`)

**Cost: 2 adaptation runs plus 2x inference forever after.** Only if Arms 1-2 pass.

Two PE-Core-L instances. APP: `LN`, readout `last`, EDL on the semantic head. OM: `N`, readout `mid`,
and critically **its own objective** — the beta-VAE reconstruction and alignment terms train *its*
trunk, not APP's. In the deployed model those gradients flow into the shared CLIP encoder (or are
cut by `detach_vae_from_encoder`, which is the same problem stated differently); here they have
somewhere of their own to go.

This is the full version of what UMAR proposed. It is also the only arm that permanently doubles
inference cost, which is why it is gated on the cheap arms showing the mechanism exists.

---

### Arm 4 — the two squeezes nobody has ablated (`width`)

**Cost: head-only retraining on cached features, minutes per cell.** Run alongside Arm 1.

- `feature_dim`: **64** (deployed) vs 128 vs 256 vs 1024 (no projection). A 1024 -> 64 bottleneck
  discards 94 % of the dimensions before either view sees the feature, and the upgrade screens that
  produced every number in §3 used the full 1024. This is an uncontrolled difference between the
  deployed system and its own evaluation.
- `vae_latent_dim`: **32** (deployed) vs 8 vs 16 vs 48. Per §4(b) this integer is the exact width of
  the channel through which OM is allowed to differ from APP. It was never swept.

Cheap, and it may move more than the backbone swap did.

---

### Arm 5 — the text-anchored semantic manifold (`text-OM`)

**Cost: no training. One text-tower forward pass over the prompt set, then head-only fitting on
cached features.** Hours, not days. Run alongside Arms 1 and 4.

Replace `BetaVAEWithAlignment` with a projection onto the span of CLIP text embeddings, per §2.4(1):

```
prompts     : the 211 FaceBench attribute names, phrased as captions
E_text      : CLIP text tower, frozen                    -> M in R^{211 x d}
P_text      : orthonormal basis of row-span(M)           -> rank <= 211, computed once
f_c         : P_text f_s          (the nameable part)
f_a         : f_s - f_c           (what cannot be named) -> artifact head
```

**Why this is the most motivated arm in the list.** The deployed OM asks a 32-dimensional MLP
bottleneck to invent a semantic manifold, and §4(b) proves the result adds no information to APP.
This defines the manifold with the object CLIP's pretraining objective actually built — the joint
image-text space — so "off-manifold" becomes a statement about nameability rather than about a
reconstruction error. It is training-free, its rank is set by the vocabulary instead of an arbitrary
latent width, and the residual cannot be a deterministic function of a 32-number bottleneck because
there is no bottleneck.

**Adaptation decision: none, in the first pass.** The whole point is that the manifold comes from
pretraining. If the arm clears its gates, the natural follow-up is LayerNorm-only on the image tower
with `P_text` recomputed, never LoRA — for the §2.3 reason.

**Two things to settle before coding.** The readout must move from the pre-projection CLS to the
**projected joint-space** embedding, or the text span is in the wrong space; and the prompt set needs
an ablation (211 attributes vs a generic face vocabulary vs a random-vector control of equal rank).
**The random-rank control is mandatory** — a rank-211 projection of a 1024-d feature removes a lot of
variance on its own, and without that control a win here is uninterpretable.

**Availability: CLIP today, PE-Core after an install.** `timm` ships PE vision towers only and
`open_clip` is not present in `discern_ext_timm`, so PE-Core's text tower needs Meta's
`perception_models` release. This arm therefore runs on **CLIP** first, which also means it is the one
arm here that does not depend on the backbone swap being right.

---

## 8. Go / no-go, decided before running

Per arm, on the 5 source-grouped dev folds, against the PE-Core-L + LN reference (fused 0.9364):

| | gate | why this number |
|---|---|---|
| **G1 diversity** | `phi(APP,OM)` <= **0.30**, down from 0.402 | below every adapted arm in §5 and at the level the frozen low-correlation readouts reach |
| **G2 both views pay** | `loo_APP > 0.005` **and** `loo_OM > 0.005` in **>= 4 of 5 folds** | the ledger must stop flipping; today one of the pair is always ~0.004 or exactly 0 |
| **G3 no zero-weight folds** | `folds_weight_zero[APP] == 0` and `[OM] == 0` | a view the fit discards is not a view |
| **G4 non-inferior** | fused dev AUROC >= 0.9364 - 0.005 | diversity bought at a real accuracy cost is not progress |

Arm 5 carries one extra, non-negotiable gate: **G0**, that it beats a random projection of equal
rank on dev. Without G0 the arm is not interpretable and does not proceed regardless of G1-G4.

**G1-G3 are the decision. G4 is a guard.** An arm that clears G1-G3 and ties on G4 is a success and
proceeds; an arm that lifts AUROC while failing G2 is the same non-result we already have four times
over and does **not** proceed. Promotion to the 10 test sets happens only after G1-G4 pass together,
and the headline claim is `fused > max_view` on more than 4/10 sets — the oracle comparison in §5,
which the current system fails.

If all five arms fail G1-G3, the honest conclusion is that two spatial views on one frozen
contrastively-pretrained foundation model cannot be made complementary, and the paper should say so
and stand on T+OM as a two-view system. That is a publishable negative and it is cheaper to reach than another operator.

---

## 9. What to change in the code

| arm | file | change |
|---|---|---|
| 1, 4 | `upgrade/extract_layers.py` | already caches all depths; add `--om-depth` to emit the OM-side cache under its own key |
| 1, 2, 3 | `upgrade/adapt_backbone.py` | `build()` currently returns one trunk and one head. Needs a second projection + beta-VAE + orthogonal residual + head; port `_geometric_view_purification` (6 lines) and `BetaVAEWithAlignment` (60 lines) from `DISCERN_Ext/src/modules/beta_vae_aligned.py` rather than re-deriving |
| 2, 3 | same | `--recipe-om {N,L,LN,frozen}` and `--readout-om {last,mid,half}`, logged into the run JSON next to `fused_qkv` |
| all | `upgrade/screen_eval.py:19` | **this is the one that matters.** It replaces only `p_APP`/`u_APP` and reuses OM from the deployed export. Every arm here produces a new OM, so this must take `p_OM`/`u_OM` from the run too, or the arms will score identically to the current ones and look like no-ops |
| all | `upgrade/screen_eval.py:125` | `phi` and `loo` are already computed and already written to the screen JSON. G1-G3 need no new instrumentation, only a gate check |

`screen_eval.py:19` is the trap. It is exactly the class of bug that cost us the DS-R scoring error
and the Part 2a silent substitution: a path that keeps working while quietly reporting a system we
are not running. The first thing any arm should do is assert that its OM standalone AUROC on CDFv2
is **not** 0.9736.

---

## 10. Risks, stated up front

1. **Arm 1 may fail because mid-depth and last-depth features are not that different.** `phi` at
   frozen `mid` is 0.221 against `last`'s 0.212 — those are correlations with the *old* CLIP OM, not
   with each other, so they do not predict the arm. Measuring APP-OM correlation *between two depths
   of one trunk* is the first number Arm 1 produces and it is worth looking at before training
   anything.
2. **The 1.31x upsample is in every PE-Core number in §3 and will be in these too.** It is a
   confound shared by all arms, so it does not affect the comparisons, but it caps what we can claim
   against CLIP in the paper. Re-extracting crops at 336 would remove it and costs a full
   preprocessing pass.
3. **Arm 3 doubles inference cost permanently.** Worth it only if it clears G1-G4 by a margin the
   bootstrap can see.
4. **None of this touches DFEval24.** All three views are at ~0.67 there and this document does not
   change that. It is the largest single gap in the system and it is a data problem.
5. **Arm 5's rank is its own confound.** A rank-211 projection removes a large share of a 1024-d
   feature's variance whatever the prompts say. G0 exists for this and is not optional.
6. **Three of the five transformations in §2.4 need something installed or checked first** —
   PE-Core's text tower, DINOv3's masked-prediction head, and the DINOv3 prefix-token slice. None of
   those is a blocker for Arms 1, 4 or 5, which is why those three go first.
7. **The 2.25x token count makes PE-Core-L adaptation 2-3x slower than CLIP's.** Arms 2 and 3 are
   4-8 GPU-hours each on GPU 1, not 1-2. Arms 1 and 4 are cheap and should go first for that reason
   as much as for the scientific one.

---

## 11. Pointers

| what | where |
|---|---|
| backbone specs, input sizes, readouts, the upsample note | `upgrade/extract_spatial.py:46-64` |
| LoRA / LN / LN recipe construction, `inject_lora`, fused-qkv handling | `upgrade/adapt_backbone.py:57-120` |
| all-depths extraction in one pass, `DEPTHS = {"mid": 2/3, "half": 1/2}` | `upgrade/extract_layers.py:39` |
| **OM's definition** — the six lines of linear algebra | `DISCERN_Ext/src/model/core_model.py:318-336` |
| **OM is deterministic at eval** — `if not self.training: return mu` | `DISCERN_Ext/src/modules/beta_vae_aligned.py:47-49` |
| the 1024 -> 64 projection both views live behind | `DISCERN_Ext/src/encoders/clip_encoder.py:52-55` |
| **the OM-is-reused trap** | `upgrade/screen_eval.py:19` |
| `phi` / `loo` / `folds_weight_zero` computation | `upgrade/screen_eval.py:118-173` |
| every number in §3 | `analysis/upgrade/screen_*.json`, `analysis/upgrade/adapt/*.json`, `analysis/upgrade/final/*_s0.json` |
| the per-view and nuisance profiles | `analysis/upgrade/part2b_profile_{clip,pe_core_l}.json` |
| **I-JEPA predictor weights, already on disk** | `branches/ijepa/IN1K-vit.h.14-300e.pth.tar`, loaded at `branches/ijepa/extract_residuals.py:48` |
| PE-Spatial-L / PE-Lang-L, loadable today | `timm` ids `vit_pe_spatial_large_patch14_448`, `vit_pe_lang_large_patch14_448` |
| CLIP text tower (discarded today) | `DISCERN_Ext/src/encoders/clip_encoder.py:29` loads the full `CLIPModel` and keeps only `.vision_model` |
| the 211-attribute vocabulary for Arm 5's prompts | FaceBench attribute list, `/data/umar/Datasets/preprocessed/*/facellava_semantic/` |
| LoRA mechanics, block anatomy, the Q/V-not-K rationale | `LORA_AND_VIT_ARCHITECTURE.md` |

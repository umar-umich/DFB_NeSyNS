# LoRA and the vision transformers in this system

Written to settle a specific disagreement and to make the code checkable. Every shape below was read
off the instantiated models, not from a paper.

---

## 1. The framing to use in a meeting

> **"We are not updating any weights of the foundation model, we add an adapted layer on top."**

That sentence is **half right, and the half that is wrong is what your colleagues are objecting to.**
They are correct: LoRA is not a head bolted on the end. It reaches *inside* the transformer and
changes what specific internal layers compute.

The accurate statement, which satisfies both sides:

> **"Every pretrained weight is frozen and never receives a gradient. LoRA adds a small trainable
> low-rank matrix pair *in parallel to* chosen internal weight matrices — inside the attention of
> every block — and adds its output to the frozen layer's output. So we change the function that
> internal layers compute, without ever modifying the foundation model's weights. The foundation
> model can be restored exactly by deleting the adapters."**

Both claims in that sentence are verifiable in our code, and section 3 shows where.

The distinction that matters scientifically: **a frozen backbone with LoRA is not the same as a
frozen backbone with a trained head.** Our I-JEPA rows R2/R2b/R3/R4 are the latter — the backbone is
untouched and only a 500k-parameter head trains. The Appearance and temporal branches are the former.
Comparing their numbers directly understates the frozen ones, which is exactly the confound recorded
in `IJEPA_LIMITATIONS_AND_DICOME_COMPARISON.md` §3b.

---

## 2. What a vision transformer block actually contains

All three backbones here are plain ViTs and share one block structure. Using CLIP ViT-L/14's block,
read from the live model:

```
x ──► LayerNorm ──► Multi-Head Self-Attention ──► + x      (residual)
  ──► LayerNorm ──► MLP (fc1 -> GELU -> fc2)     ──► + x      (residual)
```

The attention sublayer is four matrix multiplies, each a `nn.Linear`:

| projection | shape (CLIP ViT-L) | what it does |
|---|---|---|
| `q_proj` | 1024 x 1024 | turns each token into a **query**: "what am I looking for?" |
| `k_proj` | 1024 x 1024 | turns each token into a **key**: "what do I offer?" |
| `v_proj` | 1024 x 1024 | turns each token into a **value**: "what do I contribute if attended to?" |
| `out_proj` | 1024 x 1024 | mixes the attended values back to the model width |

Attention is `softmax(QK^T / sqrt(d)) V`, computed per head — 16 heads of 64 dimensions each for
ViT-L. The MLP that follows is `fc1` 1024 -> 4096, GELU, `fc2` 4096 -> 1024, and holds about two
thirds of the block's parameters.

**The three backbones, measured:**

| model | role here | blocks | width | heads | patch | params |
|---|---|---:|---:|---:|---:|---:|
| CLIP ViT-L/14 | Appearance (and Off-manifold, built from it) | 24 | 1024 | 16 | 14 | 303.2 M |
| V-JEPA 2 ViT-L | temporal video view | 24 | 1024 | 16 | — | ~300 M |
| I-JEPA ViT-H/14 encoder | the residual branch under test | **32** | **1280** | 16 | 14 | **630.8 M** |
| I-JEPA predictor | proposed R6 target | 12 | 384 | — | — | 22.4 M |

Naming differs between implementations and this trips people up when reading the code:

- **HuggingFace CLIP / V-JEPA 2** keep Q, K, V as three separate Linears: `q_proj`, `k_proj`, `v_proj`
  (CLIP) or `query`, `key`, `value` (V-JEPA 2).
- **I-JEPA's own repo** fuses them into one Linear called `qkv` of shape `3*d x d` — 3840 x 1280 in
  the encoder, 1152 x 384 in the predictor — and calls the output projection `proj`.

---

## 3. What LoRA does, exactly

A frozen linear layer computes `y = W x + b` with `W` of shape `d_out x d_in`. LoRA leaves `W` and
`b` alone and adds a parallel path:

```
y = W x + b            (frozen, no gradient ever)
    + B (A (dropout(x))) * (alpha / r)      (trainable, rank r)
```

with `A` of shape `r x d_in` and `B` of shape `d_out x r`. Because `r` is tiny — 8 here — the update
`BA` is a rank-8 matrix of the same shape as `W`, expressed in `r*(d_in + d_out)` numbers instead of
`d_in * d_out`.

Three properties that matter for the meeting question:

1. **`B` is initialised to zero**, so at step 0 the adapter contributes exactly nothing and the model
   is bit-identical to the foundation model. Training moves away from that point.
2. **The base weights never receive a gradient.** In our implementation this is explicit:
   `for p in self.base.parameters(): p.requires_grad_(False)`.
3. **It is inside the block, not on top of it.** LoRA sits on the attention projections of *every*
   block, so it changes attention itself, at every depth. This is the part your colleagues are right
   about.

`alpha / r` is a fixed scale, 16/8 = 2 here, so the adapter's contribution is not tied to `r`.

### The exact arithmetic for our branches

Per adapted Linear: `r * d_in + d_out * r` parameters.

**Appearance — CLIP ViT-L/14, targets `q_proj` and `v_proj`, r = 8:**

```
per projection : 8*1024 + 1024*8            = 16,384
per block      : 2 projections x 16,384     = 32,768
whole encoder  : 24 blocks x 32,768         = 786,432 trainable
against        : 303,200,000 frozen         = 0.26 %
```

**Temporal — V-JEPA 2 ViT-L, targets `query` and `value`, r = 8:** identical arithmetic,
**786,432** trainable. The deployed branch reports **919,938** total = 786,432 LoRA + 133,506 head
(`LayerNorm(1024) -> Linear(1024,128) -> GELU -> Linear(128,2)`), which is the number in the config.

**Proposed R6 — I-JEPA predictor, targets `attn.qkv` and `attn.proj`, r = 8, 12 blocks at width 384:**

```
qkv  : 8*384 + 1152*8 = 3,072 + 9,216       = 12,288
proj : 8*384 + 384*8                        =  6,144
per block                                    = 18,432
12 blocks                                    = 221,184 trainable
against 22.4 M predictor params              = 0.99 %
```

Note `qkv` is one fused Linear, so a single adapter there covers Q, K **and** V — unlike CLIP, where
targeting `q_proj, v_proj` deliberately skips K.

**A hypothetical I-JEPA appearance branch (§3b of the limitations doc), encoder ViT-H/14, 32 blocks
at width 1280:**

```
qkv  : 8*1280 + 3840*8 = 10,240 + 30,720    = 40,960
proj : 8*1280 + 1280*8                      = 20,480
per block                                    = 61,440
32 blocks                                    = 1,966,080 trainable
against 630.8 M frozen                       = 0.31 %
```

### Why only Q and V, and not K or the MLP?

The deployed choice — `q_proj, v_proj` — is the original LoRA paper's recommendation, and it is a
budget decision rather than a law. Q and V are the two that change *what is retrieved*; K mostly
changes *how tokens are matched*, and empirically adds little for the cost. The MLP holds about two
thirds of the parameters and is usually left alone first.

**This is a legitimate place to look for improvement**, and it is cheap to test: adding `k_proj` and
`out_proj` doubles the adapter to ~1.57 M, still 0.5 % of CLIP. Nobody in this project has run that
ablation.

---

## 4. Where this lives in our code

| what | file | what to look at |
|---|---|---|
| CLIP LoRA wiring | `DISCERN_Ext/src/encoders/clip_encoder.py:37-46` | builds `LoraConfig` and calls `get_peft_model` — this is the HuggingFace **peft** library, not our code |
| LoRA hyper-parameters | `DISCERN_Ext/eval_adaptation/configs/warm_dicome_gate_seed42.yaml:22-31` | `target_modules: [q_proj, v_proj]`, `rank: 8`, `alpha: 16`, `dropout: 0.1`, `bias: none` |
| config schema | `DISCERN_Ext/src/config.py:39-63` | `LoraConfig` / `PeftConfig` |
| **our own LoRA** (temporal) | `DFB_NeSyNS/analysis/tbiom/temporal/train_branch.py:64-100` | `LoRALinear` and `inject_lora` — written by hand, not peft |
| which modules get adapted | same file, `LORA_TARGETS` | per model family: V-JEPA 2 -> `{query, value}`, TimeSformer -> `{qkv}`, X-CLIP -> `{q_proj, v_proj}` |
| freezing | same file, `Branch.__init__` | `for p in self.backbone.parameters(): p.requires_grad_(False)` then adapters re-enabled |
| the three views assembled | `DISCERN_Ext/src/model/core_model.py:194-235` | where `semantic_feature` becomes APP and OM |
| the orthogonal artifact branch | `core_model.py:318-336` | `_geometric_view_purification` |
| I-JEPA encoder / predictor | `DFB_NeSyNS/branches/ijepa/ijepa/src/models/vision_transformer.py` | `Block`, `Attention` with the fused `qkv` |
| I-JEPA is currently FROZEN | `DFB_NeSyNS/branches/ijepa/extract_residuals.py:load_models` | `m.eval().requires_grad_(False)` — no adapters anywhere |

**Read `train_branch.py:64-80` first if you only read one thing.** It is twenty lines and it is the
whole mechanism, including the subtlety in the `weight` property: some attention implementations call
`F.linear(x, self.query.weight, bias)` directly instead of invoking the module, so the merged weight
`W + BA * alpha/r` is exposed for those paths. Gradients still reach `A` and `B`; adapter dropout does
not apply on that path. That is a real implementation detail worth knowing before trusting any
"LoRA is applied" claim on a new architecture.

---

## 5. Where improvement is plausible

1. **Widen the adapted set.** Nobody has tested `k_proj` or `out_proj`, or the MLP. Cost is one
   training run per variant and the adapters stay under 1 % of the backbone.
2. **Rank is unexamined.** r = 8 was inherited from the CLIP view and copied to the temporal branch
   without an ablation. r ∈ {4, 16, 32} is cheap to sweep and directly trades capacity against the
   overfitting we measured in G2.
3. **Depth-selective adaptation.** LoRA is applied to *all* blocks uniformly. For a face task, late
   blocks carry semantics and early blocks carry texture; adapting only the last third would be a
   smaller, more targeted intervention and is a one-line change to `inject_lora`.
4. **The frozen-versus-adapted confound is unresolved.** Our I-JEPA rows are frozen-backbone probes
   and the deployed branches are LoRA-adapted. Any comparison between them, including G2's verdict,
   is confounded until I-JEPA gets adapters. This is the single most important gap.
5. **Two views on one trunk.** Appearance and Off-manifold share the CLIP encoder, so LoRA trained
   for one is *the same adapter* the other sees. That is structural, not a hyper-parameter, and it is
   the best explanation available for four failed fusion studies. See §3b of the limitations doc.

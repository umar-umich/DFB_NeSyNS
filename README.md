
# NeSyDeFake: Neuro-Symbolic Deepfake Detection via Per-Branch Causal Discovery

**A hybrid deepfake detection framework combining foundation model features, sparse autoencoders, causal structure learning, and facial semantic attributes for interpretable, cross-dataset-generalizable detection.**

*Last updated: 2026-03-10*

---

## Table of Contents

1. [Framework Overview](#framework-overview)
2. [Architecture](#architecture)
3. [Modules in Detail](#modules-in-detail)
4. [Training Pipeline](#training-pipeline)
5. [Inference Pipeline](#inference-pipeline)
6. [Loss Functions](#loss-functions)
7. [Configuration Reference](#configuration-reference)
8. [Key Files](#key-files)

---

## Framework Overview

### The Intuition

Imagine a forensic analyst examining a face photo for forgery. They use two complementary tools:

1. **A magnifying glass** (spatial branch) — examines visual details: skin texture, edge sharpness, colour blending
2. **A UV light** (frequency branch) — reveals invisible spectral patterns: compression artifacts, frequency signatures invisible to the naked eye

For each tool, the analyst maintains two mental models:
- **"How real faces behave"** — e.g., when someone smiles, their cheeks always raise and eyes narrow (biomechanical constraints)
- **"How fakes behave"** — e.g., there's always a correlation between skin smoothness and a specific frequency artifact

To determine if a face is fake, the analyst checks: Does it **violate** real-face rules? Does it **match** fake-face patterns? Four scores — spatial violations, spatial fake-patterns, frequency violations, frequency fake-patterns — combine for the final decision.

Additionally, the analyst has a **checklist of 211 facial attributes** (hair colour, expression, action units, accessories...) extracted by an independent face analysis model. A **learned gate** selectively downweights dataset-specific attributes (background, lighting) while preserving universal facial semantics, improving cross-dataset generalization.

### Why This Works

- **Latent branches** (spatial + frequency SAE features) discover statistical patterns humans might miss
- **Semantic branch** (211 named FaceBench attributes) provides explanations — when the system flags a fake, the causal graph shows *why*: "the relationship between `smiling` and `AU6_cheek_raise` present in real faces is broken"
- **Per-branch causal graphs** reveal exactly what each generator breaks in spatial vs frequency domains
- **Source-paired training** forces artifact learning instead of identity shortcuts
- **Uniformity-Alignment loss** prevents representation collapse on the hypersphere

---

## Architecture

```
Input: face frame (224×224×3)
  │
  ├─ spatial normalization ─→ SpatialExtractor (frozen CLIP-ViT-L/14) ─→ raw_spatial (B, 1024)
  │                               │
  │                               ├─→ spatial_proj (1024→1024+LN+GELU) ───────────────┐
  │                               └─→ SAE.spatial (1024→4096 sparse, BatchTopK k=128) ─┤
  │                                                                                     │
  ├─ freq normalization ───→ FreqExtractor (FAD + frozen CLIP-ViT-L/14) ─→ raw_freq (B, 1024)
  │                               │                                                     │
  │                               ├─→ freq_proj (1024→1024+LN+GELU) ──────────────────┤
  │                               └─→ SAE.freq (1024→4096 sparse, BatchTopK k=128) ───┤
  │                                                                                     │
  │                      MultiModalFusion (attention, 2048→1024) ◄──────────────────────┘
  │                               │
  │                               ▼ fused (B, 1024)
  │
  ├─ raw frame ────────→ FacialSemanticExtractor (FaceBench Face-LLaVA-v1.5-13B, frozen)
  │                               │
  │                               ▼ semantic_attrs (B, 211) named P(present) probabilities
  │                               │
  │                      Semantic Gate: sigmoid(learnable_params) × semantic_attrs
  │                               │
  │              ┌────────────────┴────────────────────────────────┐
  │              │         CausalDiscoveryModule                   │
  │              │                                                 │
  │              │  Spatial branch (d=339):                        │
  │              │    spatial_selector(4096→128)                   │
  │              │    x = [z_spatial_128, gated_s_211]             │
  │              │    SCM_spatial_real → residuals, v_spatial_real  │
  │              │    SCM_spatial_fake → residuals, v_spatial_fake  │
  │              │                                                 │
  │              │  Frequency branch (d=339):                      │
  │              │    freq_selector(4096→128)                      │
  │              │    x = [z_freq_128, gated_s_211]                │
  │              │    SCM_freq_real → residuals, v_freq_real       │
  │              │    SCM_freq_fake → residuals, v_freq_fake       │
  │              └─────────────────┬───────────────────────────────┘
  │                                │
  │              4 zero-init violation projections (339→1024 each)
  │                                │
  │              classifier_input = fused + Δ_spatial_real + Δ_spatial_fake
  │                                      + Δ_freq_real + Δ_freq_fake
  │                                │
  │              L2 normalize → l2_embeddings (for UA loss)
  │                                │
  └──────────────────────────→ MultiTaskHead
                                   │
                                   ├─→ classification (B, 2) — real/fake logits
                                   ├─→ uncertainty (B, 1) — prediction confidence
                                   └─→ violation_score (B, 1)
```

---

## Modules in Detail

### M1: Foundation Model Feature Extractors

**Spatial branch** (`SpatialFeatureExtractor`): Frozen CLIP-ViT-L/14 (`openai/clip-vit-large-patch14`). Extracts 1024-d visual features. Backbone fully frozen; LayerNorms unfrozen in Phase 2 for GenD-style adaptation.

**Frequency branch** (`FrequencyFeatureExtractor`): FAD (Frequency-Aware Decomposition) front-end + frozen CLIP-ViT-L/14. The FAD front-end (DCT filters, learnable emphasis weights) is always trainable — it decomposes spatial input into frequency components before CLIP encoding. Outputs 1024-d features.

Both branches produce features that feed into (a) projection heads for fusion, and (b) the Sparse Autoencoder for causal discovery.

### M2: Sparse Autoencoder (SAE)

**Class:** `DualBranchSparseAutoencoder`

Per-branch BatchTopK SAE (Bussmann et al., 2024):
- Input: 1024-d raw CLIP features (pre-projection)
- Dictionary size: 4096 per branch (4× expansion)
- Target active features: k=128 (~3% sparsity)
- BatchTopK: selects top (k × batch_size) activations across the batch — no L1 coefficient tuning needed
- Loss: variance-normalized MSE + auxiliary dead-feature loss (coefficient 0.2)
- Dead feature window: 200 batches
- Input normalization: LayerNorm on CLIP features before SAE
- Decoder weights: unit-norm columns, enforced after each optimizer step

### M3: Facial Semantic Extractor (FaceBench Face-LLaVA)

**Class:** `FacialSemanticExtractor`

Extracts 211 named facial attributes from an independent pretrained face model (Face-LLaVA-v1.5-13B, Wang et al., CVPR 2025). These provide genuinely new semantic information orthogonal to the CLIP spatial/frequency features.

**Teacher-forced single-pass extraction:**

```
Image → CLIP-ViT-L@336 vision tower → 576 patch tokens (1024-d)
      → mm_projector (mlp2x_gelu) → 576 visual tokens (5120-d)
      → Construct teacher-forced prompt with all 211 attributes answered "1"
      → Single LLM forward pass (no autoregressive generation)
      → At each answer position: P(present) = sigmoid(logit("1") - logit("0"))
      → Output: (B, 211) continuous attribute probabilities
```

**211 attributes across 6 views:**

| View | Count | Examples |
|------|-------|---------|
| Appearance | 111 | `black_hair`, `wrinkled_skin`, `high_cheekbones`, `oval_face` |
| Accessories | 30 | `eyeglasses`, `hat`, `necklace`, `face_mask` |
| Makeup | 13 | `heavy_makeup`, `lipstick`, `eyeliner` |
| Surrounding | 12 | `indoor_background`, `bright_lighting`, `blurry_image` |
| Psychology/Expression | 33 | `happy`, `neutral_expression`, `AU12_lip_corner_puller` |
| Identity | 7 | `male`, `female`, `east_asian` |

### Semantic Feature Gate

**Trainable sigmoid gate** (211-d) applied to semantic attributes before causal discovery. Initialized to `sigmoid(0) = 0.5` (neutral). Learns to downweight dataset-specific attributes (e.g., `indoor_background`, `bright_lighting`) while preserving universal facial semantics (e.g., `AU6_cheek_raise`, `smooth_skin`). Improves cross-dataset generalization by preventing the causal module from overfitting to training-set-specific correlations.

### M4: Per-Branch Dual-Graph Causal Discovery

**Class:** `CausalDiscoveryModule`

Four DAGMA-DCE causal graphs — two per branch (real/fake):

```
Spatial branch (d = 128 + 211 = 339 nodes):
  A_spatial_real (339×339): causal structure in REAL faces
    e.g., smooth_skin → no_wrinkles, smiling → high_cheekbones
  A_spatial_fake (339×339): causal structure in FAKE faces
    e.g., blending_artifact ↔ skin_texture_mismatch

Frequency branch (d = 128 + 211 = 339 nodes):
  A_freq_real (339×339): frequency-semantic causality in REAL faces
    e.g., natural_high_freq → hair_texture
  A_freq_fake (339×339): frequency-semantic causality in FAKE faces
    e.g., GAN_spectral_peak ↔ face_region
```

**Per-branch feature selectors:** Each branch has a bias-free linear projection (`SparseFeatureSelector`) that maps 4096 sparse SAE features to 128 dense causally-relevant features. The causal graph discovers relationships between these 128 latent features and the 211 named semantic attributes.

**Interpretable node names:** Every node has a human-readable name:
- Nodes `[0:128]`: `z_spatial_0` .. `z_spatial_127` (data-driven SAE features)
- Nodes `[128:339]`: `black_hair`, `blonde_hair`, ..., `south_asian` (211 FaceBench attributes)

**Detection signal — four violation/conformance scores:**

| Score | Meaning |
|-------|---------|
| `v_spatial_real` | Spatial violation of real biomechanics (high for fakes) |
| `v_spatial_fake` | Spatial conformance to fake artifact patterns (high for fakes) |
| `v_freq_real` | Frequency violation of real spectral structure (high for fakes) |
| `v_freq_fake` | Frequency conformance to fake spectral artifacts (high for fakes) |

**Graph divergence for interpretability:**
```python
divergence = model.causal_module.get_graph_divergence('spatial')
# divergence['broken_by_fakes']  = edges in real but missing in fakes
# divergence['created_by_fakes'] = edges only in fakes (artificial correlations)
```

### M5: Multi-Task Classifier

**Class:** `MultiTaskHead`

Input: 1024-d (fused features + violation projections). Hidden layers: [512, 256], dropout 0.4.

| Task | Type | Output |
|------|------|--------|
| Classification | Binary CE | (B, 2) real/fake logits |
| Uncertainty | Regression (MSE) | (B, 1) prediction confidence |
| Violation score | Regression | (B, 1) causal violation magnitude |

### MultiModal Fusion

**Class:** `MultiModalFusion`

Attention-based fusion of spatial (1024-d) and frequency (1024-d) projected features. Concatenates to 2048-d, projects to 1024-d fused representation with dropout 0.3.

---

## Training Pipeline

### Source-Paired Training (GenD, WACV 2026)

Each batch contains source-matched real-fake pairs from the same video. For FF++ fake video `802_885`, the paired real frame comes from source video `802`. This forces the model to learn manipulation artifacts rather than identity or background shortcuts.

- `__getitem__` returns `{"real": real_sample, "fake": fake_sample}`
- Collator interleaves: `[real_0, fake_0, real_1, fake_1, ...]`
- Batch size N pairs → 2N frames in forward pass (batch_size=32 → 64 frames)
- Non-FF++ datasets fall back to random real pairing

### Two Training Phases

**Phase 1 — Hard Freeze Backbone [epochs 0–5]:**
- CLIP backbones fully frozen (including LayerNorms)
- All other modules train jointly: FAD frontend, projection heads, fusion, classifier, SAE, causal module, semantic extractor, semantic gate
- Loss warm-up active: causal/contrastive/sparse weights ramp from 0.01

**Phase 2 — Full End-to-End [epochs 5–50]:**
- Backbone LayerNorms unfrozen (GenD-style LN-only adaptation)
- All modules continue joint training
- Loss weights reach full values by epoch 10

### Loss Weight Warm-Up Schedule

```
                     epoch 0     epoch 5     epoch 10    epoch 50
classification:      1.0         1.0         1.0         1.0
uncertainty:         0.5         0.5         0.5         0.5
causal (real):       0.01  ───────────────→  0.3         0.3
causal (fake):       0.005 ───────────────→  0.15        0.15
contrastive (real):  0.01  ───────────────→  0.3         0.3
contrastive (fake):  0.005 ───────────────→  0.15        0.15
sparse (SAE):        0.01  ──────→  0.1      0.1         0.1
```

### Causal Warmup

At training start, 50 batches of data pre-populate the EMA adjacency buffers in the causal module before main training begins.

### Optimizer

Adam with per-module learning rate groups:

| Group | LR | Modules |
|-------|-----|---------|
| backbone_layernorms | 1e-4 | CLIP LayerNorm params (Phase 2 only) |
| always_trainable | 1e-4 | FAD front-end params |
| phase_proj | 3e-4 | Frequency phase projection |
| projection_heads | 1e-4 | spatial_proj, freq_proj |
| fusion | 2e-4 | MultiModalFusion |
| classifier | 2e-4 | MultiTaskHead |
| optional_modules | 1e-4 | causal_module, SAE, violation projections, semantic extractor, semantic gate |

Weight decay: 1e-4. Scheduler: cosine with 2-epoch linear warmup.

### Training Forward Pass

```python
# 1. Extract raw branch features (frozen CLIP backbones)
raw_spatial = spatial_extractor(spatial_frames)      # (B, 1024)
raw_freq    = frequency_extractor(freq_frames)       # (B, 1024)

# 2. Project and fuse for classifier
fused = fusion(spatial_proj(raw_spatial),
               freq_proj(raw_freq))                  # (B, 1024)

# 3. SAE (parallel path, operates on raw pre-projection features)
z_spatial, z_freq, sae_loss, sae_info = sparse_ae(
    spatial_feat=raw_spatial,
    frequency_feat=raw_freq)                         # z: (B, 4096) sparse

# 4. Semantic attributes (frozen FaceBench LLM, teacher-forced)
semantic_attrs = semantic_extractor(raw_frames)      # (B, 211)

# 5. Semantic gate (learnable soft selection)
gate = sigmoid(semantic_gate)                        # (211,)
gated_attrs = semantic_attrs * gate                  # (B, 211)

# 6. Per-branch causal discovery (4 DAGMA-DCE graphs)
causal_out = causal_module(
    z_spatial, z_freq, gated_attrs, label)

# 7. Violation fusion + classification
classifier_input = (fused
    + violation_proj_spatial_real(residuals_spatial_real)
    + violation_proj_spatial_fake(residuals_spatial_fake)
    + violation_proj_freq_real(residuals_freq_real)
    + violation_proj_freq_fake(residuals_freq_fake))

# 8. L2 normalize for UA loss (classify on un-normalized features)
l2_embeddings = F.normalize(classifier_input, p=2, dim=1)
task_outputs = classifier(classifier_input)
```

### Mixed Precision & Gradient Clipping

- AMP autocast + GradScaler for bfloat16 training
- Global gradient clipping at max_norm=1.0
- Per-branch frequency extractor clipping at max_norm=2.0
- NaN/Inf gradient detection: zeros offending gradients, GradScaler skips step

---

## Inference Pipeline

Same architecture as training with these differences:
- `label=None`: no Jacobian computation, uses EMA adjacency matrices directly
- No loss computation or backpropagation
- SAE uses learned threshold instead of batch TopK
- Semantic extractor runs normally (frozen LLM forward pass)
- Semantic gate applies learned weights
- Causal violation scores computed from EMA graphs

Output: `prob` (B,) — probability of being fake, plus per-branch violation scores and causal graph divergence for interpretability.

---

## Loss Functions

### Total Loss

```
L_total = w_cls           × L_classification
        + w_unc           × L_uncertainty
        + w_causal        × (L_structural_spatial_real + L_structural_freq_real + L_dag_real)
        + w_causal_fake   × (L_structural_spatial_fake + L_structural_freq_fake + L_dag_fake)
        + w_contrastive   × (L_contrastive_spatial_real + L_contrastive_freq_real)
        + w_contr_fake    × (L_contrastive_spatial_fake + L_contrastive_freq_fake)
        + w_sparse        × L_sae
        + α               × L_alignment
        + β               × L_uniformity
```

### Individual Losses

**Classification** (`L_cls`): Cross-entropy with class weights [1.0, 1.0].

**Uncertainty** (`L_unc`): MSE between predicted uncertainty and (1 - correctness). Teaches the model to be uncertain when wrong.

**Structural causal** (`L_structural`): Per-branch SCM reconstruction loss. Real SCMs must reconstruct real faces well (low residuals on reals); fake SCMs must reconstruct fakes well.

**DAG penalty**: DAGMA acyclicity constraint on adjacency matrices. Ensures learned causal graphs are DAGs.

**Contrastive** (`L_contrastive`): Margin loss on violation scores. Violation scores should be high for fakes, low for reals, with margin ≥ 1.0.

**SAE** (`L_sae`): Variance-normalized MSE reconstruction + dead feature auxiliary loss.

**Alignment** (`L_align`, α=0.1): Pulls same-class features together on the unit hypersphere.
```
L_align = E_{x,y same class} [||x - y||²]
```

**Uniformity** (`L_uniform`, β=0.5): Spreads all features evenly on the hypersphere, preventing representation collapse.
```
L_uniform = log E_{x,y} [e^{-2||x-y||²}]
```

---

## Configuration Reference

Key settings from `training/config/detector/nesy_defake.yaml`:

```yaml
# Data
dataset_type: nesydefake
train_dataset: [FaceForensics++]
test_dataset:  [FaceForensics++, Celeb-DF-v2]
resolution: 224
paired_training: true              # GenD source-paired real-fake batches
train_batchSize: 32                # 32 pairs = 64 frames per batch
balance_target_ratio: 0.5          # 1:1 real:fake (handled by pairing)

# Foundation models
foundation_models:
  spatial:  {model: clip-vit-large-patch14, dim: 1024, freeze: true, train_ln: false}
  frequency: {model: clip-vit-large-patch14 + FAD, dim: 1024, freeze: true, train_ln: false}

# Semantic attributes
semantic_attributes:
  backend: face_llava
  output_dim: 211
  use_llm: true                    # full 13B teacher-forced extraction
  micro_batch_size: 4

# Semantic gate
semantic_gate:
  enabled: true                    # learnable per-attribute gating

# Uniformity-Alignment (GenD recipe)
uniformity_alignment:
  enabled: true
  alignment_weight: 0.1            # α
  uniformity_weight: 0.5           # β
  l2_normalize: true

# SAE
sparse_features:
  dict_size: 4096
  target_k: 128
  dead_feature_window: 200
  aux_loss_coeff: 0.2

# Causal
causal_module:
  causal_warmup_batches: 50
  latent_variables: {z_spatial_dim: 128, z_frequency_dim: 128}
  semantic_dim: 211
  discovery: {algorithm: dagma_dce, max_parents: 3, sparsity_penalty: 0.01}

# Training
nEpochs: 50
optimizer: adam (lr=1e-4, weight_decay=1e-4)
lr_scheduler: cosine_warmup (warmup_epochs=2)
grad_clip: 1.0
mixed_precision: true

# Phases
phase1: [0, 5]   — backbone frozen, all other modules train
phase2: [5, 50]  — LayerNorms unfrozen, full end-to-end

# Loss warmup
causal:       0.01 → 0.3  over epochs [0, 10]
contrastive:  0.01 → 0.3  over epochs [0, 10]
sparse:       0.01 → 0.1  over epochs [0, 5]
```

---

## Key Files

| File | Description |
|------|-------------|
| `training/config/detector/nesy_defake.yaml` | All hyperparameters and module configuration |
| `training/detectors/nesy_defake_detector.py` | Main detector: forward pass, loss computation, metrics |
| `training/networks/nesy_defake/semantic/facial_semantic_extractor.py` | FaceBench 211-attribute extractor (vision-only + full LLM modes) |
| `training/networks/nesy_defake/causal/causal_discovery.py` | Per-branch DAGMA-DCE causal discovery (4 graphs) |
| `training/networks/nesy_defake/classifiers/sparse_autoencoder.py` | Dual-branch BatchTopK SAE |
| `training/networks/nesy_defake/classifiers/multitask_head.py` | Multi-task classification head |
| `training/networks/nesy_defake/fusion/multimodal_fusion.py` | Attention-based spatial-frequency fusion |
| `training/networks/nesy_defake/foundation_models/` | CLIP spatial and FAD-CLIP frequency extractors |
| `training/dataset/nesy_defake_dataset.py` | Dataset with source-paired training, collation |
| `training/trainer/trainer.py` | Training loop, phase transitions, mixed precision |
| `training/train.py` | Entry point, optimizer construction with per-module LR groups |

---

## Running

```bash
# Single GPU
python training/train.py \
    --detector_path training/config/detector/nesy_defake.yaml

# Multi-GPU (DDP)
torchrun --nproc_per_node=4 training/train.py \
    --detector_path training/config/detector/nesy_defake.yaml \
    --ddp
```

## Key References

- **GenD** (WACV 2026): Source-paired training, LN-only backbone adaptation, uniformity-alignment loss
- **FaceBench** (CVPR 2025): Face-LLaVA 211 facial attributes, teacher-forced extraction
- **BatchTopK SAE** (Bussmann et al., 2024): Direct sparsity control without L1 tuning
- **DAGMA-DCE** (Bello et al., 2022): Differentiable causal structure learning with acyclicity constraint
- **Wang & Isola (2020)**: Uniformity and alignment on the hypersphere

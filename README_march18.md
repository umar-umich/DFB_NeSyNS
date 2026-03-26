# NeSyDeFake: Neuro-Symbolic Deepfake Detection via Causal Graph Discovery

## Motivation

Current deepfake detectors rely on purely neural approaches (CNNs, Vision Transformers) that learn opaque, dataset-specific features. This leads to three critical problems:

1. **Poor cross-dataset generalization** — detectors trained on FaceForensics++ fail on Celeb-DF because they memorize compression artifacts and dataset-specific patterns rather than learning forensically meaningful features.

2. **No interpretability** — when a detector flags a face as fake, it cannot explain *why*. This limits trust and deployability in forensic, legal, and journalistic settings.

3. **Vulnerability to novel generators** — purely neural features overfit to the training distribution. A detector trained on Deepfakes/Face2Face fails on unseen generators because it never learned the *causal structure* of why fakes differ from reals.

NeSyDeFake addresses all three by combining neural feature extraction with symbolic causal reasoning. The key insight: **real and fake faces have fundamentally different causal structures** — the physical constraints that govern real faces (muscle groups, bone structure, lighting physics) are broken or distorted by generation pipelines. By learning and comparing these causal structures explicitly, the detector gains generalization, interpretability, and robustness.

## Framework Architecture

```
                                    NeSyDeFake Pipeline
 ┌──────────────────────────────────────────────────────────────────────────┐
 │                                                                          │
 │  Input Frame                                                             │
 │      │                                                                   │
 │      ├──► Spatial Extractor (CLIP-ViT-L/14) ──► spatial features (1024) │
 │      │         [backbone frozen, LayerNorms train — GenD regime]         │
 │      │                                                                   │
 │      ├──► Frequency Extractor (FAD-CLIP) ──────► freq features (1024)   │
 │      │         [shared CLIP backbone, FAD preprocessing]                 │
 │      │                                                                   │
 │      └──► Semantic Extractor (FaceBench) ──────► 211 face attributes    │
 │                │                                                         │
 │                ├──► Tier 1: Consistency Rules ──► 18 violation scores    │
 │                │         [deterministic, no parameters]                  │
 │                │                                                         │
 │                └──► Tier 2: Forensic Features ─► 30 pixel-level metrics │
 │                          [precomputed via BiSeNet face parsing]          │
 │                                                                          │
 │  ──── Neural Path ────────────────────────────────────────────────────── │
 │                                                                          │
 │  spatial(1024) + freq(1024)                                              │
 │      │                                                                   │
 │      └──► Attention Fusion ──► fused_features (1024)                    │
 │                                      │                                   │
 │  ──── Symbolic Causal Path ──────────┼───────────────────────────────── │
 │                                      │                                   │
 │  z_spatial ─┐                        │                                   │
 │  z_freq ────┤                        │                                   │
 │  forensic ──┘                        │                                   │
 │      │                               │                                   │
 │      └──► 4 Causal Graphs ──► residuals ──► Causal Attention Fusion     │
 │           (DAGMA-DCE)              │              │                      │
 │           spatial_real             │    causal_delta (1024)              │
 │           spatial_fake             │         │                           │
 │           freq_real                │    ┌────┘  sigmoid(causal_gate)     │
 │           freq_fake                │    │       [init -3.0, ~0.05]       │
 │                                    │    ▼                                │
 │  ──── Neuro-Symbolic Fusion ───────┼────────────────────────────────── │
 │                                    │                                     │
 │      fused_features + gate * causal_delta                                │
 │              │                                                           │
 │              ├──► Tier 3: Causal Intervention (28 features)             │
 │              │         [cross-graph scoring + do-calculus]               │
 │              │         + sigmoid(ci_gate) * ci_projection               │
 │              │                                                           │
 │              └──► Classifier ──► P(fake)                                │
 │                                                                          │
 └──────────────────────────────────────────────────────────────────────────┘
```

## Module Details

### 1. Dual-Stream Neural Backbone

**Spatial stream** (CLIP-ViT-L/14): Extracts high-level semantic and structural features from face crops. The backbone is frozen with only LayerNorm parameters trainable — the GenD training regime that preserves pretrained representations while allowing distribution adaptation.

**Frequency stream** (FAD-CLIP): The same CLIP backbone processes frequency-augmented input (via Frequency-Aware Decomposition). Captures spectral artifacts invisible in the spatial domain — GAN fingerprints, blending boundary frequencies, upsampling patterns.

Both streams share a single CLIP backbone (saving ~600MB VRAM) via batched forward pass.

### 2. Semantic Feature Hierarchy

#### Tier 0: FaceBench Attributes (211 dimensions)
Precomputed via Face-LLaVA-v1.5-13B — a vision-language model fine-tuned for face understanding. Covers 6 categories:

| Category | Count | Examples |
|----------|-------|---------|
| Appearance | 105 | hair color, face shape, skin texture, age markers |
| Accessories | 30 | glasses type, hats, piercings |
| Makeup | 13 | lipstick color, eye makeup, foundation |
| Surrounding | 12 | indoor/outdoor, lighting conditions |
| Psychology | 33 | 8 basic expressions + 25 FACS Action Units |
| Identity | 7 | gender, ethnicity |

A trainable **semantic gate** (211-d sigmoid) learns which attributes carry cross-dataset generalization signal vs. dataset-specific noise.

#### Tier 1: Cross-Attribute Consistency Rules (18 dimensions)
Deterministic, parameter-free rules that detect physically impossible attribute combinations. These are signals that generation pipelines frequently produce but real faces cannot exhibit:

| Rule Type | Count | Example |
|-----------|-------|---------|
| Gender-attribute violations | 4 | beard on female face, heavy makeup on male |
| Mutually exclusive conflicts | 4 | mouth simultaneously open and closed |
| Contradictory attributes | 1 | bald with long hair |
| Age-appearance inconsistency | 2 | elderly face with smooth skin |
| Expression-AU violations | 5 | "happy" expression without Duchenne smile muscles (AU6+AU12) |
| Symmetry/quality conflicts | 2 | simultaneously blurry and sharp |

**Why this matters for deepfakes**: Generation models predict attributes independently, while real faces have *causal* constraints between attributes. A GAN can easily produce a face labeled "happy" without activating the correct Action Units, because it never learned facial muscle anatomy — it only learned pixel correlations. These consistency rules catch exactly that kind of causal violation.

#### Tier 2: Pixel-Level Forensic Features (30 dimensions)
Precomputed via BiSeNet face parsing — classical computer vision metrics that quantify manipulation artifacts at face region boundaries:

| Feature Group | Count | What It Measures |
|---------------|-------|-----------------|
| Boundary gradients | 6 | Sobel edge magnitude at skin/eye/lip/nose boundaries |
| Regional blur | 6 | Laplacian variance per face region + cross-region ratios |
| Left-right symmetry | 4 | Intensity and landmark asymmetry (eyes, mouth, cheeks, jaw) |
| Color consistency | 4 | Chi-squared histogram distance between adjacent regions |
| Frequency anomaly | 6 | DCT high-frequency energy per region + cross-region ratios |
| Quality metrics | 4 | Anti-spoof score, detection confidence, blendshape symmetry, landmark jitter |

**Why this matters**: Deepfakes manipulate face regions independently, creating subtle discontinuities at boundaries. Real faces have smooth, physically consistent transitions. These features capture that signal without any learned parameters.

### 3. Per-Branch Dual-Graph Causal Discovery (DAGMA-DCE)

This is the core neuro-symbolic component. It learns **4 separate causal graphs** — one for each combination of branch and distribution:

| Graph | Learns | Purpose |
|-------|--------|---------|
| Spatial-Real | Causal structure of real faces in spatial domain | Baseline: how facial attributes causally relate in nature |
| Spatial-Fake | Causal structure of fake faces in spatial domain | Captures: which causal links generators break |
| Freq-Real | Causal structure of real faces in frequency domain | Baseline: natural spectral relationships |
| Freq-Fake | Causal structure of fake faces in frequency domain | Captures: spectral artifacts of generation |

#### How Causal Graph Learning Works

Each graph is a **Structural Causal Model (SCM)** — a directed acyclic graph (DAG) where edges represent direct causal effects between variables.

**Node composition** (per graph):
- 16 compressed visual features (from backbone)
- 18 consistency rule scores (Tier 1)
- 30 forensic features (Tier 2)
- **Total: 64 interpretable, forensically relevant nodes**

**Learning algorithm: DAGMA-DCE**

The SCM is a linear model: `x_hat = W @ x + b`, where the weight matrix W directly encodes causal relationships. The adjacency matrix is `A = |W|` with diagonal zeroed (no self-loops).

```
Given: x = [z_visual; consistency_rules; forensic_features]  (64-d per sample)

Step 1: Forward pass through SCM
        x_hat = W @ x + b                    (reconstruct each variable from its causes)

Step 2: Compute residuals
        residual = x - x_hat                 (what the causal model can't explain)

Step 3: Extract adjacency
        A = |W|, zero diagonal               (edge weights = causal effect magnitudes)

Step 4: Enforce DAG constraint (DAGMA, Bello et al. NeurIPS 2022)
        h(A) = -log det(sI - A*A) + d*log(s)
        h(A) = 0  iff  A is a DAG            (penalizes cycles during training)

Step 5: EMA stabilization
        A_ema = 0.99 * A_ema + 0.01 * A      (smooth graph estimates for stability)
```

**Why linear SCMs?** Linear SCMs have stronger identifiability guarantees (Peters et al., 2014). The Jacobian of a linear model IS the weight matrix — no approximation needed. This means the discovered causal graph is *exact*, not a noisy EMA estimate. For structure discovery, this precision matters more than the expressiveness of nonlinear models.

**Key insight — the detection signal is in the residuals**:
- When a *real* face passes through the *real SCM*: small residuals (the model explains the data well)
- When a *fake* face passes through the *real SCM*: large residuals (the causal structure doesn't match)
- The pattern reverses for the fake SCM
- These **residual patterns** are what distinguish real from fake, and they generalize across datasets because they capture structural violations, not surface statistics

#### What the Causal Graphs Capture

**Real-face graph** (learned from authentic faces):
- Strong edges: `happy → AU6_cheek_raise`, `happy → AU12_lip_corner_puller` (Duchenne smile musculature)
- Strong edges: `male → beard_probability`, `elderly → wrinkled_skin`
- These reflect *physical causation* — muscle groups, hormonal effects, aging processes

**Fake-face graph** (learned from deepfakes):
- Broken Duchenne pathway: `happy → AU6` edge is weak or absent (generators produce smiles without correct muscle activations)
- Spurious edges: `blur_skin → blur_eye` might be strong (generators apply uniform blur, while real faces have depth-dependent blur)
- Boundary artifacts: strong `ff_grad_skin_bg → ff_grad_eye_skin` (generators create correlated boundary artifacts across regions)

**Graph divergence** = where these two graphs differ = where fakes break natural causal structure.

### 4. Causal Violation Attention Fusion

The 4 residual vectors (one per graph) carry the detection signal, but not all residual types are equally informative for every sample. The attention fusion mechanism dynamically selects which violations matter most:

```
Query:  fused_features (what does the neural backbone see?)
Keys:   4 residual types (which causal violations exist?)
Values: 4 residual types projected to classifier dimension

attention_weights = softmax(Q @ K^T / sqrt(d))    → (B, 4) per-sample weights
causal_delta = sum(attention_weights * Values)     → (B, 1024) weighted violation signal
```

A **learnable causal gate** (scalar, initialized at sigmoid(-3.0) ~ 0.05) controls how much the causal signal influences classification. It starts nearly muted and opens as the causal module learns meaningful graphs. This prevents noisy early-training causal signals from destabilizing the classifier.

```
classifier_input = fused_features + sigmoid(causal_gate) * causal_delta
```

### 5. Tier 3: Causal Intervention Module (Neuro-Symbolic Reasoning)

This module performs **do-calculus interventions** — the defining operation of causal reasoning. It asks: *"If we force-change attribute X, does the rest of the face respond as physics predicts?"*

Three types of features are computed (28 total):

#### A. Cross-Graph Consistency Scoring (8 features)
For each branch (spatial, frequency), compare how well each sample fits the real vs fake SCM:
- **Real fit**: reconstruction error under the real SCM (low = matches real causal structure)
- **Fake fit**: reconstruction error under the fake SCM (low = matches fake causal structure)
- **Consistency score**: `real_fit - fake_fit` (positive = more real-like structure)
- **Z-semantic ratio**: how much of the error is in visual features vs forensic features

These require **zero extra SCM forwards** — they use residuals already computed by the causal module.

#### B. Do-Calculus Intervention Statistics (16 features)
For each of the 4 graphs, select the top-k most important nodes (blending attribute confidence with graph structural importance — out-degree from the learned adjacency matrix) and perform counterfactual interventions:

```
For each important semantic node:
    1. Record baseline:     x_hat = SCM(x)
    2. Intervene (do-calc): flip the attribute value (e.g., force "happy" → "not happy")
    3. Observe effect:      x_hat' = SCM(x_intervened)
    4. Measure discrepancy: disc = ||x_hat' - x_hat||

Aggregate into 4 statistics per graph:
    - Mean discrepancy:     average response magnitude
    - Max discrepancy:      largest single-attribute response
    - Entropy:              how spread out responses are
    - Structural anomaly:   coefficient of variation (std/mean)
```

**Interpretation**:
- **Real faces**: Predictable causal cascades. Flipping "happy" predictably changes AU6, AU12, and related muscle groups. Low structural anomaly.
- **Fake faces**: Anomalous responses. Flipping "happy" has inconsistent effects because the generator never learned the causal structure of facial expressions. High structural anomaly, high entropy.

#### C. Differential Intervention Response (4 features)
For the same intervention, compare how the real SCM and fake SCM respond:
- Real faces should produce similar responses under both SCMs (the face is consistent)
- Fake faces produce *different* responses (they fit the fake SCM but not the real one)

```
differential = |disc_real_SCM - disc_fake_SCM|     per intervention
Features: mean and max differential per branch
```

All 28 features are projected through a trainable layer with its own gate (`ci_gate`, initialized at sigmoid(-2.0) ~ 0.12) into the classifier.

## Training Details

### Dataset
- **Training**: FaceForensics++ (c23 compression) with source-paired real-fake training (GenD, WACV 2026)
  - Real frames augmented 3x to balance the 4:1 fake:real class ratio in FF++
- **Testing**: FaceForensics++ (in-dataset) + Celeb-DF-v2 (cross-dataset generalization)

### Training Regime
All modules train end-to-end from epoch 0 with loss weight warmup:

- **Backbone**: Frozen CLIP-ViT-L/14 with only LayerNorm parameters trainable (GenD regime — preserves pretrained representations, enables domain adaptation)
- **Causal warmup**: Before epoch 0, pre-populate EMA buffers by running real and fake samples through the SCMs. This gives the causal module stable graph estimates from the start.
- **Causal gate**: Initialized at -3.0 (sigmoid ~ 0.05), naturally suppresses causal contribution until the module learns meaningful structure. No explicit phase scheduling needed.
- **Loss warmup**: Causal loss weights ramp up over the first 15 epochs, preventing the classifier from being dominated by noisy early causal signals.

### Loss Functions

| Loss | Weight | Purpose |
|------|--------|---------|
| Cross-entropy (class-weighted) | 1.0 | Primary classification objective |
| Uniformity-alignment (GenD recipe) | 0.1 / 0.5 | L2-normalized embeddings on hypersphere for generalization |
| Causal structural (real) | 0.01 → 0.1 | SCM reconstruction quality on real faces |
| Causal structural (fake) | 0.005 → 0.05 | SCM reconstruction quality on fake faces |
| Graph divergence | 0.001 → 0.01 | Pushes real and fake graphs apart (L1 distance) |
| DAG acyclicity | penalty | DAGMA constraint: ensures learned graphs are DAGs |
| SAE sparsity | 0.01 → 0.1 | Sparse autoencoder L1 regularization (if enabled) |

### Training vs Inference Differences

| Aspect | Training | Inference |
|--------|----------|-----------|
| Causal graphs | Updated per-batch (EMA) from label-split samples | Frozen (uses learned EMA graphs) |
| SCM label routing | Real samples update real SCM, fake samples update fake SCM | All samples pass through both SCMs |
| Causal intervention (Tier 3) | Active — ci_projection trains via classification loss | Active — same computation |
| Backbone | LayerNorms train, rest frozen | Fully frozen |
| Augmentation | Albumentations pipeline + real-frame augmentation | None |

## Key Design Decisions

1. **Forensic-only causal graphs**: Only consistency rules (18) + forensic features (30) enter the causal graph, NOT the 211 demographic attributes. Demographics correlate with identity, not manipulation — including them causes the graph to learn "male → beard" instead of "boundary_gradient_anomaly → DCT_artifact".

2. **Per-branch, per-distribution graphs**: 4 separate graphs (not one shared graph) because the causal structure differs between (a) spatial vs frequency domains and (b) real vs fake distributions. The detection signal is in the *divergence* between real and fake graphs.

3. **Linear SCMs over nonlinear**: Stronger identifiability guarantees, exact graph extraction (no Jacobian approximation), and better behavior in the 64-node regime. The nonlinear expressiveness comes from the neural backbone and attention fusion, not from the SCM itself.

4. **Gated injection**: Both the causal attention signal and the intervention features have separate learnable gates initialized near zero. This prevents immature causal reasoning from destabilizing the classifier while allowing the gates to open as the causal module converges.

5. **Source-paired training**: Each training batch contains matched real-fake pairs from the same source video (GenD recipe). This forces the detector to learn manipulation-specific features rather than identity or scene features.

## Repository Structure

```
training/
├── config/detector/nesy_defake.yaml        # Full configuration
├── detectors/nesy_defake_detector.py       # Main detector (forward, losses, fusion)
├── networks/nesy_defake/
│   ├── foundation_models/
│   │   ├── spatial_extractor.py            # CLIP-ViT spatial stream
│   │   └── frequency_extractor.py          # FAD-CLIP frequency stream
│   ├── causal/
│   │   └── causal_discovery.py             # DAGMA-DCE, LinearSCM, per-branch graphs
│   └── semantic/
│       ├── facial_semantic_extractor.py    # FaceBench 211 attrs + 51 curated causal attrs
│       ├── consistency_rules.py            # Tier 1: 20 training + 13 intervention rules
│       ├── forensic_features.py            # Tier 2: 30 pixel-level features
│       └── causal_intervention.py          # Tier 3: do-calculus interventions
├── dataset/nesy_defake_dataset.py          # Paired training dataset
├── trainer/trainer.py                      # Training loop, causal warmup
└── train.py                                # Entry point, optimizer setup
```

## References

- **GenD** (WACV 2026): Source-paired training + LayerNorm-only backbone adaptation
- **DAGMA** (Bello et al., NeurIPS 2022): M-matrix acyclicity constraint for DAG learning
- **Peters et al., 2014**: Identifiability of linear SCMs for causal discovery
- **FACS** (Ekman & Friesen): Facial Action Coding System — anatomical basis for expression-AU consistency rules
- **FaceBench**: Comprehensive facial attribute benchmark (211 attributes via Face-LLaVA)

---

## v5 — Dual Sub-Graph Causal Architecture (2026-03-23)

### Motivation

The v4 causal module used a single graph per branch with all 48 semantic features (18 consistency + 30 forensic) alongside 32 latent z-features, totalling 80 nodes. This conflated two fundamentally different causal processes:

1. **Identity-constraint violations**: Static face attributes that should not co-occur (female + beard, young + gray hair) or should co-activate (happy + AU6/AU12). These are identity-level semantic contradictions introduced by face swapping.
2. **Pixel-level forensic artifacts**: Boundary gradients, regional blur, frequency anomalies, color inconsistencies. These are manipulation traces at the signal level.

Mixing these in one graph dilutes both signals — DAGMA-DCE cannot distinguish gender→beard causation from blur→boundary artifact chains.

### Architecture: Dual Sub-Graphs per Branch

```
backbone(1024) → SparseFeatureSelector → z_feature(128) → CausalCompressor → z_causal(32)
                                                                                    │
                                                            ┌───────────────────────┤
                                                            ▼                       ▼
                                                  Identity-Causal            Forensic-Pixel
                                                  z(32) + s(71) = 103       z(32) + s(30) = 62
                                                  ┌──────────┐              ┌──────────┐
                                                  │ SCM_real  │              │ SCM_real  │
                                                  │ SCM_fake  │              │ SCM_fake  │
                                                  └──────────┘              └──────────┘
                                                       │                         │
                                                       └──────────┬──────────────┘
                                                                  ▼
                                                  Concatenated residuals (165-d)
                                                  → CausalViolationAttentionFusion
```

**Total: 8 causal graphs** (2 branches × 2 sub-graphs × 2 distributions)

### Two-Stage Visual Feature Compression

The backbone outputs 1024-d features, far too large for DAGMA (validated for 20-120 nodes). Two-stage compression solves this while preserving information:

| Stage | Transformation | Purpose |
|-------|---------------|---------|
| 1 | `SparseFeatureSelector(1024→128)` | Feature selection, retains rich representation |
| 2 | `CausalCompressor(128→32)` | Linear projection for graph tractability |

The 128-d intermediate representation feeds the classifier pathway (via attention fusion), while the 32-d compressed version enters the causal graphs.

### Curated 51 FaceBench Attributes (Identity Sub-Graph)

From the full 211 FaceBench attributes, 51 are selected for causal chain formation in the identity sub-graph:

| Group | Count | Attributes | Causal Role |
|-------|-------|------------|-------------|
| **A: Gender anchors + linked** | 11 | male, female, beard, mustache, goatee, sideburns, stubble, clean_shaven, heavy_makeup, lipstick, eyeshadow | Gender→facial_hair, gender→makeup chains |
| **B: Age anchors + linked** | 9 | young_looking, middle_aged, elderly_looking, smooth_skin, wrinkled_skin, age_spots, forehead_wrinkles, gray_hair, receding_hairline | Age→skin, age→hair chains |
| **C: Structural geometry** | 10 | strong_jaw, narrow_jaw, thick_eyebrows, thin_eyebrows, bushy_eyebrows, large_nose, small_nose, broad_nose, double_chin, pointed_chin | Bone structure contradictions |
| **D: Expression-AU coherence** | 15 | neutral, happy, sad, angry, surprised + AU1, AU2, AU4, AU5, AU6, AU9, AU12, AU14, AU15, mouth_open | Expression→muscle activation chains (FACS) |
| **E: Skin tone** | 4 | fair_skin, medium_skin, dark_skin, olive_skin | Tone blending artifacts |
| **F: Symmetry** | 2 | symmetrical_face, asymmetrical_face | Swap boundary indicators |

### Expanded Consistency Rules

#### Training Rules (20 features — enter identity sub-graph as nodes)

These fire often enough during training to provide gradient signal:

| Category | Rules | Operation |
|----------|-------|-----------|
| Mutually exclusive (4) | mouth_open×closed, male×female, smiling×frowning, bright×dim_lighting | Product (both high = violation) |
| Expression-AU original (5) | \|happy-AU6\|, \|happy-AU12\|, \|surprised-mean(AU1,AU2)\|, \|sad-AU15\|, \|angry-AU4\| | Absolute difference |
| Expression-AU new (4) | \|fearful-mean(AU1,AU5)\|, \|disgusted-AU9\|, \|contemptuous-AU14\|, neutral×max(key_AUs) | Extended to all basic emotions |
| Structural (3) | double_chin×narrow_jaw, square_face×narrow_jaw, round_face×pointed_chin | Bone structure contradictions |
| Skin coherence (2) | fair_skin×dark_skin, acne×elderly | Tone/age conflicts |
| Symmetry/quality (2) | symmetrical×asymmetrical, blurry×sharp | Quality contradictions |

#### Intervention Rules (13 definitions — do-calculus at inference)

Too sparse during training (near-zero for >95% of samples) but powerful as do-calculus interventions on learned graphs:

| Rule | Intervene On | Observe | Direction | Reason for Intervention |
|------|-------------|---------|-----------|------------------------|
| Gender→beard/mustache/stubble/sideburns (4) | female | facial_hair attrs | opposite | FaceBench correctly identifies swapped face's gender; violation is face-vs-context |
| Gender→makeup/eyeliner (2) | male | heavy_makeup, eyeliner | opposite | Same as above — within-face attrs consistent |
| Gender→strong_jaw (1) | female | strong_jaw | opposite | Statistical co-occurrence |
| Bald→long_hair (1) | bald | long_hair | opposite | Contradictory hair state |
| Age→smooth_skin/wrinkles/gray/receding/spots (5) | elderly/young | appearance attrs | opposite | Age-appearance contradictions |

**Key insight**: Gender-specific rules like female×beard produce ~0 during training even on fakes because FaceBench correctly identifies the swapped face's gender. The violation is between face attributes and surrounding context (body/hair/clothing), which FaceBench doesn't capture. Intervening on the gender node at inference and observing causal cascades through the learned graph is far more powerful than checking a near-zero product.

### FaceBench Sparsity Analysis

From a sample real frame (FF++ source 021, frame 002):
- **22/211 attributes present** at 0.5 threshold, **188 absent**, 1 unsure
- Present: female, caucasian, blonde_hair, long_hair, brown_eyes, fair_skin, smooth_skin, happy, young_looking, etc.
- All AUs are NO despite happy expression — demonstrates LLM extraction noise
- Continuous probabilities [0,1] are richer than binary thresholds; the causal module operates on raw probabilities

### Node Dimension Summary

| Sub-Graph | z_causal | Semantic | Total | Count |
|-----------|----------|----------|-------|-------|
| Identity-Causal | 32 | 51 curated + 20 rules = 71 | 103 | 4 graphs (2 branches × 2 dist) |
| Forensic-Pixel | 32 | 30 forensic | 62 | 4 graphs (2 branches × 2 dist) |
| **Combined residual per branch** | — | — | **165** | Feeds attention fusion |

### Configuration Changes (nesy_defake.yaml)

```yaml
# Two-stage z compression
latent_variables:
  z_spatial_dim:    128    # was 32
  z_frequency_dim:  128    # was 32
  z_causal_dim:     32     # new — graph tractability

# Dual sub-graph replaces forensic_only
dual_subgraph: true        # was forensic_only: true

# Expanded consistency rules
consistency_rules:
  output_dim: 20           # was 18
```

### Files Modified

| File | Change |
|------|--------|
| `semantic/facial_semantic_extractor.py` | Added `CAUSAL_ATTRIBUTE_NAMES` (51), `CAUSAL_ATTRIBUTE_INDICES`, `NUM_CAUSAL_ATTRIBUTES` |
| `semantic/consistency_rules.py` | Rewritten: 20 training rules + 13 intervention rule definitions |
| `causal/causal_discovery.py` | Dual sub-graph architecture, two-stage compression, 8 graphs |
| `detectors/nesy_defake_detector.py` | Wired dual sub-graph residuals, updated structural loss, DAG penalty, graph divergence |
| `semantic/causal_intervention.py` | Updated to use identity sub-graph API for do-calculus interventions |
| `detectors/utils/graph_visualization.py` | Updated for 4 sub-graph pairs per branch |
| `trainer/trainer.py` | Updated causal warmup EMA reporting for 4 sub-graph pairs |
| `config/detector/nesy_defake.yaml` | z_dims, dual_subgraph, output_dim updates |

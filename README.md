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
│       ├── facial_semantic_extractor.py    # FaceBench 211 attributes
│       ├── consistency_rules.py            # Tier 1: 18 cross-attribute rules
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

# NeSyDeFake — Technical Training Reference

## Architecture Overview

Per-frame deepfake detector built on DeepfakeBench. No temporal branch; each frame is an independent sample.

```
spatial_frames  → SpatialExtractor (CLIP-L/14)  → raw_spatial (1024-d)
                                                  ├─→ spatial_proj (1024→proj_dim) ─┐
                                                  └─→ SAE.spatial → z_spatial        │
                                                                                      ├─→ MultiModalFusion → fused (1024-d)
freq_frames     → FreqExtractor (FAD-CLIP-L/14) → raw_freq (1024-d)                  │              │
                                                  ├─→ freq_proj (1024→proj_dim) ──────┘              │
                                                  └─→ SAE.freq → z_freq                              │
                                                                                                      │
semantic_attrs (73-d) ──────────────────────────────────────────────────────────────────┐             │
z_sae = concat(z_spatial, z_freq) (8192-d) ─────────────────────────────────────────────┴────────────┴→ CausalModule
                                                                                                           │
                                                                          ┌────────────────────────────────┤
                                                                          │                                │
                                                              SCM_real → residuals_real (329-d)    SCM_fake → residuals_fake (329-d)
                                                              A_real (real biomechanics)            A_fake (generator artifacts)
                                                              v_real = violation score              v_fake = conformance score
                                                                          │                                │
                                                          violation_proj_real (329→1024)    violation_proj_fake (329→1024)
                                                                          └────────────┬───────────────────┘
                                                                                       │ additive
                                                                         classifier_input = fused + Δ_real + Δ_fake
                                                                                       ↓
                                                                         MultiTaskHead → cls (2), uncertainty (1)
```

---

## Modules

### M1: Foundation Models

| Branch | Backbone | Output | Notes |
|--------|----------|--------|-------|
| Spatial | CLIP ViT-L/14 (`openai/clip-vit-large-patch14`) | 1024-d | Fully frozen Phase 1; LNs unfreeze at Phase 2 |
| Frequency | FAD-CLIP ViT-L/14 (same backbone) | 1024-d | Same freeze schedule; FAD front-end always trains |

**FAD (Frequency Adaptive Detection):** `output = original + α * high_freq_residual`
DCT → learnable bandpass → iDCT → additive emphasis. α initialised to 0.1, learnable.
Dataset delivers raw `[0,1]` pixels (`keep_raw: true`); CLIP normalisation applied after enhancement.

---

### M4: Sparse Autoencoder (SAE)

**Class:** `DualBranchSparseAutoencoder` (wraps two `BranchSparseAutoencoder` instances)
**File:** `training/networks/nesy_defake/classifiers/sparse_autoencoder.py`

#### Purpose

Decomposes polysemantic CLIP features into sparse monosemantic features that serve as causal variables for M3. Real CLIP representations are **polysemantic** — each dimension encodes a superposition of unrelated concepts. The SAE learns an overcomplete dictionary of monosemantic features where each active dimension represents a single interpretable concept (analogous to Anthropic's dictionary learning for LLM interpretability).

#### Design Decisions

1. **BatchTopK activation** (Bussmann et al., NeurIPS 2024 workshop) — controls average sparsity per sample without L1 coefficient tuning, allows adaptive per-sample sparsity.
2. **Per-branch SAEs** — spatial SAE captures appearance/identity features; frequency SAE captures compression artifacts and high-frequency forensic signals. Separate dictionaries prevent cross-contamination.
3. **Unit-norm decoder columns** (Anthropic convention) — prevents the model from cheating by scaling decoder columns to absorb reconstruction error. Forces each dictionary element to be a direction, not a magnitude.
4. **Pre-projection input** — SAE operates on raw 1024-d CLIP features before projection heads. The full CLIP representation is richer; projection heads compress for classification, but the SAE needs the full manifold for causal discovery.

#### BranchSparseAutoencoder: Internal Architecture

```
Input x (B, 1024)
   │
   ├─ [if normalize_inputs] LayerNorm(x.float()) → x_norm  (BF16-safe cast)
   │
   ├─ clamp(x_norm, -100.0, 100.0)               (prevent BF16 overflow into encoder)
   │
   └─ Encoder:
         x_centered = x - b_dec                   (centre on decoder bias)
         pre_act    = x_centered @ W_enc + b_enc   (W_enc: 1024 × 4096)
         z          = BatchTopK(pre_act)            (4096-d sparse)
      Decoder:
         x_hat = z @ W_dec + b_dec                 (W_dec: 4096 × 1024)
```

**Parameters:**
- `W_enc` ∈ ℝ^{1024×4096} — encoder weight, Kaiming uniform init
- `b_enc` ∈ ℝ^{4096} — encoder bias, zeros init
- `W_dec` ∈ ℝ^{4096×1024} — decoder weight, Kaiming uniform init, then column-normalised to unit norm
- `b_dec` ∈ ℝ^{1024} — decoder bias (shared pre-activation offset), zeros init

#### BatchTopK Activation

```python
# Training: select top (target_k × B) activations across entire batch
flat = pre_act.view(-1)                        # (B × 4096,)
topk_idx = torch.topk(flat, k * B)[1]
mask = zeros_like(flat).scatter_(0, topk_idx, 1.0)
z = pre_act * mask.view(B, 4096)               # average ~128 active per sample
```

```python
# Inference: use learned threshold (EMA of batch minimum during training)
z = pre_act * (pre_act > self.threshold).float()
```

The threshold EMA update at each training step:
```python
ct = result[result > 0].min()          # minimum non-zero activation in batch
threshold ← ema_decay * threshold + (1 - ema_decay) * ct     (ema_decay=0.99)
```

This means inference exactly reproduces the average density seen during training without rerunning the batch selection algorithm.

#### SAE Loss

```
L_sae = norm_recon + aux_loss_coeff * aux_loss

norm_recon = MSE(x, x_hat) / Var(x)                    (variance-normalised)
           = mean((x - x_hat)²) / mean((x - mean(x))²)
```

Variance normalisation makes the loss scale-invariant across feature dimensions and stable across training phases regardless of how much the upstream CLIP features shift.

**Dead feature auxiliary loss:**
Features inactive for `dead_feature_window` batches (default 200) are considered dead. The aux loss revives them:

```python
dead_mask = (step_counter - feature_last_active) > dead_feature_window

# Project residual through dead features only (using full ReLU, not TopK)
dead_pre  = (residual - b_dec) @ W_enc[:, dead_mask] + b_enc[dead_mask]
dead_z    = ReLU(dead_pre)
# Hard top-k on dead features to select best candidates
dead_sparse = topk_scatter(dead_z, k=min(target_k, n_dead))
# Reconstruction from dead features
dead_recon  = dead_sparse @ W_dec[dead_mask]
# Loss: how much residual can dead features explain?
aux_loss = MSE(residual, dead_recon) / Var(x)
```

Gradients flow through `aux_loss` into both `W_enc[:, dead_mask]` and `W_dec[dead_mask]`, pushing dead features to explain the current residual. `aux_loss_coeff = 0.2` (YAML; 1/32 ≈ 0.03 is the OpenAI standard, 0.2 is more aggressive revival).

#### Decoder Normalisation (post-step)

After each optimizer step, the trainer calls:
```python
model.sparse_ae.normalize_decoder_weights()
# → W_dec /= W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
```

This projects each decoder row (dictionary direction) back onto the unit sphere. Without this, the model can route around the TopK constraint by growing decoder magnitudes.

#### DualBranchSparseAutoencoder Forward

```python
# Both branches run independently; losses are summed
z_spatial, s_loss, s_info = spatial_sae(LN(raw_spatial))   # → (B, 4096)
z_freq,    f_loss, f_info = frequency_sae(LN(raw_freq))     # → (B, 4096)
total_loss = s_loss + f_loss                                # single scalar

# Downstream: concatenate for causal module
z_sae = cat([z_spatial, z_freq], dim=1)                    # (B, 8192), ~256 active
```

#### Diagnostics (reported per branch per step)

| Metric | Description |
|--------|-------------|
| `sae_{branch}_l0` | Mean active features per sample (target ≈ 128) |
| `sae_{branch}_fvu` | Fraction of Variance Unexplained = `norm_recon` |
| `sae_{branch}_dead` | Count of dead features (target: near 0) |

#### Config Summary

```yaml
sparse_features:
  enabled:             true
  dict_size:           4096       # 4× expansion of 1024-d input
  target_k:            128        # active features per branch per sample (~3% of dict)
  dead_feature_window: 200        # batches before a feature is declared dead
  aux_loss_coeff:      0.2        # dead feature revival strength
  normalize_inputs:    true       # LayerNorm raw CLIP features before SAE
```

---

### M3: Causal Discovery (Dual-Graph DAGMA-DCE)

**Class:** `CausalDiscoveryModule`
**File:** `training/networks/nesy_defake/causal/causal_discovery.py`

#### Purpose and Motivation

Two complementary DAGMA-DCE graphs operating on the same 329-d causal variable space (256 active SAE features + 73 facial semantic attributes):

**Graph_real (A_real)** — real-face biomechanical DAG:
- SCM_real trained on real frames only; A_real EMA updated from real frames only
- Captures genuine facial biomechanics: `smileLeft → cheekSquintLeft → eyeNarrow`
- `v_real` = how much a sample **violates** this structure → high for fakes, low for reals

**Graph_fake (A_fake)** — generator artifact DAG:
- SCM_fake trained on fake frames only; A_fake EMA updated from fake frames only
- Captures systematic generator patterns: artificial correlations between frequency artifacts and semantic attributes that never co-occur in real faces
- `v_fake` = how much a sample **conforms** to this structure → high for fakes, low for reals

Both scores push in the same direction through orthogonal mechanisms, forming a bidirectional detection signal stronger than either alone.

**Graph divergence (interpretability):**
```
A_real − A_fake > 0  →  edges generators BREAK   (biomechanical constraints violated)
A_fake − A_real > 0  →  edges generators CREATE  (artificial artifact correlations)
```
`get_graph_divergence()` returns `broken_by_fakes` and `created_by_fakes` for analysis.

**Why DAGMA-DCE** (Waxman et al., OJSP 2024) over standard DAGMA:
Edge weights = RMS Jacobian `∂f_j/∂x_i` over the data distribution. Interpretable as actual causal strength (not arbitrary network weights). Model-agnostic and principled for thresholding.

#### Module Architecture

```
z_sae (B, 8192) sparse ──→ SparseFeatureSelector (Linear 8192→256, no bias)
                                    │
                                    └─→ z_active (B, 256)
                                           │
                             LayerNorm(256)│          LayerNorm(73)
                                           │                │
                             z_norm (B,256)│   s_norm (B,73)│
                                           └────────────────┘
                                                    │ cat
                                               x (B, 329)   ← shared causal input
                                             ┌──────┴──────┐
                                        SCM_real        SCM_fake
                                             │                │
                                      x_hat_real (B,329) x_hat_fake (B,329)
                                      residuals_real     residuals_fake
                                             │                │
                                      A_real (real only) A_fake (fake only)  ← label-routed Jacobians
                                             │                │
                                  ViolationScorer_real  ConformanceScorer_fake
                                             │                │
                                          v_real (B,)      v_fake (B,)
                                       [high=fake]       [high=fake]
```

#### SparseFeatureSelector

```python
# Linear projection: 8192-d sparse → 256-d dense
# No activation — linear combination of monosemantic features is interpretable
proj = Linear(8192, 256, bias=False)
# Init: N(0, 1/sqrt(8192)) ≈ N(0, 0.011)
```

**Why linear, no activation?** The causal graph will operate on `z_active`. If the selector were nonlinear, position i in z_active would no longer correspond to a fixed combination of dictionary atoms, breaking interpretability of the graph nodes. A linear map ensures each z_active dimension is a weighted sum of specific SAE dictionary elements — learnable but stable.

**Why compress 8192 → 256?** The Jacobian computation for DAGMA-DCE is O(hidden × d²) in memory. At d=8192+73=8265, computing the adjacency matrix would require ~68M operations per sample. At d=329, it's ~108K per sample — a 630× reduction.

#### BatchedSCM (Structural Causal Model)

Models d=329 structural equations `x_j = f_j(x)` as a shared trunk with per-node output heads:

```
x (B, 329)
   │
   └─→ Trunk: [Linear(329,128) → Sigmoid] × (num_layers-1)   → h (B, 128)
                    │
                    └─→ Heads: Linear(128, 329)                 → x_hat (B, 329)
```

**Architecture choices:**
- **Sigmoid activations** (not ReLU): DAGMA-DCE requires once-differentiable functions for Jacobian computation. Sigmoid is infinitely differentiable everywhere, while ReLU has a non-differentiable kink at 0 that can produce zero or undefined gradients.
- **Shared trunk**: d=329 separate MLPs would be ~329× the parameters and GPU overhead. The shared trunk learns common latent interactions; per-node specialisation comes from independent rows of the output Linear.
- **Head init**: `std=0.01`, `bias=0.0` — heads start near-zero, so x_hat ≈ 0 and node_residuals ≈ x initially. This prevents the SCM from immediately memorising the data before causal structure is established.

#### DAGMADCELearner: Jacobian Computation

The full Jacobian of the BatchedSCM factors exploiting the shared trunk:

```
∂x̂_j/∂x_i = Σ_k  heads.weight[j,k] · (∂h_k/∂x_i)
```

This allows computing J_trunk first (d backward passes through the trunk only), then combining with heads.weight via matrix multiplication — avoiding d full backward passes through the complete network.

```python
# Step 1: Forward through trunk
h = scm.trunk(x)                            # (B, hidden=128)

# Step 2: Compute J_trunk_sq[i, k] = E_b[(∂h_k/∂x_i)²]
J_trunk_sq = zeros(d=329, hidden=128)
for k in range(hidden_dim):                 # 128 iterations (chunked by 32)
    grad_k = autograd.grad(h[:, k].sum(), x,
                           create_graph=training,
                           retain_graph=True)[0]   # (B, d)
    J_trunk_sq[:, k] = (grad_k**2).mean(dim=0)    # (d,)

# Step 3: Factored DCE adjacency
W_sq  = scm.heads.weight**2                # (d, hidden): W_sq[j,k]
A_sq  = J_trunk_sq @ W_sq.t()             # (d, d): A_sq[i,j]
A_dce = sqrt(A_sq + 1e-10)                # (d, d): A_dce[i,j] = DCE(i→j)

# Step 4: Zero diagonal (no self-loops)
A_dce *= (1 - eye(d))
```

The diagonal of `J_trunk_sq @ W_sq.t()` at index [i,j] gives:
`Σ_k W_sq[j,k] · J_trunk_sq[i,k]` = `Σ_k heads.weight[j,k]² · E[(∂h_k/∂x_i)²]`

This is an approximation of the exact `E[(∂x̂_j/∂x_i)²]` (drops cross terms between hidden units k,l). The approximation is accurate when hidden activations are approximately uncorrelated, which holds after sigmoid saturation.

**Chunking:** The loop processes 32 hidden dimensions per chunk to bound peak memory. At `hidden=128` and `d=329`, the intermediate gradient tensors are (B, d) = e.g. (256, 329) per step — small, but retain_graph=True keeps the computation graph alive for 128 iterations.

#### EMA Adjacency Buffer

```python
# Buffer: _A_dce_ema ∈ ℝ^{329×329}, initialized to zeros
# Updated during training after each A_dce computation:
_A_dce_ema ← 0.99 * _A_dce_ema + 0.01 * A_dce.detach()
```

The EMA serves two purposes:
1. **Stability** — a single batch's Jacobian is noisy; EMA smooths over the distribution.
2. **Inference efficiency** — at inference time, the EMA is used directly, skipping the entire Jacobian computation (no autograd overhead during evaluation).

The EMA is updated over **all frames** (real + fake). Rationale: a graph fit only on real faces can overspecialise to clean biomechanics and produce false positives for OOD manipulation methods. The L_structural loss (real frames only) keeps SCM predictions accurate for reals without restricting the graph topology.

#### DAGMA Acyclicity Constraint

```python
def dagma_acyclicity(A, s=1.0):
    d = A.shape[0]
    A_sq = A * A                               # element-wise; ensures non-negative
    M = s * eye(d) - A_sq                      # M-matrix
    sign, logabsdet = torch.linalg.slogdet(M)
    h = -logabsdet + d * log(s)               # h(A) = 0 iff A is a DAG
    return h.clamp(min=0.0)
```

From Bello et al. (NeurIPS 2022): a matrix W represents a DAG iff sI − W∘W is positive definite (an M-matrix) for sufficiently large s. The log-det characterisation is:
- `h(A) ≥ 0` always
- `h(A) = 0` iff the graph of A is a DAG
- `h(A) → ∞` as cycles strengthen

`compute_dag_penalty()` applies this to `_A_dce_ema` (not the current batch's A_dce) to penalise cycles in the running graph estimate. Using EMA rather than the batch Jacobian prevents gradient noise from the penalty destabilising the SCM.

#### ViolationScorer

```python
# Per-node squared reconstruction error
node_errors = (x - x_hat)**2                      # (B, 329)

# Weight by node in-degree (how connected is this node in the graph?)
in_degree = A_dce.sum(dim=0).detach()             # (329,): sum of incoming edges
in_degree /= (in_degree.mean() + 1e-8)            # normalise to mean=1
node_errors *= in_degree.unsqueeze(0)             # (B, 329): high-connectivity violations weighted more

# MLP → scalar violation score
violation = Linear(329, 64) → ReLU → Linear(64, 1) → squeeze  # (B,)
```

**In-degree weighting:** A central node (many causal parents) that deviates from its predicted value is more suspicious than a leaf node. Weighting by normalised in-degree focuses the violation score on nodes where the causal structure is densest.

#### CausalDiscoveryModule Forward

**Training path:**
```python
x = build_causal_input(z_sae, semantic_attrs)   # (B, 329)
x = x.detach().requires_grad_(True)              # cut from upstream graph; re-enable for Jacobian

x_hat = causal_learner.scm(x)                   # (B, 329) SCM reconstruction
node_residuals = x - x_hat                       # (B, 329)

A_dce = causal_learner.compute_adjacency_dce(x)  # (329, 329), updates EMA
violation_score = violation_scorer(x, x_hat, A_dce)  # (B,)

return violation_score, node_residuals, A_dce
```

**Inference path:**
```python
x_hat = causal_learner.scm(x)                   # no Jacobian
node_residuals = x - x_hat
violation_score = violation_scorer(x, x_hat, _A_dce_ema)  # use EMA graph
```

The `x.detach()` at the start of forward is intentional: gradients for the SCM and Jacobian computation should not propagate back into the SAE or CLIP backbone through the causal path. Each module's gradients flow through its own branch.

#### Integration with Detector (dual violation_proj)

```python
# In NeSyDeFakeHybridDetector.__init__():
violation_proj_real = Linear(329, 1024, bias=False)   # zero-init
violation_proj_fake = Linear(329, 1024, bias=False)   # zero-init

# In forward():
# violation_proj_real: large residuals_real (broken real structure) → fake signal
# violation_proj_fake: small residuals_fake (conforms to fake artifacts) → fake signal
classifier_input = fused_features
if use_causal:
    classifier_input = (fused_features
                        + violation_proj_real(residuals_real)
                        + violation_proj_fake(residuals_fake))
```

Both projections are zero-initialised → neutral through Phase 1/2. In Phase 3 they learn complementary routes: `violation_proj_real` amplifies large real-graph residuals (fake indicator), `violation_proj_fake` amplifies conformance to fake-graph patterns (also a fake indicator).

#### Node Name Mapping

The 329-d causal variable vector has the following index mapping:
```
[0:128]    → z_spatial_0 .. z_spatial_127   (spatial SAE active features)
[128:256]  → z_freq_0    .. z_freq_127      (frequency SAE active features)
[256:329]  → s_0         .. s_72            (semantic attributes)
```

Semantic attribute indices (s_0..s_72) correspond to:
```
s_0:s_16    → DeepFace: emotion (7), age (1), gender (2), race (6)
s_16:s_21   → InsightFace: pose yaw/pitch/roll (3), det_score (1), antispoof (1)
s_21:s_73   → MediaPipe: ARKit blendshapes (52)
```

#### Parameter Count Breakdown

| Submodule | Parameters | Notes |
|-----------|-----------|-------|
| `SparseFeatureSelector` (Linear 8192×256) | 2,097,152 | shared by both graphs |
| `LayerNorm` (z_norm + s_norm) | 658 | shared |
| `BatchedSCM_real` trunk + heads | ~101,504 | real graph only |
| `ViolationScorer_real` | 21,121 | real graph only |
| `BatchedSCM_fake` trunk + heads | ~101,504 | fake graph only |
| `ConformanceScorer_fake` | 21,121 | fake graph only |
| **Total CausalDiscoveryModule** | **~2.34M** | ~2.22M single-graph + ~120K for fake SCM/scorer |

| SAE submodule (per branch) | Parameters |
|----------------------------|-----------|
| `W_enc` (1024×4096) | 4,194,304 |
| `b_enc` (4096) | 4,096 |
| `W_dec` (4096×1024) | 4,194,304 |
| `b_dec` (1024) | 1,024 |
| **Total per branch** | **8,393,728** |
| **Total DualBranchSAE (2 branches)** | **~16.8M** |

#### Config Summary

```yaml
causal_module:
  enabled: false               # flipped true at phase3_start by trainer
  causal_warmup_batches: 200   # real-face-only batches to pre-populate A_dce

  latent_variables:
    z_spatial_dim:  128        # = target_k (active features, spatial branch)
    z_frequency_dim: 128       # = target_k (active features, frequency branch)
    total_sae_dim:  256        # SparseFeatureSelector output (8192 → 256)

  semantic_dim: 73

  discovery:
    algorithm:        dagma_dce
    sparsity_penalty: 0.01     # L1 on A_dce (not currently applied in loss)

  dag_learning:
    hidden_dim:         128    # BatchedSCM trunk hidden size
    num_layers:         3      # trunk depth (2 Sigmoid layers + 1 output)
    dag_penalty_weight: 0.1    # h(A_dce_ema) coefficient in L_causal
```

---

### M5: MultiTaskHead

Input: 1024-d (`fusion.projection_dim`). Tasks: classification (2-way CE), uncertainty (MSE regression).

---

## Loss Function

```
L_total = w_cls          * L_cls
        + w_unc          * L_uncertainty
        + w_causal       * (L_structural_real + L_dag_real)    ← real frames / A_real
        + w_causal_fake  * (L_structural_fake + L_dag_fake)    ← fake frames / A_fake
        + w_contrastive  * L_contrastive_real                  ← v_real separation
        + w_contrastive_fake * L_contrastive_fake              ← v_fake separation
        + w_sparse       * L_sae
```

| Term | Formula | Active |
|------|---------|--------|
| `L_cls` | CrossEntropy(logits, label) | all phases |
| `L_uncertainty` | MSE(uncertainty, 1 − correct) | all phases |
| `L_structural_real` | mean(residuals_real[real]²) — SCM_real fits real biomechanics | Phase 3 |
| `L_dag_real` | `dag_w × h(A_real_ema)` | Phase 3 |
| `L_structural_fake` | mean(residuals_fake[fake]²) — SCM_fake fits generator artifacts | Phase 3 |
| `L_dag_fake` | `0.5 × dag_w × h(A_fake_ema)` (half weight: fake graph less universal) | Phase 3 |
| `L_contrastive_real` | `ReLU(1 − (v_real[fake].mean() − v_real[real].mean()))` | Phase 3 |
| `L_contrastive_fake` | `ReLU(1 − (v_fake[fake].mean() − v_fake[real].mean()))` | Phase 3 |
| `L_sae` | `norm_recon + aux_loss_coeff * dead_feature_aux` (per branch, summed) | Phase 1–3 |

Both contrastive losses handle single-class batches: reals-only minimises the score, fakes-only maximises it.

**Loss weights:**

| Key | Phase 1 | Phase 2 | Phase 3 |
|-----|---------|---------|---------|
| `classification` | 1.0 | 1.0 | 1.0 |
| `uncertainty` | 0.5 | 0.5 | 0.5 |
| `sparse` | 0.1 | 0.1 | 0.1 |
| `causal` | 0.0 | 0.0 | 0.3 (set by trainer, real graph) |
| `causal_fake` | 0.0 | 0.0 | 0.15 (= causal × 0.5 default) |
| `contrastive` | 0.0 | 0.0 | 0.3 (set by trainer, real graph) |
| `contrastive_fake` | 0.0 | 0.0 | 0.15 (= contrastive × 0.5 default) |

`causal_fake` and `contrastive_fake` default to half of their real-graph counterparts. The fake graph is trained on a finite set of methods and may not generalise to unseen generators; the half-weight prevents overfitting to training-fake-specific artifacts. These can be set explicitly in `loss_func.weights` to override.

---

## Training Phases

### Phase 1 — Hard Freeze [epochs 0–10]

- CLIP backbones (all params incl LayerNorms) frozen (`train_layernorms: false`)
- FAD front-end (DCT filters, emphasis α) always trainable
- Trains: `spatial_proj`, `freq_proj`, `fusion`, `classifier`, `sparse_ae`
- SAE learns stable monosemantic features on static CLIP representations

### Phase 2 — GenD LayerNorm Adaptation [epochs 10–20]

At `epoch == phase2_start`, trainer calls:
1. `extractor.unfreeze_layernorms()` on both branches → LN params enter training at lr=1e-5
2. Rebuild optimizer + scheduler (LN params in `backbone_layernorms` group)

- All Phase 1 modules continue
- LNs adapt backbone normalisation to deepfake domain without shifting feature manifold

### Phase 3 — Causal Discovery [epochs 20–50]

At `epoch == phase3_start`, trainer:
1. Calls `model.enable_causal()` → sets `use_causal = True`
2. Sets `loss_func.weights.causal = 0.3`, `loss_func.weights.contrastive = 0.3`
3. Runs `_run_causal_warmup(train_loader, n_batches=200)`:
   - Freezes all params except `causal_module`
   - Iterates dataloader, filters to real frames (label==0)
   - Forward: backbone+SAE under `no_grad`; causal_module with grad (for Jacobian)
   - Populates `_A_dce_ema` with biologically valid graph before any fake data
   - Restores frozen/trainable states
4. Rebuilds optimizer + scheduler

- All modules train; violation gated fusion feeds classifier
- A_dce EMA updated from all frames; L_structural enforced on real frames only

---

## Optimizer Param Groups

Defined in `training/train.py` (`choose_optimizer`):

| Group | LR | Params |
|-------|----|--------|
| `backbone_layernorms` | 1e-5 | LayerNorm params (after Phase 2 unfreeze) |
| `always_trainable` | 1e-4 | FAD front-end (DCT, emphasis α) |
| `phase_proj` | 3e-4 | spatial_proj, freq_proj |
| `projection_heads` | 1e-4 | spatial_proj, freq_proj (overlap; check impl) |
| `fusion` | 2e-4 | MultiModalFusion |
| `classifier` | 2e-4 | MultiTaskHead |
| `optional_modules` | 1e-4 | sparse_ae, causal_module, violation_proj |

Scheduler: `cosine_warmup`, warmup_epochs=2, total=50.
Gradient clipping: global max_norm=1.0; frequency branch additional clip max_norm=2.0 (before global).
Mixed precision: BF16 (H200 — GradScaler disabled). SAE decoder normalised post-step.

---

## Semantic Features (73-d per frame)

Extracted offline; loaded from `.npz` at training time (~0.1ms per sample).

| Source | Features | Indices | Count |
|--------|----------|---------|-------|
| DeepFace | emotion (7), age (1), gender (2), race (6) | 0:16 | 16 |
| InsightFace | pose yaw/pitch/roll (3), det_score (1), antispoof (1) | 16:21 | 5 |
| MediaPipe | ARKit blendshapes | 21:73 | 52 |

Extraction: `preprocessing/extract_semantic_features.py`
Loader: `preprocessing/load_semantic_features.py` — returns `per_frame[frame_idx]` (73,) float32

---

## Key Files

| File | Role |
|------|------|
| `training/config/detector/nesy_defake.yaml` | All hyperparameters |
| `training/detectors/nesy_defake_detector.py` | Detector: forward, losses, metrics |
| `training/networks/nesy_defake/foundation_models/` | CLIP + FAD-CLIP extractors |
| `training/networks/nesy_defake/fusion/` | MultiModalFusion (attention) |
| `training/networks/nesy_defake/causal/causal_discovery.py` | DAGMA-DCE causal module |
| `training/networks/nesy_defake/classifiers/sparse_autoencoder.py` | BatchTopK SAE |
| `training/networks/nesy_defake/classifiers/multi_task_head.py` | Classifier |
| `training/trainer/trainer.py` | Train loop, phase transitions, causal warmup |
| `training/train.py` | Entry point, optimizer/scheduler construction |
| `preprocessing/extract_semantic_features.py` | Offline semantic extraction |

---

## Config Quick Reference

```yaml
sparse_features:
  enabled: true
  dict_size: 4096          # 4x expansion of 1024-d input
  target_k: 128            # active features per branch per sample
  dead_feature_window: 200 # batches before dead feature declared
  aux_loss_coeff: 0.2      # dead feature revival strength
  normalize_inputs: true

causal_module:
  enabled: false           # flipped to true at phase3_start by trainer
  causal_warmup_batches: 200
  latent_variables:
    z_spatial_dim: 128     # = target_k
    z_frequency_dim: 128   # = target_k
    total_sae_dim: 256     # SparseFeatureSelector output (8192 → 256)
  semantic_dim: 73
  dag_learning:
    hidden_dim: 128
    num_layers: 3
    dag_penalty_weight: 0.1

training_phases:
  phase1: {epochs: [0, 10]}
  phase2: {epochs: [10, 20]}
  phase3: {epochs: [20, 50], causal_loss_weight: 0.3}

foundation_models:
  spatial:  {train_layernorms: false}   # Phase 1 hard freeze
  frequency: {train_layernorms: false}  # Phase 1 hard freeze; FAD always trains
```

---

## Causal Variable Dimensions

```
z_sae (sparse, full dict):    8192-d  (4096 spatial + 4096 freq)
z_active (dense, selector):   256-d   (SparseFeatureSelector linear projection)
s_semantic:                    73-d
causal input x:               329-d   (z_active + s_semantic, shared by both graphs)

residuals_real:               329-d   (x − SCM_real(x), all frames)
residuals_fake:               329-d   (x − SCM_fake(x), all frames)
A_real:                     329×329   (real-face DCE adjacency, EMA from real frames)
A_fake:                     329×329   (fake-face DCE adjacency, EMA from fake frames)

v_real:                         (B,)  (violation score on real graph,  high=fake)
v_fake:                         (B,)  (conformance score on fake graph, high=fake)

violation_proj_real output:   1024-d  (329→1024 linear, zero-init, additive to fused)
violation_proj_fake output:   1024-d  (329→1024 linear, zero-init, additive to fused)

graph divergence:
  broken_by_fakes:           329×329  (A_real − A_fake).clamp(0) — edges generators break
  created_by_fakes:          329×329  (A_fake − A_real).clamp(0) — edges generators create
```

---

## Trainer Note: Dual Warmup at Phase 3

At `phase3_start` the trainer's `_run_causal_warmup` currently warms up only the real graph (real frames only). With dual graphs, a second fake-warmup pass is needed before Phase 3 training:

```python
# Pseudocode — trainer.py _run_causal_warmup should do both:
_causal_warmup(train_loader, n_batches=200, label_filter=0)   # real frames → A_real EMA
_causal_warmup(train_loader, n_batches=200, label_filter=1)   # fake frames → A_fake EMA
```

Without fake warmup, A_fake starts from zeros and the conformance scorer receives no informative signal at the start of Phase 3. The fake graph will converge during training, but initialising it from real fake data (before any classification gradient biases it) produces faster and more stable fake-artifact discovery.

---
---

# NeSyDeFake v2 — Per-Branch Causal Discovery with FaceBench Semantic Attributes

**Date: 2026-03-10**

> Major architectural revision: per-branch causal graphs with 211 named FaceBench facial attributes extracted via teacher-forced LLM inference. Replaces the previous single-combined-causal-space architecture described above.

---

## What Changed (TL;DR)

| Aspect | Old (v1) | New (v2) |
|--------|----------|----------|
| Semantic features | 73-d precomputed (DeepFace + InsightFace + MediaPipe) | 211 named attributes from FaceBench Face-LLaVA-v1.5-13B |
| Semantic extraction | Offline `.npz` files | Online teacher-forced single-pass LLM inference |
| Causal graphs | 2 graphs on combined space (d=329) | 4 graphs: 2 per branch (d=339 each) |
| Causal input | `[z_combined_256, s_73]` | Spatial: `[z_spatial_128, s_211]`, Freq: `[z_freq_128, s_211]` |
| Node names | Generic `s_0..s_72` | Interpretable: `black_hair`, `smiling`, `AU12_lip_corner_puller`, etc. |
| Training phases | 3 phases (causal starts at Phase 3) | 2 phases (all modules train from epoch 0 with loss warmup) |
| SAE feature selection | Single `SparseFeatureSelector(8192→256)` | Per-branch: `spatial_selector(4096→128)`, `freq_selector(4096→128)` |

---

## How It Works — For Everyone

### The Big Idea

Imagine you're a detective trying to spot a forged painting. You have two tools:

1. **A magnifying glass** (spatial branch) — looks at visual details: skin texture, edge sharpness, colour blending
2. **A UV light** (frequency branch) — reveals invisible patterns: compression artifacts, spectral signatures that the naked eye can't see

And you have a **checklist of 211 facial features** (semantic attributes) — things like "black hair", "smiling", "beard", "wrinkled skin", "eyeglasses", etc.

For each tool, you learn two rulebooks:
- **Real-face rulebook**: "In genuine photos, when someone smiles, their cheeks always raise and their eyes narrow" (biomechanical constraints)
- **Fake-face rulebook**: "In generated photos, there's always a weird correlation between skin smoothness and a specific frequency artifact"

To check if a face is fake, you apply both rulebooks with both tools:
- Does it **violate** the real-face rules? (real faces follow them, fakes don't)
- Does it **match** the fake-face rules? (fakes show these patterns, real faces don't)

Four scores in total — spatial violations, spatial fake-patterns, frequency violations, frequency fake-patterns — all combined to make the final decision.

### Why This Is Powerful

- **Latent branches** (spatial + frequency SAE features) are good at **learning patterns** — they find statistical regularities in the data that humans might miss
- **Semantic branch** (211 named attributes) is good at **explanations** — when the system says "this is fake", you can look at the causal graph and see *why*: "the relationship between `smiling` and `cheek_raise` that exists in real faces is broken in this image"
- **Separate per-branch graphs** let you see exactly what each generator breaks in the spatial vs frequency domains

---

## How It Works — Technical Details

### Architecture Overview (v2)

```
spatial_frames  → SpatialExtractor (CLIP-L/14)  → raw_spatial (1024-d)
                                                   ├─→ spatial_proj (1024→1024) ──┐
                                                   └─→ SAE.spatial → z_spatial     │
                                                       (4096-d sparse)             │
                                                                                    ├─→ MultiModalFusion → fused (1024-d)
freq_frames     → FreqExtractor (FAD-CLIP-L/14) → raw_freq (1024-d)               │
                                                   ├─→ freq_proj (1024→1024) ──────┘
                                                   └─→ SAE.freq → z_freq
                                                       (4096-d sparse)

raw_frames  → FacialSemanticExtractor (FaceBench Face-LLaVA-v1.5-13B)
                [frozen 13B LLM, teacher-forced single-pass]
              → semantic_attrs (B, 211) named attribute probabilities
                (black_hair=0.92, smiling=0.85, wrinkled_skin=0.12, ...)

                    ┌─────────────── CausalDiscoveryModule ──────────────────┐
                    │                                                         │
                    │  Spatial branch (d=339):              Freq branch (d=339):
                    │    spatial_selector(4096→128)           freq_selector(4096→128)
                    │    x_spatial = [z_spatial_128, s_211]   x_freq = [z_freq_128, s_211]
                    │         │                                    │
                    │    SCM_spatial_real  SCM_spatial_fake    SCM_freq_real  SCM_freq_fake
                    │    A_spatial_real    A_spatial_fake      A_freq_real    A_freq_fake
                    │    v_spatial_real    v_spatial_fake      v_freq_real    v_freq_fake
                    │    residuals_s_real  residuals_s_fake    residuals_f_real residuals_f_fake
                    │         │                                    │
                    └─────────┼────────────────────────────────────┼──────────┘
                              │                                    │
               violation_proj_spatial_real(339→1024)    violation_proj_freq_real(339→1024)
               violation_proj_spatial_fake(339→1024)    violation_proj_freq_fake(339→1024)
                              │                                    │
                              └─── additive fusion ────────────────┘
                                          │
                    classifier_input = fused + Δ_spatial_real + Δ_spatial_fake
                                            + Δ_freq_real + Δ_freq_fake
                                          ↓
                    MultiTaskHead → cls (2), uncertainty (1), violation_score (1)
```

### M2: Facial Semantic Extractor (FaceBench Face-LLaVA)

**Class:** `FacialSemanticExtractor`
**File:** `training/networks/nesy_defake/semantic/facial_semantic_extractor.py`

#### Why Replace Precomputed 73-d Features?

The old 73-d features (DeepFace + InsightFace + MediaPipe blendshapes) were extracted offline by multiple ad-hoc models. Problems:
1. **Pipeline fragility**: 3 separate models, each with its own preprocessing
2. **Limited coverage**: 73 features miss many discriminative facial attributes
3. **No gradient flow**: precomputed features can't adapt during training

The new approach uses a single FaceBench Face-LLaVA-v1.5-13B model (Wang et al., CVPR 2025) to extract 211 named facial attributes covering appearance, accessories, makeup, surrounding context, psychology/expressions, action units, and identity.

#### Two Modes

| Mode | Config | VRAM | Output | Use Case |
|------|--------|------|--------|----------|
| Vision-only | `use_llm: false` | ~1.2GB | `(B, output_dim)` projected features | Fast training, limited GPU |
| Full LLM | `use_llm: true` | ~28GB | `(B, 211)` named attribute probabilities | Full interpretability |

#### Teacher-Forced Single-Pass Extraction (Full LLM Mode)

Instead of running 211 separate yes/no queries through the 13B model (prohibitively expensive), we use a single teacher-forced forward pass:

```
Step 1: Image → CLIP-ViT-L/14@336 vision tower → 576 patch tokens (1024-d each)
Step 2: mm_projector (mlp2x_gelu) → 576 visual tokens (5120-d each)
Step 3: Construct prompt:
          SYSTEM: "A chat between..."
          USER: <image>\nFor each facial attribute, predict 1 if present or 0 if absent.
          ASSISTANT: black hair: 1\nblonde hair: 1\n...smiling: 1\n... (all 211, answered "1")
Step 4: Embed all text tokens via LLM embedding layer
Step 5: Concatenate: [before_embeds, visual_tokens, question_embeds, answer_embeds]
Step 6: Single LLM forward pass (teacher-forced, no autoregressive generation)
Step 7: At each of the 211 answer positions, extract:
          P(present) = sigmoid(logit("1") - logit("0"))
```

**Key optimisations:**
- `use_cache=False`: no KV cache allocation (~4GB savings per micro-batch)
- `torch.bfloat16`: no overflow risk (unlike fp16 with 13B model)
- `low_cpu_mem_usage=True`: sequential shard loading during init
- `micro_batch_size=4`: processes images in small chunks to control peak VRAM
- Answer positions pre-computed once at init by comparing "all 1" vs "all 0" tokenisations

#### 211 FaceBench Attributes (5 Views)

| View | Count | Examples |
|------|-------|---------|
| Appearance | 111 | `black_hair`, `wrinkled_skin`, `high_cheekbones`, `oval_face`, `smooth_skin` |
| Accessories | 30 | `eyeglasses`, `hat`, `necklace`, `face_mask`, `headphones` |
| Makeup | 13 | `heavy_makeup`, `lipstick`, `eyeliner`, `foundation` |
| Surrounding | 12 | `indoor_background`, `bright_lighting`, `blurry_image` |
| Psychology | 33 | `happy`, `neutral_expression`, `AU12_lip_corner_puller`, `AU6_cheek_raise` |
| Identity | 7 | `male`, `female`, `east_asian`, `caucasian` |
| **Total** | **211** | |

### M3: Per-Branch Dual-Graph Causal Discovery (v2)

**Class:** `CausalDiscoveryModule`
**File:** `training/networks/nesy_defake/causal/causal_discovery.py`

#### Architecture: Four Causal Graphs

Each latent branch (spatial, frequency) has its own pair of real/fake causal graphs operating on `[z_branch_active, semantic_attrs]`:

```
Spatial branch (d_spatial = 128 + 211 = 339):
  A_spatial_real (339×339):  spatial ↔ semantic causality in REAL faces
    e.g. smooth_skin → no_wrinkles, smiling → high_cheekbones
  A_spatial_fake (339×339):  spatial ↔ semantic causality in FAKE faces
    e.g. blending_artifact ↔ skin_texture_mismatch

Frequency branch (d_freq = 128 + 211 = 339):
  A_freq_real (339×339):  frequency ↔ semantic causality in REAL faces
    e.g. natural_high_freq → hair_texture
  A_freq_fake (339×339):  frequency ↔ semantic causality in FAKE faces
    e.g. GAN_spectral_peak ↔ face_region
```

#### Why Per-Branch Graphs?

1. **Interpretability**: See exactly what each generator breaks in spatial vs frequency domains, and how those relate to named facial semantics.
2. **Smaller causal spaces**: d=339 per branch (vs d=467 if combined), making DAGMA Jacobian computation cheaper per graph.
3. **Orthogonal signals**: Spatial and frequency artifacts manifest differently; separate graphs let the SCMs specialise.

#### Per-Branch Feature Selectors

```python
# Each branch has its own selector (no longer a single combined 8192→256)
spatial_selector = SparseFeatureSelector(4096, 128)  # spatial SAE dict → 128 active
freq_selector    = SparseFeatureSelector(4096, 128)  # freq SAE dict → 128 active
```

Each selector is a bias-free linear projection that picks causally relevant SAE dictionary elements for its branch. The causal graph then discovers relationships between these 128 latent features and the 211 named semantic attributes.

#### Interpretable Node Names

Every node in the causal graph has a human-readable name:

```
Spatial branch (339 nodes):
  [0:128]   → z_spatial_0 .. z_spatial_127    (data-driven SAE features)
  [128:339] → black_hair, blonde_hair, ..., smiling, ..., south_asian
              (211 named FaceBench attributes)

Frequency branch (339 nodes):
  [0:128]   → z_freq_0 .. z_freq_127          (data-driven SAE features)
  [128:339] → black_hair, blonde_hair, ..., smiling, ..., south_asian
              (same 211 named attributes, shared semantics)
```

When inspecting `A_spatial_real[135, 128+31]`, you're looking at the causal strength from `z_spatial_7` to `smiling` in real faces — not an opaque `node_135 → node_159`.

#### Detection Signal (Four Scores)

| Score | Meaning | High for fakes because... |
|-------|---------|--------------------------|
| `v_spatial_real` | Spatial violation of real biomechanics | Fakes break spatial-semantic relationships (e.g. smooth skin but visible pores) |
| `v_spatial_fake` | Spatial conformance to fake artifacts | Fakes exhibit spatial generator patterns (e.g. blending boundaries) |
| `v_freq_real` | Frequency violation of real structure | Fakes break frequency-semantic relationships (e.g. wrong spectral profile for hair) |
| `v_freq_fake` | Frequency conformance to fake artifacts | Fakes exhibit frequency artifacts (e.g. GAN checkerboard in spectrum) |

Aggregated: `v_real = v_spatial_real + v_freq_real`, `v_fake = v_spatial_fake + v_freq_fake`

#### Graph Divergence (Interpretability)

For each branch, comparing real vs fake graphs reveals what generators break or create:

```python
divergence = model.causal_module.get_graph_divergence('spatial')
# divergence['broken_by_fakes']   = (A_real - A_fake).clamp(min=0)
#   → edges present in real faces but missing in fakes (biomechanics violated)
# divergence['created_by_fakes']  = (A_fake - A_real).clamp(min=0)
#   → edges present only in fakes (artificial correlations)
```

Example interpretation: if `broken_by_fakes[smiling, AU6_cheek_raise]` is large, it means real faces have a strong causal link between smiling and cheek raising (Duchenne smile mechanism), but this link is weak or absent in fakes — the generator doesn't properly model this muscle coupling.

### Training Flow (v2)

#### Two Phases (Simplified from v1's Three Phases)

All modules train jointly from epoch 0 with loss weight warm-up. No delayed causal activation.

**Phase 1 — Hard Freeze Backbone [epochs 0–5]:**
- CLIP backbones fully frozen (including LayerNorms)
- All other modules train: `fad_frontend`, `projection_heads`, `fusion`, `classifier`, `sparse_ae`, `causal_module`, `semantic_extractor`
- Loss warm-up active: causal/contrastive/sparse weights ramp from 0.01

**Phase 2 — Full End-to-End [epochs 5–50]:**
- Backbone LayerNorms unfrozen (GenD-style LN adaptation)
- All modules continue joint training
- Loss weights reach full values by epoch 10

#### Loss Weight Warm-Up Schedule

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

This replaces the old phase-gated approach: instead of suddenly activating the causal module at epoch 20, all losses are active from epoch 0 but start small. This allows the causal module to co-adapt with the SAE from the beginning, leading to more stable convergence.

#### Causal Warmup (Reduced)

```python
causal_warmup_batches: 50   # down from 200 in v1
```

At the start of training, 50 batches of real-face data are passed through the causal module to pre-populate the EMA adjacency buffers. This is reduced from v1's 200 because end-to-end training from epoch 0 compensates — the causal module doesn't need a fully formed graph before seeing any gradients.

### Training Forward Pass

```python
# Step 1: Extract raw branch features
raw_spatial = spatial_extractor(spatial_frames)      # (B, 1024)
raw_freq    = frequency_extractor(freq_frames)       # (B, 1024)

# Step 2: Project and fuse for classifier
fused = fusion(spatial_proj(raw_spatial),
               freq_proj(raw_freq))                  # (B, 1024)

# Step 3: SAE (parallel path)
z_spatial, z_freq, sae_loss, sae_info = sparse_ae(
    spatial_feat=raw_spatial,
    frequency_feat=raw_freq)                         # z_*: (B, 4096) sparse

# Step 4: Semantic attributes (frozen FaceBench LLM)
semantic_attrs = semantic_extractor(raw_frames)      # (B, 211) P(present)

# Step 5: Per-branch causal discovery
causal_out = causal_module(
    z_spatial=z_spatial,       # (B, 4096) spatial SAE sparse
    z_freq=z_freq,             # (B, 4096) frequency SAE sparse
    semantic_attrs=semantic_attrs,  # (B, 211) named attributes
    label=label,               # (B,) routes Jacobians to real/fake subsets
    return_graph=True)

# Step 6: Violation fusion + classification
classifier_input = (fused
    + violation_proj_spatial_real(causal_out['residuals_spatial_real'])
    + violation_proj_spatial_fake(causal_out['residuals_spatial_fake'])
    + violation_proj_freq_real(causal_out['residuals_freq_real'])
    + violation_proj_freq_fake(causal_out['residuals_freq_fake']))

task_outputs = multitaskhead(classifier_input)
```

### Inference Forward Pass

Same as training except:
- `label=None`: no Jacobian computation, uses EMA adjacency matrices directly
- No loss computation
- SAE uses learned threshold instead of batch TopK
- Semantic extractor still runs (frozen LLM forward pass for attributes)

### Loss Function (v2)

```
L_total = w_cls           * L_cls
        + w_unc           * L_uncertainty
        + w_causal        * (L_structural_spatial_real + L_structural_freq_real + L_dag_real)
        + w_causal_fake   * (L_structural_spatial_fake + L_structural_freq_fake + L_dag_fake)
        + w_contrastive   * (L_contrastive_spatial_real + L_contrastive_freq_real)
        + w_contr_fake    * (L_contrastive_spatial_fake + L_contrastive_freq_fake)
        + w_sparse        * L_sae
```

Key difference from v1: structural and contrastive losses are summed across both branches (spatial + frequency), so each branch's causal module receives its own gradient signal.

### Causal Variable Dimensions (v2)

```
z_spatial (sparse):          4096-d  (spatial SAE dictionary)
z_freq (sparse):             4096-d  (frequency SAE dictionary)
z_spatial_active (dense):    128-d   (spatial_selector linear projection)
z_freq_active (dense):       128-d   (freq_selector linear projection)
s_semantic:                  211-d   (FaceBench named attributes)

x_spatial (causal input):    339-d   (z_spatial_active + s_semantic)
x_freq (causal input):      339-d   (z_freq_active + s_semantic)

residuals per branch:        339-d   (x − SCM(x))
A per branch:              339×339   (DCE adjacency)

v per branch:                (B,)    (violation/conformance score)

violation_proj per branch:  339→1024 (zero-init, 4 projections total)
```

### Config Quick Reference (v2)

```yaml
semantic_attributes:
  enabled: true
  backend: face_llava
  model_path: /data/umar/Repos/FaceBench
  output_dim: 211              # auto-set by LLM mode
  use_llm: true                # full 13B teacher-forced extraction
  micro_batch_size: 4          # small batches for 13B LLM

causal_module:
  enabled: true                # trains from epoch 0 (with loss warmup)
  causal_warmup_batches: 50
  latent_variables:
    z_spatial_dim: 128         # per-branch dense active features
    z_frequency_dim: 128
  semantic_dim: 211            # auto-overridden by extractor

training_phases:
  phase1: {epochs: [0, 5],  freeze_modules: [foundation_models]}
  phase2: {epochs: [5, 50], train_modules: [all]}

loss_warmup:
  causal:             {start: 0.01, end: 0.3, epochs: [0, 10]}
  contrastive:        {start: 0.01, end: 0.3, epochs: [0, 10]}
  causal_fake:        {start: 0.005, end: 0.15, epochs: [0, 10]}
  contrastive_fake:   {start: 0.005, end: 0.15, epochs: [0, 10]}
  sparse:             {start: 0.01, end: 0.1, epochs: [0, 5]}
```

### Key Files (v2)

| File | Role |
|------|------|
| `training/config/detector/nesy_defake.yaml` | All hyperparameters |
| `training/detectors/nesy_defake_detector.py` | Detector: forward, losses, metrics (per-branch) |
| `training/networks/nesy_defake/semantic/facial_semantic_extractor.py` | FaceBench 211-attribute extractor |
| `training/networks/nesy_defake/causal/causal_discovery.py` | Per-branch DAGMA-DCE (4 graphs) |
| `training/networks/nesy_defake/classifiers/sparse_autoencoder.py` | BatchTopK SAE (unchanged) |
| `training/networks/nesy_defake/classifiers/multitask_head.py` | Classifier (unchanged) |
| `training/trainer/trainer.py` | Train loop, phase transitions |
| `training/train.py` | Entry point, optimizer construction |


# NeSyDeFake — Technical Training Reference

## Architecture Overview

Per-frame deepfake detector built on DeepfakeBench. No temporal branch; each frame is an independent sample.

```
spatial_frames  → SpatialExtractor (CLIP-L/14)  → raw_spatial (1024-d)
                                                  ├─→ spatial_proj (1024→proj_dim) ─┐
                                                  └─→ SAE.spatial → z_spatial        │
                                                                                      ├─→ MultiModalFusion → fused (1024-d)
freq_frames     → FreqExtractor (FAD-CLIP-L/14) → raw_freq (1024-d)                  │         │
                                                  ├─→ freq_proj (1024→proj_dim) ──────┘         │
                                                  └─→ SAE.freq → z_freq                         │
                                                                                                 │
semantic_attrs (73-d, cached .npz) ─────────────────────────────────────────────────────┐        │
z_sae = concat(z_spatial, z_freq) (8192-d sparse) ──────────────────────────────────────┴────────┴→ CausalModule
                                                                                                      │
                                                                                          violation_proj(node_residuals: 329-d → 1024-d)
                                                                                                      │ (additive)
                                                                                          classifier_input = fused + violation_proj(residuals)
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

### M4: Sparse Autoencoder (SAE)

`DualBranchSparseAutoencoder` — one SAE per branch, operated jointly.
- Architecture: BatchTopK SAE, dict_size=4096, target_k=128 active features per branch per sample
- Input: raw 1024-d CLIP features (pre-projection, LayerNorm-normalised)
- Output: z_spatial (4096-d sparse), z_freq (4096-d sparse); get_z_sae() concatenates → 8192-d
- Loss: variance-normalised MSE + (aux_loss_coeff) * dead_feature_penalty
- Post-step: `sparse_ae.normalize_decoder_weights()` enforces unit-norm decoder columns
- Always built at init; active from Phase 1

### M3: Causal Discovery (DAGMA-DCE)

`CausalDiscoveryModule` — learns a DAG over 329 variables per frame.

**Input construction:**
```
z_sae (8192-d sparse) → SparseFeatureSelector (linear, 8192→256) → z_active (256-d)
z_active + s_semantic (73-d) → x (329-d causal variables)
```

**Graph learning:**
- `BatchedSCM`: shared trunk MLP + d=329 output heads; x → x̂
- `DAGMADCELearner`: computes A_dce via Jacobian (∂f_j/∂x_i RMS across batch)
- Acyclicity: DAGMA log-det penalty h(A) = −log det(sI − A∘A) + d·log(s)
- EMA buffer `_A_dce_ema` (decay=0.99) tracks running graph estimate

**Forward (training):**
1. Compute x̂ = SCM(x) for ALL frames → node_residuals = x − x̂ (B, 329)
2. Compute A_dce from **real frames only** (label==0) → EMA update on reals only
3. violation_score = ViolationScorer(x, x̂, A_dce) for all frames (B,)
4. Returns: `(violation_score, node_residuals, A_dce)`

**Forward (inference):** uses `_A_dce_ema`; no Jacobian computation.

**Gated fusion into classifier** (in detector, not causal module):
```python
classifier_input = fused_features + violation_proj(node_residuals)
# violation_proj: Linear(329, 1024, bias=False), zero-init → neutral in Phase 1/2
```

### M5: MultiTaskHead

Input: 1024-d (`fusion.projection_dim`). Tasks: classification (2-way CE), uncertainty (MSE regression).

---

## Loss Function

```
L_total = w_cls * L_cls
        + w_unc * L_uncertainty
        + w_causal * L_structural      ← real frames only, MSE of node residuals
        + w_causal * L_dag             ← DAGMA acyclicity penalty on A_dce_ema
        + w_contrastive * L_contrastive
        + w_sparse * L_sae
```

| Term | Formula | Active |
|------|---------|--------|
| `L_cls` | CrossEntropy(logits, label) | all phases |
| `L_uncertainty` | MSE(uncertainty, 1 − correct) | all phases |
| `L_structural` | mean((x − x̂)²) on real frames | Phase 3 |
| `L_dag` | dagma_penalty_weight × h(A_dce_ema) | Phase 3 |
| `L_contrastive` | ReLU(margin − (mean_fake_v − mean_real_v)), margin=1.0 | Phase 3 |
| `L_sae` | SAE reconstruction + dead_feature_aux | Phase 1–3 |

**Loss weights** (`loss_func.weights` in YAML):

| Key | Phase 1 | Phase 2 | Phase 3 |
|-----|---------|---------|---------|
| `classification` | 1.0 | 1.0 | 1.0 |
| `uncertainty` | 0.5 | 0.5 | 0.5 |
| `sparse` | 0.1 | 0.1 | 0.1 |
| `causal` | 0.0 | 0.0 | 0.3 (set by trainer) |
| `contrastive` | 0.0 | 0.0 | 0.3 (set by trainer) |

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
- A_dce EMA updated from real frames only throughout

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
  aux_loss_coeff: 0.2
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
z_sae (sparse, full dict):  8192-d  (4096 spatial + 4096 freq)
z_active (dense, selector): 256-d   (SparseFeatureSelector linear projection)
s_semantic:                  73-d
causal input x:             329-d   (z_active + s_semantic)
node_residuals:             329-d   (x − SCM(x), all frames)
A_dce:                    329×329   (DCE adjacency matrix)
violation_score:              (B,)  (scalar per sample, from ViolationScorer)
violation_proj output:      1024-d  (329→1024 linear, added to fused_features)
```

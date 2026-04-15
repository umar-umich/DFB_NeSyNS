# April 14 — Performance Analysis, Architecture & Recovery Plan

## NeSyDeFake Framework Architecture (Current Working State)

### End-to-End Pipeline
```
Input Frame (224x224)
    │
    ├──► Frozen CLIP ViT-L/14 (openai/clip-vit-large-patch14)
    │       │ Train only LayerNorms (GenD recipe)
    │       │ Output: 1024-d CLS token
    │       ▼
    │    spatial_proj: Linear(1024→1024) → LayerNorm → GELU
    │       │
    │       ▼
    │    Fusion MLP (attention fallback, single-branch):
    │       Linear(1024→1024) → LN → ReLU
    │       │
    │       ▼                                    ┌─────────────────────────────┐
    │    fused_features (1024-d) ────────────────►│  Classifier (multi-task)    │
    │                                            │  hidden: [512, 256]         │
    │                                            │  dropout: 0.3              │
    │                                            │  output: 2 (binary)        │
    │                                            └──────────┬──────────────────┘
    │                                                       │
    ├──► fast_semantic (58-d precomputed) ──┐               │
    │      InsightFace: gender, age, pose   │               │
    │      DeepFace: emotions, ethnicity    │               │
    │      LibreFace: 17 AUs (FACS)         ├──► Concept Branch (58→64→23)
    │      MediaPipe: landmarks, gaze       │    23 consistency rules
    │                                       │    Evidence → EDL
    │                                       │               │
    ├──► forensic_features (83-d precomp.) ─┤               │
    │      0-25: SegFormer region features  │               │
    │      26-29: dead (zeros)              ├──► CCV Causal Branch
    │      30-37: PPNC (patch noise)        │    ForensicAnomalyDetector (5 groups)
    │      38-45: CCNC (cross-channel)      │    Counterfactual reasoning
    │      46-60: SRM (rich model filters)  │    Evidence → EDL
    │      61-72: Multi-scale noise         │               │
    │      73-82: FFT spectral              │               │
    │                                       │               │
    └──► consistency_rules (23-d) ──────────┘               │
         9 expression-AU coherence                          │
         5 cross-region identity                            ▼
         2 skin/age, pose-gaze, etc.              NeSy-EDL Fusion
                                                  (CMEF + PBAS + IBDC)
                                                  Dirichlet evidence from
                                                  3 sources: spatial, concept, causal
```

### Key Design Principles
1. **Frozen CLIP + trainable LayerNorms only** (GenD, WACV 2026): preserves pretrained manifold
2. **Projection head** (`_make_projection`): `Linear → LN → GELU` — minimal nonlinear adaptation between frozen backbone and fusion. Critically needed: without it, the model is a pure linear probe with no task adaptation
3. **Fusion MLP**: when single-branch (spatial only), falls back to `Linear → LN → ReLU` — provides the learned transformation from CLIP space to classification space
4. **EDL loss** (Evidential Deep Learning): Dirichlet-based uncertainty with NeSy evidence fusion
5. **Paired training** (GenD): real-fake pairs from same FF++ source video
6. **Features always from original .pt** — augmented frames load original image's features (see "Augmentation Design" below)

### Trainable Parameter Budget (spatial-only mode)
| Module | Params | Notes |
|--------|--------|-------|
| CLIP LayerNorms | ~50K | Only unfrozen part of backbone |
| spatial_proj | ~1M | Linear(1024→1024) + LN |
| Fusion fallback | ~1M | Linear(1024→1024) + LN |
| Concept branch | ~10K | 58→64→23, small |
| CCV causal branch | ~50K | Forensic anomaly detector + CF module |
| Classifier | ~660K | 1024→512→256→2 |
| **Total trainable** | **~2.8M** | CLIP backbone: 304M frozen |

### Feature Dimensions Summary
| Feature Set | Dim | Source | Precomputed? |
|-------------|-----|--------|--------------|
| CLIP spatial | 1024 | ViT-L/14 CLS | No (live) |
| fast_semantic | 58 | InsightFace+DeepFace+LibreFace+MediaPipe | Yes (.pt) |
| forensic | 83 | SegFormer+pixel forensics | Yes (.pt) |
| consistency_rules | 23 | Derived from fast_semantic | Computed in model |

---

## Performance Drop Root Cause (RESOLVED)

### Problem
Epoch 0 AUC dropped from **0.85/0.82** (FF++/CDF, April 11) to **0.79/0.68**.

### Root Cause — Two architectural changes after working commit `e3a80f8`:

**1. Projection head removed** (`nesy_defake_detector.py`)
- Working: `_make_projection` = `Linear(1024→1024) → LayerNorm → GELU`
- Broken: `_make_adapter` = `nn.Identity()` (pass-through)

**2. Fusion MLP gutted** (`multimodal_fusion.py`)
- Working: `Linear(concat→fused) → LN → ReLU → Dropout → Linear(fused→proj) → LN`
- Broken (single-branch): Just `LayerNorm(proj_dim)` — zero learned parameters

**Impact:** Pipeline became `frozen_CLIP → LN → linear_classifier` — a pure linear probe with no adaptation capacity. The projection + fusion together provide ~2M trainable parameters for task-specific feature adaptation. Without them, only the classifier MLP (660K) could learn.

**Fix:** Reverted both files to e3a80f8. Epoch 0 AUC back to **0.83**.

---

## Augmentation Design — Why Original Features + Augmented Images Works

### The "Bug" That's Actually the Right Design
Augmented frames (`frames_aug_1/001/`) load features from `fast_semantic/001.pt` (the ORIGINAL image's features), not `fast_semantic/001_aug1.pt`. This was initially identified as a bug, but diagnostic data shows it's the correct approach:

### Diagnostic Data (from `scripts/inspect_precomputed_features.py`)
| Feature Set | Augmentation-Sensitive (corr < 0.80) | Invariant (corr >= 0.80) |
|-------------|--------------------------------------|--------------------------|
| fast_semantic (58-d) | **33/58** (57%) | 25/58 |
| forensic (83-d) | **75/83** (90%) | 8/83 |
| Expression features | corr 0.05–0.27 | — |

### Why Matched Features (augmented view) Hurts
When features are computed on augmented views, the classifier sees wildly different feature vectors for the SAME identity:
- **CLIP says:** "same face" (augmentation-robust, trained on 400M image-text pairs)
- **Semantic features say:** "different expression, different AUs, different pose" (sensitive)
- **Forensic features say:** "different noise, different frequency" (sensitive)

The concept branch can't learn stable rules ("happy face should have AU6+AU12") when AU values change randomly with each augmentation. The CCV causal branch can't learn forensic anomaly patterns when noise/FFT features are inconsistent.

### The Right Division of Labor
```
Backbone (CLIP)     →  sees augmented image  →  learns augmentation-invariant representations
Semantic/Forensic   →  from original image   →  provides stable signal for symbolic reasoning
```

This is conceptually similar to **teacher-student distillation**: the features act as a stable "teacher signal" while the backbone learns robust representations from diverse views.

---

## Two Separate Issues — Don't Conflate

### Issue A: Architecture regression (RESOLVED)
- **Cause:** `_make_projection` → `nn.Identity()`, fusion MLP → `LayerNorm`
- **Fix:** Reverted to e3a80f8
- **Status:** Fixed. Back to 83% epoch 0

### Issue B: K=12 mixed-seed augmentation noise
- **Cause:** K=12 with features computed on augmented views gave AUC ~0.55
- **Root cause:** Feature noise from augmentation-sensitive attributes drowns out symbolic signal
- **Status:** Understood. Will NOT use augmented-view features. Original features only.

---

## Potential Improvements (Safe, Incremental)

### Option 1: Residual Projection (low risk)
Add a skip connection around the projection head so CLIP features flow through directly while the projection learns a task-specific correction:
```python
# Current: x = projection(clip_out)
# Proposed: x = clip_out + alpha * projection(clip_out)   # alpha learnable, init=0.1
```
**Rationale:** Adapter/LoRA literature shows residual connections preserve pretrained features while allowing task adaptation. Init alpha small so early training is close to linear probe, then the model decides how much correction to apply.

### Option 2: Lightweight Bottleneck Adapter (low risk)
Replace `Linear(1024→1024)` projection with a bottleneck:
```python
# Linear(1024→256) → GELU → Linear(256→1024) → LayerNorm
# + skip connection from input
```
**Rationale:** 4x fewer parameters than full projection, forces the adapter to learn a low-rank correction. Standard technique from adapter-tuning literature (Houlsby et al., 2019).

### Option 3: Unfreeze Top-K CLIP Layers (medium risk)
Currently only LayerNorms are trained. Unfreezing the last 1-2 transformer blocks would give the model much more capacity for deepfake-specific features, but risks overfitting to FF++ artifacts.
**Recommendation:** Only try after OTF augmentation is working (augmentation acts as regularization).

### Option 4: Feature-Gated Fusion (low risk)
Add a learned gate that weights how much the concept/causal branches contribute:
```python
gate = sigmoid(learnable_scalar)  # init=-2.0 → gate≈0.12
output = spatial_evidence + gate * (concept_evidence + causal_evidence)
```
**Already partially implemented** as `evidence_gate` in config (concept_init=-0.5, causal_init=-0.8).

**Recommendation:** Start with Option 1 (residual projection) — safest, most principled, easy to A/B test.

---

## Experiment Sequence (Next Steps)

### Step 1: Baseline Recovery (DONE)
- Reverted detector + fusion to e3a80f8
- Config: augmented JSON + no OTF aug + original features
- Result: epoch 0 AUC ~0.83 (matches April 11 baseline)

### Step 2: Wait for Baseline Convergence
- Let current run finish. Expect ~94% CDF AUC by epoch 15-20.

### Step 3: OTF Augmentation (after Step 2 converges)
Config changes:
```yaml
train_dataset: [FaceForensics++]        # original JSON (23K real)
use_data_augmentation: true             # OTF augmentation ON
balance_classes: true                   # weighted sampler for class balance
```
Features still from original .pt. Backbone sees diverse views each epoch.

### Step 4: K=12 with Original Features (optional, after Step 3)
```yaml
train_dataset: [FaceForensics++_augmented]  # with aug1-12 entries (~13x real)
use_data_augmentation: false
balance_classes: true                       # downsample to balance
```
K=12 pre-augmented images for backbone diversity, but always load `001.pt` not `001_aug4.pt`. More diverse backbone views than K=3, features stay stable.

### Step 5: Residual Projection (after Step 3 or 4)
Add skip connection around projection head for better feature preservation.

---

## Key Files
| File | Role |
|------|------|
| `training/detectors/nesy_defake_detector.py` | Main detector, projection heads, EDL fusion |
| `training/networks/nesy_defake/fusion/multimodal_fusion.py` | Fusion MLP / attention |
| `training/networks/nesy_defake/foundation_models/spatial_extractor.py` | CLIP/DINOv2/EVA/SigLIP backbone |
| `training/dataset/nesy_defake_dataset.py` | Dataset with K-augment, feature loading |
| `training/dataset/pixel_forensic_extractor.py` | OTF forensic features (indices 30-82) |
| `training/networks/nesy_defake/ccv_branch.py` | CCV causal branch, forensic anomaly detector |
| `training/networks/nesy_defake/semantic/consistency_rules_v7.py` | 23 consistency rules |
| `training/networks/nesy_defake/semantic/refined_attributes.py` | Feature name registries |
| `preprocessing/forensic_helpers.py` | All 83 forensic feature definitions |
| `preprocessing/precompute_fast_semantic.py` | InsightFace+DeepFace+LibreFace+MediaPipe |
| `augment_and_precompute.sh` | Full pipeline: augment + precompute (K configurable) |
| `scripts/inspect_precomputed_features.py` | Feature diagnostic tool |
| `scripts/clean_dataset_json.py` | Remove stray entries from dataset JSON |

## Working Commit Reference
- **e3a80f8**: Last known good state (detector + fusion architecture)
- **Current branch:** `causal_discovery`

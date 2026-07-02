# April 15 — Corrected Architecture Documentation & Config Alignment

## NeSyDeFake Framework Architecture (Current Working State)

### End-to-End Pipeline (EDL mode: `ablation_mode: causal_edl`)
```
Input Frame (224x224)
    │
    ├──► Frozen CLIP ViT-L/14 (openai/clip-vit-large-patch14)
    │       │ Train only LayerNorms (GenD recipe)
    │       │ Output: 1024-d CLS token (spatial_raw)
    │       ▼
    │    spatial_proj: Linear(1024→1024) → LayerNorm → GELU
    │       │
    │       ▼
    │    projected (1024-d) ─────────────────────────────────────────────┐
    │       │                                                           │
    │       ▼                                                           │
    │    Classifier (MultiTaskHead)                                     │
    │       Linear(1024→512) → LN → GELU → Dropout(0.3)                │
    │       Linear(512→256)  → LN → GELU → Dropout(0.3)                │
    │       Linear(256→2)                                               │
    │       │                                                           │
    │       ▼                                                           │
    │    spatial_logits → softplus → spatial_evidence (2-d)             │
    │                                        │                          │
    ├──► fast_semantic (58-d precomputed) ──┐ │                          │
    │      InsightFace: gender, age, pose   │ │                          │
    │      DeepFace: emotions, ethnicity    │ │                          │
    │      LibreFace: 17 AUs (FACS)         ├─┤──► Concept Branch        │
    │      MediaPipe: landmarks, gaze       │ │    (58→64→2 evidence)    │
    │                                       │ │    23 consistency rules   │
    │                                       │ │    Evidence → EDL         │
    ├──► forensic_features (83-d precomp.) ─┤ │                          │
    │      0-25: SegFormer region features  │ │                          │
    │      26-29: dead (zeros)              ├─┤──► CCV Causal Branch     │
    │      30-37: PPNC (patch noise)        │ │    spatial_raw as input──┘
    │      38-45: CCNC (cross-channel)      │ │    ForensicAnomalyDetector (5 groups)
    │      46-60: SRM (rich model filters)  │ │    Counterfactual reasoning
    │      61-72: Multi-scale noise         │ │    Evidence → EDL
    │      73-82: FFT spectral              │ │
    │                                       │ │
    └──► consistency_rules (23-d) ──────────┘ │
         9 expression-AU coherence            │
         5 cross-region identity              ▼
         2 skin/age, pose-gaze, etc.    Evidence Fusion
                                        ├─ NeSy-EDL (nesy_fusion=true):
                                        │    CMEF (Confidence-Modulated Evidence Fusion)
                                        │    + PBAS (Per-Branch Auxiliary Supervision)
                                        │    + IBDC (Inter-Branch Disagreement Calibration)
                                        └─ Simple (nesy_fusion=false):
                                             total = spatial + σ(gate1)*concept + σ(gate2)*causal

                                             ▼
                                        alpha = evidence + 1
                                        prob  = alpha / sum(alpha)
                                        uncertainty = 2 / sum(alpha)
```

### Important: Fusion MLP is NOT in the EDL path
The `MultiModalFusion` module is instantiated (for multi-branch concat/attention/weighted fusion)
but the EDL forward path (`ablation_mode: causal_edl`) bypasses it entirely:
```python
# EDL path (nesy_defake_detector.py:881-887):
projected = self.spatial_proj(spatial_raw)     # NOT project_and_fuse()
task_outputs = self.classifier(projected)       # direct to classifier
spatial_evidence = F.softplus(spatial_logits)
```
The `project_and_fuse()` method (which calls fusion) is only used in the non-EDL
ablation path. With `ablation_mode: causal_edl`, the fusion module's parameters
are in the optimizer but receive no gradient (only DDP anchor keeps them alive).

### Key Design Principles
1. **Frozen CLIP + trainable LayerNorms only** (GenD, WACV 2026): preserves pretrained manifold
2. **Projection head** (`_make_projection`): `Linear(1024→1024) → LayerNorm → GELU` — minimal nonlinear adaptation between frozen backbone and classifier
3. **EDL loss** (Evidential Deep Learning): Dirichlet-based uncertainty estimation
4. **NeSy-EDL** (novel): Two evidence fusion modes — CMEF with per-branch aux losses, or simple scalar gates
5. **Paired training** (GenD): real-fake pairs from same FF++ source video
6. **Features always from original .pt** — augmented frames load original image's features (see "Augmentation Design" below)
7. **CCV causal branch receives `spatial_raw`** (raw CLIP features), not the projected features

### Trainable Parameter Budget (EDL mode, spatial-only active branch)
| Module | Params | Notes |
|--------|--------|-------|
| CLIP LayerNorms | ~102K | Only unfrozen part of backbone |
| spatial_proj | ~1M | Linear(1024→1024) + LN |
| Classifier (MultiTaskHead) | ~660K | 1024→512→256→2, LN+GELU+Dropout per layer |
| Concept branch | ~10K | 58→64→2 evidence (+ consistency rules) |
| CCV causal branch | ~62K | Constraints + forensic anomaly + counterfactual |
| CMEF (if nesy_fusion=true) | ~small | Per-branch gates + temperature |
| Evidence gates (if nesy_fusion=false) | 2 scalars | concept_gate + causal_ev_gate |
| Fusion (instantiated, unused in EDL) | ~1M | Gets no gradient in EDL path |
| **Total active** | **~1.8M** | CLIP backbone: 304M frozen |

### Feature Dimensions Summary
| Feature Set | Dim | Source | Precomputed? |
|-------------|-----|--------|--------------|
| CLIP spatial | 1024 | ViT-L/14 CLS | No (live) |
| fast_semantic | 58 | InsightFace+DeepFace+LibreFace+MediaPipe | Yes (.pt) |
| forensic | 83 | SegFormer+pixel forensics | Yes (.pt) |
| consistency_rules | 23 | Derived from fast_semantic (V7 rules) | Computed in model |

### Two EDL Modes
| Setting | Simple (`nesy_fusion: false`) | NeSy-EDL (`nesy_fusion: true`) |
|---------|-------------------------------|-------------------------------|
| Evidence fusion | `total = spatial + σ(g1)*concept + σ(g2)*causal` | CMEF: confidence-modulated weighted sum |
| Gate mechanism | Static scalar params (concept_init, causal_init) | Learned gates conditioned on evidence confidence |
| Loss | EDL NLL + KL divergence | EDL NLL + KL + per-branch aux + disagreement |
| Extra params | 2 scalars | CMEF module |
| Best result | April 8: 89.9% CDF AUC | April 11: 88.4% CDF AUC |

---

## Best Run Configurations

### Run 1: April 8 — 89.9% CDF AUC (commit `212a8db`)
- `nesy_fusion: false` (simple scalar gates)
- `combined_dim: 122` (fast 58 + vlm 64, but VLM disabled → 64 zeros)
- `use_refined_features: true`, `semantic_attributes.enabled: false`
- `class_weights: [1.8, 0.6]`
- `use_data_augmentation: true`, `train_dataset: [FaceForensics++]`
- `train_batchSize: 128`, `weight_decay: 0.0`, `manualSeed: 1024`
- `quality_lower: 65`
- Result: FF++ 0.955, CDF 0.899, avg 0.926

### Run 2: April 11 — 88.4% CDF AUC (commit `e3a80f8`)
- `nesy_fusion: true` (CMEF + PBAS + IBDC)
- `combined_dim: 58` (fast only, no VLM)
- `use_refined_features: false`
- `class_weights: [1.8, 0.6]`
- `use_data_augmentation: true`, `train_dataset: [FaceForensics++]`
- `train_batchSize: 300`, `weight_decay: 0.0`, `manualSeed: 1024`
- `quality_lower: 65`
- Result: FF++ 0.958, CDF 0.884, avg 0.921

### Key Differences Between Best Runs
| Setting | April 8 (89.9% CDF) | April 11 (88.4% CDF) |
|---------|---------------------|----------------------|
| `nesy_fusion` | `false` | `true` |
| `combined_dim` | `122` (58+64 zeros) | `58` |
| `batch_size` | `128` | `300` |
| Everything else | identical | identical |

---

## Performance Drop Root Cause (RESOLVED — April 14)

### Problem
Epoch 0 AUC dropped from **0.85/0.82** (FF++/CDF, April 11) to **0.79/0.68**.

### Root Cause — Two architectural changes after working commit `e3a80f8`:

**1. Projection head removed** (`nesy_defake_detector.py`)
- Working: `_make_projection` = `Linear(1024→1024) → LayerNorm → GELU`
- Broken: `_make_adapter` = `nn.Identity()` (pass-through)

**2. Fusion MLP gutted** (`multimodal_fusion.py`)
- Working: `Linear(concat→fused) → LN → ReLU → Dropout → Linear(fused→proj) → LN`
- Broken (single-branch): Just `LayerNorm(proj_dim)` — zero learned parameters

**Impact:** Pipeline became `frozen_CLIP → LN → linear_classifier` — a pure linear probe.
The projection head provides ~1M trainable parameters for task-specific feature adaptation.

**Fix:** Reverted both files to e3a80f8. Epoch 0 AUC back to **0.83**.

**Note:** The fusion MLP change didn't actually matter for the EDL path (fusion is bypassed),
but the projection head removal was the critical regression. The fusion revert was done for
safety since both files were changed together.

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

---

## Potential Improvements (Safe, Incremental)

### Option 1: Residual Projection (low risk)
Add a skip connection around the projection head:
```python
# Current: x = projection(clip_out)
# Proposed: x = clip_out + alpha * projection(clip_out)   # alpha learnable, init=0.1
```

### Option 2: Lightweight Bottleneck Adapter (low risk)
Replace `Linear(1024→1024)` projection with a bottleneck:
```python
# Linear(1024→256) → GELU → Linear(256→1024) → LayerNorm + skip connection
```

### Option 3: Unfreeze Top-K CLIP Layers (medium risk)
Unfreeze last 1-2 transformer blocks. Only try after OTF augmentation is working.

### Option 4: Remove Unused Fusion Module
Since fusion is bypassed in the EDL path, removing it would clean up the optimizer
and save ~1M params of memory. Low risk but cosmetic.

## Suggested Advanced Improvements by Gemini
Here are a few architectural and theoretical optimizations that align with your current framework:

1. Re-evaluate Dropout in the EDL Path
The Issue: Your MultiTaskHead classifier uses Dropout(0.3). Standard Dropout introduces stochasticity into the forward pass. However, EDL maps features deterministically to Dirichlet distribution parameters (evidence). Injecting random noise right before calculating deterministic uncertainty can cause the model's confidence scores to fluctuate wildly and degrade the calibration of the EDL loss.

The Fix: Remove Dropout from the layers immediately preceding the spatial_logits, or replace it with a lighter regularization technique like Weight Decay. If you genuinely want to capture epistemic uncertainty via Dropout, you would need to use MC-Dropout (multiple forward passes at inference), which fundamentally changes your EDL setup.

2. Feature-Conditioned Gating for "Simple" Mode
The Issue: In your simple NeSy-EDL mode (nesy_fusion: false), you use 2 static scalar parameters for the gates (concept_gate, causal_gate). A static scalar means the network applies the exact same weighting to the semantic/causal branches regardless of what is in the image.

The Fix: Upgrade these scalars to very lightweight, feature-conditioned gates. Pass the spatial_raw token through a tiny linear layer (Linear(1024 -> 2) -> Sigmoid) to dynamically output the concept_gate and causal_gate values per image. This allows the model to trust the causal branch more on highly compressed images, and the concept branch more on clear, frontal faces.

4. EDL KL-Divergence AnnealingEDL relies on a KL-divergence term to shrink the evidence of incorrect classes to zero. If this penalty is applied too strongly early in training, the model cannot explore the feature space and becomes underconfident. Ensure you have a global step-based annealing schedule for the KL term (e.g., starting at $0.0$ and scaling up to $1.0$ over the first 20% of epochs).
---

## Key Files
| File | Role |
|------|------|
| `training/detectors/nesy_defake_detector.py` | Main detector, projection head, EDL fusion, evidence gates |
| `training/networks/nesy_defake/fusion/multimodal_fusion.py` | Fusion MLP (unused in EDL path) |
| `training/networks/nesy_defake/classifiers/multitask_head.py` | Classifier: per-task MLP (LN→GELU→Dropout) |
| `training/networks/nesy_defake/foundation_models/spatial_extractor.py` | CLIP/DINOv2/EVA/SigLIP backbone |
| `training/networks/nesy_defake/losses/edl_loss.py` | Simple EDL loss (NLL + KL) |
| `training/networks/nesy_defake/losses/nesy_edl_loss.py` | NeSy-EDL loss (CMEF + PBAS + IBDC) |
| `training/dataset/nesy_defake_dataset.py` | Dataset with K-augment, feature loading |
| `training/dataset/pixel_forensic_extractor.py` | OTF forensic features (indices 30-82, dormant) |
| `training/networks/nesy_defake/ccv_branch.py` | CCV causal branch, forensic anomaly detector |
| `training/networks/nesy_defake/concept_branch.py` | Concept branch, consistency rule violations → evidence |
| `training/networks/nesy_defake/semantic/consistency_rules_v7.py` | 23 consistency rules |
| `training/networks/nesy_defake/semantic/refined_attributes.py` | Feature name registries (58 fast + 64 VLM) |
| `preprocessing/forensic_helpers.py` | All 83 forensic feature definitions |
| `preprocessing/precompute_fast_semantic.py` | InsightFace+DeepFace+LibreFace+MediaPipe |
| `augment_and_precompute.sh` | Full pipeline: augment + precompute (K configurable) |
| `scripts/inspect_precomputed_features.py` | Feature diagnostic tool |
| `scripts/clean_dataset_json.py` | Remove stray entries from dataset JSON |
| `training/train.py` | Training entry point, seed init, config snapshot to logs |

## Working Commit Reference
- **e3a80f8**: Last known good state (detector + fusion architecture)
- **212a8db**: April 8 best run code (added CCV branch support + class_weights to EDL)
- **Current branch:** `causal_discovery`

## Config Files
| Config | Purpose |
|--------|---------|
| `nesy_defake_ablation4_ccv.yaml` | Full config (VLM 122-d), matches April 8 best run |
| `nesy_defake_ablation4_ccv_novlm.yaml` | No VLM (58-d fast only), matches April 11 run |
| `nesy_defake_ablation4_ccv_novlm_otf.yaml` | No VLM + OTF forensic hybrid (experimental) |
| `nesy_defake_ablation4_ccv_novlm_k12.yaml` | K=12 pre-augmented (abandoned — feature mismatch) |


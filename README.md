# NeSyDeFake — Causal Discovery Module: Complete Design Summary

## Project Overview

NeSyDeFake is a neuro-symbolic deepfake detection framework built on top of DeepfakeBench. The system operates **per-frame** — each frame is an independent sample. There is no temporal branch.

Modules:
- **M1**: Foundation Features (spatial CLIP + frequency CLIP via FAD) — per-frame
- **M2**: Probabilistic Circuits (uncertainty estimation — deferred)
- **M3**: Causal Discovery (THE main contribution — this document) — per-frame
- **M4**: Sparse Autoencoder (monosemantic feature interpretation) — per-frame
- **M5**: Multi-task Classifier — per-frame

Current baseline: Spatial CLIP-L achieves 87.98 AUC on FF++, 75.51 on CDF-v2. Frequency branch (FAD-CLIP) implemented.

---

## The Core Idea: Causal Discovery Between Neural and Symbolic Features

### What makes this novel

We learn a **formal causal graph** connecting CLIP's latent features (compressed via Sparse Autoencoder) to grounded facial semantic attributes (extracted by SOTA face analysis models). This is a **neuro-symbolic bridge** — the graph's structural equations capture how neural representations causally relate to interpretable facial semantics.

### Why it generalizes

Causal relationships between facial attributes are properties of **real human faces** (physics + biology). They hold regardless of dataset, identity, or forgery method. A forgery that breaks the causal relationship "head pose → expression visibility" will be detected whether it's a face-swap, reenactment, or diffusion-generated face. The causal graph learned from FF++ transfers because the **structure is universal** — only the violation patterns are method-specific.

### The argument against simpler approaches

- If CLIP (300M params) already saturates in-domain (~99% AUC on FF++), why add a causal module? Because CLIP **doesn't generalize** — it drops to 75% on CDF-v2. The causal module provides a **domain-invariant regularization signal** based on universal facial structure.
- Why not learn concepts from scratch? A reviewer will ask "what grounds concept #3 to lighting?" — answer: nothing. We use **real measurements from SOTA face analysis models**, not learned fictional concepts.

---

## Architecture: Per-Frame Three-Phase Pipeline

**Every component operates on a single frame independently.** No temporal aggregation, no video-level pooling.

### Phase 1: Learn Causal Graph (DAGMA-DCE)

**Input variables per frame:** V = [Z_sae ; S_semantic]
- Z_sae = 128 spatial SAE features + 128 frequency SAE features = **256-d** (compressed from 1024-d CLIP via Sparse Autoencoder)
- S_semantic = **73 cached semantic attributes** (from preprocessing — loaded from disk per frame)
- **Total: 329 causal variables per frame**

**Algorithm:** DAGMA-DCE (Differentiable Causal Effect), chosen over NOTEARS because:
- Edge weights = differentiable causal effect (interpretable strength)
- Not opaque proxies like NOTEARS
- Log-det acyclicity penalty → 0 iff DAG

**Training:** Graph learned on **both real and fake** frames. Structural equations fit to **reals only**.
- Why both? Real-only → one-class detector → false positives on unusual reals. Seeing fakes teaches WHICH violations are discriminative.
- Why structural eqs on reals only? Real faces follow universal causal structure (physics/biology). Structural equations capture this truth.

### Phase 2: Compute Violations (per frame)

For each variable V_i in the current frame:
```
residual_i = (V_i - f_i(Parents_G(V_i); theta_i))^2
```
High residual = the causal relationship is broken in THIS frame = forgery signal.

### Phase 3: Gated Fusion with Classification (per frame)

Violation vector (329-d residuals) + CLIP features → Gated Fusion → Real/Fake prediction for this frame

### Training Objective

```
L = L_cls + λ1*L_structural(reals) + λ2*L_dag + λ3*L_sparsity + λ4*L_contrastive(violations)
```

- L_cls: Binary cross-entropy on per-frame prediction
- L_structural: MSE of structural equations on REAL frames only
- L_dag: DAGMA acyclicity constraint (log-det penalty)
- L_sparsity: L1 on adjacency matrix
- L_contrastive: Push real frame violations down, fake frame violations up

---

## Semantic Feature Design: Identity-Invariant, Fully Semantic, Per-Frame

### Critical design decision: NO identity features

ArcFace embeddings, identity drift, and identity consistency are **excluded**. Reason: they make the causal graph identity-dependent. Testing on unseen identities (different dataset = different people) causes every identity-connected edge to fire as a violation causing massive false positives. The causal graph must be **identity-invariant**.

### Critical design decision: NO temporal/video-level features

The system is **per-frame**. No temporal summaries (mean/std/delta across frames), no video-level aggregation. Each frame's 73 semantic features stand alone as causal variables. This is consistent with the spatial and frequency branches which also operate per-frame.

### What we extract (73 per-frame features)

| Source | Features | Count | Semantic Question |
|--------|----------|-------|-------------------|
| **DeepFace** | Emotion probabilities | 7 | "What expression is showing?" |
| | Age (normalized 0-1) | 1 | "How old does this person appear?" |
| | Gender probabilities | 2 | "Perceived gender?" |
| | Race probabilities | 6 | "Perceived ethnicity distribution?" |
| **InsightFace** | Head pose (yaw/pitch/roll) | 3 | "Where is the person facing?" |
| | Face quality (det_score) | 1 | "How clearly visible?" |
| | Anti-spoof score | 1 | "Does this look real?" |
| **MediaPipe** | 52 blendshape coefficients | 52 | "What is each facial muscle doing?" |

**Total per-frame: 73 semantic attributes**

### Feature index ranges

```
indices  0:16  → DeepFace (emotion 7, age 1, gender 2, race 6)
indices 16:21  → InsightFace (pose 3, quality 1, antispoof 1)
indices 21:73  → MediaPipe blendshapes (52)
```

### Why MediaPipe blendshapes are important

The 52 blendshapes (ARKit-compatible) measure specific facial muscle activations [0,1]:

1. **Bilateral symmetry**: 20 left/right pairs (browDownLeft/Right, mouthSmileLeft/Right, eyeBlinkLeft/Right). Real faces have strong L↔R causal coupling (same nerve, same muscle group). Face-swaps break this at the blending boundary → causal violation.

2. **Expression coherence**: Real smiles activate mouthSmile + cheekSquint together (Duchenne pattern). Fakes may get mouth right but miss cheek muscles. The causal graph learns these multi-variable constraints.

3. **Complements DeepFace**: DeepFace emotion gives 7 coarse categories. Blendshapes give the fine-grained muscle activations that PRODUCE those emotions. Graph can model: blendshapes → emotion, violations = forgery signature.

### Causal variable count for DAGMA-DCE

73 semantic + 256 SAE = **329 total per frame**. This is manageable for DAGMA-DCE. If needed during ablation:
- Use all 329
- PCA on 52 blendshapes → ~15 components → reduces to ~292
- Curated subset of most discriminative features

---

## Preprocessing: Semantic Feature Extraction

### File: preprocessing/extract_semantic_features.py

Mirrors DeepfakeBench's preprocessing pattern. Runs on already-extracted face crops (256x256 PNGs from preprocess.py), saves .npz files alongside frames/landmarks.

**Output structure:**
```
{preprocessed_root}/{dataset}/{sub_dataset}/
  frames/{video_name}/000.png, 001.png, ...          ← existing from preprocess.py
  landmarks/{video_name}/000.npy, ...                 ← existing from preprocess.py
  semantic_features/{video_name}.npz                  ← NEW from extract_semantic_features.py
```

Each .npz contains:
- per_frame: (T, 73) float32 — per-frame semantic attributes (T = number of extracted frames, typically 32)
- summary: (219,) float32 — video-level temporal summaries (stored but NOT used in per-frame pipeline; kept for potential future ablation only)
- feature_names: list of 73 per-frame feature names
- summary_names: list of 219 summary feature names
- extraction_info: metadata dict

**At training time, only per_frame[frame_idx] is loaded — a single (73,) vector per sample.**

### Usage

```bash
# Prerequisites
pip install deepface insightface onnxruntime-gpu mediapipe

# Download MediaPipe model (one time)
wget -O face_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

# Full extraction (restrict GPU to avoid using GPU 0)
CUDA_VISIBLE_DEVICES=2 python extract_semantic_features.py --dataset_name FaceForensics++ --comp c23

# Resume interrupted run
CUDA_VISIBLE_DEVICES=2 python extract_semantic_features.py --dataset_name FaceForensics++ --skip_existing

# DeepFace only (fastest, 16 features)
python extract_semantic_features.py --deepface_only
```

**Important: DeepFace uses detector_backend='skip'** because input is already aligned/cropped faces. No redundant face detection.

### Known issues (and fixes applied)
- **InsightFace + NumPy 2.0**: np.sctypes removed. **Fixed** with monkey-patch in extract script (no NumPy downgrade needed). Also patches np.bool, np.int, np.float, etc.
- **Speed**: ~3.4s per video (32 frames) with all 3 extractors. DeepFace is the bottleneck.
- **GPU memory**: DeepFace/TensorFlow grabs GPU memory. Use CUDA_VISIBLE_DEVICES to restrict. InsightFace capped at 2GB via ONNX Runtime config.
- **MediaPipe**: Runs CPU-only (XNNPACK). Cannot use GPU.

### File: preprocessing/load_semantic_features.py

Training-time loader for per-frame usage. Zero runtime cost (~0.1ms per sample).

```python
# In dataset __getitem__:
from preprocessing.load_semantic_features import load_semantic_features

# Load the .npz for this video
semantic_path = video_dir.parent / 'semantic_features' / f'{video_name}.npz'
semantic = load_semantic_features(semantic_path, return_per_frame=True)

# Get THIS frame's 73 semantic features
frame_semantics = semantic['per_frame'][frame_idx]  # (73,) float32
data_dict['semantic_attrs'] = torch.from_numpy(frame_semantics)
```

Handles backward compatibility: if some datasets extracted with fewer features, loader zero-pads missing columns.

---

## Dual-Branch Causal Discovery (Per-Frame)

Both spatial and frequency CLIP branches feed into the causal graph. Everything is per-frame:

```
                        For a SINGLE FRAME:

Spatial CLIP (1024-d) → SAE → 128 sparse features ──┐
                                                      ├→ V = [Z_sae(256); S_semantic(73)]
Frequency CLIP FAD (1024-d) → SAE → 128 features ───┘       = 329 causal variables
                                                              → DAGMA-DCE causal graph
                                                              → Violation vector (329-d)
                                                              → Gated fusion → Real/Fake

Cached semantic attrs (73 per frame, from .npz) ────────────→ (feeds into V above)
```

Each branch captures complementary violation patterns:
- **Spatial**: appearance-level inconsistencies (skin tone mismatch, expression incoherence)
- **Frequency**: artifact-level inconsistencies (blending boundaries, compression patterns)

---

## Frequency Branch: FAD-CLIP (Adaptive Frequency Enhancement)

**File:** networks/nesy_defake/foundation_models/frequency_feature_extractor.py

### Problem solved
Previous v3 fed ONLY high-frequency band to CLIP → residual signal with no spatial context → CLIP couldn't locate face boundary → AUC ~0.65.

### v4 solution
**Additive emphasis**: output = original + α * high_freq_residual
- CLIP sees recognizable face (pretraining works) + frequency artifacts amplified (detection works)
- Learnable emphasis weight α starts small (0.1), grows during training
- DCT → learnable bandpass filters → iDCT → frequency bands (F3Net-derived)

### Modes
- single: orig + α * high_band (best for face-swap blending)
- multi: orig + α_low * low + α_mid * mid + α_high * high (per-band emphasis)

### Critical implementation details
- Dataset delivers RAW [0,1] pixels (keep_raw: true) — DCT bandpass designed for pixel-range coefficients
- Clamp to [-0.5, 1.5] to prevent NaN in CLIP's ConvolutionBackward0
- CLIP normalization applied AFTER enhancement
- Bandpass filter values clamped to [0,1] — any value >1 amplifies DC (~1000+) → iDCT explosion → NaN

---

## Expected Performance

| Dataset | Spatial Only | + Frequency | + Causal Module |
|---------|-------------|-------------|-----------------|
| FF++ (in-domain) | 97-99 AUC | 97-99 AUC | 97-99 AUC |
| CDF-v2 (cross) | 85-90 AUC | 87-92 AUC | **94-96 AUC** |
| DFDC (cross) | 80-85 AUC | 82-87 AUC | **88-92 AUC** |
| Unseen method | 78-83 AUC | 80-85 AUC | **86-90 AUC** |

The causal module's value shows in **cross-dataset**, not in-domain (CLIP already saturates in-domain).

---

## Paper Contributions

1. **Neuro-Symbolic Bridge**: Formal causal graph connecting CLIP latent features (via SAE) to grounded facial semantic attributes from SOTA face analysis models — all operating per-frame
2. **Domain-Invariant Signal**: Causal violations are method-agnostic — trained on FF++ forgery types, transfers to unseen methods because violations defined in attribute space
3. **Dual-Branch Discovery**: Separate causal graphs for spatial and frequency CLIP branches capture complementary violation patterns
4. **Interpretable Detection**: Full explanation trace: violated attribute → causal parent SAE feature → human-readable description

---

## Files Delivered

| File | Purpose |
|------|---------|
| preprocessing/extract_semantic_features.py | Offline semantic extraction (DeepFace + InsightFace + MediaPipe) with NumPy 2.0 fix |
| preprocessing/load_semantic_features.py | Training-time per-frame loader (zero cost) |
| causal/causal_discovery_v2.py | Main causal module orchestrator |
| causal/attribute_extractor.py | Runtime attribute extraction (geometric only — may be superseded by cached semantics) |
| causal/dag_learner.py | DAGMA-DCE DAG learning |
| causal/INTEGRATION_GUIDE.py | Wiring instructions for detector |
| ROADMAP.md | 8-week implementation plan |
| nesydefake_architecture.pptx | 6-slide architecture presentation (needs update for per-frame + 73 features) |

---

## Current Status (as of March 2, 2026)

- DONE: Semantic extraction running on FF++ and Celeb-DF datasets with all 73/73 features
- DONE: DeepFace (16) + InsightFace (5) + MediaPipe blendshapes (52) all active
- DONE: InsightFace NumPy 2.0 incompatibility fixed via monkey-patch (no downgrade needed)
- DONE: Architecture slides created (6 slides, needs update for per-frame + 73 features)
- PENDING: Causal module integration into detector (next step after preprocessing completes)
- PENDING: SAE training (needed before causal graph can use Z_sae features)
- PENDING: Update slides for per-frame paradigm
- PENDING: Re-extract Celeb-DF-v1/v2 with all 73 features (earlier runs had 68, missing InsightFace)

### Immediate Next Steps

1. Let semantic extraction complete on all datasets (FF++, CDF-v1, CDF-v2, DFDC, etc.)
2. Re-extract CDF-v1/v2 (delete old .npz files, re-run with InsightFace fix active)
3. Integrate load_semantic_features.py into dataset __getitem__ — per-frame loading
4. Wire causal module into detector (follow INTEGRATION_GUIDE.py, adapted for per-frame)
5. Train spatial+frequency baseline
6. Enable causal module

---

## Key Design Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Per-frame vs per-video | **Per-frame** | Spatial and frequency branches are per-frame; no temporal branch |
| Identity features | **Excluded** | Break identity-invariance of causal graph |
| Temporal summaries | **Not used** (stored but ignored) | Per-frame pipeline; no video-level aggregation |
| Temporal branch | **Removed** | Was not working; system is purely per-frame |
| Feature extraction | **Offline cached** | Extract once with DeepFace/InsightFace/MediaPipe, load from .npz at training |
| Causal algorithm | **DAGMA-DCE** | Interpretable edge weights as differentiable causal effects |
| Structural equations | **Fit to reals only** | Real faces follow universal causal structure |
| Graph training data | **Both real and fake** | Prevents one-class detector failure mode |

---

## Key Config Settings

```yaml
# In config.yaml
preprocess:
  dataset_root_path:
    default: '/data/umar/Datasets'
  output_root_path:
    default: '/data/umar/Datasets/preprocessed'
  num_frames:
    default: 32
  mode:
    default: 'fixed_num_frames'
```

### GPU Notes
- Use CUDA_VISIBLE_DEVICES=2 (or 3) for preprocessing to keep GPU 0 free for training
- DeepFace/TensorFlow grabs GPU memory unnecessarily — restrict with env variable
- InsightFace ONNX Runtime capped at 2GB GPU memory in script
- MediaPipe runs CPU-only (XNNPACK)
# NeSyDeFake Changes — April 9-10, 2026

## 1. No-VLM Ablation Config for CCV

**File:** `training/config/detector/nesy_defake_ablation4_ccv_novlm.yaml`

Tests whether expensive FaceBench VLM features (64-d) improve detection or if fast-only semantic features (58-d) are sufficient.

**Changes from base CCV config:**
- `semantic_attributes.enabled: false` (VLM disabled)
- `use_refined_features: false`
- `concept_branch.combined_dim: 58` (fast only, was 122)
- `causal_branch.combined_dim: 58`

**Dataset fix** (`training/dataset/nesy_defake_dataset.py`):
- Added `elif self.use_fast_semantic and not self.use_refined_features` path to return fast-only 58-d features as `precomputed_attrs` instead of `torch.zeros(1)` placeholder.

**Concept branch fix** (`training/networks/nesy_defake/concept_branch.py`):
- Consistency rules index into the full 122-d combined vector by position. With 58-d input, VLM indices (>=58) caused `IndexError`.
- Fix: pad `combined_features` to 122-d with zeros before passing to consistency rules. VLM-dependent rules produce ~0 (zero × zero = no signal). The concept MLP still receives the original 58-d features.

## 2. NeurIPS Paper: NeSy-EDL Section Rewrite

**File:** `Latex_NeSy_DeFake_NeurIPS_2026/neurips_2026.tex`

Replaced Section 3.5 "Evidential deep learning with gated evidence fusion" (standard gated addition) with "Neuro-symbolic evidential deep learning (NeSy-EDL)" describing the three novel components:

- **CMEF** (Confidence-Modulated Evidence Fusion): Equations for per-sample dynamic weighting via confidence gates φ(S_b) = σ((S_b - K) / τ), with analysis of OOD trust allocation property.
- **PBAS** (Per-Branch Auxiliary Supervision): Each branch gets standalone EDL loss to prevent dominant branch gradient suppression.
- **IBDC** (Inter-Branch Disagreement Calibration): Cosine disagreement between branch Dirichlet means calibrated against fused uncertainty via BCE.
- **Total objective**: L = L_nll + λ_kl·β(t)·L_kl + λ_avu·L_avu + λ_aux·L_aux + λ_bdc·L_bdc

Updated figure caption and evidence summary table to reference CMEF instead of "gated addition".

## 3. Frequency Branch: MAE and EVA-02 Backbone Integration

### Problem with existing frequency branch

The frequency branch used FAD (F3Net Adaptive frequency enhancement) + DINOv2/CLIP as backbone. Fundamental issue: **CLIP and DINOv2 were pretrained on natural image semantics**, their representations are optimized to capture *what* is in the image, not *how* the pixels were generated. Frequency artifacts enhanced by FAD get washed out by the frozen semantic representations.

When both branches used CLIP, the detector even **shared the backbone** — making spatial and frequency features near-identical.

### Solution: Forensic-aligned pretrained backbones

Two new backbone options whose pretraining objectives align with pixel-level artifact detection:

### 3a. FAD + MAE ViT-L/16

**Config:** `training/config/detector/nesy_defake_ablation4_ccv_mae.yaml`

- **Model:** `facebook/vit-mae-large` (HuggingFace transformers)
- **Pretraining:** Masked Autoencoder — reconstructs 75% masked patches from pixels
- **Why:** Reconstruction objective forces MAE to learn exact pixel-level patterns (texture, noise, compression traces). CLIP abstracts all of this away.
- **Complementarity:** CLIP sees "what is this face?" (semantic). MAE sees "how do these pixels look?" (perceptual). These are maximally complementary — opposite ends of the semantic↔perceptual spectrum.
- **Output:** 1024-d (drop-in replacement, same as CLIP)

**Implementation** (`training/networks/nesy_defake/foundation_models/frequency_extractor.py`):
- `_MAEEncoderWrapper` class: Wraps HuggingFace `ViTMAEModel` for full-image feature extraction. `ViTMAEModel` always masks patches in its forward (designed for pretraining); the wrapper manually creates patch embeddings → prepends CLS token → adds position embeddings → runs encoder → returns CLS output.
- `_build_fad_mae()`: Loads MAE via HuggingFace, creates FAD front-end with ImageNet normalization.
- Uses shared `_forward_fad_timm()` for the 3-stage precision pipeline.

### 3b. FAD + EVA-02 ViT-L/14

**Config:** `training/config/detector/nesy_defake_ablation4_ccv_eva02.yaml`

- **Model:** `eva02_large_patch14_clip_224` (timm)
- **Pretraining:** MIM (masked image modeling) with CLIP feature targets + CLIP distillation
- **Why:** Hybrid pretraining gives both reconstruction sensitivity (from MIM) AND semantic awareness (from CLIP teacher). Features capture both the artifact and its context.
- **Trade-off vs MAE:** Less complementary to spatial CLIP (partial objective overlap) but potentially stronger standalone encoder.
- **Output:** 1024-d (drop-in)

**Implementation:**
- `_build_fad_eva02()`: Loads EVA-02 via timm with `num_classes=0`.
- Uses same `_forward_fad_timm()` shared forward.

### Shared forward: `_forward_fad_timm()`

Three-stage precision strategy (same as existing FAD+CLIP/DINOv2):
1. **Stage 1 — FAD front-end in FP32**: DCT → learnable bandpass → iDCT → emphasis → normalize
2. **Stage 2 — Backbone in BF16**: Overflow-safe, same tensor core throughput as FP16 on H200
3. **Stage 3 — SafeLayerNorm in FP32**: Gradient clamping ±100

Both MAE and EVA-02 (and any future timm backbone) use this same forward.

### Config differences from spatial-only CCV

| Setting | Spatial-only CCV | MAE/EVA-02 CCV |
|---------|-----------------|-----------------|
| `active_branches` | `['spatial']` | `['spatial', 'frequency']` |
| `frequency.name` | `fad_dinov2` | `fad_mae` / `fad_eva02` |
| `frequency.train_layernorms` | `false` | `true` |
| `frequency normalization` | CLIP values | ImageNet (MAE) / CLIP (EVA-02) |
| `train_batchSize` | 128 | 64 (two ViT-L in memory) |
| Backbone sharing | N/A | `false` (different models) |
| Fusion | passthrough (1 branch) | concat→linear (2048→1024) |

### Model comparison

| Backbone | Pretraining Objective | Pixel Sensitivity | CLIP Complementarity | Params |
|----------|----------------------|-------------------|---------------------|--------|
| CLIP ViT-L/14 (spatial) | Image-text contrastive | Low | — | 304M |
| DINOv2 ViT-L/14 (old freq) | Self-supervised | Medium | Medium | 304M |
| **MAE ViT-L/16** (new) | Pixel reconstruction | **High** | **High** | 304M |
| **EVA-02 ViT-L/14** (new) | MIM + CLIP distill | High | Medium | 304M |

## 4. NeSy-EDL Verification

Verified all three NeSy-EDL components instantiate and produce correct outputs:
```
Fused shape: (4, 2), Tau: 1.0000
Loss: 1.2935, Uncertainty: 0.4387
loss_aux: 2.8635, loss_bdc: 0.5843
```

## 5. Full Detector Smoke Tests

Both new frequency backbone variants build and produce correct output shapes:

| Config | Total Params | Trainable | Freq Backbone | Shared |
|--------|-------------|-----------|---------------|--------|
| CCV + MAE | 616,875,522 | 10,596,866 | fad_mae | No |
| CCV + EVA-02 | 616,891,842 | 10,777,058 | fad_eva02 | No |

## Files Modified

| File | Change |
|------|--------|
| `training/networks/nesy_defake/foundation_models/frequency_extractor.py` | Added `_MAEEncoderWrapper`, `_build_fad_mae`, `_build_fad_eva02`, `_forward_fad_timm`; registered in builder/dispatch dicts |
| `training/networks/nesy_defake/concept_branch.py` | Pad combined_features to 122-d for consistency rules when input < 122 |
| `training/dataset/nesy_defake_dataset.py` | Added fast-only semantic loading path when `use_refined_features: false` |
| `Latex_NeSy_DeFake_NeurIPS_2026/neurips_2026.tex` | Rewrote Section 3.5 with NeSy-EDL (CMEF/PBAS/IBDC) |

## Files Created

| File | Purpose |
|------|---------|
| `training/config/detector/nesy_defake_ablation4_ccv_novlm.yaml` | CCV without FaceBench VLM features (58-d fast only) |
| `training/config/detector/nesy_defake_ablation4_ccv_mae.yaml` | CCV + FAD+MAE frequency branch |
| `training/config/detector/nesy_defake_ablation4_ccv_eva02.yaml` | CCV + FAD+EVA-02 frequency branch |

## Updated Ablation Matrix

| # | Config | Components | What It Tests |
|---|--------|-----------|---------------|
| 1 | ablation1 | CLIP + CE | Baseline: frozen CLIP linear probe |
| 2 | ablation2_edl | CLIP + EDL | +uncertainty-aware loss |
| 3 | ablation3_concept | CLIP + EDL + Concept(122-d + 23 rules) | +symbolic consistency rules |
| 4a | ablation4_ccv | CLIP + NeSy-EDL + Concept + CCV | +learned constraints, anomaly, counterfactual, CMEF |
| 4a-novlm | ablation4_ccv_novlm | Same as 4a but fast-only (58-d) | VLM features value test |
| 4a-mae | ablation4_ccv_mae | 4a + FAD+MAE freq branch | +pixel-reconstruction frequency features |
| 4a-eva02 | ablation4_ccv_eva02 | 4a + FAD+EVA-02 freq branch | +hybrid MIM/CLIP frequency features |
| 4b | ablation4_causal | CLIP + NeSy-EDL + Concept + ImprovedSCM | +nonlinear SCMs, DAG, CMEF |

# April 14 — Performance Drop Root Cause & Recovery Plan

## Problem
Epoch 0 AUC dropped from **0.85/0.82** (FF++/CDF, April 11) to **0.79/0.68** with seemingly the same config.

## Root Cause
Three architectural changes made AFTER working commit `e3a80f8` are the culprits — **NOT** the dataset/feature loading changes:

### 1. Projection head removed (`nesy_defake_detector.py`)
- **Working (e3a80f8):** `_make_projection` = `Linear(1024→1024) → LayerNorm → GELU`
- **Current:** `_make_adapter` = `nn.Identity()` (pass-through)
- **Impact:** No learned transformation between backbone and fusion.

### 2. Fusion MLP gutted (`multimodal_fusion.py`)  
- **Working (e3a80f8):** `Linear(concat→fused) → LN → ReLU → Dropout → Linear(fused→proj) → LN`
- **Current (single-branch):** Just `LayerNorm(proj_dim)` — no learned parameters
- **Impact:** The model lost its only trainable feature transformation between frozen CLIP and classifier. With spatial-only + Identity projection + LayerNorm fusion, the entire pipeline is `frozen_CLIP → LN → classifier`, which is a pure linear probe with no adaptation capacity.

### 3. Spatial extractor additions (`spatial_extractor.py`)
- Added EVA-CLIP, SigLIP support. **Harmless for CLIP backbone** — no code path changes.

## Recovery Plan

### Step 1: Revert detector + fusion to working state
```bash
git checkout e3a80f8 -- training/detectors/nesy_defake_detector.py \
                        training/networks/nesy_defake/fusion/multimodal_fusion.py
```
This restores `_make_projection` (Linear→LN→GELU) and the 2-layer fusion MLP.

**Keep** the spatial_extractor.py changes (EVA/SigLIP support is additive, not breaking).

### Step 2: Verify baseline recovery
Run with `nesy_defake_ablation4_ccv_novlm_k12.yaml` (current config):
- `train_dataset: [FaceForensics++_augmented]` (balanced aug1-3)
- `use_data_augmentation: false`
- `k_augment: 0`
- Features load from original .pt (current `_parse_frame_path` behavior)

Expected: epoch 0 AUC should return to ~0.85/0.82.

### Step 3: OTF augmentation experiment (after baseline verified)
Change only:
- `train_dataset: [FaceForensics++]` (original, no aug entries)  
- `use_data_augmentation: true`
- `balance_classes: true` (weighted sampler for class balance)

This gives the backbone diverse augmented views each epoch while features stay from original .pt files. Should improve cross-dataset generalization.

### Step 4: K=12 experiment (after OTF verified)
If OTF works, try K=12 with matched features for zero-mismatch training.

## Two Separate Issues — Don't Conflate

### Issue A: Architecture regression (projection + fusion gutted)
- **Cause:** `_make_projection` → `nn.Identity()`, fusion MLP → `LayerNorm`
- **Fix:** Revert detector + fusion to e3a80f8
- **Confidence:** HIGH — this directly removes trainable capacity

### Issue B: Augmentation diversity + feature noise
- **Cause:** When K=12 with mixed seeds was tested, AUC was ~0.55 at epoch 0-1
- **This is a SEPARATE problem from Issue A.** Even after reverting architecture, K=12 mixed-seed augmentation may still underperform. Here's why:

The K=12 approach loads pre-augmented images where ALL features (fast_semantic 58-d + forensic 83-d) were precomputed on EACH augmented view. But the diagnostic data showed:
- 33/58 fast_semantic features have corr < 0.80 between original and augmented
- 75/83 forensic features are augmentation-sensitive (corr < 0.80)
- Expression features especially volatile: corr 0.05–0.27

**What this means for the classifier:**
When we use features computed on the augmented view, the classifier sees wildly different feature vectors for the SAME identity across different augmentation variants. The CLIP backbone features are augmentation-robust (that's what CLIP was trained for), but the semantic/forensic features are NOT. So the classifier gets conflicting signals:
- CLIP says: "same face, same identity" (robust to augmentation)
- Semantic features say: "different expression, different AUs, different pose" (sensitive to augmentation)
- Forensic features say: "different noise profile, different frequency content" (sensitive to augmentation)

This noise in the symbolic features may drown out the useful forensic/semantic signal that the concept branch and CCV causal branch rely on. The classifier can't learn stable concept rules or causal patterns when the feature vectors are inconsistent across views of the same face.

**Why the "bug" (loading original features for augmented images) actually WORKED:**
The old behavior gave the classifier STABLE semantic/forensic features (from the original clean image) while the backbone saw diverse augmented views. This is actually a smart division of labor:
- Backbone (CLIP): learns augmentation-invariant visual representations from diverse views
- Semantic/forensic features: provide stable, clean signal for the symbolic branches
- The backbone learns "what changes with augmentation doesn't matter for detection"
- The features tell the model "here's what the face ACTUALLY looks like"

**Conclusion:** The revert of Issue A (architecture) will restore baseline performance. But for augmentation experiments, we should KEEP loading original features (not augmented-view features) — the "bug" was actually the right design. OTF augmentation should only affect the backbone input, not the precomputed features.

### Augmentation Experiment Design (after baseline recovery)
1. **Baseline (Step 2):** augmented JSON + no OTF aug + original features → expect ~94% CDF AUC
2. **OTF aug (Step 3):** original JSON + OTF aug + balance_classes + original features → backbone sees diverse views, features stay clean
3. **K=12 matched features (DO NOT DO):** loading features computed on augmented views introduces too much noise into symbolic branches. The diagnostic data proves this.
4. **K=12 with original features (Step 4, optional):** augmented JSON with K=12 entries for more backbone diversity, but always load original .pt features. This is safe because only the backbone view changes, not the feature signal.

## Dataset/Feature Loading Changes (safe to keep)
- `_parse_frame_path()` helper — cleaner code, returns original video name (same behavior as before)
- `_build_augment_variant_map()` — dormant when `k_augment: 0`
- `pixel_forensic_otf` support — dormant when `pixel_otf: false`
- `clean_dataset_json.py` — removed stray non-frame entries from JSON
- `abstract_dataset.py` — unchanged from e3a80f8

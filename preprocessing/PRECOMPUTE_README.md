# Preprocessing Pipeline

## Overview

The preprocessing pipeline prepares data for NeSy-DeFake training in 3 steps:

1. **Augment real frames** — creates N augmented copies of each real frame using torchvision v2 transforms (GenD-style) to balance the 4:1 fake:real ratio in FF++.
2. **Precompute semantic features** — runs frozen Face-LLaVA 13B once over every frame, producing 211 FaceBench attribute probabilities per frame via batched per-attribute teacher-forcing.
3. **Precompute forensic features** — runs SegFormer face parsing + CPU-side pixel analysis, producing 30 forensic features per frame (boundary gradients, blur, symmetry, color, DCT, quality).

All outputs are saved as per-video `.pt` files alongside the `frames/` directory.

## Full Pipeline

```bash
bash augment_and_precompute.sh
```

Or step by step:

```bash
# Step 1: Augment real frames (3 copies)
CUDA_VISIBLE_DEVICES=1 python preprocessing/augment_real_frames.py \
    --detector_path training/config/detector/nesy_defake.yaml \
    --n_augmentations 3 --workers 16 --skip_existing

# Step 2: Precompute semantic features (211-d)
CUDA_VISIBLE_DEVICES=1 python preprocessing/precompute_semantic_features.py \
    --detector_path training/config/detector/nesy_defake.yaml \
    --attr_batch_size 128 --output_dir facellava_semantic \
    --skip_existing --device cuda:0

# Step 3: Precompute forensic features (30-d)
CUDA_VISIBLE_DEVICES=1 python preprocessing/precompute_forensic_features.py \
    --detector_path training/config/detector/nesy_defake.yaml \
    --batch_size 256 --output_dir forensic_features \
    --skip_existing --device cuda:0
```

After running, update `train_dataset` in `nesy_defake.yaml` to use `FaceForensics++_augmented`.

## Single-Image Validation

Before running the full pipeline, validate on individual images:

```bash
# Validate semantic extraction (uses the exact same code path as batch mode):
CUDA_VISIBLE_DEVICES=1 python preprocessing/precompute_semantic_features.py \
    --detector_path training/config/detector/nesy_defake.yaml \
    --demo /path/to/face.jpg --device cuda:0

# Or via demo_semantic.py (same thing, more options):
CUDA_VISIBLE_DEVICES=1 python preprocessing/demo_semantic.py \
    --demo /path/to/face.jpg --mode teacher_force \
    --detector_path training/config/detector/nesy_defake.yaml

# Compare against slow generative mode (211 individual model.generate calls):
CUDA_VISIBLE_DEVICES=1 python preprocessing/demo_semantic.py \
    --demo /path/to/face.jpg --mode generate

# Inspect an already-precomputed .pt file (no GPU needed):
python preprocessing/demo_semantic.py \
    --demo /path/to/frames/VIDEO_ID/000.png --from_precomputed
```

Each demo command produces a plot (`.png`) and text report (`.txt`) showing attribute scores.

## File Structure

```
preprocessing/
    config_utils.py                  # Shared config loading + video collection
    augment_real_frames.py           # FrameAugmenter (torchvision v2, GenD-style)
    precompute_semantic_features.py  # SemanticPrecomputer (Face-LLaVA teacher-forcing)
    precompute_forensic_features.py  # ForensicPrecomputer (SegFormer + CV features)
    forensic_helpers.py              # Pure CV functions (gradient, blur, DCT, etc.)
    demo_semantic.py                 # Single-image validation (teacher_force/generate/describe)
    demo_plot_helpers.py             # Shared plotting and text report helpers
```

## Output Layout

```
.../original_sequences/youtube/c23/
    frames/929/000.png                    # original frame
    frames_aug_1/929/000.png              # augmented copy 1
    frames_aug_2/929/000.png              # augmented copy 2
    facellava_semantic/929.pt             # {features: (N, 211), frame_paths: [...]}
    forensic_features/929.pt             # {features: (N, 30),  frame_paths: [...]}
```

## Key Classes

| Class | What it loads | What it produces |
|-------|--------------|-----------------|
| `FrameAugmenter` | Nothing (CPU-only torchvision transforms) | Augmented frame images |
| `SemanticPrecomputer` | Face-LLaVA 13B (loaded once, ~26GB) | 211 attribute probabilities per frame |
| `ForensicPrecomputer` | SegFormer + optional InsightFace/MediaPipe | 30 forensic features per frame |

## Semantic Extraction: How It Works

### Per-Attribute Teacher-Forcing (FaceBench-native approach)

Each of the 211 FaceBench attributes gets its own yes/no prompt:

```
<image>
Does this person have {attribute}?
Yes
No
Information not visible
Please directly select the appropriate option ... ASSISTANT: Yes
```

We teacher-force "Yes" as the answer and extract `P(attribute) = sigmoid(logit_yes - logit_no)` at the answer token position. This is the exact FaceBench-native evaluation protocol.

### Model Architecture (Face-LLaVA-v1.5-13B)

```
Input image (3, 336, 336)
    │
    ▼
┌──────────────────────────┐
│  CLIP ViT-L/14 @ 336px   │  Vision Tower (frozen)
│  Output: (576, 1024)      │  576 = 24×24 patches (no CLS)
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│  mm_projector (Linear)    │  Projects 1024 → 4096 (LLM hidden dim)
│  Output: (576, 4096)      │  ~4.5 MB in fp16
└──────────────────────────┘
    │
    ▼  visual embeddings replace <image> token in the text sequence
┌──────────────────────────────────────────────────────────────┐
│  LLaMA-13B (40 layers, 5120 hidden, 40 heads)               │
│  Input: [text_before_img] + [576 visual tokens] + [text_after]│
│  ≈ 636 tokens total per prompt                                │
│  Output logits: (636, 32000) per sequence                    │
└──────────────────────────────────────────────────────────────┘
    │
    ▼  extract logit_yes and logit_no at answer position
  P(attribute) = sigmoid(logit_yes - logit_no)
```

### Batching Strategy

**Per-attribute batching** (`attr_batch_size`): Multiple attribute prompts for the same image are batched into a single LLM forward pass. With `attr_batch_size=112`: `ceil(211/112) = 2` forward passes per image.

**Multi-image batching** (`image_batch_size`): Multiple images are tiled with attribute prompts. With `image_batch_size=N, attr_batch_size=112`: each pass processes `N × 112` sequences.

### Memory Analysis & OOM Fix

**The problem**: In the naive approach, we pass raw pixel tensors to the model for every sequence. With `attr_batch_size=112`:
- The vision tower (CLIP ViT-L) runs 112 times on the **same** image — completely redundant
- LLM output logits: `(112, 636, 32000) × 4 bytes ≈ 9.2 GB`
- This OOMs even with `image_batch_size=1` on a 140 GB H200

**The fix**: Two-stage pipeline that respects model sizes:

```
BEFORE (wasteful — OOMs even with image_batch_size=1):
  For each attr batch of 112:
    model(input_ids=[112 prompts], images=[112 × same pixel tensor])
    └─ internally: encode_images(112 copies) → 112 CLIP forward passes (redundant!)
    └─ LLM forward: (112, ~636, 5120) → logits (112, 636, 32000) = 9.2 GB

AFTER (two-stage pipeline):
  Stage 1 — CLIP ViT-L (~400M params, safe to batch):
    Batch-encode image_batch_size images at once through vision tower.
    visual_embs = model.encode_images(batch)  → (N, 576, 4096), ~4.5 MB each

  Stage 2 — LLaMA 13B (huge, loop per image):
    For each image's pre-computed visual_emb:
      For each attr batch of 112:
        text_emb = model.embed_tokens(text_ids)    → (112, ~60, 4096)
        inputs_embeds = merge(text_emb, visual_emb) → (112, ~636, 4096)
        model(inputs_embeds=inputs_embeds, images=None)  ← vision tower SKIPPED
        └─ logits (112, 636, 32000) = 9.2 GB peak (one image at a time)

  image_batch_size controls the CLIP batch, NOT the LLM batch.
  The LLM always processes one image's attributes at a time.
```

The key insights:
1. `model.forward()` checks `if inputs_embeds is None` — when provided, it skips `prepare_inputs_labels_for_multimodal()` entirely and goes straight to LLaMA.
2. Small models (CLIP ~400M) can be batched freely. Large models (LLaMA 13B) must loop per image to control peak memory.

### Feature Stability Analysis

Use `analyze_feature_stability.py` to verify that top-K FaceBench attributes are stable across all 32 frames of a video (they should be — hair color, gender, glasses don't change in a 10-second clip). If >90% of top-K features are common across frames, we can extract from 3 key frames and propagate, saving ~10x LLM cost.

```bash
python preprocessing/analyze_feature_stability.py \
    --detector_path training/config/detector/nesy_defake.yaml \
    --n_videos 2 --top_k 40 50 60 --image_batch_size 8
```

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

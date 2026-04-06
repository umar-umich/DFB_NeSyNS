#!/bin/bash
# Full pipeline: augment real frames + precompute all features
#
# Step 1: Augment real frames (3 copies each to balance 4:1 fake:real ratio)
# Step 2: Precompute combined semantic features (122-d) on single GPU
#         Tier A — Fast (58-d): InsightFace + MediaPipe + DeepFace
#         Tier B — VLM  (64-d): FaceBench Face-LLaVA (teacher-forced)
# Step 3: Precompute Tier 2 forensic features (30-d) for ALL frames
#
# After running, update train_dataset to use the _augmented JSON.

set -e
cd /data/umar/Repos/DFB_NeSyNS

CONFIG="training/config/detector/nesy_defake.yaml"
GPU=1

echo "============================================================"
echo "Step 1: Augmenting real frames (3 copies for 4:1 balance)"
echo "============================================================"
CUDA_VISIBLE_DEVICES=$GPU python preprocessing/augment_real_frames.py \
    --detector_path "$CONFIG" \
    --n_augmentations 3 \
    --workers 16 \
    --skip_existing

echo ""
echo "============================================================"
echo "Step 2: Precomputing combined semantic features (122-d)"
echo "  Tier A — Fast (58-d): InsightFace + MediaPipe + DeepFace"
echo "  Tier B — VLM  (64-d): FaceBench Face-LLaVA (teacher-forced)"
echo "============================================================"
CUDA_VISIBLE_DEVICES=1 python preprocessing/precompute_fast_semantic.py \
    --detector_path "$CONFIG" \
    --output_dir fast_semantic \
    # --skip_existing \
    --device cuda:0 \
    --vlm_attr_batch_size 64 \
    --vlm_image_batch_size 1

echo ""
echo "============================================================"
echo "Step 3: Precomputing forensic features for ALL frames"
echo "============================================================"
CUDA_VISIBLE_DEVICES=1 python preprocessing/precompute_forensic_features.py \
    --detector_path "$CONFIG" \
    --batch_size 256 \
    --output_dir forensic_features \
    --skip_existing \
    --device cuda:0

echo ""
echo "============================================================"
echo "Done! Next steps:"
echo "  1. Update train_dataset in nesy_defake.yaml to use"
echo "     FaceForensics++_augmented instead of FaceForensics++"
echo "  2. Run training as normal"
echo "============================================================"

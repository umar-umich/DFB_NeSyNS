#!/bin/bash
# Full pipeline: augment real frames + precompute all features
#
# Step 1: Augment real frames (3 copies each to balance 4:1 fake:real ratio)
# Step 2: Precompute fast semantic features (58-d) — InsightFace + LibreFace + MediaPipe (GPU)
# Step 3: Patch DeepFace features into fast_semantic .pt files (CPU-only, parallel)
# Step 4: Precompute forensic features (83-d)
# Step 5: Precompute FaceBench VLM features (64-d) — Face-LLaVA 13B (slowest)
#
# After Step 2 you can start training (DeepFace features are patched in later).
# Steps 3-4 are CPU/GPU and can run alongside training.
# Step 5 is the slowest (~30s/video).

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

# echo ""
# echo "============================================================"
# echo "Step 2: Precomputing fast semantic features (58-d)"
# echo "  InsightFace (GPU) + LibreFace (GPU) + MediaPipe (CPU)"
# echo "  DeepFace disabled for speed — patched in Step 3"
# echo "============================================================"
# PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=$GPU python preprocessing/precompute_fast_semantic.py \
#     --detector_path "$CONFIG" \
#     --output_dir fast_semantic \
#     --skip_existing \
#     --device cuda:0

echo ""
echo "============================================================"
echo "Step 2 complete! You can start training now."
echo ""
echo "Step 3: Patching DeepFace features (CPU-only, 8 processes)"
echo "  emotions + ethnicity entropy → indices 2-10"
echo "============================================================"
PYTHONWARNINGS=ignore python preprocessing/precompute_deepface_patch.py \
    --detector_path "$CONFIG" \
    --output_dir fast_semantic \
    --workers 16 \
    --skip_patched

# echo ""
# echo "============================================================"
# echo "Step 4: Precomputing forensic features (83-d)"
# echo "============================================================"
# CUDA_VISIBLE_DEVICES=$GPU python preprocessing/precompute_forensic_features.py \
#     --detector_path "$CONFIG" \
#     --batch_size 256 \
#     --workers 8 \
#     --output_dir forensic_features \
#     --skip_existing \
#     --device cuda:0

# echo ""
# echo "============================================================"
# echo "Step 5: Precomputing FaceBench VLM features (64-d)"
# echo "  Face-LLaVA 13B teacher-forced — this is slow (~30s/video)"
# echo "============================================================"
# CUDA_VISIBLE_DEVICES=$GPU python preprocessing/precompute_facebench_semantic.py \
#     --detector_path "$CONFIG" \
#     --output_dir facebench_semantic \
#     --skip_existing \
#     --device cuda:0 \
#     --attr_batch_size 64 \
#     --image_batch_size 1

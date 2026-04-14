#!/bin/bash
# Generate augmented variants 4–12 for K=12 pre-augment strategy.
# Variants 1–3 already exist (seed=42 from augment_and_precompute.sh).
#
# Each variant uses a different well-known seed for diversity.
# After this, run precompute on the new variants (Steps 2–4).
#
# Usage:
#   bash augment_k12.sh [GPU_ID]

set -e
cd /data/umar/Repos/DFB_NeSyNS

CONFIG="training/config/detector/nesy_defake.yaml"
GPU=${1:-1}

# Seeds for variants 4–12 (standard ML reproducibility seeds)
# Variants 1–3 used seed=42 via augment_and_precompute.sh
declare -A SEEDS=(
    [4]=0
    [5]=7
    [6]=13
    [7]=123
    [8]=256
    [9]=512
    [10]=1024
    [11]=1337
    [12]=3407
)

echo "============================================================"
echo "K=12 Augmentation: generating variants 4–12"
echo "  Variants 1–3 already exist (seed=42)"
echo "============================================================"

for k in 4 5 6 7 8 9 10 11 12; do
    seed=${SEEDS[$k]}
    echo ""
    echo "------------------------------------------------------------"
    echo "  Variant $k / 12  (seed=$seed)"
    echo "------------------------------------------------------------"
    CUDA_VISIBLE_DEVICES=$GPU python preprocessing/augment_real_frames.py \
        --detector_path "$CONFIG" \
        --n_augmentations 1 \
        --start_index "$k" \
        --seed "$seed" \
        --workers 16 \
        --skip_existing
done

echo ""
echo "============================================================"
echo "All 12 augmented variants ready."
echo ""
echo "Next steps — precompute features on variants 4–12:"
echo "  1. fast_semantic (InsightFace + LibreFace + MediaPipe + DeepFace)"
echo "  2. forensic features (SegFormer + pixel-level)"
echo ""
echo "Run:  bash precompute_k12.sh [GPU_ID]"
echo "============================================================"

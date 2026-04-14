#!/bin/bash
# Precompute features for augmented variants 4–12.
# Variants 1–3 features already exist from augment_and_precompute.sh.
#
# Steps:
#   1. fast_semantic (InsightFace + LibreFace + MediaPipe) — GPU
#   2. DeepFace patch (emotions + ethnicity entropy) — CPU parallel
#   3. forensic features (SegFormer + pixel-level) — GPU
#
# The _augmented JSON already includes aug4–aug12 entries from augment_k12.sh.
#
# Usage:
#   bash precompute_k12.sh [GPU_ID]

set -e
cd /data/umar/Repos/DFB_NeSyNS

CONFIG="training/config/detector/nesy_defake.yaml"
GPU=${1:-1}

# The precompute scripts discover frames via train_dataset in config.
# We temporarily point train_dataset → FaceForensics++_augmented so
# collect_videos_from_json picks up aug4–aug12 entries.
# Create a temp config with that override.
TMPCONFIG=$(mktemp /tmp/nesy_k12_XXXX.yaml)
sed 's/train_dataset: \[FaceForensics++\]/train_dataset: [FaceForensics++_augmented]/' \
    "$CONFIG" > "$TMPCONFIG"
trap "rm -f $TMPCONFIG" EXIT

echo "============================================================"
echo "K=12 Feature Precomputation (variants 4–12)"
echo "  Config: $CONFIG (using _augmented JSON)"
echo "  GPU: $GPU"
echo "============================================================"

echo ""
echo "Step 1: Fast semantic features (58-d)"
echo "  InsightFace (GPU) + LibreFace (GPU) + MediaPipe (CPU)"
echo "------------------------------------------------------------"
PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=$GPU python preprocessing/precompute_fast_semantic.py \
    --detector_path "$TMPCONFIG" \
    --output_dir fast_semantic \
    --skip_existing \
    --device cuda:0

echo ""
echo "Step 2: DeepFace patch (CPU-only, 8 workers)"
echo "  emotions + ethnicity entropy → indices 2-10"
echo "------------------------------------------------------------"
PYTHONWARNINGS=ignore python preprocessing/precompute_deepface_patch.py \
    --detector_path "$TMPCONFIG" \
    --output_dir fast_semantic \
    --workers 8 \
    --skip_patched

echo ""
echo "Step 3: Forensic features (83-d)"
echo "------------------------------------------------------------"
CUDA_VISIBLE_DEVICES=$GPU python preprocessing/precompute_forensic_features.py \
    --detector_path "$TMPCONFIG" \
    --batch_size 256 \
    --workers 8 \
    --output_dir forensic_features \
    --skip_existing \
    --device cuda:0

echo ""
echo "============================================================"
echo "Feature precomputation complete for all K=12 variants."
echo ""
echo "Next: update training config and dataset loader for K=12 sampling."
echo "============================================================"

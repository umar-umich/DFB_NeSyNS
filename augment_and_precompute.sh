#!/bin/bash
# Full pipeline: augment real frames + precompute all features
#
# Supports K pre-augmented variants per real frame. Default K=3 (original
# behaviour for 4:1 class balance). Set K=12 for zero-mismatch training.
#
# Variants 1–3 use seed=42 (original). Variants 4–12 each use a distinct
# well-known ML seed for diversity:
#   aug1-3: 42 | aug4: 1 | aug5: 7 | aug6: 13 | aug7: 123
#   aug8: 256 | aug9: 512 | aug10: 1024 | aug11: 1337 | aug12: 3407
#
# Steps:
#   Step 1: Augment real frames (K copies, one seed per variant)
#   Step 2: Precompute fast semantic features (58-d) — GPU
#   Step 3: Patch DeepFace features into fast_semantic .pt files — CPU
#   Step 4: Precompute forensic features (83-d) — GPU
#
# Usage:
#   bash augment_and_precompute.sh [K] [GPU_ID]
#   bash augment_and_precompute.sh              # K=3, GPU=1
#   bash augment_and_precompute.sh 12           # K=12, GPU=1
#   bash augment_and_precompute.sh 12 0         # K=12, GPU=0

set -e
cd /data/umar/Repos/DFB_NeSyNS

if [ -z "$1" ]; then
    echo "Usage: bash augment_and_precompute.sh K [GPU_ID]"
    echo "  K = number of augmented variants (e.g. 3, 12)"
    exit 1
fi

K=$1
GPU=1
CONFIG="training/config/detector/nesy_defake.yaml"

# Seed per variant index. Variants 1–3 share seed=42 (original batch).
# Variants 4+ each get a distinct seed for augmentation diversity.
declare -A VARIANT_SEEDS=(
    [4]=1    [5]=7    [6]=13   [7]=123
    [8]=256  [9]=512  [10]=1024 [11]=1337 [12]=3407
)

echo "============================================================"
echo "Augment + Precompute Pipeline  (K=$K, GPU=$GPU)"
echo "  Config: $CONFIG"
echo "============================================================"

# ── Step 1: Augment real frames ─────────────────────────────────────────

echo ""
echo "============================================================"
echo "Step 1: Augmenting real frames ($K variants)"
echo "============================================================"

# Variants 1–3: single batch with seed=42 (original behaviour)
BATCH_END=$((K < 3 ? K : 3))
echo "  Variants 1–$BATCH_END (seed=42, batch)"
CUDA_VISIBLE_DEVICES=$GPU python preprocessing/augment_real_frames.py \
    --detector_path "$CONFIG" \
    --n_augmentations "$BATCH_END" \
    --start_index 1 \
    --seed 42 \
    --workers 16 \
    --skip_existing

# Variants 4–K: one at a time with distinct seeds
for k in $(seq 4 "$K"); do
    seed=${VARIANT_SEEDS[$k]}
    if [ -z "$seed" ]; then
        echo "  ERROR: no seed defined for variant $k" >&2
        exit 1
    fi
    echo ""
    echo "  Variant $k / $K  (seed=$seed)"
    CUDA_VISIBLE_DEVICES=$GPU python preprocessing/augment_real_frames.py \
        --detector_path "$CONFIG" \
        --n_augmentations 1 \
        --start_index "$k" \
        --seed "$seed" \
        --workers 16 \
        --skip_existing
done

# ── Precompute needs the _augmented JSON to see all variants ────────────
# augment_real_frames.py already updated FaceForensics++_augmented.json.
# Create a temp config pointing train_dataset at the augmented JSON so
# the precompute scripts discover all variant frames.
TMPCONFIG=$(mktemp /tmp/nesy_precompute_XXXX.yaml)
sed 's/train_dataset: \[FaceForensics++\]/train_dataset: [FaceForensics++_augmented]/' \
    "$CONFIG" > "$TMPCONFIG"
trap "rm -f $TMPCONFIG" EXIT

# ── Step 2: Fast semantic features ──────────────────────────────────────

echo ""
echo "============================================================"
echo "Step 2: Precomputing fast semantic features (58-d)"
echo "  InsightFace (GPU) + LibreFace (GPU) + MediaPipe (CPU)"
echo "  DeepFace disabled for speed — patched in Step 3"
echo "============================================================"
PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=$GPU python preprocessing/precompute_fast_semantic.py \
    --detector_path "$TMPCONFIG" \
    --output_dir fast_semantic \
    --skip_existing \
    --device cuda:0

# ── Step 3: DeepFace patch ──────────────────────────────────────────────

echo ""
echo "============================================================"
echo "Step 3: Patching DeepFace features (CPU-only, 8 processes)"
echo "  emotions + ethnicity entropy → indices 2-10"
echo "============================================================"
PYTHONWARNINGS=ignore python preprocessing/precompute_deepface_patch.py \
    --detector_path "$TMPCONFIG" \
    --output_dir fast_semantic \
    --workers 8 \
    --skip_patched

# ── Step 4: Forensic features ──────────────────────────────────────────

echo ""
echo "============================================================"
echo "Step 4: Precomputing forensic features (83-d)"
echo "============================================================"
CUDA_VISIBLE_DEVICES=$GPU python preprocessing/precompute_forensic_features.py \
    --detector_path "$TMPCONFIG" \
    --batch_size 256 \
    --workers 8 \
    --output_dir forensic_features \
    --skip_existing \
    --device cuda:0

echo ""
echo "============================================================"
echo "Pipeline complete! K=$K variants, all features precomputed."
echo "============================================================"

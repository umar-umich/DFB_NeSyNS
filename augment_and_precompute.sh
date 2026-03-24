#!/bin/bash
# Full pipeline: augment real frames + precompute all features
#
# Step 1: Augment real frames (3 copies each to balance 4:1 fake:real ratio)
# Step 2: Precompute Face-LLaVA 211 attributes for ALL frames (original + augmented)
# Step 3: Precompute Tier 2 forensic features (30-d) for ALL frames
#
# After running, update train_dataset to use the _augmented JSON.

set -e
cd /data/umar/Repos/DFB_NeSyNS

echo "============================================================"
echo "Step 1: Augmenting real frames (3 copies for 4:1 balance)"
echo "============================================================"
python training/augment_real_frames.py \
    --detector_path training/config/detector/nesy_defake.yaml \
    --n_augmentations 3 \
    --workers 16 \
    --skip_existing

echo ""
echo "============================================================"
echo "Step 2: Precomputing Face-LLaVA features for ALL frames"
echo "============================================================"
# This picks up both original and augmented frames from the _augmented JSON
python preprocessing/precompute_semantic_features.py \
    --detector_path training/config/detector/nesy_defake.yaml \
    --batch_size 64 \
    --output_dir facellava_semantic \
    --skip_existing \
    --device cuda:1

echo ""
echo "============================================================"
echo "Step 3: Precomputing forensic features for ALL frames"
echo "============================================================"
python preprocessing/precompute_forensic_features.py \
    --detector_path training/config/detector/nesy_defake.yaml \
    --batch_size 256 \
    --output_dir forensic_features \
    --skip_existing \
    --device cuda:1

echo ""
echo "============================================================"
echo "Done! Next steps:"
echo "  1. Update train_dataset in nesy_defake.yaml to use"
echo "     FaceForensics++_augmented instead of FaceForensics++"
echo "  2. Run training as normal"
echo "============================================================"

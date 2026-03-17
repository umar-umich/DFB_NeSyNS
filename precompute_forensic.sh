#!/bin/bash
# Precompute Tier 2 forensic features (30-d) for all dataset frames.
# Run this ONCE before training with forensic features enabled.
#
# Output: .../forensic_features/<video_id>.pt alongside each frames/ directory
# Time: ~1-2 hours for FF++ (depends on GPU, InsightFace/MediaPipe availability)

cd /data/umar/Repos/DFB_NeSyNS

python preprocessing/precompute_forensic_features.py \
    --detector_path training/config/detector/nesy_defake.yaml \
    --batch_size 32 \
    --output_dir forensic_features \
    --skip_existing \
    --device cuda:1

#!/bin/bash
# Precompute Face-LLaVA 211 semantic attributes for all dataset frames.
# Run this ONCE before training with backend: precomputed.
#
# Output: .../facellava_semantic/<video_id>.pt alongside each frames/ directory
# Time: ~30-60 min for FF++ (depends on GPU and number of frames)

cd /data/umar/Repos/DFB_NeSyNS

# Use a temporary config that forces face_llava backend for extraction
python training/precompute_semantic_features.py \
    --detector_path training/config/detector/nesy_defake.yaml \
    --batch_size 32 \
    --output_dir facellava_semantic \
    --skip_existing \
    --device cuda:1

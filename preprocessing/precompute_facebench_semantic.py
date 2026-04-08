#!/usr/bin/env python3
"""
precompute_facebench_semantic.py
=================================
Precompute FaceBench VLM semantic features (64-d) for all frames using
Face-LLaVA 13B teacher-forced inference.

Only extracts the 64 VLM-only attributes (Tier B from refined_attributes.py)
that genuinely require Vision-Language Model understanding — facial hair,
makeup, skin condition, accessories, etc.

Output: facebench_semantic/{video_id}.pt → (n_frames, 64)

Usage:
  CUDA_VISIBLE_DEVICES=1 python preprocessing/precompute_facebench_semantic.py \
      --detector_path training/config/detector/nesy_defake.yaml \
      --output_dir facebench_semantic --skip_existing
"""

import argparse
import os
import sys
import traceback

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from config_utils import load_config, collect_videos_from_json


def parse_args():
    parser = argparse.ArgumentParser(
        description='Precompute FaceBench VLM semantic features (64-d)')
    parser.add_argument('--detector_path', type=str, required=True,
                        help='Path to detector YAML config')
    parser.add_argument('--output_dir', type=str,
                        default='facebench_semantic',
                        help='Subdirectory name for output .pt files')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip videos that already have output')
    parser.add_argument('--compression', type=str, default=None,
                        help='Override compression level (e.g., c23)')
    parser.add_argument('--max_frames', type=int, default=0,
                        help='Max frames per video (0=all)')
    parser.add_argument('--attr_batch_size', type=int, default=64,
                        help='Attributes per LLM forward pass (64=1 pass)')
    parser.add_argument('--image_batch_size', type=int, default=1,
                        help='Images batched through CLIP vision tower')
    parser.add_argument('--load_4bit', action='store_true',
                        help='Load LLM in 4-bit quantization')
    parser.add_argument('--load_8bit', action='store_true',
                        help='Load LLM in 8-bit quantization')
    parser.add_argument('--keyframe_stride', type=int, default=1,
                        help='Extract every Nth frame, propagate to others. '
                             'VLM attributes (hair, skin, makeup) are stable '
                             'across frames. stride=8 → 8x speedup.')
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.detector_path)
    if args.compression:
        config['compression'] = args.compression

    device = args.device if torch.cuda.is_available() else 'cpu'

    # Import VLM feature names
    sys.path.insert(
        0, os.path.join(os.path.dirname(__file__), '..', 'training'))
    from networks.nesy_defake.semantic.refined_attributes import (
        VLM_FEATURE_NAMES, NUM_VLM_FEATURES)

    print("=" * 60)
    print(f"FaceBench VLM Semantic Feature Precomputation ({NUM_VLM_FEATURES}-d)")
    print(f"  Model:  Face-LLaVA 13B (teacher-forced)")
    print(f"  Attrs:  {NUM_VLM_FEATURES} VLM-only attributes")
    print(f"  Device: {device}")
    print("=" * 60)

    # Load SemanticPrecomputer with only the 64 VLM attributes
    from precompute_semantic_features import SemanticPrecomputer

    _orig_get_attr_names = SemanticPrecomputer._get_attr_names

    @staticmethod
    def _vlm_attr_names():
        return list(VLM_FEATURE_NAMES)

    SemanticPrecomputer._get_attr_names = _vlm_attr_names
    try:
        precomputer = SemanticPrecomputer(
            config, device,
            attr_batch_size=args.attr_batch_size,
            load_4bit=args.load_4bit,
            load_8bit=args.load_8bit,
        )
    finally:
        SemanticPrecomputer._get_attr_names = _orig_get_attr_names

    print(f"[FaceBench] Loaded for {len(precomputer.attr_names)} "
          f"VLM-only attributes")

    # Collect videos
    videos = collect_videos_from_json(config)
    if not videos:
        print("No videos found. Check your config and dataset JSONs.")
        return

    print(f"\n  Videos to process: {len(videos)}")
    print(f"  Output subdir: {args.output_dir}")

    n_skipped = 0
    n_processed = 0
    n_failed = 0

    for vid_key, vid_info in tqdm(videos.items(), desc="Videos"):
        output_base = vid_info['output_dir']
        video_id = vid_info['video_id']

        output_dir = os.path.join(output_base, args.output_dir)
        output_path = os.path.join(output_dir, f"{video_id}.pt")

        if args.skip_existing and os.path.exists(output_path):
            n_skipped += 1
            continue

        frame_paths = vid_info['frames']
        if args.max_frames > 0:
            frame_paths = frame_paths[:args.max_frames]

        try:
            stride = args.keyframe_stride
            if stride > 1 and len(frame_paths) > 1:
                # Extract from keyframes only, propagate to neighbors
                keyframe_paths = frame_paths[::stride]
                result = precomputer.process_video(
                    keyframe_paths, image_batch_size=args.image_batch_size)
                if result is not None:
                    # Propagate keyframe features to all frames via
                    # nearest-keyframe assignment
                    kf_feats = result['features']  # (n_keyframes, 64)
                    n_total = len(frame_paths)
                    all_feats = torch.zeros(n_total, kf_feats.shape[1])
                    for fi in range(n_total):
                        kf_idx = min(fi // stride, kf_feats.shape[0] - 1)
                        all_feats[fi] = kf_feats[kf_idx]
                    result['features'] = all_feats
                    result['frame_paths'] = [
                        os.path.basename(p) for p in frame_paths]
            else:
                result = precomputer.process_video(
                    frame_paths, image_batch_size=args.image_batch_size)
            if result is None:
                n_failed += 1
                continue

            # Save with VLM feature names
            result['feature_names'] = list(VLM_FEATURE_NAMES)
            os.makedirs(output_dir, exist_ok=True)
            torch.save(result, output_path)
            n_processed += 1

        except Exception as e:
            print(f"  Error processing {vid_key}: {e}")
            traceback.print_exc()
            n_failed += 1
            torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print(f"FaceBench precomputation complete!")
    print(f"  Processed: {n_processed}")
    print(f"  Skipped:   {n_skipped}")
    print(f"  Failed:    {n_failed}")
    print(f"  Output:    */{args.output_dir}/<video_id>.pt")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()

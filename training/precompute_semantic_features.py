#!/usr/bin/env python3
"""
precompute_semantic_features.py
================================
Precompute Face-LLaVA semantic attributes for all frames and save to disk.

Runs the frozen Face-LLaVA 13B model once over every frame in the dataset
JSONs, producing per-video .pt files with shape (n_frames, 211) containing
the 211 FaceBench attribute probabilities.

Output structure (mirrors frames/ directory):
  .../original_sequences/youtube/c23/facellava_semantic/929.pt
  .../manipulated_sequences/Deepfakes/c23/facellava_semantic/802_885.pt

Each .pt file contains:
  {'features': Tensor(n_frames, 211), 'frame_paths': List[str]}

Usage:
  python training/precompute_semantic_features.py \
      --detector_path training/config/detector/nesy_defake.yaml \
      --batch_size 32 \
      --output_dir facellava_semantic

This eliminates the need to run a 13B LLM during training (~5x speedup).
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Precompute Face-LLaVA semantic features for all frames')
    parser.add_argument('--detector_path', type=str, required=True,
                        help='Path to detector YAML config')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Micro-batch size for Face-LLaVA inference')
    parser.add_argument('--output_dir', type=str, default='facellava_semantic',
                        help='Output subdirectory name (placed alongside frames/)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run inference on')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip videos that already have precomputed features')
    parser.add_argument('--compression', type=str, default=None,
                        help='Override compression level (e.g., c23)')
    return parser.parse_args()


def load_config(detector_path):
    """Load and merge detector + train configs."""
    with open(detector_path) as f:
        config = yaml.safe_load(f)
    train_cfg_path = os.path.join(
        os.path.dirname(detector_path), '..', 'config', 'train_config.yaml')
    if not os.path.exists(train_cfg_path):
        train_cfg_path = './training/config/train_config.yaml'
    if os.path.exists(train_cfg_path):
        with open(train_cfg_path) as f:
            train_config = yaml.safe_load(f)
        if 'label_dict' in config:
            train_config['label_dict'] = config['label_dict']
        config.update(train_config)
    return config


def collect_videos_from_json(config):
    """
    Parse dataset JSONs and return a dict:
      video_id -> {'frames': [path1, path2, ...], 'output_dir': str}
    """
    json_folder = config['dataset_json_folder']
    compression = config.get('compression', 'c23')

    all_datasets = set()
    for key in ('train_dataset', 'test_dataset'):
        ds = config.get(key, [])
        if isinstance(ds, str):
            ds = [ds]
        all_datasets.update(ds)

    videos = {}
    total_frames = 0

    for dataset_name in sorted(all_datasets):
        json_path = os.path.join(json_folder, f'{dataset_name}.json')
        if not os.path.exists(json_path):
            print(f"  WARNING: JSON not found: {json_path}, skipping")
            continue

        with open(json_path) as f:
            data = json.load(f)

        for top_key, top_val in data.items():
            for label_key, label_val in top_val.items():
                for mode_key, mode_val in label_val.items():
                    if compression not in mode_val:
                        continue
                    comp_val = mode_val[compression]
                    for video_id, video_data in comp_val.items():
                        frames = video_data.get('frames', [])
                        if not frames:
                            continue

                        # Determine output directory from frame path
                        # Handles both 'frames/' and 'frames_aug_N/' directories
                        sample_frame = frames[0]
                        sep = '/' if '/' in sample_frame else '\\'
                        parts = sample_frame.split(sep)
                        frames_dir_idx = None
                        for pi, part in enumerate(parts):
                            if part == 'frames' or part.startswith('frames_aug_'):
                                frames_dir_idx = pi
                                break
                        if frames_dir_idx is not None:
                            base_dir = sep.join(parts[:frames_dir_idx])
                        else:
                            base_dir = os.path.dirname(
                                os.path.dirname(sample_frame))

                        vid_key = f"{base_dir}/{video_id}"
                        if vid_key not in videos:
                            videos[vid_key] = {
                                'frames': sorted(frames),
                                'output_dir': base_dir,
                                'video_id': video_id,
                            }
                            total_frames += len(frames)
                        else:
                            # Merge frames from different splits
                            existing = set(videos[vid_key]['frames'])
                            for fp in frames:
                                if fp not in existing:
                                    videos[vid_key]['frames'].append(fp)
                                    total_frames += 1
                            videos[vid_key]['frames'].sort()

    print(f"\nCollected {len(videos)} videos, {total_frames} total frames")
    return videos


def build_extractor(config, device, batch_size):
    """Build the Face-LLaVA semantic extractor."""
    from copy import deepcopy

    # Deep copy to avoid mutating the original config
    extract_config = deepcopy(config)
    sem_cfg = extract_config.get('semantic_attributes', {})

    # Force face_llava backend (even if config says 'precomputed')
    sem_cfg['backend'] = 'face_llava'
    sem_cfg['micro_batch_size'] = batch_size
    # Ensure use_llm is true for 211-attribute extraction
    sem_cfg['use_llm'] = sem_cfg.get('use_llm', True)
    extract_config['semantic_attributes'] = sem_cfg

    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from networks.nesy_defake.semantic import FacialSemanticExtractor

    extractor = FacialSemanticExtractor(extract_config)
    extractor = extractor.to(device)
    extractor.eval()

    return extractor


def precompute_all(args):
    """Main precomputation loop."""
    config = load_config(args.detector_path)
    if args.compression:
        config['compression'] = args.compression

    print("=" * 60)
    print("Face-LLaVA Semantic Feature Precomputation")
    print("=" * 60)

    # Collect all videos
    videos = collect_videos_from_json(config)
    if not videos:
        print("No videos found. Check your config and dataset JSONs.")
        return

    # Build extractor
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nBuilding Face-LLaVA extractor on {device}...")
    extractor = build_extractor(config, device, args.batch_size)

    use_llm = getattr(extractor, '_use_llm', False)
    print(f"  Mode: {'Full LLM (211 attributes)' if use_llm else 'Vision-only'}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Output subdir: {args.output_dir}")

    # Process videos
    n_skipped = 0
    n_processed = 0
    n_failed = 0

    for vid_key, vid_info in tqdm(videos.items(), desc="Videos"):
        output_base = vid_info['output_dir']
        video_id = vid_info['video_id']
        frames = vid_info['frames']

        output_dir = os.path.join(output_base, args.output_dir)
        output_path = os.path.join(output_dir, f'{video_id}.pt')

        if args.skip_existing and os.path.exists(output_path):
            n_skipped += 1
            continue

        try:
            # Load all frames for this video
            from PIL import Image
            import numpy as np

            frame_tensors = []
            valid_paths = []

            for fp in frames:
                if not os.path.exists(fp):
                    continue
                try:
                    img = Image.open(fp).convert('RGB')
                    img_np = np.array(img).astype(np.float32) / 255.0
                    # HWC -> CHW
                    img_t = torch.from_numpy(img_np).permute(2, 0, 1)
                    frame_tensors.append(img_t)
                    valid_paths.append(fp)
                except Exception as e:
                    print(f"  Warning: failed to load {fp}: {e}")

            if not frame_tensors:
                n_failed += 1
                continue

            # Stack and run through extractor
            batch = torch.stack(frame_tensors).to(device)

            with torch.no_grad():
                features = extractor(raw_images=batch)  # (N, 211)

            # Save
            os.makedirs(output_dir, exist_ok=True)
            torch.save({
                'features': features.cpu(),
                'frame_paths': valid_paths,
            }, output_path)

            n_processed += 1

        except Exception as e:
            print(f"  Error processing {vid_key}: {e}")
            n_failed += 1
            # Try to free GPU memory
            torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print(f"Precomputation complete!")
    print(f"  Processed: {n_processed}")
    print(f"  Skipped:   {n_skipped}")
    print(f"  Failed:    {n_failed}")
    print(f"  Output:    */{args.output_dir}/<video_id>.pt")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    args = parse_args()
    precompute_all(args)

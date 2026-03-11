#!/usr/bin/env python3
"""
augment_real_frames.py
=======================
Augment real frames to balance the 4:1 fake:real class imbalance in FF++.

Creates N augmented copies of each real frame and saves them to parallel
directories alongside the original frames/ directory:
  .../original_sequences/youtube/c23/frames/929/000.png        (original)
  .../original_sequences/youtube/c23/frames_aug_1/929/000.png  (augmented copy 1)
  .../original_sequences/youtube/c23/frames_aug_2/929/000.png  (augmented copy 2)
  .../original_sequences/youtube/c23/frames_aug_3/929/000.png  (augmented copy 3)

Then updates (or creates) a new dataset JSON with the augmented frames
included as additional FF-real entries so the dataloader picks them up.

After running this script:
  1. Run precompute_semantic_features.py on the augmented frames
  2. Training will see ~equal real:fake frame counts

Usage:
  python training/augment_real_frames.py \
      --detector_path training/config/detector/nesy_defake.yaml \
      --n_augmentations 3 \
      --workers 16
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Add training dir to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Augment real frames for class-balanced training')
    parser.add_argument('--detector_path', type=str, required=True,
                        help='Path to detector YAML config')
    parser.add_argument('--n_augmentations', type=int, default=3,
                        help='Number of augmented copies per real frame (3 for 4:1 ratio)')
    parser.add_argument('--workers', type=int, default=16,
                        help='Number of parallel workers for augmentation')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip frames that already have augmented copies')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()


def load_config(detector_path):
    import yaml
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


def build_augmentation_pipeline(config):
    """Build the same augmentation pipeline used during training."""
    import albumentations as A
    from dataset.albu import IsotropicResize

    aug_cfg = config.get('data_aug', {})
    resolution = config.get('resolution', 224)

    transform = A.Compose([
        A.HorizontalFlip(p=aug_cfg.get('flip_prob', 0.5)),
        A.Rotate(
            limit=aug_cfg.get('rotate_limit', [-10, 10]),
            p=aug_cfg.get('rotate_prob', 0.3)),
        A.GaussianBlur(
            blur_limit=aug_cfg.get('blur_limit', [3, 5]),
            p=aug_cfg.get('blur_prob', 0.1)),
        A.OneOf([
            IsotropicResize(max_side=resolution,
                            interpolation_down=cv2.INTER_AREA,
                            interpolation_up=cv2.INTER_CUBIC),
            IsotropicResize(max_side=resolution,
                            interpolation_down=cv2.INTER_AREA,
                            interpolation_up=cv2.INTER_LINEAR),
            IsotropicResize(max_side=resolution,
                            interpolation_down=cv2.INTER_LINEAR,
                            interpolation_up=cv2.INTER_LINEAR),
        ], p=1),
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=aug_cfg.get('brightness_limit', [-0.1, 0.1]),
                contrast_limit=aug_cfg.get('contrast_limit', [-0.1, 0.1])),
            A.FancyPCA(),
            A.HueSaturationValue()
        ], p=0.5),
        A.ImageCompression(
            quality_lower=aug_cfg.get('quality_lower', 80),
            quality_upper=aug_cfg.get('quality_upper', 100),
            p=0.2),
    ])
    return transform


def collect_real_frames(config):
    """Collect all real frame paths from dataset JSONs (train split only)."""
    json_folder = config['dataset_json_folder']
    compression = config.get('compression', 'c23')

    all_datasets = config.get('train_dataset', [])
    if isinstance(all_datasets, str):
        all_datasets = [all_datasets]

    # video_id -> list of frame paths
    real_videos = {}
    total_frames = 0

    for dataset_name in all_datasets:
        json_path = os.path.join(json_folder, f'{dataset_name}.json')
        if not os.path.exists(json_path):
            print(f"  WARNING: JSON not found: {json_path}")
            continue

        with open(json_path) as f:
            data = json.load(f)

        label_dict = config.get('label_dict', {})

        for top_key, top_val in data.items():
            for label_key, label_val in top_val.items():
                # Only collect real frames
                label = label_dict.get(label_key)
                if label != 0:  # 0 = real
                    continue

                if 'train' not in label_val:
                    continue
                train_val = label_val['train']

                if compression not in train_val:
                    continue

                for video_id, video_data in train_val[compression].items():
                    frames = video_data.get('frames', [])
                    if not frames:
                        continue
                    real_videos[video_id] = {
                        'frames': sorted(frames),
                        'label': label_key,
                    }
                    total_frames += len(frames)

    print(f"  Found {len(real_videos)} real videos, {total_frames} frames")
    return real_videos


def augment_single_video(args_tuple):
    """Augment all frames of a single video. Runs in a worker process."""
    (video_id, frame_paths, n_augs, aug_config, resolution,
     skip_existing, seed) = args_tuple

    # Build augmentation pipeline in each worker
    import albumentations as A
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from dataset.albu import IsotropicResize

    aug_cfg = aug_config
    transform = A.Compose([
        A.HorizontalFlip(p=aug_cfg.get('flip_prob', 0.5)),
        A.Rotate(
            limit=aug_cfg.get('rotate_limit', [-10, 10]),
            p=aug_cfg.get('rotate_prob', 0.3)),
        A.GaussianBlur(
            blur_limit=aug_cfg.get('blur_limit', [3, 5]),
            p=aug_cfg.get('blur_prob', 0.1)),
        A.OneOf([
            IsotropicResize(max_side=resolution,
                            interpolation_down=cv2.INTER_AREA,
                            interpolation_up=cv2.INTER_CUBIC),
            IsotropicResize(max_side=resolution,
                            interpolation_down=cv2.INTER_AREA,
                            interpolation_up=cv2.INTER_LINEAR),
            IsotropicResize(max_side=resolution,
                            interpolation_down=cv2.INTER_LINEAR,
                            interpolation_up=cv2.INTER_LINEAR),
        ], p=1),
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=aug_cfg.get('brightness_limit', [-0.1, 0.1]),
                contrast_limit=aug_cfg.get('contrast_limit', [-0.1, 0.1])),
            A.FancyPCA(),
            A.HueSaturationValue()
        ], p=0.5),
        A.ImageCompression(
            quality_lower=aug_cfg.get('quality_lower', 80),
            quality_upper=aug_cfg.get('quality_upper', 100),
            p=0.2),
    ])

    random.seed(seed + hash(video_id))
    np.random.seed((seed + hash(video_id)) % (2**32))

    augmented_paths = defaultdict(list)  # aug_idx -> list of paths
    n_created = 0

    for frame_path in frame_paths:
        if not os.path.exists(frame_path):
            continue

        # Parse path to find frames/ directory
        sep = '/' if '/' in frame_path else '\\'
        parts = frame_path.split(sep)
        if 'frames' not in parts:
            continue

        frames_idx = parts.index('frames')
        frame_filename = parts[-1]

        # Read original image
        img = cv2.imread(frame_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for aug_i in range(1, n_augs + 1):
            # Build output path: replace 'frames' with 'frames_aug_N'
            aug_parts = list(parts)
            aug_parts[frames_idx] = f'frames_aug_{aug_i}'
            aug_path = sep.join(aug_parts)

            if skip_existing and os.path.exists(aug_path):
                augmented_paths[aug_i].append(aug_path)
                continue

            # Apply augmentation
            augmented = transform(image=img)['image']

            # Save
            os.makedirs(os.path.dirname(aug_path), exist_ok=True)
            aug_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
            cv2.imwrite(aug_path, aug_bgr)

            augmented_paths[aug_i].append(aug_path)
            n_created += 1

    return video_id, dict(augmented_paths), n_created


def update_dataset_json(config, real_videos, n_augs):
    """
    Create an updated dataset JSON that includes augmented real frames.

    The augmented frames are added as new video entries with modified paths
    (frames/ -> frames_aug_N/). This way the dataloader picks them up
    automatically as additional real training data.
    """
    json_folder = config['dataset_json_folder']
    compression = config.get('compression', 'c23')

    for dataset_name in config.get('train_dataset', []):
        json_path = os.path.join(json_folder, f'{dataset_name}.json')
        if not os.path.exists(json_path):
            continue

        with open(json_path) as f:
            data = json.load(f)

        # Find the real label section
        label_dict = config.get('label_dict', {})
        modified = False

        for top_key, top_val in data.items():
            for label_key, label_val in top_val.items():
                label = label_dict.get(label_key)
                if label != 0:  # only augment reals
                    continue

                if 'train' not in label_val:
                    continue

                if compression not in label_val['train']:
                    continue

                train_data = label_val['train'][compression]

                # Add augmented video entries
                for video_id, vid_info in list(real_videos.items()):
                    if video_id not in train_data:
                        continue

                    original_frames = train_data[video_id]['frames']
                    original_label = train_data[video_id]['label']

                    for aug_i in range(1, n_augs + 1):
                        aug_video_id = f'{video_id}_aug{aug_i}'
                        if aug_video_id in train_data:
                            continue  # already exists

                        # Create augmented frame paths
                        aug_frames = []
                        for fp in original_frames:
                            aug_fp = fp.replace('/frames/', f'/frames_aug_{aug_i}/')
                            aug_frames.append(aug_fp)

                        train_data[aug_video_id] = {
                            'label': original_label,
                            'frames': aug_frames,
                        }
                        modified = True

        if modified:
            # Rename top-level key to match the augmented filename
            # (abstract_dataset.py looks up dataset_info[dataset_name])
            aug_dataset_name = f'{dataset_name}_augmented'
            if dataset_name in data and aug_dataset_name not in data:
                data[aug_dataset_name] = data.pop(dataset_name)

            # Save to a new file (don't overwrite original)
            aug_json_path = os.path.join(
                json_folder, f'{dataset_name}_augmented.json')
            with open(aug_json_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"\n  Updated JSON saved to: {aug_json_path}")
            print(f"  To use: set train_dataset to ['{dataset_name}_augmented'] "
                  f"in your config, or rename the file.")

            # Also count final balance
            n_real = sum(1 for vid, vinfo in train_data.items()
                         if not any(vid.startswith(f'{v}_aug')
                                    for v in real_videos))
            n_real_aug = sum(1 for vid in train_data
                             if any(vid.startswith(f'{v}_aug')
                                    for v in real_videos))
            print(f"  Real videos (original): {n_real}")
            print(f"  Real videos (augmented): {n_real_aug}")
            print(f"  Total real videos: {n_real + n_real_aug}")


def main():
    args = parse_args()
    config = load_config(args.detector_path)

    print("=" * 60)
    print("Real Frame Augmentation for Class Balance")
    print("=" * 60)
    print(f"  Augmentation copies: {args.n_augmentations}")
    print(f"  Workers: {args.workers}")
    print(f"  Skip existing: {args.skip_existing}")
    print()

    # Collect real frames
    print("Collecting real frames...")
    real_videos = collect_real_frames(config)

    if not real_videos:
        print("No real frames found!")
        return

    # Prepare work items
    aug_config = config.get('data_aug', {})
    resolution = config.get('resolution', 224)

    work_items = [
        (vid_id, vid_info['frames'], args.n_augmentations,
         aug_config, resolution, args.skip_existing, args.seed)
        for vid_id, vid_info in real_videos.items()
    ]

    # Process with multiprocessing
    print(f"\nAugmenting {len(work_items)} videos "
          f"({args.n_augmentations} copies each)...")

    total_created = 0

    if args.workers <= 1:
        for item in tqdm(work_items, desc="Videos"):
            _, _, n = augment_single_video(item)
            total_created += n
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(augment_single_video, item): item[0]
                       for item in work_items}
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="Videos"):
                try:
                    _, _, n = future.result()
                    total_created += n
                except Exception as e:
                    vid_id = futures[future]
                    print(f"\n  Error processing {vid_id}: {e}")

    print(f"\n  Total augmented frames created: {total_created}")

    # Update dataset JSON
    print("\nUpdating dataset JSON...")
    update_dataset_json(config, real_videos, args.n_augmentations)

    print(f"\n{'=' * 60}")
    print("Next steps:")
    print("  1. Run precompute_semantic_features.py on augmented frames")
    print("  2. Update train_dataset in config to use the _augmented JSON")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()

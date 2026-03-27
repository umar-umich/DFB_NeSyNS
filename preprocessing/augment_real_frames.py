#!/usr/bin/env python3
"""
augment_real_frames.py
=======================
Augment real frames to balance the 4:1 fake:real class imbalance in FF++.

Creates N augmented copies of each real frame using torchvision v2 transforms
(GenD-style) and saves them to parallel directories alongside the original
frames/ directory.

Usage:
  python preprocessing/augment_real_frames.py \
      --detector_path training/config/detector/nesy_defake.yaml \
      --n_augmentations 3 --workers 16
"""

import argparse
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image
from torchvision.transforms import v2 as Tv2
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from config_utils import load_config, collect_real_frames, update_dataset_json


class FrameAugmenter:
    """Augments frames using a torchvision v2 pipeline (GenD-style)."""

    def __init__(self, config: dict):
        self.config = config
        self.transform = self._build_pipeline()

    def _build_pipeline(self) -> Tv2.Compose:
        aug = self.config.get('data_aug', {})
        resolution = self.config.get('resolution', 224)
        transforms = []

        flip_p = aug.get('flip_prob', 0.5)
        if flip_p > 0:
            transforms.append(Tv2.RandomHorizontalFlip(p=flip_p))

        degrees = aug.get('rotate_limit', [-10, 10])
        if isinstance(degrees, list):
            degrees = max(abs(degrees[0]), abs(degrees[1]))
        translate = aug.get('affine_translate', [0.1, 0.1])
        scale = aug.get('affine_scale', [0.9, 1.1])
        if degrees > 0 or translate is not None or scale is not None:
            transforms.append(Tv2.RandomAffine(
                degrees=degrees,
                translate=tuple(translate) if translate else None,
                scale=tuple(scale) if scale else None,
            ))

        blur_p = aug.get('blur_prob', 0.1)
        blur_kernel = aug.get('blur_kernel_size', 7)
        blur_sigma = aug.get('blur_sigma', [0.1, 2.0])
        if blur_p > 0:
            transforms.append(Tv2.RandomApply(
                [Tv2.GaussianBlur(kernel_size=blur_kernel, sigma=blur_sigma)],
                p=blur_p,
            ))

        brightness = aug.get('brightness_limit', 0.1)
        contrast = aug.get('contrast_limit', 0.1)
        if isinstance(brightness, list):
            brightness = max(abs(brightness[0]), abs(brightness[1]))
        if isinstance(contrast, list):
            contrast = max(abs(contrast[0]), abs(contrast[1]))
        if brightness > 0 or contrast > 0:
            transforms.append(Tv2.ColorJitter(
                brightness=brightness, contrast=contrast))

        quality_lower = aug.get('quality_lower', 40)
        quality_upper = aug.get('quality_upper', 100)
        if quality_lower < 100:
            transforms.append(Tv2.JPEG([quality_lower, quality_upper]))

        transforms.append(Tv2.Resize((resolution, resolution)))

        noise_sigma = aug.get('gaussian_noise_sigma', 0.0)
        if noise_sigma > 0:
            transforms.append(Tv2.Compose([
                Tv2.ToTensor(),
                Tv2.GaussianNoise(0.0, noise_sigma),
                Tv2.ToPILImage(),
            ]))

        return Tv2.Compose(transforms)

    def augment_frame(self, img: Image.Image) -> Image.Image:
        return self.transform(img)


def _augment_single_video(args_tuple):
    """Worker function for multiprocessing. Builds its own pipeline."""
    (video_id, frame_paths, n_augs, config,
     skip_existing, seed, start_index) = args_tuple

    augmenter = FrameAugmenter(config)
    random.seed(seed + hash(video_id))
    np.random.seed((seed + hash(video_id)) % (2**32))

    augmented_paths = defaultdict(list)
    n_created = 0

    for frame_path in frame_paths:
        if not os.path.exists(frame_path):
            continue

        sep = '/' if '/' in frame_path else '\\'
        parts = frame_path.split(sep)
        if 'frames' not in parts:
            continue
        frames_idx = parts.index('frames')

        try:
            img = Image.open(frame_path).convert('RGB')
        except Exception:
            continue

        for aug_i in range(start_index, start_index + n_augs):
            aug_parts = list(parts)
            aug_parts[frames_idx] = f'frames_aug_{aug_i}'
            aug_path = sep.join(aug_parts)

            if skip_existing and os.path.exists(aug_path):
                augmented_paths[aug_i].append(aug_path)
                continue

            augmented = augmenter.augment_frame(img)
            os.makedirs(os.path.dirname(aug_path), exist_ok=True)
            augmented.save(aug_path)
            augmented_paths[aug_i].append(aug_path)
            n_created += 1

    return video_id, dict(augmented_paths), n_created


def parse_args():
    parser = argparse.ArgumentParser(
        description='Augment real frames for class-balanced training')
    parser.add_argument('--detector_path', type=str, required=True)
    parser.add_argument('--n_augmentations', type=int, default=3)
    parser.add_argument('--start_index', type=int, default=None)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.detector_path)

    print("=" * 60)
    print("Real Frame Augmentation for Class Balance")
    print("=" * 60)

    print("Collecting real frames...")
    real_videos = collect_real_frames(config)
    if not real_videos:
        print("No real frames found!")
        return

    # Auto-detect start_index
    start_index = args.start_index
    if start_index is None:
        sample_path = next(iter(real_videos.values()))['frames'][0]
        frames_dir = os.path.dirname(os.path.dirname(sample_path))
        existing_augs = [
            d for d in os.listdir(frames_dir)
            if os.path.isdir(os.path.join(frames_dir, d))
            and d.startswith('frames_aug_')
        ]
        if existing_augs:
            max_existing = max(int(d.split('_')[-1]) for d in existing_augs)
            start_index = max_existing + 1
            print(f"  Found existing augmentations up to frames_aug_{max_existing}")
        else:
            start_index = 1

    end_index = start_index + args.n_augmentations - 1
    print(f"  Creating augmentations: frames_aug_{start_index} to frames_aug_{end_index}")
    print(f"  Workers: {args.workers}")
    print()

    work_items = [
        (vid_id, vid_info['frames'], args.n_augmentations,
         config, args.skip_existing, args.seed, start_index)
        for vid_id, vid_info in real_videos.items()
    ]

    print(f"\nAugmenting {len(work_items)} videos "
          f"({args.n_augmentations} copies each)...")

    total_created = 0
    if args.workers <= 1:
        for item in tqdm(work_items, desc="Videos"):
            _, _, n = _augment_single_video(item)
            total_created += n
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_augment_single_video, item): item[0]
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

    print("\nUpdating dataset JSON...")
    update_dataset_json(config, real_videos, args.n_augmentations, start_index)

    print(f"\n{'=' * 60}")
    print("Next steps:")
    print("  1. Run precompute_semantic_features.py on augmented frames")
    print("  2. Update train_dataset in config to use the _augmented JSON")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()

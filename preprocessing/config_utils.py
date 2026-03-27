"""
config_utils.py
===============
Shared configuration loading and dataset video/frame collection utilities.
Used by augment_real_frames, precompute_semantic_features, and
precompute_forensic_features.
"""

import json
import os

import yaml


def load_config(detector_path: str) -> dict:
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


def collect_videos_from_json(config: dict) -> dict:
    """
    Parse dataset JSONs and return a dict:
      vid_key -> {'frames': [path1, ...], 'output_dir': str, 'video_id': str}
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
                    first_val = next(iter(mode_val.values()), {})
                    if isinstance(first_val, dict) and 'frames' in first_val:
                        video_items = mode_val.items()
                    elif compression in mode_val:
                        video_items = mode_val[compression].items()
                    else:
                        continue

                    for video_id, video_data in video_items:
                        frames = video_data.get('frames', [])
                        if not frames:
                            continue

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
                            existing = set(videos[vid_key]['frames'])
                            for fp in frames:
                                if fp not in existing:
                                    videos[vid_key]['frames'].append(fp)
                                    total_frames += 1
                            videos[vid_key]['frames'].sort()

    print(f"\nCollected {len(videos)} videos, {total_frames} total frames")
    return videos


def collect_real_frames(config: dict) -> dict:
    """
    Collect all real frame paths from dataset JSONs (train split only).

    Returns:
        dict: video_id -> {'frames': [path1, ...], 'label': str}
    """
    json_folder = config['dataset_json_folder']
    compression = config.get('compression', 'c23')

    all_datasets = config.get('train_dataset', [])
    if isinstance(all_datasets, str):
        all_datasets = [all_datasets]

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
                label = label_dict.get(label_key)
                if label != 0:  # 0 = real
                    continue
                if 'train' not in label_val:
                    continue
                train_val = label_val['train']
                if compression not in train_val:
                    continue

                for video_id, video_data in train_val[compression].items():
                    if '_aug' in video_id:
                        continue
                    frames = video_data.get('frames', [])
                    if not frames:
                        continue
                    orig_frames = [f for f in frames if '/frames/' in f
                                   and '/frames_aug_' not in f]
                    if not orig_frames:
                        continue
                    real_videos[video_id] = {
                        'frames': sorted(orig_frames),
                        'label': label_key,
                    }
                    total_frames += len(orig_frames)

    print(f"  Found {len(real_videos)} real videos, {total_frames} frames")
    return real_videos


def update_dataset_json(config: dict, real_videos: dict,
                        n_augs: int, start_index: int = 1):
    """
    Create an updated dataset JSON that includes augmented real frames.
    Augmented frames are added as new video entries with modified paths.
    """
    json_folder = config['dataset_json_folder']
    compression = config.get('compression', 'c23')

    for dataset_name in config.get('train_dataset', []):
        json_path = os.path.join(json_folder, f'{dataset_name}.json')
        if not os.path.exists(json_path):
            continue

        with open(json_path) as f:
            data = json.load(f)

        label_dict = config.get('label_dict', {})
        modified = False

        for top_key, top_val in data.items():
            for label_key, label_val in top_val.items():
                label = label_dict.get(label_key)
                if label != 0:
                    continue
                if 'train' not in label_val:
                    continue
                if compression not in label_val['train']:
                    continue

                train_data = label_val['train'][compression]

                for video_id in list(real_videos.keys()):
                    if video_id not in train_data:
                        continue

                    original_frames = train_data[video_id]['frames']
                    original_label = train_data[video_id]['label']

                    for aug_i in range(start_index, start_index + n_augs):
                        aug_video_id = f'{video_id}_aug{aug_i}'
                        if aug_video_id in train_data:
                            continue
                        aug_frames = [
                            fp.replace('/frames/', f'/frames_aug_{aug_i}/')
                            for fp in original_frames
                        ]
                        train_data[aug_video_id] = {
                            'label': original_label,
                            'frames': aug_frames,
                        }
                        modified = True

        if modified:
            if dataset_name.endswith('_augmented'):
                aug_dataset_name = dataset_name
            else:
                aug_dataset_name = f'{dataset_name}_augmented'

            if dataset_name in data and aug_dataset_name not in data:
                data[aug_dataset_name] = data.pop(dataset_name)

            aug_json_path = os.path.join(
                json_folder, f'{aug_dataset_name}.json')
            with open(aug_json_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"\n  Updated JSON saved to: {aug_json_path}")
            print(f"  To use: set train_dataset to ['{aug_dataset_name}'] "
                  f"in your config, or rename the file.")

            n_real = sum(1 for vid in train_data
                         if not any(vid.startswith(f'{v}_aug')
                                    for v in real_videos))
            n_real_aug = sum(1 for vid in train_data
                             if any(vid.startswith(f'{v}_aug')
                                    for v in real_videos))
            print(f"  Real videos (original): {n_real}")
            print(f"  Real videos (augmented): {n_real_aug}")
            print(f"  Total real videos: {n_real + n_real_aug}")

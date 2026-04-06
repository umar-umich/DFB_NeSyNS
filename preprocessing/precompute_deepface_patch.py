#!/usr/bin/env python3
"""
precompute_deepface_patch.py
==============================
Patch existing fast_semantic/*.pt files with DeepFace features (indices 2-10).
Runs CPU-only (~207ms/frame) — no GPU needed, can run alongside training.

DeepFace fills:
  [2]  ethnicity_entropy  (normalized Shannon entropy of race probabilities)
  [3]  neutral            (emotion scores, 0-1)
  [4]  happy
  [5]  sad
  [6]  angry
  [7]  surprise
  [8]  fear
  [9]  disgust
  [10] contempt           (always 0 — DeepFace doesn't support it)

Usage:
  python preprocessing/precompute_deepface_patch.py \
      --detector_path training/config/detector/nesy_defake.yaml \
      --output_dir fast_semantic --workers 8
"""

import argparse
import os
import sys
import warnings

# Force TF to CPU-only before any import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
except ImportError:
    pass

import cv2
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from config_utils import load_config, collect_videos_from_json

# DeepFace feature indices in the 58-d vector
DEEPFACE_INDICES = [2, 3, 4, 5, 6, 7, 8, 9, 10]


def extract_deepface_for_frame(image_rgb: np.ndarray) -> np.ndarray:
    """Extract DeepFace features for a single frame. Returns (9,) array."""
    feats = np.zeros(9, dtype=np.float32)
    try:
        from deepface import DeepFace
        result = DeepFace.analyze(
            image_rgb, actions=['emotion', 'race'],
            enforce_detection=False, silent=True,
            detector_backend='skip')

        if result and len(result) > 0:
            r = result[0]
            emo = r.get('emotion', {})
            # feats[0] = ethnicity_entropy (index 2 in full vector)
            # feats[1:8] = emotions (indices 3-9)
            # feats[8] = contempt (index 10, always 0)
            for emo_name, local_idx in [
                ('neutral', 1), ('happy', 2), ('sad', 3),
                ('angry', 4), ('surprise', 5), ('fear', 6),
                ('disgust', 7),
            ]:
                feats[local_idx] = float(emo.get(emo_name, 0.0)) / 100.0

            race = r.get('race', {})
            if race:
                probs = np.array(
                    [v / 100.0 for v in race.values()],
                    dtype=np.float32)
                probs = probs / (probs.sum() + 1e-8)
                entropy = -np.sum(
                    probs * np.log(probs + 1e-8)) / np.log(len(probs))
                feats[0] = float(entropy)
    except Exception:
        pass
    return feats


def _process_frame_path(frame_path):
    """Worker function: load frame and extract DeepFace features."""
    try:
        bgr = cv2.imread(frame_path)
        if bgr is None:
            return np.zeros(9, dtype=np.float32)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return extract_deepface_for_frame(rgb)
    except Exception:
        return np.zeros(9, dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Patch fast_semantic .pt files with DeepFace features')
    parser.add_argument('--detector_path', type=str, required=True,
                        help='Path to detector YAML config')
    parser.add_argument('--output_dir', type=str, default='fast_semantic',
                        help='Subdirectory with existing .pt files to patch')
    parser.add_argument('--compression', type=str, default=None)
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of processes for parallel extraction. '
                             'Each process loads its own TF/DeepFace instance.')
    parser.add_argument('--max_frames', type=int, default=0,
                        help='Max frames per video (0=all)')
    parser.add_argument('--skip_patched', action='store_true',
                        help='Skip videos where DeepFace features are '
                             'already non-zero')
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.detector_path)
    if args.compression:
        config['compression'] = args.compression

    print("=" * 60)
    print("DeepFace Feature Patch (CPU-only)")
    print("  Patching existing fast_semantic .pt files")
    print("  Features: ethnicity_entropy, 7 emotions (indices 2-10)")
    print(f"  Workers: {args.workers} processes")
    print("=" * 60)

    videos = collect_videos_from_json(config)
    if not videos:
        print("No videos found.")
        return

    # Filter to only videos that have existing .pt files
    to_patch = []
    for vid_key, vid_info in videos.items():
        output_base = vid_info['output_dir']
        video_id = vid_info['video_id']
        pt_path = os.path.join(output_base, args.output_dir,
                               f"{video_id}.pt")
        if os.path.exists(pt_path):
            to_patch.append((vid_key, vid_info, pt_path))

    print(f"  Total videos: {len(videos)}")
    print(f"  With existing .pt: {len(to_patch)}")

    if not to_patch:
        print("No .pt files found to patch. Run precompute_fast_semantic.py first.")
        return

    n_patched = 0
    n_skipped = 0
    n_failed = 0

    # Use ProcessPoolExecutor for true parallelism (bypasses GIL)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for vid_key, vid_info, pt_path in tqdm(to_patch, desc="Patching"):
            try:
                data = torch.load(pt_path, weights_only=False)
                features = data['features']  # (n_frames, 58)

                # Check if already patched (any emotion feature non-zero)
                if args.skip_patched:
                    emotion_cols = features[:, 3:10]
                    if emotion_cols.abs().sum().item() > 0:
                        n_skipped += 1
                        continue

                frame_paths = vid_info['frames']
                if args.max_frames > 0:
                    frame_paths = frame_paths[:args.max_frames]

                # Extract DeepFace features in parallel processes
                futures = [pool.submit(_process_frame_path, fp)
                           for fp in frame_paths]
                df_features = [f.result() for f in futures]

                # Patch the features tensor
                for i, df_feat in enumerate(df_features):
                    if i < features.shape[0]:
                        for local_j, global_j in enumerate(DEEPFACE_INDICES):
                            features[i, global_j] = float(df_feat[local_j])

                data['features'] = features
                torch.save(data, pt_path)
                n_patched += 1

            except Exception as e:
                print(f"  Error patching {vid_key}: {e}")
                n_failed += 1

    print(f"\n{'=' * 60}")
    print(f"DeepFace patching complete!")
    print(f"  Patched:  {n_patched}")
    print(f"  Skipped:  {n_skipped}")
    print(f"  Failed:   {n_failed}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
precompute_forensic_features.py
================================
Precompute Tier 2 pixel-level forensic features (30-d) for all frames.

Loads SegFormer (+ optional InsightFace/MediaPipe) ONCE, then processes
all videos producing per-video .pt files with shape (n_frames, 30).

Output structure:
  .../original_sequences/youtube/c23/forensic_features/929.pt

Usage:
  python preprocessing/precompute_forensic_features.py \
      --detector_path training/config/detector/nesy_defake.yaml \
      --batch_size 256 --output_dir forensic_features
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from config_utils import load_config, collect_videos_from_json
from forensic_helpers import FEATURE_NAMES, extract_forensic_features


class ForensicPrecomputer:
    """
    Loads SegFormer face parser (+ optional InsightFace/MediaPipe) once
    and provides batch processing of video frames.
    """

    def __init__(self, device: str = 'cuda:0',
                 use_insightface: bool = True,
                 use_mediapipe: bool = True):
        self.device = device
        self._face_parser = None
        self._image_processor = None
        self._insightface_app = None
        self._mediapipe_detector = None

        self._init_face_parser()
        if use_insightface:
            self._init_insightface()
        if use_mediapipe:
            self._init_mediapipe()

    # ------------------------------------------------------------------
    # Model initialization
    # ------------------------------------------------------------------

    def _init_face_parser(self):
        from transformers import (SegformerImageProcessor,
                                  SegformerForSemanticSegmentation)
        print("[ForensicFeatures] Loading SegFormer face parser...")
        model_name = "jonathandinu/face-parsing"
        self._image_processor = SegformerImageProcessor.from_pretrained(
            model_name)
        self._face_parser = SegformerForSemanticSegmentation.from_pretrained(
            model_name).to(self.device).eval()
        print(f"[ForensicFeatures] SegFormer loaded on {self.device}")

    def _init_insightface(self):
        try:
            from insightface.app import FaceAnalysis
            self._insightface_app = FaceAnalysis(
                name='buffalo_l',
                providers=['CUDAExecutionProvider'])
            self._insightface_app.prepare(ctx_id=0, det_size=(224, 224))
            print("[ForensicFeatures] InsightFace loaded")
        except Exception as e:
            print(f"[ForensicFeatures] InsightFace not available: {e}")

    def _init_mediapipe(self):
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision as mp_vision

            model_paths = [
                os.path.expanduser(
                    '~/.mediapipe/models/face_landmarker.task'),
                '/data/umar/models/face_landmarker.task',
            ]
            model_path = next(
                (p for p in model_paths if os.path.exists(p)), None)
            if model_path is None:
                print("[ForensicFeatures] MediaPipe model not found, skipping")
                return

            options = mp_vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=model_path),
                output_face_blendshapes=True,
                num_faces=1,
            )
            self._mediapipe_detector = (
                mp_vision.FaceLandmarker.create_from_options(options))
            print("[ForensicFeatures] MediaPipe loaded")
        except Exception as e:
            print(f"[ForensicFeatures] MediaPipe not available: {e}")

    # ------------------------------------------------------------------
    # Inference methods
    # ------------------------------------------------------------------

    def get_parsing_maps_batch(self, images_rgb: list) -> list:
        pil_imgs = [Image.fromarray(img) for img in images_rgb]
        inputs = self._image_processor(
            images=pil_imgs, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._face_parser(**inputs)
        logits = outputs.logits

        parsing_maps = []
        for i, img in enumerate(images_rgb):
            upsampled = F.interpolate(
                logits[i:i+1], size=img.shape[:2],
                mode='bilinear', align_corners=False)
            parsing_maps.append(
                upsampled.argmax(dim=1).squeeze(0).cpu().numpy())
        return parsing_maps

    def _extract_insightface_quality(self, image_bgr: np.ndarray) -> tuple:
        if self._insightface_app is None:
            return 0.0, 0.0
        try:
            faces = self._insightface_app.get(image_bgr)
            if len(faces) == 0:
                return 0.0, 0.0
            face = faces[0]
            antispoof = float(getattr(face, 'antispoof', 0.0) or 0.0)
            det_score = float(face.det_score)
            return antispoof, det_score
        except Exception:
            return 0.0, 0.0

    def _extract_blendshape_symmetry(self, image_rgb: np.ndarray) -> float:
        if self._mediapipe_detector is None:
            return 0.0
        try:
            import mediapipe as mp
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=image_rgb)
            result = self._mediapipe_detector.detect(mp_image)

            if (not result.face_blendshapes
                    or len(result.face_blendshapes) == 0):
                return 0.0

            bs = {b.category_name: b.score
                  for b in result.face_blendshapes[0]}

            bilateral_pairs = [
                ('browDownLeft', 'browDownRight'),
                ('browOuterUpLeft', 'browOuterUpRight'),
                ('cheekSquintLeft', 'cheekSquintRight'),
                ('eyeBlinkLeft', 'eyeBlinkRight'),
                ('eyeLookDownLeft', 'eyeLookDownRight'),
                ('eyeLookInLeft', 'eyeLookInRight'),
                ('eyeLookOutLeft', 'eyeLookOutRight'),
                ('eyeLookUpLeft', 'eyeLookUpRight'),
                ('eyeSquintLeft', 'eyeSquintRight'),
                ('eyeWideLeft', 'eyeWideRight'),
                ('mouthSmileLeft', 'mouthSmileRight'),
                ('mouthFrownLeft', 'mouthFrownRight'),
                ('mouthDimpleLeft', 'mouthDimpleRight'),
                ('mouthStretchLeft', 'mouthStretchRight'),
                ('noseSneerLeft', 'noseSneerRight'),
            ]
            diffs = [abs(bs.get(l, 0.0) - bs.get(r, 0.0))
                     for l, r in bilateral_pairs]
            return float(np.mean(diffs)) if diffs else 0.0
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------

    def process_video(self, frame_paths: list,
                      batch_size: int = 32) -> dict:
        """
        Process all frames of a video and return forensic features.

        Returns:
            {'features': Tensor(n_frames, 30), 'frame_paths': List[str],
             'feature_names': List[str]}
        """
        loaded = []
        for frame_path in frame_paths:
            try:
                image_bgr = cv2.imread(frame_path)
                if image_bgr is None:
                    loaded.append(None)
                    continue
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                loaded.append((image_rgb, image_bgr))
            except Exception as e:
                print(f"  [WARNING] Failed to load {frame_path}: {e}")
                loaded.append(None)

        all_features = [None] * len(frame_paths)

        for batch_start in range(0, len(loaded), batch_size):
            batch_end = min(batch_start + batch_size, len(loaded))

            batch_indices = []
            batch_images_rgb = []
            for i in range(batch_start, batch_end):
                if loaded[i] is not None:
                    batch_indices.append(i)
                    batch_images_rgb.append(loaded[i][0])

            if not batch_images_rgb:
                for i in range(batch_start, batch_end):
                    all_features[i] = np.zeros(30, dtype=np.float32)
                continue

            try:
                parsing_maps = self.get_parsing_maps_batch(batch_images_rgb)
            except Exception as e:
                print(f"  [WARNING] Batch parsing failed: {e}, "
                      f"falling back to zeros")
                parsing_maps = [
                    np.zeros(img.shape[:2], dtype=np.int64)
                    for img in batch_images_rgb
                ]

            for j, idx in enumerate(batch_indices):
                image_rgb, image_bgr = loaded[idx]
                try:
                    feats = extract_forensic_features(
                        image_rgb, parsing_maps[j])

                    if self._insightface_app is not None:
                        antispoof, det_score = (
                            self._extract_insightface_quality(image_bgr))
                        feats[26] = antispoof
                        feats[27] = det_score

                    if self._mediapipe_detector is not None:
                        feats[28] = self._extract_blendshape_symmetry(
                            image_rgb)

                    all_features[idx] = feats
                except Exception as e:
                    print(f"  [WARNING] Failed on {frame_paths[idx]}: {e}")
                    all_features[idx] = np.zeros(30, dtype=np.float32)

            torch.cuda.empty_cache()

        for i in range(len(all_features)):
            if all_features[i] is None:
                all_features[i] = np.zeros(30, dtype=np.float32)

        features_tensor = torch.from_numpy(np.stack(all_features))
        return {
            'features': features_tensor,
            'frame_paths': [os.path.basename(p) for p in frame_paths],
            'feature_names': FEATURE_NAMES,
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Precompute Tier 2 forensic features')
    parser.add_argument('--detector_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='forensic_features')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--compression', type=str, default=None)
    parser.add_argument('--no_insightface', action='store_true')
    parser.add_argument('--no_mediapipe', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_frames', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.detector_path)
    if args.compression:
        config['compression'] = args.compression

    print("=" * 60)
    print("Forensic Feature Precomputation (Tier 2, 30-d)")
    print("=" * 60)

    precomputer = ForensicPrecomputer(
        device=args.device,
        use_insightface=not args.no_insightface,
        use_mediapipe=not args.no_mediapipe,
    )

    videos = collect_videos_from_json(config)
    if not videos:
        print("No videos found. Check your config and dataset JSONs.")
        return

    print(f"\n  Output subdir: {args.output_dir}")
    print(f"  Batch size: {args.batch_size}")

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
            result = precomputer.process_video(
                frame_paths, batch_size=args.batch_size)

            os.makedirs(output_dir, exist_ok=True)
            torch.save(result, output_path)
            n_processed += 1

        except Exception as e:
            print(f"  Error processing {vid_key}: {e}")
            n_failed += 1
            torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print(f"Precomputation complete!")
    print(f"  Processed: {n_processed}")
    print(f"  Skipped:   {n_skipped}")
    print(f"  Failed:    {n_failed}")
    print(f"  Output:    */{args.output_dir}/<video_id>.pt")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()

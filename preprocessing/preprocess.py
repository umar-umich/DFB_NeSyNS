# Data pre-processing script for deepfake datasets.
# v2: RetinaFace (ONNX) replaces dlib for face detection + alignment.
#     Ported from GenD (WACV 2026) for better recall on profile/occluded faces.
#     Uses cv2.estimateAffinePartial2D (LMEDS) instead of skimage SimilarityTransform.
#     Adds "at_least" frame selection mode (max-spread permutation) from GenD.

"""
Output directory structure (unchanged from v1):

  {dataset_path}/frames/{video_stem}/{NNN}.png      — 256x256 aligned face crops
  {dataset_path}/landmarks/{video_stem}/{NNN}.npy    — 5-point landmarks (post-alignment)
  {dataset_path}/masks/{video_stem}/{NNN}.png         — binary mask (if available)

Original dataset structure before preprocessing:

-FaceForensics++
    -original_sequences
        -youtube
            -c23
                -videos
                    *.mp4
    -manipulated_sequences
        -Deepfakes / Face2Face / FaceSwap / NeuralTextures / FaceShifter / DeepFakeDetection
            -c23
                -videos

-Celeb-DF-v1/v2
    -Celeb-synthesis / Celeb-real / YouTube-real
        -videos

-DFDCP
    -method_A / method_B / original_videos

-DeeperForensics-1.0
    -manipulated_videos / source_videos

-UADFV
    -fake / real
"""

import os
import sys
import time
import heapq
import cv2
import yaml
import logging
import datetime
import glob
import concurrent.futures
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from retinaface import RetinaFace, prepare_model


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def create_logger(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# ---------------------------------------------------------------------------
# Frame selection (ported from GenD)
# ---------------------------------------------------------------------------

def max_spread_permutation_pq(N, start=0):
    """Generate a permutation of 0..N-1 maximizing minimum distance to
    previously chosen elements at each step (priority-queue based)."""
    if not (0 <= start < N):
        raise ValueError("`start` must be in the range [0, N-1]")

    chosen = [start]
    dist = {i: abs(i - start) for i in range(N) if i != start}

    heap = [(-d, i) for i, d in dist.items()]
    heapq.heapify(heap)

    while heap:
        while True:
            neg_d, candidate = heapq.heappop(heap)
            current = -neg_d
            if dist.get(candidate, -1) == current:
                break
        chosen.append(candidate)
        del dist[candidate]
        for other in list(dist.keys()):
            new_d = abs(other - candidate)
            if new_d < dist[other]:
                dist[other] = new_d
                heapq.heappush(heap, (-new_d, other))

    return chosen


def get_frame_indices(total_frames, mode, num_frames, stride):
    """Return frame indices to extract based on mode."""
    if mode == 'fixed_num_frames':
        return np.linspace(0, total_frames - 1, num_frames, endpoint=True, dtype=int)
    elif mode == 'fixed_stride':
        return np.arange(0, total_frames, stride, dtype=int)
    elif mode == 'at_least':
        return max_spread_permutation_pq(total_frames, start=total_frames // 2)
    else:
        raise ValueError(f"Invalid mode: {mode}")


# ---------------------------------------------------------------------------
# Face alignment (ported from GenD — cv2.estimateAffinePartial2D + LMEDS)
# ---------------------------------------------------------------------------

def align_face(img, landmarks, target_size=(256, 256), scale=1.3, mask=None):
    """
    Align and crop face using 5-point landmarks.
    Uses cv2.estimateAffinePartial2D with LMEDS (robust to outliers).

    Args:
        img: BGR image
        landmarks: (5, 2) array — left_eye, right_eye, nose, left_mouth, right_mouth
        target_size: (width, height) output size
        scale: context margin around face (1.3 = 30% extra)
        mask: optional mask to warp with same transform

    Returns:
        (aligned_face, aligned_mask_or_None)
    """
    # Normalized reference points (GenD template)
    dst = np.array([
        [0.34, 0.46],
        [0.66, 0.46],
        [0.50, 0.64],
        [0.37, 0.82],
        [0.63, 0.82],
    ], dtype=np.float32)

    dst[:, 0] *= target_size[0]
    dst[:, 1] *= target_size[1]

    margin_rate = scale - 1
    x_margin = target_size[0] * margin_rate / 2.0
    y_margin = target_size[1] * margin_rate / 2.0

    dst[:, 0] += x_margin
    dst[:, 1] += y_margin
    dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
    dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

    src = landmarks.astype(np.float32)
    M = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)[0]

    if M is None:
        return None, None

    aligned = cv2.warpAffine(img, M, target_size, flags=cv2.INTER_LINEAR)

    aligned_mask = None
    if mask is not None:
        aligned_mask = cv2.warpAffine(mask, M, target_size, flags=cv2.INTER_NEAREST)

    return aligned, aligned_mask


# ---------------------------------------------------------------------------
# Video processing
# ---------------------------------------------------------------------------

def process_video(
    movie_path: Path,
    mask_path: Path,
    save_root: Path,
    model: RetinaFace,
    mode: str,
    num_frames: int,
    stride: int,
    target_size: tuple = (256, 256),
    scale: float = 1.3,
    logger=None,
):
    """Process a single video: detect faces, align, save crops + landmarks."""

    cap_org = cv2.VideoCapture(str(movie_path))
    if not cap_org.isOpened():
        if logger:
            logger.error(f"Failed to open {movie_path}")
        return

    cap_mask = None
    if mask_path is not None:
        cap_mask = cv2.VideoCapture(str(mask_path))
        if not cap_mask.isOpened():
            if logger:
                logger.error(f"Failed to open mask {mask_path}")
            cap_mask = None

    frame_count = int(cap_org.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        if logger:
            logger.warning(f"Empty video: {movie_path}")
        cap_org.release()
        if cap_mask:
            cap_mask.release()
        return

    frame_idxs = get_frame_indices(frame_count, mode, num_frames, stride)
    # For at_least mode, only take num_frames
    if mode == 'at_least':
        frame_idxs = frame_idxs[:num_frames]

    # Convert to set for O(1) lookup
    frame_set = set(int(idx) for idx in frame_idxs)

    # Output paths
    frames_dir = save_root / 'frames' / movie_path.stem
    landmarks_dir = save_root / 'landmarks' / movie_path.stem
    masks_dir = save_root / 'masks' / movie_path.stem

    num_saved = 0
    for cnt_frame in range(frame_count):
        ret_org, frame_org = cap_org.read()
        frame_mask = None
        if cap_mask is not None:
            ret_mask, frame_mask = cap_mask.read()
            if not ret_mask:
                frame_mask = None

        if not ret_org:
            break

        if cnt_frame not in frame_set:
            continue

        # Detect faces with RetinaFace
        try:
            xyxy, kpss = model.detect(frame_org)
        except Exception as e:
            if logger:
                logger.warning(f"Detection error frame {cnt_frame} of {movie_path}: {e}")
            continue

        # Fallback for sources that are *already* tight face crops (e.g. Celeb-DF-v3
        # FaceReenact/TalkingFace ship 256x256 pre-cropped video). RetinaFace expects the
        # face to be a sub-region of a larger image and finds nothing when it fills the
        # frame. A replicate border restores the surrounding context; measured 0/12 -> 12/12
        # detections on a previously-failing DaGAN clip, with no change on clips that
        # already worked. Upscaling does NOT help, which confirms context and not
        # resolution is the constraint.
        # Landmarks are shifted back into original-frame coordinates so alignment runs on
        # the unpadded frame exactly as it does for normally-detected frames — recovered
        # frames are therefore identical in convention to the ones already extracted.
        if len(xyxy) == 0:
            pad = max(frame_org.shape[0], frame_org.shape[1]) // 4
            try:
                padded = cv2.copyMakeBorder(frame_org, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
                xyxy, kpss = model.detect(padded)
            except Exception as e:
                if logger:
                    logger.warning(f"Padded detection error frame {cnt_frame} of {movie_path}: {e}")
                continue
            if len(xyxy):
                xyxy = xyxy.copy()
                kpss = kpss.copy()
                xyxy[:, [0, 2]] -= pad
                xyxy[:, [1, 3]] -= pad
                kpss[:, :, 0] -= pad
                kpss[:, :, 1] -= pad

        if len(xyxy) == 0:
            if logger:
                logger.warning(f"No faces in frame {cnt_frame} of {movie_path}")
            continue

        # Select face: if mask available, pick face with most mask overlap;
        # otherwise pick largest face.
        selected_landmarks = None

        if frame_mask is not None and frame_mask.sum() > 0:
            mask_gray = cv2.cvtColor(frame_mask, cv2.COLOR_BGR2GRAY) if len(frame_mask.shape) == 3 else frame_mask
            mask_binary = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)[1]

            max_intersection = 0
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i, :4].astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(mask_binary.shape[1], x2)
                y2 = min(mask_binary.shape[0], y2)
                face_region = mask_binary[y1:y2, x1:x2]
                intersection = face_region.sum()
                if intersection > max_intersection:
                    max_intersection = intersection
                    selected_landmarks = kpss[i]

        if selected_landmarks is None:
            # Pick largest face
            areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            idx = np.argmax(areas)
            selected_landmarks = kpss[idx]

        # Align face
        aligned_face, aligned_mask = align_face(
            frame_org, selected_landmarks,
            target_size=target_size, scale=scale,
            mask=frame_mask,
        )

        if aligned_face is None:
            if logger:
                logger.warning(f"Alignment failed frame {cnt_frame} of {movie_path}")
            continue

        # Save aligned face
        frames_dir.mkdir(parents=True, exist_ok=True)
        image_path = frames_dir / f"{cnt_frame:03d}.png"
        if not image_path.is_file():
            cv2.imwrite(str(image_path), aligned_face)

        # Save 5-point landmarks (post-alignment reference)
        landmarks_dir.mkdir(parents=True, exist_ok=True)
        land_path = landmarks_dir / f"{cnt_frame:03d}.npy"
        np.save(str(land_path), selected_landmarks)

        # Save mask
        if aligned_mask is not None:
            masks_dir.mkdir(parents=True, exist_ok=True)
            mask_save_path = masks_dir / f"{cnt_frame:03d}.png"
            _, binary_mask = cv2.threshold(aligned_mask, 1, 255, cv2.THRESH_BINARY)
            if len(binary_mask.shape) == 3:
                binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(str(mask_save_path), binary_mask)

        num_saved += 1

    cap_org.release()
    if cap_mask:
        cap_mask.release()

    total_target = len(frame_set)
    if num_saved < total_target and logger:
        logger.warning(
            f"{movie_path.stem}: only {num_saved}/{total_target} frames extracted successfully"
        )
        failed_log_path = save_root / 'failed_videos.txt'
        with open(failed_log_path, 'a') as f:
            f.write(f"{movie_path.stem},{num_saved},{total_target}\n")


def already_extracted(output_path, video_stem, mode, num_frames):
    """True if this video's frames were already fully extracted by a previous run.

    Only claims completeness for the modes with a known up-front target count.
    Videos that came out short (a clip with fewer frames than num_frames, or frames
    where detection failed) are deliberately not skipped, so a re-run retries them.
    """
    if mode not in ('fixed_num_frames', 'at_least'):
        return False
    frames_dir = Path(output_path) / 'frames' / video_stem
    if not frames_dir.is_dir():
        return False
    return sum(1 for f in os.scandir(frames_dir) if f.name.endswith('.png')) >= num_frames


def preprocess(dataset_path, mask_path, output_path, mode, num_frames, stride, logger, model,
               allowed_videos=None, skip_existing=False, video_exts=('.mp4',)):
    """Process all videos in a dataset directory.

    Args:
        dataset_path: Source directory containing videos.
        mask_path: Source directory containing mask videos (or None).
        output_path: Separate output directory for frames/landmarks/masks.
        mode: Frame selection mode.
        num_frames: Number of frames to extract.
        stride: Stride for fixed_stride mode.
        logger: Logger instance.
        model: RetinaFace model instance.
        allowed_videos: Optional set of absolute Path objects. If provided, only
            videos whose path is in the set will be processed (used for
            test-list filtering on Celeb-DF-v3).
        skip_existing: Skip videos already fully extracted into output_path. Face
            detection dominates runtime and runs even when the output PNG exists,
            so this is what makes re-running over a partly-processed dataset cheap.
        video_exts: Container extensions to pick up. Defaults to mp4-only, matching the
            original behaviour for every existing dataset. Deepfake-Eval-2024 is
            in-the-wild media and contains one .webm (a Fake in the official test split),
            which an mp4-only glob would silently drop.
    """
    movies_path_list = sorted([
        Path(p)
        for ext in video_exts
        for p in glob.glob(os.path.join(dataset_path, f'**/*{ext}'), recursive=True)
    ])
    if allowed_videos is not None:
        movies_path_list = [p for p in movies_path_list if p.resolve() in allowed_videos]
    if len(movies_path_list) == 0:
        logger.error(f"No videos found in {dataset_path}")
        return
    found = len(movies_path_list)

    num_skipped = 0
    if skip_existing:
        kept = [p for p in movies_path_list
                if not already_extracted(output_path, p.stem, mode, num_frames)]
        num_skipped = found - len(kept)
        movies_path_list = kept

    logger.info(
        f"{found} videos found in {dataset_path}"
        + (f" — {num_skipped} already extracted, {len(movies_path_list)} to process"
           if skip_existing else "")
    )
    if not movies_path_list:
        logger.info(f"Nothing to do for {dataset_path}")
        return

    # Initialize failed-videos log for this sub-dataset
    failed_log_path = Path(output_path) / 'failed_videos.txt'
    with open(failed_log_path, 'w') as f:
        f.write("video_name,extracted_frames,total_frames\n")

    masks_path_list = []
    if mask_path is not None:
        masks_path_list = sorted([
            Path(p) for p in glob.glob(os.path.join(mask_path, '**/*.mp4'), recursive=True)
        ])
        logger.info(f"{len(masks_path_list)} masks found in {mask_path}")

    start_time = time.monotonic()

    # Note: RetinaFace ONNX model is NOT thread-safe when using GPU.
    # Use ThreadPoolExecutor with max_workers=1 for GPU, or process sequentially.
    # For CPU-only, multiple workers are fine since each thread gets its own session.
    num_processes = min(os.cpu_count(), 8)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for movie_path in movies_path_list:
            video_mask_path = None
            if mask_path is not None:
                video_mask_path = next(
                    (p for p in masks_path_list if p.stem == movie_path.stem), None
                )
                if video_mask_path is None:
                    logger.warning(f"No mask for video {movie_path}")

            futures.append(
                executor.submit(
                    process_video,
                    movie_path,
                    video_mask_path,
                    Path(output_path),
                    model,
                    mode,
                    num_frames,
                    stride,
                    logger=logger,
                )
            )

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(movies_path_list)):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error processing video: {e}")

    duration_minutes = (time.monotonic() - start_time) / 60
    logger.info(f"Total time taken: {duration_minutes:.2f} minutes")


if __name__ == '__main__':
    yaml_path = './config.yaml'
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.parser.ParserError as e:
        print("YAML file parsing error:", e)
        sys.exit(1)

    dataset_name = config['preprocess']['dataset_name']['default']
    dataset_root_path = config['preprocess']['dataset_root_path']['default']
    output_root_path = config['preprocess']['output_root_path']['default']
    comp = config['preprocess']['comp']['default']
    mode = config['preprocess']['mode']['default']
    stride = config['preprocess']['stride']['default']
    num_frames = config['preprocess']['num_frames']['default']
    # Optional keys — defaulted here so older config.yaml copies keep working.
    skip_existing = config['preprocess'].get('skip_existing', {}).get('default', True)
    celebdfv3_families = config['preprocess'].get('celebdfv3_families', {}).get(
        'default', ['FaceSwap', 'FaceReenact', 'TalkingFace'])
    # Opt-in override of the Deepfake-Eval-2024 no-leak guard. Default False: the train
    # ("Finetuning Set") split stays unextracted unless explicitly requested.
    eval24_include_finetuning_split = bool(
        config['preprocess'].get('eval24_include_finetuning_split', {}).get('default', False))

    # Source dataset path (original videos)
    dataset_path = Path(os.path.join(dataset_root_path, dataset_name))

    # Output root — mirrors the dataset sub-structure beneath it
    output_base = Path(output_root_path) / dataset_name
    output_base.mkdir(parents=True, exist_ok=True)

    # Create logger
    log_path = f'./logs/{dataset_name}.log'
    logger = create_logger(log_path)

    # Initialize RetinaFace model (once, shared across all videos)
    logger.info("Initializing RetinaFace model...")
    model = prepare_model(det_thres=0.5, nms_thresh=0.4)
    logger.info("RetinaFace model ready.")

    # Default container extensions; a dataset branch may widen this.
    video_exts = ('.mp4',)

    # Define sub-dataset paths based on dataset name
    ## FaceForensics++
    if dataset_name == 'FaceForensics++':
        sub_dataset_names = [
            "original_sequences/youtube", "original_sequences/actors",
            "manipulated_sequences/Deepfakes",
            "manipulated_sequences/Face2Face", "manipulated_sequences/FaceSwap",
            "manipulated_sequences/NeuralTextures", "manipulated_sequences/FaceShifter",
            "manipulated_sequences/DeepFakeDetection",
        ]
        sub_dataset_paths = [Path(os.path.join(dataset_path, name, comp)) for name in sub_dataset_names]
        mask_dataset_names = [
            "manipulated_sequences/Deepfakes", "manipulated_sequences/Face2Face",
            "manipulated_sequences/FaceSwap", "manipulated_sequences/NeuralTextures",
            "manipulated_sequences/DeepFakeDetection",
        ]
        mask_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in mask_dataset_names]

    ## Celeb-DF-v1
    elif dataset_name == 'Celeb-DF-v1':
        sub_dataset_names = ['Celeb-real', 'Celeb-synthesis', 'YouTube-real']
        sub_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in sub_dataset_names]

    ## Celeb-DF-v2
    elif dataset_name == 'Celeb-DF-v2':
        sub_dataset_names = ['Celeb-real', 'Celeb-synthesis', 'YouTube-real']
        sub_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in sub_dataset_names]

    ## Celeb-DF-v3 (all manipulation families + both real folders, restricted to test list)
    ## Layout: Celeb-synthesis/<family>/<generator>/*.mp4
    ##   FaceSwap     — BlendFace, Celeb-DF-v2, GHOST, HifiFace, InSwapper,
    ##                  MobileFaceSwap, SimSwap, UniFace          (8 generators)
    ##   FaceReenact  — DaGAN, FSRT, HyperReenact, LIA, LivePortrait, MCNET, TPSMM   (7)
    ##   TalkingFace  — AniTalker, EchoMimic, EDTalk, FLOAT, IP_LAP,
    ##                  Real3DPortrait, SadTalker                 (7)
    elif dataset_name == 'Celeb-DF-v3':
        test_list_path = dataset_path / 'List_of_testing_videos.txt'
        if not test_list_path.is_file():
            raise FileNotFoundError(f"Missing test list: {test_list_path}")
        with open(test_list_path) as f:
            test_rels = [line.strip().split()[1] for line in f if line.strip()]
        allowed_videos = {(dataset_path / rel).resolve() for rel in test_rels}

        synthesis_root = dataset_path / 'Celeb-synthesis'
        if not synthesis_root.is_dir():
            raise FileNotFoundError(f"Celeb-synthesis root missing: {synthesis_root}")

        # Families are discovered from disk rather than hardcoded, then intersected with
        # the configured selection, so a newly-added family is picked up automatically.
        available_families = sorted(p.name for p in synthesis_root.iterdir() if p.is_dir())
        unknown = [f for f in celebdfv3_families if f not in available_families]
        if unknown:
            raise ValueError(
                f"Configured Celeb-DF-v3 families not present on disk: {unknown}. "
                f"Available: {available_families}"
            )
        families = [f for f in available_families if f in celebdfv3_families]
        logger.info(f"Celeb-DF-v3 families selected: {families} (available: {available_families})")

        # Each generator is its own sub-dataset so that identical video stems across
        # generators (e.g. id0_id1_0001 appears under every FaceSwap generator) land in
        # separate output directories instead of overwriting each other.
        sub_dataset_paths = []
        for family in families:
            generators = sorted(p for p in (synthesis_root / family).iterdir() if p.is_dir())
            if not generators:
                logger.warning(f"No generator directories under {synthesis_root / family}")
            sub_dataset_paths += generators
        sub_dataset_paths += [dataset_path / 'Celeb-real', dataset_path / 'YouTube-real']

    ## DFDCP
    elif dataset_name == 'DFDCP':
        sub_dataset_names = ['original_videos', 'method_A', 'method_B']
        sub_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in sub_dataset_names]

    ## DFDC
    elif dataset_name == 'DFDC':
        sub_train_dataset_names = ["dfdc_train_part_" + str(i) for i in range(0, 50)]
        sub_train_dataset_paths = [Path(os.path.join(dataset_path, 'train', name)) for name in sub_train_dataset_names]
        sub_dataset_paths = [Path(os.path.join(dataset_path, 'test'))] + sub_train_dataset_paths

    ## DeeperForensics-1.0
    elif dataset_name == 'DeeperForensics-1.0':
        real_sub = ['source_videos/' + n for n in os.listdir(os.path.join(dataset_path, 'source_videos'))]
        fake_sub = ['manipulated_videos/' + n for n in os.listdir(os.path.join(dataset_path, 'manipulated_videos'))]
        sub_dataset_names = real_sub + fake_sub
        sub_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in sub_dataset_names]

    ## UADFV
    elif dataset_name == 'UADFV':
        sub_dataset_names = ['fake', 'real']
        sub_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in sub_dataset_names]

    ## Deepfake-Eval-2024 (Chandra et al., 2025) — in-the-wild deployment benchmark
    ## Flat directory of 2,036 videos; the class lives in the metadata CSV, not the layout,
    ## so real/fake separation happens in rearrange.py (same pattern as DFDC test).
    ## By default only the official `test` split (815 videos: 386 fake / 429 real) is
    ## extracted — it is used zero-shot, and the 1,221-video `train` (a.k.a. "Finetuning Set")
    ## split is left untouched so it cannot leak into any finetuning by accident.
    ##
    ## `eval24_include_finetuning_split: true` overrides that guard and additionally extracts
    ## the 1,221 train videos. Authorised 2026-08-12 to build an out-of-domain *validation*
    ## set (VALmix) for checkpoint selection. The guard stays default-off and every override
    ## is logged at WARNING, because it has a real cost:
    ##
    ##   Selecting checkpoints on in-the-wild 2024 media means the Deepfake-Eval-2024 *test*
    ##   AUC is no longer a strictly zero-shot number — it becomes a held-out test with
    ##   in-distribution model selection. That benchmark is the headline deployment target,
    ##   so any paper using an override-built validation set must say so explicitly.
    ##
    ## The test split itself is never contaminated: train and test are disjoint by the CSV's
    ## own `Finetuning Set` column, and they are staged under different source names.
    elif dataset_name == 'Deepfake-Eval-2024':
        # The on-disk directory name differs from the canonical dataset name.
        dataset_path = Path(dataset_root_path) / 'video_eval24'
        metadata_path = dataset_path / 'video-metadata-publish-with-links.csv'
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing metadata CSV: {metadata_path}")
        metadata = pd.read_csv(metadata_path)
        test_rows = metadata[metadata['Finetuning Set'] == 'test']
        wanted_rows = test_rows
        if eval24_include_finetuning_split:
            train_rows = metadata[metadata['Finetuning Set'] == 'train']
            wanted_rows = pd.concat([test_rows, train_rows])
            logger.warning(
                "eval24_include_finetuning_split=True — extracting the %d-video train "
                "('Finetuning Set') split IN ADDITION to the %d test videos. This overrides "
                "the default no-leak guard. Deepfake-Eval-2024 test AUC is only zero-shot if "
                "these train videos are never used for training or model selection.",
                len(train_rows), len(test_rows),
            )
        allowed_videos = {(dataset_path / f).resolve() for f in wanted_rows['Filename']}
        missing = [f for f in test_rows['Filename'] if not (dataset_path / f).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} Deepfake-Eval-2024 test videos missing on disk, "
                f"e.g. {missing[:3]}"
            )
        train_note = ("INCLUDED (guard overridden)" if eval24_include_finetuning_split
                      else f"({len(metadata) - len(test_rows)}) deliberately skipped")
        logger.info(
            f"Deepfake-Eval-2024: {len(allowed_videos)} videos to extract "
            f"({(wanted_rows['Video Ground Truth'] == 'Fake').sum()} fake / "
            f"{(wanted_rows['Video Ground Truth'] == 'Real').sum()} real); "
            f"train split {train_note}"
        )
        # One test video is .webm; an mp4-only glob would silently drop it.
        video_exts = ('.mp4', '.webm')
        sub_dataset_paths = [dataset_path]
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized")

    # Check paths
    if not Path(dataset_path).exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    if 'sub_dataset_paths' in dir() and len(sub_dataset_paths) != 0:
        for sub_dataset_path in sub_dataset_paths:
            if not Path(sub_dataset_path).exists():
                logger.error(f"Sub Dataset path does not exist: {sub_dataset_path}")
                sys.exit(1)

        video_filter = allowed_videos if 'allowed_videos' in dir() else None

        for sub_dataset_path in sub_dataset_paths:
            # Compute output path mirroring source structure under output_base
            relative_path = sub_dataset_path.relative_to(dataset_path)
            output_path = output_base / relative_path
            output_path.mkdir(parents=True, exist_ok=True)

            # Only part of FF++ has masks
            if dataset_name == 'FaceForensics++' and sub_dataset_path.parent in mask_dataset_paths:
                mask_dataset_path = os.path.join(sub_dataset_path.parent, "masks")
                preprocess(sub_dataset_path, mask_dataset_path, output_path, mode, num_frames, stride, logger, model,
                           allowed_videos=video_filter, skip_existing=skip_existing, video_exts=video_exts)
            else:
                preprocess(sub_dataset_path, None, output_path, mode, num_frames, stride, logger, model,
                           allowed_videos=video_filter, skip_existing=skip_existing, video_exts=video_exts)
    else:
        logger.error(f"No sub-dataset paths found")
        sys.exit(1)

    logger.info("Face cropping complete!")

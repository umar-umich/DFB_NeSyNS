# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-29
# description: Data pre-processing script for deepfake dataset.


"""
Original dataset structure before the preprocessing:

-FaceForensics++
    -original_sequences
        -youtube
            -c23
                -videos
                    *.mp4
    -manipulated_sequences
        -Deepfakes
            -c23
                -videos
        -Face2Face
            -c23
                -videos
        -FaceSwap
            -c23
                -videos
        -NeuralTextures
            -c23
                -videos
        -FaceShifter
            -c23
                -videos
        -DeepFakeDetection
            -c23
                -videos
                
-Celeb-DF-v1/v2
    -Celeb-synthesis
        -videos
    -Celeb-real
        -videos
    -YouTube-real
        -videos

-DFDCP
    -method_A
    -method_B
    -original_videos

-DeeperForensics-1.0
    -manipulated_videos
    -source_videos

We then additionally obtain "frames", "landmarks", and "mask" directories in same directory as the "videos" folder.
"""


import os
import sys
import time
import cv2
import dlib
import yaml
import logging
import datetime
import glob
import concurrent.futures
import numpy as np
from tqdm import tqdm
from pathlib import Path
from imutils import face_utils
from skimage import transform as trans


# Minimum number of successfully extracted face frames required to keep a video.
# Videos that yield fewer than this many good frames are discarded and logged.
# Videos at or above this threshold are kept and padded to num_frames by
# cyclically repeating the available frames (standard practice in video models).
MIN_FRAMES_THRESHOLD = 8


def create_logger(log_path):
    """
    Creates a logger object and saves all messages to a file.

    Args:
        log_path (str): The path to save the log file.

    Returns:
        logger: The logger object.
    """
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


def get_keypts(image, face, predictor, face_detector):
    shape = predictor(image, face)
    
    leye   = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
    reye   = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
    nose   = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
    lmouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
    rmouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)
    
    pts = np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)
    return pts


def extract_aligned_face_dlib(face_detector, predictor, image, res=256, mask=None):
    def img_align_crop(img, landmark=None, outsize=None, scale=1.3, mask=None):
        M = None
        target_size = [112, 112]
        dst = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)

        if target_size[1] == 112:
            dst[:, 0] += 8.0

        dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0]
        dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]

        target_size = outsize

        margin_rate = scale - 1
        x_margin = target_size[0] * margin_rate / 2.
        y_margin = target_size[1] * margin_rate / 2.

        dst[:, 0] += x_margin
        dst[:, 1] += y_margin

        dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
        dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

        src = landmark.astype(np.float32)

        tform = trans.SimilarityTransform()
        tform.estimate(src, dst)
        M = tform.params[0:2, :]

        img = cv2.warpAffine(img, M, (target_size[1], target_size[0]))

        if outsize is not None:
            img = cv2.resize(img, (outsize[1], outsize[0]))
        
        if mask is not None:
            mask = cv2.warpAffine(mask, M, (target_size[1], target_size[0]))
            mask = cv2.resize(mask, (outsize[1], outsize[0]))
            return img, mask
        else:
            return img, None

    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = face_detector(rgb, 1)

    if len(faces):
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        landmarks = get_keypts(rgb, face, predictor, face_detector)
        cropped_face, mask_face = img_align_crop(rgb, landmarks, outsize=(res, res), mask=mask)
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
        
        face_align = face_detector(cropped_face, 1)
        if len(face_align) == 0:
            return None, None, None
        landmark = predictor(cropped_face, face_align[0])
        landmark = face_utils.shape_to_np(landmark)

        return cropped_face, landmark, mask_face
    else:
        return None, None, None


def video_manipulate(
    movie_path: Path,
    mask_path: Path,
    save_path: Path,
    mode: str,
    num_frames: int,
    stride: int,
) -> None:
    """
    Processes a single video file by detecting and cropping the largest face
    in each frame and saving the results.

    Frame saving policy
    -------------------
    - If extracted good frames  >= num_frames  : save exactly num_frames frames
                                                  (original behaviour).
    - If extracted good frames  >= MIN_FRAMES_THRESHOLD (8)
                              and < num_frames : pad to num_frames by cycling
                                                  through the available frames
                                                  and save. This preserves short
                                                  but usable videos.
    - If extracted good frames  < MIN_FRAMES_THRESHOLD : discard the video and
                                                  log it to failed_videos.txt
                                                  (original behaviour).

    Padding is done by cycling (modular index), which is the standard approach
    used by VideoMAE, TimeSformer, etc. to handle variable-length clips. The
    saved file structure is always exactly num_frames images so no downstream
    dataset loader changes are needed.
    """

    face_detector = dlib.get_frontal_face_detector()
    predictor_path = './dlib_tools/shape_predictor_81_face_landmarks.dat'
    if not os.path.exists(predictor_path):
        logger.error(f"Predictor path does not exist: {predictor_path}")
        sys.exit()
    face_predictor = dlib.shape_predictor(predictor_path)

    def facecrop(
        org_path: Path,
        mask_path: Path,
        save_path: Path,
        mode: str,
        num_frames: int,
        stride: int,
        face_predictor: dlib.shape_predictor,
        face_detector: dlib.fhog_object_detector,
        margin: float = 0.5,
        visualization: bool = False,
    ) -> None:

        assert org_path.exists(), f"Video file {org_path} does not exist."
        cap_org = cv2.VideoCapture(str(org_path))
        if not cap_org.isOpened():
            logger.error(f"Failed to open {org_path}")
            return

        if mask_path is not None:
            cap_mask = cv2.VideoCapture(str(mask_path))
            if not cap_mask.isOpened():
                logger.error(f"Failed to open {mask_path}")
                return

        frame_count_org = int(cap_org.get(cv2.CAP_PROP_FRAME_COUNT))

        if mode == 'fixed_num_frames':
            frame_idxs = np.linspace(0, frame_count_org - 1, num_frames, endpoint=True, dtype=int)
            desired_frames = num_frames
        elif mode == 'fixed_stride':
            frame_idxs = np.arange(0, frame_count_org, stride, dtype=int)
            desired_frames = num_frames if num_frames > 0 else len(frame_idxs)

        # Iterate from end → start so dark/occluded frames at the start are
        # skipped naturally once we have enough good frames.
        frame_idxs_reversed = frame_idxs[::-1]

        extracted_frames = []

        for frame_idx in frame_idxs_reversed:
            if len(extracted_frames) >= desired_frames:
                break

            cap_org.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret_org, frame_org = cap_org.read()
            if not ret_org:
                logger.warning(f"Failed to read frame {frame_idx} of {org_path}")
                continue

            frame_mask = None
            if mask_path is not None:
                cap_mask.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ret_mask, frame_mask = cap_mask.read()
                if not ret_mask:
                    logger.warning(f"Failed to read mask {frame_idx} of {mask_path}")
                    continue

            cropped_face, landmarks, masks = extract_aligned_face_dlib(
                face_detector, face_predictor, frame_org, mask=frame_mask)

            if cropped_face is None:
                logger.warning(f"No face in frame {frame_idx} of {org_path}")
                continue
            if landmarks is None:
                logger.warning(f"No landmarks in frame {frame_idx} of {org_path}")
                continue

            extracted_frames.append({
                'original_frame_idx': int(frame_idx),
                'cropped_face': cropped_face,
                'landmarks': landmarks,
                'masks': masks if mask_path is not None else None,
            })

        successfully_processed = len(extracted_frames)

        # ── Decide what to do based on how many good frames we got ────────
        if successfully_processed < MIN_FRAMES_THRESHOLD:
            # Too few usable frames — discard the video entirely.
            logger.warning(
                f"Skipping {org_path.stem}: only {successfully_processed} good frames "
                f"(minimum required: {MIN_FRAMES_THRESHOLD})"
            )
            failed_log_path = save_path / 'failed_videos.txt'
            with open(failed_log_path, 'a') as f:
                f.write(f"{org_path.stem},{successfully_processed},{desired_frames}\n")
            cap_org.release()
            if mask_path is not None:
                cap_mask.release()
            return

        if successfully_processed < desired_frames:
            # Enough frames to be useful but fewer than desired — pad by cycling.
            # Example: 10 good frames, desired 32 → indices 0..9, 0..9, 0..9, 0..1
            logger.warning(
                f"Padding {org_path.stem}: {successfully_processed} good frames "
                f"padded to {desired_frames} by cycling "
                f"(minimum threshold {MIN_FRAMES_THRESHOLD} was met)"
            )
            padded_frames = [
                extracted_frames[i % successfully_processed]
                for i in range(desired_frames)
            ]
            # Reassign so the save loop below always works on a desired_frames-length list
            extracted_frames = padded_frames

        # Sort by original frame index to maintain chronological order.
        # (Padded duplicates share their source frame's index — sort is stable
        #  so duplicates stay in the order they were appended.)
        extracted_frames.sort(key=lambda x: x['original_frame_idx'])

        # ── Save all frames ───────────────────────────────────────────────
        for save_idx, frame_data in enumerate(extracted_frames):
            cropped_face = frame_data['cropped_face']
            landmarks    = frame_data['landmarks']
            masks        = frame_data['masks']

            save_path_ = save_path / 'frames' / org_path.stem
            save_path_.mkdir(parents=True, exist_ok=True)

            image_path = save_path_ / f"{save_idx:03d}.png"
            if not image_path.is_file():
                cv2.imwrite(str(image_path), cropped_face)

            land_path = save_path / 'landmarks' / org_path.stem / f"{save_idx:03d}.npy"
            os.makedirs(os.path.dirname(land_path), exist_ok=True)
            np.save(str(land_path), landmarks)

            if masks is not None:
                mask_save_path = save_path / 'masks' / org_path.stem / f"{save_idx:03d}.png"
                os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)
                _, binary_mask = cv2.threshold(masks, 1, 255, cv2.THRESH_BINARY)
                cv2.imwrite(str(mask_save_path), binary_mask)

        cap_org.release()
        if mask_path is not None:
            cap_mask.release()

    try:
        facecrop(movie_path, mask_path, save_path, mode, num_frames, stride, face_predictor, face_detector)
    except Exception as e:
        logger.error(f"Error processing video {movie_path}: {e}")


def preprocess(dataset_path, mask_path, output_path, mode, num_frames, stride, logger):
    movies_path_list = sorted([Path(p) for p in glob.glob(os.path.join(dataset_path, '**/*.mp4'), recursive=True)])
    if len(movies_path_list) == 0:
        logger.error(f"No videos found in {dataset_path}")
        sys.exit()
    logger.info(f"{len(movies_path_list)} videos found in {dataset_path}")

    failed_log_path = output_path / 'failed_videos.txt'
    if failed_log_path.exists():
        failed_log_path.unlink()
    with open(failed_log_path, 'w') as f:
        f.write("video_name,extracted_frames,desired_frames\n")

    masks_path_list = []
    if mask_path is not None and os.path.exists(mask_path):
        masks_path_list = sorted([Path(p) for p in glob.glob(os.path.join(mask_path, '**/*.mp4'), recursive=True)])
        if len(masks_path_list) == 0:
            logger.warning(f"No masks found in {mask_path}")
        else:
            logger.info(f"{len(masks_path_list)} masks found in {mask_path}")

    start_time = time.monotonic()
    num_processes = os.cpu_count()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for movie_path in movies_path_list:
            current_mask_path = None
            if mask_path is not None and len(masks_path_list) > 0:
                if movie_path.stem in [path.stem for path in masks_path_list]:
                    current_mask_path = next(
                        (path for path in masks_path_list if path.stem == movie_path.stem), None
                    )

            futures.append(
                executor.submit(
                    video_manipulate,
                    movie_path,
                    current_mask_path,
                    output_path,
                    mode,
                    num_frames,
                    stride,
                )
            )

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(movies_path_list)):
            logger.info(f"Current time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error processing video: {e}")

        end_time = time.monotonic()
        duration_minutes = (end_time - start_time) / 60
        logger.info(f"Total time taken: {duration_minutes:.2f} minutes")

        failed_log_path = output_path / 'failed_videos.txt'
        if failed_log_path.exists():
            with open(failed_log_path, 'r') as f:
                lines = f.readlines()
                failed_count = len(lines) - 1  # subtract header
                if failed_count > 0:
                    logger.warning(
                        f"{failed_count} videos failed (< {MIN_FRAMES_THRESHOLD} good frames). "
                        f"Check {failed_log_path}"
                    )
                else:
                    logger.info("All videos processed successfully!")
                    failed_log_path.unlink()


if __name__ == '__main__':
    yaml_path = './config.yaml'
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.parser.ParserError as e:
        print("YAML file parsing error:", e)

    dataset_name      = config['preprocess']['dataset_name']['default']
    dataset_root_path = config['preprocess']['dataset_root_path']['default']
    output_root_path  = config['preprocess']['output_root_path']['default']
    comp              = config['preprocess']['comp']['default']
    mode              = config['preprocess']['mode']['default']
    stride            = config['preprocess']['stride']['default']
    num_frames        = config['preprocess']['num_frames']['default']

    dataset_path = Path(os.path.join(dataset_root_path, dataset_name))

    output_base = Path(output_root_path) / dataset_name
    output_base.mkdir(parents=True, exist_ok=True)

    log_path = f'./logs/{dataset_name}.log'
    logger = create_logger(log_path)

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

    elif dataset_name == 'Celeb-DF-v1':
        sub_dataset_names = ['Celeb-real', 'Celeb-synthesis', 'YouTube-real']
        sub_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in sub_dataset_names]

    elif dataset_name == 'Celeb-DF-v2':
        sub_dataset_names = ['Celeb-real', 'Celeb-synthesis', 'YouTube-real']
        sub_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in sub_dataset_names]

    elif dataset_name == 'DFDCP':
        sub_dataset_names = ['original_videos', 'method_A', 'method_B']
        sub_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in sub_dataset_names]

    elif dataset_name == 'DFDC':
        sub_dataset_names = ['test', 'train']
        sub_train_dataset_names = ["dfdc_train_part_" + str(i) for i in range(0, 50)]
        sub_train_dataset_paths = [
            Path(os.path.join(dataset_path, 'train', name)) for name in sub_train_dataset_names
        ]
        sub_dataset_paths = [Path(os.path.join(dataset_path, 'test'))] + sub_train_dataset_paths

    elif dataset_name == 'DeeperForensics-1.0':
        real_sub_dataset_names = [
            'source_videos/' + name
            for name in os.listdir(os.path.join(dataset_path, 'source_videos'))
        ]
        fake_sub_dataset_names = [
            'manipulated_videos/' + name
            for name in os.listdir(os.path.join(dataset_path, 'manipulated_videos'))
        ]
        real_sub_dataset_names.extend(fake_sub_dataset_names)
        sub_dataset_names = real_sub_dataset_names
        sub_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in sub_dataset_names]

    elif dataset_name == 'UADFV':
        sub_dataset_names = ['fake', 'real']
        sub_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in sub_dataset_names]

    else:
        raise ValueError(f"Dataset {dataset_name} not recognized")

    if not Path(dataset_path).exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        sys.exit()

    if 'sub_dataset_paths' in globals() and len(sub_dataset_paths) != 0:
        for sub_dataset_path in sub_dataset_paths:
            if not Path(sub_dataset_path).exists():
                logger.error(f"Sub Dataset path does not exist: {sub_dataset_path}")
                sys.exit()

        for sub_dataset_path in sub_dataset_paths:
            relative_path = sub_dataset_path.relative_to(dataset_path)
            output_path = output_base / relative_path
            output_path.mkdir(parents=True, exist_ok=True)

            if dataset_name == 'FaceForensics++' and sub_dataset_path.parent in mask_dataset_paths:
                mask_dataset_path = os.path.join(sub_dataset_path.parent, "masks")
                preprocess(sub_dataset_path, mask_dataset_path, output_path, mode, num_frames, stride, logger)
            else:
                preprocess(sub_dataset_path, None, output_path, mode, num_frames, stride, logger)
    else:
        logger.error(f"Sub Dataset path does not exist: {sub_dataset_paths}")
        sys.exit()

    logger.info("Face cropping complete!")
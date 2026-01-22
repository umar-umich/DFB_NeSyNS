# author: Modified for Temporal Deepfake Detection
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2026-01-21
# description: Temporal-aware data pre-processing script for deepfake dataset with stabilized face crops.

"""
Temporal preprocessing strategy with stride:
- Extract N segments (e.g., 8 for broader temporal coverage) from each video
- Each segment contains M frames (e.g., 16 for VideoMAE) with stride S
- Stride allows skipping frames: stride=1 (consecutive), stride=2 (skip 1), stride=3 (skip 2)
- Detect face in multiple candidate frames per segment
- Choose the transformation from the frame with the largest/best face
- Apply that transformation to all frames in the segment for temporal stability
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


def create_logger(log_path):
    """
    Creates a logger object and saves all messages to a file.

    Args:
        log_path (str): The path to save the log file.

    Returns:
        logger: The logger object.
    """
    # Create logger object
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create file handler and set the formatter
    fh = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(fh)

    # Add a stream handler to print to console
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def get_keypts(image, face, predictor, face_detector):
    """Extract 5 key facial landmarks for alignment."""
    # detect the facial landmarks for the selected face
    shape = predictor(image, face)
    
    # select the key points for the eyes, nose, and mouth
    leye = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
    reye = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
    nose = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
    lmouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
    rmouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)
    
    pts = np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)

    return pts


def extract_aligned_face_dlib(face_detector, predictor, image, res=256, mask=None):
    """
    Extract and align face using dlib with similarity transform.
    Returns transformation matrix for applying to other frames.
    """
    def img_align_crop(img, landmark=None, outsize=None, scale=1.3, mask=None):
        """ 
        align and crop the face according to the given bbox and landmarks
        landmark: 5 key points
        """

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

        # move
        dst[:, 0] += x_margin
        dst[:, 1] += y_margin

        # resize
        dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
        dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

        src = landmark.astype(np.float32)

        # use skimage tranformation
        tform = trans.SimilarityTransform()
        tform.estimate(src, dst)
        M = tform.params[0:2, :]

        img = cv2.warpAffine(img, M, (target_size[1], target_size[0]))

        if outsize is not None:
            img = cv2.resize(img, (outsize[1], outsize[0]))
        
        if mask is not None:
            mask = cv2.warpAffine(mask, M, (target_size[1], target_size[0]))
            mask = cv2.resize(mask, (outsize[1], outsize[0]))
            return img, mask, M
        else:
            return img, None, M

    # Image size
    height, width = image.shape[:2]

    # Convert to rgb
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect with dlib
    faces = face_detector(rgb, 1)
    if len(faces):
        # For now only take the biggest face
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        
        # Get the landmarks/parts for the face in box d only with the five key points
        landmarks = get_keypts(rgb, face, predictor, face_detector)

        # Align and crop the face
        cropped_face, mask_face, transform_matrix = img_align_crop(rgb, landmarks, outsize=(res, res), mask=mask)
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
        
        # Extract the all landmarks from the aligned face
        face_align = face_detector(cropped_face, 1)
        if len(face_align) == 0:
            return None, None, None, None, 0
        landmark = predictor(cropped_face, face_align[0])
        landmark = face_utils.shape_to_np(landmark)

        # Return face size as quality metric
        face_size = face.width() * face.height()
        
        return cropped_face, landmark, mask_face, transform_matrix, face_size
    
    else:
        return None, None, None, None, 0


def apply_transform_to_frame(frame, transform_matrix, mask=None, res=256):
    """Apply pre-computed transformation matrix to frame for temporal stability."""
    target_size = (res, res)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    transformed = cv2.warpAffine(rgb, transform_matrix, target_size)
    transformed = cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR)
    
    if mask is not None:
        mask_transformed = cv2.warpAffine(mask, transform_matrix, target_size)
        return transformed, mask_transformed
    
    return transformed, None


def video_manipulate(
    movie_path: Path,
    mask_path: Path,
    save_path: Path,
    num_segments: int, 
    frames_per_segment: int,
    frame_stride: int,
    logger
) -> None:
    """
    Processes a single video file with temporal stability and stride.
    
    Strategy:
    1. Divide video into N segments
    2. For each segment, extract frames_per_segment frames with stride
    3. Detect faces in candidate frames (first, middle, last + 2 random)
    4. Use the transformation from the frame with the largest face
    5. Apply to all frames in the segment for temporal consistency
    
    Args:
        movie_path (Path): Path to the video file to process.
        mask_path (Path): Path to the mask file (if available).
        save_path (Path): Path to save preprocessed outputs.
        num_segments (int): Number of temporal segments to extract.
        frames_per_segment (int): Number of frames per segment.
        frame_stride (int): Stride between frames (1=consecutive, 2=skip 1, etc).
        logger: Logger instance.

    Returns:
        None
    """

    # Define face detector and predictor models
    face_detector = dlib.get_frontal_face_detector()
    predictor_path = './dlib_tools/shape_predictor_81_face_landmarks.dat'
    ## Check if predictor path exists
    if not os.path.exists(predictor_path):
        logger.error(f"Predictor path does not exist: {predictor_path}")
        sys.exit()
    face_predictor = dlib.shape_predictor(predictor_path)
    
    # Open the video file
    assert movie_path.exists(), f"Video file {movie_path} does not exist."
    cap_org = cv2.VideoCapture(str(movie_path))
    if not cap_org.isOpened():
        logger.error(f"Failed to open {movie_path}")
        return

    cap_mask = None
    if mask_path is not None and mask_path.exists():
        cap_mask = cv2.VideoCapture(str(mask_path))
        if not cap_mask.isOpened():
            logger.warning(f"Failed to open mask {mask_path}")
            cap_mask = None
    
    # Get the number of frames in the video
    frame_count = int(cap_org.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frames needed per segment considering stride
    frames_needed_per_segment = (frames_per_segment - 1) * frame_stride + 1
    
    # Validate video length
    min_required_frames = frames_needed_per_segment * num_segments
    if frame_count < frames_needed_per_segment:
        logger.warning(f"Video {movie_path.stem} has only {frame_count} frames, "
                      f"need at least {frames_needed_per_segment} frames per segment "
                      f"(stride={frame_stride})")
        # Adjust num_segments to fit available frames
        actual_num_segments = max(1, frame_count // frames_needed_per_segment)
        if actual_num_segments < num_segments:
            logger.warning(f"Reducing to {actual_num_segments} segments for {movie_path.stem}")
            num_segments = actual_num_segments
    
    # Calculate evenly spaced segment start positions
    # Use max to ensure we don't go negative
    max_start = max(0, frame_count - frames_needed_per_segment)
    segment_boundaries = np.linspace(0, max_start, num_segments, endpoint=True, dtype=int)
    
    logger.info(f"Processing {movie_path.stem}: {frame_count} frames -> "
                f"{num_segments} segments × {frames_per_segment} frames (stride={frame_stride}) = "
                f"{num_segments * frames_per_segment} extracted frames")
    logger.debug(f"Segment boundaries: {segment_boundaries}, "
                f"frames needed per segment: {frames_needed_per_segment}")
    
    # Process each segment
    segments_processed = 0
    for seg_idx, seg_start in enumerate(segment_boundaries):
        seg_end = seg_start + frames_needed_per_segment
        
        if seg_end > frame_count:
            logger.warning(f"Segment {seg_idx} extends beyond video length, skipping")
            continue
        
        logger.debug(f"Segment {seg_idx}: frames {seg_start}-{seg_end-1} (stride={frame_stride})")
        
        # Generate frame indices for this segment with stride
        frame_indices = list(range(seg_start, seg_end, frame_stride))[:frames_per_segment]
        
        if len(frame_indices) < frames_per_segment:
            logger.warning(f"Segment {seg_idx} only has {len(frame_indices)} frames "
                          f"(need {frames_per_segment}), skipping")
            continue
        
        # Find best anchor frame for face detection in this segment
        # Check: first, last, middle, and 2 additional frames
        candidate_positions = [
            0,  # First frame
            len(frame_indices) - 1,  # Last frame
            len(frame_indices) // 2,  # Middle frame
            len(frame_indices) // 4,  # Quarter frame
            3 * len(frame_indices) // 4  # Three-quarter frame
        ]
        
        candidate_indices = [frame_indices[pos] for pos in candidate_positions 
                           if 0 <= pos < len(frame_indices)]
        
        best_transform = None
        best_face_size = 0
        best_anchor_idx = None
        
        # Try each candidate frame to find the best face
        for candidate_idx in candidate_indices:
            cap_org.set(cv2.CAP_PROP_POS_FRAMES, candidate_idx)
            ret, candidate_frame = cap_org.read()
            
            if not ret:
                logger.warning(f"Failed to read candidate frame {candidate_idx}")
                continue
            
            # Detect face and get transformation
            _, _, _, transform_matrix, face_size = extract_aligned_face_dlib(
                face_detector, face_predictor, candidate_frame
            )
            
            if transform_matrix is not None and face_size > best_face_size:
                best_transform = transform_matrix
                best_face_size = face_size
                best_anchor_idx = candidate_idx
        
        if best_transform is None:
            logger.warning(f"No face detected in any candidate frame of segment {seg_idx} "
                          f"in {movie_path.stem}")
            continue
        
        logger.debug(f"Segment {seg_idx}: Using transformation from frame {best_anchor_idx} "
                    f"(face size: {best_face_size})")
        
        # Apply the best transformation to all frames in the segment
        frames_saved = 0
        for output_idx, frame_idx in enumerate(frame_indices):
            cap_org.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap_org.read()
            
            if not ret:
                logger.warning(f"Failed to read frame {frame_idx} in {movie_path}")
                continue
            
            # Read corresponding mask if available
            frame_mask = None
            if cap_mask is not None:
                cap_mask.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret_mask, frame_mask = cap_mask.read()
                if not ret_mask:
                    frame_mask = None
            
            # Apply stabilized transformation
            cropped_face, mask_cropped = apply_transform_to_frame(
                frame, best_transform, mask=frame_mask
            )
            
            # Extract landmarks from cropped face for verification
            face_align = face_detector(cropped_face, 1)
            if len(face_align) == 0:
                logger.warning(f"No face in transformed frame {frame_idx} of {movie_path}")
                continue
            
            landmark = face_predictor(cropped_face, face_align[0])
            landmark = face_utils.shape_to_np(landmark)
            
            # Save outputs with segment structure
            # Use output_idx for sequential naming (0, 1, 2, ...)
            
            # Save cropped face
            save_path_ = save_path / 'frames' / movie_path.stem / f'segment_{seg_idx:02d}'
            save_path_.mkdir(parents=True, exist_ok=True)
            image_path = save_path_ / f"{output_idx:03d}.png"
            cv2.imwrite(str(image_path), cropped_face)

            # Save landmarks
            land_path = save_path / 'landmarks' / movie_path.stem / f'segment_{seg_idx:02d}' / f"{output_idx:03d}.npy"
            land_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(land_path), landmark)

            # Save mask
            if mask_cropped is not None:
                mask_save_path = save_path / 'masks' / movie_path.stem / f'segment_{seg_idx:02d}' / f"{output_idx:03d}.png"
                mask_save_path.parent.mkdir(parents=True, exist_ok=True)
                _, binary_mask = cv2.threshold(mask_cropped, 1, 255, cv2.THRESH_BINARY)
                cv2.imwrite(str(mask_save_path), binary_mask)
            
            frames_saved += 1
        
        logger.debug(f"Segment {seg_idx}: Saved {frames_saved}/{frames_per_segment} frames")
        segments_processed += 1

    # Release the video capture
    cap_org.release()
    if cap_mask is not None:
        cap_mask.release()
    
    logger.info(f"✓ {movie_path.stem}: Processed {segments_processed}/{num_segments} segments successfully")


def preprocess(dataset_path, mask_path, output_path, num_segments, frames_per_segment, frame_stride, logger):
    """
    Main preprocessing function with temporal extraction and stride.
    
    Args:
        dataset_path: Path to input videos
        mask_path: Path to mask videos (if available)
        output_path: Path to save preprocessed outputs
        num_segments: Number of temporal segments
        frames_per_segment: Frames per segment
        frame_stride: Stride between frames
        logger: Logger instance
    """
    # Define paths to videos in dataset
    movies_path_list = sorted([Path(p) for p in glob.glob(os.path.join(dataset_path, '**/*.mp4'), recursive=True)])
    if len(movies_path_list) == 0:
        logger.error(f"No videos found in {dataset_path}")
        return
    logger.info(f"{len(movies_path_list)} videos found in {dataset_path}")
    
    # Define paths to masks in dataset
    masks_path_list = []
    if mask_path is not None and os.path.exists(mask_path):
        masks_path_list = sorted([Path(p) for p in glob.glob(os.path.join(mask_path, '**/*.mp4'), recursive=True)])
        if len(masks_path_list) == 0:
            logger.warning(f"No masks found in {mask_path}")
        else:
            logger.info(f"{len(masks_path_list)} masks found in {mask_path}")    
    
    # Start timer
    start_time = time.monotonic()

    # Define the number of processes based on CPU capabilities
    num_processes = os.cpu_count()

    # Use multiprocessing to process videos in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for movie_path in movies_path_list:
            # Check if there is a mask for the video
            current_mask_path = None
            if mask_path is not None and len(masks_path_list) > 0:
                if movie_path.stem in [path.stem for path in masks_path_list]:
                    current_mask_path = next((path for path in masks_path_list if path.stem == movie_path.stem), None)
            
            # Create a future for each video and submit it for processing
            futures.append(
                executor.submit(
                    video_manipulate,
                    movie_path,
                    current_mask_path,
                    output_path,
                    num_segments,
                    frames_per_segment,
                    frame_stride,
                    logger
                )
            )
        
        # Wait for all futures to complete and log any errors
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(movies_path_list)):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error processing video: {e}")
                import traceback
                logger.error(traceback.format_exc())
            
        # End timer
        end_time = time.monotonic()
        duration_minutes = (end_time - start_time) / 60
        logger.info(f"Total time taken: {duration_minutes:.2f} minutes")


if __name__ == '__main__':
    # from config.yaml load parameters
    yaml_path = './config_temporal.yaml'
    # open the yaml file
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.parser.ParserError as e:
        print("YAML file parsing error:", e)
        sys.exit()
    except FileNotFoundError:
        print(f"Config file not found: {yaml_path}")
        sys.exit()

    # Get the parameters
    dataset_name = config['preprocess']['dataset_name']['default']
    dataset_root_path = config['preprocess']['dataset_root_path']['default']
    output_root_path = config['preprocess']['output_root_path']['default']
    comp = config['preprocess']['comp']['default']
    num_segments = config['preprocess']['num_segments']['default']
    frames_per_segment = config['preprocess']['frames_per_segment']['default']
    frame_stride = config['preprocess']['frame_stride']['default']
    
    # use dataset_name and dataset_root_path to get dataset_path
    dataset_path = Path(os.path.join(dataset_root_path, dataset_name))
    
    # Create output directory
    output_base = Path(output_root_path) / f"{dataset_name}_temporal_s{frame_stride}"
    output_base.mkdir(parents=True, exist_ok=True)

    # Create logger
    log_dir = Path('./logs')
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f'{dataset_name}_temporal_s{frame_stride}.log'
    logger = create_logger(str(log_path))
    
    logger.info(f"="*80)
    logger.info(f"Starting temporal preprocessing for {dataset_name}")
    logger.info(f"Input: {dataset_path}")
    logger.info(f"Output: {output_base}")
    logger.info(f"Segments: {num_segments}, Frames per segment: {frames_per_segment}, Stride: {frame_stride}")
    logger.info(f"Temporal span per segment: {(frames_per_segment - 1) * frame_stride + 1} frames")
    logger.info(f"="*80)

    # Define dataset path based on the input arguments
    ## FaceForensics++
    if dataset_name == 'FaceForensics++':
        sub_dataset_names = ["original_sequences/youtube", "original_sequences/actors",
                             "manipulated_sequences/Deepfakes",
                             "manipulated_sequences/Face2Face", "manipulated_sequences/FaceSwap",
                             "manipulated_sequences/NeuralTextures", "manipulated_sequences/FaceShifter",
                             "manipulated_sequences/DeepFakeDetection"]
        sub_dataset_paths = [Path(os.path.join(dataset_path, name, comp)) for name in sub_dataset_names]
        
        # mask
        mask_dataset_names = ["manipulated_sequences/Deepfakes", "manipulated_sequences/Face2Face",
                            "manipulated_sequences/FaceSwap", "manipulated_sequences/NeuralTextures",
                            "manipulated_sequences/DeepFakeDetection"]
        mask_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in mask_dataset_names]
    
    ## Celeb-DF-v1
    elif dataset_name == 'Celeb-DF-v1':
        sub_dataset_names = ['Celeb-real', 'Celeb-synthesis', 'YouTube-real']
        sub_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in sub_dataset_names]
        mask_dataset_paths = []
    
    ## Celeb-DF-v2
    elif dataset_name == 'Celeb-DF-v2':
        sub_dataset_names = ['Celeb-real', 'Celeb-synthesis', 'YouTube-real']
        sub_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in sub_dataset_names]
        mask_dataset_paths = []
    
    ## DFDCP
    elif dataset_name == 'DFDCP':
        sub_dataset_names = ['original_videos', 'method_A', 'method_B']
        sub_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in sub_dataset_names]
        mask_dataset_paths = []

    ## DFDC
    elif dataset_name == 'DFDC':
        # train dataset is too large, so we split it into 50 parts
        sub_train_dataset_names = ["dfdc_train_part_" + str(i) for i in range(0, 50)]
        sub_train_dataset_paths = [Path(os.path.join(dataset_path, 'train', name)) for name in sub_train_dataset_names]
        sub_dataset_paths = [Path(os.path.join(dataset_path, 'test'))] + sub_train_dataset_paths
        mask_dataset_paths = []
   
    ## DeeperForensics-1.0
    elif dataset_name == 'DeeperForensics-1.0':
        real_sub_dataset_names = ['source_videos/' + name for name in os.listdir(os.path.join(dataset_path, 'source_videos'))]
        fake_sub_dataset_names = ['manipulated_videos/' + name for name in os.listdir(os.path.join(dataset_path, 'manipulated_videos'))]
        real_sub_dataset_names.extend(fake_sub_dataset_names)
        sub_dataset_names = real_sub_dataset_names
        sub_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in sub_dataset_names]
        mask_dataset_paths = []
        
    ## UADFV
    elif dataset_name == 'UADFV':
        sub_dataset_names = ['fake', 'real']
        sub_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in sub_dataset_names]
        mask_dataset_paths = []
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized")
    
    # Check if dataset path exists
    if not Path(dataset_path).exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        sys.exit()

    if 'sub_dataset_paths' in globals() and len(sub_dataset_paths) != 0:
        # Check if sub_dataset path exists
        for sub_dataset_path in sub_dataset_paths:
            if not Path(sub_dataset_path).exists():
                logger.warning(f"Sub Dataset path does not exist: {sub_dataset_path}, skipping")
                continue
        
        # preprocess each sub_dataset
        for sub_dataset_path in sub_dataset_paths:
            if not sub_dataset_path.exists():
                continue
                
            logger.info(f"\n{'='*60}\nProcessing: {sub_dataset_path}\n{'='*60}")
            
            # Create output subdirectory matching the input structure
            relative_path = sub_dataset_path.relative_to(dataset_path)
            output_path = output_base / relative_path
            output_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Output will be saved to: {output_path}")
            
            # only part of FaceForensics++ has mask
            mask_path = None
            if dataset_name == 'FaceForensics++' and sub_dataset_path.parent in mask_dataset_paths:
                mask_dataset_path = os.path.join(sub_dataset_path.parent, "masks")
                if os.path.exists(mask_dataset_path):
                    mask_path = mask_dataset_path
                    logger.info(f"Masks found at: {mask_path}")
            
            preprocess(sub_dataset_path, mask_path, output_path, num_segments, frames_per_segment, frame_stride, logger)
    else:
        logger.error(f"Sub Dataset paths not defined")
        sys.exit()
    
    logger.info(f"\n{'='*80}")
    logger.info("✓ Temporal face cropping complete!")
    logger.info(f"Output saved to: {output_base}")
    logger.info(f"{'='*80}\n")
#!/usr/bin/env python3

import cv2
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple


class GenDPreprocessor:
    """Handles all video preprocessing including frame extraction and face alignment."""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        stride: int = 10,
        detect_face: bool = False,
        face_confidence: float = 0.4,
        scale: float = 1.3,
        face_detector=None,
        feature_extractor=None,
    ):
        self.target_size = target_size
        self.stride = stride
        self.detect_face = detect_face
        self.face_confidence = face_confidence
        self.scale = scale
        
        self.face_detector = face_detector
        self.feature_extractor = feature_extractor
        self.frames, self.frame_indices = None, None
        
        # Destination landmarks for face alignment
        self.dst_landmarks = np.array(
            [
                [0.34, 0.46],
                [0.66, 0.46],
                [0.5, 0.64],
                [0.37, 0.82],
                [0.63, 0.82],
            ],
            dtype=np.float32,
        )
    
    def align_face(self, img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Align face based on detected landmarks."""
        dst = self.dst_landmarks.copy()
        
        # Calculate inter-landmark distances
        desired = np.linalg.norm(
            landmarks[:, None, :] - landmarks[None, :, :], axis=-1
        )
        dst_d = np.linalg.norm(dst[:, None, :] - dst[None, :, :], axis=-1)
        
        upper = np.triu_indices(len(dst), k=1)
        desired = desired[upper]
        dst_d = dst_d[upper]
        
        # Determine target size
        approx = np.round(np.mean(desired / dst_d) * self.scale).astype(int)
        tgt = (approx, approx)
        
        # Scale destination landmarks
        dst[:, 0] *= tgt[0]
        dst[:, 1] *= tgt[1]
        
        # Add margins
        margin_rate = self.scale - 1
        x_margin = tgt[0] * margin_rate / 2.0
        y_margin = tgt[1] * margin_rate / 2.0
        
        dst[:, 0] += x_margin
        dst[:, 1] += y_margin
        
        dst[:, 0] *= tgt[0] / (tgt[0] + 2 * x_margin)
        dst[:, 1] *= tgt[1] / (tgt[1] + 2 * y_margin)
        
        # Estimate affine transformation
        M = cv2.estimateAffinePartial2D(
            landmarks.astype(np.float32), dst, method=cv2.LMEDS
        )[0]
        
        aligned = cv2.warpAffine(img, M, tgt, flags=cv2.INTER_LINEAR)
        return aligned
    
    def extract_frames(self, video_path: str) -> Tuple[List[np.ndarray], List[int]]:
        """Extract frames from video at specified stride intervals."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = list(np.arange(0, total_frames, self.stride, dtype=int))
        
        frames = []
        extracted_indices = []
        
        for fid in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            
            if ret:
                frames.append(frame)
                extracted_indices.append(fid)
        
        cap.release()
        return frames, extracted_indices
    
    def detect_largest_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect and return the largest face in the frame."""
        if self.face_detector is None:
            raise ValueError("Face detector not initialized")
        
        xyxy, landmarks = self.face_detector.detect(frame)
        
        if len(xyxy) == 0:
            return None
        
        # Select the largest face
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        idx = np.argmax(areas)
        lm = landmarks[idx]
        
        # Align face
        aligned_face = self.align_face(frame, lm)
        return aligned_face
    
    def process_frame(self, frame: np.ndarray) -> Optional[Image.Image]:
        """Process a single frame to extract face or resize."""
        if self.detect_face:
            face = self.detect_largest_face(frame)
            if face is None:
                return None
        else:
            face = cv2.resize(frame, self.target_size)
        
        # Convert to RGB and PIL Image
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        return Image.fromarray(face_rgb)
    
    def preprocess_video(self, video_path: str = None) -> Tuple[List, List[int]]:
        """
        Preprocess video and return list of tensors ready for model inference.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (preprocessed_tensors, frame_indices)
        """
        if self.feature_extractor is None:
            raise ValueError("Feature extractor not initialized")
        
        # Extract frames if path given
        if video_path:
            self.frames, self.frame_indices = self.extract_frames(video_path)
        
        # Process each frame
        processed_tensors = []
        processed_indices = []
        
        for frame, idx in zip(self.frames, self.frame_indices):
            processed_face = self.process_frame(frame)
            
            if processed_face is not None:
                tensor = self.feature_extractor.preprocess(processed_face)
                processed_tensors.append(tensor)
                processed_indices.append(idx)
        
        return processed_tensors, processed_indices, self.frames
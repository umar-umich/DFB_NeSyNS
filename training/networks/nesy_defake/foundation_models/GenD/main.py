#!/usr/bin/env python3

import os
import numpy as np
import argparse
import yaml  # Added for YAML support
from typing import Dict, Any, Optional
from pathlib import Path

from deeptect.video.GenD.model import GenD
from deeptect.video.GenD.retinaface import prepare_model

from deeptect.base import AbstractDetector
from base import AbstractDetector
# from deeptect.video_n.GenD.preprocess import VideoPreprocessor
from deeptect.video.GenD.preprocess import GenDPreprocessor
from huggingface_hub import login


class GenDDetector(AbstractDetector):
    """Main detector class that orchestrates preprocessing and inference."""

    def __init__(self, config_path: str):
        # We pass None to super() if it expects a dict,
        # or handle it according to your base class requirements
        super().__init__(None)

        # Load values from YAML file
        self.config = self._load_config(config_path)

        # Assign values directly in constructor
        self.device = self.config.get("device", "cuda")
        self.model_name = self.config.get("model_name", "yermandy/GenD_PE_L")
        self.thresh = float(self.config.get("thresh", 0.5))
        self.scale = float(self.config.get("scale", 1.3))
        # YAML lists convert automatically to lists; we cast to tuple for cv2/torch compatibility
        self.target_size = tuple(self.config.get("target_size", [256, 256]))
        self.detect_face = bool(self.config.get("detect_face", False))
        self.face_conf = float(self.config.get("face_conf", 0.4))
        self.stride = int(self.config.get("stride", 10))
        self.video_path = self.config.get("video", None)
        self.face_detector_path = self.config.get("face_detector", None)

        # Components
        self.detector = None
        self.face_detector = None
        self.preprocessor = None
        self.frames_idx = None

    def _load_config(self, path: str) -> Dict[str, Any]:
        """Helper to safely load YAML files."""
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def load_model(self, model_weights_dir, weights_path, s3_manager):
        """Load the GenD model and initialize all components."""

        face_detector_path = os.path.join(model_weights_dir, self.face_detector_path)

        idx = weights_path.find("model_weights")
        S3_MODEL_KEY = weights_path[idx:]            
        S3_FACEDET_KEY = face_detector_path[idx:]   

        if not os.path.exists(weights_path):
            print(f"Downloading model weights from s3 to {weights_path}...")
            s3_manager.download_directory(
                s3_prefix = S3_MODEL_KEY,
                local_dir = weights_path
            )
            print("Download complete")

        # Authenticate with Hugging Face
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            login(token=hf_token)
        # Load main detection model
        self.detector = GenD.from_pretrained(weights_path, local_files_only=True)
        self.detector.to(self.device)
        self.detector.eval()

        # Initialize face detector if needed
        if self.detect_face:
            if not os.path.exists(face_detector_path):
                print(f"Downloading face detector weights from s3 to {face_detector_path}...")
                s3_manager.download_specific_file(
                s3_key = S3_FACEDET_KEY,
                local_path = face_detector_path
                )
                print("Download complete")
            self.face_detector = prepare_model(face_detector_path, self.face_conf)

        # Initialize preprocessor
        self.preprocessor = GenDPreprocessor(
            target_size=self.target_size,
            stride=self.stride,
            detect_face=self.detect_face,
            face_confidence=self.face_conf,
            scale=self.scale,
            face_detector=self.face_detector,
            feature_extractor=self.detector.feature_extractor,
        )

        return self.detector, self.preprocessor

    def set_input(self, video_path_override: Optional[str] = None):
        """Preprocess video frames using VideoPreprocessor."""
        path = video_path_override or self.video_path

        assert path is not None, "Video path must be provided via config or argument"
        assert self.preprocessor is not None, "Model not loaded. Call load_model() first."

        self.frames_tensors, self.frames_idx = self.preprocessor.preprocess_video(path)


    def get_predictions(self):
        """Run inference on preprocessed frames."""
        # detector.load_model(model_name)

        preds = []
        for tensor in self.frames_tensors:
            tensor = tensor.unsqueeze(0).to(self.device)
            prob = self.detector(tensor).softmax(dim=-1)[0, -1].item()
            preds.append(prob)

        if not preds:
            return {"prediction": "Unknown", "confidence": 0.0}

        preds = np.asarray(preds)
        avg = float(np.mean(preds))
        label = "fake" if avg > self.thresh else "real"
        avg_conf = round(avg if label == "fake" else 1 - avg, 2)

        # Calculate max_prob_frame
        frame_labels = [1 if p > self.thresh else 0 for p in preds]
        num_label = 0 if label == "real" else 1

        if num_label == 0:
            preds_adjusted = 1 - preds
        else:
            preds_adjusted = preds.copy()

        idxs = np.where(np.array(frame_labels) == num_label)[0]

        max_prob_frame = (
            idxs[np.argmax(preds_adjusted[idxs])]
            if len(idxs)
            else int(preds.argmax())
        )

        return label, avg_conf, len(preds), max_prob_frame

        # return label, avg_conf, len(preds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./gend.yaml", help="Path to config file")
    parser.add_argument("--video", help="Optional: Override video path in config")
    args = parser.parse_args()

    # Initialize by passing the path
    detector = GenDDetector(args.config)


    # We can still override the video path from CLI if needed
    detector.set_input(video_path_override=args.video)

    pred = detector.get_predictions(model_name=detector.model_name)
    print(pred)

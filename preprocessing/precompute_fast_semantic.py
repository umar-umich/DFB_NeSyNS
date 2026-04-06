#!/usr/bin/env python3
"""
precompute_fast_semantic.py
============================
Precompute fast semantic features (58-d) for all frames using:
  - InsightFace: gender, age, head pose, face detection confidence
  - DeepFace:    emotion recognition, ethnicity entropy (CPU-only)
  - LibreFace:   17 FACS-validated AUs (DISFA intensity + BP4D detection)
  - MediaPipe:   landmarks, gaze, geometry, symmetry, 6 fallback AUs

Output: fast_semantic/{video_id}.pt → (n_frames, 58)

Usage:
  python preprocessing/precompute_fast_semantic.py \
      --detector_path training/config/detector/nesy_defake.yaml \
      --output_dir fast_semantic --skip_existing
"""

import argparse
import os
import sys
import traceback

# Force TensorFlow (loaded by DeepFace) to CPU-only BEFORE it sees GPUs.
# We pre-import TF here, restrict it to CPU, then let PyTorch take the GPU.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
except ImportError:
    pass  # DeepFace won't work either, handled gracefully below

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from config_utils import load_config, collect_videos_from_json

# Feature count must match refined_attributes.py
NUM_FAST_FEATURES = 58


def _libreface_opts(ckpt_path, download_id, data='DISFA', device='cpu'):
    """Create a config-like namespace for LibreFace solvers."""
    from types import SimpleNamespace
    return SimpleNamespace(
        ckpt_path=ckpt_path,
        weights_download_id=download_id,
        device=device,
        image_size=256,
        crop_size=224,
        model_name='resnet',
        half_precision=False,
        num_labels=12,
        data=data,
        data_root='',
        batch_size=32,
        num_workers=0,
        dropout=0.1,
        fm_distillation=False,
        hidden_dim=128,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Fast Semantic Feature Extractor
# ═══════════════════════════════════════════════════════════════════════════

class FastSemanticExtractor:
    """
    Extracts 58 semantic features per frame using specialized fast models:
      - InsightFace: gender, age, head pose, face detection confidence
      - DeepFace:    emotion recognition, ethnicity entropy
      - LibreFace:   17 FACS-validated AUs (DISFA intensity + BP4D detection)
      - MediaPipe:   landmarks, gaze, geometry, symmetry, 6 fallback AUs
    """

    def __init__(self, device: str = 'cuda:0',
                 use_mediapipe: bool = True,
                 use_deepface: bool = True):
        self.device = device
        self._insightface_app = None
        self._mediapipe_detector = None
        self._use_deepface = use_deepface
        self._libreface_det_solver = None
        self._libreface_int_solver = None

        self._init_insightface(device)
        if use_mediapipe:
            self._init_mediapipe()
        self._init_libreface(device)

    # ------------------------------------------------------------------
    # Model initialization
    # ------------------------------------------------------------------

    def _init_insightface(self, device: str):
        from insightface.app import FaceAnalysis
        if 'cuda' in device and torch.cuda.is_available():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            ctx_id = int(device.split(':')[-1]) if ':' in device else 0
        else:
            providers = ['CPUExecutionProvider']
            ctx_id = -1
        self._insightface_app = FaceAnalysis(
            name='buffalo_l', providers=providers)
        self._insightface_app.prepare(ctx_id=ctx_id, det_size=(320, 320))
        _prov = 'GPU' if ctx_id >= 0 else 'CPU'
        print(f"[FastSemantic] InsightFace buffalo_l loaded ({_prov})")

    def _init_mediapipe(self):
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision as mp_vision

            model_paths = [
                os.path.join(os.path.dirname(__file__),
                             'face_landmarker.task'),
                os.path.expanduser(
                    '~/.mediapipe/models/face_landmarker.task'),
                '/data/umar/models/face_landmarker.task',
            ]
            model_path = next(
                (p for p in model_paths if os.path.exists(p)), None)
            if model_path is None:
                print("[FastSemantic] MediaPipe model not found, "
                      "landmark features will be zero")
                return

            options = mp_vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=model_path),
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=1,
            )
            self._mediapipe_detector = (
                mp_vision.FaceLandmarker.create_from_options(options))
            print(f"[FastSemantic] MediaPipe FaceLandmarker loaded "
                  f"from {model_path}")
        except Exception as e:
            print(f"[FastSemantic] MediaPipe not available: {e}")

    def _init_libreface(self, device: str):
        """Load LibreFace AU models (ResNet18, ~45MB each, GPU if available)."""
        try:
            # LibreFace's __init__.py eagerly imports mediapipe which may
            # not be installed.  Import the solver submodules directly by
            # pre-seeding sys.modules with a stub libreface package and
            # loading the real utils first.
            import importlib.util as _ilu

            _pkg_dir = None
            _spec = _ilu.find_spec('libreface')
            if _spec and _spec.origin:
                _pkg_dir = os.path.dirname(_spec.origin)

            if _pkg_dir is None:
                raise ImportError("libreface package not found")

            # Ensure libreface.utils is available (needed by solvers)
            if 'libreface.utils' not in sys.modules:
                _uspec = _ilu.spec_from_file_location(
                    'libreface.utils',
                    os.path.join(_pkg_dir, 'utils.py'))
                _umod = _ilu.module_from_spec(_uspec)
                sys.modules['libreface.utils'] = _umod
                _uspec.loader.exec_module(_umod)

            # Helper to register a stub package with __path__
            import types

            from importlib.machinery import ModuleSpec as _ModuleSpec

            def _stub_pkg(name, path):
                if name not in sys.modules:
                    m = types.ModuleType(name)
                    m.__path__ = [path]
                    m.__spec__ = _ModuleSpec(name, None, is_package=True)
                    sys.modules[name] = m

            def _load_mod(name, filepath):
                if name not in sys.modules:
                    s = _ilu.spec_from_file_location(name, filepath)
                    m = _ilu.module_from_spec(s)
                    sys.modules[name] = m
                    s.loader.exec_module(m)

            # Ensure libreface stub package exists
            _stub_pkg('libreface', _pkg_dir)

            # AU Recognition subpackages and models
            _rec_dir = os.path.join(_pkg_dir, 'AU_Recognition')
            _rec_models_dir = os.path.join(_rec_dir, 'models')
            _stub_pkg('libreface.AU_Recognition', _rec_dir)
            _stub_pkg('libreface.AU_Recognition.models', _rec_models_dir)
            # masking_generator must load before mae (mae imports it)
            _load_mod('libreface.AU_Recognition.models.masking_generator',
                       os.path.join(_rec_models_dir, 'masking_generator.py'))
            _load_mod('libreface.AU_Recognition.models.resnet18',
                       os.path.join(_rec_models_dir, 'resnet18.py'))
            _load_mod('libreface.AU_Recognition.models.mae',
                       os.path.join(_rec_models_dir, 'mae.py'))

            # AU Detection subpackages and models
            _det_dir = os.path.join(_pkg_dir, 'AU_Detection')
            _det_models_dir = os.path.join(_det_dir, 'models')
            _stub_pkg('libreface.AU_Detection', _det_dir)
            _stub_pkg('libreface.AU_Detection.models', _det_models_dir)
            _load_mod('libreface.AU_Detection.models.resnet18',
                       os.path.join(_det_models_dir, 'resnet18.py'))

            # Now import the solvers (their deps are pre-loaded)
            _int_spec = _ilu.spec_from_file_location(
                'libreface.AU_Recognition.solver_inference_image',
                os.path.join(_pkg_dir, 'AU_Recognition',
                             'solver_inference_image.py'))
            _int_mod = _ilu.module_from_spec(_int_spec)
            sys.modules[
                'libreface.AU_Recognition.solver_inference_image'] = _int_mod
            _int_spec.loader.exec_module(_int_mod)
            solver_inference_image = _int_mod.solver_inference_image

            _det_spec = _ilu.spec_from_file_location(
                'libreface.AU_Detection.solver_inference_image',
                os.path.join(_pkg_dir, 'AU_Detection',
                             'solver_inference_image.py'))
            _det_mod = _ilu.module_from_spec(_det_spec)
            sys.modules[
                'libreface.AU_Detection.solver_inference_image'] = _det_mod
            _det_spec.loader.exec_module(_det_mod)
            solver_in_domain_image = _det_mod.solver_in_domain_image

            weights_dir = os.path.join(
                os.path.dirname(__file__), '..', 'weights_libreface')

            # Use GPU for LibreFace if available (ResNet18 is small)
            _lf_device = device if torch.cuda.is_available() else 'cpu'

            # AU intensity model (DISFA): AUs 1,2,4,5,6,9,12,15,17,20,25,26
            int_opts = _libreface_opts(
                f'{weights_dir}/AU_Recognition/weights/resnet.pt',
                '14qEnWRew2snhdMdOVyqKFJ5rq5VZrfAX',
                data='DISFA', device=_lf_device)
            self._libreface_int_solver = solver_inference_image(int_opts)
            self._libreface_int_solver.load_best_ckpt()

            # AU detection model (BP4D): AUs 1,2,4,6,7,10,12,14,15,17,23,24
            det_opts = _libreface_opts(
                f'{weights_dir}/AU_Detection/weights/resnet.pt',
                '17v_vxQ09upLG3Yh0Zlx12rpblP7uoA8x',
                data='BP4D', device=_lf_device)
            self._libreface_det_solver = solver_in_domain_image(det_opts)
            self._libreface_det_solver.load_best_ckpt()

            # Store dataset classes for batch processing
            self._AU_Recognition_Dataset = _int_mod.AU_Recognition_Dataset
            self._AU_Detection_Dataset = _det_mod.AU_Detection_Dataset

            print(f"[FastSemantic] LibreFace AU models loaded ({_lf_device})")
        except ImportError as e:
            print(f"[FastSemantic] LibreFace not installed — AU features "
                  f"will use MediaPipe blendshapes only. "
                  f"Install: pip install libreface --no-deps ({e})")
        except Exception as e:
            print(f"[FastSemantic] LibreFace init failed: {e}")
            import traceback
            traceback.print_exc()

    # LibreFace AU number → feature index mapping
    # Intensity model (DISFA, 0-5 scale → normalized to 0-1)
    _LIBREFACE_INT_AUS = {1: 11, 2: 12, 4: 13, 5: 14, 6: 15, 9: 16,
                          12: 18, 15: 20, 17: 21, 20: 23, 25: 26, 26: 27}
    # Detection model (BP4D, binary) — only for AUs not in intensity model
    _LIBREFACE_DET_ONLY_AUS = {7: 31, 10: 17, 14: 19, 23: 24, 24: 25}

    # MediaPipe blendshape fallback — only for 6 AUs LibreFace doesn't cover
    _BLENDSHAPE_FALLBACK = {
        'mouthPucker':      22,   # AU18 lip pucker
        'mouthRollLower':   28,   # AU28 lip suck
        'eyeBlinkLeft':     29,   # AU43 eyes closed (avg L+R)
        'eyeBlinkLeft_2':   30,   # AU45 blink (same as 43)
        'mouthLowerDownLeft': 32, # AU16 lower lip depressor (avg L+R)
        'mouthFunnel':      33,   # AU22 lip funneler
    }
    _BLENDSHAPE_FALLBACK_LR = {
        'eyeBlinkLeft':       'eyeBlinkRight',
        'mouthLowerDownLeft': 'mouthLowerDownRight',
    }

    # ------------------------------------------------------------------
    # Per-frame feature extraction
    # ------------------------------------------------------------------

    def extract_frame(self, image_rgb: np.ndarray,
                      image_bgr: np.ndarray) -> np.ndarray:
        """
        Extract 58-d fast semantic features from a single frame.

        Returns:
            (58,) float32 feature vector
        """
        feats = np.zeros(NUM_FAST_FEATURES, dtype=np.float32)
        H, W = image_rgb.shape[:2]

        # ── InsightFace: demographics, pose, confidence ──────────────
        face = self._extract_insightface(image_bgr)
        if face is not None:
            gender_raw = int(face.get('gender', 0))
            feats[0] = gender_raw * 2.0 - 1.0  # fs_gender_score
            feats[1] = float(face.get('age', 30)) / 100.0  # fs_age_score
            feats[2] = 0.5  # default ethnicity_entropy

            pose = face.get('pose', None)
            if pose is not None and len(pose) >= 3:
                feats[34] = float(pose[0]) / 90.0  # yaw
                feats[35] = float(pose[1]) / 90.0  # pitch
                feats[36] = float(pose[2]) / 90.0  # roll

            feats[51] = float(face.get('det_score', 0.0))

        # ── DeepFace: emotion + ethnicity (CPU-only) ────────────────
        if self._use_deepface:
            self._extract_deepface(image_rgb, feats)

        # ── MediaPipe: landmarks, gaze, geometry, symmetry, AUs ─────
        if self._mediapipe_detector is not None:
            self._extract_mediapipe(image_rgb, H, W, feats)

        # ── Image quality: blur ──────────────────────────────────────
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        feats[54] = min(laplacian_var / 1000.0, 1.0)

        return feats

    def _extract_insightface(self, image_bgr: np.ndarray):
        if self._insightface_app is None:
            return None
        try:
            faces = self._insightface_app.get(image_bgr)
            if len(faces) == 0:
                return None
            return max(faces, key=lambda f: f.get('det_score', 0.0))
        except Exception:
            return None

    def _extract_deepface(self, image_rgb: np.ndarray,
                          feats: np.ndarray):
        """Extract emotion and ethnicity using DeepFace (CPU-only)."""
        try:
            if not hasattr(self, '_deepface'):
                import warnings as _w
                with _w.catch_warnings():
                    _w.simplefilter("ignore")
                    from deepface import DeepFace
                    self._deepface = DeepFace

            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                result = self._deepface.analyze(
                    image_rgb, actions=['emotion', 'race'],
                    enforce_detection=False, silent=True,
                    detector_backend='skip')

            if result and len(result) > 0:
                r = result[0]
                emo = r.get('emotion', {})
                for emo_name, idx in [
                    ('neutral', 3), ('happy', 4), ('sad', 5),
                    ('angry', 6), ('surprise', 7), ('fear', 8),
                    ('disgust', 9),
                ]:
                    feats[idx] = float(emo.get(emo_name, 0.0)) / 100.0
                feats[10] = 0.0  # contempt (DeepFace lacks it)

                race = r.get('race', {})
                if race:
                    probs = np.array(
                        [v / 100.0 for v in race.values()],
                        dtype=np.float32)
                    probs = probs / (probs.sum() + 1e-8)
                    entropy = -np.sum(
                        probs * np.log(probs + 1e-8)) / np.log(len(probs))
                    feats[2] = float(entropy)
        except Exception:
            pass

    def _extract_mediapipe(self, image_rgb: np.ndarray,
                           H: int, W: int, feats: np.ndarray):
        """Extract landmarks, gaze, geometry, symmetry, and AUs."""
        try:
            import mediapipe as mp
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=image_rgb)
            result = self._mediapipe_detector.detect(mp_image)

            if not result.face_landmarks or len(result.face_landmarks) == 0:
                return

            lm = result.face_landmarks[0]
            pts = np.array([(p.x * W, p.y * H, p.z * W)
                            for p in lm], dtype=np.float32)

            feats[52] = 1.0  # landmark confidence

            face_bbox_area = (
                (pts[:, 0].max() - pts[:, 0].min())
                * (pts[:, 1].max() - pts[:, 1].min()))
            feats[53] = min(face_bbox_area / (H * W), 1.0)

            if len(pts) >= 478:
                left_iris = pts[468:473].mean(axis=0)[:2]
                left_outer = pts[33][:2]
                left_inner = pts[133][:2]
                left_center = (left_outer + left_inner) / 2.0
                left_width = np.linalg.norm(left_inner - left_outer) + 1e-6
                left_gaze = (left_iris - left_center) / left_width
                feats[37] = float(left_gaze[0])
                feats[38] = float(left_gaze[1])

                right_iris = pts[473:478].mean(axis=0)[:2]
                right_outer = pts[362][:2]
                right_inner = pts[263][:2]
                right_center = (right_outer + right_inner) / 2.0
                right_width = np.linalg.norm(
                    right_inner - right_outer) + 1e-6
                right_gaze = (right_iris - right_center) / right_width
                feats[39] = float(right_gaze[0])
                feats[40] = float(right_gaze[1])

                left_ear = self._eye_aspect_ratio(
                    pts, top=159, bottom=145, inner=133, outer=33)
                feats[41] = left_ear
                right_ear = self._eye_aspect_ratio(
                    pts, top=386, bottom=374, inner=263, outer=362)
                feats[42] = right_ear

            mouth_h = np.linalg.norm(pts[13][:2] - pts[14][:2])
            mouth_w = np.linalg.norm(pts[61][:2] - pts[291][:2]) + 1e-6
            feats[43] = float(mouth_h / mouth_w)

            face_w = pts[:, 0].max() - pts[:, 0].min() + 1e-6
            face_h = pts[:, 1].max() - pts[:, 1].min() + 1e-6

            if len(pts) >= 478:
                ipd = np.linalg.norm(left_iris - right_iris)
                feats[44] = float(ipd / face_w)
            else:
                feats[44] = 0.3

            nose_w = np.linalg.norm(pts[102][:2] - pts[331][:2])
            feats[45] = float(nose_w / face_w)
            jaw_w = np.linalg.norm(pts[234][:2] - pts[454][:2])
            feats[46] = float(jaw_w / face_w)
            feats[47] = float(face_w / face_h)

            chin = pts[152][:2]
            jaw_l = pts[234][:2]
            jaw_r = pts[454][:2]
            v1 = jaw_l - chin
            v2 = jaw_r - chin
            cos_angle = np.dot(v1, v2) / (
                np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            feats[48] = float(np.arccos(cos_angle) / np.pi)

            brow_h = abs(pts[70][1] - pts[159][1])
            feats[49] = float(brow_h / face_h)
            mouth_nose_d = abs(pts[1][1] - pts[13][1])
            feats[50] = float(mouth_nose_d / face_h)

            if len(pts) >= 478:
                feats[55] = abs(left_ear - right_ear) / (
                    (left_ear + right_ear) / 2.0 + 1e-6)
            mouth_l_y = pts[61][1]
            mouth_r_y = pts[291][1]
            feats[56] = abs(mouth_l_y - mouth_r_y) / (face_h + 1e-6)
            jaw_l_d = np.linalg.norm(pts[234][:2] - chin)
            jaw_r_d = np.linalg.norm(pts[454][:2] - chin)
            feats[57] = abs(jaw_l_d - jaw_r_d) / (
                (jaw_l_d + jaw_r_d) / 2.0 + 1e-6)

            # ── Blendshape fallback AUs (6 AUs not covered by LibreFace) ──
            if (result.face_blendshapes
                    and len(result.face_blendshapes) > 0):
                bs = result.face_blendshapes[0]
                bs_dict = {b.category_name: b.score for b in bs}
                for bs_name, au_idx in self._BLENDSHAPE_FALLBACK.items():
                    real_name = bs_name.rstrip('_2')
                    val = bs_dict.get(real_name, 0.0)
                    if real_name in self._BLENDSHAPE_FALLBACK_LR:
                        right_name = self._BLENDSHAPE_FALLBACK_LR[real_name]
                        val = (val + bs_dict.get(right_name, 0.0)) / 2.0
                    feats[au_idx] = float(val)

        except Exception:
            pass

    @staticmethod
    def _eye_aspect_ratio(pts, top, bottom, inner, outer):
        vert = np.linalg.norm(pts[top][:2] - pts[bottom][:2])
        horiz = np.linalg.norm(pts[inner][:2] - pts[outer][:2]) + 1e-6
        return float(vert / horiz)

    # ------------------------------------------------------------------
    # LibreFace batch AU extraction
    # ------------------------------------------------------------------

    # ImageNet normalization constants
    _IMGNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _IMGNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _extract_libreface_batch(self, images_rgb: list,
                                 all_features: list):
        """
        Run LibreFace AU models on pre-loaded frames (GPU batched).
        Uses OpenCV/numpy for transforms (no PIL overhead).
        Intensity AUs normalized from 0-5 → 0-1.
        """
        if self._libreface_int_solver is None:
            return

        try:
            # Prepare batch with OpenCV (much faster than PIL)
            valid_idx = []
            batch_np = []
            for i, img in enumerate(images_rgb):
                if img is None:
                    continue
                # Resize shortest side to 256, center crop 224
                h, w = img.shape[:2]
                scale = 256.0 / min(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                resized = cv2.resize(img, (new_w, new_h),
                                     interpolation=cv2.INTER_LINEAR)
                # Center crop 224x224
                y0 = (new_h - 224) // 2
                x0 = (new_w - 224) // 2
                crop = resized[y0:y0+224, x0:x0+224]
                # Normalize: to float [0,1], then ImageNet norm
                crop_f = crop.astype(np.float32) / 255.0
                crop_f = (crop_f - self._IMGNET_MEAN) / self._IMGNET_STD
                # HWC → CHW
                batch_np.append(crop_f.transpose(2, 0, 1))
                valid_idx.append(i)

            if not batch_np:
                return

            batch = torch.from_numpy(np.stack(batch_np))  # (N, 3, 224, 224)
            device = self._libreface_int_solver.device

            # --- AU Intensity (DISFA): 12 AUs, 0-5 scale ---
            with torch.no_grad():
                self._libreface_int_solver.eval()
                int_preds = self._libreface_int_solver.model(
                    batch.to(device))
                int_preds = torch.clamp(int_preds * 5.0, min=0.0, max=5.0)
                int_preds = int_preds.cpu()

            for k, i in enumerate(valid_idx):
                for j, au_num in enumerate(
                        self._libreface_int_solver.aus):
                    if au_num in self._LIBREFACE_INT_AUS:
                        feat_idx = self._LIBREFACE_INT_AUS[au_num]
                        all_features[i][feat_idx] = float(
                            int_preds[k][j].item()) / 5.0

            # --- AU Detection (BP4D): 12 AUs, binary ---
            if self._libreface_det_solver is not None:
                with torch.no_grad():
                    self._libreface_det_solver.eval()
                    det_preds = self._libreface_det_solver.model(
                        batch.to(device))
                    det_preds = (det_preds >= 0.5).int().cpu()

                for k, i in enumerate(valid_idx):
                    for j, au_num in enumerate(
                            self._libreface_det_solver.aus):
                        if au_num in self._LIBREFACE_DET_ONLY_AUS:
                            feat_idx = self._LIBREFACE_DET_ONLY_AUS[au_num]
                            all_features[i][feat_idx] = float(
                                det_preds[k][j].item())

        except Exception as e:
            print(f"  [WARNING] LibreFace batch failed: {e}")
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Video processing
    # ------------------------------------------------------------------

    def _extract_cpu_features(self, image_rgb: np.ndarray,
                              H: int, W: int,
                              feats: np.ndarray):
        """CPU-only features: DeepFace + MediaPipe (thread-safe)."""
        if self._use_deepface:
            self._extract_deepface(image_rgb, feats)
        if self._mediapipe_detector is not None:
            self._extract_mediapipe(image_rgb, H, W, feats)
        gray = cv2.cvtColor(
            cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        feats[54] = min(laplacian_var / 1000.0, 1.0)

    def process_video(self, frame_paths: list,
                      num_workers: int = 4) -> dict:
        """
        Process all frames of a video and return fast semantic features.

        Pipeline (GPU + CPU run in parallel):
          - Thread 0: InsightFace GPU → sequential per frame (~47ms/frame)
          - Threads 1-N: DeepFace CPU + MediaPipe CPU → parallel (~220ms/frame)
          - After both: LibreFace GPU → batched (~2ms/frame)

        Args:
            frame_paths: List of image file paths.
            num_workers: Threads for CPU-bound DeepFace/MediaPipe extraction.
        """
        n = len(frame_paths)
        all_features = [np.zeros(NUM_FAST_FEATURES, dtype=np.float32)
                        for _ in range(n)]
        loaded = [None] * n

        # ── Step 1: Load all frames ─────────────────────────────────────
        for i, fp in enumerate(frame_paths):
            try:
                bgr = cv2.imread(fp)
                if bgr is not None:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    loaded[i] = (rgb, bgr)
            except Exception:
                pass

        # ── Step 2: GPU + CPU in parallel ───────────────────────────────
        # InsightFace GPU runs in one thread while DeepFace/MediaPipe
        # CPU work runs in a thread pool — different hardware, true overlap.
        from concurrent.futures import ThreadPoolExecutor

        def _gpu_work():
            """InsightFace on GPU — must be single-threaded."""
            for i in range(n):
                if loaded[i] is None:
                    continue
                _, bgr = loaded[i]
                face = self._extract_insightface(bgr)
                if face is not None:
                    feats = all_features[i]
                    gender_raw = int(face.get('gender', 0))
                    feats[0] = gender_raw * 2.0 - 1.0
                    feats[1] = float(face.get('age', 30)) / 100.0
                    feats[2] = 0.5
                    pose = face.get('pose', None)
                    if pose is not None and len(pose) >= 3:
                        feats[34] = float(pose[0]) / 90.0
                        feats[35] = float(pose[1]) / 90.0
                        feats[36] = float(pose[2]) / 90.0
                    feats[51] = float(face.get('det_score', 0.0))

        def _cpu_work(i):
            """DeepFace + MediaPipe on CPU — safe to parallelize."""
            if loaded[i] is None:
                return
            rgb, _ = loaded[i]
            H, W = rgb.shape[:2]
            self._extract_cpu_features(rgb, H, W, all_features[i])

        with ThreadPoolExecutor(max_workers=num_workers + 1) as pool:
            # Submit GPU work as one task
            gpu_future = pool.submit(_gpu_work)
            # Submit CPU work as N parallel tasks
            cpu_futures = [pool.submit(_cpu_work, i) for i in range(n)]
            # Wait for all
            gpu_future.result()
            for f in cpu_futures:
                f.result()

        # ── Step 3: LibreFace AUs on GPU — batched ──────────────────────
        # Pass pre-loaded RGB images (avoids re-reading from disk)
        images_rgb = [loaded[i][0] if loaded[i] is not None else None
                      for i in range(n)]
        self._extract_libreface_batch(images_rgb, all_features)

        valid_paths = [os.path.basename(p) for p in frame_paths]
        features_tensor = torch.from_numpy(np.stack(all_features))

        sys.path.insert(
            0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from networks.nesy_defake.semantic.refined_attributes import (
            FAST_FEATURE_NAMES)

        return {
            'features': features_tensor,
            'frame_paths': valid_paths,
            'feature_names': list(FAST_FEATURE_NAMES),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description='Precompute fast semantic features (58-d)')
    parser.add_argument('--detector_path', type=str, required=True,
                        help='Path to detector YAML config')
    parser.add_argument('--output_dir', type=str, default='fast_semantic',
                        help='Subdirectory name for output .pt files')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip videos that already have output')
    parser.add_argument('--compression', type=str, default=None,
                        help='Override compression level (e.g., c23)')
    parser.add_argument('--no_mediapipe', action='store_true',
                        help='Disable MediaPipe (landmarks will be zero)')
    parser.add_argument('--no_deepface', action='store_true', default=True,
                        help='Disable DeepFace (emotions/ethnicity zero). '
                             'Default: disabled (adds ~20h for 8 features '
                             'redundant with FaceBench VLM)')
    parser.add_argument('--use_deepface', action='store_true',
                        help='Enable DeepFace (slow, ~207ms/frame CPU)')
    parser.add_argument('--max_frames', type=int, default=0,
                        help='Max frames per video (0=all)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Threads for CPU extraction (DeepFace/MediaPipe). '
                             'Default 1 is optimal without DeepFace. '
                             'Use 4-8 if --use_deepface is enabled.')
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.detector_path)
    if args.compression:
        config['compression'] = args.compression

    print("=" * 60)
    print("Fast Semantic Feature Precomputation (58-d)")
    print("  InsightFace: gender, age, pose, detection confidence")
    print("  DeepFace:    emotions, ethnicity entropy (CPU)")
    print("  LibreFace:   17 FACS-validated AUs (DISFA + BP4D)")
    print("  MediaPipe:   landmarks, gaze, geometry, 6 fallback AUs")
    print("=" * 60)

    use_deepface = args.use_deepface and not args.no_deepface
    extractor = FastSemanticExtractor(
        device=args.device,
        use_mediapipe=not args.no_mediapipe,
        use_deepface=use_deepface,
    )

    videos = collect_videos_from_json(config)
    if not videos:
        print("No videos found. Check your config and dataset JSONs.")
        return

    print(f"\n  Videos to process: {len(videos)}")
    print(f"  Output subdir: {args.output_dir}")
    print(f"  Workers: {args.workers} threads (CPU extractors)")

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
            result = extractor.process_video(
                    frame_paths, num_workers=args.workers)
            if result is not None:
                os.makedirs(output_dir, exist_ok=True)
                torch.save(result, output_path)
                n_processed += 1
            else:
                n_failed += 1
        except Exception as e:
            print(f"  Error processing {vid_key}: {e}")
            traceback.print_exc()
            n_failed += 1

    print(f"\n{'=' * 60}")
    print(f"Fast semantic precomputation complete!")
    print(f"  Processed: {n_processed}")
    print(f"  Skipped:   {n_skipped}")
    print(f"  Failed:    {n_failed}")
    print(f"  Output:    */{args.output_dir}/<video_id>.pt")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()

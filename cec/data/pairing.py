"""FF++ fake -> paired real, with the pilot's re-alignment.

The paired real is the counterfactual "what these pixels would look like if the
face had not been manipulated". The subtlety, and the reason this is ported
rather than reinvented: the pre-aligned youtube crops on disk are aligned
INDEPENDENTLY of the fake crops and are not pixel-registered to them
(background MSE outside the mask ~800). Re-aligning the RAW youtube frame with
the FAKE crop's landmarks brings that to ~9 (compression noise only), so fake
and real share pixel geometry and a region swap changes only the manipulated
content, not the framing.

Ported from scripts/intervention_pilot.py (build_sample, read_raw_frame) and
scripts/pilot1_rev3.py (build_sample_rev3). `align_face` is imported from
preprocessing/preprocess.py; the QC constants come from cec/registration.

Not imported from intervention_pilot directly on purpose: that module imports
pilot_detectors, which does a module-level os.chdir into the GenD repo. This
module stays chdir-free.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from cec.registration import load_params

# align_face lives in the preprocessing package, not on the default path.
_PREPROC = Path(__file__).resolve().parents[2] / "preprocessing"
if str(_PREPROC) not in sys.path:
    sys.path.append(str(_PREPROC))
from preprocess import align_face  # noqa: E402

_params = load_params()
_data = _params.data["ffpp"]
_qc = _params.qc

PP = Path(_data["root"])
RAW = Path(_data["raw_root"])
COMPRESSION = _data["compression"]

QC_MSE = _qc["max_bg_mse"]
FG_MIN, FG_MAX = _qc["fg_min"], _qc["fg_max"]


@dataclass
class PairSample:
    """One fake frame and its pixel-registered paired real."""

    method: str
    vid: str
    frame: str
    fake: np.ndarray            # BGR uint8, the manipulated crop as scored by the pilot
    real: np.ndarray            # BGR uint8, raw youtube frame re-aligned to the fake's landmarks
    mask: np.ndarray            # bool HxW, the GT manipulated region
    landmarks: np.ndarray       # (5, 2) float32, raw-frame coords
    fg: float                   # manipulated-region area fraction
    bg_mse: float               # MSE outside the mask; the registration QC number

    @property
    def source_id(self) -> str:
        return self.vid.split("_")[0]

    @property
    def image_path(self) -> Path:
        """Path to the image under test (proposer inference is path-based, for caching).

        Fakes -> the manipulated frame PNG. Reals (method 'youtube-real') -> the
        youtube original frame, since there is no manipulated_sequences path.
        """
        if self.method == "youtube-real":
            return PP / "original_sequences" / "youtube" / COMPRESSION / "frames" / \
                self.source_id / f"{self.frame}.png"
        return fake_paths(self.method, self.vid, self.frame)[0]

    # Back-compat alias (older callers used fake_path).
    @property
    def fake_path(self) -> Path:
        return self.image_path


def read_raw_frame(video_path, idx) -> Optional[np.ndarray]:
    """Read one frame from a video by index (ported verbatim from the pilot)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def fake_dir(method: str) -> Path:
    return PP / "manipulated_sequences" / method / COMPRESSION


def fake_paths(method: str, vid: str, frame: str):
    """(frame_png, mask_png, landmarks_npy) for a manipulated frame."""
    base = fake_dir(method)
    return (
        base / "frames" / vid / f"{frame}.png",
        base / "masks" / vid / f"{frame}.png",
        base / "landmarks" / vid / f"{frame}.npy",
    )


def source_video(vid: str) -> Path:
    """The raw youtube source video for a fake's target identity."""
    return RAW / "original_sequences" / "youtube" / COMPRESSION / "videos" / \
        f"{vid.split('_')[0]}.mp4"


def resolve_image_path(image_label: str) -> Path:
    """Record 'image' label ('method/vid/frame') -> the actual frame PNG.

    Reals are labeled 'youtube-real/<sid>_<sid>/<frame>' -> the youtube original;
    everything else is a manipulated frame. Used by the multimodal DPO builder to
    attach each preference pair's image.
    """
    method, vid, frame = image_label.split("/")
    if method == "youtube-real":
        sid = vid.split("_")[0]
        return PP / "original_sequences" / "youtube" / COMPRESSION / "frames" / sid / f"{frame}.png"
    return fake_paths(method, vid, frame)[0]


def split_video_ids(split: str) -> set:
    """FF++ `<a>_<b>` video ids belonging to a split ('train'|'test'|'val').

    Splits are stored as lists of source pairs [a, b]; a fake video `a_b` (or
    `b_a`) belongs to the split iff [a, b] is listed. Used to keep region-gate
    calibration (train) disjoint from evaluation (test), per the codebook rule.
    """
    import json
    pairs = json.loads((PP / f"{split}.json").read_text())
    vids = set()
    for a, b in pairs:
        vids.add(f"{a}_{b}")
        vids.add(f"{b}_{a}")
    return vids


def align_paired_real(landmarks, vid: str, frame, offset: int = 0) -> Optional[np.ndarray]:
    """Re-align the raw youtube frame to the fake's landmarks. offset shifts the
    source frame (used by the real-repair control, which reads a +/-1 frame)."""
    raw = read_raw_frame(source_video(vid), int(frame) + offset)
    if raw is None:
        return None
    real, _ = align_face(raw, landmarks)
    return real


def verify_pair(fake, real, mask) -> tuple[bool, float]:
    """Is the paired real pixel-registered to the fake? Returns (ok, bg_mse).

    bg_mse is the MSE OUTSIDE the mask — where fake and real should be identical
    up to compression. Above QC_MSE the crops are not registered and the sample
    must be dropped, not repaired.
    """
    outside = ~mask
    bg_mse = float(((fake.astype(float) - real.astype(float))[outside] ** 2).mean())
    return bg_mse <= QC_MSE, bg_mse


def real_landmarks_path(source_id: str, frame: str) -> Path:
    return PP / "original_sequences" / "youtube" / COMPRESSION / "landmarks" / source_id / f"{frame}.npy"


def build_real_pair(source_id: str, frame: str) -> Optional["PairSample"]:
    """A PairSample for a REAL youtube image, for the real-image audit path (T15).

    The image under test is the real crop; its paired "real" is an ADJACENT frame
    re-aligned to the same landmarks (the real-offset construction). There is no
    manipulation, so repairing any region should barely move p_fake (NM ~= 0) —
    the reals anchor the null and let FP-claim rate be measured. `mask` is the
    inner-face composite region (no GT mask exists for a real).
    """
    from cec.masks.regions import composite_mask

    lpath = real_landmarks_path(source_id, frame)
    if not lpath.exists():
        return None
    landmarks = np.load(lpath)
    # image under test: raw youtube frame re-aligned with its own landmarks.
    fake = align_paired_real(landmarks, f"{source_id}_{source_id}", frame, offset=0)
    if fake is None:
        return None
    real = align_paired_real(landmarks, f"{source_id}_{source_id}", frame, offset=1)
    if real is None:
        real = align_paired_real(landmarks, f"{source_id}_{source_id}", frame, offset=-1)
    if real is None:
        return None
    mask = composite_mask(landmarks)
    if mask is None:
        return None
    return PairSample(
        method="youtube-real", vid=f"{source_id}_{source_id}", frame=frame,
        fake=fake, real=real, mask=mask, landmarks=landmarks, fg=float(mask.mean()), bg_mse=0.0,
    )


def build_pair(method: str, vid: str, frame: str) -> Optional[PairSample]:
    """Load a fake frame and its verified paired real, or None if unusable.

    Returns None on: missing files, an implausible manipulated-region fraction
    (outside [FG_MIN, FG_MAX]), a failed alignment, or a background MSE above
    QC_MSE (crops not registered). These are the pilot's QC gates, unchanged.
    """
    fpath, mpath, lpath = fake_paths(method, vid, frame)
    fake = cv2.imread(str(fpath))
    mask_raw = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
    if fake is None or mask_raw is None or not lpath.exists():
        return None

    mask = mask_raw > 127
    fg = float(mask.mean())
    if not (FG_MIN <= fg <= FG_MAX):
        return None

    landmarks = np.load(lpath)
    real = align_paired_real(landmarks, vid, frame)
    if real is None:
        return None

    ok, bg_mse = verify_pair(fake, real, mask)
    if not ok:
        return None

    return PairSample(
        method=method, vid=vid, frame=frame,
        fake=fake, real=real, mask=mask, landmarks=landmarks,
        fg=fg, bg_mse=bg_mse,
    )

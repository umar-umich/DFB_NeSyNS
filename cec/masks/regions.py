"""Location vocabulary -> binary region masks, from the 5-point landmarks.

The codebook (docs/pilots/CODEBOOK.md section 2) names 14 regions. The landmarks
on disk are 5-point (left_eye, right_eye, nose, left_mouth, right_mouth) in
raw-frame coordinates. This module builds a deterministic mask for each codebook
region in the ALIGNED 256 crop frame, anchored to those five points.

Deterministic is the point: the codebook forbids a grounding model on the
critical path ("Predicates are bound to deterministic facial landmark regions.
No free text, no grounding model"). Five points cannot resolve every region
anatomically — forehead, hairline, and the cheek/jaw split are geometric
extrapolations, not landmarked boundaries. Implementation v2 Task 3 anticipates
this: "BiSeNet upgrade only if Pilot 1L shows landmark regions too coarse." The
per-region `fidelity` tag below records which are anchored vs extrapolated so
Pilot 1L can make that call with evidence.

whole_face is NOT a spatial region: the codebook verifies it by spectral
intervention only. `region_mask('whole_face', ...)` returns None; callers route
it to the spectral instrument (see cec.registration.vocab.routes_to_spectral).
"""
from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np

from cec.registration.vocab import SPATIAL_REGIONS, WHOLE_FACE

# align_face's template + defaults, mirrored here because align_face returns the
# warped image but not the transform, and region geometry needs the transform.
# Kept byte-identical to preprocessing/preprocess.py:align_face; verified in the
# Task 3 smoke test by warping with this matrix and diffing against align_face.
_TEMPLATE = np.array([
    [0.34, 0.46],   # left_eye
    [0.66, 0.46],   # right_eye
    [0.50, 0.64],   # nose
    [0.37, 0.82],   # left_mouth
    [0.63, 0.82],   # right_mouth
], dtype=np.float32)
TARGET_SIZE = (256, 256)
SCALE = 1.3

# How trustworthy each region is, given only 5 landmarks. Recorded, not enforced.
FIDELITY: Dict[str, str] = {
    "left_eye": "anchored", "right_eye": "anchored", "inter_ocular": "anchored",
    "nose": "anchored", "mouth": "anchored", "nasolabial": "anchored",
    "left_cheek": "extrapolated", "right_cheek": "extrapolated",
    "jawline": "extrapolated", "chin": "extrapolated",
    "forehead": "extrapolated", "hairline": "extrapolated",
    "face_boundary": "extrapolated",
}


def align_matrix(landmarks, target_size=TARGET_SIZE, scale=SCALE):
    """The partial-affine matrix mapping raw landmarks -> aligned crop.

    Reproduces preprocessing/preprocess.py:align_face exactly. Returns a 2x3
    matrix, or None if the fit fails (same failure mode as align_face).
    """
    dst = _TEMPLATE.copy()
    dst[:, 0] *= target_size[0]
    dst[:, 1] *= target_size[1]

    margin_rate = scale - 1
    x_margin = target_size[0] * margin_rate / 2.0
    y_margin = target_size[1] * margin_rate / 2.0
    dst[:, 0] += x_margin
    dst[:, 1] += y_margin
    dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
    dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

    M = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), dst, method=cv2.LMEDS)[0]
    return M


def aligned_landmarks(landmarks, target_size=TARGET_SIZE, scale=SCALE):
    """The 5 landmarks in aligned-crop coordinates. Shape (5, 2), or None."""
    M = align_matrix(landmarks, target_size, scale)
    if M is None:
        return None
    pts = np.hstack([landmarks.astype(np.float32), np.ones((len(landmarks), 1), np.float32)])
    return (pts @ M.T)[:, :2]


class FaceGeometry:
    """Anchors and scale for one aligned face, from which regions are built.

    All regions are drawn relative to these five points and the inter-ocular
    distance, so they scale with the face rather than assuming the canonical
    template positions (alignment is a 4-DOF similarity fit, so real landmarks
    sit near but not exactly on the template).
    """

    def __init__(self, landmarks, target_size=TARGET_SIZE, scale=SCALE):
        pts = aligned_landmarks(landmarks, target_size, scale)
        if pts is None:
            raise ValueError("alignment failed; cannot build regions for this frame")
        self.w, self.h = target_size
        # p_* are the five landmark POINTS; the region methods (nose(), mouth(),
        # ...) share those names, so the points must not shadow the methods.
        self.p_le, self.p_re, self.p_nose, self.p_lm, self.p_rm = pts
        self.eye_c = (self.p_le + self.p_re) / 2.0
        self.mouth_c = (self.p_lm + self.p_rm) / 2.0
        self.iod = float(np.linalg.norm(self.p_re - self.p_le))   # inter-ocular distance
        self.eye_mouth = float(np.linalg.norm(self.mouth_c - self.eye_c))
        self.face_c = self.eye_c + 0.55 * (self.mouth_c - self.eye_c)

    # -- primitives ----------------------------------------------------------
    def _blank(self):
        return np.zeros((self.h, self.w), dtype=np.uint8)

    def _ellipse(self, center, ax, ay, angle=0.0):
        m = self._blank()
        cv2.ellipse(m, (int(round(center[0])), int(round(center[1]))),
                    (max(1, int(round(ax))), max(1, int(round(ay)))),
                    angle, 0, 360, 255, -1)
        return m.astype(bool)

    def _face_ellipse(self) -> np.ndarray:
        """The whole-face support: everything else is intersected with this."""
        return self._ellipse(self.face_c, 1.15 * self.iod, 0.95 * self.eye_mouth + 0.6 * self.iod)

    # -- codebook regions ----------------------------------------------------
    def left_eye(self):
        return self._ellipse(self.p_le, 0.42 * self.iod, 0.30 * self.iod)

    def right_eye(self):
        return self._ellipse(self.p_re, 0.42 * self.iod, 0.30 * self.iod)

    def inter_ocular(self):
        c = self.eye_c - np.array([0.0, 0.05 * self.iod])
        m = self._ellipse(c, 0.40 * self.iod, 0.45 * self.iod)
        return m & ~self.left_eye() & ~self.right_eye()

    def nose(self):
        c = self.p_nose + np.array([0.0, -0.05 * self.iod])
        return self._ellipse(c, 0.34 * self.iod, 0.50 * self.iod)

    def mouth(self):
        return self._ellipse(self.mouth_c, 0.62 * self.iod, 0.32 * self.iod)

    def nasolabial(self):
        """Folds from nose to mouth corners: two bands, minus nose and mouth."""
        m = self._blank()
        for corner in (self.p_lm, self.p_rm):
            mid = (self.p_nose + corner) / 2.0
            cv2.circle(m, (int(mid[0]), int(mid[1])), int(0.30 * self.iod), 255, -1)
        band = m.astype(bool) & self._face_ellipse()
        return band & ~self.nose() & ~self.mouth()

    def _lower_face(self):
        """Below the mouth line, inside the face ellipse: jaw + chin support."""
        m = self._blank()
        y0 = int(self.mouth_c[1])
        m[y0:, :] = 255
        return m.astype(bool) & self._face_ellipse()

    def chin(self):
        c = self.mouth_c + np.array([0.0, 0.70 * self.iod])
        return self._ellipse(c, 0.45 * self.iod, 0.40 * self.iod) & self._face_ellipse()

    def jawline(self):
        """Lower-face band excluding the central chin — the jaw/neck seam."""
        return self._lower_face() & ~self.chin() & ~self.mouth()

    def left_cheek(self):
        c = (self.p_le + self.p_lm) / 2.0 + np.array([-0.25 * self.iod, 0.0])
        return self._ellipse(c, 0.40 * self.iod, 0.55 * self.iod) & self._face_ellipse() \
            & ~self.nose() & ~self.nasolabial()

    def right_cheek(self):
        c = (self.p_re + self.p_rm) / 2.0 + np.array([0.25 * self.iod, 0.0])
        return self._ellipse(c, 0.40 * self.iod, 0.55 * self.iod) & self._face_ellipse() \
            & ~self.nose() & ~self.nasolabial()

    def forehead(self):
        c = self.eye_c - np.array([0.0, 0.75 * self.iod])
        return self._ellipse(c, 0.95 * self.iod, 0.55 * self.iod) & self._face_ellipse()

    def hairline(self):
        c = self.eye_c - np.array([0.0, 1.15 * self.iod])
        m = self._ellipse(c, 1.05 * self.iod, 0.40 * self.iod)
        return m & self._face_ellipse()

    def face_boundary(self):
        """The outer blend seam: a ring just inside the face-ellipse perimeter.

        Codebook: interventions here are masked strictly to the face side of the
        seam, so background pixels are never altered. This ring is entirely
        inside the face ellipse, satisfying that constraint by construction.
        """
        face = self._face_ellipse()
        eroded = cv2.erode(face.astype(np.uint8), np.ones((3, 3), np.uint8),
                           iterations=max(2, int(0.18 * self.iod))).astype(bool)
        return face & ~eroded


# name -> method on FaceGeometry
_BUILDERS = {r: r for r in SPATIAL_REGIONS}


def region_mask(region_id: str, landmarks, target_size=TARGET_SIZE,
                scale=SCALE) -> Optional[np.ndarray]:
    """Binary mask (bool HxW) for a codebook region, or None if not spatial.

    whole_face -> None (not a landmark region; use composite_mask for the
    whole-face COMPOSITE scope). An unknown region_id raises KeyError, so a
    validator bug surfaces rather than silently producing an empty mask.
    """
    if region_id == WHOLE_FACE:
        return None
    if region_id not in _BUILDERS:
        raise KeyError(
            f"'{region_id}' is not a codebook region. "
            f"Spatial: {sorted(SPATIAL_REGIONS)}; plus whole_face (composite)."
        )
    geom = FaceGeometry(landmarks, target_size, scale)
    return getattr(geom, _BUILDERS[region_id])()


def composite_mask(landmarks, target_size=TARGET_SIZE, scale=SCALE) -> Optional[np.ndarray]:
    """The inner-face union mask, for the whole-face COMPOSITE scope (T13).

    This is the full inner-face region derivable from landmarks (the face
    ellipse), NOT a small landmark region. Repairing it is the SAME operation the
    frozen `gate:` block was calibrated on (Pilot rev3 full-mask repair), so
    composite claims are judged against `gate:` without lowering any bar.
    Returns None if alignment fails.
    """
    try:
        geom = FaceGeometry(landmarks, target_size, scale)
    except ValueError:
        return None
    return geom._face_ellipse()

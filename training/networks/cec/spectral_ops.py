"""
networks/cec/spectral_ops.py
============================
Study-1B spectral interventions, for the spectral certification arm.

Frequency and noise claims bind to whole_face and cannot be verified by spatial
repair — a global generator fingerprint survives local inpainting. They get
predicate-targeted frequency-domain interventions instead: perturb the cited
spectral structure, re-query the frequency detector, read the p(fake) drop.

  notch_hf           zero a high-frequency annulus in the 2D FFT (the band a
                     frequency_anomaly would cite), then invert.
  checkerboard_supp  attenuate strong isolated peaks (upsampling harmonics)
                     outside the low-frequency disk to the local median.
  residual_renorm    shrink the SRM-style high-frequency residual energy toward
                     a reference crop's residual energy (tests noise_inconsistency).

Ported verbatim from scripts/pilot1_spectral.py. Model-free (numpy/cv2 only), so
this lives beside the detector wiring but pulls in no weights.
"""
from __future__ import annotations

import cv2
import numpy as np

# Default intervention parameters (pilot1_spectral.py signatures).
NOTCH_R_LO, NOTCH_R_HI = 0.55, 0.85
CHECKER_R_MIN, CHECKER_PCT = 0.35, 99.5


def _fft_channels(img):
    """Per-channel shifted FFT of a float BGR image."""
    return [np.fft.fftshift(np.fft.fft2(img[:, :, c])) for c in range(3)]


def _ifft_channels(fchs, shape):
    out = np.zeros(shape, np.float32)
    for c, F in enumerate(fchs):
        out[:, :, c] = np.real(np.fft.ifft2(np.fft.ifftshift(F)))
    return np.clip(out, 0, 255).astype(np.uint8)


def _radius_grid(h, w):
    cy, cx = h / 2.0, w / 2.0
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    return r / r.max()  # normalized 0..1


def notch_hf(img_bgr, r_lo=NOTCH_R_LO, r_hi=NOTCH_R_HI):
    """Zero a high-frequency annulus r_lo..r_hi (normalized radius)."""
    f = img_bgr.astype(np.float32)
    h, w = f.shape[:2]
    r = _radius_grid(h, w)
    band = (r >= r_lo) & (r <= r_hi)
    fchs = _fft_channels(f)
    for F in fchs:
        F[band] = 0.0
    return _ifft_channels(fchs, f.shape)


def checkerboard_supp(img_bgr, r_min=CHECKER_R_MIN, pct=CHECKER_PCT):
    """Attenuate strong isolated peaks (upsampling harmonics) outside the low-freq disk."""
    f = img_bgr.astype(np.float32)
    h, w = f.shape[:2]
    r = _radius_grid(h, w)
    outer = r >= r_min
    fchs = _fft_channels(f)
    for F in fchs:
        mag = np.abs(F)
        thr = np.percentile(mag[outer], pct)
        peaks = outer & (mag > thr)
        if peaks.sum():
            med = np.median(mag[outer])
            scale = np.where(peaks, med / (mag + 1e-6), 1.0)
            F *= scale
    return _ifft_channels(fchs, f.shape)


def residual_renorm(img_bgr, ref_bgr):
    """Shrink the high-freq residual std toward the reference crop's residual std."""
    def residual(x):
        blur = cv2.GaussianBlur(x, (5, 5), 0).astype(np.float32)
        return x.astype(np.float32) - blur, blur

    res, blur = residual(img_bgr)
    ref_res, _ = residual(ref_bgr)
    s_img = res.std() + 1e-6
    s_ref = ref_res.std() + 1e-6
    factor = min(1.0, s_ref / s_img)  # only shrink, never amplify
    out = blur + res * factor
    return np.clip(out, 0, 255).astype(np.uint8)


# intervention_type (codebook) -> callable. residual_renorm needs a reference.
INTERVENTIONS = {
    "spectral_notch": notch_hf,
    "checkerboard_suppress": checkerboard_supp,
    "residual_renormalize": residual_renorm,
}

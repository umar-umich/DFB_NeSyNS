"""The control battery: what separates a causal drop from a fragile one.

A large p(fake) drop after repair is only evidence that the cited region is
load-bearing if the drop survives these controls:

  matched_blur / matched_shift  An inert, non-semantic distortion of the SAME
        region, its magnitude tuned so its LPIPS distance to the original equals
        the repair's LPIPS distance. If a perceptually-equal blur drops the score
        as much as the repair, the detector was reacting to perturbation
        magnitude, not the manipulated content.
  wrong_region  The same Poisson repair applied to an equal-area region OUTSIDE
        the manipulation, hugging the mask boundary. Must be near-inert: a drop
        here means the test is not specific to the cited region.
  real_offset  The same repair op applied to the paired REAL frame (real content
        from a +/-1 frame). Measures the operation's own artifact; its
        distribution is the normalization floor, not required to be zero.

All ported from scripts/pilot1_rev3.py and scripts/intervention_pilot.py. The
Task 4 acceptance gate depends on byte-identical behaviour.
"""
from __future__ import annotations

import cv2
import numpy as np

from .repair import poisson_paste

# Corruption magnitude sweeps, from pilot1_rev3.py:match_corruption.
BLUR_KSIZES = [3, 5, 7, 9, 11, 15, 19, 25, 31, 41, 55, 71]
SHIFT_DISTS = [1, 2, 3, 4, 6, 8, 10, 13, 16, 20, 25, 30]


# --------------------------------------------------------------------------- corruptions
def blur_region(img_bgr, region_mask, ksize):
    """Gaussian blur applied only inside region_mask (pilot1_rev3.blur_region)."""
    if ksize < 3:
        return img_bgr.copy()
    k = ksize | 1  # kernel must be odd
    blurred = cv2.GaussianBlur(img_bgr, (k, k), 0)
    out = img_bgr.copy()
    out[region_mask] = blurred[region_mask]
    return out


def shift_region(img_bgr, region_mask, dx):
    """Shift the region's pixel content by dx px, region only (pilot1_rev3.shift_region)."""
    out = img_bgr.copy()
    ys, xs = np.where(region_mask)
    nx = np.clip(xs + dx, 0, img_bgr.shape[1] - 1)
    out[ys, xs] = img_bgr[ys, nx]
    return out


# --------------------------------------------------------------------------- LPIPS matching
class Lpips:
    """LPIPS(alex) distance between BGR uint8 crops (pilot1_rev3.Lpips)."""

    def __init__(self, device):
        import lpips
        import torch
        self.torch = torch
        self.net = lpips.LPIPS(net="alex", verbose=False).eval().to(device)
        self.device = device

    def _t(self, x_bgr):
        x = cv2.cvtColor(x_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        return self.torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def dist(self, a_bgr, b_bgr):
        with self.torch.no_grad():
            return float(self.net(self._t(a_bgr), self._t(b_bgr)).item())

    def dist_batch(self, ref_bgr, cand_list):
        r = self._t(ref_bgr)
        with self.torch.no_grad():
            return [float(self.net(r, self._t(c)).item()) for c in cand_list]


def match_corruption(orig_bgr, region_mask, target_lpips, lp, kind):
    """Return the corruption whose LPIPS to orig is closest to target_lpips.

    kind is 'blur' or 'shift'. Returns (image, achieved_lpips, param). Ported
    from pilot1_rev3.match_corruption.
    """
    if kind == "blur":
        params = BLUR_KSIZES
        cands = [blur_region(orig_bgr, region_mask, k) for k in params]
    elif kind == "shift":
        params = SHIFT_DISTS
        cands = [shift_region(orig_bgr, region_mask, d) for d in params]
    else:
        raise ValueError(f"unknown corruption kind '{kind}'")
    dists = lp.dist_batch(orig_bgr, cands)
    j = int(np.argmin([abs(d - target_lpips) for d in dists]))
    return cands[j], dists[j], params[j]


# --------------------------------------------------------------------------- wrong region
def make_control_region(mask):
    """Equal-area region OUTSIDE the manipulation, hugging the mask boundary.

    A ring grown outward from the mask until it holds >= area(mask) pixels, then
    trimmed to exactly area(mask) by proximity. Same area, fully disjoint, and
    adjacent to the manipulation, so it cannot be dismissed as "edited somewhere
    unrelated." Ported verbatim from intervention_pilot.make_control_region.
    """
    A = int(mask.sum())
    if A == 0:
        return None
    inv = (~mask).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5).flatten()
    m = mask.astype(np.uint8)
    for _ in range(80):
        d = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=1).astype(bool)
        ring = d & ~mask
        if ring.sum() >= A:
            idx = np.where(ring.flatten())[0]
            sel = idx[np.argsort(dist[idx])][:A]
            ctrl = np.zeros(mask.size, bool)
            ctrl[sel] = True
            return ctrl.reshape(mask.shape)
        m = d.astype(np.uint8)
    out = ~mask
    if out.sum() >= A:
        idx = np.where(out.flatten())[0]
        sel = idx[np.argsort(dist[idx])][:A]
        ctrl = np.zeros(mask.size, bool)
        ctrl[sel] = True
        return ctrl.reshape(mask.shape)
    return None


# --------------------------------------------------------------------------- full battery
def build_variants(pair, lp):
    """Build all repair/control variants for a PairSample, over its GT mask.

    Returns dict name -> BGR image:
      orig          the fake, unmodified
      gt_repair     Poisson repair of the GT mask with paired-real pixels
      match_blur    LPIPS-matched blur of the GT mask
      match_shift   LPIPS-matched shift of the GT mask
      wrong_region  Poisson repair of an equal-area region outside the mask
      real_orig     the paired real, unmodified (for the real-offset baseline)
      real_offset   the repair op applied to the real (+/-1 frame)  [if available]

    Mirrors pilot1_rev3.build_sample_rev3 + the matched-corruption step in main().
    """
    from cec.data.pairing import align_paired_real

    fake, real, mask = pair.fake, pair.real, pair.mask
    variants = {
        "orig": fake,
        "gt_repair": poisson_paste(fake, real, mask),
        "real_orig": real,
    }

    # LPIPS-matched corruptions, tuned to the gt-repair LPIPS distance.
    target = lp.dist(fake, variants["gt_repair"])
    variants["match_blur"], _, _ = match_corruption(fake, mask, target, lp, "blur")
    variants["match_shift"], _, _ = match_corruption(fake, mask, target, lp, "shift")

    # Wrong-region: same repair, equal-area region outside the mask.
    ctrl = make_control_region(mask)
    if ctrl is not None:
        variants["wrong_region"] = poisson_paste(fake, real, ctrl)

    # Real-offset: the repair op on real content from an adjacent frame.
    real_b = align_paired_real(pair.landmarks, pair.vid, pair.frame, offset=1)
    if real_b is None:
        real_b = align_paired_real(pair.landmarks, pair.vid, pair.frame, offset=-1)
    if real_b is not None:
        variants["real_offset"] = poisson_paste(real, real_b, mask)

    return variants

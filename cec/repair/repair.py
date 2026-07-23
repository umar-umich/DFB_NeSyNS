"""Region repair: replace a cited region with paired-real pixels.

The core intervention. To certify a spatial claim the gate repairs the cited
region with the paired real and re-queries the detector; the p(fake) drop is the
Necessity Margin. Poisson blending (seamlessClone) is used rather than a hard
paste so the repaired region carries no seam of its own that the detector could
react to — the point is to measure the manipulated *content*, not a new artifact
the repair introduced.

Ported verbatim from scripts/pilot1_rev3.py:poisson_paste. Byte-for-byte
behaviour matters: the Task 4 acceptance gate reproduces effort's rev3 numbers
(Δgt = 0.648 / Δblur = +0.001 on Deepfakes/257_420/762), and any change to the
blend would move them.
"""
from __future__ import annotations

import cv2
import numpy as np

# seamlessClone degenerates on tiny regions; below this pixel count, hard-paste.
MIN_POISSON_PX = 20


def poisson_paste(dst_bgr, src_bgr, region_mask):
    """Poisson-blend src into dst over region_mask (bool HxW).

    Masked to the region, so background pixels are untouched. Falls back to a
    hard paste when the region is too small for seamlessClone (< MIN_POISSON_PX)
    or when seamlessClone raises on a degenerate region.
    """
    m = region_mask.astype(np.uint8) * 255
    ys, xs = np.where(region_mask)
    if len(ys) < MIN_POISSON_PX:
        return hard_paste(dst_bgr, src_bgr, region_mask)
    cy = int((ys.min() + ys.max()) / 2)
    cx = int((xs.min() + xs.max()) / 2)
    try:
        return cv2.seamlessClone(src_bgr, dst_bgr, m, (cx, cy), cv2.NORMAL_CLONE)
    except cv2.error:
        return hard_paste(dst_bgr, src_bgr, region_mask)


def hard_paste(dst_bgr, src_bgr, region_mask):
    """Copy src pixels into dst over region_mask, no blending.

    This is the v1 repair op (intervention_pilot.py:revert). Kept as the Poisson
    fallback for tiny regions; NOT the production repair (v1's hard paste was
    retired in favour of Poisson — see docs/cec/LOG.md, CONFLICT 2 resolution).
    """
    out = dst_bgr.copy()
    out[region_mask] = src_bgr[region_mask]
    return out

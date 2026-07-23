"""
networks/cec/inference.py
=========================
Batched p(fake) inference for the frozen CEC instruments.

Ported from `pfake_batch` in `scripts/pilot_detectors.py`. The gate re-queries a
detector once per claim per intervention, so interventions x claims x models is
the cost centre — batching is what makes the audit run affordable.
"""

import numpy as np
import torch

from .gend_registry import pfake_from_tensor, to_pil_rgb

DEFAULT_BATCH_SIZE = 64


@torch.no_grad()
def pfake_batch(model, transform, images, device, batch_size=DEFAULT_BATCH_SIZE):
    """p(fake) for a list of images (BGR uint8 arrays or PIL).

    Returns a float32 ndarray of shape (len(images),).
    """
    if len(images) == 0:
        return np.empty(0, dtype=np.float32)

    out = []
    for start in range(0, len(images), batch_size):
        chunk = images[start:start + batch_size]
        x = torch.stack([transform(to_pil_rgb(im)) for im in chunk]).to(device)
        out.append(pfake_from_tensor(model, x).cpu().numpy())
    return np.concatenate(out).astype(np.float32)

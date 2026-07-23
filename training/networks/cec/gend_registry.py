"""
networks/cec/gend_registry.py
=============================
Adapter onto the frozen GenD detector registry at /data/umar/Repos/GenD_NeSy.

Ported from `scripts/pilot_detectors.py`, which produced every frozen anchor in
CURRENT_STATE ledger item 6. The load path is deliberately identical, because a
different transform or a different checkpoint resolution silently renumbers all
of them.

Two deviations from the pilot script, both forced by living inside DeepfakeBench
rather than in a standalone script:

1. **chdir is scoped, not module-level.** GenD resolves checkpoints by RELATIVE
   path (fsfm string-matches its exact checkpoint path), so construction must
   run with cwd == the GenD repo. The pilot does `os.chdir(GEND)` at import,
   which is safe for a one-shot script but would break DFB's own relative
   config/weight resolution here. `gend_cwd()` wraps construction only; the cwd
   is always restored.
2. **sys.path is appended, not prepended.** The pilot does `sys.path.insert(0, GEND)`.
   Prepending would let GenD's top-level `datasets/`, `config/`, `scripts/`, and
   `results/` directories shadow same-named modules — including, via PEP-420
   namespace packages, the HuggingFace `datasets` library. `src` and `run` are
   unique to GenD, so appending resolves them identically while shadowing nothing.

Neither deviation changes which weights are loaded or which transform is applied;
the Task 4 acceptance gate re-checks that against the pilot's own numbers.
"""

import contextlib
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# The GenD checkout. Absolute everywhere except inside gend_cwd().
GEND_ROOT = Path("/data/umar/Repos/GenD_NeSy")


@contextlib.contextmanager
def gend_cwd(gend_root=GEND_ROOT):
    """Run a block with cwd == the GenD repo, then restore.

    Required around GenD model construction ONLY. Never hold this across DFB
    work: DFB resolves its own config and weight paths relatively.
    """
    previous = os.getcwd()
    os.chdir(str(gend_root))
    try:
        yield
    finally:
        os.chdir(previous)


def _ensure_importable(gend_root=GEND_ROOT):
    """Make GenD's `src` and `run` importable without shadowing DFB or site-packages."""
    path = str(gend_root)
    if path not in sys.path:
        sys.path.append(path)


def build_gend_model(spec, device, gend_root=GEND_ROOT):
    """Build a frozen GenD detector from a registration spec.

    spec: dict with `checkpoint` (relative to the GenD repo, or an HF hub id) and
          optionally `model_type` and `zoom`.

    Returns (model, transform) where transform maps a PIL RGB image to a tensor.
    The model is eval()'d and moved to `device`: these instruments are frozen and
    are never trained (CURRENT_STATE rule 4).
    """
    _ensure_importable(gend_root)
    with gend_cwd(gend_root):
        from src.config import Config, CustomPreprocessing  # noqa: E402
        from run import load_model  # noqa: E402

        kwargs = {"checkpoint": spec["checkpoint"]}
        if spec.get("model_type"):
            kwargs["model_type"] = spec["model_type"]
        if spec.get("zoom"):
            kwargs["custom_preprocessing"] = CustomPreprocessing(zoom_factor=spec["zoom"])

        config = Config(**kwargs)
        model = load_model(config)
        model.load_checkpoint(config.checkpoint)

    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model, model.get_preprocessing()


@torch.no_grad()
def pfake_from_tensor(model, x):
    """p(fake) for a preprocessed batch tensor. logits_labels is [real, fake]."""
    return F.softmax(model(x).logits_labels, dim=1)[:, 1]


def to_pil_rgb(image):
    """Accept the pilot's BGR uint8 arrays or a PIL image; return PIL RGB.

    The harness reads frames with cv2, so BGR uint8 is the native currency here;
    getting the channel order wrong would quietly change every score.
    """
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    raise TypeError(f"expected PIL.Image or BGR uint8 ndarray, got {type(image)}")

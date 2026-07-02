"""Frozen-detector registry for the intervention verifier pilot.

All detectors come from /data/umar/Repos/GenD_NeSy and share the interface
    model(tensor) -> HeadOutput(logits_labels)        # [real, fake] logits
    model.get_preprocessing() -> (PIL.Image -> tensor)
so p(fake) = softmax(logits)[:, 1].

The GenD repo loads its checkpoints by RELATIVE path (FSFM even matches the
exact string), so we chdir into it once at import. Data paths in the pilot are
absolute, so this does not affect them.
"""
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

GEND = "/data/umar/Repos/GenD_NeSy"
sys.path.insert(0, GEND)
os.chdir(GEND)
from src.config import Config, CustomPreprocessing   # noqa: E402
from run import load_model                            # noqa: E402

# name -> how to build the Config. checkpoints are relative to the GenD repo.
REGISTRY = {
    "effort": dict(model_type="effort",
                   checkpoint="weights/Effort/effort_clip_L14_trainOn_FaceForensic.pth"),
    "forada": dict(checkpoint="weights/ForAda/ForAdackpt_best.pth"),
    "fsfm":   dict(checkpoint="weights/FS-VFM/FS-VFM-ViT-L.pth", zoom=1.3),
    "gend":   dict(checkpoint="yermandy/GenD_CLIP_L_14"),
}


def available():
    return list(REGISTRY.keys())


def load_detector(name, device):
    """Return (model, transform) where transform maps a PIL RGB image to a tensor."""
    if name not in REGISTRY:
        raise ValueError(f"Unknown detector '{name}'. Choices: {available()}")
    spec = REGISTRY[name]
    kw = {"checkpoint": spec["checkpoint"]}
    if spec.get("model_type"):
        kw["model_type"] = spec["model_type"]
    if spec.get("zoom"):
        kw["custom_preprocessing"] = CustomPreprocessing(zoom_factor=spec["zoom"])
    config = Config(**kw)
    model = load_model(config)
    model.load_checkpoint(config.checkpoint)
    model.eval().to(device)
    return model, model.get_preprocessing()


@torch.no_grad()
def pfake_batch(model, transform, bgr_list, device, bs=64):
    """bgr_list: HxWx3 uint8 BGR arrays -> np.array of p(fake)."""
    probs = []
    for i in range(0, len(bgr_list), bs):
        chunk = bgr_list[i:i + bs]
        x = torch.stack([
            transform(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))) for im in chunk
        ]).to(device)
        out = model(x)
        logits = out.logits_labels
        probs.append(F.softmax(logits, 1)[:, 1].cpu().numpy())
    return np.concatenate(probs)

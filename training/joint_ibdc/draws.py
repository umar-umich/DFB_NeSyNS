#!/usr/bin/env python3
"""The augmentation draw, in ONE place, shared by the spatial and temporal FF++ train extractors.

Option B, matched draws: one arm-B setting per (VIDEO, draw index), used by EVERY view. Keying on
the video rather than the frame is what makes the matching possible at all -- the temporal view
applies one setting per clip, so a per-frame spatial draw could never align with it, and a sample
would then have no single "augmentation level" for G0b to report against.

This is a deliberate choice for the joint study and NOT a reproduction of deployed training, which
is unmatched in two ways: its spatial arm B draws per SAMPLE, and its spatial views are not
augmented at all (`warm_dicome_gate_seed42.yaml` has no `training_arm`; the arm-B spatial config is a
separate file). The video view's arm B, drawn per clip, is the only part of this that deployment
also does.

Both extractors import from here so the two caches cannot drift apart; `test_draws.py` asserts that
a frame key and its video resolve to the same setting.
"""
from __future__ import annotations

import re
import sys
import zlib
from pathlib import Path

import numpy as np

EXT = Path("/data/umar/Repos/DISCERN_Ext")
if str(EXT) not in sys.path:
    sys.path.insert(0, str(EXT))
from src.dataset.training_arms import SharedProcessing  # noqa: E402  the grid training used

SEED = 42
CLASS_DIRS = {"real", "fake", "frames", "images"}


def video_of(key: str) -> str:
    """THE video rule (analysis/tbiom/video_id.py): the full directory path, and the frame itself
    when the parent is a bare class name."""
    k = str(key).split("::", 1)[-1].rstrip("/")
    directory, _, _frame = k.rpartition("/")
    if directory.rsplit("/", 1)[-1].lower() in CLASS_DIRS:
        return re.sub(r"\.[A-Za-z0-9]+$", "", k)
    return directory


def as_video(s: str) -> str:
    """Accept either a frame key or an already-resolved video id, and return the video id.

    The two are told apart by whether the LAST path component carries a file extension. Getting this
    wrong is silent: an over-eager video_of on an id would strip one more component and the spatial
    and temporal caches would draw different settings for the same video, which is precisely the
    matching that option B exists to guarantee. test_draws.py pins it.
    """
    k = str(s).split("::", 1)[-1].rstrip("/")
    last = k.rsplit("/", 1)[-1]
    return video_of(k) if re.search(r"\.[A-Za-z0-9]+$", last) else k


def setting_for(key_or_video: str, draw: int, arm: str = "B", seed: int = SEED,
                proc: SharedProcessing | None = None):
    """One setting per (video, draw). Accepts a frame key or a video id; both give the same answer."""
    if arm != "B":
        return ("none", None)
    proc = proc or SharedProcessing(enabled=True)
    rng = np.random.default_rng([seed, draw, zlib.crc32(as_video(key_or_video).encode())])
    return proc.sample(rng)


def apply_to(img_uint8, setting, proc: SharedProcessing | None = None):
    proc = proc or SharedProcessing(enabled=True)
    return img_uint8 if setting[0] == "none" else proc.apply(img_uint8, setting)

"""DF40 sample loading for the spectral arm (Pilot S).

DF40 is spectral-only in CEC (no masks, own cropper — see docs/cec/LOG.md). For
Pilot S we need real and fake crops from its GENERATIVE families, where a
frequency detector should separate. Those families ship their own geometry-
matched reals in a `real/` subdir alongside `fake/`, so no FF++ pairing is
needed and no crop-distribution confound is introduced.

Layout varies per family (flat, or one dir per clip, png or jpg), so images are
globbed recursively under each family's fake/ and real/ roots.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np

DF40_TEST = Path("/data/umar/Datasets/df40/test")

# Generative families with a real/ subdir: GANs + diffusion, the spectral arm's
# fair test. Verified on disk 2026-07-16.
GENERATIVE_FAMILIES = {
    "stargan": "gan",
    "starganv2": "gan",
    "CollabDiff": "diffusion",
    "MidJourney": "diffusion",
}

_IMG_EXT = ("*.png", "*.jpg", "*.jpeg")


# Detectors' frozen anchors were computed on 256x256 crops; DF40 native sizes
# vary (256 .. 1024). Crops are resized to this at load so batches stack and the
# resolution matches what the detectors expect.
CROP_SIZE = 256


@dataclass
class DF40Sample:
    family: str
    label: int          # 1 = fake, 0 = real
    path: Path
    image: np.ndarray   # BGR uint8, resized to CROP_SIZE
    native_hw: tuple     # (h, w) BEFORE resize — the resolution-confound audit


def _glob_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    out: List[Path] = []
    for ext in _IMG_EXT:
        out.extend(root.rglob(ext))
    # drop macOS resource-fork junk
    return sorted(p for p in out if "__MACOSX" not in p.parts)


def list_family(family: str):
    """(fake_paths, real_paths) for a DF40 family, sorted."""
    fam_dir = DF40_TEST / family
    return _glob_images(fam_dir / "fake"), _glob_images(fam_dir / "real")


def load_samples(family: str, n_per_class: int, rng) -> List[DF40Sample]:
    """Up to n_per_class fakes + n_per_class reals from a family, shuffled by rng."""
    fakes, reals = list_family(family)
    if not fakes or not reals:
        raise FileNotFoundError(
            f"DF40 family '{family}' missing fake/ or real/ images "
            f"(fakes={len(fakes)}, reals={len(reals)})."
        )
    rng.shuffle(fakes)
    rng.shuffle(reals)

    samples: List[DF40Sample] = []
    for label, paths in ((1, fakes), (0, reals)):
        for p in paths[:n_per_class]:
            img = cv2.imread(str(p))
            if img is None:
                continue
            native_hw = img.shape[:2]
            if native_hw != (CROP_SIZE, CROP_SIZE):
                # INTER_AREA for down, INTER_CUBIC for up — cv2 picks by direction
                # only via flag, so choose per-image.
                interp = cv2.INTER_AREA if native_hw[0] > CROP_SIZE else cv2.INTER_CUBIC
                img = cv2.resize(img, (CROP_SIZE, CROP_SIZE), interpolation=interp)
            samples.append(DF40Sample(family=family, label=label, path=p,
                                      image=img, native_hw=native_hw))
    return samples


def family_native_resolution(samples: List[DF40Sample]):
    """Modal native (h,w) for fakes vs reals, and whether they match.

    A mismatch means a uniform resize injects a different interpolation
    fingerprint into each class — a spectral AUC on such a family is confounded.
    """
    from collections import Counter

    def modal(label):
        hws = [s.native_hw for s in samples if s.label == label]
        return Counter(hws).most_common(1)[0][0] if hws else None

    fake_hw, real_hw = modal(1), modal(0)
    return {"fake": fake_hw, "real": real_hw, "res_matched": fake_hw == real_hw}

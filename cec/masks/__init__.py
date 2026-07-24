"""CEC masks: codebook location vocabulary -> deterministic binary region masks."""

from .regions import (  # noqa: F401
    FIDELITY,
    FaceGeometry,
    align_matrix,
    aligned_landmarks,
    composite_mask,
    region_mask,
)

__all__ = [
    "region_mask",
    "composite_mask",
    "FaceGeometry",
    "align_matrix",
    "aligned_landmarks",
    "FIDELITY",
]

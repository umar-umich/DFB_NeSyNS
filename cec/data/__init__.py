"""CEC data: pairing FF++ fakes with pixel-registered paired reals."""

from .pairing import (  # noqa: F401
    PairSample,
    align_paired_real,
    build_pair,
    fake_paths,
    read_raw_frame,
    source_video,
    verify_pair,
)

__all__ = [
    "PairSample",
    "build_pair",
    "verify_pair",
    "align_paired_real",
    "fake_paths",
    "source_video",
    "read_raw_frame",
]

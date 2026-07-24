"""CEC data: pairing FF++ fakes with pixel-registered paired reals."""

from .pairing import (  # noqa: F401
    PairSample,
    align_paired_real,
    build_pair,
    build_real_pair,
    fake_paths,
    read_raw_frame,
    source_video,
    split_video_ids,
    verify_pair,
)

__all__ = [
    "PairSample",
    "build_pair",
    "build_real_pair",
    "verify_pair",
    "align_paired_real",
    "fake_paths",
    "source_video",
    "split_video_ids",
    "read_raw_frame",
]

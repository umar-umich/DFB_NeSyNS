#!/usr/bin/env python3
"""The I-JEPA video rule must be THE video rule.

    <discern_ext python> -m pytest branches/ijepa/test_video_rule.py -v

`analysis/tbiom/video_id.py` is the single implementation of video identity in this project, and its
docstring records two occasions where a local shortcut silently regrouped frames and reversed a
conclusion. This asserts that `branches/ijepa/data.py:video_of` agrees with it on real keys drawn
from every dataset the extraction touches, so a future divergence fails a test rather than quietly
changing a number.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
SPLITS = Path("/data/umar/Repos/DISCERN_Ext/eval_adaptation/data/splits")
sys.path.insert(0, str(ROOT / "branches/ijepa"))
sys.path.insert(0, str(ROOT / "analysis/tbiom"))

from data import video_of                 # noqa: E402  the module under test
from video_id import video_id             # noqa: E402  the canonical rule

DATASETS = ["FFpp_train", "FFpp_val", "VALmix", "VALmixV2", "CDFv2", "CDFv3", "DF40",
            "DFD", "DFDC", "DFDCP", "DFDM", "DFEval24", "FaceShifter", "FFIW", "IDForge", "KoDF"]


def sample_keys(ds, per_manifest=40):
    d = SPLITS / ds
    if not d.is_dir():
        return []
    out = []
    for m in sorted(d.glob("*.txt")):
        lines = [l.strip() for l in m.read_text().splitlines() if l.strip()]
        out += lines[:per_manifest] + lines[-per_manifest:]
    return out


@pytest.mark.parametrize("ds", DATASETS)
def test_matches_canonical_rule(ds):
    keys = sample_keys(ds)
    if not keys:
        pytest.skip(f"{ds} has no manifests")
    want = video_id(pd.Series(keys)).tolist()
    got = [video_of(k) for k in keys]
    bad = [(k, g, w) for k, g, w in zip(keys, got, want) if g != w]
    assert not bad, f"{ds}: {len(bad)} of {len(keys)} disagree, e.g. {bad[:3]}"


def test_basename_collision_is_not_merged():
    """The DF40 failure the canonical rule exists to prevent."""
    a = "DF40/blendface-cdf/id0_0000/frame_0.png"
    b = "DF40/CollabDiff-real/id0_0000/frame_0.png"
    assert video_of(a) != video_of(b)
    assert video_of(a) == video_id(pd.Series([a]))[0]


def test_flat_whole_image_methods_do_not_collapse():
    """A bare class-name parent means each frame is its own video."""
    keys = [f"DF40/ddim-cdf/fake/{i:05d}.png" for i in range(5)]
    got = [video_of(k) for k in keys]
    assert len(set(got)) == 5, got
    assert got == video_id(pd.Series(keys)).tolist()


def test_h5_double_colon_form():
    k = "eval_adaptation/data/h5/FFpp_val.h5::FFpp_val/Deepfakes/Deepfakes__004_982/000.png"
    assert video_of(k) == "FFpp_val/Deepfakes/Deepfakes__004_982"
    assert video_of(k) == video_id(pd.Series([k]))[0]

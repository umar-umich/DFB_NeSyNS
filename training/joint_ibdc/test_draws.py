#!/usr/bin/env python3
"""The matched draw must be one rule, and it must agree for a frame key and its video."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from draws import setting_for, video_of, SharedProcessing  # noqa: E402

SPL = Path("/data/umar/Repos/DISCERN_Ext/eval_adaptation/data/splits/FFpp_train")


def keys(n=4000):
    out = []
    for m in sorted(SPL.glob("*.txt")):
        out += [l.strip() for l in m.read_text().splitlines() if l.strip()][:n // 5]
    return out


def test_frame_key_and_video_id_agree():
    """The spatial extractor passes a frame key, the temporal one a video id. Same answer."""
    for k in keys(500):
        assert setting_for(k, 0) == setting_for(video_of(k), 0), k


def test_constant_within_a_video():
    byvid = {}
    for k in keys():
        byvid.setdefault(video_of(k), set()).add(setting_for(k, 0))
    bad = {v: s for v, s in byvid.items() if len(s) != 1}
    assert not bad, list(bad.items())[:3]


def test_reproducible_and_independent_across_draws():
    ks = keys(2000)
    assert [setting_for(k, 0) for k in ks] == [setting_for(k, 0) for k in ks]
    d0 = [setting_for(k, 0) for k in ks]
    d1 = [setting_for(k, 1) for k in ks]
    frac = sum(a != b for a, b in zip(d0, d1)) / len(ks)
    assert 0.6 < frac < 0.9, frac            # the grid predicts 0.78


def test_mix_matches_arm_b():
    c = Counter(f"{a}:{b}" for a, b in (setting_for(k, 0) for k in keys()))
    tot = sum(c.values())
    assert 0.33 < c["none:None"] / tot < 0.47, c
    for sev in ("jpeg:70", "jpeg:50", "jpeg:30", "resize:0.75", "resize:0.5", "resize:0.35"):
        assert 0.05 < c[sev] / tot < 0.16, (sev, c[sev] / tot)


def test_arm_none_is_inert():
    assert {setting_for(k, d, arm="none") for k in keys(300) for d in range(3)} == {("none", None)}


def test_temporal_cache_settings_match_the_shared_rule():
    """If the temporal cache exists, what it RECORDED must equal what the rule gives.

    This is the end-to-end statement of option B: the spatial extractor draws from the same rule, so
    agreement here means the two caches cannot disagree about a video's augmentation.
    """
    import numpy as np
    C = Path("/data/umar/Repos/DFB_NeSyNS/logs/joint_ibdc/cache")
    proc = SharedProcessing(enabled=True)
    checked = 0
    for d in range(3):
        f = C / f"FFpp_train_T_B_d{d}.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        for v, rec in zip(z["video"], z["setting"]):
            a, b = setting_for(str(v), d, proc=proc)
            assert str(rec) == (f"{a}:{b}" if b is not None else "none"), (v, rec, a, b)
            checked += 1
    if checked == 0:
        pytest.skip("temporal cache not extracted yet")
    assert checked >= 3600

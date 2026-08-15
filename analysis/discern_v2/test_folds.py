#!/usr/bin/env python3
"""Unit tests for the leave-one-generator-out fold construction.

`logo_folds_two_class` decides what every A2a/A3 number means, and its failure modes are
silent rather than loud: a single-class fold produces "no result" instead of an error, and a
real video split across train and test inflates every probe without ever looking wrong. Both
bugs were live in the first version of these scripts, so they are pinned here.

    python analysis/discern_v2/test_folds.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def _toy(n_gen: int = 4, n_fake: int = 25, n_real_vid: int = 20, frames: int = 3):
    """Frame-level toy set: n_gen generators of fakes plus a shared pool of real videos."""
    methods, labels, vids = [], [], []
    for g in range(n_gen):
        for i in range(n_fake):
            for f in range(frames):
                methods.append(f"gen{g}")
                labels.append(1)
                vids.append(f"fake_{g}_{i}")
    for i in range(n_real_vid):
        for f in range(frames):
            methods.append("real")
            labels.append(0)
            vids.append(f"real_{i}")
    return np.array(methods), np.array(labels), np.array(vids)


def test_folds_are_two_class():
    m, y, v = _toy()
    folds = C.logo_folds_two_class(m, y, min_per_group=20, video_ids=v)
    assert folds, "no folds produced -- the single-class bug is back"
    for held, tr, te in folds:
        assert len(np.unique(y[te])) == 2, f"{held}: test side is single-class"
        assert len(np.unique(y[tr])) == 2, f"{held}: train side is single-class"
    print(f"  ok: {len(folds)} folds, all two-class on both sides")


def test_held_out_generator_never_trains():
    m, y, v = _toy()
    for held, tr, te in C.logo_folds_two_class(m, y, min_per_group=20, video_ids=v):
        assert not (m[tr] == held).any(), f"{held}: held-out generator leaked into train"
        assert (m[te] == held).any(), f"{held}: held-out generator missing from test"
    print("  ok: held-out generator is absent from train and present in test")


def test_no_real_video_on_both_sides():
    m, y, v = _toy()
    for held, tr, te in C.logo_folds_two_class(m, y, min_per_group=20, video_ids=v):
        shared = set(v[tr]) & set(v[te])
        assert not shared, f"{held}: {len(shared)} videos on both sides, e.g. {list(shared)[:3]}"
    print("  ok: no video appears on both sides of any fold")


def test_frame_level_split_is_by_video():
    """Without video_ids the split is row-wise; with them it must respect video boundaries."""
    m, y, v = _toy()
    naive = C.logo_folds_two_class(m, y, min_per_group=20)          # row-wise
    byvid = C.logo_folds_two_class(m, y, min_per_group=20, video_ids=v)
    naive_shared = max(len(set(v[tr]) & set(v[te])) for _, tr, te in naive)
    byvid_shared = max(len(set(v[tr]) & set(v[te])) for _, tr, te in byvid)
    assert naive_shared > 0, "toy set cannot demonstrate the leak"
    assert byvid_shared == 0
    print(f"  ok: row-wise split leaks {naive_shared} videos, video-aware split leaks 0")


def test_small_generators_are_skipped():
    m, y, v = _toy(n_gen=3, n_fake=25)
    m = np.concatenate([m, np.array(["tiny"] * 6)])
    y = np.concatenate([y, np.ones(6, dtype=int)])
    v = np.concatenate([v, np.array([f"tiny_{i}" for i in range(6)])])
    held = [h for h, _, _ in C.logo_folds_two_class(m, y, min_per_group=20, video_ids=v)]
    assert "tiny" not in held, "a 6-sample generator produced a fold"
    print("  ok: under-sized generator groups are skipped, not silently reported")


def test_determinism():
    m, y, v = _toy()
    a = C.logo_folds_two_class(m, y, min_per_group=20, video_ids=v)
    b = C.logo_folds_two_class(m, y, min_per_group=20, video_ids=v)
    for (h1, tr1, te1), (h2, tr2, te2) in zip(a, b):
        assert h1 == h2 and np.array_equal(tr1, tr2) and np.array_equal(te1, te2)
    print("  ok: fold construction is deterministic across calls")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"running {len(fns)} fold tests\n")
    for fn in fns:
        print(f"{fn.__name__}:")
        fn()
    print("\nall fold tests passed")

#!/usr/bin/env python3
"""Loading the I-JEPA residual cache, with labels and video grouping.

KEYS ARE INDEXED FROM THE END, ALWAYS. A cache key is a manifest line, `.../<source>/<video>/<frame>`.
The source is `parts[-3]` and the video is `parts[-2]`. Indexing from the FRONT is what silently broke
the leakage audit: VALmix paths are three deep and VALmix-v2 paths are four, so a front index read the
wrong component for one of them and produced a false PASS. The label rule itself
(`0 if "real" in source.lower() else 1`) is DISCERN_Ext's own, from
`src/dataset/deepfake.py:_build_labels`, reused rather than re-derived; it was checked against every
manifest of FFpp_train, FFpp_val, VALmix, VALmixV2, CDFv2, KoDF and DF40 with no malformed key and no
case where the manifest stem and the source component disagreed.

Shards are transparent: `load("IDForge")` concatenates `IDForge.s0of4 .. s3of4` in shard order. A
dataset extracted whole loads from the single file.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

CACHE = Path("/data/umar/Repos/DFB_NeSyNS/analysis/ijepa/cache")
ARRAYS = ("eps", "rbar", "pooled", "sbar", "q")


def source_of(key: str) -> str:
    return str(key).split("/")[-3]


CLASS_DIRS = {"real", "fake", "frames", "images"}          # mirrors analysis/tbiom/video_id.py:42


def video_of(key: str) -> str:
    """THE video-identity rule, from `analysis/tbiom/video_id.py`, not a local shortcut.

    The video is the DIRECTORY CONTAINING THE FRAME AS A FULL PATH -- not the directory's
    basename. DF40 borrows its authentic halves, so a real and a fake video can share the
    basename `id0_0000`; grouping on the basename silently merges them and `max(label)` then
    depends on which frames an export happened to sample. That bug has already reversed two
    conclusions in this project. When the parent directory is a bare class name, the frames are
    flat whole images and each frame is its own video, or every generated image would collapse
    into one.

    `test_video_rule.py` asserts this agrees with the canonical implementation on real keys.
    """
    k = str(key).split("::", 1)[-1].rstrip("/")
    directory, _, _frame = k.rpartition("/")
    parent = directory.rsplit("/", 1)[-1]
    if parent.lower() in CLASS_DIRS:
        return re.sub(r"\.[A-Za-z0-9]+$", "", k)          # degenerate: the frame IS the video
    return directory


def label_of(key: str) -> int:
    """DISCERN_Ext's rule, applied to the source component."""
    return 0 if "real" in source_of(key).lower() else 1


def labels_of(keys) -> np.ndarray:
    return np.array([label_of(k) for k in keys], dtype=np.int64)


def videos_of(keys) -> np.ndarray:
    return np.array([video_of(k) for k in keys])


def sources_of(keys) -> np.ndarray:
    return np.array([source_of(k) for k in keys])


def shard_files(stem: str) -> list[Path]:
    """Every file belonging to `stem`, in shard order. Empty if nothing is cached yet."""
    whole = CACHE / f"{stem}.npz"
    if whole.exists():
        return [whole]
    pat = re.compile(rf"^{re.escape(stem)}\.s(\d+)of(\d+)\.npz$")
    found = []
    for p in CACHE.glob(f"{stem}.s*of*.npz"):
        m = pat.match(p.name)
        if m:
            found.append((int(m.group(1)), int(m.group(2)), p))
    if not found:
        return []
    n = found[0][1]
    if any(t != n for _, t, _ in found):
        raise ValueError(f"{stem}: mixed shard counts {sorted({t for _, t, _ in found})}")
    found.sort()
    have = [s for s, _, _ in found]
    if have != list(range(n)):
        missing = sorted(set(range(n)) - set(have))
        raise FileNotFoundError(f"{stem}: shards {missing} of {n} are not cached yet")
    return [p for _, _, p in found]


def load(stem: str, arrays=ARRAYS) -> dict:
    """Load one cached dataset, concatenating shards. Returns arrays plus keys, y, video, source."""
    files = shard_files(stem)
    if not files:
        raise FileNotFoundError(f"nothing cached for {stem!r} under {CACHE}")
    parts = {a: [] for a in arrays}
    keys = []
    for f in files:
        with np.load(f, allow_pickle=True) as z:
            for a in arrays:
                if a not in z:
                    raise KeyError(f"{f.name} has no {a!r}; it predates the current extractor")
                parts[a].append(z[a])
            keys.append(z["keys"])
    out = {a: np.concatenate(parts[a]) for a in arrays}
    out["keys"] = np.concatenate(keys)
    n = len(out["keys"])
    for a in arrays:
        if len(out[a]) != n:
            raise ValueError(f"{stem}: {a} has {len(out[a])} rows against {n} keys")
    if len(set(map(str, out["keys"]))) != n:
        raise ValueError(f"{stem}: duplicate keys across shards")
    out["y"] = labels_of(out["keys"])
    out["video"] = videos_of(out["keys"])
    out["source"] = sources_of(out["keys"])
    return out


def available() -> dict:
    """What is cached right now, as stem -> frame count, for progress reporting."""
    seen = {}
    for p in sorted(CACHE.glob("*.npz")):
        stem = re.sub(r"\.s\d+of\d+$", "", p.stem)
        with np.load(p, allow_pickle=True) as z:
            seen[stem] = seen.get(stem, 0) + len(z["keys"])
    return seen


def video_mean(x: np.ndarray, videos: np.ndarray):
    """Average frame-level values within each video. Returns (video order, means)."""
    order, inv = np.unique(videos, return_inverse=True)
    cnt = np.bincount(inv, minlength=len(order)).astype(np.float64)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return order, (np.bincount(inv, weights=x, minlength=len(order)) / cnt)
    out = np.stack([np.bincount(inv, weights=x[:, c], minlength=len(order)) / cnt
                    for c in range(x.shape[1])], axis=1)
    return order, out


def _selftest():
    # the rule that the leakage audit paid for: source and video come from the END
    k3 = "CDFv2/Celeb-real/id0_0000/frame_000.png"          # three deep
    k4 = "CDFv3valG/CDFv3valG__Celeb-real/id0/vid1/f.png"   # four deep
    assert source_of(k3) == "Celeb-real" and video_of(k3) == "CDFv2/Celeb-real/id0_0000"
    assert source_of(k4) == "id0" and video_of(k4) == "CDFv3valG/CDFv3valG__Celeb-real/id0/vid1"
    # a flat whole-image generator: the parent is a bare class name, so the frame is the video
    assert video_of("DF40/ddim-cdf/fake/00001.png") == "DF40/ddim-cdf/fake/00001"
    # the collision the rule exists to prevent: same basename, different videos
    assert video_of("DF40/x-real/id0_0000/f.png") != video_of("DF40/x-fake/id0_0000/f.png")
    assert label_of(k3) == 0

    for s, want in [("Celeb-real", 0), ("Celeb-synthesis", 1), ("YouTube-real", 0),
                    ("FFpp-real", 0), ("Deepfakes", 1), ("NeuralTextures", 1),
                    ("adv-real", 0), ("fake-dffs", 1), ("CollabDiff-real", 0), ("blendface-cdf", 1)]:
        assert label_of(f"ds/{s}/vid/f.png") == want, s

    vids = np.array(["a", "a", "b", "b", "b"])
    o, m = video_mean(np.array([1.0, 3.0, 0.0, 0.0, 3.0]), vids)
    assert list(o) == ["a", "b"] and np.allclose(m, [2.0, 1.0])
    o, m = video_mean(np.array([[1.0, 2.0], [3.0, 4.0], [0, 0], [0, 0], [3.0, 6.0]]), vids)
    assert np.allclose(m, [[2.0, 3.0], [1.0, 2.0]])

    print("data.py self-test OK")
    have = available()
    if have:
        print(f"  cached so far: {len(have)} stems, {sum(have.values())} frames")
        for k, v in sorted(have.items()):
            print(f"    {k:28s} {v:8d}")
    else:
        print("  cache is still empty; extraction is running")


if __name__ == "__main__":
    _selftest()

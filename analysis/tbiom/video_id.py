#!/usr/bin/env python3
"""THE video-identity rule. One implementation, guarded by a regression test.

    python analysis/tbiom/video_id.py --check      # run the regression check

WHY THIS FILE EXISTS AS ITS OWN MODULE
--------------------------------------
Getting video identity wrong does not raise. It silently regroups frames, and the resulting
AUROC is a real number computed over the wrong units — which is why it has twice reversed a
scientific conclusion in this project:

  * Stage 1 (DF40): grouping on the directory BASENAME merged a real video and a fake one that
    shared the name `id0_0000`, because DF40 borrows its authentic halves. `max(label)` then
    depended on which frames each export happened to sample.
  * Step 1 (DiCoME): trusting the h5 dataset's own `video` field grouped all of FF++ into 144
    groups named after the manipulation method (`Deepfakes`), Celeb-DF-v2 by identity rather
    than video (`id0`), and every DFD video into the single group `1`. That produced a fused
    CDFv2 AUROC of 0.9223 instead of 0.9731 and inverted the Step-1 verdict.

Both bugs are the same shape: a plausible-looking field or a shorter key. So the rule lives in
one place, is used by every consumer, and the expected video count per dataset is asserted below.
A change that regroups anything now fails a test instead of quietly changing a headline.

THE RULE
--------
A video is the DIRECTORY CONTAINING THE FRAME, as a full path. Not the directory's basename
(collides across a corpus's real and fake halves), and not any metadata field (mis-parses).
For flat whole-image methods — where the parent directory is a bare class name like `fake/`, so
every generated image would collapse into one "video" — the frame itself is the video.

This matches DiCoME's own `_save_video_level_report`, which groups on `path.split("/")[-2]`,
while being additionally safe against basename collisions.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

# A parent directory with one of these names is a CLASS, not a video.
CLASS_DIRS = {"real", "fake", "frames", "images"}

# Expected video counts, verified against the corpora's own dataset JSONs / split files.
# `None` means "not yet pinned" rather than "any value" — the check reports those separately so
# an unpinned dataset cannot pass silently.
EXPECTED_VIDEOS = {
    # our own eval_v1 / score_fpad exports
    "FaceForensics__": 700,      # 140 real + 4 x 140 manipulations
    "Celeb_DF_v2": 518,
    "Celeb_DF_v1": 100,
    "DFDCP": 654,
    "DFDC": 4704,
    "DeepFakeDetection": 3431,   # 363 real + 3,068 fake
    "Deepfake_Eval_2024": 814,
    "UADFV": 98,
    "VALmix": 1350,              # 450 each from CDFv2val / DFDCPval / DFEval24val
    # DiCoME's h5 exports, same corpora seen through its own split files
    "FFpp": 700,
    "CDFv2": 518,
    "DFD": 3431,
    "DFEval24": 814,
}


def video_id(keys: pd.Series) -> pd.Series:
    """The video each frame belongs to. See the module docstring for why this is the rule.

    Accepts either a bare frame path or DiCoME's `<h5 path>::<key>` form.
    """
    k = keys.astype(str).str.split("::", n=1).str[-1].str.rstrip("/")
    directory = k.str.rsplit("/", n=1).str[0]
    parent = directory.str.rsplit("/", n=1).str[-1]
    degenerate = parent.str.lower().isin(CLASS_DIRS)
    stem = k.str.replace(r"\.[A-Za-z0-9]+$", "", regex=True)
    return directory.where(~degenerate, stem)


def to_video_level(df: pd.DataFrame, prob_col: str, key_col: str = "key",
                   label_col: str = "label") -> pd.DataFrame:
    """Mean fake-probability per video, the aggregation the spec fixes."""
    out = pd.DataFrame({"video": video_id(df[key_col]),
                        "p": df[prob_col].to_numpy(),
                        "y": df[label_col].to_numpy()})
    return out.groupby("video", as_index=False).agg(p=("p", "mean"), y=("y", "max"))


def check(verbose: bool = True) -> int:
    """Regression check: the rule's behaviour, and video counts on every export on disk."""
    import glob
    from pathlib import Path

    failures, unpinned, checked = [], [], 0

    # --- behavioural cases, independent of what happens to be on disk ---------------------
    cases = [
        # (input, expected, why)
        ("/d/Celeb-DF-v2/Celeb-real/frames/id0_0000/012.png",
         "/d/Celeb-DF-v2/Celeb-real/frames/id0_0000",
         "ordinary video: the containing directory"),
        ("/d/df40/test/danet/cdf/frames/id0_0000/012.png",
         "/d/df40/test/danet/cdf/frames/id0_0000",
         "same BASENAME as the real above, must stay distinct"),
        ("/d/df40/test/stargan/fake/00931.jpg",
         "/d/df40/test/stargan/fake/00931",
         "flat whole-image method: parent is a class, so the frame IS the video"),
        ("data/h5/FFpp.h5::FFpp/Deepfakes/Deepfakes__000_003/000.png",
         "FFpp/Deepfakes/Deepfakes__000_003",
         "DiCoME's <h5>::<key> form, and NOT the `Deepfakes` method name"),
    ]
    got = video_id(pd.Series([c[0] for c in cases]))
    for (inp, want, why), have in zip(cases, got):
        checked += 1
        if have != want:
            failures.append(f"{why}\n      in   {inp}\n      want {want}\n      got  {have}")

    # the collision case, stated as the property that matters
    pair = video_id(pd.Series([cases[0][0], cases[1][0]]))
    checked += 1
    if pair.iloc[0] == pair.iloc[1]:
        failures.append("a real and a fake video sharing a basename collapsed into one video")

    # --- video counts on every export present ---------------------------------------------
    roots = ["logs/tbiom/crossdataset/*/", "logs/tbiom/ffpp_df40_eval/*/*/",
             "logs/tbiom/valmix/*/", "logs/tbiom/step1/*.csv"]
    seen: dict[str, set] = {}
    for pattern in roots:
        for path in glob.glob(pattern):
            p = Path(path)
            files = [p] if p.suffix == ".csv" else sorted(p.glob("*.parquet"))
            if not files:
                continue
            name = next((d for d in EXPECTED_VIDEOS if d in p.name), None)
            if name is None:
                continue
            try:
                frames = pd.concat(
                    [pd.read_csv(f) if f.suffix == ".csv" else pd.read_parquet(f)
                     for f in files], ignore_index=True)
            except Exception as exc:                      # noqa: BLE001
                failures.append(f"{p}: unreadable ({exc})")
                continue
            if "key" not in frames:
                continue
            n = video_id(frames["key"]).nunique()
            seen.setdefault(name, set()).add((str(p), n))

    for name, entries in sorted(seen.items()):
        want = EXPECTED_VIDEOS.get(name)
        for src, n in sorted(entries):
            checked += 1
            if want is None:
                unpinned.append(f"{name}: {n} videos ({src})")
            elif n != want:
                failures.append(f"{name}: expected {want} videos, got {n}  ({src})")

    if verbose:
        print(f"checked {checked} cases across {len(seen)} datasets")
        for u in unpinned:
            print(f"  UNPINNED {u}")
        for f in failures:
            print(f"  FAIL {f}")
    if failures:
        print(f"\n{len(failures)} FAILURE(S) — video grouping is wrong, and a wrong grouping "
              f"silently changes every downstream AUROC rather than raising.")
        return 1
    print("\nall video-identity checks passed")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    sys.exit(check() if a.check else 0)

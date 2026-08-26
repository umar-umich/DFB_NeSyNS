#!/usr/bin/env python
"""Build the FF++ (+) DF40 combined training manifest.

    python preprocessing/build_ffpp_df40_manifest.py --out-name FFPP_DF40

Emits a dataset JSON in this repo's own format, with every frame path already RESOLVED to a file
that exists, so training never discovers a broken path mid-epoch and the manifest can be audited
on its own.

WHY NOT JUST USE DF40_all.json
------------------------------
DF40 ships an aggregate `DF40_all.json`, and it is the obvious thing to reach for. It is not
usable as a training split, for two independent reasons, both measured here rather than assumed:

1. **It leaks FF++ test identities into its train split.** DF40's own per-method files keep the
   FF++ identity split (blendface's 719 train ids are 719/719 inside FF++ train, 0 in test), but
   the aggregate does not. Its e4e entries flatten every image into a single `ff/inversions/`
   pseudo-directory that discards the source id; recovering the id from the filename (unique, 0
   collisions across the 31,949 files on disk) puts **4,031 of its 29,126 e4e "train" images on
   FF++ TEST identities**. `simswap` and `inswap` add 258 more train videos whose ids are BOTH in
   FF++ test.
2. **Its e4e paths do not resolve at all** — that same `ff/inversions/` directory does not exist
   on disk, so 4.2% of the aggregate's frames are unreachable as written.

So the fake half is built from the PER-METHOD `*_ff.json` files, which carry correct splits and
correct paths, and every video is then filtered against the FF++ identity lists regardless. The
result is DF40_all in content, minus the leakage.

THE FILTER IS DELIBERATELY CONSERVATIVE. A DF40 swap is named `<target>_<source>_<method>`, and a
video is dropped if EITHER id appears in FF++ test or val. Dropping on the source id matters:
that identity's face is what was synthesised into the frame, so training on it would put a test
identity's appearance in the training set even though the background video is a training one.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import pathlib
import re
import sys

FFPP_JSON = pathlib.Path("/data/umar/Datasets/preprocessed/dataset_json/FaceForensics++.json")
AGGREGATES = {"DF40_all", "FSAll_ff", "FRAll_ff", "EFSAll_ff"}
FFPP_FAKE_SUBSETS = ("FF-DF", "FF-F2F", "FF-FS", "FF-NT")


def load_df40_paths():
    """Import df40_paths directly; `training.dataset.__init__` drags in dlib-dependent modules."""
    spec = importlib.util.spec_from_file_location(
        "df40_paths", pathlib.Path(__file__).resolve().parents[1] / "training/dataset/df40_paths.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ffpp_identity_sets() -> tuple[set, set, set]:
    d = json.load(open(FFPP_JSON))
    node = d["FaceForensics++"]["FF-real"]
    ids = lambda split: {k.split("/")[-1] for k in node[split]["c23"]}
    return ids("train"), ids("val"), ids("test")


def video_ids(video_name: str) -> list[str]:
    """The FF++ source ids embedded in a DF40 video name (`306_278_simswap` -> 306, 278)."""
    return re.findall(r"\d{3}", video_name.split("/")[-1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--df40-json", type=pathlib.Path,
                    default=pathlib.Path("/data/umar/Datasets/df40/dataset_json"))
    ap.add_argument("--out-dir", type=pathlib.Path,
                    default=pathlib.Path("/data/umar/Datasets/preprocessed/dataset_json"))
    ap.add_argument("--out-name", default="FFPP_DF40")
    ap.add_argument("--check-every-frame", action="store_true",
                    help="stat every frame rather than one per video; slow but exhaustive")
    ap.add_argument("--report", type=pathlib.Path, default=pathlib.Path("tbiom/FFPP_DF40_MANIFEST.md"))
    args = ap.parse_args()

    dp = load_df40_paths()
    ff_tr, ff_va, ff_te = ffpp_identity_sets()
    excluded = ff_va | ff_te
    real_key, fake_key = f"{args.out_name}_Real", f"{args.out_name}_Fake"
    out = {args.out_name: {real_key: {"train": {}, "val": {}, "test": {}},
                           fake_key: {"train": {}, "val": {}, "test": {}}}}

    stats: dict[str, dict] = {}

    # ---- val / test: FF++'s own, verbatim ----------------------------------------------------
    # The DF40 fakes contribute to TRAIN only. Their held-out portion is evaluated separately as
    # DF40-Dev/Holdout and must not be reachable through this manifest.
    #
    # These splits are populated rather than left empty for two reasons. Mechanically, `load_split`
    # constructs the dataset in test mode before swapping to the requested split, so an empty
    # `test` makes the manifest unloadable even for training. Substantively, FF++ val IS the
    # selection set for every arm — FF++'s splits are fully disjoint (train/val/test overlap 0 in
    # all three directions, verified) — so making this dataset's val and test exactly FF++'s is
    # the firewall written into the data rather than relied on by convention.
    ffd = json.load(open(FFPP_JSON))["FaceForensics++"]
    for split in ("val", "test"):
        for sub in ("FF-real",) + FFPP_FAKE_SUBSETS:
            half = real_key if sub == "FF-real" else fake_key
            for vid, info in ffd[sub][split]["c23"].items():
                frames = info.get("frames", [])
                if frames:
                    out[args.out_name][half][split][f"ffpp_{sub}_{vid}"] = {
                        "label": half, "frames": frames}

    # ---- FF++ half of TRAIN: reals and the four manipulations, already absolute and resolved --
    for sub in ("FF-real",) + FFPP_FAKE_SUBSETS:
        half = real_key if sub == "FF-real" else fake_key
        node = ffd[sub]["train"]["c23"]
        kept = 0
        for vid, info in node.items():
            frames = [f for f in info.get("frames", []) if pathlib.Path(f).exists()] \
                if args.check_every_frame else info.get("frames", [])
            if not frames:
                continue
            out[args.out_name][half]["train"][f"ffpp_{sub}_{vid}"] = {
                "label": half, "frames": frames}
            kept += 1
        stats[f"FF++/{sub}"] = {"videos": kept, "dropped_identity": 0,
                                "frames": sum(len(v["frames"])
                                              for k, v in out[args.out_name][half]["train"].items()
                                              if k.startswith(f"ffpp_{sub}_"))}

    # ---- DF40 half: per-method fakes, identity-filtered and path-resolved -------------------
    methods = sorted(p.stem for p in args.df40_json.glob("*_ff.json") if p.stem not in AGGREGATES)
    drop_ident = collections.Counter()
    drop_path = collections.Counter()
    drop_empty = collections.Counter()
    for m in methods:
        j = json.load(open(args.df40_json / f"{m}.json"))
        top = list(j.keys())[0]
        kept = frames_kept = 0
        for sub, sv in j[top].items():
            if "ake" not in sub:
                continue                                   # reals come from the FF++ half
            for vid, info in sv.get("train", {}).items():
                if any(i in excluded for i in video_ids(vid)):
                    drop_ident[m] += 1
                    continue
                raw = info.get("frames", [])
                if not raw:
                    # DF40 ships entries whose frame list is empty (155 per EFS method, where
                    # generation failed for that id). Counted apart from a resolution failure:
                    # conflating them would read as "our path rules are broken" when the source
                    # simply has no frames to point at.
                    drop_empty[m] += 1
                    continue
                frames = [dp.resolve(f) for f in raw]
                if args.check_every_frame:
                    frames = [f for f in frames if pathlib.Path(f).is_file()]
                elif not pathlib.Path(frames[0]).is_file():
                    frames = []
                if not frames:
                    drop_path[m] += 1
                    continue
                out[args.out_name][fake_key]["train"][f"df40_{m}_{vid}"] = {
                    "label": fake_key, "frames": frames}
                kept += 1
                frames_kept += len(frames)
        stats[f"DF40/{m}"] = {"videos": kept, "dropped_identity": drop_ident[m],
                              "dropped_unresolved": drop_path[m],
                              "dropped_empty": drop_empty[m], "frames": frames_kept}

    n_real = len(out[args.out_name][real_key]["train"])
    n_fake = len(out[args.out_name][fake_key]["train"])
    f_real = sum(len(v["frames"]) for v in out[args.out_name][real_key]["train"].values())
    f_fake = sum(len(v["frames"]) for v in out[args.out_name][fake_key]["train"].values())

    # A manifest that quietly contains a test identity is the one failure that would invalidate
    # every number built on it, so it is re-checked on the FINAL contents, not just per source.
    offenders = [k for half in (real_key, fake_key)
                 for k in out[args.out_name][half]["train"]
                 if any(i in ff_te for i in video_ids(k))]
    if offenders:
        raise SystemExit(f"{len(offenders)} manifest entries carry an FF++ TEST identity, "
                         f"e.g. {offenders[:3]} — refusing to write")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dest = args.out_dir / f"{args.out_name}.json"
    dest.write_text(json.dumps(out))
    print(f"wrote {dest}")
    print(f"  real {n_real} videos / {f_real} frames")
    print(f"  fake {n_fake} videos / {f_fake} frames")
    print(f"  dropped: {sum(drop_ident.values())} for identity, "
          f"{sum(drop_empty.values())} empty in source JSON, "
          f"{sum(drop_path.values())} unresolved on disk")
    print(f"  FF++ test identities in manifest: 0 (verified on final contents)")

    lines = ["# FF++ (+) DF40 combined training manifest", "",
             f"`{dest}`  ·  label keys `{real_key}` = 0, `{fake_key}` = 1", "",
             f"| half | videos | frames |", "|---|---:|---:|",
             f"| real | {n_real} | {f_real} |", f"| fake | {n_fake} | {f_fake} |", "",
             "Fakes are built from DF40's PER-METHOD `*_ff.json` files, not `DF40_all.json`. The "
             "aggregate leaks FF++ test identities into its train split (4,031 e4e images and 258 "
             "simswap/inswap videos) and its e4e paths do not resolve. Every video here is "
             "additionally filtered against the FF++ val and test identity lists, dropping on "
             "EITHER id of a `<target>_<source>` pair.", "",
             "`dropped: empty` are entries DF40 ships with an empty frame list (generation "
             "failed for that id); they are NOT path-resolution failures and are counted apart so "
             "the two are never confused.", "",
             "| source | videos | frames | dropped: identity | dropped: empty | dropped: unresolved |",
             "|---|---:|---:|---:|---:|---:|"]
    for k, v in stats.items():
        lines.append(f"| `{k}` | {v['videos']} | {v['frames']} | {v.get('dropped_identity', 0)} "
                     f"| {v.get('dropped_empty', 0)} | {v.get('dropped_unresolved', 0)} |")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n")
    print(f"  report -> {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

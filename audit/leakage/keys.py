#!/usr/bin/env python3
"""Shared identifier rules for the leakage audit. Read-only; no model or fusion code is touched.

A manifest line is `<dataset>/<subset>/<video>/<frame>.png`, so the video key is the third path component. From that
video key this module derives the two coarser keys the audit needs:

  SOURCE VIDEO -- the original footage a clip was made from. A face-swap fake derives from two sources, and both are
                  returned, because either one appearing in training is a leak.
  IDENTITY     -- the person, where the dataset encodes it. Returned as "not available" otherwise, never guessed.

Every rule below is derived from the actual manifest contents (verified by `python audit/leakage/keys.py`), not from
dataset documentation. Where a dataset's naming does not support a level, that is recorded rather than approximated.
"""
from __future__ import annotations
import re
from pathlib import Path

NA = ()          # "not available at this level"


def video_of(line: str) -> str:
    """The video directory is always the component immediately before the frame file.

    Depth differs between pools -- `VALmix/<subset>/<video>/<frame>` is three deep while
    `VALmixV2/<component>/<subset>/<video>/<frame>` is four -- so indexing from the front silently returns a subset
    name for the deeper layout. Indexing from the back is correct for both."""
    return line.strip().split("/")[-2]


def dataset_of(line: str) -> str:
    return line.strip().split("/")[0]


# Which naming rule applies to a line, once the development-pool wrappers are unwrapped. The wrapper directory says
# where the videos came from, and that is what decides the source/identity rule.
ORIGIN = [(re.compile(r"^VALmixV2/CDFv3valG/"), "CDFv3"), (re.compile(r"^VALmixV2/FFIWval/"), "FFIW"),
          (re.compile(r"^VALmixV2/DF40train/"), "DF40-ffpair"), (re.compile(r"^VALmixV2/DS\d+val/"), "in-the-wild"),
          (re.compile(r"^VALmix/CDFv2val-"), "CDFv2"), (re.compile(r"^VALmix/FFppval-"), "FFpp"),
          (re.compile(r"^VALmix/"), "in-the-wild")]


def origin_of(line: str) -> str:
    for rx, name in ORIGIN:
        if rx.match(line.strip()):
            return name
    return dataset_of(line)


def _ffpp_pair(tail: str):
    """FF++ video names: `001_870` is a fake from sources 001 and 870; `004` is real source 004."""
    m = re.fullmatch(r"(\d{3})_(\d{3})", tail)
    if m:
        return (m.group(1), m.group(2))
    return (tail,) if re.fullmatch(r"\d{3}", tail) else NA


def _celeb(tail: str):
    """Celeb-DF: `idX_idY_NNNN` fake (identities X and Y), `idX_NNNN` real, `NNNNN` YouTube-real."""
    m = re.fullmatch(r"(id\d+)_(id\d+)_(\d+)", tail)
    if m:
        return (m.group(1), m.group(2)), (f"{m.group(1)}_{m.group(3)}",)
    m = re.fullmatch(r"(id\d+)_(\d+)", tail)
    if m:
        return (m.group(1),), (tail,)
    return NA, (tail,)


# Identifier namespaces. `id10` in Celeb-DF and `id10` in IDForge are different people, and a bare-string
# intersection across datasets would report a leak that does not exist. Every key is therefore prefixed with the
# family whose naming rule produced it, and only keys in the same family can ever match.
NS = {"FFpp": "ffpp", "FFpp_train": "ffpp", "FFpp_val": "ffpp", "DF40-ffpair": "ffpp",
      "CDFv2": "celeb", "CDFv3": "celeb", "DFDM": "celeb", "DF40": "celeb-or-ffpp",
      "IDForge": "idforge", "DFD": "dfd", "KoDF": "kodf", "FFIW": "ffiw", "DFDC": "dfdc",
      "DFDCP": "dfdcp", "DFEval24": "dfeval24", "in-the-wild": "itw"}


def _ns(dataset: str, video: str) -> str:
    """DF40 mixes two families in one dataset, so its namespace is decided per video by the naming shape."""
    base = NS.get(dataset, dataset.lower())
    if base != "celeb-or-ffpp":
        return base
    return "ffpp" if re.search(r"__\d{3}_\d{3}$|__\d{3}$", video) or re.fullmatch(r"\d{3}(_\d{3})?", video) \
        else "celeb"


def sources_and_identity(dataset: str, video: str):
    """Returns (source_videos, identities), each key prefixed by its namespace. Either may be NA."""
    v = re.sub(r"^cdf[23]-", "", video)              # VALmix v1 prefixes Celeb-DF videos with the set it came from
    d = dataset
    ns = _ns(dataset, video)
    tag = lambda t: tuple(f"{ns}:{x}" for x in t)

    if d == "DF40-ffpair":                           # DF40's FF++-sourced training videos are named `AAA_BBB`
        return tag(_ffpp_pair(v)), NA
    if "__" in v:                                    # generator/manipulation prefix, e.g. `danet-cdf__id0_id16_0003`
        head, _, v = v.partition("__")
    else:
        head = ""

    if d.startswith("FFpp") or (d == "DF40" and head.endswith("-ff")) or d == "DF40-ff-real":
        return tag(_ffpp_pair(v)), NA                # FF++ gives no identity labels
    if d.startswith("CDF") or (d == "DF40" and head.endswith("-cdf")) or head == "DF40-cdf-real":
        ids, src = _celeb(v)
        return tag(src), tag(ids)
    if d == "DF40":                                  # deepfacelab and the shared real pools
        ids, src = _celeb(v)
        return tag(src), tag(ids)
    if d == "DFDM":                                  # `id0_id1_0000_4dfaker` -> Celeb-DF naming plus a method suffix
        core = re.sub(r"_\d?[a-zA-Z-]+\d*$", "", v)
        ids, src = _celeb(core)
        return tag(src), tag(ids)
    if d == "IDForge":                               # `id00_scene_0023_2_infoswap`
        m = re.match(r"(id\d+)_scene_(\d+)", v)
        return (tag((f"{m.group(1)}_scene_{m.group(2)}",)), tag((m.group(1),))) if m else (NA, NA)
    if d == "DFD":                                   # `01_02__exit_phone_room__HASH` / `01__exit_phone_room`
        m = re.match(r"(\d{2})(?:_(\d{2}))?", v)
        if m:
            ids = tuple(x for x in (m.group(1), m.group(2)) if x)
            return tag((v,)), tag(ids)
        return tag((v,)), NA
    if d == "KoDF":
        parts = v.split("_")
        return tag(tuple(p for p in parts[:2] if len(p) > 8) or (v,)), NA
    return tag((v,)), NA                             # FFIW, DFDC, DFDCP, DFEval24: video-level only


UNSUPPORTED_IDENTITY = {"FFpp", "FFpp_train", "FFpp_val", "FFIW", "DFDC", "DFDCP", "DFEval24", "KoDF"}


def manifest_videos(path: Path):
    """{video -> n_frames} for one manifest."""
    out: dict[str, int] = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            v = video_of(line)
            out[v] = out.get(v, 0) + 1
    return out


def manifest_keys(path: Path):
    """-> (videos {v: frames}, sources set, identities set), using each line's own origin rule."""
    vids, src, ids = {}, set(), set()
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        v = video_of(line)
        vids[v] = vids.get(v, 0) + 1
        s, i = sources_and_identity(origin_of(line), v)
        src.update(s); ids.update(i)
    return vids, src, ids


if __name__ == "__main__":                            # self-test against real manifest samples
    DEPTH = [("VALmixV2/CDFv3valG/Celeb-real/id0_0002/frame_0000.png", "id0_0002", "CDFv3"),
             ("VALmixV2/FFIWval/real/val_00000000/frame_0047.png", "val_00000000", "FFIW"),
             ("VALmixV2/DF40train/danet/002_006/000.png", "002_006", "DF40-ffpair"),
             ("VALmix/CDFv2val-Celeb-real/cdf2-id0_0005/000.png", "cdf2-id0_0005", "CDFv2"),
             ("CDFv2/Celeb-synthesis/id0_id16_0003/000.png", "id0_id16_0003", "CDFv2")]
    for line, ev, eo in DEPTH:
        gv, go = video_of(line), origin_of(line)
        good = gv == ev and go == eo
        print(("PASS " if good else "FAIL ") + f"depth/origin  {line[:52]:52s} -> {gv}, {go}")
    print()
    CASES = [("FFpp_train", "Deepfakes__001_870", ("ffpp:001", "ffpp:870"), NA),
             ("FFpp_val", "004", ("ffpp:004",), NA),
             ("CDFv2", "id0_id16_0003", ("celeb:id0_0003",), ("celeb:id0", "celeb:id16")),
             ("CDFv3", "id0_0001", ("celeb:id0_0001",), ("celeb:id0",)),
             ("DF40", "danet-cdf__id0_id16_0003", ("celeb:id0_0003",), ("celeb:id0", "celeb:id16")),
             ("DF40", "danet-ff__000_003", ("ffpp:000", "ffpp:003"), NA),
             ("DFDM", "id0_id1_0000_4dfaker", ("celeb:id0_0000",), ("celeb:id0", "celeb:id1")),
             ("IDForge", "id00_scene_0023_2_infoswap", ("idforge:id00_scene_0023",), ("idforge:id00",))]
    ok = True
    for ds, v, esrc, eid in CASES:
        s, i = sources_and_identity(ds, v)
        good = tuple(s) == tuple(esrc) and tuple(i) == tuple(eid)
        ok &= good
        print(("PASS " if good else "FAIL ") + f"{ds:10s} {v:32s} -> sources {s}, identities {i}")
    print("\nself-test:", "all rules behave as documented" if ok else "RULE MISMATCH")

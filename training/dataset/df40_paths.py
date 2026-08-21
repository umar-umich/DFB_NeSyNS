"""DF40 frame-path resolution (spec §21's DF40 family/method rows).

DF40's shipped JSONs store frame paths relative to a root directory that does not exist on this
machine — `deepfakes_detection_datasets/…` — and there is no single rewrite that works, because
DF40 borrows its authentic halves from other corpora and its generated halves are not laid out
uniformly:

    borrowed authentic   deepfakes_detection_datasets/<CORPUS>/…
                     ->  <DF40_ROOT>/real/<CORPUS>/…
    video methods        deepfakes_detection_datasets/DF40/<m>/<subset>/frames/<vid>/<f>.png
                     ->  <DF40_ROOT>/test/<m>/<subset>/frames/<vid>/<f>.png
    image methods        deepfakes_detection_datasets/DF40/<m>/<half>/<f>.jpg
                     ->  <DF40_ROOT>/test/<m>/<half>/<half>/<f>.jpg          (doubled directory)

So `resolve()` tries candidates and takes the one that exists. Deriving a single rule from one
method is how this went wrong the first time: the doubled-directory rule, generalised from
`stargan`, resolved only 21.6% of `danet_cdf` — its authentic half and none of its fakes — and the
run would have "succeeded" on a fifth of the data. That is why resolution is measured per method
rather than assumed to work.

Resolution is reported, never assumed
-------------------------------------
`verify()` reports the fraction of frames that resolve per method. A method below the threshold is
excluded with its rate recorded, so a DF40 number is never computed over a partially-missing
method without that being visible in the report.

Two things to know before reading DF40 results (§21)
---------------------------------------------------
* **`DF40_all.json` is unusable**: every entry sits under `train` and its `test` split is empty.
  Use the per-method JSONs, which is also what §21's "family/method breakdown" asks for.
* **DF40 mixes two different problems.** Its face-swap and reenactment methods (danet, facedancer,
  blendface, MRAA, …) are face manipulations of real footage, comparable to FF++/CDF. Its
  whole-image generators (MidJourney, StyleGAN2/3/XL, VQGAN, CollabDiff, DiT, SiT, RDDM) produce
  entire synthetic images at up to 1024x1024, not manipulated face crops. Pooling both into one
  "DF40 AUROC" averages over two tasks; `FACE_MANIPULATION` and `WHOLE_IMAGE_SYNTHESIS` below keep
  them separable so the report can say which is which.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DF40_ROOT = Path("/data/umar/Datasets/df40")
DF40_JSON_DIR = DF40_ROOT / "dataset_json"
RELATIVE_ROOT = "deepfakes_detection_datasets/"

# Some JSONs store the path with a leading `./`, and e4e's store the authors' full cluster path.
# Both mean the same relative location; stripped so one set of rules covers all of them.
ABSOLUTE_PREFIXES = (
    "/Youtu_Pangu_Security_Public/youtu-pangu-public/zhiyuanyan/",
    "./",
)

# Method directories whose on-disk name is not the JSON's method name. Every entry below was
# confirmed by listing the directory, not inferred from the name — `whichisreal` in particular
# ships under `whichfaceisreal`, and the extra nesting levels (`heygen/heygen_new`,
# `styleclip/styleclip`) are archive-extraction artifacts that no naming rule predicts.
METHOD_DIR_ALIASES = {
    "heygen": "heygen/heygen_new",
    "styleclip": "styleclip/styleclip",
    "whichisreal": "whichfaceisreal/whichfaceisreal",
    "rddm": "RDDM",
}

# DF40's Celeb-DF-driven halves name the SOURCE pool in the JSON (`cdf/Celeb-real/...`) but are
# stored under a directory that names what they are (`cdf/Fake_from_Celeb-real/...`). Note the
# capitalisation change on the second one: the archive writes `Youtube`, the JSON writes
# `YouTube`, so this cannot be a case-insensitive lookup on one rule.
CDF_SOURCE_DIRS = {
    "Celeb-real": "Fake_from_Celeb-real",
    "YouTube-real": "Fake_from_Youtube-real",
}

# §21 asks for a family/method breakdown. This is a COARSE two-way split by task type, not
# DF40's own four-family taxonomy (face swapping / face reenactment / entire face synthesis /
# face editing) — it separates "entire image is synthetic" from "real footage was manipulated",
# which is the distinction that makes a pooled AUROC misleading. Use DF40's taxonomy for the
# paper's family rows; use this to avoid averaging two different tasks together.
WHOLE_IMAGE_SYNTHESIS = {
    "CollabDiff", "DiT", "SiT", "RDDM", "MidJourney", "VQGAN", "StyleGAN2", "StyleGAN3",
    "StyleGANXL", "PixArt", "SDXL", "whichisreal",
}


def _normalise(path: str) -> str:
    """Strip the prefixes that stop a path being recognised as relative at all.

    `e4e_ff` writes `./deepfakes_detection_datasets/...` and `e4e_cdf` writes the authors'
    absolute cluster path. Both previously fell through `startswith(RELATIVE_ROOT)` and were
    returned verbatim, so BOTH e4e arms resolved at exactly 0.0 — not because the frames were
    missing (they are on disk) but because two characters at the front of the string sent the
    path down the "already absolute, another corpus" branch.
    """
    for prefix in ABSOLUTE_PREFIXES:
        if path.startswith(prefix):
            path = path[len(prefix):]
    return path


def candidates(path: str, root: Path = DF40_ROOT) -> list[str]:
    """Every plausible on-disk location for one DF40 frame path, most likely first.

    DF40's generated halves are NOT laid out uniformly, which is why this returns candidates
    instead of one rewrite:

        video methods (danet, fomm, ...)  DF40/<m>/<subset>/frames/<vid>/<f>.png
                                     ->   test/<m>/<subset>/frames/<vid>/<f>.png
        image methods (stargan, ...)      DF40/<m>/<half>/<f>.jpg
                                     ->   test/<m>/<half>/<half>/<f>.jpg      (doubled half)
        extracted archives                DF40/<m>/<half>/<f>.jpg
                                     ->   test/<m>/<m>/<half>/<f>.jpg         (doubled method)
        cdf-driven synthesis              DF40/<m>/cdf/Celeb-real/<vid>/<f>.png
                                     ->   test/<m>/cdf/Fake_from_Celeb-real/<vid>/<f>.png

    A single rule derived from one method silently loses the other's frames -- which is exactly
    what happened here: the doubled rule generalised from `stargan` resolved only 21.6% of
    `danet_cdf`, i.e. its authentic half and none of its fakes.
    """
    path = _normalise(path)
    if not path.startswith(RELATIVE_ROOT):
        return [path]                                  # already absolute, or another corpus
    tail = path[len(RELATIVE_ROOT):]
    out: list[str] = []
    if tail.startswith("DF40/"):
        inner = tail[len("DF40/"):]
        parts = inner.split("/")
        method, rest = parts[0], parts[1:]
        # Every spelling of the method directory worth trying, in order of likelihood.
        method_dirs = [method]
        alias = METHOD_DIR_ALIASES.get(method)
        if alias:
            method_dirs.insert(0, alias)
        method_dirs.append(f"{method}/{method}")       # archive extracted into its own name

        for mdir in method_dirs:
            out.append(str(root / "test" / mdir / "/".join(rest)))
            if len(rest) >= 2:                         # doubled-half image layout
                out.append(str(root / "test" / mdir / rest[0] / rest[0] / "/".join(rest[1:])))
            if len(rest) >= 2 and rest[1] in CDF_SOURCE_DIRS:
                renamed = [rest[0], CDF_SOURCE_DIRS[rest[1]], *rest[2:]]
                out.append(str(root / "test" / mdir / "/".join(renamed)))
    else:
        # Pre-DF40-namespace methods: `simswap/cdfv2/frames/...` is stored as
        # `test/simswap/cdf/frames/...`.
        parts = tail.split("/")
        if len(parts) >= 2 and parts[1] == "cdfv2":
            out.append(str(root / "test" / parts[0] / "cdf" / "/".join(parts[2:])))
        out.append(str(root / "real" / tail))           # borrowed authentic halves
        out.append(str(Path("/data/umar/Datasets/preprocessed") / tail))
    # de-duplicate, preserving order: several rules coincide for the simple layouts
    seen, unique = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def resolve(path: str, root: Path = DF40_ROOT) -> str:
    """First candidate that exists on disk; the first candidate otherwise.

    Returning an unresolved path rather than None keeps the caller's reporting honest: it counts
    what it could not find instead of quietly shortening the evaluation set.
    """
    options = candidates(path, root)
    for option in options:
        if os.path.isfile(option):
            return option
    return options[0]


def resolves(path: str, root: Path = DF40_ROOT) -> bool:
    return any(os.path.isfile(c) for c in candidates(path, root))


def available_methods(json_dir: Path = DF40_JSON_DIR) -> list[str]:
    """Per-method JSON names, excluding the unusable aggregate."""
    return sorted(p.stem for p in Path(json_dir).glob("*.json") if p.stem != "DF40_all")


def family_of(method: str) -> str:
    base = method.split("_")[0]
    return ("whole_image_synthesis" if base in WHOLE_IMAGE_SYNTHESIS
            else "face_manipulation")


def verify(method: str, json_dir: Path = DF40_JSON_DIR, root: Path = DF40_ROOT,
           split: str = "test", sample_per_video: int = 2) -> dict:
    """Resolution rate for one method, sampled rather than exhaustive.

    Sampling keeps the check cheap over 40 methods; a method whose paths are broken is broken
    uniformly, and a partial breakage still shows up as a rate below 1.0.
    """
    path = Path(json_dir) / f"{method}.json"
    if not path.is_file():
        return {"method": method, "status": "no json"}
    blob = json.loads(path.read_text())
    root_key = next(iter(blob))
    checked = resolved = n_videos = 0
    per_label: dict[str, int] = {}
    for label, sections in blob[root_key].items():
        section = sections.get(split) or {}
        per_label[label] = len(section)
        for video, info in section.items():
            frames = (info or {}).get("frames") or []
            n_videos += 1
            for frame in frames[:sample_per_video]:
                checked += 1
                resolved += resolves(frame, root)
    return {
        "method": method,
        "family": family_of(method),
        "status": "ok" if checked else "empty split",
        "videos": per_label,
        "n_videos": n_videos,
        "frames_checked": checked,
        "resolution_rate": (resolved / checked) if checked else 0.0,
    }


def usable_methods(min_rate: float = 0.99, **kwargs) -> tuple[list[dict], list[dict]]:
    """(usable, excluded) method reports, so exclusions are recorded rather than silent."""
    reports = [verify(m, **kwargs) for m in available_methods(kwargs.get("json_dir",
                                                                        DF40_JSON_DIR))]
    usable = [r for r in reports if r.get("status") == "ok" and r["resolution_rate"] >= min_rate]
    excluded = [r for r in reports if r not in usable]
    return usable, excluded


def label_dict_for(methods: list[str], json_dir: Path = DF40_JSON_DIR) -> dict[str, int]:
    """`{'danet_Real': 0, 'danet_Fake': 1, ...}` for the requested methods.

    The loader resolves a video's class through `config['label_dict']` and RAISES on an unknown
    key, and DF40 names its labels per method (`danet_Real`, `stargan_Fake`, …) — none of which
    are in the shared test config's 28 entries. Generated here and injected per run rather than
    added to `training/config/test_config.yaml`, because that file is shared with every other
    experiment in the repo and this is one evaluation's concern.

    Read from each JSON rather than pattern-built from the method name: the label keys do not
    always match the file stem (`danet_cdf.json` declares `danet_Real` / `danet_Fake`), so
    guessing them would raise for exactly the methods whose naming differs.
    """
    mapping: dict[str, int] = {}
    for method in methods:
        path = Path(json_dir) / f"{method}.json"
        if not path.is_file():
            continue
        blob = json.loads(path.read_text())
        for label in blob[next(iter(blob))]:
            lowered = label.lower()
            if lowered.endswith("real"):
                mapping[label] = 0
            elif lowered.endswith("fake"):
                mapping[label] = 1
            else:
                raise ValueError(
                    f"{method}: cannot tell whether label {label!r} is real or fake; refusing to "
                    f"guess, because a flipped label silently inverts that method's AUROC")
    return mapping


def remap_dataset(dataset, root: Path = DF40_ROOT) -> dict:
    """Rewrite an already-constructed dataset's image list in place.

    Returns a report. Frames that still do not resolve are DROPPED and counted: keeping them
    would raise mid-epoch on a missing file, and silently substituting another frame would make
    the evaluation set differ from what the report claims.
    """
    images, labels, dropped = [], [], 0
    for path, label in zip(dataset.image_list, dataset.label_list):
        candidate = resolve(path if isinstance(path, str) else path[0], root)
        if os.path.isfile(candidate):
            images.append(candidate)
            labels.append(label)
        else:
            dropped += 1
    if not images:
        raise SystemExit(
            "no DF40 frame resolved after remapping — check DF40_ROOT and the archive layout "
            "before trusting any DF40 number")
    dataset.image_list, dataset.label_list = images, labels
    dataset.data_dict = {"image": images, "label": labels}
    dataset._build_source_video_maps()
    return {"kept": len(images), "dropped": dropped,
            "resolution_rate": len(images) / (len(images) + dropped)}


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Report DF40 path-resolution rates per method")
    ap.add_argument("--json-dir", type=Path, default=DF40_JSON_DIR)
    ap.add_argument("--root", type=Path, default=DF40_ROOT)
    ap.add_argument("--split", default="test")
    ap.add_argument("--min-rate", type=float, default=0.99)
    args = ap.parse_args()

    reports = [verify(m, args.json_dir, args.root, args.split) for m in
               available_methods(args.json_dir)]
    ok = [r for r in reports if r.get("status") == "ok" and r["resolution_rate"] >= args.min_rate]
    bad = [r for r in reports if r not in ok]

    print(f"{len(reports)} DF40 method JSONs in {args.json_dir}\n")
    for group, title in ((ok, "USABLE"), (bad, "EXCLUDED")):
        print(f"--- {title} ({len(group)}) ---")
        for r in sorted(group, key=lambda r: r.get("method", "")):
            rate = r.get("resolution_rate")
            print(f"  {r['method']:24s} {r.get('family', '-'):22s} "
                  f"videos={r.get('n_videos', 0):5d} "
                  f"resolved={rate:.3f}" if rate is not None else
                  f"  {r['method']:24s} {r.get('status')}")
        print()
    families: dict[str, int] = {}
    for r in ok:
        families[r["family"]] = families.get(r["family"], 0) + 1
    print(f"usable by family: {families}")
    print("\nNote: whole_image_synthesis methods are full synthetic images, not manipulated face "
          "crops — report them separately from face_manipulation (§21).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

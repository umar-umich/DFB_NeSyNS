#!/usr/bin/env python
"""Rebuild VALmix as a video-id manifest over OUR OWN preprocessed tree.

    python preprocessing/build_valmix_manifest.py

VALmix is the diverse validation split the brief selects checkpoints on: 450 videos each from
Celeb-DF-v2, DFDCP and Deepfake-Eval-2024, balanced so the macro-average across domains is not
dominated by one corpus. It exists only as a 5 GB HDF5 in a different repo
(`DiCoME/eval_adaptation/data/h5/VALmix.h5`), which nothing in this repo can read, so nothing
here has ever actually used it.

The video LIST is taken from that HDF5, since that is where the split was defined; the FRAMES are
then resolved against `/data/umar/Datasets/preprocessed/`. Those pixels were previously verified
byte-identical to the HDF5's (8 frames across 4 videos, mean absolute difference 0.000 in RGB
order), so this changes nothing about what selection sees while removing a 5 GB cross-repo
dependency and making the split inspectable and diffable.

WHY VALmix AND NOT THE CORPORA'S OWN val SPLITS. For Celeb-DF-v2 and DFDCP the shipped `val`
split IS the shipped `test` split (518/518 and 652/652 verified), and Deepfake-Eval-2024 ships no
val at all. Selecting a checkpoint on those would be selecting on test. VALmix is carved from
videos absent from all three test lists, which this script re-verifies rather than trusting.

WHAT IT COSTS. Adopting VALmix makes Celeb-DF-v2, DFDCP and Deepfake-Eval-2024 domain-seen rather
than strict zero-shot. Celeb-DF-v1, Celeb-DF-v3, DFD, DFDC and UADFV stay strictly zero-shot.
Any claim of zero-shot generalisation belongs to the second group.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

PRE = pathlib.Path("/data/umar/Datasets/preprocessed")
H5 = pathlib.Path("/data/umar/Repos/DiCoME/eval_adaptation/data/h5/VALmix.h5")

# group name -> (relative frames directory, label). Labels are explicit, never inferred from the
# group name: `CDFv2val-Celeb-synthesis` is fake while `CDFv2val-Celeb-real` is real, and a rule
# keyed on the substring "real" would silently mislabel `DFDCPval-real`'s siblings.
GROUPS = {
    "CDFv2val-Celeb-real":       ("Celeb-DF-v2/Celeb-real/frames", 0),
    "CDFv2val-Celeb-synthesis":  ("Celeb-DF-v2/Celeb-synthesis/frames", 1),
    "CDFv2val-YouTube-real":     ("Celeb-DF-v2/YouTube-real/frames", 0),
    "DFDCPval-method_A":         ("DFDCP/method_A/frames", 1),
    "DFDCPval-method_B":         ("DFDCP/method_B/frames", 1),
    "DFDCPval-real":             ("DFDCP/original_videos/frames", 0),
    "DFEval24val-fake":          ("Deepfake-Eval-2024/frames", 1),
    "DFEval24val-real":          ("Deepfake-Eval-2024/frames", 0),
}
DOMAIN_OF = {g: g.split("-")[0] for g in GROUPS}

TEST_JSONS = {
    "CDFv2val": ("Celeb-DF-v2.json", "Celeb-DF-v2"),
    "DFDCPval": ("DFDCP.json", "DFDCP"),
    "DFEval24val": ("Deepfake-Eval-2024.json", "Deepfake-Eval-2024"),
}


def test_video_ids(json_name: str, root_key: str) -> set[str]:
    d = json.load(open(PRE / "dataset_json" / json_name))
    out: set[str] = set()
    for sub in d[root_key].values():
        for vid in sub.get("test", {}):
            out.add(vid.split("/")[-1])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--h5", type=pathlib.Path, default=H5)
    ap.add_argument("--out-dir", type=pathlib.Path, default=PRE / "dataset_json")
    ap.add_argument("--out-name", default="VALmix")
    ap.add_argument("--report", type=pathlib.Path, default=pathlib.Path("tbiom/VALMIX.md"))
    args = ap.parse_args()

    import h5py

    if not args.h5.is_file():
        raise SystemExit(f"{args.h5} not found — it defines the split and cannot be regenerated "
                         f"here. Recover it before rebuilding the manifest.")

    real_key, fake_key = f"{args.out_name}_Real", f"{args.out_name}_Fake"
    out = {args.out_name: {real_key: {"train": {}, "val": {}, "test": {}},
                           fake_key: {"train": {}, "val": {}, "test": {}}}}

    per_group: dict[str, dict] = {}
    missing: list[str] = []
    with h5py.File(args.h5, "r") as f:
        root = f[args.out_name]
        unknown = set(root.keys()) - set(GROUPS)
        if unknown:
            raise SystemExit(f"unmapped VALmix groups {sorted(unknown)} — every group needs an "
                             f"explicit directory and label, never a guessed one")
        for group, (reldir, label) in GROUPS.items():
            if group not in root:
                raise SystemExit(f"group {group} absent from {args.h5}")
            half = fake_key if label == 1 else real_key
            kept = frames_kept = 0
            for raw in root[group].keys():
                vid = raw.split("-", 1)[1] if "-" in raw else raw   # strip the `cdf2-`/`dfdcp-`
                vdir = PRE / reldir / vid
                frames = sorted(str(p) for p in vdir.glob("*.png")) if vdir.is_dir() else []
                if not frames:
                    missing.append(f"{group}/{vid}")
                    continue
                entry = {"label": half, "frames": frames}
                # Written under BOTH keys, and `test` is a LOADER ARTIFACT, not a claim.
                # `abstract_dataset` supports only the modes `train` and `test` — it raises
                # NotImplementedError on `val` — and `load_split` constructs in test mode before
                # it can swap splits. So a val-only manifest is simply unloadable in this
                # codebase. VALmix remains a VALIDATION set: it selects checkpoints and nothing
                # else, and a number computed on it must never be reported as a test result.
                # The two keys hold identical content, so nothing differs but the name.
                out[args.out_name][half]["val"][f"{DOMAIN_OF[group]}_{vid}"] = entry
                out[args.out_name][half]["test"][f"{DOMAIN_OF[group]}_{vid}"] = dict(entry)
                kept += 1
                frames_kept += len(frames)
            per_group[group] = {"videos": kept, "frames": frames_kept,
                                "label": label, "dir": reldir}

    if missing:
        raise SystemExit(f"{len(missing)} VALmix videos are not on disk, e.g. {missing[:3]}. "
                         f"Selection must read every video the split defines, so this is refused "
                         f"rather than silently shrinking the set.")

    # Re-verify the brief's requirement on the FINAL manifest, not on the HDF5 it came from.
    overlaps = {}
    for domain, (jname, rkey) in TEST_JSONS.items():
        te = test_video_ids(jname, rkey)
        mine = {k.split("_", 1)[1] for half in (real_key, fake_key)
                for k in out[args.out_name][half]["val"] if k.startswith(domain + "_")}
        overlaps[domain] = sorted(mine & te)
    bad = {d: v for d, v in overlaps.items() if v}
    if bad:
        raise SystemExit(f"VALmix overlaps a TEST split: "
                         f"{ {d: (len(v), v[:3]) for d, v in bad.items()} }. Selection would then "
                         f"be selection on test; refusing to write.")

    n_real = len(out[args.out_name][real_key]["val"])
    n_fake = len(out[args.out_name][fake_key]["val"])
    f_real = sum(len(v["frames"]) for v in out[args.out_name][real_key]["val"].values())
    f_fake = sum(len(v["frames"]) for v in out[args.out_name][fake_key]["val"].values())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dest = args.out_dir / f"{args.out_name}.json"
    dest.write_text(json.dumps(out))
    print(f"wrote {dest}")
    print(f"  real {n_real} videos / {f_real} frames")
    print(f"  fake {n_fake} videos / {f_fake} frames")
    for d, v in overlaps.items():
        print(f"  {d}: 0 overlap with its test split (checked {len(v)} collisions)")

    by_domain: dict[str, dict] = {}
    for g, s in per_group.items():
        dom = by_domain.setdefault(DOMAIN_OF[g], {"videos": 0, "frames": 0})
        dom["videos"] += s["videos"]
        dom["frames"] += s["frames"]

    lines = ["# VALmix — the diverse validation split", "",
             f"`{dest}`  ·  label keys `{real_key}` = 0, `{fake_key}` = 1  ·  split key `val`", "",
             "Rebuilt as a video-id manifest over `/data/umar/Datasets/preprocessed/`. The video "
             "LIST comes from the original HDF5, since that is where the split was defined; the "
             "FRAMES are ours. Those pixels were verified byte-identical, so selection sees the "
             "same images while the 5 GB cross-repo dependency goes away.", "",
             "| domain | videos | frames |", "|---|---:|---:|"]
    for dom, s in by_domain.items():
        lines.append(f"| {dom} | {s['videos']} | {s['frames']} |")
    lines += [f"| **total** | **{n_real + n_fake}** | **{f_real + f_fake}** |", "",
              f"Real / fake: {n_real} / {n_fake}.", "",
              "## Why this and not each corpus's own `val`", "",
              "For Celeb-DF-v2 and DFDCP the shipped `val` split IS the shipped `test` split "
              "(518/518 and 652/652), and Deepfake-Eval-2024 ships no `val` at all. Selecting on "
              "those would be selecting on test. Every video here was re-checked against its "
              "corpus's test list on the FINAL manifest: zero overlap in all three domains, and "
              "the builder refuses to write otherwise.", "",
              "## The `test` key is a loader artifact", "",
              "`abstract_dataset` supports only the modes `train` and `test` — it raises "
              "NotImplementedError on `val` — and `load_split` constructs in test mode before it "
              "can swap splits, so a val-only manifest is unloadable in this codebase. The `val` "
              "and `test` keys therefore hold IDENTICAL content. VALmix is a validation set: it "
              "selects checkpoints and nothing else, and a number computed on it must never be "
              "reported as a test result.", "",
              "## What adopting it costs", "",
              "| corpus | status |", "|---|---|",
              "| Celeb-DF-v2, DFDCP, Deepfake-Eval-2024 | **domain-seen** — not strict zero-shot |",
              "| Celeb-DF-v1, Celeb-DF-v3, DFD, DFDC, UADFV | strict zero-shot |", "",
              "Any zero-shot claim in the paper belongs to the second row.", "",
              "## Per group", "", "| group | label | videos | frames | directory |",
              "|---|---|---:|---:|---|"]
    for g, s in per_group.items():
        lines.append(f"| `{g}` | {'fake' if s['label'] else 'real'} | {s['videos']} | "
                     f"{s['frames']} | `{s['dir']}` |")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n")
    print(f"  report -> {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

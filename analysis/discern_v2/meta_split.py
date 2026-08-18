#!/usr/bin/env python3
"""§12 — the VAL_select / VAL_meta split, and the 5-fold cross-fit inside VAL_meta.

    VAL_select  60%   expert checkpoint selection (§11)
    VAL_meta    40%   applicability gates + risk model, via 5-fold cross-fitting (§12, §18)

Built from FF++ **validation** videos only. The split is deterministic, video-level, and
source-disjoint, and the folds inside VAL_meta are source-disjoint too, because §19 lists exactly
that as a leakage guard.

Source-disjointness is stricter here than the ported helper
-----------------------------------------------------------
DISCERN's `_extract_source_video` maps a FF++ fake like `802_885` to `802` — its target identity.
That is right for GenD-style pairing, but it is not enough for a leakage guard: `802_885` also
contains the *source* identity 885, so grouping on `802` alone would let real video `885` (or the
fake `885_xxx`) land in the other partition while the same face appears in both. Here every id in
a video name joins one group, computed by union-find, so a face cannot cross the partition
boundary through the second identity. Some groups therefore become large chains — that is the
honest cost of the guarantee, and the group-size distribution is reported so it is visible rather
than surprising.

Stratification (§12: "where possible")
--------------------------------------
Groups, not videos, are the unit of assignment, so exact per-method proportions are not always
achievable — a single group can contain several manipulations. Groups are stratified by their
manipulation signature and assigned largest-first within each stratum to whichever partition is
furthest below its target share. That keeps both the 60/40 split and the per-method balance close
without ever splitting a group.

🔴 UMAR-RUNS (reads the dataset JSON; no model, no GPU):

    python analysis/discern_v2/meta_split.py --out configs/discern_v2/meta_split.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_JSON = Path("/data/umar/Datasets/preprocessed/dataset_json/FaceForensics++.json")
REAL_SUBSETS = ("FF-real",)
VAL_SELECT_FRACTION = 0.60
N_FOLDS = 5
SEED = 42


# ---------------------------------------------------------------------------
# video inventory
# ---------------------------------------------------------------------------


def identities(video_name: str) -> list[str]:
    """Every identity in a FF++ video name.

    `929` -> ["929"];  `802_885` -> ["802", "885"]. Both ids matter for disjointness: the second
    is the source face that was swapped in, and it appears elsewhere in the dataset under its own
    name.
    """
    return [p for p in re.split(r"[_-]", str(video_name)) if p.isdigit()] or [str(video_name)]


def load_videos(json_path: Path, split: str = "val", compression: str = "c23") -> list[dict]:
    """Every FF++ video in `split`, with its subset, label and frame count."""
    blob = json.loads(Path(json_path).read_text())
    root = blob["FaceForensics++"]
    videos = []
    for subset, per_label in root.items():
        section = per_label.get(split)
        if section is None:
            continue
        section = section.get(compression, section)
        for name, info in section.items():
            frames = info.get("frames") or []
            videos.append({
                "video": name,
                "subset": subset,                       # FF-real / FF-DF / FF-F2F / ...
                "label": 0 if subset in REAL_SUBSETS else 1,
                "n_frames": len(frames),
                "identities": identities(name),
            })
    if not videos:
        raise SystemExit(f"no videos found in {json_path} for split={split!r}")
    return videos


# ---------------------------------------------------------------------------
# grouping
# ---------------------------------------------------------------------------


def group_videos(videos: list[dict]) -> dict[str, list[dict]]:
    """Union-find over identities -> {group_id: [videos]}.

    Two videos share a group if they share any identity, transitively. A group is the smallest
    unit that can be assigned to a partition without splitting a face across the boundary.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)          # deterministic: lower id wins

    for v in videos:
        ids = v["identities"]
        for other in ids[1:]:
            union(ids[0], other)

    groups: dict[str, list[dict]] = defaultdict(list)
    for v in videos:
        groups[find(v["identities"][0])].append(v)
    return dict(groups)


def stratum_of(members: list[dict]) -> str:
    """A group's manipulation signature — the stratification key."""
    subsets = sorted({m["subset"] for m in members})
    return "+".join(subsets)


# ---------------------------------------------------------------------------
# assignment
# ---------------------------------------------------------------------------


def assign(groups: dict[str, list[dict]], fraction: float = VAL_SELECT_FRACTION
           ) -> tuple[dict[str, str], dict]:
    """Deterministically assign whole groups to VAL_select / VAL_meta.

    Within each stratum, groups are taken largest-first (by video count, ties broken by group id
    so the result is reproducible) and given to whichever partition is furthest below its target
    share. Largest-first matters: assigning a big group last can overshoot the target by more
    than the whole remaining budget.
    """
    assignment: dict[str, str] = {}
    per_stratum = defaultdict(list)
    for gid, members in groups.items():
        per_stratum[stratum_of(members)].append((gid, members))

    totals = {"VAL_select": 0, "VAL_meta": 0}
    for stratum in sorted(per_stratum):
        ordered = sorted(per_stratum[stratum], key=lambda kv: (-len(kv[1]), kv[0]))
        counts = {"VAL_select": 0, "VAL_meta": 0}
        for gid, members in ordered:
            total = counts["VAL_select"] + counts["VAL_meta"]
            target_select = fraction * (total + len(members))
            # deficit-driven: whichever partition is further below where it should be
            part = "VAL_select" if counts["VAL_select"] < target_select else "VAL_meta"
            assignment[gid] = part
            counts[part] += len(members)
        totals["VAL_select"] += counts["VAL_select"]
        totals["VAL_meta"] += counts["VAL_meta"]
    return assignment, totals


def make_folds(groups: dict[str, list[dict]], meta_gids: list[str],
               n_folds: int = N_FOLDS) -> dict[str, int]:
    """Group-disjoint 5-fold assignment inside VAL_meta (§12, §19).

    Balanced by video count rather than by group count: groups differ in size, so equal numbers
    of groups would give very unequal folds, and a fold with almost no fake videos cannot produce
    a usable out-of-fold gate prediction.
    """
    ordered = sorted(meta_gids, key=lambda g: (-len(groups[g]), g))
    load = [0] * n_folds
    fold_of: dict[str, int] = {}
    for gid in ordered:
        k = min(range(n_folds), key=lambda i: (load[i], i))
        fold_of[gid] = k
        load[k] += len(groups[gid])
    return fold_of


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------


def verify(videos: list[dict], groups: dict[str, list[dict]], assignment: dict[str, str],
           fold_of: dict[str, int]) -> dict:
    """§19's leakage guard, computed rather than asserted in prose.

    Checks identity disjointness directly on the identities themselves, not on the group ids —
    the grouping is what is being verified, so trusting it here would make the check circular.
    """
    ids_by_part = defaultdict(set)
    for gid, members in groups.items():
        for m in members:
            ids_by_part[assignment[gid]].update(m["identities"])
    shared = ids_by_part["VAL_select"] & ids_by_part["VAL_meta"]
    if shared:
        raise AssertionError(
            f"{len(shared)} identities appear in BOTH partitions (e.g. {sorted(shared)[:5]}); "
            f"the split is not source-disjoint")

    ids_by_fold = defaultdict(set)
    for gid, fold in fold_of.items():
        for m in groups[gid]:
            ids_by_fold[fold].update(m["identities"])
    for a in sorted(ids_by_fold):
        for b in sorted(ids_by_fold):
            if a < b and (ids_by_fold[a] & ids_by_fold[b]):
                raise AssertionError(f"folds {a} and {b} share identities")

    assigned = {v["video"] for gid, members in groups.items() for v in members}
    if assigned != {v["video"] for v in videos}:
        raise AssertionError("some videos were lost or duplicated during grouping")
    return {"identities_val_select": len(ids_by_part["VAL_select"]),
            "identities_val_meta": len(ids_by_part["VAL_meta"]),
            "shared_identities": 0}


def summarise(groups: dict[str, list[dict]], assignment: dict[str, str],
              fold_of: dict[str, int]) -> dict:
    per_part = defaultdict(lambda: {"videos": 0, "real": 0, "fake": 0,
                                    "frames": 0, "by_subset": defaultdict(int)})
    for gid, members in groups.items():
        part = assignment[gid]
        for m in members:
            p = per_part[part]
            p["videos"] += 1
            p["real" if m["label"] == 0 else "fake"] += 1
            p["frames"] += m["n_frames"]
            p["by_subset"][m["subset"]] += 1

    per_fold = defaultdict(lambda: {"videos": 0, "real": 0, "fake": 0})
    for gid, fold in fold_of.items():
        for m in groups[gid]:
            f = per_fold[f"fold_{fold}"]
            f["videos"] += 1
            f["real" if m["label"] == 0 else "fake"] += 1

    sizes = sorted((len(m) for m in groups.values()), reverse=True)
    return {
        "partitions": {k: {**v, "by_subset": dict(v["by_subset"])} for k, v in per_part.items()},
        "folds": dict(per_fold),
        "groups": {"n": len(groups), "largest": sizes[:5],
                   "median_size": sizes[len(sizes) // 2] if sizes else 0},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset-json", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--split", default="val",
                    help="§12 uses ONLY original FF++ validation videos")
    ap.add_argument("--compression", default="c23")
    ap.add_argument("--val-select-fraction", type=float, default=VAL_SELECT_FRACTION)
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--out", type=Path, default=REPO / "configs" / "discern_v2" / "meta_split.json")
    args = ap.parse_args()

    videos = load_videos(args.dataset_json, args.split, args.compression)
    groups = group_videos(videos)
    assignment, totals = assign(groups, args.val_select_fraction)
    meta_gids = [g for g, part in assignment.items() if part == "VAL_meta"]
    fold_of = make_folds(groups, meta_gids, args.folds)

    checks = verify(videos, groups, assignment, fold_of)
    summary = summarise(groups, assignment, fold_of)

    video_to_part = {m["video"]: assignment[gid]
                     for gid, members in groups.items() for m in members}
    video_to_fold = {m["video"]: fold_of[gid]
                     for gid in meta_gids for m in groups[gid]}

    n_total = len(videos)
    achieved = totals["VAL_select"] / max(1, n_total)
    print(f"FF++ {args.split} ({args.compression}): {n_total} videos in {len(groups)} "
          f"identity groups")
    print(f"  VAL_select {totals['VAL_select']} videos ({achieved:.1%}, target "
          f"{args.val_select_fraction:.0%})   VAL_meta {totals['VAL_meta']}")
    for part, s in summary["partitions"].items():
        print(f"  {part}: real {s['real']} / fake {s['fake']}, {s['frames']} frames, "
              f"{dict(s['by_subset'])}")
    for fold, s in sorted(summary["folds"].items()):
        print(f"  {fold}: {s['videos']} videos (real {s['real']} / fake {s['fake']})")
    print(f"  leakage check: {checks}")

    payload = {
        "spec": "V1 §12 — VAL_select/VAL_meta with 5-fold cross-fitting",
        "source": {"dataset_json": str(args.dataset_json), "split": args.split,
                   "compression": args.compression},
        "config": {"val_select_fraction": args.val_select_fraction, "n_folds": args.folds,
                   "seed": SEED, "unit": "identity group (union-find over video-name ids)"},
        "counts": {"videos": n_total, "groups": len(groups), **totals,
                   "achieved_val_select_fraction": achieved},
        "summary": summary,
        "checks": checks,
        "video_to_partition": video_to_part,
        "video_to_fold": video_to_fold,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

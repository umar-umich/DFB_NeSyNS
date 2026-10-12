#!/usr/bin/env python3
"""Pack a GenD test set from Dr Kutub's extracted frames into our evaluation format.

    <discern_ext python> upgrade/branchfusion/pack_gend.py --dataset DeepSpeak-2.0

Produces the four things a dataset needs before any branch can score it:
  1. a symlink tree in OUR space, carrying only the IN-SCOPE generators;
  2. an h5 pack keyed by relative path, written HERE rather than by DISCERN_Ext's
     `folder_to_h5_dataset`: that tool's directory walk does not traverse symlinked directories,
     so pointed at our tree it wrote zero images and reported success. The format it produces is
     reproduced exactly -- one uint8 dataset per image, keyed by the path relative to the
     dataset root -- and verified against an existing pack before use;
  3. one split .txt per subset, listing those keys;
  4. a `run1auxedl_<ds>.yaml` config, copied from an existing one so every field we do not
     set stays byte-identical to the sets already in the table.

NOTHING IS WRITTEN INTO /data/kutub. The frames are read through symlinks and the outputs land in
our tree, which is a standing constraint on this project and not a stylistic choice.

IN-SCOPE GENERATORS are named per dataset rather than filtered by rule, because the rule
(`OUT_OF_SCOPE`) is under review and a dataset packed against a moving rule cannot be compared
with one packed last week. Lip-sync families -- wav2lip, retalking, diff2lip, latentsync, memo --
are excluded here for the same reason they are excluded everywhere else: the detector is scored
on face swap and reenactment, and the audio-driven halves are a different task.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
EXT = Path("/data/umar/Repos/DISCERN_Ext")
SRC = Path("/data/kutub/datasets/GEND/extracted/test")
PACK = ROOT / "logs/branchfusion/packroot"

#: dataset -> (real subsets, in-scope fake subsets). Names are the directory names under SRC.
SETS = {
    "DeepSpeak-1.1": (["test-real"],
                      ["test-facefusion", "test-facefusion_gan", "test-facefusion_live"]),
    "DeepSpeak-2.0": (["test-real"],
                      ["test-facefusion", "test-hellomeme", "test-liveportrait"]),
    "UADFV":         (["real"], ["fake"]),
}
#: UADFV is spelled UADFD in Dr Kutub's tree; ours uses the canonical name from the paper.
SRC_DIR = {"UADFV": "UADFD"}
IMG = {".png", ".jpg", ".jpeg"}


def build_tree(ds: str) -> Path:
    """packroot/<ds>/<ds>/<subset> -> the real directory. The doubled name is deliberate: the h5
    key must begin with the dataset, and the tool keys by path relative to `image_root`."""
    reals, fakes = SETS[ds]
    src = SRC / SRC_DIR.get(ds, ds)
    root = PACK / ds
    inner = root / ds
    inner.mkdir(parents=True, exist_ok=True)
    for sub in reals + fakes:
        t, link = src / sub, inner / sub
        if not t.is_dir():
            raise SystemExit(f"missing source subset {t}")
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(t)
    print(f"  tree {inner} -> {len(reals + fakes)} subsets", flush=True)
    return root


def write_splits(ds: str) -> list[str]:
    reals, fakes = SETS[ds]
    src = SRC / SRC_DIR.get(ds, ds)
    out = EXT / "eval_adaptation/data/splits" / ds
    out.mkdir(parents=True, exist_ok=True)
    rel = []
    for sub in reals + fakes:
        keys = []
        for vid in sorted(os.listdir(src / sub)):
            d = src / sub / vid
            if not d.is_dir():
                continue
            for fr in sorted(os.listdir(d)):
                if Path(fr).suffix.lower() in IMG:
                    keys.append(f"{ds}/{sub}/{vid}/{fr}")
        (out / f"{sub}.txt").write_text("\n".join(keys) + "\n")
        rel.append(f"eval_adaptation/data/splits/{ds}/{sub}.txt")
        print(f"  split {sub}: {len(keys)} frames", flush=True)
    return rel


def write_h5(ds: str, out: Path) -> None:
    """One uint8 dataset per image, keyed `<ds>/<subset>/<video>/<frame>` -- the format the
    existing packs use. Written to a .partial and renamed, so an interrupted run never leaves a
    half pack that later looks complete."""
    import h5py
    import numpy as np
    from PIL import Image
    reals, fakes = SETS[ds]
    src = SRC / SRC_DIR.get(ds, ds)
    tmp = out.with_suffix(".partial.h5")
    n = 0
    with h5py.File(tmp, "w") as h:
        for sub in reals + fakes:
            for vid in sorted(os.listdir(src / sub)):
                d = src / sub / vid
                if not d.is_dir():
                    continue
                for fr in sorted(os.listdir(d)):
                    if Path(fr).suffix.lower() not in IMG:
                        continue
                    a = np.asarray(Image.open(d / fr).convert("RGB"), dtype=np.uint8)
                    h.create_dataset(f"{ds}/{sub}/{vid}/{fr}", data=a)
                    n += 1
            print(f"    h5 {sub}: {n} images so far", flush=True)
    tmp.replace(out)
    print(f"  h5 {out} ({n} images)", flush=True)


def write_config(ds: str, rel: list[str]) -> Path:
    """Copied from DFDM's config so every unrelated field stays identical."""
    base = yaml.safe_load(open(EXT / "eval_adaptation/configs/run1auxedl_DFDM.yaml"))
    base["tst_h5_path"] = f"eval_adaptation/data/h5/{ds}.h5"
    base["tst_files"] = {ds: rel}
    f = EXT / "eval_adaptation/configs" / f"run1auxedl_{ds}.yaml"
    yaml.safe_dump(base, open(f, "w"), sort_keys=False)
    print(f"  config {f}", flush=True)
    return f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(SETS))
    a = ap.parse_args()
    ds = a.dataset
    print(f"packing {ds}", flush=True)
    root = build_tree(ds)
    rel = write_splits(ds)
    h5 = EXT / "eval_adaptation/data/h5" / f"{ds}.h5"
    if h5.exists():
        print(f"  h5 exists, kept: {h5}", flush=True)
    else:
        write_h5(ds, h5)
    write_config(ds, rel)
    print(f"done {ds}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Cross-manipulation evaluation on DF40's individual face-swap generators: configs and export job lists.

    python analysis/tbiom/df40_prepare.py

DiCoME's cross-manipulation table covers UniFace, BlendFace, MobileFaceSwap, e4s, FaceDancer, FSGAN, InSwapper and
SimSwap. DF40 ships two more face-swap generators that the table leaves out -- the classical graphics **FaceSwap** and
**DeepFaceLab** -- and every generator exists in both a CDF-sourced and an FF++-sourced version. This evaluates all ten
generators in both sources, so the comparison covers what was reported and what was not.

Each generator is scored against the real pool DF40 ships for its source (`DF40-cdf-real`, `DF40-ff-real`, and
DeepFaceLab's own `deepfacelab-real`), so a generator's AUROC is never computed against another source's genuine
videos. Three config families are written per generator, all pointing at the same splits: `warmgate_*` (our reference
spatial architecture), `dicome_*` (our trained DiCoME baseline) and `run1auxedl_*` (what the temporal branch's
dataset lookup reads).
"""
from __future__ import annotations
import re
from pathlib import Path

EXT = Path("/data/umar/Repos/DISCERN_Ext")
CFG, SPL = EXT / "eval_adaptation/configs", "eval_adaptation/data/splits/DF40"
LOG = Path("/data/umar/Repos/DFB_NeSyNS/logs/tbiom/df40")
S23 = Path("/data/umar/Repos/DFB_NeSyNS/logs/tbiom/stage23")
METHODS = ["blendface", "e4s", "facedancer", "faceswap", "fsgan", "inswap", "mobileswap", "simswap", "uniface"]
# the two DiCoME's published cross-manipulation table does not list
NOT_IN_DICOME_TABLE = {"faceswap", "deepfacelab"}
OURS = EXT / ("runs/discern_ext_train/warm_dicome_gate_seed42/checkpoints/"
              "discern_ext-best-epoch=19-val_auroc_video=0.9956.ckpt")
DICOME = EXT / ("runs/_baseline_dicome/dicome_train/ffpp_reproduce_seed42/checkpoints/"
                "dicome-best-epoch=05-val_auroc_video=0.9953.ckpt")
TEMPLATES = {"warmgate": "warmgate_CDFv2.yaml", "dicome": "dicome_KoDF.yaml", "run1auxedl": "run1auxedl_CDFv2.yaml"}


def datasets():
    """dataset name -> (fake split, real split)."""
    out = {}
    for m in METHODS:
        for src, real in (("cdf", "DF40-cdf-real"), ("ff", "DF40-ff-real")):
            if (EXT / SPL / f"{m}-{src}.txt").exists():
                out[f"DF40_{m}_{src}"] = (f"{m}-{src}", real)
    if (EXT / SPL / "deepfacelab.txt").exists():
        out["DF40_deepfacelab"] = ("deepfacelab", "deepfacelab-real")
    return out


def write_configs(ds):
    n = 0
    for name, (fake, real) in ds.items():
        files = "\n".join(f"  - {SPL}/{s}.txt" for s in (real, fake))       # real first: label comes from the subset name
        for fam, tmpl in TEMPLATES.items():
            out = CFG / f"{fam}_{name}.yaml"
            if out.exists():
                continue
            t = (CFG / tmpl).read_text()
            t = re.sub(r"^run_name: .*$", f"run_name: {fam}_{name}", t, flags=re.M)
            t = re.sub(r"^tst_h5_path: .*$", "tst_h5_path: eval_adaptation/data/h5/DF40.h5", t, flags=re.M)
            t = re.sub(r"^tst_files:\n(?:  \S+:\n)(?:  - .*\n)+", f"tst_files:\n  {name}:\n{files}\n", t, flags=re.M)
            out.write_text(t); n += 1
    return n


def main():
    LOG.mkdir(parents=True, exist_ok=True)
    ds = datasets()
    n = write_configs(ds)
    print(f"{len(ds)} generator datasets, {n} configs written")
    for tag, ck, fam in (("ours", OURS, "warmgate"), ("dicome", DICOME, "dicome")):
        jobs = [f"{ck}|{fam}_{name}.yaml|{name}|{S23}/{tag}_{name}.csv|" for name in ds]
        (LOG / f"jobs_{tag}.txt").write_text("\n".join(jobs) + "\n")
        print(f"  {tag}: {len(jobs)} export jobs -> {LOG / f'jobs_{tag}.txt'}")
    (LOG / "datasets.txt").write_text("\n".join(ds) + "\n")
    print("generators:", ", ".join(sorted({n.split('_')[1] for n in ds})))
    print("not in DiCoME's published table:", ", ".join(sorted(NOT_IN_DICOME_TABLE)))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate the missing per-generator DF40 configs for the four-family results table.

    python tbiom/tables/make_configs.py            # dry run, lists what it would write
    python tbiom/tables/make_configs.py --write

PER GENERATOR, not per family. The table reports mean +- standard deviation ACROSS the generators
inside each family, so every generator needs its own AUROC and therefore its own config. The 41
configs that already exist cover FS and FR; EFS, FE and the standalone editors do not exist yet.

FAMILIES follow DF40's own published taxonomy, which is what makes the numbers comparable with
published DF40 tables rather than bespoke to us. Note that DF40 places `wav2lip` in face
reenactment while our v3.1 objective excludes it as lip-sync-only; the taxonomy is followed here and
the difference is recorded in the report.

REAL SIDE. Each generator is scored against the real set of its own domain -- `DF40-cdf-real` for
`-cdf`, `DF40-ff-real` for `-ff`. The standalone editors carry their OWN paired reals and are wired
to those instead, which is why they cannot be averaged into the same column without saying so.
"""
from __future__ import annotations

import argparse, re, sys
from pathlib import Path

EXT = Path("/data/umar/Repos/DISCERN_Ext")
CONFIGS = EXT / "eval_adaptation/configs"
SPLITS = EXT / "eval_adaptation/data/splits/DF40"

FAMILIES = {
    "FS":  ["blendface", "e4s", "facedancer", "faceswap", "fsgan", "inswap", "mobileswap",
            "simswap", "uniface"],
    "FR":  ["danet", "facevid2vid", "fomm", "hyperreenact", "lia", "mcnet", "MRAA",
            "one_shot_free", "pirender", "sadtalker", "tpsm", "wav2lip"],
    "EFS": ["ddim", "DiT", "rddm", "sd2.1", "SiT", "StyleGAN2", "StyleGAN3", "StyleGANXL", "VQGAN"],
    "FE":  ["e4e"],
}
STANDALONE = {"stargan": "stargan-real", "starganv2": "starganv2-real", "styleclip": "styleclip-real"}
REAL = {"cdf": "DF40-cdf-real", "ff": "DF40-ff-real"}

# one template per arm; each already exists for blendface_cdf and differs only in checkpoint wiring
ARMS = {"warmgate": "warmgate_DF40_blendface_cdf.yaml",     # ours
        "dicome":   "dicome_DF40_blendface_cdf.yaml"}        # DiCoME, our trained weights


def write_one(tmpl_text, arm, name, real_split, fake_split, write):
    body = "\n".join(f"  - eval_adaptation/data/splits/DF40/{s}.txt" for s in (real_split, fake_split))
    out = tmpl_text
    out = re.sub(r"^run_name: .*$", f"run_name: {arm}_{name}", out, flags=re.M)
    out = re.sub(r"^tst_h5_path: .*$", "tst_h5_path: eval_adaptation/data/h5/DF40.h5", out, flags=re.M)
    out = re.sub(r"^tst_files:\n(?:  \S+:\n)(?:  - .*\n)+", f"tst_files:\n  {name}:\n{body}\n",
                 out, flags=re.M)
    p = CONFIGS / f"{arm}_{name}.yaml"
    if write:
        p.write_text(out)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    planned, missing_splits, skipped = [], [], 0
    for arm, tmpl in ARMS.items():
        t = CONFIGS / tmpl
        if not t.is_file():
            print(f"  !! template missing for arm {arm}: {tmpl}"); continue
        text = t.read_text()
        for fam, gens in FAMILIES.items():
            for g in gens:
                for var in ("cdf", "ff"):
                    stem, name = f"{g}-{var}", f"DF40_{g}_{var}"
                    if not (SPLITS / f"{stem}.txt").is_file():
                        missing_splits.append(stem); continue
                    if (CONFIGS / f"{arm}_{name}.yaml").is_file():
                        skipped += 1; continue
                    planned.append(write_one(text, arm, name, REAL[var], stem, a.write))
        for g, real in STANDALONE.items():
            name = f"DF40_{g}"
            if not (SPLITS / f"{g}.txt").is_file():
                missing_splits.append(g); continue
            if (CONFIGS / f"{arm}_{name}.yaml").is_file():
                skipped += 1; continue
            planned.append(write_one(text, arm, name, real, g, a.write))

    print(f"  {'wrote' if a.write else 'would write'} {len(planned)} configs; {skipped} already existed")
    for p in planned[:6]:
        print(f"    {p.name}")
    if len(planned) > 6:
        print(f"    ... and {len(planned)-6} more")
    if missing_splits:
        print(f"  splits absent (skipped): {sorted(set(missing_splits))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

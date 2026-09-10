#!/usr/bin/env python3
"""DF40 as four OOD family subsets, using DF40's own published taxonomy.

    python analysis/tbiom/df40_families.py --write-configs

WHY FAMILIES AND NOT 80 SUBSETS. DF40 ships 80 splits. Exporting each gives 80 CSVs and 80
AUROCs, which is a table nobody reads and a multiple-comparisons problem nobody wants. DF40's
own paper groups its generators into four manipulation families, so grouping that way is both
the smaller table and the citable one -- other papers report the same four, which makes our
numbers comparable instead of bespoke.

    FS   face swapping          identity replaced
    FR   face reenactment       expression/pose driven, identity kept
    EFS  entire face synthesis  no source face at all
    FE   face editing           attribute-level edits

The bundling mechanism is the one CDFv3 already uses: many split files under one `tst_files`
group, read out as a single dataset.

THE `-ff` SUBSETS ARE EXCLUDED, and this is the important decision. Every DF40 generator ships
in two domain variants: `-cdf` built on Celeb-DF source videos, `-ff` built on FaceForensics++
source videos. FF++ is OUR TRAINING SET. DF40 re-indexes filenames (`DF40-ff-real__000`), so
the original FF++ video ids are not recoverable from paths, which means we cannot currently
prove the `-ff` subsets avoid our training videos. Until that is resolved by frame matching or
from DF40's documentation, an `-ff` number could be measuring memorisation and would be
indefensible in the paper. Including them would double the table for material we cannot
certify, so they are left out and the reason is recorded.

WHAT ELSE IS EXCLUDED, and why. Eight standalone subsets (CollabDiff, MidJourney, whichisr3al,
stargan, starganv2, styleclip, deepfacelab, heygen) each carry their OWN paired real set rather
than drawing on DF40-cdf-real, and most are image-level rather than video-level. Mixing them
into the families would put two different real pools behind one AUROC. They are listed here as
`STANDALONE` so they can be run as separate small sets on purpose, not forgotten.

THE REAL SIDE IS SHARED AND THIN, and any writeup has to say so. All four families score
against the same 178 DF40-cdf-real videos, so the four AUROCs are NOT independent -- a
threshold artifact on those reals moves all four together. 178 real videos against several
thousand fakes is also a wide confidence interval on the real side. This is a property of how
DF40 is distributed, not a choice, but it is a caveat and not a footnote.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXT = Path("/data/umar/Repos/DISCERN_Ext")
SPLITS = EXT / "eval_adaptation/data/splits/DF40"
CONFIGS = EXT / "eval_adaptation/configs"
REAL = "DF40-cdf-real"

FAMILIES = {
    # face swapping -- identity replaced. 9 methods.
    "DF40_FS": ["blendface-cdf", "e4s-cdf", "facedancer-cdf", "faceswap-cdf", "fsgan-cdf",
                "inswap-cdf", "mobileswap-cdf", "simswap-cdf", "uniface-cdf"],
    # face reenactment -- expression and pose driven, identity preserved. 12 methods.
    "DF40_FR": ["danet-cdf", "facevid2vid-cdf", "fomm-cdf", "hyperreenact-cdf", "lia-cdf",
                "mcnet-cdf", "MRAA-cdf", "one_shot_free-cdf", "pirender-cdf", "sadtalker-cdf",
                "tpsm-cdf", "wav2lip-cdf"],
    # entire face synthesis -- no source face. 9 methods, and the ONLY latent-diffusion content
    # available to us anywhere: FF++ has none, which is why Branch 3 reads near chance in-domain.
    "DF40_EFS": ["ddim-cdf", "DiT-cdf", "rddm-cdf", "sd2.1-cdf", "SiT-cdf", "StyleGAN2-cdf",
                 "StyleGAN3-cdf", "StyleGANXL-cdf", "VQGAN-cdf"],
    # face editing -- attribute-level. Only one -cdf method exists; the rest are standalone.
    "DF40_FE": ["e4e-cdf"],
}

# Own paired reals and/or image-level. Run separately if wanted; never folded into a family.
STANDALONE = {
    "deepfacelab": "deepfacelab-real",      # video, face swap, classic DFL
    "heygen": "heygen-real",                # video, commercial
    "CollabDiff": "CollabDiff-real",        # image, EFS
    "MidJourney": "MidJourney-real",        # image, EFS
    "whichisr3al": "whichisr3al-real",      # image, EFS
    "stargan": "stargan-real",              # image, FE
    "starganv2": "starganv2-real",          # image, FE
    "styleclip": "styleclip-real",          # image, FE
}

BASE = "run1auxedl_CDFv3.yaml"              # the frozen arm's export config, as the template


def write_configs(arms=("run1auxedl",)) -> int:
    import re
    tmpl = (CONFIGS / BASE).read_text()
    n = 0
    for arm in arms:
        for fam, subs in FAMILIES.items():
            missing = [s for s in subs + [REAL] if not (SPLITS / f"{s}.txt").is_file()]
            if missing:
                print(f"  !! {fam}: missing splits {missing}"); continue
            files = "\n".join(f"  - eval_adaptation/data/splits/DF40/{s}.txt"
                              for s in [REAL] + subs)
            out = tmpl
            out = re.sub(r"^run_name: .*$", f"run_name: {arm}_{fam}", out, flags=re.M)
            out = re.sub(r"^tst_h5_path: .*$",
                         "tst_h5_path: eval_adaptation/data/h5/DF40.h5", out, flags=re.M)
            out = re.sub(r"^tst_files:\n(?:  \w+:\n)(?:  - .*\n)+",
                         f"tst_files:\n  {fam}:\n{files}\n", out, flags=re.M)
            p = CONFIGS / f"{arm}_{fam}.yaml"
            p.write_text(out); n += 1
            print(f"  wrote {p.name}  ({len(subs)} methods + {REAL})")
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write-configs", action="store_true")
    args = ap.parse_args()

    print(__doc__.split("\n\n")[1].strip(), "\n")
    for fam, subs in FAMILIES.items():
        print(f"{fam:10} {len(subs):2d} methods: {', '.join(subs)}")
    print(f"\nreal pool for all four: {REAL}")
    print(f"excluded -ff variants: {len([p for p in SPLITS.glob('*-ff.txt')])} splits "
          f"(FF++-sourced, provenance unresolved)")
    print(f"standalone (own reals): {', '.join(STANDALONE)}")
    if args.write_configs:
        print()
        n = write_configs()
        print(f"\n{n} configs written")
    return 0


if __name__ == "__main__":
    sys.exit(main())

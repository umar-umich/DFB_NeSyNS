#!/usr/bin/env python3
"""DF40's four-family taxonomy, the real set each generator is scored against, and the method rows.

Single source of truth: `make_configs.py` writes one config per entry here and `build_table.py`
reads one result per entry, so the table can never drift from what was actually exported.

FAMILIES follow DF40's own published taxonomy rather than a bespoke grouping, which is what keeps
our columns comparable with published DF40 tables. DF40 places `wav2lip` in face reenactment while
our v3.1 objective excludes it as lip-sync-only; the taxonomy is followed and the difference is
recorded in the report.

REAL SIDE. Every generator is scored against the real set of its own domain -- `DF40-cdf-real` for
a `-cdf` variant, `DF40-ff-real` for `-ff`. `stargan`, `starganv2` and `styleclip` ship their OWN
paired reals and are wired to those, which is DF40's own arrangement: its download notes split the
corpus into 31 "known" methods that share the FF++/Celeb-DF real pools and 9 "unknown" ones that
bundle reals in the folder. That is a packaging difference, not a taxonomic one -- all four editors
are face editing and are reported in ONE column. Averaging cells scored against different real
pools is already what every other family does, since `-cdf` and `-ff` cells use different pools.
"""
from __future__ import annotations

FAMILIES = {
    "FS":  ["blendface", "e4s", "facedancer", "faceswap", "fsgan", "inswap", "mobileswap",
            "simswap", "uniface"],
    "FR":  ["danet", "facevid2vid", "fomm", "hyperreenact", "lia", "mcnet", "MRAA",
            "one_shot_free", "pirender", "sadtalker", "tpsm", "wav2lip"],
    "EFS": ["ddim", "DiT", "rddm", "sd2.1", "SiT", "StyleGAN2", "StyleGAN3", "StyleGANXL", "VQGAN"],
    "FE":  ["e4e"],
}
# Face editing, DF40's "unknown" half: same family as e4e, own bundled reals.
STANDALONE = {"stargan": "stargan-real", "starganv2": "starganv2-real", "styleclip": "styleclip-real"}
REAL = {"cdf": "DF40-cdf-real", "ff": "DF40-ff-real"}
VARIANTS = ("cdf", "ff")

FAMILY_NAME = {"FS": "Face swap", "FR": "Reenactment", "EFS": "Synthesis", "FE": "Editing"}
FAMILY_ORDER = ["FS", "FR", "EFS", "FE"]

# `wav2lip` is audio-driven lip-sync: the frames exist as images but the manipulation is only
# legible over time, so it is flagged wherever a per-generator number is printed.
LIPSYNC = {"wav2lip"}

# DF40's EFS/FE/standalone sets are image collections, not videos -- one "video" per image. The
# temporal view still runs (it sees a repeated frame); the methods section records that rule.
IMAGE_LEVEL_FAMILIES = {"EFS", "FE"}


def cells():
    """Every (family, generator, variant, set_name, fake_split, real_split) the table covers."""
    for fam, gens in FAMILIES.items():
        for g in gens:
            for var in VARIANTS:
                yield fam, g, var, f"DF40_{g}_{var}", f"{g}-{var}", REAL[var]
    for g, real in STANDALONE.items():
        yield "FE", g, None, f"DF40_{g}", g, real


# method rows: label -> (stage23 output prefix, kind).  kind picks the scorer in build_table.py.
# The second operator row is the SAME three views and the same per-split fitting; only the
# combination rule differs, so the two rows isolate the operator.
OPERATOR_ARMS = {"DISCERN (ours, PCR6+-R)": "PCR6P-R"}

OURS_ROWS = {
    "DISCERN (ours, DS-R)":      ("ours", "dsr"),
    "DISCERN (ours, PCR6+-R)":   ("ours", "operator"),
    "DiCoME (our training)":     ("dicome", "views"),
    "DiCoME (released weights)": ("dicomerel", "views"),
}
# third-party rows read the GenD-repo runner's test_predictions.csv
BASELINE_ROWS = {
    "GenD-CLIP": ("GenD_CLIP", "gend"),
    "Effort":    ("Effort", "gend"),
    "ForAda":    ("ForAda", "gend"),
    "FSFM":      ("FSFM", "gend"),
}

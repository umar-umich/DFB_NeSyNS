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
# Operator rows: same three views, same opinions. `fit` means the reliability triple is grid-searched
# on each split's fitting fold; `None` means no discounting at all (reliability pinned to 1), which is
# what "Dempster, no R" means.
#
# NOTE FOR READING THE TABLE: PCR6+-R against plain Dempster changes TWO things at once -- the
# combination rule AND the reliability discounting, so the difference between the two rows is not
# the operator on its own. Measured over these 65 sets: Dempster 0.9535, Dempster-R 0.9571,
# PCR6+-R 0.9562 -- the discounting accounts for the movement, and the operator held at fixed
# discounting is -0.0009 (ahead on 20 of 65 sets). Where PCR6+ does win on its own is reenactment:
# 0.9347 against Dempster-R's 0.9317 and Dempster's 0.9305, ahead on 13 of 24 sets.
# Dempster-R and plain Dempster were both dropped from the printed table on UMAR's instruction
# (2026-09-29) after this was put to him, and the single row is now labelled just "DISCERN (ours)".
# The operator it uses is PCR6+ with reliability; the label no longer says so, so the report's
# Scoring paragraph has to. Re-adding either row is one line here and in OURS_ROWS:
#   ("Dempster-R", "fit")  -> reliability, Dempster's rule
#   ("Dempster-R", None)   -> no discounting at all
OPERATOR_ARMS = {"DISCERN (ours)": ("PCR6P-R", "fit")}

OURS_ROWS = {
    "DISCERN (ours)":            ("ours", "operator"),   # three views, PCR6+ with reliability
    "DiCoME":                    ("dicome", "views"),   # OUR retraining, not the release
    # `dicomerel` (the authors' released DiCoME weights) was dropped from the printed table on
    # UMAR's instruction (2026-09-29). Its 65 exports are on disk as logs/tbiom/stage23/dicomerel_*
    # and it re-enters with one line here; it stays useful as a reproduction check on our retraining.
}
# third-party rows read the GenD-repo runner's test_predictions.csv
BASELINE_ROWS = {
    "ForAda":        ("ForAda", "gend"),
    "Effort":        ("Effort", "gend"),
    "FSFM":          ("FSFM", "gend"),
    "FSFM-Adapter":  ("FSFM_Adapter", "gend"),
    "GenD-DINO":     ("GenD_DINO", "gend"),
    "GenD-PE":       ("GenD_PE", "gend"),
    "GenD-CLIP":     ("GenD_CLIP", "gend"),
}


# PRINT ORDER, stated explicitly rather than inherited from the dicts above. Those dicts group rows
# by which SCORER they need, which is not the order a reader wants: DiCoME is prior work but shares
# our three-view replay, so grouping by scorer floated it below us. Third-party first, then DiCoME,
# then ours last, with the proposed operator on the final line.
ROW_ORDER = ["ForAda", "Effort", "FSFM", "FSFM-Adapter", "GenD-DINO", "GenD-PE", "GenD-CLIP",
             "DiCoME", "DISCERN (ours)"]


def row_order(present) -> list:
    """`present` in ROW_ORDER, with anything unlisted appended so a new row can never vanish."""
    known = [m for m in ROW_ORDER if m in present]
    return known + [m for m in present if m not in ROW_ORDER]

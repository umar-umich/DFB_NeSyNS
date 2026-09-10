#!/usr/bin/env python3
"""Canonical display names for the three views — defined ONCE, imported everywhere.

The figures and the reports must call the branches the same thing, and before this module they
did not: the t-SNE figures said "Appearance / Off-manifold residual / Bona-fide face prior"
while the result tables still said "semantic (CLIP) / artifact (beta-VAE) / FS-VFM". Two names
for one branch across a paper's figures and tables is the kind of inconsistency a reader trips
on and a reviewer remarks on.

THE KEYS ARE IDENTIFIERS AND DO NOT CHANGE. `semantic`, `artifact` and `fsvfm` appear in
exported CSV column names (`p_semantic`, `u_fsvfm`), in `fusion_views` in the model config, and
in export prefixes. Renaming those would break every stored artifact and every checkpoint's
config. Only the DISPLAY strings are renamed, and only at the point of writing a table.
"""
from __future__ import annotations

VIEW_NAME = {
    "semantic": "Appearance",
    "artifact": "Off-manifold residual",
    "fsvfm":    "Bona-fide face prior",
    "__all__":  "Fused",
    "fused":    "Fused",
}

# Older mechanism-flavoured labels, kept ONLY so historical reports can be mapped forward
# mechanically rather than by hand. Not for new writing.
LEGACY = {
    "semantic (CLIP)": "Appearance",
    "artifact (β-VAE)": "Off-manifold residual",
    "artifact (beta-VAE)": "Off-manifold residual",
    "FS-VFM": "Bona-fide face prior",
}


# The two Branch-3 experts, named by WHAT THEY WERE TRAINED TO DO rather than by their source.
# Arm B's expert is trained to preserve a real-face representation; arm C's is a checkpoint
# fine-tuned discriminatively for detection. Both descriptions are accurate and neither is a
# pointer to a particular publication, which keeps table labels functional.
#
# This is a LABELLING choice, not an attribution one. Branch 3 is built on a third-party
# pretrained encoder and arm C uses a third-party released fine-tuned checkpoint; the method
# section and the references still have to say so. Shorthand in a results table is fine;
# leaving the provenance out of the paper entirely would not be.
BRANCH3 = {
    "student": "preservation Br3",       # ours: LoRA student trained for representation preservation
    "ft":      "discriminative Br3",     # external: checkpoint fine-tuned for detection
}


def name(key: str) -> str:
    """Display name for a view key; unknown keys pass through unchanged."""
    return VIEW_NAME.get(key, key)


def order() -> list[str]:
    """The three views in the order every table and figure presents them."""
    return ["semantic", "artifact", "fsvfm"]

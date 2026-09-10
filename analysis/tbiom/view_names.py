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


def name(key: str) -> str:
    """Display name for a view key; unknown keys pass through unchanged."""
    return VIEW_NAME.get(key, key)


def order() -> list[str]:
    """The three views in the order every table and figure presents them."""
    return ["semantic", "artifact", "fsvfm"]

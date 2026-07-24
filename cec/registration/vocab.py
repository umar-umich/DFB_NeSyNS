"""The frozen claim vocabulary, from docs/pilots/CODEBOOK.md (revision 3).

CODEBOOK.md is authoritative (CURRENT_STATE document map). This module is the
machine-readable copy of section 2 (regions) and section 3 (predicate types).
The frozen proposer prompt embeds these same strings; `check_prompt_vocab()`
asserts the two have not drifted, and is called by the Task 1 self-check.

Anything a proposer emits outside these sets is coerced by the Task 6
validator: unknown artifact -> OTHER, unmappable location -> null.
"""
from __future__ import annotations

from typing import FrozenSet

# --- section 3: predicate types -------------------------------------------
# Grouped by the forensic family they probe, as in the codebook.
BLENDING_PREDICATES = frozenset({"boundary_artifact", "blend_seam_visible"})
SPECTRAL_PREDICATES = frozenset({"frequency_anomaly", "noise_inconsistency"})
GEOMETRY_PREDICATES = frozenset({"geometry_inconsistency", "identity_drift"})
TEXTURE_PREDICATES = frozenset({"texture_anomaly", "physiology_violation"})
GLOBAL_PREDICATES = frozenset({"lighting_inconsistency"})

ARTIFACTS: FrozenSet[str] = (
    BLENDING_PREDICATES
    | SPECTRAL_PREDICATES
    | GEOMETRY_PREDICATES
    | TEXTURE_PREDICATES
    | GLOBAL_PREDICATES
)

OTHER = "OTHER"  # the validator's sink for out-of-vocabulary artifacts

# --- section 2: landmark regions ------------------------------------------
# whole_face is global: never a spatial repair target, verified only by
# spectral interventions (codebook rev-2 change).
WHOLE_FACE = "whole_face"

SPATIAL_REGIONS: FrozenSet[str] = frozenset({
    "left_eye", "right_eye", "inter_ocular", "nose", "nasolabial", "mouth",
    "jawline", "left_cheek", "right_cheek", "forehead", "hairline", "chin",
    "face_boundary",
})

REGIONS: FrozenSet[str] = SPATIAL_REGIONS | {WHOLE_FACE}


def routes_to_spectral(artifact: str, location: str | None) -> bool:
    """Whole-face frequency/noise claims are the (dropped) spectral instrument's.

    Spectral was dropped after Pilot S, so these are UNTESTABLE — but they stay
    distinct from whole-face COMPOSITE claims (T13). Do not conflate.
    """
    return location == WHOLE_FACE and artifact in SPECTRAL_PREDICATES


def claim_scope(artifact: str, location: str | None) -> str:
    """Route a claim to a certification scope (T13). One of:

      region      a specific landmark region -> repair that region, judge against
                  the region-calibrated `gate_region:` block.
      composite   whole_face + a NON-spectral artifact -> repair the full inner-face
                  region, judge against the frozen full-mask `gate:` block. (The
                  same operation `gate:` was calibrated on, so no bar is "lowered".)
      spectral    whole_face + a frequency/noise artifact -> UNTESTABLE (spectral
                  instrument dropped after Pilot S; characterization only).
      untestable  no locus / unmappable.
    """
    if location == WHOLE_FACE:
        return "spectral" if artifact in SPECTRAL_PREDICATES else "composite"
    if location in SPATIAL_REGIONS:
        return "region"
    return "untestable"


def check_prompt_vocab(prompt_text: str) -> None:
    """Fail loudly if the frozen prompt and this module have drifted apart."""
    missing_artifacts = sorted(a for a in ARTIFACTS if a not in prompt_text)
    missing_regions = sorted(r for r in REGIONS if r not in prompt_text)
    if missing_artifacts or missing_regions:
        raise AssertionError(
            "frozen prompt is out of sync with the codebook vocabulary — "
            f"missing artifacts {missing_artifacts}, missing regions {missing_regions}"
        )

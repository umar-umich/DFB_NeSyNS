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
    """Whole-face frequency/noise claims go to the spectral instrument.

    The gate's routing rule (Implementation v2, Task 7): spatial -> mask+repair
    vs the CLIP detector; whole-face spectral -> intervention vs the frequency
    detector; everything else -> UNTESTABLE.
    """
    return location == WHOLE_FACE and artifact in SPECTRAL_PREDICATES


def check_prompt_vocab(prompt_text: str) -> None:
    """Fail loudly if the frozen prompt and this module have drifted apart."""
    missing_artifacts = sorted(a for a in ARTIFACTS if a not in prompt_text)
    missing_regions = sorted(r for r in REGIONS if r not in prompt_text)
    if missing_artifacts or missing_regions:
        raise AssertionError(
            "frozen prompt is out of sync with the codebook vocabulary — "
            f"missing artifacts {missing_artifacts}, missing regions {missing_regions}"
        )

"""CEC Disclosure Policy: what a Certified Evidence Record is allowed to display.

Performs NO certification — three-tier disclosure (region > composite > abstain),
two deployment modes (certified_mode / screening_mode), optional
detector-vs-explainer disagreement suppression.
"""
from .output_policy import (  # noqa: F401
    ABSTENTION,
    CERTIFIED_MODE,
    COMPOSITE_STATEMENT,
    DISAGREEMENT,
    SCREENING_MODE,
    DisclosurePolicy,
    assemble,
    render,
)

__all__ = [
    "DisclosurePolicy", "CERTIFIED_MODE", "SCREENING_MODE",
    "ABSTENTION", "DISAGREEMENT", "COMPOSITE_STATEMENT",
    "assemble", "render",
]

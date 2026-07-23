"""CEC repair + controls: the counterfactual intervention and its control battery.

  repair.py    poisson_paste (production), hard_paste (fallback)
  controls.py  LPIPS-matched blur/shift, wrong-region, real-offset; build_variants
"""

from .controls import (  # noqa: F401
    Lpips,
    blur_region,
    build_variants,
    make_control_region,
    match_corruption,
    shift_region,
)
from .repair import hard_paste, poisson_paste  # noqa: F401

__all__ = [
    "poisson_paste",
    "hard_paste",
    "Lpips",
    "blur_region",
    "shift_region",
    "match_corruption",
    "make_control_region",
    "build_variants",
]

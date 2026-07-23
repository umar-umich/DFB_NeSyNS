"""CEC network guts: the frozen-instrument plumbing.

`cec_instrument_detector.py` only wires these together, per this repo's house
style (the detector file is a wiring file; the work lives here).

  gend_registry.py  GenD registry adapter — scoped chdir, model construction
  inference.py      batched p(fake)
"""

from .gend_registry import (  # noqa: F401
    GEND_ROOT,
    build_gend_model,
    gend_cwd,
    pfake_from_tensor,
    to_pil_rgb,
)
from .inference import pfake_batch  # noqa: F401

__all__ = [
    "GEND_ROOT",
    "build_gend_model",
    "gend_cwd",
    "pfake_from_tensor",
    "to_pil_rgb",
    "pfake_batch",
]

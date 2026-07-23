"""CEC instruments: the things that certify a claim.

Thin adapters onto the frozen detector classes registered in
`training/detectors/cec_instrument_detector.py`. Two instruments, because
Study 1B showed CLIP detectors ignore spectral cues (Δp≈0) while spectral
separates at AUC 0.94–0.99:

  spatial.py    CLIP-family detector + paired-real region repair
  spectral.py   frequency detector + spectral interventions   (Task 5, not built)
"""

from .spatial import INSTRUMENTS, SpatialInstrument  # noqa: F401
from .spectral import SPECTRAL_INSTRUMENTS, SpectralInstrument  # noqa: F401

__all__ = [
    "SpatialInstrument", "INSTRUMENTS",
    "SpectralInstrument", "SPECTRAL_INSTRUMENTS",
]

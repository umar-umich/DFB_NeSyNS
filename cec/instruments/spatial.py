"""The spatial instrument: a frozen CLIP-family detector, queried counterfactually.

This is a THIN ADAPTER. The model code lives in DeepfakeBench house style at
`training/detectors/cec_instrument_detector.py` (+ `training/networks/cec/`);
this file only exposes the Implementation-v2 API on top of it and owns no
implementation of its own.

    from cec.instruments.spatial import SpatialInstrument

    inst = SpatialInstrument("effort")
    nm = inst.necessity_margin(original_bgr, repaired_bgr)   # p_fake(orig) - p_fake(repaired)

The gate (Task 7) reads `necessity_margin`; everything else here supports it.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

# DeepfakeBench resolves its own modules relative to training/, exactly as
# training/detectors/__init__.py does when run from there.
_TRAINING_DIR = Path(__file__).resolve().parents[2] / "training"
if str(_TRAINING_DIR) not in sys.path:
    sys.path.append(str(_TRAINING_DIR))

from cec.registration import load_detectors  # noqa: E402

INSTRUMENTS = ("effort", "fsfm", "gend", "forada")


class SpatialInstrument:
    """One frozen spatial detector. Construct once per detector; it holds weights.

    Loading is lazy so that importing this module does not pull in torch or the
    GenD checkout — the validator and the record readers import it for names only.
    """

    def __init__(self, name: str, device: str | None = None):
        if name not in INSTRUMENTS:
            raise ValueError(f"unknown spatial instrument '{name}'. Choices: {INSTRUMENTS}")
        self.name = name
        self._device = device
        self._detector = None

    # -- lazy load -----------------------------------------------------------
    @property
    def detector(self):
        if self._detector is None:
            from metrics.registry import DETECTOR  # noqa: E402
            import detectors  # noqa: F401,E402  (populates the registry)

            cls = DETECTOR[f"cec_{self.name}"]
            self._detector = cls({"device": self._device})
        return self._detector

    @property
    def anchor(self) -> float:
        """The frozen gt-repair drop median this instrument must keep reproducing."""
        return load_detectors()["anchors"]["gt_repair_drop_median"][self.name]

    # -- the API Implementation v2 asks for -----------------------------------
    def p_fake(self, image) -> float:
        """p(fake) for one image (BGR uint8 array or PIL)."""
        return float(self.p_fake_batch([image])[0])

    def p_fake_batch(self, images: Sequence, batch_size: int = 64) -> np.ndarray:
        """p(fake) for many images. Batched: the gate re-queries once per claim."""
        return self.detector.p_fake(list(images), batch_size=batch_size)

    # -- what the gate actually calls ----------------------------------------
    def necessity_margin(self, original, intervened) -> float:
        """NM = p_fake(original) - p_fake(intervened), on this instrument.

        A raw per-detector score drop. NOT a probability; never compared or
        averaged across detectors (params.yaml `reporting`).
        """
        p = self.p_fake_batch([original, intervened])
        return float(p[0] - p[1])

    def necessity_margins(self, original, interventions: Iterable) -> List[float]:
        """NM for several interventions on one image, in one batched pass."""
        interventions = list(interventions)
        p = self.p_fake_batch([original] + interventions)
        return [float(p[0] - q) for q in p[1:]]

    def __repr__(self) -> str:
        state = "loaded" if self._detector is not None else "lazy"
        return f"SpatialInstrument({self.name!r}, {state})"

"""The spectral instrument: a frozen frequency detector, queried via spectral interventions.

The second certification instrument. Study 1B showed CLIP-family detectors are
near-blind to spectral cues (Δp≈0 under frequency interventions) while a
frequency detector separates at AUC 0.94–0.99 — so whole_face frequency/noise
claims, which spatial repair cannot touch, are certified here instead.

Thin adapter, like spatial.py: the model is the frozen `cec_freqnet` detector
registered in training/detectors/cec_instrument_detector.py; the interventions
are ported in training/networks/cec/spectral_ops.py. This file owns neither.

    from cec.instruments.spectral import SpectralInstrument

    inst = SpectralInstrument("freqnet")
    nm = inst.necessity_margin(orig_bgr, inst.intervene("spectral_notch", orig_bgr))

Pilot S (Task 5) decides whether this instrument is adopted; the pre-committed
fallback is a single instrument + spectral characterization.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

_TRAINING_DIR = Path(__file__).resolve().parents[2] / "training"
if str(_TRAINING_DIR) not in sys.path:
    sys.path.append(str(_TRAINING_DIR))

# Spectral detector candidates for Pilot S. freqnet is on disk; npr needs a
# download (pins.yaml). Names must match registered `cec_<name>` detectors.
SPECTRAL_INSTRUMENTS = ("freqnet", "npr")


class SpectralInstrument:
    """One frozen frequency detector plus the Study-1B intervention battery.

    Lazy-loads the model, so importing this module stays cheap.
    """

    def __init__(self, name: str = "freqnet", device: str | None = None):
        if name not in SPECTRAL_INSTRUMENTS:
            raise ValueError(
                f"unknown spectral instrument '{name}'. Choices: {SPECTRAL_INSTRUMENTS}"
            )
        self.name = name
        self._device = device
        self._detector = None

    @property
    def detector(self):
        if self._detector is None:
            from metrics.registry import DETECTOR  # noqa: E402
            import detectors  # noqa: F401,E402  (populates the registry)

            key = f"cec_{self.name}"
            if key not in DETECTOR.data:
                raise KeyError(
                    f"'{key}' is not registered. If this is 'npr', the weights are "
                    f"not on disk yet — see cec/registration/pins.yaml (🔴 download)."
                )
            self._detector = DETECTOR[key]({"device": self._device})
        return self._detector

    # -- interventions -------------------------------------------------------
    def intervene(self, intervention_type: str, image, reference=None):
        """Apply a named spectral intervention to a BGR image.

        residual_renormalize needs a reference crop (the paired real); the other
        two are self-contained. Returns a BGR image.
        """
        from networks.cec.spectral_ops import INTERVENTIONS  # noqa: E402

        if intervention_type not in INTERVENTIONS:
            raise KeyError(
                f"unknown intervention '{intervention_type}'. "
                f"Known: {sorted(INTERVENTIONS)}."
            )
        op = INTERVENTIONS[intervention_type]
        if intervention_type == "residual_renormalize":
            if reference is None:
                raise ValueError("residual_renormalize needs a reference crop")
            return op(image, reference)
        return op(image)

    # -- scoring -------------------------------------------------------------
    def p_fake(self, image) -> float:
        return float(self.p_fake_batch([image])[0])

    def p_fake_batch(self, images: Sequence, batch_size: int = 64) -> np.ndarray:
        return self.detector.p_fake(list(images), batch_size=batch_size)

    def necessity_margin(self, original, intervened) -> float:
        """NM = p_fake(original) - p_fake(intervened), on this instrument.

        Raw score drop; not a probability, never compared across detectors.
        """
        p = self.p_fake_batch([original, intervened])
        return float(p[0] - p[1])

    def necessity_margins(self, original, intervened_list) -> list[float]:
        """NM for several interventions on one image, in one batched pass."""
        intervened_list = list(intervened_list)
        p = self.p_fake_batch([original] + intervened_list)
        return [float(p[0] - q) for q in p[1:]]

    def __repr__(self) -> str:
        state = "loaded" if self._detector is not None else "lazy"
        return f"SpectralInstrument({self.name!r}, {state})"

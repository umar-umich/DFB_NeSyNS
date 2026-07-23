"""
detectors/cec_instrument_detector.py
====================================
CEC instrument — a frozen SOTA detector, wrapped for counterfactual
certification.

Forward path (one path only; there is nothing to branch on):
    image -> GenD transform -> frozen detector -> logits [real, fake]
          -> softmax -> p(fake)

The gate calls this twice per claim — once on the original image, once on the
intervened image — and reads the drop as the Necessity Margin. Instruments are
registered from one class, differing only by checkpoint:

    spatial  cec_effort · cec_fsfm · cec_gend · cec_forada
             (CLIP-family; certify by repairing the cited region)
    spectral cec_freqnet
             (frequency detector; certify by spectral interventions on whole_face)

Checkpoints, preprocessing rationale, and the frozen anchors live in
`cec/registration/detectors.yaml`, which is the single source of truth; this
file reads it rather than restating any path.

WHY THIS IS FROZEN. These detectors own the verdict and the accuracy numbers.
Nothing here is ever trained (CURRENT_STATE rule 4: detectors and instruments
stay frozen; only the proposer is preference-tuned). `AbstractDetector` is a
*training* abstraction, so its training-only contract is implemented as a hard
refusal — see `build_loss` below. The abstract base then enforces the freeze
invariant instead of fighting it.

PREPROCESSING. Each detector is wrapped through GenD's own
`model.get_preprocessing()`, NOT DeepfakeBench's. This amends Implementation v2
Task 2, whose "use DFB preprocessing" instruction is unsatisfiable for three of
the four (they are not in the DFB registry) and would silently renumber every
frozen anchor for all four. Decided by Umar 2026-07-16; see docs/cec/LOG.md.
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks.cec import GEND_ROOT, build_gend_model, pfake_batch, to_pil_rgb
from networks.cec.npr import build_npr_model

logger = logging.getLogger(__name__)

# cec/registration/detectors.yaml, relative to this file (training/detectors/).
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRATION = _REPO_ROOT / "cec" / "registration" / "detectors.yaml"

_FROZEN_ERROR = (
    "CEC instruments are frozen and are never trained. CURRENT_STATE rule 4: "
    "detectors and instruments stay frozen; only the proposer is preference-tuned "
    "(offline DPO, after the audit freeze). If you are here because you wanted to "
    "train this, you want cec/dpo/ instead."
)


def load_registration(path=None):
    """Read cec/registration/detectors.yaml. Plain YAML — no import of the cec package."""
    path = Path(path) if path else DEFAULT_REGISTRATION
    if not path.exists():
        raise FileNotFoundError(f"CEC detector registration missing: {path}")
    with path.open() as fh:
        return yaml.safe_load(fh)


class CECInstrumentDetector(AbstractDetector):
    """One frozen GenD detector, exposed as a DeepfakeBench detector.

    config keys:
        instrument        one of effort / fsfm / gend / forada     (required)
        registration_file path to detectors.yaml    (optional; defaults as above)
        gend_root         GenD checkout             (optional)
        device            torch device              (optional; default cuda if available)
    """

    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        self.config = config

        self.instrument = config.get("instrument")
        if not self.instrument:
            raise ValueError(
                "CECInstrumentDetector needs config['instrument'], one of "
                "effort / fsfm / gend / forada."
            )

        registration = load_registration(config.get("registration_file"))
        detectors = registration["detectors"]
        if self.instrument not in detectors:
            raise KeyError(
                f"unknown instrument '{self.instrument}'. "
                f"Registered: {sorted(detectors)}."
            )
        self.spec = detectors[self.instrument]
        self.gend_root = Path(config.get("gend_root", GEND_ROOT))

        # The anchor this instrument must keep reproducing (ledger item 6).
        self.anchor_gt_repair_drop_median = self.spec.get("anchor_gt_repair_drop_median")

        self.device = torch.device(
            config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.backbone, self.transform = self.build_backbone(config)

    # -- construction --------------------------------------------------------
    def build_backbone(self, config):
        # loader selects the construction path: 'gend' (default) delegates to the
        # GenD registry; 'npr' loads the native NPR module (not a GenD model).
        loader = self.spec.get("loader", "gend")
        if loader == "gend":
            model, transform = build_gend_model(self.spec, self.device, self.gend_root)
        elif loader == "npr":
            model, transform = build_npr_model(self.spec, self.device)
        else:
            raise ValueError(
                f"unknown loader '{loader}' for instrument '{self.instrument}'. "
                f"Known: 'gend', 'npr'."
            )
        logger.info(
            "CEC instrument '%s' loaded frozen from %s (anchor gt-repair drop %.3f)",
            self.instrument, self.spec["checkpoint"],
            self.anchor_gt_repair_drop_median or float("nan"),
        )
        return model, transform

    # -- inference -----------------------------------------------------------
    def features(self, data_dict):
        """Logits [real, fake] for a preprocessed batch under data_dict['image']."""
        return self.backbone(data_dict["image"]).logits_labels

    def classifier(self, features):
        """Identity: the frozen detector's head already produced the logits."""
        return features

    def forward(self, data_dict, inference=True):
        logits = self.classifier(self.features(data_dict))
        prob = torch.softmax(logits, dim=1)[:, 1]
        return {"cls": logits, "prob": prob, "feat": logits}

    @torch.no_grad()
    def p_fake(self, images, batch_size=64):
        """p(fake) for raw images (BGR uint8 arrays or PIL) — the gate's entry point."""
        return pfake_batch(
            self.backbone, self.transform, images, self.device, batch_size=batch_size
        )

    def preprocess(self, image):
        """GenD's own transform: PIL/BGR image -> model-ready tensor (no batch dim)."""
        return self.transform(to_pil_rgb(image))

    # -- training contract: refused on purpose -------------------------------
    def build_loss(self, config):
        raise NotImplementedError(_FROZEN_ERROR)

    def get_losses(self, data_dict, pred_dict):
        raise NotImplementedError(_FROZEN_ERROR)

    def get_train_metrics(self, data_dict, pred_dict):
        raise NotImplementedError(_FROZEN_ERROR)

    def train(self, mode=True):
        """Refuse train mode. A frozen instrument in train() would drift BN/dropout."""
        if mode:
            raise NotImplementedError(_FROZEN_ERROR)
        return super().train(False)


def _register(instrument):
    """Register one thin subclass per instrument, so DFB configs can name them."""

    @DETECTOR.register_module(module_name=f"cec_{instrument}")
    class _Instrument(CECInstrumentDetector):
        __doc__ = f"Frozen CEC spatial instrument: {instrument}."

        def __init__(self, config=None):
            config = dict(config or {})
            config.setdefault("instrument", instrument)
            super().__init__(config)

    _Instrument.__name__ = f"CEC{instrument.capitalize()}Detector"
    _Instrument.__qualname__ = _Instrument.__name__
    return _Instrument


CECEffortDetector = _register("effort")
CECFsfmDetector = _register("fsfm")
CECGendDetector = _register("gend")
CECForadaDetector = _register("forada")
CECFreqnetDetector = _register("freqnet")
CECNprDetector = _register("npr")

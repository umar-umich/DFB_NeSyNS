"""Branch D — multi-rate forensic response `e_rate` (brief Stage 2).

    x -> frozen FS-VFM ViT-L/16                       [frozen encoder, shared with Branch B]
      -> frozen MR-VAE, evaluated at K betas          [frozen mechanism, never trains]
      -> R(x) = [r_beta_1, ..., r_beta_K]             the rate-distortion response
      -> standardize with FF++ REAL statistics        [frozen calibrator]
      -> H_rate (the only trainable part)             -> e_rate (B, 2)

Why the operator sits on FS-VFM features and not the anchor's
--------------------------------------------------------------
The brief says to attach the Phase-1 MR-VAE frozen. The MR-VAE is a **feature-space** operator: it
consumes a 64-d projected semantic feature. Its two existing checkpoints live in DiCoME's LoRA-CLIP
feature space and in the D-ladder backbone's, and **neither is V1's Branch-A space** — which does
not exist until Stage 4 trains its LoRA. "Frozen before detector training" and "operates on the
detector's own feature" cannot both hold for a feature-space operator, so one of them has to give.

Hosting it on the frozen FS-VFM embedding keeps *frozen* and gives up *the anchor's feature*. That
is the right trade for this branch: FS-VFM is fixed, already cached, and independent of Stage 4, so
R(x) measures how compressible the **image's face representation** is under a bona-fide-trained
rate-distortion model — a property of the image, not of the classifier's current opinion of it.
Hosting it on Branch A instead would make the response a function of a representation that was
itself trained to separate real from fake, and "the anchor's feature is hard to compress" would be
a restatement of the anchor. See `phase2/REPO_MAP.md` item 5 for the alternatives considered.

Polarity is learned, never assumed
-----------------------------------
The brief is explicit: do not hardcode "larger residual means fake". `H_rate` reads the whole
standardized K-vector, so it can map either direction — or a *shape* of the curve that is neither
monotone direction — onto either class. Nothing in this module encodes a sign.

What the head can see, and what it deliberately cannot
-------------------------------------------------------
Inputs are the K distortions and their finite differences across log-beta (the discrete slope of
the rate-distortion curve). The slopes are supplied because the brief's Stage 2.3 question is
whether R(x) carries *structure* rather than magnitude, and a linear head over raw distortions can
only read magnitude and offset. Nothing identifying the dataset, generator or family enters.

Availability
------------
Non-finite response for a sample sets `valid = 0`, and the caller turns that into a vacuous
opinion. The head never invents an evidence value for a sample the operator could not process.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .projectors import BETA_GRID, MRVAEProjector
from .reference import ResidualCalibrator

logger = logging.getLogger(__name__)


class FrozenRateOperator(nn.Module):
    """The MR-VAE, frozen, evaluated at the exact beta grid it was trained with.

    The grid is read from the artifact rather than from `projectors.BETA_GRID`, and mismatches
    raise. The response is only interpretable at the rates the FiLM conditioning actually saw:
    evaluating at betas outside the training range extrapolates the conditioning network, and the
    resulting curve would look like a rate response while measuring extrapolation error.
    """

    def __init__(self, artifact: Path | str):
        super().__init__()
        blob = torch.load(str(artifact), map_location="cpu", weights_only=False)
        self.feature_dim = int(blob["feature_dim"])
        self.latent_dim = int(blob["latent_dim"])
        # hidden_dim travels WITH the weights and is read back here. Rebuilding at the module
        # default and then loading fails outright — better than loading silently, but it blocks
        # the branch, so the width has to come from the artifact rather than from a default that
        # happened to match when the fit was run.
        self.hidden_dim = int(blob.get("hidden_dim") or 0) or None
        self.beta_grid: tuple[float, ...] = tuple(float(b) for b in blob["beta_grid"])
        self.provenance = blob.get("provenance", {})

        if tuple(round(b, 6) for b in self.beta_grid) != tuple(round(b, 6) for b in BETA_GRID):
            logger.warning(
                "artifact beta grid %s differs from projectors.BETA_GRID %s — using the "
                "ARTIFACT's grid, because the response is only defined at the rates this "
                "operator was trained on", self.beta_grid, BETA_GRID)

        self.projector = MRVAEProjector(self.feature_dim, self.latent_dim,
                                        hidden_dim=self.hidden_dim)
        self.projector.load_state_dict(blob["projector_state"])
        self.freeze()

    def freeze(self) -> "FrozenRateOperator":
        for p in self.projector.parameters():
            p.requires_grad_(False)
        self.projector.eval()
        return self

    def train(self, mode: bool = True) -> "FrozenRateOperator":
        """Stay in eval under `model.train()`. Without this the FiLM/BN state would drift and the
        response would move over a run for reasons unrelated to the data."""
        super().train(mode)
        self.projector.eval()
        return self

    def assert_frozen(self) -> None:
        trainable = [n for n, p in self.projector.named_parameters() if p.requires_grad]
        if trainable:
            raise RuntimeError(f"MR-VAE parameters are trainable: {trainable[:5]}")
        if self.projector.training:
            raise RuntimeError("MR-VAE is in train mode; its response would drift during a run")

    @torch.no_grad()
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """(B, feature_dim) -> R(x) (B, K), per-sample distortion at each beta in the grid."""
        return self.projector.rate_distortion_response(features)


class RateEvidenceBranch(nn.Module):
    """`R(x)` -> frozen standardization -> a small head -> non-negative evidence `(B, 2)`."""

    def __init__(self, operator: FrozenRateOperator, hidden_dim: int = 32,
                 use_slopes: bool = True):
        super().__init__()
        self.operator = operator
        self.use_slopes = use_slopes
        self.k = len(operator.beta_grid)
        self.calibrator = ResidualCalibrator(self.k)
        in_dim = self.k + (self.k - 1 if use_slopes else 0)
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2))
        # log-beta spacing, held as a buffer so the slopes are a property of the operator's grid
        self.register_buffer(
            "log_beta", torch.log(torch.tensor(operator.beta_grid, dtype=torch.float32)))

    def load_calibrator(self, state: dict) -> "RateEvidenceBranch":
        self.calibrator.load_state_dict(state)
        for p in self.calibrator.parameters():
            p.requires_grad_(False)
        return self

    def assert_frozen(self) -> None:
        self.operator.assert_frozen()

    def _features(self, response: torch.Tensor) -> torch.Tensor:
        z = self.calibrator(response)
        if not self.use_slopes:
            return z
        # discrete d(distortion)/d(log beta): the SHAPE of the curve, which a linear map over
        # raw distortions cannot express
        slopes = (z[:, 1:] - z[:, :-1]) / (self.log_beta[1:] - self.log_beta[:-1]).clamp_min(1e-6)
        return torch.cat([z, slopes], dim=1)

    def forward(self, encoder_features: torch.Tensor) -> dict:
        response = self.operator(encoder_features)
        valid = torch.isfinite(response).all(dim=1)
        safe = torch.nan_to_num(response, nan=0.0, posinf=0.0, neginf=0.0)
        feature = self._features(safe)
        return {
            "evidence": F.softplus(self.head(feature)),
            "feature": feature,
            "valid": valid,
            "raw_stats": {f"rate_r_{i}": response[:, i] for i in range(self.k)},
            "response": response,
        }


def build_rate_branch(artifact: Path | str, hidden_dim: int = 32,
                      use_slopes: bool = True) -> RateEvidenceBranch:
    """Build the branch and load the frozen standardization written beside the operator."""
    operator = FrozenRateOperator(artifact)
    branch = RateEvidenceBranch(operator, hidden_dim=hidden_dim, use_slopes=use_slopes)
    blob = torch.load(str(artifact), map_location="cpu", weights_only=False)
    if "calibrator_state" in blob:
        branch.load_calibrator(blob["calibrator_state"])
    else:
        raise SystemExit(
            f"{artifact} has no calibrator_state. The head would then see raw distortions whose "
            f"scale depends on the encoder and the fit, and 'standardized against authentic "
            f"video' would be a claim the artifact does not support.")
    branch.assert_frozen()
    return branch

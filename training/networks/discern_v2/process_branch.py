"""Branch C — generative-process evidence `e_proc` (V1 spec §5).

Wraps the validated P2a frozen-VAE reconstruction operator and turns its error statistics into
Dirichlet evidence:

    x (aligned crop, [0,1] pixels)
      -> frozen SDXL/LDM first-stage AE, VAE-native normalization   [operator, §5]
      -> K reconstruction-error statistics
      -> standardize with FF++ REAL TRAINING statistics             [frozen calibrator, §5]
      -> H_proc (the only trainable part)                           -> e_proc (B, 2)

Three properties are structural rather than conventional:

* **The VAE never trains.** The operator freezes it and keeps it in eval even under
  `model.train()`; without that, its norm layers would drift and the residual would move over a
  run for reasons unrelated to the data.
* **Polarity is learned, never assumed.** §5 is explicit: AEROBLADE shows LDM-generated content
  can reconstruct *more* faithfully, so a LOW residual can be evidence of generation. The head
  may map either direction to fake; nothing here encodes "large error = fake".
* **Statistics are standardized with authentic-training statistics only**, held as buffers so
  they cannot receive gradient, for the same reason the reference branch is: the head should see
  "how unusual is this reconstruction relative to authentic video", and folding fake statistics
  into the yardstick shrinks exactly the deviation being measured.

Input normalization (§6)
------------------------
The operator un-normalizes its input before handing [-1, 1] to the AE, so it must be told which
normalization produced that input. This branch feeds it `raw_frames` ([0, 1] pixels) and sets
mean 0 / std 1, which makes the un-normalization an identity and keeps the VAE on its own native
scale. Passing CLIP-normalized `spatial_frames` with CLIP's constants would also work, but §6's
rule is that branches start from the same source frame and normalize independently — and the
identity path has one less place to get the constants wrong.

Availability (§14.1)
--------------------
A catastrophic operator failure (non-finite statistics) sets `branch_valid_proc = 0`. The caller
turns that into a vacuous opinion at the opinion level; this branch never invents an evidence
value for a sample it could not process, because the head has no way to express "I could not
look at this" in evidence space.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dirichlet import DirichletState, to_dirichlet
from .process_residual import STAT_NAMES, ProcessResidualOperator, build_operator
from .reference import ResidualCalibrator

logger = logging.getLogger(__name__)

# raw_frames are [0, 1], so the operator's un-normalisation must be the identity
RAW_MEAN = (0.0, 0.0, 0.0)
RAW_STD = (1.0, 1.0, 1.0)


class ProcessEvidenceBranch(nn.Module):
    """Specialist 2: is this observation consistent with an LDM decode?"""

    name = "proc"

    def __init__(self, vae_path: str | Path, resolution: int = 256,
                 num_classes: int = 2, hidden_dim: int = 32,
                 stats_artifact: str | Path | None = None,
                 input_mean: tuple[float, float, float] = RAW_MEAN,
                 input_std: tuple[float, float, float] = RAW_STD):
        super().__init__()
        self.operator: ProcessResidualOperator = build_operator(
            vae_path=vae_path, resolution=resolution,
            input_mean=input_mean, input_std=input_std)
        self.n_stats = self.operator.n_stats
        self.stat_names = STAT_NAMES

        self.calibrator = ResidualCalibrator(self.n_stats)
        if stats_artifact is not None:
            self.load_stats(stats_artifact)

        # The only trainable part. Deliberately small: it reads six hand-named statistics, and
        # capacity here would let it memorise FF++ rather than read the process signal.
        self.head = nn.Sequential(
            nn.Linear(self.n_stats, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_classes))

    # ------------------------------------------------------------------ calibration

    def load_stats(self, artifact: str | Path) -> "ProcessEvidenceBranch":
        """Load FF++ authentic-training statistics fit offline by `fit_process_stats.py`."""
        blob = torch.load(str(artifact), map_location="cpu", weights_only=False)
        if int(blob["n_stats"]) != self.n_stats:
            raise ValueError(
                f"stats artifact has {blob['n_stats']} statistics, operator produces "
                f"{self.n_stats}; they describe different operators")
        self.calibrator.load_state_dict(blob["calibrator_state"])
        self.stats_provenance = blob.get("provenance", {})
        logger.info(f"  Process branch: calibrated on {self.stats_provenance.get('n_real', '?')} "
                    f"FF++ authentic training frames")
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        self.operator.eval()       # the operator enforces this too; belt and braces
        self.calibrator.eval()
        return self

    def assert_frozen(self) -> None:
        live = [n for n, p in self.operator.named_parameters() if p.requires_grad]
        if live:
            raise RuntimeError(f"the process VAE has trainable parameters: {live[:5]}")
        live_cal = [n for n, p in self.calibrator.named_parameters() if p.requires_grad]
        if live_cal:
            raise RuntimeError(f"the process calibrator has trainable params: {live_cal}")
        if self.operator.training:
            raise RuntimeError("the process VAE is in train mode; it must stay in eval")

    # ------------------------------------------------------------------ forward

    def statistics(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) in [0, 1] -> (B, K) raw error statistics. No gradient by construction."""
        with torch.no_grad():
            return self.operator(images)

    def forward(self, images: torch.Tensor) -> dict:
        stats = self.statistics(images)

        # §14.1: a sample the operator could not process gets flagged, not guessed at. The
        # statistics are neutralised so a NaN cannot propagate into the head and poison the
        # whole batch's gradient — the flag, not the value, is what the caller acts on.
        valid = torch.isfinite(stats).all(dim=1)
        stats = torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)

        calibrated = self.calibrator(stats)
        evidence = F.softplus(self.head(calibrated))
        return {
            "evidence": evidence,                 # (B, K) non-negative
            "feature": calibrated,                # standardized statistics, for the gate
            "valid": valid,                       # (B,) bool -> branch_valid_proc
            "raw_stats": stats,                   # for the §20 domain-detector audit
        }

    def dirichlet(self, images: torch.Tensor) -> DirichletState:
        return to_dirichlet(self.forward(images)["evidence"])

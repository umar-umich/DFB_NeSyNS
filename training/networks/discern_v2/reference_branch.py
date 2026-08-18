"""ReferenceEvidenceBranch — Specialist 1 in the three-slot architecture.

Wraps a Stage-I artifact (a frozen reals-only reference plus its residual calibrator) and
turns the residual into Dirichlet evidence. The only trainable part is the evidence head:

    f_0 --[frozen reference]--> P(f_0) --> r = f_0 - P(f_0)
        --[frozen calibrator]--> r_tilde --> [trainable head] --> e_ref

Three properties are structural rather than conventional:

* **The reference is loaded, never constructed fresh.** Stage II has no code path that can
  create an unfitted reference, so it cannot accidentally co-train one.
* **The head sees the *calibrated* residual**, standardized by authentic-training statistics.
  Without that the head's first layer has to absorb a scale that differs per arm, which makes
  the C0/C1/C2/C3 comparison partly a comparison of input scales.
* **Polarity is learned, not assumed.** The head may map either a large or a small residual to
  fake. For the process branch that is essential (AEROBLADE: LDM output can reconstruct *more*
  faithfully), and it costs nothing to hold the same discipline here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from .branches import BranchOutput, EvidenceBranch
from .reference import FrozenReference, ResidualCalibrator, build_reference

logger = logging.getLogger(__name__)


class ReferenceEvidenceBranch(EvidenceBranch):
    """Evidence from the distance between an observation and the authentic manifold."""

    name = "reference"

    def __init__(self, artifact_path: str | Path, hidden_dim: int = 64,
                 num_classes: int = 2, evidence_activation: str = "softplus",
                 input_dim: int | None = None, map_location: str = "cpu"):
        blob = torch.load(str(artifact_path), map_location=map_location, weights_only=False)
        feature_dim = int(blob["feature_dim"])
        latent_dim = int(blob["latent_dim"])
        # head input: the calibrated residual (D) plus its magnitude and angle
        super().__init__(in_dim=feature_dim + 2, hidden_dim=hidden_dim,
                         num_classes=num_classes, evidence_activation=evidence_activation)

        self.arm = str(blob["arm"])
        self.objective = str(blob["objective"])
        self.artifact_path = str(artifact_path)

        ref: FrozenReference = build_reference(self.arm, feature_dim, latent_dim)
        ref.load_state_dict(blob["reference_state"])
        ref.freeze()
        self.reference = ref

        cal = ResidualCalibrator(feature_dim)
        cal.load_state_dict(blob["calibrator_state"])
        self.calibrator = cal

        # DISCERN's visual feature may be wider than the space the reference was fit in. That
        # is NOT something to paper over with a learned projection here: the reference was fit
        # on a specific frozen feature space, and mapping into it with trainable weights would
        # reintroduce exactly the drift the frozen protocol exists to prevent.
        if input_dim is not None and input_dim != feature_dim:
            raise ValueError(
                f"reference was fit on {feature_dim}-d features but the branch is being fed "
                f"{input_dim}-d. Feed it the same frozen encoder space it was fit on (see "
                f"Task 0's dual-encoder option) rather than projecting into it — a trainable "
                f"map into the reference's space is a drifting encoder by another name.")
        logger.info(f"  Reference branch: arm={self.arm} objective={self.objective} "
                    f"dim={feature_dim} (frozen) from {Path(artifact_path).name}")

    def train(self, mode: bool = True):
        """Keep the frozen parts in eval regardless of the enclosing model's mode."""
        super().train(mode)
        self.reference.eval()
        self.calibrator.eval()
        return self

    def represent(self, f0: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # no_grad around the frozen path: it cannot learn, and excluding it from the graph
        # keeps the audit "did the reference receive gradient?" trivially true.
        with torch.no_grad():
            desc = self.reference(f0)
            r_tilde = self.calibrator(desc.residual)
        feats = torch.cat([r_tilde, desc.norm.unsqueeze(1), desc.angle.unsqueeze(1)], dim=1)
        return feats, {
            "reference_residual_norm": desc.norm,
            "reference_angle": desc.angle,
            "reference_residual": desc.residual,
        }

    def assert_frozen(self) -> None:
        """Re-check the protocol from the branch. Cheap, and called from the training path."""
        self.reference.assert_frozen_protocol()
        live = [n for n, p in self.calibrator.named_parameters() if p.requires_grad]
        if live:
            raise RuntimeError(f"residual calibrator has trainable params: {live}")


def build_reference_branch(cfg: dict) -> ReferenceEvidenceBranch:
    """Construct from the `discern_v2.reference` config block."""
    path = cfg.get("artifact_path")
    if not path:
        raise ValueError(
            "reference branch needs `artifact_path` pointing at a Stage-I artifact produced by "
            "analysis/discern_v2/fit_reference.py. There is deliberately no way to construct "
            "an unfitted reference here — Stage II must not be able to co-train one.")
    return ReferenceEvidenceBranch(
        artifact_path=path,
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        input_dim=(int(cfg["input_dim"]) if cfg.get("input_dim") else None),
    )

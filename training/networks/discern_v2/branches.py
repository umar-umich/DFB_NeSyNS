"""DISCERN v2 evidence branches — one contract, three implementations, all toggleable.

Every branch returns the same thing:

    evidence     (B, 2)   raw non-negative Dirichlet evidence, real/fake
    features     (...)    representation kept for analysis and the applicability gate
    diagnostics  {...}    branch-specific extras (residual norms, rate response, ...)

alpha / S / p / vacuity are NOT computed here. They come from `dirichlet.to_dirichlet`, so a
DiCoME-ported branch and a DISCERN-native branch can never drift apart on the definition of
vacuity. `BranchOutput.state` exposes them for callers that want the full evidential view.

Toggling
--------
Each branch is independently switchable from config, which is what makes the D1 sub-ablation
(D1-V visual only / D1-M manifold only / D1-VM both) a config switch rather than a code
change. `build_branches(cfg)` returns only the enabled ones and refuses to return an empty
set -- a run with every branch off would train a bare classifier and still be labelled D1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .dirichlet import DirichletState, logits_to_evidence, to_dirichlet
from .projectors import build_projector


@dataclass
class BranchOutput:
    """The common branch return. `state` is derived centrally, never by the branch."""

    evidence: torch.Tensor
    features: torch.Tensor
    diagnostics: dict = field(default_factory=dict)

    @property
    def state(self) -> DirichletState:
        return to_dirichlet(self.evidence)

    def as_dict(self) -> dict:
        """Flat dict view for the existing v1 fusion path, which expects `evidence`."""
        s = self.state
        return {"evidence": s.evidence, "alpha": s.alpha, "p": s.p, "vacuity": s.vacuity,
                "features": self.features, "diagnostics": self.diagnostics}


class EvidenceBranch(nn.Module):
    """Base class: an evidence head plus the shared contract.

    Subclasses build `self.head` mapping their representation to (B, 2) logits and implement
    `represent()`. Keeping the head here means every branch turns logits into evidence the
    same way.
    """

    name: str = "branch"

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_classes: int = 2,
                 evidence_activation: str = "softplus"):
        super().__init__()
        self.evidence_activation = evidence_activation
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_classes))

    def represent(self, *args, **kwargs) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError

    def forward(self, *args, **kwargs) -> BranchOutput:
        features, diagnostics = self.represent(*args, **kwargs)
        evidence = logits_to_evidence(self.head(features), self.evidence_activation)
        return BranchOutput(evidence=evidence, features=features, diagnostics=diagnostics)


class VisualEvidenceBranch(EvidenceBranch):
    """The CLIP visual branch — the strong baseline everything else is measured against.

    Deliberately thin: the Phase-1 instructions say to keep the CLIP branch unchanged, so
    this wraps an already-computed visual feature rather than re-implementing the backbone.
    The phase success criterion is that the manifold and process mechanisms transfer
    *without destroying this baseline*, so it must stay exactly what v1 had.
    """

    name = "visual"

    def represent(self, visual_feature: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return visual_feature, {}


class ManifoldEvidenceBranch(EvidenceBranch):
    """Manifold-residual branch with a swappable projector.

    The mechanism (DiCoME's Geometric View Purification): a projector reconstructs a
    manifold-consistent feature `f_c` from the semantic feature `f_s`, and the residual
    `f_r = f_s - f_c` is what carries forensic signal — the part of the observation the
    natural-face manifold cannot explain.

    Which projector produces `f_c` is a config string. Nothing here assumes MR-VAE: the rate
    response is exported only when the projector advertises one, so selecting P1a or P1b
    later is a one-line config change.
    """

    name = "manifold"

    def __init__(self, feature_dim: int, latent_dim: int = 32, projector: str = "mr_vae",
                 hidden_dim: int = 64, num_classes: int = 2,
                 evidence_activation: str = "softplus", export_rate_response: bool = True,
                 input_dim: int | None = None):
        # the head reads the residual, which has the same width as the semantic feature
        super().__init__(in_dim=feature_dim, hidden_dim=hidden_dim, num_classes=num_classes,
                         evidence_activation=evidence_activation)
        # DISCERN's visual feature is 1024-d; DiCoME's f_s was 64-d and every ported
        # projector has a 32-unit hidden layer sized for that. Feeding 1024 straight in
        # would run the projector far outside the regime it was validated in (a 32x
        # bottleneck instead of 2x), so an input projection brings the feature down to
        # `feature_dim` first and the ported code then operates exactly as in the pilot.
        # When input_dim == feature_dim (or is omitted) this is an identity and the branch
        # is byte-equivalent to the pilot.
        self.input_proj = (nn.Linear(input_dim, feature_dim)
                           if input_dim is not None and input_dim != feature_dim
                           else nn.Identity())
        self.projector = build_projector(projector, feature_dim, latent_dim)
        self.projector_name = projector
        self.export_rate_response = export_rate_response

    def represent(self, semantic_feature: torch.Tensor) -> tuple[torch.Tensor, dict]:
        semantic_feature = self.input_proj(semantic_feature)
        z, mu, log_var, recon = self.projector(semantic_feature)
        residual = semantic_feature - recon

        diagnostics = {
            "z": z, "mu": mu, "log_var": log_var, "reconstruction": recon,
            "residual_norm": residual.norm(dim=1),
            # cosine distortion at the operating point: the scalar version of the response
            "distortion": 1.0 - torch.cosine_similarity(semantic_feature, recon, dim=1),
        }
        if self.export_rate_response and self.projector.has_rate_response:
            # no_grad: the response is a diagnostic read at K extra betas, not a training
            # path. Letting it backprop would train the projector K+1 times per step.
            with torch.no_grad():
                diagnostics["rate_response"] = self.projector.rate_distortion_response(
                    semantic_feature)
        return residual, diagnostics

    def extra_loss(self, semantic_feature: torch.Tensor, dataset_size: int):
        """Projector-specific loss (beta-TCVAE's TC decomposition), or None.

        Applies the same input projection as `represent`, so the loss is computed on the
        tensor the projector actually saw rather than the raw backbone feature.
        """
        semantic_feature = self.input_proj(semantic_feature)
        z, mu, log_var, _ = self.projector(semantic_feature)
        return self.projector.extra_loss(semantic_feature, z, mu, log_var, dataset_size)


class ProcessEvidenceBranch(EvidenceBranch):
    """P2a process-residual branch: AEROBLADE-style LDM first-stage reconstruction residual.

    Asks a different question from CLIP's: not "does this look manipulated?" but "was this
    image ever decoded by a latent diffusion model?". That difference is the entire reason it
    is a third view rather than more of the same evidence.

    The operator is frozen and produces K error-map statistics; only this head trains.
    """

    name = "process"

    def __init__(self, operator: nn.Module, stat_dim: int, hidden_dim: int = 64,
                 num_classes: int = 2, evidence_activation: str = "softplus"):
        super().__init__(in_dim=stat_dim, hidden_dim=hidden_dim, num_classes=num_classes,
                         evidence_activation=evidence_activation)
        self.operator = operator

    def represent(self, images: torch.Tensor) -> tuple[torch.Tensor, dict]:
        stats = self.operator(images)
        return stats, {"process_stats": stats}


def build_branches(cfg: dict) -> nn.ModuleDict:
    """Construct exactly the branches the config enables.

    cfg shape (see the D-ladder configs):

        discern_v2:
          visual:   {enabled: true,  feature_dim: 512}
          manifold: {enabled: true,  feature_dim: 64, latent_dim: 32, projector: mr_vae}
          process:  {enabled: true,  vae_path: ..., resolution: 256}

    Refuses an all-off configuration: that would silently train a bare classifier while
    still being labelled a D-ladder rung.
    """
    v2 = cfg.get("discern_v2", cfg)
    branches: dict[str, nn.Module] = {}

    vis = v2.get("visual", {})
    if vis.get("enabled", False):
        branches["visual"] = VisualEvidenceBranch(in_dim=int(vis["feature_dim"]))

    man = v2.get("manifold", {})
    if man.get("enabled", False):
        branches["manifold"] = ManifoldEvidenceBranch(
            feature_dim=int(man["feature_dim"]),
            latent_dim=int(man.get("latent_dim", 32)),
            projector=str(man.get("projector", "mr_vae")),
            export_rate_response=bool(man.get("export_rate_response", True)),
            input_dim=(int(man["input_dim"]) if man.get("input_dim") else None))

    proc = v2.get("process", {})
    if proc.get("enabled", False):
        from .process_residual import build_operator
        op = build_operator(vae_path=proc.get("vae_path"),
                            resolution=int(proc.get("resolution", 256)),
                            input_mean=tuple(proc.get("input_mean", (0.48145466, 0.4578275,
                                                                     0.40821073))),
                            input_std=tuple(proc.get("input_std", (0.26862954, 0.26130258,
                                                                   0.27577711))))
        branches["process"] = ProcessEvidenceBranch(operator=op, stat_dim=op.n_stats)

    if not branches:
        raise ValueError(
            "no DISCERN v2 branch is enabled. An all-off config trains a bare classifier "
            "that would still be reported as a D-ladder rung; enable at least one of "
            "visual / manifold / process.")
    return nn.ModuleDict(branches)

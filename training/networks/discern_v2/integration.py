"""Wiring the DISCERN v2 branches into the existing detector.

Kept in its own module, and additive by construction, so the v1 forward path is provably
untouched: with `manifold_v2` and `process_v2` both false, `build_v2_stack` returns None and
the detector's behaviour is bit-identical to before this file existed. That property is what
makes D0 (v1 reproduction) and D1-D3 comparable at all, so it is tested rather than asserted.

How the v2 evidence joins the fused Dirichlet
---------------------------------------------
The v1 fusion (`EvidenceFusion`) already produces `total_evidence` from
spatial + concept + causal. The v2 branches are added on top with their own learned scalar
gates, mirroring v1's static-gate arithmetic exactly:

    total_evidence <- total_evidence + sigmoid(gate_b) * evidence_b

Two reasons for that rather than extending EvidenceFusion itself:

* v1 code stays untouched, so no D0/D1 comparison can be contaminated by a refactor;
* the gates start near zero (`gate_init = -2.0`, sigmoid ~= 0.12), so a freshly enabled v2
  branch begins as a small perturbation of the v1 system and has to earn its weight. A
  branch injected at full strength would swamp the CLIP baseline in early epochs, which is
  precisely the "collapse of the CLIP baseline" the D3 gate is meant to detect.

alpha/S/p/u are NOT recomputed here. The detector derives them from the fused evidence in
one place, as it already does.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from .branches import build_branches

logger = logging.getLogger(__name__)


class DiscernV2Stack(nn.Module):
    """The enabled v2 branches plus their fusion gates."""

    def __init__(self, cfg: dict, gate_init: float = -2.0):
        super().__init__()
        self.branches = build_branches(cfg)
        self.gates = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(float(gate_init)))
            for name in self.branches})
        v2 = cfg.get("discern_v2", cfg)
        self.manifold_input = str(v2.get("manifold", {}).get("input", "projected"))
        logger.info(f"  DISCERN v2      : branches={list(self.branches)} "
                    f"(gate_init={gate_init}, manifold_input={self.manifold_input})")

    def forward(self, total_evidence: torch.Tensor, *, visual_feature: torch.Tensor,
                images: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        """Add gated v2 evidence to the already-fused v1 evidence.

        Returns (total_evidence, diagnostics). Diagnostics carry each branch's raw evidence
        and gate so the per-sample logger and the applicability gate can read them without
        re-running the branches.
        """
        diag: dict = {}
        for name, branch in self.branches.items():
            if name == "manifold":
                out = branch(visual_feature)
            elif name == "process":
                if images is None:
                    raise ValueError(
                        "process_v2 is enabled but no images were passed to the v2 stack; "
                        "the process residual is computed from pixels, not features.")
                out = branch(images)
            elif name == "visual":
                # The v1 spatial head already supplies visual evidence; enabling the v2
                # visual branch on top would double-count it. D1-V is expressed by turning
                # the OTHER v2 branches off, not by adding a second visual head.
                continue
            else:
                raise KeyError(f"unhandled v2 branch {name!r}")

            gate = torch.sigmoid(self.gates[name])
            total_evidence = total_evidence + gate * out.evidence
            diag[f"{name}_evidence"] = out.evidence
            diag[f"{name}_gate"] = gate.detach()
            state = out.state
            diag[f"{name}_vacuity"] = state.vacuity.detach()
            for k, v in out.diagnostics.items():
                if torch.is_tensor(v):
                    diag[f"{name}_{k}"] = v.detach()
        return total_evidence, diag

    def extra_losses(self, visual_feature: torch.Tensor, dataset_size: int) -> dict:
        """Projector-specific loss terms (currently only beta-TCVAE's TC decomposition)."""
        out = {}
        # nn.ModuleDict supports __contains__/__getitem__ but not .get()
        if "manifold" in self.branches:
            term = self.branches["manifold"].extra_loss(visual_feature, dataset_size)
            if term is not None:
                out["manifold_tc"] = term
        return out


def build_v2_stack(config: dict, gate_init: float = -2.0) -> DiscernV2Stack | None:
    """Return the v2 stack, or None when no v2 mechanism is enabled.

    Returning None rather than an empty stack is deliberate: the detector can then guard the
    whole v2 path with `if self.v2 is not None`, and a v1 run executes exactly the code it
    executed before, with no extra tensor ops and no extra parameters in the optimizer.
    """
    if not (config.get("manifold_v2", False) or config.get("process_v2", False)):
        return None

    v2cfg = dict(config.get("discern_v2", {}) or {})
    # the top-level flags are the master switches; a branch section cannot turn itself on
    # behind their back, so the ladder configs remain readable at a glance
    if not config.get("manifold_v2", False) and "manifold" in v2cfg:
        v2cfg["manifold"] = {**v2cfg["manifold"], "enabled": False}
    if not config.get("process_v2", False) and "process" in v2cfg:
        v2cfg["process"] = {**v2cfg["process"], "enabled": False}
    # the v1 spatial head already provides visual evidence
    if "visual" in v2cfg:
        v2cfg["visual"] = {**v2cfg["visual"], "enabled": False}

    if not any((v2cfg.get(k) or {}).get("enabled") for k in ("manifold", "process")):
        return None
    return DiscernV2Stack({"discern_v2": v2cfg}, gate_init=gate_init)

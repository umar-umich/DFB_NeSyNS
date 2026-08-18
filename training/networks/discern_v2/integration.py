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

from .applicability import ApplicabilityGate
from .branches import build_branches
from .dirichlet import to_dirichlet

logger = logging.getLogger(__name__)

# The v2 branches the applicability gate may route over. The visual branch is excluded
# because it IS the baseline the gate measures everything against.
ROUTABLE = ("manifold", "process")


class DiscernV2Stack(nn.Module):
    """The enabled v2 branches plus their fusion gates.

    With `discern_v2.applicability.enabled` false (D0-D3) the forward pass is exactly what
    it was before D4 existed: no gate is constructed, no extra parameters enter the
    optimizer, and every branch contributes unconditionally. D4 is the only rung that
    changes behaviour here, which is what keeps D4 - D3 a clean measurement of the gate.
    """

    def __init__(self, cfg: dict, gate_init: float = -2.0):
        super().__init__()
        self.branches = build_branches(cfg)
        self.gates = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(float(gate_init)))
            for name in self.branches})
        v2 = cfg.get("discern_v2", cfg)
        self.manifold_input = str(v2.get("manifold", {}).get("input", "projected"))

        app_cfg = dict(v2.get("applicability", {}) or {})
        self.applicability: ApplicabilityGate | None = None
        if app_cfg.pop("enabled", False):
            routable = [n for n in self.branches if n in ROUTABLE]
            if not routable:
                raise ValueError(
                    "applicability gate is enabled but no routable branch is on. D4 is "
                    "D3 + routing; with neither manifold_v2 nor process_v2 there is "
                    "nothing to route and the rung would silently be D0.")
            self.applicability = ApplicabilityGate(routable, **app_cfg)

        logger.info(f"  DISCERN v2      : branches={list(self.branches)} "
                    f"(gate_init={gate_init}, manifold_input={self.manifold_input}, "
                    f"applicability={'on' if self.applicability is not None else 'off'})")

    def forward(self, total_evidence: torch.Tensor, *, visual_feature: torch.Tensor,
                images: torch.Tensor | None = None,
                baseline_evidence: torch.Tensor | None = None,
                labels: torch.Tensor | None = None,
                epoch: int | None = None) -> tuple[torch.Tensor, dict]:
        """Add gated v2 evidence to the already-fused v1 evidence.

        Returns (total_evidence, diagnostics). Diagnostics carry each branch's raw evidence
        and gate so the per-sample logger and the applicability gate can read them without
        re-running the branches.

        D4 only: `baseline_evidence` is the v1 spatial (visual) evidence, which is the
        reference the gate scores each specialist against; `labels` supervise the gate and
        are used ONLY as a target (see applicability.py); `epoch` drives the optional
        freeze. All three are ignored when the gate is off, so D0-D3 call this exactly as
        they did before.
        """
        diag: dict = {}
        base_state = None
        if self.applicability is not None:
            if baseline_evidence is None:
                raise ValueError(
                    "applicability gate is enabled but baseline_evidence is None; the gate "
                    "scores each specialist relative to the visual baseline and cannot be "
                    "evaluated without it.")
            base_state = to_dirichlet(baseline_evidence)
            if epoch is not None:
                # drives both the warmup window and the A2b freeze; set on eval passes too
                # so validation reflects the routing regime the model is actually in
                self.applicability.set_epoch(epoch)

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
            state = out.state
            contribution = gate * out.evidence

            # -- D4: per-sample routing -----------------------------------------------
            # The mask is hard and detached, so a branch cannot learn to talk its way past
            # the gate; the gate is trained by its own BCE term instead.
            if self.applicability is not None and name in self.applicability.specialists:
                # one head evaluation: the logit trains the gate, q routes it
                logit = self.applicability.logit(name, base_state, state)
                q = torch.sigmoid(logit)
                route = self.applicability.route(q)
                contribution = route.unsqueeze(1) * contribution
                diag[f"{name}_q"] = q.detach()
                diag[f"{name}_route"] = route
                if labels is not None:
                    # NOT detached: this is the term that trains the gate. It is the one
                    # entry in `diag` carrying gradient, hence the distinct key prefix.
                    diag[f"applicability_loss_{name}"] = self.applicability.gate_loss(
                        name, logit, base_state, state, labels)

            total_evidence = total_evidence + contribution
            diag[f"{name}_evidence"] = out.evidence
            diag[f"{name}_gate"] = gate.detach()
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

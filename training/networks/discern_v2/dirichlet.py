"""The single shared Dirichlet utility for DISCERN v2 evidence branches.

Every branch outputs raw `evidence`; the four evidential quantities are derived HERE and
nowhere else:

    alpha_b = e_b + 1
    S_b     = sum_k alpha_b,k
    p_b     = alpha_b / S_b
    u_b     = K / S_b

Why centrally
-------------
DISCERN v2 mixes branches ported out of DiCoME with branches native to DISCERN. If each
branch derived its own alpha/S/p/u, the two lineages would drift -- a different epsilon, a
sum that keeps or drops a dimension, a vacuity defined as `K/S` in one place and
`1 - max(p)` in another -- and the fusion would silently be combining quantities that are
not on the same scale. Nothing downstream can detect that, so it is prevented structurally
instead: branches return evidence, this module returns everything else.

This matches `losses/edl_loss.py::evidence_to_dirichlet` (alpha = e+1, S = sum alpha,
u = K/S) so v1 and v2 agree; that method stays where it is for the v1 loss path, and this
module is what the v2 branch contract uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DirichletState:
    """The four derived quantities, kept together so they cannot be recomputed piecemeal."""

    evidence: torch.Tensor   # (B, K) raw non-negative evidence, as produced by the branch
    alpha: torch.Tensor      # (B, K) concentration
    strength: torch.Tensor   # (B, 1) S
    p: torch.Tensor          # (B, K) expected probability
    vacuity: torch.Tensor    # (B,)   epistemic uncertainty, K / S

    @property
    def num_classes(self) -> int:
        return int(self.evidence.shape[1])

    @property
    def belief(self) -> torch.Tensor:
        """belief_k = e_k / S — the subjective-logic mass on each class (V1 spec §7).

        Lives here rather than in the fusion module so that `sum_k belief_k + u == 1` holds by
        construction from the same alpha/S every branch already shares: computing belief from a
        separately re-derived strength is how two modules end up with opinions that are each
        internally consistent and jointly incomparable.
        """
        return self.evidence / self.strength

    def fake_prob(self, fake_index: int = 1) -> torch.Tensor:
        """p(fake) as a (B,) tensor -- the scalar the analysis exports consume."""
        return self.p[:, fake_index]


def to_dirichlet(evidence: torch.Tensor) -> DirichletState:
    """evidence (B, K) -> the full evidential state.

    `evidence` must already be non-negative (branches apply softplus/exp/relu to their
    logits). It is clamped at zero rather than silently accepted: a negative entry would
    make alpha < 1 and produce a vacuity above K, which is not a valid Dirichlet and would
    poison any gate reading it.
    """
    if evidence.dim() != 2:
        raise ValueError(f"evidence must be (B, K), got {tuple(evidence.shape)}")
    evidence = evidence.clamp_min(0.0)
    alpha = evidence + 1.0
    strength = alpha.sum(dim=1, keepdim=True)
    p = alpha / strength
    vacuity = (evidence.shape[1] / strength).squeeze(1)
    return DirichletState(evidence=evidence, alpha=alpha, strength=strength, p=p,
                          vacuity=vacuity)


def logits_to_evidence(logits: torch.Tensor, activation: str = "softplus") -> torch.Tensor:
    """Map a head's raw logits to non-negative evidence.

    softplus is the default because it is smooth everywhere; relu zeroes the gradient for
    any class the head currently disbelieves, which makes a branch that starts out wrong
    slow to recover.
    """
    if activation == "softplus":
        return torch.nn.functional.softplus(logits)
    if activation == "relu":
        return torch.relu(logits)
    if activation == "exp":
        # clamped: exp on an untrained head overflows to inf and S becomes nan
        return torch.exp(logits.clamp(-10.0, 10.0))
    raise ValueError(f"unknown evidence activation {activation!r}")


def conflict(a: DirichletState, b: DirichletState, fake_index: int = 1) -> torch.Tensor:
    """Cross-branch conflict |p_a(fake) - p_b(fake)| -- a label-free gate feature.

    Lives here rather than in the gate so that A2's offline analysis and the online D4
    fusion compute conflict identically; a gate trained on one definition and deployed on
    another is a silent distribution shift.
    """
    return (a.fake_prob(fake_index) - b.fake_prob(fake_index)).abs()

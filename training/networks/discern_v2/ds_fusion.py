"""Subjective-logic opinions, applicability discounting, and Dempster-Shafer fusion.

V1 spec §14, §16, §17. This module is the reasoning layer — the part of DISCERN v2 that is
supposed to be the contribution — so every quantity it produces is defined once, here, and the
branches only supply evidence.

The pipeline, in the order the spec fixes it:

    e_b (>=0, from a branch)
      -> Opinion:      belief_b,k = e_b,k / S_b,  u_b = K / S_b          (§7)
      -> discount:     belief'_b  = q_b * belief_b,  u'_b = (1-q_b) + q_b*u_b   (§14.2)
      -> DS combine:   omega_sem (+) omega'_ref (+) omega'_proc           (§14.2)
      -> back to EDL:  S_f = K / u_f,  e_f = belief_f * S_f               (§16)
      -> reliability:  V, C, A                                            (§17)

Why discounting rather than a weighted evidence sum
---------------------------------------------------
A weighted *sum* answers "how much evidence?", and a specialist with weight 0 contributes
nothing — which in a two-class Dirichlet is indistinguishable from evidence for Real. Shafer
discounting answers "how much do we trust this opinion?": at `q_b = 0` the opinion becomes
*vacuous* (`u' = 1`), i.e. ignorance, so an inapplicable specialist cannot push the fusion
toward Real. That distinction is the whole point of §14.1's availability handling too.

Ported, not reinvented
----------------------
`_combine_two` is DiCoME's `DS_Combin` / `_combine_two_opinions`
(`/data/umar/Repos/DiCoME/src/model/core_model.py:57-99`), transcribed with the same conflict
definition and the same fusion algebra so our numbers stay comparable with the reproduced
baseline. Two deliberate differences, both required by the spec:

* **The `1 - conflict` divisor is epsilon-guarded.** DiCoME's is unclamped and divides by zero
  at total conflict, producing inf/NaN that propagates into every downstream reliability signal.
* **Combination is n-ary** by sequential folding, because V1 fuses three opinions rather than
  DiCoME's two. DS is associative in exact arithmetic; `assert_order_invariant` measures whether
  the epsilon-guarded floating-point version stays so, as §15 requires.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .dirichlet import DirichletState, to_dirichlet

# Guards the `1 - conflict` divisor. Not a tolerance to tune: it is the difference between a
# defined answer and inf/NaN when two confident opinions disagree completely, which is precisely
# the OOD situation this layer exists to handle.
CONFLICT_EPS = 1e-6


@dataclass(frozen=True)
class Opinion:
    """A binomial/multinomial subjective-logic opinion: per-class belief plus vacuity.

    Invariant, asserted rather than assumed: `sum_k belief_k + u == 1`.
    """

    belief: torch.Tensor      # (B, K)
    vacuity: torch.Tensor     # (B, 1)

    @property
    def num_classes(self) -> int:
        return int(self.belief.shape[1])

    @property
    def batch_size(self) -> int:
        return int(self.belief.shape[0])

    def assert_normalized(self, tol: float = 1e-4) -> "Opinion":
        total = self.belief.sum(dim=1, keepdim=True) + self.vacuity
        err = (total - 1.0).abs().max()
        if not torch.isfinite(err) or err > tol:
            raise ValueError(
                f"opinion is not normalized: max |sum(belief) + u - 1| = {float(err):.3e}. "
                f"An unnormalized opinion is not a subjective-logic opinion and every "
                f"downstream quantity (conflict, vacuity, V/C/A) silently stops meaning what "
                f"its name says.")
        return self

    def probability(self) -> torch.Tensor:
        """Projected probability p_k = belief_k + u/K — the decision-relevant distribution."""
        return self.belief + self.vacuity / self.num_classes

    def fake_prob(self, fake_index: int = 1) -> torch.Tensor:
        return self.probability()[:, fake_index]

    # -- conversions -------------------------------------------------------------------

    @staticmethod
    def from_evidence(evidence: torch.Tensor) -> "Opinion":
        """e (B, K) -> opinion, via the centralized Dirichlet utility (§7)."""
        return Opinion.from_dirichlet(to_dirichlet(evidence))

    @staticmethod
    def from_dirichlet(state: DirichletState) -> "Opinion":
        belief = state.evidence / state.strength
        return Opinion(belief=belief, vacuity=state.vacuity.unsqueeze(1))

    @staticmethod
    def vacuous(batch_size: int, num_classes: int = 2, *, device=None,
                dtype=torch.float32) -> "Opinion":
        """Total ignorance: no belief anywhere, `u = 1`.

        This is what §14.1 requires when a specialist has no valid input. It is NOT the same as
        zero evidence for one class or a uniform *probability*: a vacuous opinion says "I have
        nothing to contribute", and DS combination leaves the other opinions untouched by it.
        """
        return Opinion(
            belief=torch.zeros(batch_size, num_classes, device=device, dtype=dtype),
            vacuity=torch.ones(batch_size, 1, device=device, dtype=dtype))

    def to_dirichlet(self, eps: float = CONFLICT_EPS) -> DirichletState:
        """§16: S = K / u, e = belief * S, alpha = e + 1.

        `u` is clamped away from 0 before the division: a fully certain opinion implies infinite
        evidence, which is not representable and would make alpha inf.
        """
        u = self.vacuity.clamp(eps, 1.0)
        strength = self.num_classes / u
        evidence = self.belief * strength
        return to_dirichlet(evidence)


def discount(opinion: Opinion, q: torch.Tensor) -> Opinion:
    """§14.2 Shafer discounting by applicability `q in [0, 1]`.

        belief'_k = q * belief_k
        u'        = (1 - q) + q * u

    `q -> 1` keeps the opinion; `q -> 0` turns it into ignorance rather than into evidence for
    Real. Normalization is preserved exactly: sum_k q*b_k + (1-q) + q*u = q*(sum b + u) + 1 - q.
    """
    if q.dim() == 1:
        q = q.unsqueeze(1)
    if q.shape[0] != opinion.batch_size or q.shape[1] != 1:
        raise ValueError(f"q must be (B,) or (B, 1) with B={opinion.batch_size}, got {tuple(q.shape)}")
    q = q.clamp(0.0, 1.0)
    return Opinion(belief=q * opinion.belief,
                   vacuity=(1.0 - q) + q * opinion.vacuity)


def apply_validity(opinion: Opinion, valid: torch.Tensor) -> Opinion:
    """§14.1: an unavailable specialist becomes vacuous, at the opinion level.

    `valid` is (B,) or (B, 1), 1 = the branch had a usable input. Where it is 0 the opinion is
    replaced by total ignorance — not dropped, not zero-filled, and not left for the evidence
    head to interpret a garbage input as Real or Fake.
    """
    if valid.dim() == 1:
        valid = valid.unsqueeze(1)
    valid = valid.to(opinion.belief.dtype)
    vac = Opinion.vacuous(opinion.batch_size, opinion.num_classes,
                          device=opinion.belief.device, dtype=opinion.belief.dtype)
    return Opinion(belief=valid * opinion.belief + (1 - valid) * vac.belief,
                   vacuity=valid * opinion.vacuity + (1 - valid) * vac.vacuity)


def _combine_two(a: Opinion, b: Opinion, eps: float = CONFLICT_EPS
                 ) -> tuple[Opinion, torch.Tensor, torch.Tensor]:
    """One Dempster combination. Returns (fused opinion, raw conflict, degenerate mask).

    Ported from DiCoME `_combine_two_opinions`; the conflict is the off-diagonal mass of the
    belief outer product, `sum_{i != j} b_a,i * b_b,j`. `degenerate` marks samples where the
    combination was undefined (total conflict) and the vacuous fallback below was used.
    """
    if a.num_classes != b.num_classes:
        raise ValueError(f"class mismatch: {a.num_classes} vs {b.num_classes}")
    k = a.num_classes
    outer = torch.bmm(a.belief.view(-1, k, 1), b.belief.view(-1, 1, k))
    conflict = (outer.sum(dim=(1, 2)) - torch.diagonal(outer, dim1=-2, dim2=-1).sum(dim=1))
    conflict = conflict.clamp(0.0, 1.0).view(-1, 1)

    # the guard DiCoME lacks: at total conflict the divisor is 0 and every downstream signal
    # becomes NaN rather than "these two views disagree completely"
    denom = (1.0 - conflict).clamp_min(eps)

    belief = (a.belief * b.belief
              + a.belief * b.vacuity
              + b.belief * a.vacuity) / denom
    vacuity = (a.vacuity * b.vacuity) / denom

    # TOTAL conflict is not merely a small divisor — Dempster's rule is undefined there. Two
    # certain, opposite opinions put all mass on the empty set, so every numerator above is 0
    # too and the clamp alone would return a zero "opinion" that is not normalized. The
    # principled reading is that the views annihilate and nothing is known, so the result is the
    # VACUOUS opinion: V then reports "no usable evidence" while C (computed from the
    # pre-fusion opinions) still reports that informed experts disagreed. Collapsing both into
    # one number is exactly the failure §17 exists to avoid.
    mass = belief.sum(dim=1, keepdim=True) + vacuity
    degenerate = mass < 1e-6
    belief = torch.where(degenerate, torch.zeros_like(belief), belief / mass.clamp_min(eps))
    vacuity = torch.where(degenerate, torch.ones_like(vacuity), vacuity / mass.clamp_min(eps))

    fused = Opinion(belief=belief, vacuity=vacuity).assert_normalized()
    return fused, conflict.squeeze(1), degenerate.squeeze(1)


def ds_combine(opinions: list[Opinion], eps: float = CONFLICT_EPS
               ) -> tuple[Opinion, dict]:
    """Fuse n opinions by sequential Dempster combination (§14.2).

    Order is fixed by the caller and recorded in the diagnostics; §15 requires the order to be
    documented and its effect measured rather than assumed away.
    """
    if not opinions:
        raise ValueError("nothing to combine")
    fused = opinions[0].assert_normalized()
    conflicts, degenerates = [], []
    for nxt in opinions[1:]:
        fused, c, d = _combine_two(fused, nxt, eps)
        conflicts.append(c)
        degenerates.append(d)
    diagnostics = {
        "pairwise_conflict": torch.stack(conflicts, dim=1) if conflicts
        else torch.zeros(fused.batch_size, 0, device=fused.belief.device),
        # logged per sample: how often the fusion hit undefined total conflict is a property of
        # the data, and a silent fallback that nobody counts is indistinguishable from one that
        # never fires
        "degenerate": (torch.stack(degenerates, dim=1).any(dim=1) if degenerates
                       else torch.zeros(fused.batch_size, dtype=torch.bool,
                                        device=fused.belief.device)),
        "n_opinions": len(opinions),
    }
    # raw DS conflict, logged separately from the applicability-weighted C of §17 because they
    # answer different questions: this one ignores whether the disagreeing views were applicable
    diagnostics["ds_conflict_max"] = (diagnostics["pairwise_conflict"].max(dim=1).values
                                      if conflicts else torch.zeros(fused.batch_size,
                                                                    device=fused.belief.device))
    return fused, diagnostics


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Jensen-Shannon divergence in bits, so it is bounded in [0, 1] for any K.

    Bounded on purpose: `C` in §17 is a reliability signal that gets compared across samples and
    fed to a small risk model, and an unbounded nat-valued divergence would put that model's
    input on a scale that changes with K.
    """
    m = 0.5 * (p + q)
    def _kl(x, y):
        return (x * (torch.log2(x.clamp_min(eps)) - torch.log2(y.clamp_min(eps)))).sum(dim=1)
    return (0.5 * _kl(p, m) + 0.5 * _kl(q, m)).clamp_min(0.0)


def reliability(opinions: dict[str, Opinion], q: dict[str, torch.Tensor],
                fused: Opinion, specialists: tuple[str, ...] = ("ref", "proc"),
                eps: float = 1e-6) -> dict[str, torch.Tensor]:
    """§17 V / C / A — three reliability signals that must not be collapsed into one.

        V = u_f                                     insufficient fused evidence
        w_b = q_b * (1 - u_b)                       informative-and-applicable weight
        C = sum_{b<c} w_b w_c JS(p_b, p_c) / (sum w_b w_c + eps)
        A = 1 - mean_{b in specialists} w_b         specialist support absent

    They are separated because they fail differently. High `V` means nobody had evidence; high
    `C` means informed experts disagreed; high `A` means the anchor is carrying the sample alone.
    A single fused uncertainty cannot tell those apart, and DiCoME's `u_f = u_0 * u_1` is exactly
    the case where two views ignorant *for the same reason* fuse into confidence.

    `q` must contain a weight for every opinion; the anchor's is 1 by construction (§13).
    """
    names = list(opinions)
    missing = [n for n in names if n not in q]
    if missing:
        raise ValueError(f"no applicability weight for {missing}; the anchor's is 1 by definition")

    w = {}
    for n in names:
        qn = q[n]
        qn = qn.unsqueeze(1) if qn.dim() == 1 else qn
        w[n] = (qn.clamp(0.0, 1.0) * (1.0 - opinions[n].vacuity)).squeeze(1)   # (B,)

    num = torch.zeros_like(w[names[0]])
    den = torch.zeros_like(w[names[0]])
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pair = w[a] * w[b]
            num = num + pair * js_divergence(opinions[a].probability(), opinions[b].probability())
            den = den + pair
    conflict = num / (den + eps)

    present = [n for n in specialists if n in w]
    if present:
        support = torch.stack([w[n] for n in present], dim=1).mean(dim=1)
    else:
        # no specialist in the fusion at all: the anchor is unsupported by definition
        support = torch.zeros_like(conflict)
    return {
        "V": fused.vacuity.squeeze(1),
        "C": conflict,
        "A": 1.0 - support,
        "weights": w,
    }


def assert_order_invariant(opinions: list[Opinion], tol: float = 1e-4,
                           eps: float = CONFLICT_EPS) -> float:
    """§15: does combination order change the result beyond numerical tolerance?

    Returns the largest observed deviation across permutations. DS is associative and commutative
    in exact arithmetic, so a large value means the epsilon guard is engaging (near-total
    conflict) — which is information about the samples, not a bug, and must be reported rather
    than hidden behind a fixed order.
    """
    from itertools import permutations

    base, _ = ds_combine(opinions, eps)
    worst = 0.0
    for order in permutations(range(len(opinions))):
        other, _ = ds_combine([opinions[i] for i in order], eps)
        worst = max(worst,
                    float((other.belief - base.belief).abs().max()),
                    float((other.vacuity - base.vacuity).abs().max()))
    if worst > tol:
        raise AssertionError(
            f"combination order changes the fused opinion by {worst:.3e} > {tol:.1e}. Fix the "
            f"order in the fusion path and record it, or the deployed result depends on an "
            f"implementation detail.")
    return worst

"""D4 — applicability-aware fusion.

D3 adds every enabled v2 branch to the fused evidence unconditionally: each branch has one
learned scalar gate, the same for every sample. That is what makes D3 "the honest system"
and also what limits it. A1/A2 showed the manifold and process branches *rescue* some
sources and *harm* others, and the harm is not uniform — so a single scalar cannot express
"trust the manifold branch here, ignore it there".

D4 asks whether that decision can be made per sample, from label-free evidence state alone:

    total_evidence <- total_evidence + route_b * sigmoid(gate_b) * evidence_b

where `route_b in {0, 1}` comes from a small head predicting q_b = P(branch b beats the
visual baseline on THIS sample), thresholded at `tau`.

Ported, not reinvented
----------------------
The feature set, the target, and the routing rule are taken from the offline pilot
`analysis/discern_v2/a2_gate.py`, which is what produced the +0.106 mean gate-recovery
verdict that unblocked this arm. A gate trained on one feature definition and deployed on
another is a silent distribution shift, so the two must agree; `dirichlet.conflict()`
already exists for exactly this reason. Deviations from the pilot are marked DEVIATION
below, with a reason.

The A2b protocol, enforced here
-------------------------------
`analysis/discern_v2/A2_gate/A2b_PROTOCOL.md` is the binding spec:

* **Inputs are strictly label-free and identical at train and inference.** Permitted:
  per-branch evidence, alpha, p, vacuity, cross-branch conflict and disagreement. Forbidden:
  the label, `dloss_b`, generator ID, dataset ID, forgery-family ID. `assert_label_free()`
  is ported here and runs on the feature names, so a future edit that adds a leaky feature
  fails loudly instead of quietly inflating the result.
* **The label is permitted as a training TARGET, never as an input.** `dloss_b` is computed
  from the label to supervise the gate; it never enters the feature vector. These are
  different things and the protocol turns on the distinction.
* **Trained only on permitted data.** Our training set is FF++, so this holds by
  construction — but `freeze_after_epoch` makes it explicit and testable, and inference
  never updates the gate under any setting.

Why the routing mask is hard, and detached
------------------------------------------
`route_b` is a hard {0,1} mask taken under `no_grad`. Two reasons, both deliberate:

* the pilot fits the gate and *then* routes with it; a soft differentiable mask would be a
  different estimator, and the reported recovery number would no longer describe what runs;
* with a soft mask, the branches receive gradient through the gate and can learn to inflate
  their own predicted utility. The gate would then measure "which branch argues loudest",
  not "which branch is actually right". Detaching removes that channel entirely.

The gate head is trained by its own BCE term instead, which is what `gate_loss` returns.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dirichlet import DirichletState, conflict, to_dirichlet

logger = logging.getLogger(__name__)

# Ported verbatim from analysis/discern_v2/a2_gate.py. Kept as a literal copy rather than
# imported: `analysis/` is offline pandas/sklearn code and importing it into the training
# path would drag those dependencies into every run.
FORBIDDEN_FEATURE_TOKENS = ("label", "dloss", "family", "method", "generator", "source",
                            "dataset", "split")


def assert_label_free(names: list[str]) -> None:
    """Fail loudly if any feature name looks like it leaks the label or the source.

    A naming-level check cannot prove a feature is label-free, and does not try to. It
    catches the realistic failure — someone adds `dloss_manifold` to the feature dict
    because it is sitting right there and is very predictive — which is precisely how an
    applicability result gets silently inflated.
    """
    bad = [n for n in names if any(t in n.lower() for t in FORBIDDEN_FEATURE_TOKENS)]
    if bad:
        raise ValueError(
            f"applicability gate features must be label-free; forbidden name(s): {bad}. "
            f"See analysis/discern_v2/A2_gate/A2b_PROTOCOL.md")


# The feature block built per specialist branch, in a fixed order. Named explicitly so
# assert_label_free() has something to check and so the width is not a magic number.
FEATURE_NAMES = ["p_vis", "u_vis", "margin_vis",
                 "p_spec", "u_spec", "margin_spec",
                 "conflict", "disagree", "u_ratio"]
assert_label_free(FEATURE_NAMES)
FEATURE_DIM = len(FEATURE_NAMES)


def gate_features(baseline: DirichletState, specialist: DirichletState,
                  fake_index: int = 1) -> torch.Tensor:
    """Label-free evidence state for one specialist, relative to the visual baseline.

    Mirrors `a2_gate.gate_features`. The gate's job is to notice "these two branches
    disagree and the visual one is unusually uncertain", which is not visible from either
    branch's score alone — hence conflict, disagreement and the vacuity ratio are explicit
    features rather than something an MLP is expected to rediscover from raw scores.

    DEVIATION from the pilot: the pilot also consumed `ps/pa/us/ua` sub-scores when the
    exported frame carried them. Those are pilot-export artefacts with no counterpart in the
    online forward pass, so they are omitted rather than faked.

    Returns (B, FEATURE_DIM).
    """
    p_vis = baseline.fake_prob(fake_index)
    p_spec = specialist.fake_prob(fake_index)
    u_vis = baseline.vacuity
    u_spec = specialist.vacuity
    cols = [
        p_vis,
        u_vis,
        (p_vis - 0.5).abs(),                       # margin: confidence, direction-free
        p_spec,
        u_spec,
        (p_spec - 0.5).abs(),
        conflict(baseline, specialist, fake_index),
        ((p_vis > 0.5) != (p_spec > 0.5)).float(),  # hard disagreement
        u_spec / (u_vis + 1e-8),
    ]
    return torch.stack(cols, dim=1)


def _nll(state: DirichletState, labels: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Per-sample negative log-likelihood of the true class under a branch's Dirichlet mean.

    This is the `loss_b` the pilot differenced to build its target. Using p (the Dirichlet
    mean) rather than a separate head keeps online and offline definitions aligned.

    Forced to float32: under AMP `p` arrives as fp16, where a confident branch's p_true is
    small enough that its log loses most of its precision. The target is a *comparison* of
    two such logs, so fp16 noise flips the label on exactly the borderline samples the gate
    most needs to learn from.
    """
    p_true = state.p.float().gather(1, labels.view(-1, 1).long()).squeeze(1)
    return -torch.log(p_true.clamp_min(eps))


class ApplicabilityGate(nn.Module):
    """Per-sample routing head over the v2 specialist branches.

    One head per specialist (`manifold`, `process`). Heads are independent: the A2 finding
    is per branch, and a shared head would let a strong branch mask a weak one.
    """

    def __init__(self, specialists: list[str], hidden: int = 32, tau: float = 0.95,
                 loss_weight: float = 0.1, freeze_after_epoch: int | None = None,
                 warmup_epochs: int = 5, fake_index: int = 1):
        super().__init__()
        if not specialists:
            raise ValueError("ApplicabilityGate needs at least one specialist branch")
        if not 0.0 < tau < 1.0:
            raise ValueError(f"tau must be in (0, 1), got {tau}")
        self.specialists = list(specialists)
        self.tau = float(tau)
        self.loss_weight = float(loss_weight)
        self.freeze_after_epoch = freeze_after_epoch
        self.warmup_epochs = int(warmup_epochs)
        self.fake_index = int(fake_index)
        self._frozen = False
        self._epoch = 0
        self.heads = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(FEATURE_DIM, hidden), nn.ReLU(),
                                nn.Linear(hidden, 1))
            for name in self.specialists})
        logger.info(f"  Applicability   : specialists={self.specialists} tau={self.tau} "
                    f"hidden={hidden} loss_weight={self.loss_weight} "
                    f"warmup_epochs={self.warmup_epochs} "
                    f"freeze_after_epoch={self.freeze_after_epoch}")

    @property
    def warming_up(self) -> bool:
        """During warmup D4 routes everything, i.e. it *is* D3.

        Without this the arm deadlocks, and the failure is silent. A freshly initialised
        head outputs q ~= 0.5, which never clears tau = 0.95, so every specialist is routed
        out; routed out, they receive no gradient from the classification loss and stay at
        their random initialisation; still random, they never beat the visual baseline, so
        the gate's own target stays 0 and it learns to route nothing, forever. The run would
        complete, report plausible numbers, and be measuring a system with no specialists at
        all — strictly worse than D3, for reasons invisible in the loss curve.

        Routing everything for the first `warmup_epochs` breaks the cycle: the branches
        train exactly as they do in D3, so by the time the gate engages its target is
        meaningful. It also makes D4 - D3 easier to read, since D4 starts life *as* D3 and
        the gate is the only thing that subsequently differs.

        Setting warmup_epochs: 0 is legitimate when initialising from a trained D3(P1d)
        checkpoint — there the branches are already trained and the deadlock cannot form.
        """
        return self._epoch < self.warmup_epochs

    def set_epoch(self, epoch: int) -> None:
        """Track the epoch for warmup, and apply the A2b freeze. Idempotent."""
        self._epoch = int(epoch)
        if self.freeze_after_epoch is None or self._frozen:
            return
        if epoch >= self.freeze_after_epoch:
            for p in self.parameters():
                p.requires_grad_(False)
            self._frozen = True
            logger.info(f"  Applicability   : gate FROZEN at epoch {epoch}")

    # kept as the old name so callers written against it keep working
    maybe_freeze = set_epoch

    def logit(self, name: str, baseline: DirichletState,
              specialist: DirichletState) -> torch.Tensor:
        """Raw head output for branch b, before the sigmoid.

        The logit rather than q is the quantity that leaves this method, because the loss
        must be `binary_cross_entropy_with_logits`: plain `binary_cross_entropy` raises
        under AMP ("unsafe to autocast"), since it cannot be made numerically safe in fp16
        once the sigmoid has already been taken. Fusing the two is both autocast-safe and
        better conditioned, so q is derived for routing only.
        """
        feats = gate_features(baseline, specialist, self.fake_index)
        return self.heads[name](feats).squeeze(1)

    def q(self, name: str, baseline: DirichletState,
          specialist: DirichletState) -> torch.Tensor:
        """Predicted utility q_b in [0, 1] — P(branch b beats the visual baseline here)."""
        return torch.sigmoid(self.logit(name, baseline, specialist))

    def route(self, q: torch.Tensor) -> torch.Tensor:
        """Hard routing mask at tau, with no gradient path (see module docstring).

        Conservative by construction: when no specialist is confidently better, the visual
        baseline is kept untouched. tau = 0.95 is the A2c setting — it cuts worst-source
        harm 72% and halves routing to 47%, at ~20% of the mean gain. An applicability gate
        that harms what the baseline already gets right has failed at its only job.

        During warmup everything is routed, so the arm is exactly D3 (see `warming_up`).
        """
        if self.warming_up:
            return torch.ones_like(q.detach())
        return (q.detach() >= self.tau).float()

    def gate_loss(self, name: str, logit: torch.Tensor, baseline: DirichletState,
                  specialist: DirichletState, labels: torch.Tensor) -> torch.Tensor:
        """BCE-with-logits against the pilot's target: does this specialist beat the
        visual baseline on this sample?

        Takes the LOGIT, not q — see `logit()` for why the fused form is required under AMP.

        The label enters HERE and only here, as a target. `dloss` is never returned to the
        feature path — see the A2b note in the module docstring.
        """
        with torch.no_grad():
            dloss = _nll(baseline, labels) - _nll(specialist, labels)
            target = (dloss > 0).float()
        return F.binary_cross_entropy_with_logits(logit.float(), target)

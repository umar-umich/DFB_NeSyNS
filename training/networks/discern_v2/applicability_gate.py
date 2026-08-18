"""§13 — fusion-utility-supervised applicability gates `q_ref`, `q_proc`.

The gate must learn the quantity it actually controls: *does admitting this specialist into the
conflict-aware fusion improve the decision for this sample?* §13 is explicit that the tempting
alternative — supervising on standalone specialist quality, `CE(p_sem, y) - CE(p_b, y)` — asks a
different and weaker question and can diverge from fusion utility. So the target comes from the
same DS mechanism that deploys:

    p_sem(+)b = DS(omega_sem, omega_b)            undiscounted, pairwise, same combination + eps
    Delta_b   = CE(p_sem, y) - CE(p_sem(+)b, y)
    t_b       = 1[Delta_b > delta],  delta = 0     config, never tuned on OOD

`q_b = sigmoid(H_app(features))` then predicts the probability that letting specialist `b` into
fusion improves the outcome, trained with BCE on VAL_meta folds.

The named approximation (write this into the paper)
---------------------------------------------------
The target is pairwise and full-admission; deployment is three-way and soft-discounted
(`omega_sem (+) omega'_ref (+) omega'_proc`). This is a much better proxy than standalone quality,
but it cannot see interactions where reference and process each help alone yet jointly overshoot
or conflict. Three-way Shapley-style marginal utility is deferred and pulled in only if §20's
plain-DS-vs-DS+applicability diagnostic shows the applicability layer failing to beat plain DS.
The V1 method is **fusion-utility-supervised applicability**, not generic utility supervision.

The label is a TRAINING TARGET, never an input
----------------------------------------------
`t_b` is computed from the label; the gate's *features* are strictly label-free and identical at
train and inference time. `assert_label_free` (ported from the Phase-1 A2b protocol, which is
where the discipline was validated) runs on the feature names, so an edit that adds a leaky
feature fails loudly instead of quietly inflating the result. Generator, manipulation, dataset and
family identity are forbidden inputs even though they are sitting right there and are predictive.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .applicability import FORBIDDEN_FEATURE_TOKENS, assert_label_free
from .ds_fusion import Opinion, ds_combine

logger = logging.getLogger(__name__)

# The label-free feature set. Compact on purpose: §13 recommends the branch's own evidential
# state plus cross-branch disagreement, and every extra input is another chance to smuggle in
# something that correlates with the source rather than with applicability.
FEATURE_NAMES = (
    "p_sem_fake",       # anchor's projected p(fake)
    "u_sem",            # anchor vacuity
    "p_b_fake",         # specialist's projected p(fake)
    "u_b",              # specialist vacuity
    "abs_disagree",     # |p_sem_fake - p_b_fake|
    "js_disagree",      # JS divergence between the two projected distributions
    "b_margin",         # |p_b_fake - 0.5| — how decided the specialist is
    "sem_margin",       # |p_sem_fake - 0.5|
)
assert_label_free(list(FEATURE_NAMES))

DEFAULT_DELTA = 0.0


def _cross_entropy(p: torch.Tensor, labels: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Per-sample CE of a probability vector (not logits) against integer labels."""
    return F.nll_loss(torch.log(p.clamp_min(eps)), labels, reduction="none")


@torch.no_grad()
def fusion_utility_target(sem: Opinion, spec: Opinion, labels: torch.Tensor,
                          delta: float = DEFAULT_DELTA) -> dict:
    """§13's target: does admitting this specialist into DS fusion help THIS sample?

    Undiscounted and pairwise, using the same centralized DS the deployment path uses — a target
    computed with a different combination rule would train the gate against a fusion that never
    runs.
    """
    fused, _ = ds_combine([sem, spec])
    ce_sem = _cross_entropy(sem.probability(), labels)
    ce_fused = _cross_entropy(fused.probability(), labels)
    improvement = ce_sem - ce_fused
    return {
        "target": (improvement > delta).float(),
        "delta_fuse": improvement,
        "ce_sem": ce_sem,
        "ce_fused": ce_fused,
    }


def gate_features(sem: Opinion, spec: Opinion) -> torch.Tensor:
    """(B, len(FEATURE_NAMES)) label-free features, identical at train and inference."""
    from .ds_fusion import js_divergence

    p_sem = sem.probability()
    p_b = spec.probability()
    sem_fake, b_fake = p_sem[:, 1], p_b[:, 1]
    return torch.stack([
        sem_fake,
        sem.vacuity.squeeze(1),
        b_fake,
        spec.vacuity.squeeze(1),
        (sem_fake - b_fake).abs(),
        js_divergence(p_sem, p_b),
        (b_fake - 0.5).abs(),
        (sem_fake - 0.5).abs(),
    ], dim=1)


class ApplicabilityGate(nn.Module):
    """`q_b = sigmoid(H_app(h_b))`. One gate per specialist; the anchor has no gate (q_sem = 1)."""

    def __init__(self, n_features: int = len(FEATURE_NAMES), hidden: int = 32,
                 feature_names: tuple[str, ...] = FEATURE_NAMES):
        super().__init__()
        assert_label_free(list(feature_names))
        self.feature_names = tuple(feature_names)
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Features -> q in [0, 1]. Takes features only: the gate is structurally incapable of
        reading a label at inference because there is no argument through which one could arrive.
        """
        if features.shape[1] != len(self.feature_names):
            raise ValueError(
                f"gate expects {len(self.feature_names)} features "
                f"{self.feature_names}, got {features.shape[1]}")
        return torch.sigmoid(self.net(features)).squeeze(1)

    def loss(self, features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """BCE against the fusion-utility target (§13)."""
        return F.binary_cross_entropy(self(features).clamp(1e-6, 1 - 1e-6), target)


def train_gate(features: torch.Tensor, target: torch.Tensor, *, hidden: int = 32,
               epochs: int = 200, lr: float = 1e-2, seed: int = 42,
               weight_decay: float = 1e-4) -> tuple[ApplicabilityGate, dict]:
    """Fit one gate. Small, full-batch, deterministic — it reads eight scalars.

    Class weighting is deliberately absent: `q_b` is used as a probability of usefulness, and
    rebalancing would distort exactly the quantity the discounting step consumes.
    """
    torch.manual_seed(seed)
    gate = ApplicabilityGate(n_features=features.shape[1], hidden=hidden)
    opt = torch.optim.Adam(gate.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = gate.loss(features, target)
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
    return gate, {"final_loss": history[-1], "positive_rate": float(target.mean()),
                  "n": int(len(target))}


def cross_fit(features: torch.Tensor, target: torch.Tensor, folds: torch.Tensor,
              **kwargs) -> dict:
    """§12's 5-fold cross-fitting: out-of-fold `q`, then one gate refit on all of VAL_meta.

    Out-of-fold predictions are what the §18 risk model trains on. Fitting the risk model on
    in-sample `q` would let it calibrate against a gate that had already seen those samples, and
    the resulting defer policy would look better offline than it can ever behave.
    """
    q_oof = torch.zeros_like(target)
    per_fold = {}
    for k in sorted(set(int(f) for f in folds.tolist())):
        held = folds == k
        if held.all() or not held.any():
            raise ValueError(f"fold {k} is empty or covers everything")
        gate, info = train_gate(features[~held], target[~held], **kwargs)
        with torch.no_grad():
            q_oof[held] = gate(features[held])
        per_fold[f"fold_{k}"] = {**info, "n_held_out": int(held.sum())}

    deployment_gate, info = train_gate(features, target, **kwargs)
    return {
        "q_out_of_fold": q_oof,
        "gate": deployment_gate,
        "per_fold": per_fold,
        "deployment": info,
        "metrics": gate_metrics(q_oof, target),
    }


def gate_metrics(q: torch.Tensor, target: torch.Tensor) -> dict:
    """§13 requires the gate to be logged against its own target, not against the task label."""
    q_np = q.detach().cpu().numpy()
    t_np = target.detach().cpu().numpy()
    out = {"accuracy": float(((q_np >= 0.5) == (t_np >= 0.5)).mean()),
           "positive_rate": float(t_np.mean()),
           "mean_q": float(q_np.mean())}
    try:
        from sklearn.metrics import roc_auc_score
        out["auroc"] = (float(roc_auc_score(t_np, q_np)) if len(set(t_np.tolist())) > 1
                        else float("nan"))
    except ImportError:                                    # pragma: no cover
        out["auroc"] = float("nan")
    # A gate that always says "admit" scores well on accuracy whenever the target is mostly 1;
    # reported so that case is visible rather than flattering.
    out["majority_baseline_accuracy"] = float(max(t_np.mean(), 1 - t_np.mean()))
    return out


__all__ = ["FEATURE_NAMES", "FORBIDDEN_FEATURE_TOKENS", "ApplicabilityGate", "cross_fit",
           "fusion_utility_target", "gate_features", "gate_metrics", "train_gate"]

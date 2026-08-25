"""§18 — Real / Fake / Defer: a deliberately small risk model over V, C, A and the fused margin.

    features = [V, C, A, |p_fused(fake) - 0.5|]
    target   = 1 if the FUSED prediction is wrong
    model    = logistic regression (§18: "No large MLP in V1")

Fit on VAL_meta using the **out-of-fold** `q_b` from §12's cross-fitting, so the risk model never
sees a `q` produced by a gate that had already been trained on that sample. Without that, the
defer policy calibrates against an optimistic gate and looks better offline than it can behave.

Why a linear model is the right size here
-----------------------------------------
The claim under test is that V, C and A carry *separable* information about when the system is
wrong. A capacious model would make that claim unfalsifiable: it could recover an error signal
from almost any four correlated inputs, and a good risk-coverage curve would no longer be
evidence about the reliability decomposition. Four coefficients are also readable — the paper can
state which reliability signal is doing the work, and the sign of each.

Threshold discipline (§18, §11)
-------------------------------
Every threshold — the Real/Fake decision boundary and the defer cutoff — is derived on FF++
validation only and frozen for every OOD source. `freeze_thresholds()` returns an immutable
`DeferPolicy`; nothing downstream can recompute a threshold on the data it is being evaluated on,
because the policy carries the numbers with it and records where they came from.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

FEATURE_NAMES = ("V", "C", "A", "fused_margin")
REAL, FAKE, DEFER = 0, 1, 2
DECISION_NAMES = {REAL: "Real", FAKE: "Fake", DEFER: "Defer"}


def risk_features(V: torch.Tensor, C: torch.Tensor, A: torch.Tensor,
                  prob_fake: torch.Tensor) -> torch.Tensor:
    """(B, 4) — the reliability triple plus how decided the fused opinion is.

    `fused_margin = |p - 0.5|` rather than `p` itself: the risk model must predict *error*, and
    error is symmetric in the class, so handing it a signed probability would invite it to learn
    a class prior instead of a reliability signal.
    """
    return torch.stack([V, C, A, (prob_fake - 0.5).abs()], dim=1)


class RiskModel(nn.Module):
    """Logistic regression on the four reliability features. P(the fused prediction is wrong)."""

    def __init__(self, n_features: int = len(FEATURE_NAMES),
                 feature_names: tuple[str, ...] = FEATURE_NAMES):
        super().__init__()
        self.feature_names = tuple(feature_names)
        self.linear = nn.Linear(n_features, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[1] != len(self.feature_names):
            raise ValueError(f"expected {len(self.feature_names)} features "
                             f"{self.feature_names}, got {features.shape[1]}")
        return torch.sigmoid(self.linear(features)).squeeze(1)

    def coefficients(self) -> dict:
        """The readable part: which reliability signal predicts error, and in which direction."""
        w = self.linear.weight.detach().squeeze(0)
        return {**{n: float(v) for n, v in zip(self.feature_names, w)},
                "bias": float(self.linear.bias.detach())}


def fit_risk_model(features: torch.Tensor, wrong: torch.Tensor, *, epochs: int = 500,
                   lr: float = 0.05, weight_decay: float = 1e-3, seed: int = 42
                   ) -> tuple[RiskModel, dict]:
    """Fit on VAL_meta. `wrong` is 1 where the fused prediction was incorrect."""
    if features.shape[0] != wrong.shape[0]:
        raise ValueError(f"features {features.shape[0]} vs targets {wrong.shape[0]}")
    if len(torch.unique(wrong)) < 2:
        raise ValueError(
            "the fused model is either always right or always wrong on this split, so there is "
            "no error signal to calibrate against; check the split before trusting any defer "
            "policy fit here")
    torch.manual_seed(seed)
    model = RiskModel(features.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = nn.BCELoss()
    history = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(model(features).clamp(1e-6, 1 - 1e-6), wrong.float())
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
    return model, {"final_loss": history[-1], "error_rate": float(wrong.float().mean()),
                   "n": int(len(wrong)), "coefficients": model.coefficients()}


# ---------------------------------------------------------------------------
# selective prediction
# ---------------------------------------------------------------------------


def error_detection_auroc(risk: torch.Tensor, wrong: torch.Tensor) -> float:
    """Does the risk score rank errors above correct predictions?"""
    from sklearn.metrics import roc_auc_score

    y = wrong.detach().cpu().numpy()
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, risk.detach().cpu().numpy()))


def risk_coverage_curve(risk: torch.Tensor, wrong: torch.Tensor,
                        n_points: int = 21) -> list[dict]:
    """Selective risk at a sweep of coverage levels.

    At coverage `c` the system answers on the `c` fraction of samples it considers least risky.
    A working reliability model gives selective risk that falls monotonically as coverage drops;
    a flat curve means the risk score is not ordering errors at all, whatever its AUROC.
    """
    order = torch.argsort(risk)
    errors = wrong.float()[order]
    n = len(errors)
    out = []
    for c in np.linspace(1.0, 0.05, n_points):
        k = max(1, int(round(c * n)))
        out.append({"coverage": k / n,
                    "selective_risk": float(errors[:k].mean()),
                    "n_answered": k})
    return out


def coverage_at_budget(risk: torch.Tensor, wrong: torch.Tensor,
                       abstention_budget: float = 0.10) -> dict:
    """§18's fixed-budget report: defer the riskiest `budget` fraction, then measure."""
    n = len(risk)
    n_defer = int(round(abstention_budget * n))
    order = torch.argsort(risk)
    answered = order[:n - n_defer] if n_defer else order
    errors = wrong.float()
    return {
        "abstention_budget": abstention_budget,
        "coverage": len(answered) / n,
        "selective_risk": float(errors[answered].mean()) if len(answered) else float("nan"),
        "full_coverage_risk": float(errors.mean()),
        "risk_reduction": float(errors.mean()) - (float(errors[answered].mean())
                                                  if len(answered) else float("nan")),
        "n_deferred": int(n_defer),
    }


# ---------------------------------------------------------------------------
# the frozen policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeferPolicy:
    """Thresholds derived on FF++ validation and frozen for every OOD source (§18).

    Frozen dataclass on purpose: a policy that can be recomputed downstream is a policy that will
    eventually be recomputed on the evaluation data, which is the leak §11 and §18 both forbid.
    `provenance` records where the numbers came from so a results table can state it.
    """

    decision_threshold: float      # p_fused(fake) above this -> Fake
    risk_threshold: float          # risk above this -> Defer
    provenance: str

    def decide(self, prob_fake: torch.Tensor, risk: torch.Tensor) -> torch.Tensor:
        """-> tensor of REAL / FAKE / DEFER."""
        decision = torch.where(prob_fake >= self.decision_threshold,
                               torch.full_like(prob_fake, FAKE),
                               torch.full_like(prob_fake, REAL))
        return torch.where(risk > self.risk_threshold,
                           torch.full_like(prob_fake, DEFER), decision).long()

    def as_dict(self) -> dict:
        return {"decision_threshold": self.decision_threshold,
                "risk_threshold": self.risk_threshold, "provenance": self.provenance}


def eer_threshold(labels: torch.Tensor, prob_fake: torch.Tensor) -> float:
    """Equal-error point on the protocol source. Never computed on an OOD source."""
    from sklearn.metrics import roc_curve

    y = labels.detach().cpu().numpy()
    p = prob_fake.detach().cpu().numpy()
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[int(np.nanargmin(np.abs(fpr - (1 - tpr))))])


def freeze_thresholds(labels: torch.Tensor, prob_fake: torch.Tensor, risk: torch.Tensor,
                      abstention_budget: float = 0.10,
                      source: str = "FF++ VAL_meta (out-of-fold q)") -> DeferPolicy:
    """Derive both thresholds on the protocol source, once, and freeze them.

    The defer cutoff is the `budget` quantile of the risk score rather than a fixed number: risk
    scores are not comparable across models, but "the riskiest 10%" is, and §18 asks for a fixed
    abstention budget rather than a fixed risk value.
    """
    decision = eer_threshold(labels, prob_fake)
    cutoff = float(torch.quantile(risk, 1.0 - abstention_budget))
    return DeferPolicy(
        decision_threshold=decision,
        risk_threshold=cutoff,
        provenance=(f"decision = EER on {source}; defer = {abstention_budget:.0%} abstention "
                    f"budget on {source}; both frozen for all OOD sources"))


def evaluate_policy(policy: DeferPolicy, labels: torch.Tensor, prob_fake: torch.Tensor,
                    risk: torch.Tensor) -> dict:
    """Apply a FROZEN policy to a new source and report what it did there."""
    decisions = policy.decide(prob_fake, risk)
    deferred = decisions == DEFER
    answered = ~deferred
    correct = (decisions == labels) & answered
    return {
        "n": int(len(labels)),
        "coverage": float(answered.float().mean()),
        "deferred": int(deferred.sum()),
        "selective_accuracy": (float(correct.sum()) / float(answered.sum())
                               if int(answered.sum()) else float("nan")),
        "full_coverage_accuracy": float(
            ((prob_fake >= policy.decision_threshold).long() == labels).float().mean()),
        "decisions": {DECISION_NAMES[k]: int((decisions == k).sum()) for k in DECISION_NAMES},
    }


def report(risk: torch.Tensor, wrong: torch.Tensor, labels: torch.Tensor,
           prob_fake: torch.Tensor, abstention_budget: float = 0.10,
           coverage_points: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.5)) -> dict:
    """Everything §18 asks to be reported, in one place."""
    curve = risk_coverage_curve(risk, wrong)
    by_coverage = {}
    for c in coverage_points:
        nearest = min(curve, key=lambda row: abs(row["coverage"] - c))
        by_coverage[f"coverage_{c:.2f}"] = nearest
    return {
        "error_detection_auroc": error_detection_auroc(risk, wrong),
        "risk_coverage_curve": curve,
        "selective_risk_at": by_coverage,
        "fixed_budget": coverage_at_budget(risk, wrong, abstention_budget),
        "base_error_rate": float(wrong.float().mean()),
        "n": int(len(risk)),
    }

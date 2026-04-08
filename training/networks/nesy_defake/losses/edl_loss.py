"""
losses/edl_loss.py
==================
Evidential Deep Learning loss for Dirichlet-based uncertainty estimation.

Adapted from FFDBackbone (PAMI) for the NeSyDeFake ablation pipeline.

Given logits (B, K):
  evidence  = softplus(logits)             # non-negative evidence
  alpha     = evidence + 1                 # Dirichlet concentration params
  S         = alpha.sum(dim=1)             # Dirichlet strength
  pred      = alpha / S                    # expected probability
  uncertainty = K / S                      # epistemic uncertainty

Loss = EDL_log_likelihood + annealed_KL_divergence [+ AVU calibration]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialLoss(nn.Module):
    """
    Evidential Deep Learning loss with log-likelihood formulation,
    annealed KL regularisation, and optional AVU calibration.
    """

    def __init__(
        self,
        num_classes: int = 2,
        annealing_epochs: int = 10,
        kl_weight: float = 0.1,
        avu_weight: float = 0.0,
        class_weights: list = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.annealing_epochs = annealing_epochs
        self.kl_weight = kl_weight
        self.avu_weight = avu_weight
        self.eps = 1e-7
        if class_weights is not None:
            self.register_buffer(
                '_class_weights', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self._class_weights = None

    # ------------------------------------------------------------------
    #  Core EDL quantities
    # ------------------------------------------------------------------

    @staticmethod
    def logits_to_evidence(logits: torch.Tensor) -> torch.Tensor:
        """Non-negative evidence via softplus (smooth, gradient-friendly)."""
        return F.softplus(logits)

    def evidence_to_dirichlet(self, evidence: torch.Tensor):
        """Returns (alpha, S, uncertainty)."""
        alpha = evidence + 1.0
        S = alpha.sum(dim=1, keepdim=True)
        uncertainty = self.num_classes / S
        return alpha, S, uncertainty

    # ------------------------------------------------------------------
    #  Loss components
    # ------------------------------------------------------------------

    def _log_likelihood(self, alpha, S, y_onehot):
        """Type-II maximum likelihood: E_{Dir}[log p(y | theta)]."""
        ll = torch.sum(
            y_onehot * (torch.digamma(alpha) - torch.digamma(S)),
            dim=1, keepdim=True,
        )
        return -ll  # minimise negative log-likelihood

    def _kl_divergence(self, alpha, y_onehot):
        """KL(Dir(alpha_tilde) || Dir(1,...,1)) removing correct-class evidence."""
        alpha_tilde = (1.0 - y_onehot) * (alpha - 1.0) + 1.0
        S_tilde = alpha_tilde.sum(dim=1, keepdim=True)

        # Analytic KL between two Dirichlet distributions
        ones = torch.ones_like(alpha_tilde)
        S_ones = ones.sum(dim=1, keepdim=True)

        kl = (
            torch.lgamma(S_tilde) - torch.lgamma(S_ones)
            - torch.sum(torch.lgamma(alpha_tilde), dim=1, keepdim=True)
            + torch.sum(torch.lgamma(ones), dim=1, keepdim=True)
            + torch.sum(
                (alpha_tilde - ones)
                * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)),
                dim=1, keepdim=True,
            )
        )
        return kl

    def _avu_loss(self, alpha, S, y_onehot, target, uncertainty):
        """
        Accuracy-vs-Uncertainty calibration (from PAMI FFDBackbone):
        penalise confident-wrong, reward uncertain-wrong.
        """
        pred_scores, pred_cls = torch.max(alpha / S, dim=1, keepdim=True)
        acc_match = (pred_cls == target.unsqueeze(1)).float()

        acc_uncertain = -pred_scores * torch.log(
            1.0 - uncertainty + self.eps)
        inacc_certain = -(1.0 - pred_scores) * torch.log(
            uncertainty + self.eps)

        avu = acc_match * acc_uncertain + (1.0 - acc_match) * inacc_certain
        return avu

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        evidence: torch.Tensor,
        target: torch.Tensor,
        epoch: int = 0,
    ) -> dict:
        """
        Args:
            evidence: (B, K) non-negative evidence (already through softplus).
                      Caller is responsible for computing evidence from logits
                      or from fused multi-branch evidence.
            target:   (B,) integer class labels.
            epoch:    current epoch for KL annealing.

        Returns:
            dict with 'loss', 'alpha', 'uncertainty', 'pred',
                 'loss_nll', 'loss_kl', 'loss_avu'.
        """
        # Force FP32 for numerical stability (digamma, lgamma)
        evidence = evidence.float()

        alpha, S, uncertainty = self.evidence_to_dirichlet(evidence)
        y_onehot = F.one_hot(target, self.num_classes).float().to(evidence.device)

        # 1. Log-likelihood (per-sample class-weighted if configured)
        nll_per_sample = self._log_likelihood(alpha, S, y_onehot)  # (B, 1)
        if self._class_weights is not None:
            w = self._class_weights.to(evidence.device)[target]  # (B,)
            loss_nll = (nll_per_sample.squeeze(1) * w).mean()
        else:
            loss_nll = nll_per_sample.mean()

        # 2. Annealed KL divergence
        anneal = min(1.0, epoch / max(self.annealing_epochs, 1))
        loss_kl = self._kl_divergence(alpha, y_onehot).mean()

        loss = loss_nll + self.kl_weight * anneal * loss_kl

        # 3. Optional AVU calibration
        loss_avu = torch.tensor(0.0, device=evidence.device)
        if self.avu_weight > 0:
            loss_avu = self._avu_loss(
                alpha, S, y_onehot, target, uncertainty).mean()
            loss = loss + self.avu_weight * loss_avu

        # Prediction & diagnostics
        pred = alpha / S  # Dirichlet mean

        # Evidence statistics (for logging)
        _, preds = torch.max(pred, dim=1)
        correct = (preds == target).float()
        total_ev = evidence.sum(dim=1)
        ev_succ = (total_ev * correct).sum() / (correct.sum() + self.eps)
        ev_fail = (total_ev * (1.0 - correct)).sum() / (
            (1.0 - correct).sum() + self.eps)

        return {
            'loss': loss,
            'alpha': alpha,
            'uncertainty': uncertainty.squeeze(1),
            'pred': pred,
            'loss_nll': loss_nll.detach(),
            'loss_kl': loss_kl.detach(),
            'loss_avu': loss_avu.detach(),
            'evidence_succ': ev_succ.detach(),
            'evidence_fail': ev_fail.detach(),
        }

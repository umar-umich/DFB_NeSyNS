"""
losses/nesy_edl_loss.py
========================
Neuro-Symbolic Evidential Deep Learning (NeSy-EDL).

Novel multi-branch evidential framework that goes beyond the standard
single-source EDL (Sensoy et al., 2018) adapted in FFDBackbone (PAMI).

Three novel components on top of standard EDL:

  1. **Confidence-Modulated Evidence Fusion (CMEF)**:
     Replace static scalar gates (e_total = e_s + g*e_c) with per-sample
     dynamic weighting based on each branch's Dirichlet strength.
     Branches that are uncertain about a sample contribute less evidence;
     branches that are confident contribute more. This naturally allocates
     trust to symbolic reasoning when neural perception is uncertain
     --- precisely the OOD regime where generalization matters.

  2. **Per-Branch Auxiliary Supervision (PBAS)**:
     Each branch receives its own lightweight EDL loss, ensuring it learns
     independently meaningful evidence before fusion. Without this, a
     dominant branch (spatial/CLIP) can suppress gradient flow to weaker
     branches, causing them to learn noise.

  3. **Inter-Branch Disagreement Calibration (IBDC)**:
     A novel loss term that ties epistemic uncertainty to branch agreement.
     When neural and symbolic branches disagree on a sample, the fused
     prediction SHOULD be uncertain. When they agree, it can be confident.
     This produces calibrated uncertainty that reflects reasoning conflict,
     not just evidence magnitude.

Together, these form a principled neuro-symbolic uncertainty framework
where symbolic knowledge directly shapes the epistemic landscape.
"""

import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class NeSyEvidentialLoss(nn.Module):
    """
    Neuro-Symbolic Evidential Deep Learning loss.

    Extends standard EDL with multi-branch awareness:
      - Per-branch auxiliary supervision
      - Inter-branch disagreement calibration
      - Class-weighted NLL
      - Annealed KL + AVU (inherited from standard EDL)
    """

    def __init__(
        self,
        num_classes: int = 2,
        annealing_epochs: int = 10,
        kl_weight: float = 0.15,
        avu_weight: float = 0.1,
        aux_weight: float = 0.1,
        disagreement_weight: float = 0.05,
        class_weights: list = None,
    ):
        super().__init__()
        self.K = num_classes
        self.annealing_epochs = annealing_epochs
        self.kl_weight = kl_weight
        self.avu_weight = avu_weight
        self.aux_weight = aux_weight
        self.disagreement_weight = disagreement_weight
        self.eps = 1e-7

        if class_weights is not None:
            self.register_buffer(
                '_class_weights',
                torch.tensor(class_weights, dtype=torch.float32))
        else:
            self._class_weights = None

        logger.info(
            f"[NeSy-EDL] K={num_classes}, kl={kl_weight}, avu={avu_weight}, "
            f"aux={aux_weight}, disagree={disagreement_weight}, "
            f"class_w={class_weights}")

    # ------------------------------------------------------------------
    #  Core EDL quantities
    # ------------------------------------------------------------------

    @staticmethod
    def evidence_to_dirichlet(evidence: torch.Tensor):
        """(B, K) evidence → (alpha, S, uncertainty)."""
        alpha = evidence + 1.0
        S = alpha.sum(dim=1, keepdim=True)
        K = evidence.shape[1]
        uncertainty = K / S
        return alpha, S, uncertainty

    # ------------------------------------------------------------------
    #  Standard EDL components
    # ------------------------------------------------------------------

    def _nll(self, alpha, S, y_onehot):
        """Type-II maximum likelihood: E_{Dir}[log p(y | theta)]."""
        ll = torch.sum(
            y_onehot * (torch.digamma(alpha) - torch.digamma(S)),
            dim=1, keepdim=True)
        return -ll

    def _kl(self, alpha, y_onehot):
        """KL(Dir(alpha_tilde) || Dir(1)) removing correct-class evidence."""
        alpha_tilde = (1.0 - y_onehot) * (alpha - 1.0) + 1.0
        S_tilde = alpha_tilde.sum(dim=1, keepdim=True)
        ones = torch.ones_like(alpha_tilde)
        S_ones = ones.sum(dim=1, keepdim=True)
        kl = (
            torch.lgamma(S_tilde) - torch.lgamma(S_ones)
            - torch.sum(torch.lgamma(alpha_tilde), dim=1, keepdim=True)
            + torch.sum(torch.lgamma(ones), dim=1, keepdim=True)
            + torch.sum(
                (alpha_tilde - ones)
                * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)),
                dim=1, keepdim=True))
        return kl

    def _avu(self, alpha, S, target, uncertainty):
        """Accuracy-vs-Uncertainty calibration."""
        pred_scores, pred_cls = torch.max(alpha / S, dim=1, keepdim=True)
        acc_match = (pred_cls == target.unsqueeze(1)).float()
        acc_uncertain = -pred_scores * torch.log(
            1.0 - uncertainty + self.eps)
        inacc_certain = -(1.0 - pred_scores) * torch.log(
            uncertainty + self.eps)
        return acc_match * acc_uncertain + (1.0 - acc_match) * inacc_certain

    def _single_branch_edl(self, evidence, target, y_onehot, epoch):
        """Compute EDL loss for a single branch (used for aux losses)."""
        evidence = evidence.float()
        alpha, S, uncertainty = self.evidence_to_dirichlet(evidence)

        nll = self._nll(alpha, S, y_onehot)
        if self._class_weights is not None:
            w = self._class_weights.to(evidence.device)[target]
            loss_nll = (nll.squeeze(1) * w).mean()
        else:
            loss_nll = nll.mean()

        anneal = min(1.0, epoch / max(self.annealing_epochs, 1))
        loss_kl = self._kl(alpha, y_onehot).mean()
        return loss_nll + self.kl_weight * anneal * loss_kl

    # ------------------------------------------------------------------
    #  Novel Component 3: Inter-Branch Disagreement Calibration (IBDC)
    # ------------------------------------------------------------------

    def _disagreement_calibration(
        self,
        branch_evidences: list,
        fused_uncertainty: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encourage fused uncertainty to reflect inter-branch agreement.

        When branches disagree on the predicted class, fused uncertainty
        should be HIGH. When they agree, it should be LOW.

        Uses cosine distance between branch Dirichlet means as the
        disagreement measure, and binary cross-entropy to calibrate:

          d(x) = 1 - cos(p_neural(x), p_symbolic(x))
          L_bdc = -d * log(u) - (1-d) * log(1-u)

        This is novel: standard EDL has no mechanism for uncertainty
        to reflect reasoning CONFLICT between evidence sources.
        """
        if len(branch_evidences) < 2:
            return torch.zeros(1, device=fused_uncertainty.device).squeeze()

        # Compute Dirichlet means for each branch
        branch_probs = []
        for ev in branch_evidences:
            alpha = ev + 1.0
            S = alpha.sum(dim=1, keepdim=True)
            branch_probs.append(alpha / S)  # (B, K)

        # Average pairwise cosine disagreement across all branch pairs
        disagreement = torch.zeros(
            branch_probs[0].shape[0], device=fused_uncertainty.device)
        n_pairs = 0
        for i in range(len(branch_probs)):
            for j in range(i + 1, len(branch_probs)):
                cos_sim = F.cosine_similarity(
                    branch_probs[i], branch_probs[j], dim=1)
                disagreement = disagreement + (1.0 - cos_sim)
                n_pairs += 1
        disagreement = (disagreement / max(n_pairs, 1)).clamp(0, 1)

        # Binary cross-entropy: disagreement → high uncertainty
        u = fused_uncertainty.squeeze().clamp(self.eps, 1.0 - self.eps)
        d = disagreement.detach()  # stop gradient on disagreement target
        bdc = -(d * torch.log(u) + (1 - d) * torch.log(1 - u))
        return bdc.mean()

    # ------------------------------------------------------------------
    #  Forward: full NeSy-EDL loss
    # ------------------------------------------------------------------

    def forward(
        self,
        fused_evidence: torch.Tensor,
        target: torch.Tensor,
        epoch: int = 0,
        branch_evidences: dict = None,
    ) -> dict:
        """
        Args:
            fused_evidence:   (B, K) fused evidence from CMEF.
            target:           (B,) integer class labels.
            epoch:            current epoch for KL annealing.
            branch_evidences: dict mapping branch name → (B, K) evidence.
                              e.g. {'spatial': ..., 'concept': ..., 'causal': ...}
                              If None, falls back to standard single-source EDL.
        Returns:
            dict with 'loss', 'alpha', 'uncertainty', 'pred', plus diagnostics.
        """
        fused_evidence = fused_evidence.float()
        device = fused_evidence.device
        y_onehot = F.one_hot(target, self.K).float().to(device)

        # ── Primary EDL loss on fused evidence ────────────────────────
        alpha, S, uncertainty = self.evidence_to_dirichlet(fused_evidence)

        nll = self._nll(alpha, S, y_onehot)
        if self._class_weights is not None:
            w = self._class_weights.to(device)[target]
            loss_nll = (nll.squeeze(1) * w).mean()
        else:
            loss_nll = nll.mean()

        anneal = min(1.0, epoch / max(self.annealing_epochs, 1))
        loss_kl = self._kl(alpha, y_onehot).mean()
        loss = loss_nll + self.kl_weight * anneal * loss_kl

        # AVU calibration
        loss_avu = torch.zeros(1, device=device).squeeze()
        if self.avu_weight > 0:
            loss_avu = self._avu(alpha, S, target, uncertainty).mean()
            loss = loss + self.avu_weight * loss_avu

        # ── Novel: Per-Branch Auxiliary Supervision (PBAS) ────────────
        loss_aux = torch.zeros(1, device=device).squeeze()
        if branch_evidences is not None and self.aux_weight > 0:
            for name, ev in branch_evidences.items():
                if ev is not None:
                    loss_aux = loss_aux + self._single_branch_edl(
                        ev, target, y_onehot, epoch)
            loss = loss + self.aux_weight * loss_aux

        # ── Novel: Inter-Branch Disagreement Calibration (IBDC) ───────
        loss_bdc = torch.zeros(1, device=device).squeeze()
        if (branch_evidences is not None
                and self.disagreement_weight > 0
                and len(branch_evidences) >= 2):
            ev_list = [ev for ev in branch_evidences.values() if ev is not None]
            if len(ev_list) >= 2:
                loss_bdc = self._disagreement_calibration(
                    ev_list, uncertainty)
                loss = loss + self.disagreement_weight * loss_bdc

        # ── Diagnostics ───────────────────────────────────────────────
        pred = alpha / S
        _, preds = torch.max(pred, dim=1)
        correct = (preds == target).float()
        total_ev = fused_evidence.sum(dim=1)
        ev_succ = (total_ev * correct).sum() / (correct.sum() + self.eps)
        ev_fail = (total_ev * (1 - correct)).sum() / (
            (1 - correct).sum() + self.eps)

        return {
            'loss': loss,
            'alpha': alpha,
            'uncertainty': uncertainty.squeeze(1),
            'pred': pred,
            'loss_nll': loss_nll.detach(),
            'loss_kl': loss_kl.detach(),
            'loss_avu': loss_avu.detach(),
            'loss_aux': loss_aux.detach(),
            'loss_bdc': loss_bdc.detach(),
            'evidence_succ': ev_succ.detach(),
            'evidence_fail': ev_fail.detach(),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Novel Component 1: Confidence-Modulated Evidence Fusion (CMEF)
# ═══════════════════════════════════════════════════════════════════════════

class ConfidenceModulatedEvidenceFusion(nn.Module):
    """
    Confidence-Modulated Evidence Fusion (CMEF).

    Replaces static scalar gates with per-sample dynamic weighting
    based on each branch's Dirichlet strength:

      Standard:  e_total = e_s + sigma(g1) * e_c + sigma(g2) * e_a
      CMEF:      e_total = e_s + sigma(g1) * phi(S_c) * e_c
                               + sigma(g2) * phi(S_a) * e_a

    where phi(S_b) = sigmoid((S_b - K) / tau) is a confidence gate
    that smoothly transitions from ~0.5 (uncertain branch, S ≈ K) to
    ~1.0 (confident branch, S >> K).

    Key property: on OOD data where the spatial/neural branch is
    uncertain, symbolic branches with higher confidence automatically
    get higher relative weight — enabling generalization through
    dynamic neural-symbolic trust allocation.

    The learnable temperature tau controls how sharply confidence
    gates transition. Initialized to 1.0 (moderate sensitivity).
    """

    def __init__(
        self,
        num_symbolic_branches: int = 2,
        num_classes: int = 2,
        gate_inits: list = None,
    ):
        super().__init__()
        self.K = num_classes

        # Learnable base gates (one per symbolic branch)
        if gate_inits is None:
            gate_inits = [-0.5] * num_symbolic_branches
        self.gates = nn.ParameterList([
            nn.Parameter(torch.tensor(float(g))) for g in gate_inits])

        # Learnable temperature for confidence modulation
        self.log_tau = nn.Parameter(torch.tensor(0.0))  # tau = exp(0) = 1.0

        logger.info(
            f"[CMEF] {num_symbolic_branches} symbolic branches, "
            f"K={num_classes}, gate_inits={gate_inits}")

    def confidence_weight(self, evidence: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample confidence weight from branch evidence.

        phi(S) = sigmoid((S - K) / tau)

        When S = K (uniform Dirichlet): phi ≈ 0.5
        When S >> K (high confidence): phi → 1.0
        When S < K (degenerate): phi < 0.5

        Returns: (B, 1) confidence weights in (0, 1).
        """
        alpha = evidence + 1.0
        S = alpha.sum(dim=1, keepdim=True)  # (B, 1)
        tau = self.log_tau.exp().clamp(min=0.1)
        return torch.sigmoid((S - self.K) / tau)

    def forward(
        self,
        spatial_evidence: torch.Tensor,
        symbolic_evidences: list,
    ) -> tuple:
        """
        Args:
            spatial_evidence:   (B, K) base neural evidence (always full weight)
            symbolic_evidences: list of (B, K) evidence from symbolic branches

        Returns:
            fused_evidence: (B, K) confidence-modulated fused evidence
            diagnostics:    dict with gate values and confidence weights
        """
        fused = spatial_evidence.clone()
        diagnostics = {
            'spatial_evidence_mean': spatial_evidence.detach().sum(1).mean(),
        }

        for i, (ev, gate) in enumerate(zip(symbolic_evidences, self.gates)):
            base_gate = torch.sigmoid(gate)
            conf = self.confidence_weight(ev)  # (B, 1)
            modulated_weight = base_gate * conf  # (B, 1) per-sample weight
            fused = fused + modulated_weight * ev

            diagnostics[f'branch_{i}_gate'] = base_gate.detach()
            diagnostics[f'branch_{i}_conf_mean'] = conf.detach().mean()

        diagnostics['tau'] = self.log_tau.exp().detach()

        return fused, diagnostics

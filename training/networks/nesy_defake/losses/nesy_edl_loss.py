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
        ibdc_version: str = 'v1',
        symbolic_reweight_enabled: bool = False,
        symbolic_reweight_factor: float = 2.0,
        symbolic_reweight_warmup: int = 8,
    ):
        super().__init__()
        self.K = num_classes
        # S9: symbolic-guided hard-sample reweighting (training only).
        self.symbolic_reweight_enabled = bool(symbolic_reweight_enabled)
        self.symbolic_reweight_factor = float(symbolic_reweight_factor)
        self.symbolic_reweight_warmup = int(symbolic_reweight_warmup)
        self.annealing_epochs = annealing_epochs
        self.kl_weight = kl_weight
        self.avu_weight = avu_weight
        self.aux_weight = aux_weight
        self.disagreement_weight = disagreement_weight
        if ibdc_version not in ('v1', 'v2', 'v3'):
            raise ValueError(
                f"Unknown ibdc_version: {ibdc_version!r} "
                f"(expected 'v1', 'v2', or 'v3')")
        self.ibdc_version = ibdc_version
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
            f"ibdc={ibdc_version}, sym_reweight="
            f"{self.symbolic_reweight_enabled}(x{self.symbolic_reweight_factor},"
            f"warmup={self.symbolic_reweight_warmup}), class_w={class_weights}")

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
        branch_names: list = None,
    ) -> tuple:
        """
        Encourage fused uncertainty to reflect inter-branch agreement.

        When branches disagree on the predicted class, fused uncertainty
        should be HIGH. When they agree, it should be LOW.

        Uses cosine distance between branch Dirichlet means as the
        disagreement measure, and binary cross-entropy to calibrate:

          d(x) = 1 - cos(p_neural(x), p_symbolic(x))
          L_bdc = -d * log(u) - (1-d) * log(1-u)

        Three variants (config `edl.ibdc_version`), differing in aggregation and
        the loss form on fused vacuity u = K/S (gradient flows through u only):
          v1 (default) — uniform-mean disagreement d, symmetric BCE(u; d).
          v2 (S2)      — vacuity-weighted d (each pair weighted by q_i q_j,
                         q_b = 1 - K/S_b, q detached):
                           d = Σ_{i<j} q_i q_j (1-cos(p_i,p_j)) / (Σ_{i<j} q_i q_j + ε)
                         same symmetric BCE(u; d).
          v3 (R3)      — v2's q-weighted d, but a ONE-SIDED hinge instead of BCE:
                           L = relu(d - u)^2
                         raises u only when disagreement exceeds it, inert
                         otherwise (never suppresses u under agreement).

        Returns (bdc_loss_scalar, {branch_name: mean_commitment_q}).

        This is novel: standard EDL has no mechanism for uncertainty
        to reflect reasoning CONFLICT between evidence sources.
        """
        if len(branch_evidences) < 2:
            return torch.zeros((), device=fused_uncertainty.device), {}

        # Dirichlet mean + commitment q_b = 1 - K/S_b for each branch.
        branch_probs, q_list = [], []
        for ev in branch_evidences:
            alpha = ev + 1.0
            S = alpha.sum(dim=1, keepdim=True)              # (B, 1)
            branch_probs.append(alpha / S)                  # (B, K)
            vacuity = self.K / S.squeeze(1)                 # (B,) in (0, 1]
            q_list.append((1.0 - vacuity).clamp(0.0, 1.0))  # (B,) commitment

        B = branch_probs[0].shape[0]
        device = fused_uncertainty.device
        num = torch.zeros(B, device=device)
        den = torch.zeros(B, device=device)
        for i in range(len(branch_probs)):
            for j in range(i + 1, len(branch_probs)):
                dis_ij = 1.0 - F.cosine_similarity(
                    branch_probs[i], branch_probs[j], dim=1)   # (B,)
                if self.ibdc_version in ('v2', 'v3'):
                    w = (q_list[i] * q_list[j]).detach()       # q detached
                else:
                    w = torch.ones(B, device=device)
                num = num + w * dis_ij
                den = den + w
        # v1: den = n_pairs (≥1) so clamp_min is a no-op → byte-identical to the
        # original uniform mean. v2/v3: den can be ~0 if all branches are vacuous.
        disagreement = (num / den.clamp_min(self.eps)).clamp(0, 1)

        # u = fused vacuity K/S (gradient flows through u only); d is detached.
        u = fused_uncertainty.squeeze().clamp(self.eps, 1.0 - self.eps)
        d = disagreement.detach()  # stop gradient on disagreement target
        if self.ibdc_version == 'v3':
            # v3 (one-sided hinge): raise u only when disagreement EXCEEDS it;
            # inert otherwise, so it never suppresses u under confident-or-
            # ignorant agreement (unlike the symmetric BCE of v1/v2).
            bdc = torch.relu(d - u).pow(2)
        else:
            # Binary cross-entropy: disagreement → high uncertainty
            bdc = -(d * torch.log(u) + (1 - d) * torch.log(1 - u))

        names = branch_names or [f'branch{i}' for i in range(len(q_list))]
        q_means = {nm: q.detach().mean() for nm, q in zip(names, q_list)}
        return bdc.mean(), q_means

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
        nll_ps = nll.squeeze(1)                          # (B,) per-sample NLL
        weights = torch.ones_like(nll_ps)
        if self._class_weights is not None:
            weights = weights * self._class_weights.to(device)[target]

        # S9: symbolic-guided hard-sample reweighting (fused NLL term only).
        # After warmup, upweight samples where the spatial stream is WRONG but
        # the symbolic streams (concept+causal) are RIGHT — the cases where the
        # neuro-symbolic signal should correct the neural one. Training data
        # only (uses in-batch labels); disabled by default.
        reweight_frac = torch.zeros((), device=device)
        if (self.symbolic_reweight_enabled
                and epoch >= self.symbolic_reweight_warmup
                and branch_evidences is not None):
            spat = branch_evidences.get('spatial')
            sym_parts = [branch_evidences.get(k) for k in ('concept', 'causal')]
            sym_parts = [e for e in sym_parts if e is not None]
            if spat is not None and len(sym_parts) >= 1:
                sym = sum(sym_parts)
                spat_pred = spat.argmax(dim=1)
                sym_pred = sym.argmax(dim=1)
                hard = ((spat_pred != target)
                        & (sym_pred == target)).float()  # (B,)
                weights = weights * (
                    1.0 + (self.symbolic_reweight_factor - 1.0) * hard)
                reweight_frac = hard.mean().detach()
        loss_nll = (nll_ps * weights).mean()

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
        ibdc_q_means = {}
        if (branch_evidences is not None
                and self.disagreement_weight > 0
                and len(branch_evidences) >= 2):
            named = [(n, ev) for n, ev in branch_evidences.items()
                     if ev is not None]
            if len(named) >= 2:
                names = [n for n, _ in named]
                ev_list = [ev for _, ev in named]
                loss_bdc, ibdc_q_means = self._disagreement_calibration(
                    ev_list, uncertainty, names)
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
            'ibdc_q_means': ibdc_q_means,   # {branch: mean commitment q_b}
            'symbolic_reweight_frac': reweight_frac.detach(),
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

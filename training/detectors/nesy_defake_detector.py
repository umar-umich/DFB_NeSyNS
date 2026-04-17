"""
detectors/nesy_defake_detector.py
=================================
NeSy-DeFake hybrid detector (EDL evidence fusion).

Forward paths:
  1. Ablation 1 (ablation_spatial_only=True, ablation_mode='spatial_ce')
       CLIP spatial backbone -> spatial_proj -> MultiTaskHead (CE).
  2. Ablation 2/3/4 (ablation_mode in {spatial_edl, concept_edl, causal_edl})
       spatial_proj -> softplus evidence
       + optional concept_branch evidence (Ablation 3+)
       + optional causal_branch evidence (Ablation 4: CCV / ImprovedSCM / Simple)
       fused via NeSy-EDL (CMEF + PBAS + IBDC) or static scalar gates.

Dead pipelines removed (frequency branch, MSCAN CausalDiscoveryModule,
DualBranchSparseAutoencoder, CausalViolationAttentionFusion, SemanticProjection,
FacialSemanticExtractor in-detector, CausalInterventionModule, MultiModalFusion,
per-tier norms, semantic gate, consistency-rule detector member). Precomputed
semantic/forensic features arrive via `data_dict['precomputed_attrs']` and
`data_dict['forensic_features']`; the dataset produces them.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics.base_metrics_class import calculate_metrics_for_train
from .base_detector import AbstractDetector
from detectors import DETECTOR

from networks.nesy_defake.foundation_models import SpatialFeatureExtractor
from networks.nesy_defake.classifiers import MultiTaskHead
from networks.nesy_defake.losses import alignment, uniformity

logger = logging.getLogger(__name__)


def _zero_grad_anchor(module: nn.Module, device: torch.device) -> torch.Tensor:
    anchor = torch.zeros(1, device=device, dtype=torch.float32)
    for p in module.parameters():
        if p.requires_grad:
            anchor = anchor + p.sum() * 0.0
    return anchor


# --------------------------------------------------------------------------- #
# Projection-head variants (selected via config['projection_head']['type'])    #
# --------------------------------------------------------------------------- #

def _make_projection(in_dim: int, out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    layers = [nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU()]
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class ResidualProjection(nn.Module):
    """output = x + alpha * projection(x). Alpha learnable, small-init.

    Early training behaves like a linear probe; model learns how much
    task-specific correction to apply via alpha.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0,
                 alpha_init: float = 0.1):
        super().__init__()
        self.proj = _make_projection(in_dim, out_dim, dropout)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * self.proj(x)


class BottleneckAdapter(nn.Module):
    """Down -> act -> up, skip-connected (Houlsby et al., 2019 — no alpha)."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0,
                 bottleneck_ratio: int = 4):
        super().__init__()
        neck_dim = in_dim // bottleneck_ratio
        self.down = nn.Linear(in_dim, neck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(neck_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.dropout(self.up(self.act(self.down(x)))))


class HoulsbyAdapter(nn.Module):
    """Bottleneck + learnable residual scale (Houlsby et al., ICML 2019)."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0,
                 bottleneck_ratio: int = 4, alpha_init: float = 0.1):
        super().__init__()
        neck_dim = in_dim // bottleneck_ratio
        self.down = nn.Linear(in_dim, neck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(neck_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adapted = self.dropout(self.up(self.act(self.down(x))))
        return self.norm(x + self.alpha * adapted)


class FeatureConditionedGate(nn.Module):
    """Per-image evidence-branch gating via Linear(backbone) -> Sigmoid.

    gate_values = sigmoid(Linear(spatial_raw))  # (B, num_gates)

    Replaces static scalar gates when evidence_gate.conditioned=true —
    lets the model trust the causal branch more on compressed images,
    concept branch more on clear frontal faces, etc.
    """

    def __init__(self, input_dim: int, num_gates: int, gate_inits: list = None):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_gates)
        with torch.no_grad():
            nn.init.zeros_(self.linear.weight)
            if gate_inits is not None:
                for i, val in enumerate(gate_inits):
                    self.linear.bias[i] = val
            else:
                nn.init.constant_(self.linear.bias, -1.0)

    def forward(self, spatial_raw: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(spatial_raw))


@DETECTOR.register_module(module_name='nesydefake_hybrid')
class NeSyDeFakeHybridDetector(AbstractDetector):
    """Spatial-only NeSy-DeFake detector with EDL evidence fusion."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ablation_spatial_only = config.get('ablation_spatial_only', False)
        self.ablation_mode = config.get('ablation_mode', 'spatial_ce')
        # ablation_mode values:
        #   'spatial_ce'   -> Ablation 1 (GenD baseline, CE loss)
        #   'spatial_edl'  -> Ablation 2 (spatial + EDL loss)
        #   'concept_edl'  -> Ablation 3 (spatial + concept branch + EDL)
        #   'causal_edl'   -> Ablation 4 (spatial + concept + causal + EDL)
        self._current_epoch = 0

        self._use_edl = self.ablation_mode in (
            'spatial_edl', 'concept_edl', 'causal_edl')
        self._use_concept_branch = self.ablation_mode in (
            'concept_edl', 'causal_edl')
        self._use_causal_branch = self.ablation_mode == 'causal_edl'
        self._use_nesy_edl = False  # overridden below if edl.nesy_fusion=true

        # -- Backbone + projection head -------------------------------------
        self.build_backbone(config)
        proj_dim = config['foundation_models']['spatial']['output_dim']

        # -- Uniformity-Alignment regularizer (GenD recipe) ------------------
        # Applied on `pred['l2_embeddings']` = F.normalize(projected, p=2, dim=1).
        # Gradient flows only through spatial_proj + backbone LayerNorms; does
        # NOT touch concept_branch, causal_branch (CCV/SCM), CMEF, or IBDC.
        ua_cfg = config.get('uniformity_alignment', {})
        self.use_ua_loss = bool(ua_cfg.get('enabled', False))
        self.ua_alpha = float(ua_cfg.get('alignment_weight', 1.0))
        self.ua_beta = float(ua_cfg.get('uniformity_weight', 1.0))
        self.ua_align_power = float(ua_cfg.get('alignment_power', 2.0))
        self.ua_uniform_t = float(ua_cfg.get('uniformity_t', 2.0))
        if self.use_ua_loss:
            logger.info(
                f"  UA loss         : alignment={self.ua_alpha} "
                f"(pow={self.ua_align_power}), "
                f"uniformity={self.ua_beta} (t={self.ua_uniform_t})")

        # -- Classifier ------------------------------------------------------
        # EDL maps features to Dirichlet parameters deterministically — dropout
        # before evidence computation adds noise to uncertainty estimates, so
        # edl_classifier_dropout overrides classifier dropout in EDL mode.
        if self._use_edl and 'edl_classifier_dropout' in config:
            config['classifier']['dropout'] = config['edl_classifier_dropout']
            logger.info(
                f"  EDL dropout     : classifier dropout overridden to "
                f"{config['edl_classifier_dropout']}")
        self.multitaskhead = MultiTaskHead(config)
        self.loss_weights = config['loss_func']['weights']
        self._base_loss_weights = dict(config['loss_func']['weights'])
        self.build_loss(config)
        self.loss_warmup_cfg = config.get('loss_warmup', {})

        # -- EDL evidence loss (Ablation 2/3/4) ------------------------------
        if self._use_edl:
            edl_cfg = config.get('edl', {})
            use_nesy_edl = edl_cfg.get('nesy_fusion', False)
            if use_nesy_edl:
                from networks.nesy_defake.losses.nesy_edl_loss import (
                    NeSyEvidentialLoss, ConfidenceModulatedEvidenceFusion)
                self.edl_loss = NeSyEvidentialLoss(
                    num_classes=edl_cfg.get('num_classes', 2),
                    annealing_epochs=edl_cfg.get('annealing_epochs', 10),
                    kl_weight=edl_cfg.get('kl_weight', 0.15),
                    avu_weight=edl_cfg.get('avu_weight', 0.1),
                    aux_weight=edl_cfg.get('aux_weight', 0.1),
                    disagreement_weight=edl_cfg.get('disagreement_weight', 0.05),
                    class_weights=config.get('class_weights', None),
                )
                gate_cfg = config.get('evidence_gate', {})
                num_sym = (1 if self._use_concept_branch else 0) \
                        + (1 if self._use_causal_branch else 0)
                gate_inits = []
                if self._use_concept_branch:
                    gate_inits.append(gate_cfg.get('concept_init', -0.5))
                if self._use_causal_branch:
                    gate_inits.append(gate_cfg.get('causal_init', -0.8))
                self.cmef = ConfidenceModulatedEvidenceFusion(
                    num_symbolic_branches=num_sym,
                    num_classes=edl_cfg.get('num_classes', 2),
                    gate_inits=gate_inits,
                )
                logger.info("  NeSy-EDL        : CMEF + PBAS + IBDC enabled")
            else:
                from networks.nesy_defake.losses.edl_loss import EvidentialLoss
                self.edl_loss = EvidentialLoss(
                    num_classes=edl_cfg.get('num_classes', 2),
                    annealing_epochs=edl_cfg.get('annealing_epochs', 10),
                    kl_weight=edl_cfg.get('kl_weight', 0.1),
                    avu_weight=edl_cfg.get('avu_weight', 0.0),
                    class_weights=config.get('class_weights', None),
                )
            self._use_nesy_edl = use_nesy_edl
            logger.info(
                f"  EDL loss        : annealing="
                f"{edl_cfg.get('annealing_epochs', 10)} epochs")

        # -- Concept branch (Ablation 3+) ------------------------------------
        if self._use_concept_branch:
            from networks.nesy_defake.concept_branch import ConceptBranch
            cb_cfg = config.get('concept_branch', {})
            self.concept_branch = ConceptBranch(
                combined_dim=cb_cfg.get('combined_dim', 122),
                rules_dim=cb_cfg.get('rules_dim', 23),
                hidden_dim=cb_cfg.get('hidden_dim', 64),
                dropout=cb_cfg.get('dropout', 0.2),
            )
            gate_cfg = config.get('evidence_gate', {})
            self.concept_gate = nn.Parameter(
                torch.tensor(float(gate_cfg.get('concept_init', -3.0))))
            logger.info(
                f"  Concept branch  : gate_init="
                f"{gate_cfg.get('concept_init', -3.0)}")

        # -- Causal branch (Ablation 4: CCV / ImprovedSCM / Simple) ----------
        if self._use_causal_branch:
            sc_cfg = config.get('causal_branch', {})
            causal_type = sc_cfg.get('type', 'simple')
            if causal_type == 'ccv':
                from networks.nesy_defake.ccv_branch import (
                    CausalConstraintVerificationBranch)
                self.causal_branch = CausalConstraintVerificationBranch(
                    combined_dim=sc_cfg.get('combined_dim', 122),
                    rules_dim=sc_cfg.get('rules_dim', 23),
                    forensic_dim=sc_cfg.get('forensic_dim', 83),
                    backbone_dim=sc_cfg.get('backbone_dim', 1024),
                    num_constraints=sc_cfg.get('num_constraints', 16),
                    constraint_hidden=sc_cfg.get('constraint_hidden', 48),
                    forensic_bottleneck=sc_cfg.get('forensic_bottleneck', 8),
                    cf_z_dim=sc_cfg.get('cf_z_dim', 32),
                    cf_hidden_dim=sc_cfg.get('cf_hidden_dim', 64),
                    evidence_hidden=sc_cfg.get('evidence_hidden', 64),
                    dropout=sc_cfg.get('dropout', 0.2),
                )
            elif causal_type == 'improved_scm':
                from networks.nesy_defake.improved_scm_branch import (
                    ImprovedCausalBranch)
                self.causal_branch = ImprovedCausalBranch(
                    backbone_dim=sc_cfg.get('backbone_dim', 1024),
                    z_causal_dim=sc_cfg.get('z_causal_dim', 32),
                    curated_dim=sc_cfg.get('curated_dim', 51),
                    rules_dim=sc_cfg.get('rules_dim', 23),
                    forensic_dim=sc_cfg.get('forensic_dim', 83),
                    scm_hidden_dim=sc_cfg.get('scm_hidden_dim', 64),
                    summary_dim=sc_cfg.get('summary_dim', 8),
                    evidence_hidden=sc_cfg.get('evidence_hidden', 64),
                    sparsity_penalty=sc_cfg.get('sparsity_penalty', 0.01),
                    divergence_weight=sc_cfg.get('divergence_weight', 0.1),
                    recon_weight=sc_cfg.get('recon_weight', 0.5),
                )
            else:
                from networks.nesy_defake.concept_branch import (
                    SimplifiedCausalBranch)
                self.causal_branch = SimplifiedCausalBranch(
                    backbone_dim=sc_cfg.get('backbone_dim', 1024),
                    z_causal_dim=sc_cfg.get('z_causal_dim', 32),
                    curated_dim=sc_cfg.get('curated_dim', 51),
                    rules_dim=sc_cfg.get('rules_dim', 23),
                    forensic_dim=sc_cfg.get('forensic_dim', 83),
                    hidden_dim=sc_cfg.get('hidden_dim', 64),
                    sparsity_penalty=sc_cfg.get('sparsity_penalty', 0.01),
                )
            self._causal_branch_type = causal_type
            gate_cfg = config.get('evidence_gate', {})
            self.causal_ev_gate = nn.Parameter(
                torch.tensor(float(gate_cfg.get('causal_init', -3.0))))
            self._dag_penalty_weight = sc_cfg.get('dag_penalty_weight', 0.05)
            logger.info(
                f"  Causal branch   : type={causal_type}, "
                f"gate_init={gate_cfg.get('causal_init', -3.0)}")

        # -- Feature-conditioned evidence gates ------------------------------
        gate_cfg = config.get('evidence_gate', {})
        self._conditioned_gates = gate_cfg.get('conditioned', False)
        if self._conditioned_gates and (
                self._use_concept_branch or self._use_causal_branch):
            gate_inits = []
            self._gate_names = []
            if self._use_concept_branch:
                gate_inits.append(float(gate_cfg.get('concept_init', -0.5)))
                self._gate_names.append('concept')
            if self._use_causal_branch:
                gate_inits.append(float(gate_cfg.get('causal_init', -0.8)))
                self._gate_names.append('causal')
            backbone_dim = config['foundation_models']['spatial']['output_dim']
            self.conditioned_gate = FeatureConditionedGate(
                input_dim=backbone_dim,
                num_gates=len(gate_inits),
                gate_inits=gate_inits,
            )
            logger.info(
                f"  Evidence gates  : feature-conditioned "
                f"({backbone_dim}->{len(gate_inits)}), "
                f"init={dict(zip(self._gate_names, gate_inits))}")

        self._try_compile_frozen_modules()

        logger.info("NeSyDeFake detector initialised (spatial + EDL)")
        logger.info(f"  Ablation mode   : {self.ablation_mode}")
        logger.info(f"  EDL enabled     : {self._use_edl}")
        logger.info(f"  Projection dim  : {proj_dim}")

    # ------------------------------------------------------------------ #
    #  Construction                                                        #
    # ------------------------------------------------------------------ #

    def build_backbone(self, config: dict) -> None:
        self.spatial_extractor = SpatialFeatureExtractor(config)

        proj_dim = config['foundation_models']['spatial']['output_dim']
        dropout_cfg = config.get('projection_dropout', {})
        proj_cfg = config.get('projection_head', {})
        proj_type = proj_cfg.get('type', 'standard')
        dropout = dropout_cfg.get('spatial', 0.2)

        if proj_type == 'residual':
            alpha_init = proj_cfg.get('alpha_init', 0.1)
            self.spatial_proj = ResidualProjection(
                proj_dim, proj_dim, dropout, alpha_init)
        elif proj_type == 'bottleneck':
            ratio = proj_cfg.get('bottleneck_ratio', 4)
            self.spatial_proj = BottleneckAdapter(
                proj_dim, proj_dim, dropout, ratio)
        elif proj_type == 'houlsby':
            ratio = proj_cfg.get('bottleneck_ratio', 4)
            alpha_init = proj_cfg.get('alpha_init', 0.1)
            self.spatial_proj = HoulsbyAdapter(
                proj_dim, proj_dim, dropout, ratio, alpha_init)
        else:
            self.spatial_proj = _make_projection(proj_dim, proj_dim, dropout)

        if proj_type != 'standard':
            extra = {k: v for k, v in proj_cfg.items() if k != 'type'}
            logger.info(f"  Projection head : type={proj_type}, {extra}")

    def _try_compile_frozen_modules(self) -> None:
        """torch.compile on the frozen CLIP vision backbone (speed optimization).

        Uses default (inductor) mode, NOT reduce-overhead — the latter uses
        CUDA graphs that overwrite output tensors on replay, breaking
        downstream consumers outside the compiled region.
        """
        if not hasattr(torch, 'compile'):
            return
        backbone = getattr(self.spatial_extractor, 'backbone', None)
        if backbone is None:
            return
        if any(p.requires_grad for p in backbone.parameters()):
            return
        try:
            self.spatial_extractor.backbone = torch.compile(
                backbone, fullgraph=False)
            logger.info("  torch.compile   : spatial_extractor.backbone")
        except Exception as e:
            logger.warning(f"  torch.compile failed for spatial backbone: {e}")

    def update_loss_warmup(self, epoch: int, total_epochs: int) -> None:
        self._current_epoch = epoch
        for loss_name, schedule in self.loss_warmup_cfg.items():
            start_epoch = schedule.get('start_epoch', 0)
            end_epoch = schedule.get('end_epoch', 10)
            start_w = schedule.get('start_weight', 0.01)
            end_w = schedule.get('end_weight', 0.3)
            if epoch < start_epoch:
                w = start_w
            elif epoch >= end_epoch:
                w = end_w
            else:
                progress = (epoch - start_epoch) / max(end_epoch - start_epoch, 1)
                w = start_w + progress * (end_w - start_w)
            self.loss_weights[loss_name] = w

    def build_loss(self, config: dict) -> None:
        cw = config.get('class_weights', None)
        # Label smoothing is load-bearing when UA loss is on: without it,
        # CE drives logits to infinity and alignment cannot shape the sphere.
        ls = float(config.get('label_smoothing', 0.0))
        weight = torch.tensor(cw, dtype=torch.float32) if cw is not None else None
        self.cls_loss = nn.CrossEntropyLoss(weight=weight, label_smoothing=ls)
        if ls > 0:
            logger.info(f"  CE label_smoothing : {ls}")
        self.reg_loss = nn.MSELoss()

    # ------------------------------------------------------------------ #
    #  Feature extraction                                                  #
    # ------------------------------------------------------------------ #

    def extract_raw_features(self, data_dict: dict) -> dict:
        spatial_input = data_dict['spatial_frames']
        if self.spatial_extractor.needs_resize:
            spatial_input = F.interpolate(
                spatial_input,
                size=(self.spatial_extractor.required_size,
                      self.spatial_extractor.required_size),
                mode='bilinear', align_corners=False)
        return {'spatial_raw': self.spatial_extractor(spatial_input)}

    def features(self, data_dict: dict) -> torch.Tensor:
        return self.spatial_proj(self.extract_raw_features(data_dict)['spatial_raw'])

    def classifier(self, features: torch.Tensor) -> dict:
        return self.multitaskhead(features)

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(self, data_dict: dict, inference: bool = False) -> dict:
        device = data_dict['label'].device

        # -- Ablation 1: spatial + CE ---------------------------------------
        if self.ablation_spatial_only and not self._use_edl:
            raw_feats = self.extract_raw_features(data_dict)
            spatial_raw = raw_feats['spatial_raw']
            projected = self.spatial_proj(spatial_raw)
            l2_embeddings = F.normalize(projected, p=2, dim=1)
            task_outputs = self.classifier(projected)
            cls_logits = task_outputs['classification']
            prob = torch.softmax(cls_logits, dim=1)[:, 1]
            return {
                'cls':            cls_logits,
                'prob':           prob,
                'feat':           projected,
                'l2_embeddings':  l2_embeddings,
                'spatial_feat':   spatial_raw,
                'task_outputs':   task_outputs,
            }

        # -- Ablation 2/3/4: spatial + (concept + causal) + EDL -------------
        if self._use_edl:
            raw_feats = self.extract_raw_features(data_dict)
            spatial_raw = raw_feats['spatial_raw']
            projected = self.spatial_proj(spatial_raw)
            l2_embeddings = F.normalize(projected, p=2, dim=1)

            task_outputs = self.classifier(projected)
            spatial_logits = task_outputs['classification']         # (B, 2)
            spatial_evidence = F.softplus(spatial_logits.float())   # (B, 2)

            # Concept evidence (Ablation 3+)
            concept_out = None
            if self._use_concept_branch:
                combined_features = data_dict.get('precomputed_attrs')
                if combined_features is not None:
                    combined_features = combined_features.to(device)
                    concept_out = self.concept_branch(combined_features)

            # Causal evidence (Ablation 4)
            causal_out = None
            if self._use_causal_branch and concept_out is not None:
                forensic_features = data_dict.get('forensic_features')
                if forensic_features is not None:
                    forensic_features = forensic_features.to(device)
                    labels = data_dict.get('label')
                    if labels is not None:
                        labels = labels.to(device)
                    causal_out = self.causal_branch(
                        spatial_raw=spatial_raw,
                        combined_features=combined_features,
                        violations=concept_out['violations'],
                        forensic_features=forensic_features,
                        labels=labels,
                    )

            # -- Evidence fusion --------------------------------------------
            branch_evidences = {'spatial': spatial_evidence}

            if self._use_nesy_edl and hasattr(self, 'cmef'):
                # Novel: Confidence-Modulated Evidence Fusion (replaces CMEF's
                # predecessor CausalViolationAttentionFusion — same per-sample
                # "pick which signal to trust" idea, placed in EDL space).
                symbolic_evs = []
                if concept_out is not None:
                    symbolic_evs.append(concept_out['evidence'])
                    branch_evidences['concept'] = concept_out['evidence']
                if causal_out is not None:
                    symbolic_evs.append(causal_out['evidence'])
                    branch_evidences['causal'] = causal_out['evidence']
                total_evidence, cmef_diag = self.cmef(
                    spatial_evidence, symbolic_evs)
            elif self._conditioned_gates and hasattr(self, 'conditioned_gate'):
                # Feature-conditioned gates: per-image weighting from backbone
                gate_vals = self.conditioned_gate(spatial_raw)  # (B, num_gates)
                total_evidence = spatial_evidence
                gi = 0
                if concept_out is not None:
                    concept_gate = gate_vals[:, gi].unsqueeze(1)
                    total_evidence = total_evidence + concept_gate * concept_out['evidence']
                    branch_evidences['concept'] = concept_out['evidence']
                    gi += 1
                if causal_out is not None:
                    causal_gate = gate_vals[:, gi].unsqueeze(1)
                    total_evidence = total_evidence + causal_gate * causal_out['evidence']
                    branch_evidences['causal'] = causal_out['evidence']
                cmef_diag = None
            else:
                # Standard: static scalar gates
                total_evidence = spatial_evidence
                if concept_out is not None:
                    concept_gate = torch.sigmoid(self.concept_gate)
                    total_evidence = total_evidence + concept_gate * concept_out['evidence']
                    branch_evidences['concept'] = concept_out['evidence']
                if causal_out is not None:
                    causal_gate = torch.sigmoid(self.causal_ev_gate)
                    total_evidence = total_evidence + causal_gate * causal_out['evidence']
                    branch_evidences['causal'] = causal_out['evidence']
                cmef_diag = None

            # Dirichlet prediction from fused evidence
            alpha = total_evidence + 1.0
            S = alpha.sum(dim=1, keepdim=True)
            prob = (alpha / S)[:, 1]                 # P(fake)
            uncertainty = 2.0 / S.squeeze(1)

            pred = {
                'cls':               spatial_logits,
                'prob':              prob,
                'total_evidence':    total_evidence,
                'spatial_evidence':  spatial_evidence,
                'branch_evidences':  branch_evidences,
                'alpha':             alpha,
                'uncertainty':       uncertainty,
                'feat':              projected,
                'l2_embeddings':     l2_embeddings,
                'spatial_feat':      spatial_raw,
                'task_outputs':      task_outputs,
            }
            if concept_out is not None:
                pred['concept_evidence'] = concept_out['evidence']
                if self._use_nesy_edl and cmef_diag is not None:
                    pred['concept_gate'] = cmef_diag.get(
                        'branch_0_gate', torch.tensor(0.0))
                    pred['concept_conf'] = cmef_diag.get(
                        'branch_0_conf_mean', torch.tensor(0.0))
                elif self._conditioned_gates and hasattr(self, 'conditioned_gate'):
                    pred['concept_gate'] = concept_gate.mean().detach()
                elif hasattr(self, 'concept_gate'):
                    pred['concept_gate'] = torch.sigmoid(self.concept_gate).detach()
                pred['violations'] = concept_out['violations']
            if causal_out is not None:
                pred['causal_evidence'] = causal_out['evidence']
                if self._use_nesy_edl and cmef_diag is not None:
                    idx = 1 if concept_out is not None else 0
                    pred['causal_gate'] = cmef_diag.get(
                        f'branch_{idx}_gate', torch.tensor(0.0))
                    pred['causal_conf'] = cmef_diag.get(
                        f'branch_{idx}_conf_mean', torch.tensor(0.0))
                elif self._conditioned_gates and hasattr(self, 'conditioned_gate'):
                    pred['causal_gate'] = causal_gate.mean().detach()
                elif hasattr(self, 'causal_ev_gate'):
                    pred['causal_gate'] = torch.sigmoid(self.causal_ev_gate).detach()
                pred['dag_penalty'] = causal_out['dag_penalty']
                for k, v in causal_out.items():
                    if k.startswith('A_'):
                        pred[k] = v
                for ccv_key in ('violation_scores', 'anomaly_scores',
                                'counterfactual_residual'):
                    if ccv_key in causal_out:
                        pred[ccv_key] = causal_out[ccv_key]
            if cmef_diag is not None and 'tau' in cmef_diag:
                pred['cmef_tau'] = cmef_diag['tau']
            return pred

        raise RuntimeError(
            f"No active forward path for ablation_mode={self.ablation_mode!r}, "
            f"ablation_spatial_only={self.ablation_spatial_only}")

    # ------------------------------------------------------------------ #
    #  Loss                                                                #
    # ------------------------------------------------------------------ #

    def _compute_ua_loss(self, pred_dict: dict, label: torch.Tensor,
                         device: torch.device) -> tuple:
        """Uniformity-Alignment auxiliary loss (GenD recipe).

        Returns (ua_align, ua_uniform, ua_weighted_sum). All three are
        scalar tensors on *device*; zeros if UA is disabled or embeddings
        are missing.
        """
        zero = torch.zeros((), device=device)
        if not self.use_ua_loss:
            return zero, zero, zero
        l2_emb = pred_dict.get('l2_embeddings')
        if l2_emb is None:
            return zero, zero, zero
        l2_emb = l2_emb.float()
        ua_align = alignment(l2_emb, label, alpha=self.ua_align_power)
        ua_uniform = uniformity(l2_emb, t=self.ua_uniform_t)
        weighted = self.ua_alpha * ua_align + self.ua_beta * ua_uniform
        return ua_align, ua_uniform, weighted

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        device = label.device

        # Move class weights to correct device/dtype
        if hasattr(self.cls_loss, 'weight') and self.cls_loss.weight is not None:
            target_dtype = pred_dict['cls'].dtype
            if (self.cls_loss.weight.device != device
                    or self.cls_loss.weight.dtype != target_dtype):
                self.cls_loss.weight = self.cls_loss.weight.to(
                    device=device, dtype=target_dtype)

        cls_loss = self.cls_loss(pred_dict['cls'], label)
        ua_align, ua_uniform, ua_weighted = self._compute_ua_loss(
            pred_dict, label, device)

        # -- Ablation 2/3/4: EDL loss ---------------------------------------
        if self._use_edl:
            branch_evs = pred_dict.get('branch_evidences', None)
            if self._use_nesy_edl and branch_evs is not None:
                edl_out = self.edl_loss(
                    fused_evidence=pred_dict['total_evidence'],
                    target=label,
                    epoch=self._current_epoch,
                    branch_evidences=branch_evs,
                )
            else:
                edl_out = self.edl_loss(
                    evidence=pred_dict['total_evidence'],
                    target=label,
                    epoch=self._current_epoch,
                )
            total_loss = edl_out['loss']

            # DAG penalty for Ablation 4
            dag_penalty = torch.zeros(1, device=device).squeeze()
            if self._use_causal_branch and 'dag_penalty' in pred_dict:
                dag_penalty = pred_dict['dag_penalty']
                total_loss = total_loss + self._dag_penalty_weight * dag_penalty

            # UA auxiliary regularizer on L2-normalized embeddings
            total_loss = total_loss + ua_weighted

            loss_dict = {
                'overall':          total_loss,
                'classification':   edl_out['loss'].detach(),
                'edl_nll':          edl_out['loss_nll'],
                'edl_kl':           edl_out['loss_kl'],
                'dag_penalty':      dag_penalty.detach()
                                    if isinstance(dag_penalty, torch.Tensor)
                                    else dag_penalty,
                'ua_alignment':     ua_align.detach(),
                'ua_uniformity':    ua_uniform.detach(),
            }
            if 'loss_aux' in edl_out:
                loss_dict['edl_aux'] = edl_out['loss_aux']
            if 'loss_bdc' in edl_out:
                loss_dict['edl_bdc'] = edl_out['loss_bdc']
            return loss_dict

        # -- Ablation 1: pure CE + UA ---------------------------------------
        def _scalar(t):
            return t.squeeze() if isinstance(t, torch.Tensor) else t

        overall = cls_loss + ua_weighted
        return {
            'overall':        _scalar(overall),
            'classification': _scalar(cls_loss),
            'ua_alignment':   _scalar(ua_align.detach()),
            'ua_uniformity':  _scalar(ua_uniform.detach()),
        }

    # ------------------------------------------------------------------ #
    #  Metrics                                                             #
    # ------------------------------------------------------------------ #

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(
            label.detach().float(), pred.detach().float())
        metrics = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

        if self._use_edl:
            if 'uncertainty' in pred_dict:
                metrics['uncertainty_mean'] = float(
                    pred_dict['uncertainty'].detach().mean().item())
            if 'concept_gate' in pred_dict:
                metrics['concept_gate'] = float(pred_dict['concept_gate'].item())
            if 'causal_gate' in pred_dict:
                metrics['causal_gate'] = float(pred_dict['causal_gate'].item())
            if 'concept_conf' in pred_dict:
                metrics['concept_conf'] = float(pred_dict['concept_conf'].item())
            if 'causal_conf' in pred_dict:
                metrics['causal_conf'] = float(pred_dict['causal_conf'].item())
            if 'cmef_tau' in pred_dict:
                metrics['cmef_tau'] = float(pred_dict['cmef_tau'].item())

        self.video_names = []
        return metrics

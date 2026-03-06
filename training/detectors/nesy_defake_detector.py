"""
detectors/nesy_defake_hybrid_detector.py
=========================================
FRAME-LEVEL + SAE INTEGRATION (Step 5b):

  SAE wiring: The SAE operates on raw branch features BEFORE projection heads.
  This is critical because:
    - The SAE needs the full 1024-d CLIP representation (not the projected version)
    - SAE features feed the causal module (parallel path)
    - Projected features feed the classifier (main path)

  Data flow:
    spatial_frames ──→ SpatialExtractor ──→ raw_spatial (1024-d) ──┬→ spatial_proj → fused → classifier
                                                                    └→ SAE.spatial → z_spatial ─┐
    freq_frames ────→ FreqExtractor ────→ raw_freq (1024-d) ──────┬→ freq_proj → fused → classifier
                                                                    └→ SAE.freq → z_freq ──────┤
                                                                                                ├→ Z_sae (concat)
    semantic_attrs (73-d from dataset) ─────────────────────────────────────────────────────────┤
                                                                                                ↓
                                                                                    CausalModule (when enabled)
                                                                                                ↓
                                                                                    violation_vector → classifier
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics.base_metrics_class import calculate_metrics_for_train
from .base_detector import AbstractDetector
from detectors import DETECTOR

from networks.nesy_defake.foundation_models import (
    FrequencyFeatureExtractor,
    SpatialFeatureExtractor,
)
from networks.nesy_defake.fusion import MultiModalFusion
from networks.nesy_defake.causal import CausalDiscoveryModule
from networks.nesy_defake.classifiers import MultiTaskHead
from networks.nesy_defake.classifiers.sparse_autoencoder import DualBranchSparseAutoencoder

logger = logging.getLogger(__name__)
ALL_BRANCHES = ('spatial', 'frequency')


def _zero_grad_anchor(module: nn.Module, device: torch.device) -> torch.Tensor:
    anchor = torch.zeros(1, device=device, dtype=torch.float32)
    for p in module.parameters():
        if p.requires_grad:
            anchor = anchor + p.sum() * 0.0
    return anchor


def _make_projection(in_dim: int, out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    layers = [nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU()]
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


@DETECTOR.register_module(module_name='nesydefake_hybrid')
class NeSyDeFakeHybridDetector(AbstractDetector):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.build_backbone(config)
        self.fusion = MultiModalFusion(config)

        # ── Module 4: Sparse Autoencoder ──────────────────────────────────
        # Always built (DDP requires all params present at init).
        # use_sparse controls whether it's actually used in forward().
        self.use_sparse = config['sparse_features']['enabled']
        self.sparse_ae = DualBranchSparseAutoencoder(config)
        logger.info(f"  SAE output dim  : {self.sparse_ae.output_dim}")

        # ── Module 3: Causal Discovery ────────────────────────────────────
        # Always built (DDP consistency); use_causal flipped by enable_causal()
        # at phase3_start via trainer. Module is frozen-out via loss weight=0.
        self.use_causal = config['causal_module']['enabled']
        self.causal_module = CausalDiscoveryModule(config)

        # ── Violation → classifier gated fusion ───────────────────────────
        # Projects 329-d node residual vector into proj_dim space and adds to
        # fused_features before classifier (additive: no input_dim change).
        causal_cfg = config['causal_module']
        violation_dim = (causal_cfg['latent_variables']['total_sae_dim']
                         + causal_cfg['semantic_dim'])
        proj_dim = config['fusion']['projection_dim']
        self.violation_proj = nn.Linear(violation_dim, proj_dim, bias=False)
        nn.init.zeros_(self.violation_proj.weight)  # start neutral; trains in phase3

        # ── Module 5: Classifier ──────────────────────────────────────────
        self.multitaskhead = MultiTaskHead(config)
        self.loss_weights = config['loss_func']['weights']
        self.build_loss(config)

        logger.info("NeSyDeFake Hybrid Detector initialised (FRAME-LEVEL + SAE)")
        logger.info(f"  Active branches : {sorted(self.active_branches)}")
        logger.info(f"  Fused dim       : {config['fusion']['fused_dim']}")
        logger.info(f"  Projection dim  : {config['fusion']['projection_dim']}")
        logger.info(f"  SAE enabled     : {self.use_sparse}")
        logger.info(f"  Causal enabled  : {self.use_causal}")

    # ------------------------------------------------------------------ #
    #  Construction helpers                                                #
    # ------------------------------------------------------------------ #

    def build_backbone(self, config: dict) -> None:
        self.spatial_extractor = SpatialFeatureExtractor(config)
        self.frequency_extractor = FrequencyFeatureExtractor(config)

        active = set(config.get('active_branches', list(ALL_BRANCHES)))
        active = active & set(ALL_BRANCHES)
        self.active_branches = active

        fm = config['foundation_models']
        proj_dim = config['fusion']['projection_dim']

        branch_dims = {
            'spatial': fm['spatial']['output_dim'],
            'frequency': self.frequency_extractor.output_dim,
        }

        dropout_cfg = config.get('projection_dropout', {})
        self.spatial_proj = _make_projection(
            branch_dims['spatial'], proj_dim,
            dropout=dropout_cfg.get('spatial', 0.2))
        self.frequency_proj = _make_projection(
            branch_dims['frequency'], proj_dim,
            dropout=dropout_cfg.get('frequency', 0.1))

        fused_dim = proj_dim * sum(1 for b in ALL_BRANCHES if b in active)
        config['fusion']['fused_dim'] = fused_dim

        for name in ALL_BRANCHES:
            if name not in active:
                self._freeze_module(getattr(self, f'{name}_extractor'), name)

    def _freeze_module(self, module: nn.Module, name: str) -> None:
        for p in module.parameters():
            p.requires_grad = False
        logger.info(f"  Frozen branch   : {name}")

    def enable_causal(self) -> None:
        """Called by trainer at phase3_start to activate causal discovery."""
        self.use_causal = True
        logger.info("Causal module enabled (phase 3)")

    def enable_sparse(self) -> None:
        """Called by trainer if SAE needs to be enabled at a phase transition."""
        self.use_sparse = True
        logger.info("SAE enabled")

    def build_loss(self, config: dict) -> None:
        cw = config.get('class_weights', None)
        if cw is not None:
            weight = torch.tensor(cw, dtype=torch.float32)
            self.cls_loss = nn.CrossEntropyLoss(weight=weight)
        else:
            self.cls_loss = nn.CrossEntropyLoss()
        self.reg_loss = nn.MSELoss()

    # ------------------------------------------------------------------ #
    #  Feature extraction                                                  #
    # ------------------------------------------------------------------ #

    def extract_raw_features(self, data_dict: dict) -> dict:
        """
        Extract raw (pre-projection) branch features.
        These feed both the projection heads AND the SAE.

        Returns dict with 'spatial_raw', 'frequency_raw' tensors.
        """
        raw = {}
        if 'spatial' in self.active_branches:
            raw['spatial_raw'] = self.spatial_extractor(data_dict['spatial_frames'])
        if 'frequency' in self.active_branches:
            raw['frequency_raw'] = self.frequency_extractor(data_dict['freq_frames'])
        return raw
    
    def features(self):
        pass # maybe replaced with project_and_fuse for SAE 

    def project_and_fuse(self, raw_feats: dict) -> torch.Tensor:
        """
        Apply projection heads and fuse.
        """
        spatial_proj = (
            self.spatial_proj(raw_feats['spatial_raw'])
            if 'spatial_raw' in raw_feats else None
        )
        freq_proj = (
            self.frequency_proj(raw_feats['frequency_raw'])
            if 'frequency_raw' in raw_feats else None
        )
        return self.fusion(spatial_proj, freq_proj)

    def classifier(self, features: torch.Tensor) -> dict:
        return self.multitaskhead(features)

    # ------------------------------------------------------------------ #
    #  Forward pass                                                        #
    # ------------------------------------------------------------------ #

    def forward(self, data_dict: dict, inference: bool = False) -> dict:
        device = data_dict['label'].device

        # ── Step 1: Extract raw branch features ───────────────────────────
        raw_feats = self.extract_raw_features(data_dict)

        # ── Step 2: Project and fuse for classifier ───────────────────────
        fused_features = self.project_and_fuse(raw_feats)

        # ── Step 3: SAE on raw features (parallel path) ──────────────────
        z_spatial = None
        z_freq = None
        sae_loss = torch.zeros(1, device=device)
        sae_info = {}

        if self.use_sparse:
            z_spatial, z_freq, sae_loss, sae_info = self.sparse_ae(
                spatial_feat=raw_feats.get('spatial_raw'),
                frequency_feat=raw_feats.get('frequency_raw'),
            )

        # ── Step 4: Causal module (uses SAE features + semantic attrs) ────
        violation_score = None
        node_residuals = None
        causal_dag = None

        if self.use_causal:
            z_sae = self.sparse_ae.get_z_sae(z_spatial, z_freq) if self.use_sparse else None
            semantic_attrs = data_dict.get('semantic_attrs', None)
            label = data_dict.get('label', None) if not inference else None

            # Causal module takes only z_sae and semantic_attrs — the two
            # semantically grounded inputs that form causal variables.
            # fused_features is NOT passed: it conflates spatial/frequency
            # projections that are already summarised by z_sae, and passing
            # the fused representation would introduce redundancy and pollute
            # the causal graph with classification-head artifacts.
            violation_score, node_residuals, causal_dag = self.causal_module(
                z_sae=z_sae,
                semantic_attrs=semantic_attrs,
                label=label,
                return_graph=True,
            )

        # ── Step 5: Gated violation fusion + Classification ───────────────
        # Project the 329-d node-residual vector into proj_dim and ADD to
        # fused_features. classifier.input_dim stays 1024 (additive, not cat).
        # violation_proj is zero-init so phase1/2 classifier is unaffected.
        classifier_input = fused_features
        if self.use_causal and node_residuals is not None:
            classifier_input = fused_features + self.violation_proj(node_residuals)

        task_outputs = self.classifier(classifier_input)
        cls_logits = task_outputs['classification']
        prob = torch.softmax(cls_logits, dim=1)[:, 1]

        pred_dict = {
            'cls': cls_logits,
            'prob': prob,
            'feat': fused_features,
            'spatial_feat': raw_feats.get('spatial_raw'),
            'frequency_feat': raw_feats.get('frequency_raw'),
            'z_spatial': z_spatial,
            'z_freq': z_freq,
            'uncertainty': task_outputs.get('uncertainty'),
            'violation_score': violation_score,
            'node_residuals': node_residuals,
            'task_outputs': task_outputs,
            'sae_loss': sae_loss,
            'sae_info': sae_info,
            'causal_dag': causal_dag,
        }
        return pred_dict

    # ------------------------------------------------------------------ #
    #  Loss computation                                                    #
    # ------------------------------------------------------------------ #

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        device = label.device

        # Move class weights to correct device
        if hasattr(self.cls_loss, 'weight') and self.cls_loss.weight is not None:
            target_dtype = pred_dict['cls'].dtype
            if (self.cls_loss.weight.device != device
                    or self.cls_loss.weight.dtype != target_dtype):
                self.cls_loss.weight = self.cls_loss.weight.to(
                    device=device, dtype=target_dtype)

        cls_loss = self.cls_loss(pred_dict['cls'], label)

        # Uncertainty loss
        uncertainty_loss = torch.zeros(1, device=device)
        if pred_dict.get('uncertainty') is not None:
            pred_label = pred_dict['cls'].argmax(dim=1)
            is_correct = (pred_label == label).float()
            uncertainty_loss = self.reg_loss(
                pred_dict['uncertainty'].squeeze(), 1 - is_correct)

        # Causal losses (only active when use_causal=True)
        causal_loss = torch.zeros(1, device=device)      # structural + DAG penalty
        contrastive_loss = torch.zeros(1, device=device)  # margin contrastive on violations
        if self.use_causal:
            node_residuals = pred_dict.get('node_residuals')
            violation_score = pred_dict.get('violation_score')

            # L_structural: SCM reconstruction quality on REAL frames only
            # (structural equations fit to real-face biomechanics)
            if node_residuals is not None:
                real_mask = (label == 0)
                if real_mask.any():
                    causal_loss = node_residuals[real_mask].pow(2).mean()

            # DAG acyclicity penalty (added to structural loss)
            if pred_dict.get('causal_dag') is not None:
                dag_penalty = self.causal_module.causal_learner.compute_dag_penalty()
                causal_loss = causal_loss + (
                    self.config['causal_module']['dag_learning']['dag_penalty_weight']
                    * dag_penalty)

            # L_contrastive: push fake violations UP, real violations DOWN
            if violation_score is not None:
                real_mask = (label == 0)
                fake_mask = (label == 1)
                if real_mask.any() and fake_mask.any():
                    margin = 1.0
                    real_v = violation_score[real_mask].mean()
                    fake_v = violation_score[fake_mask].mean()
                    contrastive_loss = F.relu(margin - (fake_v - real_v))
                elif real_mask.any():
                    # Only reals in batch: minimise their violation scores
                    contrastive_loss = violation_score[real_mask].mean().clamp(min=0)
                elif fake_mask.any():
                    # Only fakes in batch: maximise their violation scores
                    contrastive_loss = F.relu(1.0 - violation_score[fake_mask].mean())

        # SAE loss (reconstruction + auxiliary)
        sae_loss = pred_dict.get('sae_loss', torch.zeros(1, device=device))
        if not isinstance(sae_loss, torch.Tensor):
            sae_loss = torch.zeros(1, device=device)

        # DDP anchor
        ddp_anchor = torch.zeros(1, device=device)
        branch_pairs = {
            'spatial': (self.spatial_extractor, self.spatial_proj),
            'frequency': (self.frequency_extractor, self.frequency_proj),
        }
        for name, (extractor, proj) in branch_pairs.items():
            if name not in self.active_branches:
                ddp_anchor = ddp_anchor + _zero_grad_anchor(extractor, device)
                ddp_anchor = ddp_anchor + _zero_grad_anchor(proj, device)

        for attr in ('causal_module', 'sparse_ae', 'multitaskhead', 'violation_proj'):
            mod = getattr(self, attr, None)
            if mod is not None:
                ddp_anchor = ddp_anchor + _zero_grad_anchor(mod, device)

        total_loss = (
            self.loss_weights['classification'] * cls_loss
            + self.loss_weights.get('uncertainty', 0.0) * uncertainty_loss
            + self.loss_weights.get('causal', 0.0) * causal_loss
            + self.loss_weights.get('contrastive', 0.0) * contrastive_loss
            + self.loss_weights.get('sparse', 0.0) * sae_loss
            + ddp_anchor
        )

        def _scalar(t):
            return t.squeeze() if isinstance(t, torch.Tensor) else t

        return {
            'overall': _scalar(total_loss),
            'classification': _scalar(cls_loss),
            'uncertainty': _scalar(uncertainty_loss),
            'causal': _scalar(causal_loss),
            'contrastive': _scalar(contrastive_loss),
            'sparse': _scalar(sae_loss),
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

        if pred_dict.get('uncertainty') is not None:
            metrics['mean_uncertainty'] = float(
                pred_dict['uncertainty'].detach().float().cpu().numpy().mean())

        # SAE diagnostics
        sae_info = pred_dict.get('sae_info', {})
        for branch_name, branch_info in sae_info.items():
            if isinstance(branch_info, dict):
                metrics[f'sae_{branch_name}_l0'] = float(branch_info.get('l0', 0))
                metrics[f'sae_{branch_name}_fvu'] = float(branch_info.get('fvu', 0))
                metrics[f'sae_{branch_name}_dead'] = int(branch_info.get('n_dead', 0))

        self.video_names = []
        return metrics
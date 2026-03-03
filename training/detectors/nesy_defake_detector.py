"""
detectors/nesy_defake_hybrid_detector.py
=========================================
FRAME-LEVEL REVISION (Step 5):

  Major changes from video-level version:
    - Temporal branch REMOVED entirely (extractor, projection, consistency loss)
    - Input is now (B, C, H, W) per branch, not (B, T, C, H, W)
    - No more _pool_and_project — extractors receive flat batches directly
    - Semantic attributes (73-d) passed through data_dict for causal module
    - active_branches now only supports 'spatial' and 'frequency'

  Preserved from previous version:
    - Class-weighted CrossEntropyLoss (Fix 1)
    - Projection head dropout (Fix 2)
    - DDP zero-grad anchors for frozen modules
    - Per-module loss weighting
    - Causal module, sparse autoencoder, semantic grounding (when enabled)
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
from networks.nesy_defake.classifiers import MultiTaskHead, SparseAutoencoder

logger = logging.getLogger(__name__)
ALL_BRANCHES = ('spatial', 'frequency')


def _zero_grad_anchor(module: nn.Module, device: torch.device) -> torch.Tensor:
    """DDP anchor: ensures all params participate in backward even if unused."""
    anchor = torch.zeros(1, device=device, dtype=torch.float32)
    for p in module.parameters():
        if p.requires_grad:
            anchor = anchor + p.sum() * 0.0
    return anchor


def _make_projection(in_dim: int, out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    """
    Learnable branch projection head: Linear -> LayerNorm -> GELU -> Dropout.
    """
    layers: list = [
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.GELU(),
    ]
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

        # ── Causal Module ─────────────────────────────────────────────────
        self.use_causal = config['causal_module']['enabled']
        if self.use_causal:
            self.causal_module = CausalDiscoveryModule(config)

        # ── Sparse Autoencoder ────────────────────────────────────────────
        self.use_sparse = config['sparse_features']['enabled']
        if self.use_sparse:
            self.sparse_ae = SparseAutoencoder(config)

        # ── Classifier ────────────────────────────────────────────────────
        self.multitaskhead = MultiTaskHead(config)
        self.loss_weights = config['loss_func']['weights']
        self.build_loss(config)

        logger.info("NeSyDeFake Hybrid Detector initialised (FRAME-LEVEL)")
        logger.info(f"  Active branches : {sorted(self.active_branches)}")
        logger.info(f"  Fused dim       : {config['fusion']['fused_dim']}")
        logger.info(f"  Projection dim  : {config['fusion']['projection_dim']}")

    # ------------------------------------------------------------------ #
    #  Construction helpers                                                #
    # ------------------------------------------------------------------ #

    def build_backbone(self, config: dict) -> None:
        self.spatial_extractor   = SpatialFeatureExtractor(config)
        self.frequency_extractor = FrequencyFeatureExtractor(config)

        active = set(config.get('active_branches', list(ALL_BRANCHES)))
        # Filter out temporal if someone left it in config
        active = active & set(ALL_BRANCHES)
        self.active_branches = active

        fm       = config['foundation_models']
        proj_dim = config['fusion']['projection_dim']

        branch_dims = {
            'spatial':   fm['spatial']['output_dim'],
            'frequency': self.frequency_extractor.output_dim,
        }

        # Per-branch dropout from config
        dropout_cfg = config.get('projection_dropout', {})
        self.spatial_proj = _make_projection(
            branch_dims['spatial'], proj_dim,
            dropout=dropout_cfg.get('spatial', 0.2))
        self.frequency_proj = _make_projection(
            branch_dims['frequency'], proj_dim,
            dropout=dropout_cfg.get('frequency', 0.1))

        fused_dim = proj_dim * sum(1 for b in ALL_BRANCHES if b in active)
        config['fusion']['fused_dim'] = fused_dim

        logger.info(
            f"  Branch dims     : "
            f"spatial={branch_dims['spatial']}, "
            f"frequency={branch_dims['frequency']}"
        )
        logger.info(f"  Projection dim  : {proj_dim}")
        logger.info(
            f"  Proj dropout    : "
            f"spatial={dropout_cfg.get('spatial', 0.2)}, "
            f"frequency={dropout_cfg.get('frequency', 0.1)}"
        )
        logger.info(
            f"  Fused dim       : {fused_dim} "
            f"(active: {sorted(active)}, {len(active)} x {proj_dim})"
        )

        # Freeze inactive branches
        for name in ALL_BRANCHES:
            if name not in active:
                self._freeze_module(
                    getattr(self, f'{name}_extractor'), name)

    def _freeze_module(self, module: nn.Module, name: str) -> None:
        for p in module.parameters():
            p.requires_grad = False
        logger.info(f"  Frozen branch   : {name}")

    def build_loss(self, config: dict) -> None:
        """CrossEntropyLoss with optional class weights."""
        cw = config.get('class_weights', None)
        if cw is not None:
            weight = torch.tensor(cw, dtype=torch.float32)
            self.cls_loss = nn.CrossEntropyLoss(weight=weight)
            logger.info(f"  Class weights   : real={cw[0]}, fake={cw[1]}")
        else:
            self.cls_loss = nn.CrossEntropyLoss()
            logger.info("  Class weights   : none (uniform)")

        self.reg_loss = nn.MSELoss()
        self.l1_loss  = nn.L1Loss()

    # ------------------------------------------------------------------ #
    #  Feature extraction — frame-level (no temporal dimension)            #
    # ------------------------------------------------------------------ #

    def extract_spatial_features(self, spatial_frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spatial_frames: (B, C, H, W)
        Returns:
            (B, projection_dim)
        """
        feats = self.spatial_extractor(spatial_frames)   # (B, spatial_dim)
        return self.spatial_proj(feats)                  # (B, proj_dim)

    def extract_frequency_features(self, freq_frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            freq_frames: (B, C, H, W)
        Returns:
            (B, projection_dim)
        """
        feats = self.frequency_extractor(freq_frames)    # (B, freq_dim)
        return self.frequency_proj(feats)                # (B, proj_dim)

    def features(self, data_dict: dict) -> tuple:
        """
        Extract and fuse features from active branches.

        Returns:
            (fused_features, spatial_feat, frequency_feat)
        """
        spatial_feat = (
            self.extract_spatial_features(data_dict['spatial_frames'])
            if 'spatial' in self.active_branches else None
        )
        frequency_feat = (
            self.extract_frequency_features(data_dict['freq_frames'])
            if 'frequency' in self.active_branches else None
        )

        # Fusion expects (temporal=None, spatial, frequency)
        fused_features = self.fusion(spatial_feat, frequency_feat)
        return fused_features, spatial_feat, frequency_feat

    def classifier(self, features: torch.Tensor) -> dict:
        return self.multitaskhead(features)

    # ------------------------------------------------------------------ #
    #  Forward pass                                                        #
    # ------------------------------------------------------------------ #

    def forward(self, data_dict: dict, inference: bool = False) -> dict:
        device = data_dict['label'].device

        # ── Feature extraction ────────────────────────────────────────────
        fused_features, spatial_feat, frequency_feat = self.features(data_dict)

        # ── Causal module ─────────────────────────────────────────────────
        violation_score = None
        causal_dag      = None
        causal_concepts = None
        classifier_input = fused_features

        if self.use_causal:
            # Pass semantic attributes from dataset for causal discovery
            semantic_attrs = data_dict.get('semantic_attrs', None)
            if not inference:
                violation_score, causal_dag, causal_concepts = self.causal_module(
                    fused_features,
                    semantic_attrs=semantic_attrs,
                    return_graph=True,
                )
            else:
                violation_score = self.causal_module(
                    fused_features,
                    semantic_attrs=semantic_attrs,
                )

        # ── Sparse autoencoder ────────────────────────────────────────────
        sparse_loss = None
        if self.use_sparse:
            sparse_features, sparse_loss = self.sparse_ae(classifier_input)
            classifier_input = sparse_features

        # ── Classification ────────────────────────────────────────────────
        task_outputs = self.classifier(classifier_input)
        cls_logits   = task_outputs['classification']
        prob         = torch.softmax(cls_logits, dim=1)[:, 1]

        pred_dict = {
            'cls':               cls_logits,
            'prob':              prob,
            'feat':              fused_features,
            'spatial_feat':      spatial_feat,
            'frequency_feat':    frequency_feat,
            'causal_concepts':   causal_concepts,
            'uncertainty':       task_outputs.get('uncertainty'),
            'violation_score':   task_outputs.get('violation_score'),
            'task_outputs':      task_outputs,
            'sparse_loss':       sparse_loss,
            'causal_dag':        causal_dag,
        }
        return pred_dict

    # ------------------------------------------------------------------ #
    #  Loss computation                                                    #
    # ------------------------------------------------------------------ #

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label  = data_dict['label']
        device = label.device

        # Move class weights to correct device/dtype on first call
        if (hasattr(self.cls_loss, 'weight')
                and self.cls_loss.weight is not None):
            target_dtype = pred_dict['cls'].dtype
            if (self.cls_loss.weight.device != device
                    or self.cls_loss.weight.dtype != target_dtype):
                self.cls_loss.weight = self.cls_loss.weight.to(
                    device=device, dtype=target_dtype)

        cls_loss = self.cls_loss(pred_dict['cls'], label)

        # ── Uncertainty loss ──────────────────────────────────────────────
        uncertainty_loss = torch.zeros(1, device=device)
        if pred_dict.get('uncertainty') is not None:
            pred_label = pred_dict['cls'].argmax(dim=1)
            is_correct = (pred_label == label).float()
            uncertainty_loss = self.reg_loss(
                pred_dict['uncertainty'].squeeze(), 1 - is_correct)

        # ── Causal loss ───────────────────────────────────────────────────
        causal_loss = torch.zeros(1, device=device)
        if self.use_causal and pred_dict.get('violation_score') is not None:
            causal_loss = self.reg_loss(
                pred_dict['violation_score'], label.float())
            if pred_dict.get('causal_dag') is not None:
                dag_penalty = self.causal_module.causal_learner.compute_dag_penalty()
                causal_loss = causal_loss + (
                    self.config['causal_module']['dag_learning']['dag_penalty_weight']
                    * dag_penalty
                )

        # ── Sparse loss ───────────────────────────────────────────────────
        sparse_loss = pred_dict.get('sparse_loss')
        if sparse_loss is None or not isinstance(sparse_loss, torch.Tensor):
            sparse_loss = torch.zeros(1, device=device)

        # ── DDP anchor for frozen/unused modules ──────────────────────────
        ddp_anchor = torch.zeros(1, device=device)
        branch_pairs = {
            'spatial':   (self.spatial_extractor,   self.spatial_proj),
            'frequency': (self.frequency_extractor, self.frequency_proj),
        }
        for name, (extractor, proj) in branch_pairs.items():
            if name not in self.active_branches:
                ddp_anchor = ddp_anchor + _zero_grad_anchor(extractor, device)
                ddp_anchor = ddp_anchor + _zero_grad_anchor(proj, device)

        for attr in ('causal_module', 'sparse_ae', 'multitaskhead'):
            mod = getattr(self, attr, None)
            if mod is not None:
                ddp_anchor = ddp_anchor + _zero_grad_anchor(mod, device)

        # ── Total loss ────────────────────────────────────────────────────
        total_loss = (
            self.loss_weights['classification']                 * cls_loss
            + self.loss_weights.get('uncertainty',    0.0)     * uncertainty_loss
            + self.loss_weights.get('causal',         0.0)     * causal_loss
            + self.loss_weights.get('sparse',         0.0)     * sparse_loss
            + ddp_anchor
        )

        def _scalar(t):
            return t.squeeze() if isinstance(t, torch.Tensor) else t

        return {
            'overall':        _scalar(total_loss),
            'classification': _scalar(cls_loss),
            'uncertainty':    _scalar(uncertainty_loss),
            'causal':         _scalar(causal_loss),
            'sparse':         _scalar(sparse_loss),
        }

    # ------------------------------------------------------------------ #
    #  Metrics                                                             #
    # ------------------------------------------------------------------ #

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred  = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(
            label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        if pred_dict.get('uncertainty') is not None:
            uncertainty = pred_dict['uncertainty'].detach().cpu().numpy()
            metric_batch_dict['mean_uncertainty'] = float(uncertainty.mean())
        self.video_names = []
        return metric_batch_dict
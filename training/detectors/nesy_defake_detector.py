"""
detectors/nesy_defake_hybrid_detector.py  — ABLATION-AWARE VERSION
===================================================================
Changes from original:
  1. build_backbone() computes and injects config['fusion']['fused_dim']
     dynamically from active_branches so MultiModalFusion always sees the
     correct input dimension (Option A).
  2. features() passes None for inactive branches instead of running them,
     saving GPU memory and compute.
  3. The DDP anchor in get_losses() still covers ALL branch parameters
     (including inactive ones) so DDP never complains.
"""

import logging
import torch
import torch.nn as nn
import yaml

from metrics.base_metrics_class import calculate_metrics_for_train
from .base_detector import AbstractDetector
from detectors import DETECTOR

from networks.nesy_defake.foundation_models import (
    TemporalFeatureExtractor,
    FrequencyFeatureExtractor,
    SpatialFeatureExtractor,
)
from networks.nesy_defake.fusion import MultiModalFusion
from networks.nesy_defake.causal import CausalDiscoveryModule
from networks.nesy_defake.classifiers import MultiTaskHead, SparseAutoencoder
from utils.semantic_grounding import SemanticGroundingModule

logger = logging.getLogger(__name__)
ALL_BRANCHES = ('temporal', 'spatial', 'frequency')


def _zero_grad_anchor(module: nn.Module, device: torch.device) -> torch.Tensor:
    anchor = torch.zeros(1, device=device, dtype=torch.float32)
    for p in module.parameters():
        if p.requires_grad:
            anchor = anchor + p.sum() * 0.0
    return anchor


@DETECTOR.register_module(module_name='nesydefake_hybrid')
class NeSyDeFakeHybridDetector(AbstractDetector):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # ── Backbone (sets self.active_branches & patches fused_dim) ──────
        self.build_backbone(config)

        # ── Fusion ────────────────────────────────────────────────────────
        self.fusion = MultiModalFusion(config)

        # ── Semantic Grounding ────────────────────────────────────────────
        self.use_semantic_grounding = config.get(
            'semantic_grounding', {}).get('enabled', False)
        if self.use_semantic_grounding:
            self.semantic_grounding = SemanticGroundingModule(
                config,
                feature_dim=config['fusion']['projection_dim'],
            )
            self.grounded_feature_dim = self.semantic_grounding.output_dim
        else:
            self.grounded_feature_dim = config['fusion']['projection_dim']

        # ── Causal Discovery ──────────────────────────────────────────────
        self.use_causal = config['causal_module']['enabled']
        if self.use_causal:
            self.causal_module = CausalDiscoveryModule(config)

        # ── Sparse Features ───────────────────────────────────────────────
        self.use_sparse = config['sparse_features']['enabled']
        if self.use_sparse:
            self.sparse_ae = SparseAutoencoder(config)

        # ── Multi-task Classifier ─────────────────────────────────────────
        self.multitaskhead = MultiTaskHead(config)

        # ── Loss ──────────────────────────────────────────────────────────
        self.loss_weights = config['loss_func']['weights']
        self.build_loss(config)

        # ── Processing config ─────────────────────────────────────────────
        self.num_clips = config.get('num_clips_per_video', 32)
        self.clip_size  = config.get('clip_size', 16)

        logger.info("NeSyDeFake Hybrid Detector initialised (ablation-aware)")
        logger.info(f"  - Active branches : {sorted(self.active_branches)}")
        logger.info(f"  - Fused dim       : {config['fusion']['fused_dim']}")
        logger.info(f"  - Projection dim  : {config['fusion']['projection_dim']}")

    # ------------------------------------------------------------------ #
    #  Construction helpers                                                #
    # ------------------------------------------------------------------ #

    def build_backbone(self, config: dict) -> None:
        """
        Instantiate all three extractors, freeze inactive ones, and —
        crucially — patch config['fusion']['fused_dim'] to match the sum
        of active-branch output dimensions so MultiModalFusion is correct.
        """
        self.temporal_extractor  = TemporalFeatureExtractor(config)
        self.spatial_extractor   = SpatialFeatureExtractor(config)
        self.frequency_extractor = FrequencyFeatureExtractor(config)

        active = set(config.get('active_branches', list(ALL_BRANCHES)))
        self.active_branches = active

        # ── Per-branch output dims ────────────────────────────────────────
        fm = config['foundation_models']
        branch_dims = {
            'temporal':  fm['temporal']['output_dim'],
            'spatial':   fm['spatial']['output_dim'],
            'frequency': fm['frequency']['output_dim'],
        }

        # ── Option A: compute fused_dim from active branches only ─────────
        fused_dim = sum(branch_dims[b] for b in ALL_BRANCHES if b in active)
        config['fusion']['fused_dim'] = fused_dim   # patch in-place for MultiModalFusion

        logger.info(
            f"  - Branch dims     : temporal={branch_dims['temporal']}, "
            f"spatial={branch_dims['spatial']}, frequency={branch_dims['frequency']}"
        )
        logger.info(
            f"  - Computed fused_dim={fused_dim} "
            f"(active: {sorted(active)})"
        )

        # ── Freeze inactive branch weights ────────────────────────────────
        branch_map = {
            'temporal':  self.temporal_extractor,
            'spatial':   self.spatial_extractor,
            'frequency': self.frequency_extractor,
        }
        for name, module in branch_map.items():
            if name not in active:
                self._freeze_module(module, name)

    def _freeze_module(self, module: nn.Module, name: str) -> None:
        for p in module.parameters():
            p.requires_grad = False
        logger.info(f"  - Frozen branch   : {name}")

    def build_loss(self, config: dict) -> None:
        self.cls_loss = nn.CrossEntropyLoss()
        self.reg_loss = nn.MSELoss()
        self.l1_loss  = nn.L1Loss()

    # ------------------------------------------------------------------ #
    #  Feature extraction                                                  #
    # ------------------------------------------------------------------ #

    def extract_temporal_features(self, video_clips: torch.Tensor) -> torch.Tensor:
        return self.temporal_extractor(video_clips)

    def extract_spatial_features(self, spatial_frames: torch.Tensor) -> torch.Tensor:
        return self.spatial_extractor(spatial_frames)

    def extract_frequency_features(self, frequency_frames: torch.Tensor) -> torch.Tensor:
        return self.frequency_extractor(frequency_frames)

    def features(self, data_dict: dict) -> tuple:
        """
        Run only the active branches.  Inactive branches return None,
        which MultiModalFusion accepts and skips.
        """
        temporal_feat  = (self.extract_temporal_features(data_dict['temporal_clip'])
                          if 'temporal' in self.active_branches else None)
        spatial_feat   = (self.extract_spatial_features(data_dict['spatial_frame'])
                          if 'spatial' in self.active_branches else None)
        frequency_feat = (self.extract_frequency_features(data_dict['frequency_frame'])
                          if 'frequency' in self.active_branches else None)

        fused_features = self.fusion(temporal_feat, spatial_feat, frequency_feat)
        return fused_features, temporal_feat, spatial_feat, frequency_feat

    def classifier(self, features: torch.Tensor) -> dict:
        return self.multitaskhead(features)

    # ------------------------------------------------------------------ #
    #  Forward pass                                                        #
    # ------------------------------------------------------------------ #

    def forward(self, data_dict: dict, inference: bool = False) -> dict:
        device = data_dict['label'].device

        fused_features, temporal_feat, spatial_feat, frequency_feat = \
            self.features(data_dict)

        # ── Semantic Grounding ────────────────────────────────────────────
        semantic_concepts = None
        grounded_features = fused_features
        if self.use_semantic_grounding:
            grounded_features, semantic_concepts = self.semantic_grounding(
                fused_features,
                data_dict['raw_frame'],
            )

        # ── Causal Discovery ──────────────────────────────────────────────
        violation_score = None
        causal_dag      = None
        causal_concepts = None
        if self.use_causal:
            if not inference:
                violation_score, causal_dag, causal_concepts = self.causal_module(
                    grounded_features,
                    explicit_concepts=semantic_concepts,
                    return_graph=True,
                )
            else:
                violation_score = self.causal_module(
                    grounded_features,
                    explicit_concepts=semantic_concepts,
                )

        # ── Sparse Features ───────────────────────────────────────────────
        sparse_loss = None
        if self.use_sparse:
            sparse_features, sparse_loss = self.sparse_ae(grounded_features)
            classifier_input = sparse_features
        else:
            classifier_input = grounded_features

        # ── Classification ────────────────────────────────────────────────
        task_outputs = self.classifier(classifier_input)
        cls_logits   = task_outputs['classification']
        prob         = torch.softmax(cls_logits, dim=1)[:, 1]

        pred_dict = {
            'cls':               cls_logits,
            'prob':              prob,
            'feat':              fused_features,
            'grounded_feat':     grounded_features,
            'temporal_feat':     temporal_feat,
            'spatial_feat':      spatial_feat,
            'frequency_feat':    frequency_feat,
            'semantic_concepts': semantic_concepts,
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

        cls_loss = self.cls_loss(pred_dict['cls'], label)

        uncertainty_loss = torch.zeros(1, device=device)
        if pred_dict.get('uncertainty') is not None:
            pred_label = pred_dict['cls'].argmax(dim=1)
            is_correct = (pred_label == label).float()
            uncertainty_loss = self.reg_loss(
                pred_dict['uncertainty'].squeeze(), 1 - is_correct)

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

        sparse_loss = pred_dict.get('sparse_loss')
        if sparse_loss is None or not isinstance(sparse_loss, torch.Tensor):
            sparse_loss = torch.zeros(1, device=device)

        # ── DDP anchor: touch ALL branch params regardless of active set ──
        ddp_anchor = torch.zeros(1, device=device)
        branch_map = {
            'temporal':  self.temporal_extractor,
            'spatial':   self.spatial_extractor,
            'frequency': self.frequency_extractor,
        }
        for name, module in branch_map.items():
            if name not in self.active_branches:
                ddp_anchor = ddp_anchor + _zero_grad_anchor(module, device)

        if self.use_causal:
            ddp_anchor = ddp_anchor + _zero_grad_anchor(self.causal_module, device)
        if self.use_sparse:
            ddp_anchor = ddp_anchor + _zero_grad_anchor(self.sparse_ae, device)
        if self.use_semantic_grounding:
            ddp_anchor = ddp_anchor + _zero_grad_anchor(self.semantic_grounding, device)
        ddp_anchor = ddp_anchor + _zero_grad_anchor(self.multitaskhead, device)

        total_loss = (
            self.loss_weights['classification']          * cls_loss
            + self.loss_weights.get('uncertainty', 0.0) * uncertainty_loss
            + self.loss_weights.get('causal',      0.0) * causal_loss
            + self.loss_weights.get('sparse',      0.0) * sparse_loss
            + ddp_anchor
        )

        return {
            'overall':        total_loss,
            'classification': cls_loss,
            'uncertainty':    uncertainty_loss,
            'causal':         causal_loss,
            'sparse':         sparse_loss,
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
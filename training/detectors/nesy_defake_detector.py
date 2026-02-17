"""
NeSyDeFake: Neural-Symbolic Deepfake Detection with Causal Discovery
Updated Detector - Uses ForensicAdapter as frozen spatial feature extractor
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
    SpatialFeatureExtractor
)
from networks.nesy_defake.fusion import MultiModalFusion
from networks.nesy_defake.causal import CausalDiscoveryModule
from networks.nesy_defake.classifiers import MultiTaskHead, SparseAutoencoder

from utils.semantic_grounding import SemanticGroundingModule

logger = logging.getLogger(__name__)
ALL_BRANCHES = ('temporal', 'spatial', 'frequency')


def _zero_grad_anchor(module: nn.Module, device: torch.device) -> torch.Tensor:
    """
    Returns a scalar zero tensor that is connected to ALL parameters of
    `module` via a sum-of-zeros operation.

    This forces DDP to register gradients for every parameter in `module`
    even when the module is not used in a given forward pass, satisfying
    DDP's requirement that all parameters participate in every backward pass.
    The mathematical contribution to the loss is exactly 0.0.

    Args:
        module: the nn.Module whose parameters need to be anchored.
        device: the device the loss tensor should live on.

    Returns:
        A scalar tensor with value 0.0, attached to the computation graph
        through every parameter in `module`.
    """
    anchor = torch.zeros(1, device=device, dtype=torch.float32)
    for p in module.parameters():
        if p.requires_grad:
            anchor = anchor + p.sum() * 0.0
    return anchor


@DETECTOR.register_module(module_name='nesydefake_hybrid')
class NeSyDeFakeHybridDetector(AbstractDetector):
    """
    NeSyDeFake Detector with ForensicAdapter as the spatial feature extractor.

    The vanilla SpatialFeatureExtractor (plain CLIP image encoder) is replaced
    by a fully frozen ForensicAdapterDetector whose extract_spatial_features()
    method returns (N, 768) forensic-aware feature vectors.  Every other module
    (temporal, frequency, fusion, causal, sparse, multi-task head) is unchanged.

    Config YAML additions required:
        forensic_adapter:
            weights_path: /path/to/fa_ckpt_best.pth
            clip_model_name: ViT-L/14
            vit_name: <adapter_vit_name>
            num_quires: <int>
            fusion_map: <dict>
            mlp_dim: <int>
            mlp_out_dim: <int>
            head_num: <int>
            device: cuda
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Module 1: Foundation Model Feature Extractors
        self.build_backbone(config)

        # Module 3: Fusion
        self.fusion = MultiModalFusion(config)

        # Module 4: Semantic Grounding (Optional)
        self.use_semantic_grounding = config.get('semantic_grounding', {}).get('enabled', False)
        if self.use_semantic_grounding:
            self.semantic_grounding = SemanticGroundingModule(
                config,
                feature_dim=config['fusion']['projection_dim']
            )
            self.grounded_feature_dim = self.semantic_grounding.output_dim
        else:
            self.grounded_feature_dim = config['fusion']['projection_dim']

        # Module 5: Causal Discovery (Optional)
        self.use_causal = config['causal_module']['enabled']
        if self.use_causal:
            self.causal_module = CausalDiscoveryModule(config)

        # Module 6: Sparse Features (Optional)
        self.use_sparse = config['sparse_features']['enabled']
        if self.use_sparse:
            self.sparse_ae = SparseAutoencoder(config)

        # Module 7: Multi-task Classifier
        self.multitaskhead = MultiTaskHead(config)

        # Loss configuration
        self.loss_weights = config['loss_func']['weights']
        self.build_loss(config)

        # Processing configuration
        self.num_clips = config.get('num_clips_per_video', 32)
        self.clip_size = config.get('clip_size', 16)

        logger.info("NeSyDeFake Hybrid Detector initialised successfully")
        logger.info(f"  - Processing {self.num_clips} clips per video")
        logger.info(f"  - Clip size: {self.clip_size} frames")

    # ------------------------------------------------------------------ #
    #  Construction helpers                                                #
    # ------------------------------------------------------------------ #

    def build_backbone(self, config):
        self.temporal_extractor  = TemporalFeatureExtractor(config)
        self.spatial_extractor   = SpatialFeatureExtractor(config)
        self.frequency_extractor = FrequencyFeatureExtractor(config)

        # ── NEW: freeze inactive branches ─────────────────────────────────────
        # Default: all branches active (identical to original behaviour)
        active = set(config.get('active_branches', list(ALL_BRANCHES)))
        self.active_branches = active

        # branch_map = {
        #     'temporal':  self.temporal_extractor,
        #     'spatial':   self.spatial_extractor,
        #     'frequency': self.frequency_extractor,
        # }
        # for name, module in branch_map.items():
        #     if name not in active:
        #         self._freeze_module(module, name)

        # active_str   = ', '.join(sorted(active))
        # inactive_str = ', '.join(sorted(set(ALL_BRANCHES) - active)) or 'none'
        # logger.info(f"  - Active branches  : {active_str}")
        # logger.info(f"  - Frozen branches  : {inactive_str}")


    def build_loss(self, config: dict) -> None:
        self.cls_loss = nn.CrossEntropyLoss()
        self.reg_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    # ------------------------------------------------------------------ #
    #  Feature extraction                                                  #
    # ------------------------------------------------------------------ #

    def extract_temporal_features(self, video_clips: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video_clips: (B, clip_size, C, H, W)
        Returns:
            (B, D_temporal)
        """
        return self.temporal_extractor(video_clips)

    def extract_spatial_features(self, spatial_frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spatial_frames: (B, C, H, W)
        Returns:
            (B, D_spatial)
        """
        return self.spatial_extractor(spatial_frames)

    def extract_frequency_features(self, frequency_frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frequency_frames: (B, C, H, W)
        Returns:
            (B, D_frequency)
        """
        return self.frequency_extractor(frequency_frames)

    def features(self, data_dict: dict) -> tuple:
        """
        Extract and fuse multi-modal features.

        Args:
            data_dict must contain:
                'temporal_clip'   : (B, clip_size, C, H, W)
                'spatial_frame'   : (B, C, H, W)
                'frequency_frame' : (B, C, H, W)
                'label'           : (B,)
        Returns:
            fused_features : (B, D_fused)
            temporal_feat  : (B, D_temporal)
            spatial_feat   : (B, 768)
            frequency_feat : (B, D_frequency)
        """
        temporal_feat  = self.extract_temporal_features(data_dict['temporal_clip'])
        spatial_feat   = self.extract_spatial_features(data_dict['spatial_frame'])
        frequency_feat = self.extract_frequency_features(data_dict['frequency_frame'])
        fused_features = self.fusion(temporal_feat, spatial_feat, frequency_feat)
        return fused_features, temporal_feat, spatial_feat, frequency_feat

    def classifier(self, features: torch.Tensor) -> dict:
        return self.multitaskhead(features)

    # ------------------------------------------------------------------ #
    #  Forward pass                                                        #
    # ------------------------------------------------------------------ #

    def forward(self, data_dict: dict, inference: bool = False) -> dict:
        """
        Forward pass.

        Args:
            data_dict: dict containing
                'temporal_clip'   : (B, clip_size, C, H, W)
                'spatial_frame'   : (B, C, H, W)
                'frequency_frame' : (B, C, H, W)
                'label'           : (B,)
                'raw_frame'       : (B, C, H, W)  — only needed if semantic
                                    grounding is enabled
            inference: whether in inference mode
        Returns:
            pred_dict
        """
        device = data_dict['label'].device

        # Step 1: Multi-modal feature extraction
        fused_features, temporal_feat, spatial_feat, frequency_feat = self.features(data_dict)

        # Step 2: Semantic Grounding (Optional)
        semantic_concepts = None
        grounded_features = fused_features

        if self.use_semantic_grounding:
            representative_frames = data_dict['raw_frame']
            grounded_features, semantic_concepts = self.semantic_grounding(
                fused_features,
                representative_frames,
            )

        # Step 3: Causal Discovery (Optional)
        # NOTE: violation_score is ALWAYS computed as a real tensor (never
        # hardcoded to 0) so that causal_module parameters always receive
        # gradients through it, keeping DDP happy with find_unused_parameters=False.
        violation_score = None
        causal_dag = None
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

        # Step 4: Sparse Features (Optional)
        sparse_loss = None
        if self.use_sparse:
            sparse_features, sparse_loss = self.sparse_ae(grounded_features)
            classifier_input = sparse_features
        else:
            classifier_input = grounded_features

        # Step 5: Multi-task Classification
        # ALL task heads (including violation_score head) are always run so
        # that their parameters always participate in the forward pass.
        task_outputs = self.classifier(classifier_input)
        cls_logits = task_outputs['classification']
        prob = torch.softmax(cls_logits, dim=1)[:, 1]

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
            'violation_score':   task_outputs.get('violation_score'),  # real tensor, not 0
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

        # ── Classification loss (unchanged) ──────────────────────────────────
        cls_loss = self.cls_loss(pred_dict['cls'], label)

        # ── Uncertainty loss (unchanged) ──────────────────────────────────────
        uncertainty_loss = torch.zeros(1, device=device)
        if pred_dict.get('uncertainty') is not None:
            pred_label = pred_dict['cls'].argmax(dim=1)
            is_correct = (pred_label == label).float()
            uncertainty_loss = self.reg_loss(
                pred_dict['uncertainty'].squeeze(),
                1 - is_correct,
            )

        # ── Causal violation loss (unchanged) ────────────────────────────────
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

        # ── Sparse loss (unchanged) ───────────────────────────────────────────
        sparse_loss = pred_dict.get('sparse_loss')
        if sparse_loss is None or not isinstance(sparse_loss, torch.Tensor):
            sparse_loss = torch.zeros(1, device=device)

        # ── DDP anchor (expanded to cover frozen branches) ────────────────────
        # Frozen branch parameters still need to appear in the gradient graph
        # so DDP doesn't complain. _zero_grad_anchor adds exactly 0.0 to loss.
        ddp_anchor = torch.zeros(1, device=device)

        branch_map = {
            'temporal':  self.temporal_extractor,
            'spatial':   self.spatial_extractor,
            'frequency': self.frequency_extractor,
        }
        for name, module in branch_map.items():
            if name not in self.active_branches:
                ddp_anchor = ddp_anchor + _zero_grad_anchor(module, device)

        # Optional modules (same as before)
        if self.use_causal:
            ddp_anchor = ddp_anchor + _zero_grad_anchor(self.causal_module, device)
        if self.use_sparse:
            ddp_anchor = ddp_anchor + _zero_grad_anchor(self.sparse_ae, device)
        if self.use_semantic_grounding:
            ddp_anchor = ddp_anchor + _zero_grad_anchor(self.semantic_grounding, device)
        ddp_anchor = ddp_anchor + _zero_grad_anchor(self.multitaskhead, device)

        # ── Total loss ────────────────────────────────────────────────────────
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
        pred = pred_dict['cls']

        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

        if pred_dict.get('uncertainty') is not None:
            uncertainty = pred_dict['uncertainty'].detach().cpu().numpy()
            metric_batch_dict['mean_uncertainty'] = float(uncertainty.mean())

        self.video_names = []
        return metric_batch_dict
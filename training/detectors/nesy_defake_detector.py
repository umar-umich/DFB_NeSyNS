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

# ForensicAdapterDetector lives in the detectors folder alongside this file.
# Import directly rather than through DETECTOR registry to avoid name conflicts.
# from detectors.f_adapter import ForensicAdapterDetector

logger = logging.getLogger(__name__)


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
        

        # ── Module 3: Fusion ──────────────────────────────────────────────
        self.fusion = MultiModalFusion(config)

        # ── Module 4: Semantic Grounding (Optional) ───────────────────────
        self.use_semantic_grounding = config.get('semantic_grounding', {}).get('enabled', False)
        if self.use_semantic_grounding:
            self.semantic_grounding = SemanticGroundingModule(
                config,
                feature_dim=config['fusion']['projection_dim']
            )
            self.grounded_feature_dim = self.semantic_grounding.output_dim
        else:
            self.grounded_feature_dim = config['fusion']['projection_dim']

        # ── Module 5: Causal Discovery (Optional) ────────────────────────
        self.use_causal = config['causal_module']['enabled']
        if self.use_causal:
            self.causal_module = CausalDiscoveryModule(config)

        # ── Module 6: Sparse Features (Optional) ─────────────────────────
        self.use_sparse = config['sparse_features']['enabled']
        if self.use_sparse:
            self.sparse_ae = SparseAutoencoder(config)

        # ── Module 7: Multi-task Classifier ──────────────────────────────
        self.multitaskhead = MultiTaskHead(config)

        # ── Loss configuration ────────────────────────────────────────────
        self.loss_weights = config['loss_func']['weights']
        self.build_loss(config)

        # ── Processing configuration ──────────────────────────────────────
        self.num_clips = config.get('num_clips_per_video', 32)
        self.clip_size = config.get('clip_size', 16)

        logger.info("NeSyDeFake Hybrid Detector initialised successfully")
        logger.info(f"  - Spatial extractor: ForensicAdapter (frozen, 768D output)")
        logger.info(f"  - Processing {self.num_clips} clips per video")
        logger.info(f"  - Clip size: {self.clip_size} frames")

    # ------------------------------------------------------------------ #
    #  ForensicAdapter construction and freezing                           #
    # ------------------------------------------------------------------ #

    # def _build_forensic_spatial_extractor(self, config: dict) -> None:
        
    #     fa_cfg = config['foundation_models']['spatial']
    #     device = fa_cfg.get('device', config.get('device', 'cuda'))

    #     logger.info(f"Building ForensicAdapter spatial extractor ...")
    #     logger.info(f"  CLIP backbone : {fa_cfg['clip_model_name']}")
    #     logger.info(f"  num_quires    : {fa_cfg['num_quires']}")
    #     logger.info(f"  model_path  : {fa_cfg['model_path']}")

    #     # Instantiate FA with its original constructor signature
    #     self.forensic_adapter = ForensicAdapterDetector(
    #         clip_name=fa_cfg['clip_model_name'],
    #         adapter_vit_name=fa_cfg['vit_name'],
    #         num_quires=fa_cfg['num_quires'],
    #         fusion_map=fa_cfg['fusion_map'],
    #         mlp_dim=fa_cfg['mlp_dim'],
    #         mlp_out_dim=fa_cfg['mlp_out_dim'],
    #         head_num=fa_cfg['head_num'],
    #         device=device,
    #     )

    #     # Load pretrained weights
    #     # FA checkpoints are saved as raw state_dicts (see test.py in FA repo)
    #     weights_path = fa_cfg['model_path']
    #     ckpt = torch.load(weights_path, map_location=device)

    #     # Handle both raw state_dict and wrapped {'model': state_dict} formats
    #     state_dict = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
    #     missing, unexpected = self.forensic_adapter.load_state_dict(
    #         state_dict, strict=False
    #     )
    #     if missing:
    #         logger.warning(f"ForensicAdapter: missing keys ({len(missing)}): {missing[:5]} ...")
    #     if unexpected:
    #         logger.warning(f"ForensicAdapter: unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")

    #     logger.info(f"ForensicAdapter weights loaded from {weights_path}")

    #     # Freeze all parameters — FA is used as a fixed feature extractor only.
    #     # NeSyDeFake's novelty is in the causal/symbolic reasoning modules,
    #     # not in the feature extraction stage.
    #     for param in self.forensic_adapter.parameters():
    #         param.requires_grad = False
    #     self.forensic_adapter.eval()

    #     logger.info("ForensicAdapter frozen (requires_grad=False for all params)")

    # def _ensure_fa_frozen(self) -> None:
    #     """
    #     Guard: re-enforce eval mode and no-grad on FA at the start of every
    #     forward pass. This ensures FA stays frozen even if a training loop
    #     calls model.train() globally (which would otherwise flip FA's
    #     BatchNorm / Dropout layers into training mode).
    #     """
    #     self.forensic_adapter.eval()
    #     for param in self.forensic_adapter.parameters():
    #         param.requires_grad = False

    # ------------------------------------------------------------------ #
    #  Loss and metric helpers                                             #
    # ------------------------------------------------------------------ #


    def build_backbone(self, config):
        self.temporal_extractor = TemporalFeatureExtractor(config)
        self.frequency_extractor = FrequencyFeatureExtractor(config)

        # if config['foundation_models']['spatial']['name'].lower() == 'forensic_adapter':
        # # ── Module 2: ForensicAdapter as Frozen Spatial Extractor ─────────
        #     self._build_forensic_spatial_extractor(config)
        # else:
        self.spatial_extractor = SpatialFeatureExtractor(config)

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
                'spatial_frame'   : (B, C, H, W)  ← middle frame per clip
                'frequency_frame' : (B, C, H, W)
                'label'           : (B,)
        Returns:
            fused_features : (B, D_fused)
            temporal_feat  : (B, D_temporal)
            spatial_feat   : (B, 768)          ← from ForensicAdapter
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
        # ── Step 1: Multi-modal feature extraction ─────────────────────
        fused_features, temporal_feat, spatial_feat, frequency_feat = self.features(data_dict)

        # ── Step 2: Semantic Grounding (Optional) ──────────────────────
        semantic_concepts = None
        grounded_features = fused_features

        if self.use_semantic_grounding:
            representative_frames = data_dict['raw_frame']  # (B, C, H, W)
            grounded_features, semantic_concepts = self.semantic_grounding(
                fused_features,
                representative_frames,
            )

        # ── Step 3: Causal Discovery (Optional) ────────────────────────
        violation_score = None
        causal_dag = None
        causal_concepts = None

        if self.use_causal and not inference:
            violation_score, causal_dag, causal_concepts = self.causal_module(
                grounded_features,
                explicit_concepts=semantic_concepts,
                return_graph=True,
            )
        elif self.use_causal:
            violation_score = self.causal_module(
                grounded_features,
                explicit_concepts=semantic_concepts,
            )

        # ── Step 4: Sparse Features (Optional) ─────────────────────────
        sparse_loss = None
        if self.use_sparse:
            sparse_features, sparse_loss = self.sparse_ae(grounded_features)
            classifier_input = sparse_features
        else:
            classifier_input = grounded_features

        # ── Step 5: Multi-task Classification ──────────────────────────
        task_outputs = self.classifier(classifier_input)
        cls_logits = task_outputs['classification']
        prob = torch.softmax(cls_logits, dim=1)[:, 1]

        pred_dict = {
            'cls': cls_logits,
            'prob': prob,
            'feat': fused_features,
            'grounded_feat': grounded_features,
            'temporal_feat': temporal_feat,
            'spatial_feat': spatial_feat,      # (B, 768) from ForensicAdapter
            'frequency_feat': frequency_feat,
            'semantic_concepts': semantic_concepts,
            'causal_concepts': causal_concepts,
            'uncertainty': task_outputs.get('uncertainty'),
            'violation_score': violation_score,
            'task_outputs': task_outputs,
            'sparse_loss': sparse_loss,
            'causal_dag': causal_dag,
        }

        return pred_dict

    # ------------------------------------------------------------------ #
    #  Loss computation                                                    #
    # ------------------------------------------------------------------ #

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        device = label.device

        # Classification loss
        cls_loss = self.cls_loss(pred_dict['cls'], label)

        # Uncertainty loss
        uncertainty_loss = torch.tensor(0.0, device=device)
        if pred_dict.get('uncertainty') is not None:
            pred_label = pred_dict['cls'].argmax(dim=1)
            is_correct = (pred_label == label).float()
            target_uncertainty = 1 - is_correct
            uncertainty_loss = self.reg_loss(
                pred_dict['uncertainty'].squeeze(),
                target_uncertainty,
            )

        # Causal violation loss
        causal_loss = torch.tensor(0.0, device=device)
        if self.use_causal and pred_dict.get('violation_score') is not None:
            target_violation = label.float()
            causal_loss = self.reg_loss(pred_dict['violation_score'], target_violation)
            if pred_dict.get('causal_dag') is not None:
                dag_penalty = self.causal_module.causal_learner.compute_dag_penalty()
                causal_loss += (
                    self.config['causal_module']['dag_learning']['dag_penalty_weight']
                    * dag_penalty
                )

        # Sparse loss
        sparse_loss = pred_dict.get('sparse_loss')
        if sparse_loss is None or not isinstance(sparse_loss, torch.Tensor):
            sparse_loss = torch.tensor(0.0, device=device)

        total_loss = (
            self.loss_weights['classification'] * cls_loss
            + self.loss_weights.get('uncertainty', 0) * uncertainty_loss
            + self.loss_weights.get('causal', 0) * causal_loss
            + self.loss_weights.get('sparse', 0) * sparse_loss
        )

        return {
            'overall': total_loss,
            'classification': cls_loss,
            'uncertainty': uncertainty_loss,
            'causal': causal_loss,
            'sparse': sparse_loss,
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
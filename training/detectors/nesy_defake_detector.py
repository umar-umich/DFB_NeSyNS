"""
NeSyDeFake: Neural-Symbolic Deepfake Detection with Causal Discovery
Main Detector - Orchestrates all modules
"""

import logging
import torch
import torch.nn as nn

from metrics.base_metrics_class import calculate_metrics_for_train
from .base_detector import AbstractDetector
from detectors import DETECTOR

# CORRECTED IMPORTS - Use nesydefake sub-package
from networks.nesy_defake.foundation_models import (
    TemporalFeatureExtractor,
    SpatialFeatureExtractor, 
    FrequencyFeatureExtractor
)
from networks.nesy_defake.fusion import MultiModalFusion
from networks.nesy_defake.causal import CausalDiscoveryModule
from networks.nesy_defake.classifiers import MultiTaskHead, SparseAutoencoder

# Semantic grounding is in utils (already correct)
from utils.semantic_grounding import SemanticGroundingModule

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='nesydefake_hybrid')
class NeSyDeFakeHybridDetector(AbstractDetector):
    """NeSyDeFake Detector - properly integrated with DeepfakeBench"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Module 1: Foundation Model Feature Extractors
        self.build_backbone(config)
        
        # Fusion Module
        self.fusion = MultiModalFusion(config)
        
        # Module 2: Semantic Grounding (Optional)
        self.use_semantic_grounding = config.get('semantic_grounding', {}).get('enabled', False)
        if self.use_semantic_grounding:
            self.semantic_grounding = SemanticGroundingModule(
                config, 
                feature_dim=config['fusion']['projection_dim']
            )
            self.grounded_feature_dim = self.semantic_grounding.output_dim
        else:
            self.grounded_feature_dim = config['fusion']['projection_dim']
        
        # Module 3: Causal Discovery (Optional)
        self.use_causal = config['causal_module']['enabled']
        if self.use_causal:
            self.causal_module = CausalDiscoveryModule(config)
        
        # Module 4: Sparse Features (Optional)
        self.use_sparse = config['sparse_features']['enabled']
        if self.use_sparse:
            self.sparse_ae = SparseAutoencoder(config)
        
        # Module 5: Multi-task Classifier
        self.multitaskhead = MultiTaskHead(config)
        
        # Loss configuration
        self.loss_weights = config['loss_func']['weights']
        self.build_loss(config)
        
        logger.info("NeSyDeFake Hybrid Detector initialized successfully")
    
    def build_backbone(self, config):
        """Initialize foundation model feature extractors"""
        self.temporal_extractor = TemporalFeatureExtractor(config)
        self.spatial_extractor = SpatialFeatureExtractor(config)
        self.frequency_extractor = FrequencyFeatureExtractor(config)
    
    def build_loss(self, config):
        """Build loss functions for each task"""
        self.cls_loss = nn.CrossEntropyLoss()
        self.reg_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    
    def extract_temporal_features(self, video_clips):
        """
        Extract temporal features using VideoMAE
        
        Args:
            video_clips: (B, T, C, H, W) - video clips
        Returns:
            temporal_features: (B, D)
        """
        return self.temporal_extractor(video_clips)
    
    def extract_spatial_features(self, frames):
        """
        Extract spatial features using CLIP
        
        Args:
            frames: (B, num_frames, C, H, W)
        Returns:
            spatial_features: (B, D)
        """
        B, num_frames = frames.shape[:2]
        
        # Process each frame
        frame_features = []
        for i in range(num_frames):
            frame = frames[:, i]
            feat = self.spatial_extractor(frame)
            frame_features.append(feat)
        
        # Average across frames
        spatial_features = torch.stack(frame_features, dim=1).mean(dim=1)
        return spatial_features
    
    def extract_frequency_features(self, frames):
        """
        Extract frequency features using SRM+ResNet
        
        Args:
            frames: (B, num_frames, C, H, W)
        Returns:
            frequency_features: (B, D)
        """
        B, num_frames = frames.shape[:2]
        
        # Process each frame
        frame_features = []
        for i in range(num_frames):
            frame = frames[:, i]
            feat = self.frequency_extractor(frame)
            frame_features.append(feat)
        
        # Average across frames
        frequency_features = torch.stack(frame_features, dim=1).mean(dim=1)
        return frequency_features
    
    def features(self, data_dict: dict) -> tuple:
        """
        Extract and fuse multi-modal features
        
        Args:
            data_dict: Dictionary containing 'image' (B, T, C, H, W)
        Returns:
            fused_features: (B, D)
            temporal_feat: (B, D1)
            spatial_feat: (B, D2)
            frequency_feat: (B, D3)
        """
        # Extract features from each modality
        temporal_feat = self.extract_temporal_features(data_dict['image'])
        spatial_feat = self.extract_spatial_features(data_dict['image'])
        frequency_feat = self.extract_frequency_features(data_dict['image'])
        
        # Fuse multi-modal features
        fused_features = self.fusion(temporal_feat, spatial_feat, frequency_feat)
        
        return fused_features, temporal_feat, spatial_feat, frequency_feat
    
    def classifier(self, features: torch.tensor) -> dict:
        """
        Classify features using multi-task head
        
        Args:
            features: (B, D)
        Returns:
            task_outputs: dict of predictions
        """
        return self.multitaskhead(features)
    
    def forward(self, data_dict: dict, inference=False) -> dict:
        """
        Forward pass
        
        Args:
            data_dict: Dictionary containing:
                - 'image': (B, T, C, H, W) - video clips
                - 'label': (B,) - ground truth labels
            inference: whether in inference mode
        Returns:
            pred_dict: Dictionary containing all predictions and features
        """
        # Step 1: Extract and fuse multi-modal features
        fused_features, temporal_feat, spatial_feat, frequency_feat = self.features(data_dict)
        
        # Step 2: Semantic Grounding (Optional)
        semantic_concepts = None
        grounded_features = fused_features
        
        if self.use_semantic_grounding:
            grounded_features, semantic_concepts = self.semantic_grounding(
                fused_features,
                data_dict['image']
            )
        
        # Step 3: Causal Discovery (Optional)
        violation_score = None
        causal_dag = None
        causal_concepts = None
        
        if self.use_causal and not inference:
            violation_score, causal_dag, causal_concepts = self.causal_module(
                grounded_features, 
                explicit_concepts=semantic_concepts,
                return_graph=True
            )
        elif self.use_causal:
            violation_score = self.causal_module(
                grounded_features,
                explicit_concepts=semantic_concepts
            )
        
        # Step 4: Sparse Features (Optional)
        sparse_loss = None
        if self.use_sparse:
            sparse_features, sparse_loss = self.sparse_ae(grounded_features)
            classifier_input = sparse_features
        else:
            classifier_input = grounded_features
        
        # Step 5: Multi-task Classification
        task_outputs = self.classifier(classifier_input)
        
        # Get probabilities
        cls_logits = task_outputs['classification']
        prob = torch.softmax(cls_logits, dim=1)[:, 1]
        
        # Build prediction dictionary
        pred_dict = {
            'cls': cls_logits,
            'prob': prob,
            'feat': fused_features,
            'grounded_feat': grounded_features,
            'temporal_feat': temporal_feat,
            'spatial_feat': spatial_feat,
            'frequency_feat': frequency_feat,
            'semantic_concepts': semantic_concepts,
            'causal_concepts': causal_concepts,
            'uncertainty': task_outputs.get('uncertainty'),
            'violation_score': violation_score,
            'task_outputs': task_outputs,
            'sparse_loss': sparse_loss,
            'causal_dag': causal_dag
        }
        
        return pred_dict
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        """
        Compute all losses
        
        Args:
            data_dict: Ground truth data
            pred_dict: Model predictions
        Returns:
            loss_dict: Dictionary of all losses
        """
        label = data_dict['label']
        
        # Classification loss
        cls_loss = self.cls_loss(pred_dict['cls'], label)
        
        # Uncertainty loss (if enabled)
        uncertainty_loss = 0
        if pred_dict.get('uncertainty') is not None:
            pred_label = pred_dict['cls'].argmax(dim=1)
            is_correct = (pred_label == label).float()
            target_uncertainty = 1 - is_correct
            uncertainty_loss = self.reg_loss(
                pred_dict['uncertainty'].squeeze(), 
                target_uncertainty
            )
        
        # Causal violation loss
        causal_loss = 0
        if self.use_causal and pred_dict.get('violation_score') is not None:
            target_violation = label.float()
            causal_loss = self.reg_loss(pred_dict['violation_score'], target_violation)
            
            # Add DAG acyclicity penalty
            if pred_dict.get('causal_dag') is not None:
                dag_penalty = self.causal_module.causal_learner.compute_dag_penalty()
                causal_loss += self.config['causal_module']['dag_learning']['dag_penalty_weight'] * dag_penalty
        
        # Sparse loss
        sparse_loss = pred_dict.get('sparse_loss', 0)
        if sparse_loss == 0:
            sparse_loss = torch.tensor(0.0, device=cls_loss.device)
        
        # Total loss
        total_loss = (
            self.loss_weights['classification'] * cls_loss +
            self.loss_weights.get('uncertainty', 0) * uncertainty_loss +
            self.loss_weights.get('causal', 0) * causal_loss +
            self.loss_weights.get('sparse', 0) * sparse_loss
        )
        
        loss_dict = {
            'overall': total_loss,
            'classification': cls_loss,
            'uncertainty': uncertainty_loss if isinstance(uncertainty_loss, torch.Tensor) else torch.tensor(uncertainty_loss),
            'causal': causal_loss if isinstance(causal_loss, torch.Tensor) else torch.tensor(causal_loss),
            'sparse': sparse_loss
        }
        
        return loss_dict
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        """
        Compute training metrics
        
        Args:
            data_dict: Ground truth data
            pred_dict: Model predictions
        Returns:
            metric_batch_dict: Dictionary of metrics
        """
        label = data_dict['label']
        pred = pred_dict['cls']
        
        # Standard metrics
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        
        # Add uncertainty calibration if available
        if pred_dict.get('uncertainty') is not None:
            uncertainty = pred_dict['uncertainty'].detach().cpu().numpy()
            metric_batch_dict['mean_uncertainty'] = float(uncertainty.mean())
        
        self.video_names = []
        return metric_batch_dict

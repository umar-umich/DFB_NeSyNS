"""
NeSyDeFake: Neural-Symbolic Deepfake Detection with Causal Discovery
Multi-stream hybrid architecture combining:
- VideoMAE (temporal)
- CLIP (spatial) 
- SRM+ResNet (frequency)
- Causal Discovery Module
- Sparse Monosemantic Features (optional)
"""

import os
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics.base_metrics_class import calculate_metrics_for_train
from .base_detector import AbstractDetector
from detectors import DETECTOR
from loss import LOSSFUNC

# Import semantic grounding module
from utils.semantic_grounding import SemanticGroundingModule, DeepFaceSemanticExtractor

logger = logging.getLogger(__name__)


# ============================================================================
# MODULE 1: FOUNDATION MODEL FEATURE EXTRACTORS
# ============================================================================

class TemporalFeatureExtractor(nn.Module):
    """VideoMAE-based temporal feature extractor"""
    def __init__(self, config):
        super().__init__()
        from transformers import VideoMAEModel
        
        model_path = config['foundation_models']['temporal']['model_path']
        self.backbone = VideoMAEModel.from_pretrained(model_path)
        self.output_dim = config['foundation_models']['temporal']['output_dim']
        
        # Normalization layer
        self.fc_norm = nn.LayerNorm(self.output_dim)
        
        if config['foundation_models']['temporal']['freeze_backbone']:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) - batch of video clips
        Returns:
            features: (B, D) - temporal features
        """
        outputs = self.backbone(x, output_hidden_states=True)
        sequence_output = outputs[0]  # (B, num_patches, hidden_dim)
        features = self.fc_norm(sequence_output.mean(1))  # Global average pooling
        return features


class SpatialFeatureExtractor(nn.Module):
    """CLIP-based spatial feature extractor"""
    def __init__(self, config):
        super().__init__()
        from transformers import CLIPVisionModel
        
        model_path = config['foundation_models']['spatial']['model_path']
        self.backbone = CLIPVisionModel.from_pretrained(model_path)
        self.output_dim = config['foundation_models']['spatial']['output_dim']
        
        if config['foundation_models']['spatial']['freeze_backbone']:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) - batch of images
        Returns:
            features: (B, D) - spatial features
        """
        outputs = self.backbone(pixel_values=x, output_hidden_states=True)
        # Use pooled output (CLS token)
        features = outputs.pooler_output
        return features


class FrequencyFeatureExtractor(nn.Module):
    """SRM + ResNet for frequency domain analysis"""
    def __init__(self, config):
        super().__init__()
        
        # SRM filters for artifact detection
        self.srm_conv = self._build_srm_layer(
            config['foundation_models']['frequency']['srm_filters']
        )
        
        # ResNet backbone
        self.resnet = self._build_resnet(
            config['foundation_models']['frequency']['resnet_depth']
        )
        
        self.output_dim = config['foundation_models']['frequency']['output_dim']
        
        if config['foundation_models']['frequency']['freeze_srm']:
            for param in self.srm_conv.parameters():
                param.requires_grad = False
    
    def _build_srm_layer(self, num_filters):
        """Build SRM filter bank"""
        # Implement standard SRM filters for steganalysis
        # These are fixed filters designed to detect manipulation artifacts
        srm_weights = self._get_srm_filters(num_filters)
        
        conv = nn.Conv2d(3, num_filters, kernel_size=5, padding=2, bias=False)
        conv.weight.data = torch.from_numpy(srm_weights).float()
        
        return conv
    
    def _get_srm_filters(self, num_filters):
        """Get SRM filter bank (simplified version)"""
        # In practice, use the full 30 SRM filters from steganalysis literature
        # Here's a simplified version with basic edge/noise detection filters
        filters = []
        
        # Basic high-pass filters for edge detection
        # Filter 1: Horizontal edge
        f1 = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [-1, -1, 4, -1, -1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        
        # Filter 2: Vertical edge
        f2 = np.array([
            [0, 0, -1, 0, 0],
            [0, 0, -1, 0, 0],
            [0, 0, 4, 0, 0],
            [0, 0, -1, 0, 0],
            [0, 0, -1, 0, 0]
        ])
        
        # Filter 3: Square edge
        f3 = np.array([
            [0, 0, 0, 0, 0],
            [0, -1, -1, -1, 0],
            [0, -1, 8, -1, 0],
            [0, -1, -1, -1, 0],
            [0, 0, 0, 0, 0]
        ])
        
        # Replicate basic filters across RGB channels and to reach num_filters
        base_filters = [f1, f2, f3]
        for i in range(num_filters):
            f = base_filters[i % len(base_filters)]
            # Create 3-channel version (one per RGB)
            f_rgb = np.stack([f, f, f], axis=0)  # (3, 5, 5)
            filters.append(f_rgb)
        
        return np.array(filters)  # (num_filters, 3, 5, 5)
    
    def _build_resnet(self, depth):
        """Build ResNet backbone"""
        import torchvision.models as models
        
        if depth == 18:
            resnet = models.resnet18(pretrained=True)
        elif depth == 34:
            resnet = models.resnet34(pretrained=True)
        else:
            resnet = models.resnet50(pretrained=True)
        
        # Modify first conv to accept SRM filter outputs
        original_conv1 = resnet.conv1
        resnet.conv1 = nn.Conv2d(
            self.srm_conv.out_channels, 
            64, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        
        # Remove final FC layer
        resnet.fc = nn.Identity()
        
        return resnet
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) - batch of images
        Returns:
            features: (B, D) - frequency features
        """
        # Apply SRM filters
        srm_out = self.srm_conv(x)
        
        # Process through ResNet
        features = self.resnet(srm_out)
        
        return features


# ============================================================================
# FUSION MODULE
# ============================================================================

class MultiModalFusion(nn.Module):
    """Fuse features from multiple streams"""
    def __init__(self, config):
        super().__init__()
        
        self.fusion_type = config['fusion']['type']
        temporal_dim = config['foundation_models']['temporal']['output_dim']
        spatial_dim = config['foundation_models']['spatial']['output_dim']
        frequency_dim = config['foundation_models']['frequency']['output_dim']
        
        self.fused_dim = config['fusion']['fused_dim']
        self.projection_dim = config['fusion']['projection_dim']
        
        if self.fusion_type == 'concat':
            self.fusion = nn.Sequential(
                nn.Linear(temporal_dim + spatial_dim + frequency_dim, self.fused_dim),
                nn.LayerNorm(self.fused_dim),
                nn.ReLU(),
                nn.Dropout(config['fusion']['dropout']),
                nn.Linear(self.fused_dim, self.projection_dim),
                nn.LayerNorm(self.projection_dim)
            )
        elif self.fusion_type == 'attention':
            self.fusion = CrossModalAttention(
                temporal_dim, spatial_dim, frequency_dim, self.projection_dim
            )
        else:
            raise NotImplementedError(f"Fusion type {self.fusion_type} not implemented")
    
    def forward(self, temporal_feat, spatial_feat, frequency_feat):
        """
        Args:
            temporal_feat: (B, D1)
            spatial_feat: (B, D2)
            frequency_feat: (B, D3)
        Returns:
            fused_features: (B, projection_dim)
        """
        if self.fusion_type == 'concat':
            combined = torch.cat([temporal_feat, spatial_feat, frequency_feat], dim=1)
            fused = self.fusion(combined)
        else:
            fused = self.fusion(temporal_feat, spatial_feat, frequency_feat)
        
        return fused


class CrossModalAttention(nn.Module):
    """Cross-modal attention fusion"""
    def __init__(self, temp_dim, spat_dim, freq_dim, output_dim):
        super().__init__()
        
        # Project all to same dimension
        hidden_dim = 512
        self.temp_proj = nn.Linear(temp_dim, hidden_dim)
        self.spat_proj = nn.Linear(spat_dim, hidden_dim)
        self.freq_proj = nn.Linear(freq_dim, hidden_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, temporal_feat, spatial_feat, frequency_feat):
        # Project to common space
        temp = self.temp_proj(temporal_feat).unsqueeze(1)  # (B, 1, H)
        spat = self.spat_proj(spatial_feat).unsqueeze(1)
        freq = self.freq_proj(frequency_feat).unsqueeze(1)
        
        # Stack as sequence
        sequence = torch.cat([temp, spat, freq], dim=1)  # (B, 3, H)
        
        # Self-attention across modalities
        attended, _ = self.attention(sequence, sequence, sequence)
        
        # Pool and project
        pooled = attended.mean(dim=1)  # (B, H)
        output = self.output_proj(pooled)
        
        return output


# ============================================================================
# MODULE 3: CAUSAL DISCOVERY MODULE
# ============================================================================

class CausalDiscoveryModule(nn.Module):
    """
    Module 3: Learn causal structure from features
    
    Can work with:
    1. Implicit concepts (learned from latent variables)
    2. Explicit concepts (from DeepFace semantic grounding)
    3. Combined (both implicit and explicit)
    """
    def __init__(self, config):
        super().__init__()
        
        self.config = config['causal_module']
        
        # Check if semantic grounding is enabled
        self.use_explicit_concepts = config.get('semantic_grounding', {}).get('enabled', False)
        
        # Feature to latent variable mapping
        if self.use_explicit_concepts:
            # Use smaller latent dimensions since we have explicit concepts
            input_dim = config.get('semantic_grounding', {}).get('fuse_with_features', True)
            if input_dim:
                # Using fused features (implicit + explicit)
                input_dim = 256  # Output from SemanticGroundingModule
            else:
                # Using only explicit concepts
                input_dim = self._calculate_semantic_dim(config)
        else:
            # Using only implicit features
            input_dim = config['fusion']['projection_dim']
        
        self.latent_encoder = LatentVariableEncoder(
            input_dim=input_dim,
            z_spatial_dim=self.config['latent_variables']['z_spatial_dim'],
            z_temporal_dim=self.config['latent_variables']['z_temporal_dim'],
            z_frequency_dim=self.config['latent_variables']['z_frequency_dim']
        )
        
        # Calculate total concept dimension
        # Implicit latent concepts
        total_latent_dim = self.config['latent_variables']['total_latent_dim']
        
        # Explicit semantic concepts (if enabled)
        explicit_concept_dim = 0
        if self.use_explicit_concepts:
            explicit_concept_dim = self._calculate_semantic_dim(config)
        
        # Total concepts = latent + explicit
        total_concept_dim = total_latent_dim + explicit_concept_dim
        
        # Semantic concept extractor (for implicit concepts)
        num_implicit_concepts = self.config['semantic_concepts']['concept_dim']
        self.concept_extractor = SemanticConceptExtractor(
            latent_dim=total_latent_dim,
            num_concepts=num_implicit_concepts
        )
        
        # Total concepts for causal learning
        self.num_total_concepts = num_implicit_concepts + explicit_concept_dim
        
        # Causal structure learner
        algorithm = self.config['discovery']['algorithm']
        if algorithm == 'structural_causal_circuits':
            self.causal_learner = StructuralCausalCircuits(
                num_variables=self.num_total_concepts,
                hidden_dim=self.config['dag_learning']['hidden_dim'],
                num_layers=self.config['dag_learning']['num_layers']
            )
        else:
            raise NotImplementedError(f"Algorithm {algorithm} not implemented")
        
        # Causal reasoning module
        self.causal_reasoner = CausalReasoner(
            num_concepts=self.num_total_concepts
        )
        
        logger.info(
            f"Causal Discovery Module initialized: "
            f"implicit_concepts={num_implicit_concepts}, "
            f"explicit_concepts={explicit_concept_dim}, "
            f"total_concepts={self.num_total_concepts}"
        )
    
    def _calculate_semantic_dim(self, config):
        """Calculate dimension of explicit semantic concepts"""
        sg_config = config.get('semantic_grounding', {})
        dim = 0
        if sg_config.get('extract_age', False):
            dim += 1
        if sg_config.get('extract_gender', False):
            dim += 2
        if sg_config.get('extract_emotion', False):
            dim += 7
        if sg_config.get('extract_race', False):
            dim += 6
        return dim
    
    def forward(self, features, explicit_concepts=None, return_graph=False):
        """
        Args:
            features: (B, D) - grounded features (implicit + explicit combined)
            explicit_concepts: (B, E) - explicit semantic concepts from DeepFace (optional)
            return_graph: whether to return the learned DAG
        Returns:
            violation_score: (B,) - causal violation scores
            dag: adjacency matrix (optional)
            all_concepts: (B, total_concepts) - all concepts (implicit + explicit)
        """
        # Extract latent variables from features
        z_spatial, z_temporal, z_frequency = self.latent_encoder(features)
        z_all = torch.cat([z_spatial, z_temporal, z_frequency], dim=1)
        
        # Extract implicit semantic concepts
        implicit_concepts = self.concept_extractor(z_all)  # (B, num_implicit_concepts)
        
        # Combine with explicit concepts if available
        if explicit_concepts is not None and self.use_explicit_concepts:
            # Concatenate implicit and explicit concepts
            all_concepts = torch.cat([implicit_concepts, explicit_concepts], dim=1)
        else:
            all_concepts = implicit_concepts
        
        # Learn causal structure
        dag = self.causal_learner(all_concepts)  # (num_concepts, num_concepts)
        
        # Compute causal violation scores
        violation_scores = self.causal_reasoner(all_concepts, dag)
        
        if return_graph:
            return violation_scores, dag, all_concepts
        return violation_scores
    
    def get_concept_names(self, config) -> List[str]:
        """Get names of all concepts (implicit + explicit)"""
        # Implicit concept names
        implicit_names = config['causal_module']['semantic_concepts']['concepts']
        
        # Explicit concept names (if enabled)
        explicit_names = []
        if self.use_explicit_concepts:
            sg_config = config.get('semantic_grounding', {})
            if sg_config.get('extract_age'):
                explicit_names.append('age')
            if sg_config.get('extract_gender'):
                explicit_names.extend(['gender_man', 'gender_woman'])
            if sg_config.get('extract_emotion'):
                emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
                explicit_names.extend([f'emotion_{e}' for e in emotions])
            if sg_config.get('extract_race'):
                races = ['asian', 'indian', 'black', 'white', 'middle_eastern', 'latino_hispanic']
                explicit_names.extend([f'race_{r}' for r in races])
        
        return implicit_names + explicit_names


class LatentVariableEncoder(nn.Module):
    """Encode features into latent variables"""
    def __init__(self, input_dim, z_spatial_dim, z_temporal_dim, z_frequency_dim):
        super().__init__()
        
        self.z_spatial_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, z_spatial_dim)
        )
        
        self.z_temporal_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, z_temporal_dim)
        )
        
        self.z_frequency_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, z_frequency_dim)
        )
    
    def forward(self, x):
        z_spatial = self.z_spatial_encoder(x)
        z_temporal = self.z_temporal_encoder(x)
        z_frequency = self.z_frequency_encoder(x)
        return z_spatial, z_temporal, z_frequency


class SemanticConceptExtractor(nn.Module):
    """Extract interpretable semantic concepts"""
    def __init__(self, latent_dim, num_concepts):
        super().__init__()
        
        self.concept_net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_concepts),
            nn.Sigmoid()  # Concepts as probabilities
        )
    
    def forward(self, z):
        return self.concept_net(z)


class StructuralCausalCircuits(nn.Module):
    """Learn DAG structure using differentiable causal discovery"""
    def __init__(self, num_variables, hidden_dim, num_layers):
        super().__init__()
        
        self.num_variables = num_variables
        
        # Learnable adjacency matrix (with acyclicity constraint)
        self.adjacency_logits = nn.Parameter(
            torch.randn(num_variables, num_variables) * 0.1
        )
        
        # MLP for each variable conditioned on parents
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_variables, hidden_dim),
                nn.ReLU(),
                *[layer for _ in range(num_layers - 1) 
                  for layer in (nn.Linear(hidden_dim, hidden_dim), nn.ReLU())],
                nn.Linear(hidden_dim, 1)
            )
            for _ in range(num_variables)
        ])
    
    def forward(self, concepts):
        """
        Args:
            concepts: (B, num_variables)
        Returns:
            dag: (num_variables, num_variables) adjacency matrix
        """
        # Get adjacency matrix with Gumbel-Softmax for differentiability
        dag = torch.sigmoid(self.adjacency_logits)
        
        # Enforce acyclicity (remove self-loops)
        dag = dag * (1 - torch.eye(self.num_variables, device=dag.device))
        
        return dag
    
    def compute_dag_penalty(self):
        """Compute acyclicity penalty h(W) = tr(e^W) - d"""
        dag = torch.sigmoid(self.adjacency_logits)
        dag = dag * (1 - torch.eye(self.num_variables, device=dag.device))
        
        # Matrix exponential trace penalty
        expm = torch.matrix_exp(dag)
        penalty = torch.trace(expm) - self.num_variables
        
        return penalty


class CausalReasoner(nn.Module):
    """Compute causal violation scores using interventions"""
    def __init__(self, num_concepts):
        super().__init__()
        self.num_concepts = num_concepts
    
    def forward(self, concepts, dag):
        """
        Args:
            concepts: (B, num_concepts)
            dag: (num_concepts, num_concepts)
        Returns:
            violation_scores: (B,)
        """
        batch_size = concepts.size(0)
        
        # Compute expected values based on causal parents
        violation_scores = []
        
        for i in range(self.num_concepts):
            # Get parents of concept i
            parents = dag[:, i]  # (num_concepts,)
            
            # Expected value based on parents
            expected = torch.matmul(concepts, parents.unsqueeze(1))  # (B, 1)
            
            # Violation = |observed - expected|
            violation = torch.abs(concepts[:, i:i+1] - expected)
            violation_scores.append(violation)
        
        # Average violation across all concepts
        total_violation = torch.cat(violation_scores, dim=1).mean(dim=1)
        
        return total_violation


# ============================================================================
# MODULE 4: SPARSE MONOSEMANTIC FEATURES (Optional)
# ============================================================================

class SparseAutoencoder(nn.Module):
    """Sparse autoencoder for discovering interpretable features"""
    def __init__(self, config):
        super().__init__()
        
        input_dim = config['sparse_features']['sparse_autoencoder']['input_dim']
        expansion_factor = config['sparse_features']['sparse_autoencoder']['expansion_factor']
        hidden_dim = input_dim * expansion_factor
        
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        
        self.l1_coef = config['sparse_features']['sparse_autoencoder']['l1_coefficient']
    
    def forward(self, x):
        # Encode with ReLU activation for sparsity
        hidden = F.relu(self.encoder(x))
        
        # Decode
        reconstructed = self.decoder(hidden)
        
        # Compute sparsity loss
        sparsity_loss = self.l1_coef * torch.abs(hidden).sum(dim=1).mean()
        
        # Reconstruction loss
        reconstruction_loss = F.mse_loss(reconstructed, x)
        
        return hidden, reconstruction_loss + sparsity_loss


# ============================================================================
# MODULE 5: MULTI-TASK CLASSIFIER
# ============================================================================

class MultiTaskHead(nn.Module):
    """Multi-task classifier for final predictions"""
    def __init__(self, config):
        super().__init__()
        
        input_dim = config['classifier']['input_dim']
        hidden_dims = config['classifier']['hidden_dims']
        dropout = config['classifier']['dropout']
        
        # Shared backbone
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.shared_backbone = nn.Sequential(*layers)
        
        # Task-specific heads
        self.tasks = config['classifier']['tasks']
        self.task_heads = nn.ModuleDict()
        
        for task in self.tasks:
            if task['type'] == 'binary':
                self.task_heads[task['name']] = nn.Linear(prev_dim, task['output_dim'])
            elif task['type'] == 'regression':
                self.task_heads[task['name']] = nn.Linear(prev_dim, task['output_dim'])
    
    def forward(self, x):
        shared_features = self.shared_backbone(x)
        
        outputs = {}
        for task in self.tasks:
            outputs[task['name']] = self.task_heads[task['name']](shared_features)
        
        return outputs


# ============================================================================
# MAIN HYBRID DETECTOR
# ============================================================================

@DETECTOR.register_module(module_name='nesydefake_hybrid')
class NeSyDeFakeHybridDetector(AbstractDetector):
    """
    NeSyDeFake: Neural-Symbolic Deepfake Detection
    Hybrid multi-stream architecture with causal discovery
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Module 1: Foundation Models
        self.build_backbone(config)
        # Fusion module
        self.fusion = MultiModalFusion(config)
        
        # Module 2: Semantic Grounding (Expert Grounding - Explicit)
        self.use_semantic_grounding = config.get('semantic_grounding', {}).get('enabled', False)
        if self.use_semantic_grounding:
            self.semantic_grounding = SemanticGroundingModule(
                config, 
                feature_dim=config['fusion']['projection_dim']
            )
            # Update feature dimension for subsequent modules
            self.grounded_feature_dim = self.semantic_grounding.output_dim
        else:
            self.grounded_feature_dim = config['fusion']['projection_dim']
        
        # Module 3: Causal Discovery (if enabled)
        self.use_causal = config['causal_module']['enabled']
        if self.use_causal:
            self.causal_module = CausalDiscoveryModule(config)
        
        # Module 4: Sparse Features (if enabled)
        self.use_sparse = config['sparse_features']['enabled']
        if self.use_sparse:
            self.sparse_ae = SparseAutoencoder(config)
        
        # Module 5: Multi-task Classifier
        self.multitasthead = MultiTaskHead(config)
        
        # Loss functions
        self.loss_weights = config['loss_func']['weights']
        self.build_loss(config)

    def build_backbone(self, config):
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
        Args:
            video_clips: (B, T, C, H, W) - single clip per batch
        Returns:
            temporal_features: (B, D)
        """
        # video_clips is already (B, T, C, H, W) - pass directly to VideoMAE
        feat = self.temporal_extractor(video_clips)  # (B, D)
        return feat

    # def extract_temporal_features(self, video_clips):
    #     """
    #     Args:
    #         video_clips: (B, num_clips, T, C, H, W)
    #     Returns:
    #         temporal_features: (B, D)
    #     """
    #     B, num_clips = video_clips.shape[:2]
        
    #     # Process each clip
    #     clip_features = []
    #     for i in range(num_clips):
    #         clip = video_clips[:, i]  # (B, T, C, H, W)
    #         feat = self.temporal_extractor(clip)  # (B, D)
    #         clip_features.append(feat)
        
    #     # Average across clips
    #     temporal_features = torch.stack(clip_features, dim=1).mean(dim=1)
        
    #     return temporal_features
    
    def extract_spatial_features(self, frames):
        """
        Args:
            frames: (B, num_frames, C, H, W)
        Returns:
            spatial_features: (B, D)
        """
        B, num_frames = frames.shape[:2]
        
        # Process each frame
        frame_features = []
        for i in range(num_frames):
            frame = frames[:, i]  # (B, C, H, W)
            feat = self.spatial_extractor(frame)  # (B, D)
            frame_features.append(feat)
        
        # Average across frames
        spatial_features = torch.stack(frame_features, dim=1).mean(dim=1)
        
        return spatial_features
    
    def extract_frequency_features(self, frames):
        """
        Args:
            frames: (B, num_frames, C, H, W)
        Returns:
            frequency_features: (B, D)
        """
        B, num_frames = frames.shape[:2]
        
        # Process each frame
        frame_features = []
        for i in range(num_frames):
            frame = frames[:, i]  # (B, C, H, W)
            feat = self.frequency_extractor(frame)  # (B, D)
            frame_features.append(feat)
        
        # Average across frames
        frequency_features = torch.stack(frame_features, dim=1).mean(dim=1)
        
        return frequency_features

    def features(self, data_dict: dict) -> torch.tensor:
        # b, t, c, h, w = data_dict['image'].shape
        # frame_input = data_dict['image'].reshape(-1, c, h, w)
        # # get frame-level features
        # frame_level_features = self.backbone.features(frame_input)
        # frame_level_features = F.adaptive_avg_pool2d(frame_level_features, (1, 1)).reshape(b, t, -1)
        # # get video-level features
        # video_level_features = self.temporal_module(frame_level_features)[0][:, -1, :]

        # outputs = self.backbone(data_dict['image'], output_hidden_states=True)
        # sequence_output = outputs[0]
        # video_level_features = self.fc_norm(sequence_output.mean(1))
        # return video_level_features

        temporal_feat = self.extract_temporal_features(data_dict['image'])
        spatial_feat = self.extract_spatial_features(data_dict['image'])
        frequency_feat = self.extract_frequency_features(data_dict['image'])

        # Fuse multi-modal features
        fused_features = self.fusion(temporal_feat, spatial_feat, frequency_feat)
        return fused_features, temporal_feat, spatial_feat, frequency_feat


    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.multitasthead(features)
    
    # def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
    #     label = data_dict['label']
    #     pred = pred_dict['cls']
    #     loss = self.loss_func(pred, label)
    #     loss_dict = {'overall': loss}
    #     return loss_dict

    
    def forward(self, data_dict: dict, inference=False) -> dict:
        """
        Args:
            data_dict: Dictionary containing:
                - 'temporal_clips': (B, num_clips, T, C, H, W)
                - 'spatial_frames': (B, num_frames, C, H, W)
                - 'frequency_frames': (B, num_frames, C, H, W)
                - 'label': (B,)
        Returns:
            pred_dict: Dictionary containing predictions and features
        """
        # Extract features from each stream
        # temporal_feat = self.extract_temporal_features(data_dict['temporal_clips'])
        # spatial_feat = self.extract_spatial_features(data_dict['spatial_frames'])
        # frequency_feat = self.extract_frequency_features(data_dict['frequency_frames'])
        
        # Fuse multi-modal features
        fused_features, temporal_feat, spatial_feat, frequency_feat = self.features(data_dict)
        
        # Module 2: Semantic Grounding (Optional)
        semantic_concepts = None
        grounded_features = fused_features
        
        if self.use_semantic_grounding:
            # Extract explicit semantic concepts using DeepFace
            grounded_features, semantic_concepts = self.semantic_grounding(
                fused_features,
                data_dict['image']  # Use spatial frames for face analysis
            )
        
        # Module 3: Causal Discovery
        violation_score = None
        causal_dag = None
        causal_concepts = None
        
        if self.use_causal and not inference:
            violation_score, causal_dag, causal_concepts = self.causal_module(
                grounded_features, 
                explicit_concepts=semantic_concepts,  # Pass explicit concepts
                return_graph=True
            )
        elif self.use_causal:
            violation_score = self.causal_module(
                grounded_features,
                explicit_concepts=semantic_concepts
            )
        
        # Optional: Sparse autoencoder
        sparse_loss = None
        if self.use_sparse:
            sparse_features, sparse_loss = self.sparse_ae(grounded_features)
            classifier_input = sparse_features
        else:
            classifier_input = grounded_features
        
        # Multi-task classification
        task_outputs = self.classifier(classifier_input)
        
        # Get probabilities
        cls_logits = task_outputs['classification']
        prob = torch.softmax(cls_logits, dim=1)[:, 1]
        
        # Build prediction dict
        pred_dict = {
            'cls': cls_logits,
            'prob': prob,
            'feat': fused_features,
            'grounded_feat': grounded_features,
            'temporal_feat': temporal_feat,
            'spatial_feat': spatial_feat,
            'frequency_feat': frequency_feat,
            'semantic_concepts': semantic_concepts,  # Explicit concepts from DeepFace
            'causal_concepts': causal_concepts,  # Learned concepts from causal module
            'uncertainty': task_outputs.get('uncertainty'),
            'violation_score': violation_score,
            'task_outputs': task_outputs,
            'sparse_loss': sparse_loss,
            'causal_dag': causal_dag
        }
        
        return pred_dict
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        """Compute all losses"""
        label = data_dict['label']
        
        # Classification loss
        cls_loss = self.cls_loss(pred_dict['cls'], label)
        
        # Uncertainty loss (if enabled)
        uncertainty_loss = 0
        if pred_dict.get('uncertainty') is not None:
            # Uncertainty should be higher for misclassified samples
            pred_label = pred_dict['cls'].argmax(dim=1)
            is_correct = (pred_label == label).float()
            target_uncertainty = 1 - is_correct  # High uncertainty for wrong predictions
            uncertainty_loss = self.reg_loss(
                pred_dict['uncertainty'].squeeze(), 
                target_uncertainty
            )
        
        # Causal violation loss
        causal_loss = 0
        if self.use_causal and pred_dict.get('violation_score') is not None:
            # Real videos should have low violation, fake videos high violation
            target_violation = label.float()  # 0 for real, 1 for fake
            causal_loss = self.reg_loss(pred_dict['violation_score'], target_violation)
            
            # Add DAG acyclicity penalty
            if pred_dict.get('causal_dag') is not None:
                dag_penalty = self.causal_module.causal_learner.compute_dag_penalty()
                causal_loss += self.config['causal_module']['dag_learning']['dag_penalty_weight'] * dag_penalty
        
        # Sparse loss
        sparse_loss = pred_dict.get('sparse_loss', 0)
        
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
            'uncertainty': uncertainty_loss,
            'causal': causal_loss,
            'sparse': sparse_loss
        }
        
        return loss_dict
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        """Compute training metrics"""
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
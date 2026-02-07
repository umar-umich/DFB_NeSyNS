"""
Multi-Modal Fusion Module
"""

import torch
import torch.nn as nn
from .cross_modal_attention import CrossModalAttention


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
        elif self.fusion_type == 'weighted':
            self._build_weighted_fusion(temporal_dim, spatial_dim, frequency_dim)
        else:
            raise NotImplementedError(f"Fusion type {self.fusion_type} not implemented")
    
    def _build_weighted_fusion(self, temp_dim, spat_dim, freq_dim):
        """Build weighted fusion with learnable weights"""
        self.temp_proj = nn.Linear(temp_dim, self.projection_dim)
        self.spat_proj = nn.Linear(spat_dim, self.projection_dim)
        self.freq_proj = nn.Linear(freq_dim, self.projection_dim)
        
        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)
        
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
        elif self.fusion_type == 'attention':
            fused = self.fusion(temporal_feat, spatial_feat, frequency_feat)
        elif self.fusion_type == 'weighted':
            # Project to common dimension
            temp = self.temp_proj(temporal_feat)
            spat = self.spat_proj(spatial_feat)
            freq = self.freq_proj(frequency_feat)
            
            # Normalize weights
            weights = torch.softmax(self.fusion_weights, dim=0)
            
            # Weighted sum
            fused = weights[0] * temp + weights[1] * spat + weights[2] * freq
        
        return fused

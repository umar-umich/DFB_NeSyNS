"""
Latent Variable Encoder
"""

import torch
import torch.nn as nn


class LatentVariableEncoder(nn.Module):
    """Encode features into disentangled latent variables"""
    
    def __init__(self, input_dim, z_spatial_dim, z_temporal_dim, z_frequency_dim):
        super().__init__()
        
        self.z_spatial_dim = z_spatial_dim
        self.z_temporal_dim = z_temporal_dim
        self.z_frequency_dim = z_frequency_dim
        
        # Separate encoders for each latent dimension
        self.z_spatial_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, z_spatial_dim)
        )
        
        self.z_temporal_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, z_temporal_dim)
        )
        
        self.z_frequency_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, z_frequency_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, D) - input features
        Returns:
            z_spatial: (B, z_spatial_dim)
            z_temporal: (B, z_temporal_dim)
            z_frequency: (B, z_frequency_dim)
        """
        z_spatial = self.z_spatial_encoder(x)
        z_temporal = self.z_temporal_encoder(x)
        z_frequency = self.z_frequency_encoder(x)
        
        return z_spatial, z_temporal, z_frequency

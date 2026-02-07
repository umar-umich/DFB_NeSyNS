"""
Cross-Modal Attention Fusion
"""

import torch
import torch.nn as nn


class CrossModalAttention(nn.Module):
    """Cross-modal attention fusion"""
    
    def __init__(self, temp_dim, spat_dim, freq_dim, output_dim, num_heads=8):
        super().__init__()
        
        # Project all modalities to same dimension
        hidden_dim = 512
        self.temp_proj = nn.Linear(temp_dim, hidden_dim)
        self.spat_proj = nn.Linear(spat_dim, hidden_dim)
        self.freq_proj = nn.Linear(freq_dim, hidden_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            hidden_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, temporal_feat, spatial_feat, frequency_feat):
        """
        Args:
            temporal_feat: (B, D1)
            spatial_feat: (B, D2)
            frequency_feat: (B, D3)
        Returns:
            fused: (B, output_dim)
        """
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

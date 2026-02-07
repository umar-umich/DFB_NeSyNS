"""
Temporal Feature Extractor using VideoMAE
"""

import torch
import torch.nn as nn
from transformers import VideoMAEModel


class TemporalFeatureExtractor(nn.Module):
    """VideoMAE-based temporal feature extractor"""
    
    def __init__(self, config):
        super().__init__()
        
        model_path = config['foundation_models']['temporal']['model_path']
        self.output_dim = config['foundation_models']['temporal']['output_dim']
        
        # Load pretrained VideoMAE
        self.backbone = VideoMAEModel.from_pretrained(model_path)
        
        # Normalization layer
        self.fc_norm = nn.LayerNorm(self.output_dim)
        
        if config['foundation_models']['temporal']['freeze_backbone']:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze backbone parameters for transfer learning"""
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
        
        # Global average pooling over patches
        features = self.fc_norm(sequence_output.mean(1))
        
        return features

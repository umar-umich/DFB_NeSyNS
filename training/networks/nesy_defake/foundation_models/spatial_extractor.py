"""
Spatial Feature Extractor using CLIP
"""

import torch
import torch.nn as nn
from transformers import CLIPVisionModel


class SpatialFeatureExtractor(nn.Module):
    """CLIP-based spatial feature extractor"""
    
    def __init__(self, config):
        super().__init__()
        
        model_path = config['foundation_models']['spatial']['model_path']
        self.output_dim = config['foundation_models']['spatial']['output_dim']
        
        # Load pretrained CLIP vision encoder
        self.backbone = CLIPVisionModel.from_pretrained(model_path)
        
        if config['foundation_models']['spatial']['freeze_backbone']:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze backbone parameters for transfer learning"""
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

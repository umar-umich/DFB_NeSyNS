"""
Frequency Feature Extractor using SRM + ResNet
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


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
        srm_weights = self._get_srm_filters(num_filters)
        
        conv = nn.Conv2d(3, num_filters, kernel_size=5, padding=2, bias=False)
        conv.weight.data = torch.from_numpy(srm_weights).float()
        
        return conv
    
    def _get_srm_filters(self, num_filters):
        """
        Get SRM filter bank for steganalysis
        
        These are fixed filters designed to detect manipulation artifacts.
        In practice, use the full 30 SRM filters from steganalysis literature.
        """
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
        if depth == 18:
            resnet = models.resnet18(pretrained=True)
        elif depth == 34:
            resnet = models.resnet34(pretrained=True)
        else:
            resnet = models.resnet50(pretrained=True)
        
        # Modify first conv to accept SRM filter outputs
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

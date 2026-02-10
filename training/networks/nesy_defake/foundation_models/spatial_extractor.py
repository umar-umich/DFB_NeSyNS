"""
Spatial Feature Extractor - Multiple backbone options for deepfake detection
Supports: CLIP variants, DINOv2, ConvNeXt, Swin Transformer
"""

import torch
import torch.nn as nn
from transformers import (
    CLIPVisionModel, 
    CLIPProcessor,
    AutoImageProcessor,
    Dinov2Model,
)


class SpatialFeatureExtractor(nn.Module):
    """
    Multi-backbone spatial feature extractor
    
    Supported models (ranked by deepfake detection performance):
    1. DINOv2 (Best for fine-grained features, excellent generalization)
    2. CLIP-ViT-L/14-336 (Strong semantic understanding, good scale)
    3. ConvNeXt-V2 (Strong CNN features, robust to compression)
    4. EVA-CLIP (Largest CLIP variant, excellent features)
    5. Swin-V2 (Hierarchical features, good for artifacts)
    """
    
    def __init__(self, config):
        super().__init__()
        
        spatial_config = config['foundation_models']['spatial']
        self.model_name = spatial_config.get('name', 'clip')
        self.model_path = spatial_config['model_path']
        self.output_dim = spatial_config['output_dim']
        self.freeze_backbone = spatial_config.get('freeze_backbone', False)

        # Get required input size for this model
        self.required_size = self._get_required_input_size()
        self.default_input_size = 224
        # Initialize the appropriate backbone
        self._build_backbone()

        # Add resize if needed
        if self.required_size != self.default_input_size:
            print(f" Adding resize: {self.default_input_size}×{self.default_input_size} → {self.required_size}×{self.required_size}")
            self.resize = nn.functional.interpolate
            self.needs_resize = True
        else:
            print(f" Input size matches: {self.required_size}×{self.required_size}")
            self.needs_resize = False

        # Add projection layer if output dims don't match
        if self.backbone_dim != self.output_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.backbone_dim, self.output_dim),
                nn.LayerNorm(self.output_dim),
                nn.GELU()
            )
        else:
            self.projection = nn.Identity()
        
        if self.freeze_backbone:
            self._freeze_backbone()
        
        print(f"Spatial Extractor: {self.model_name} ({self.model_path})")
        print(f"  Input: {self.default_input_size}×{self.default_input_size} → Model: {self.required_size}×{self.required_size}")
        print(f"  Backbone dim: {self.backbone_dim} -> Output dim: {self.output_dim}")
        print(f"  Frozen: {self.freeze_backbone}")


    def _get_required_input_size(self):
        """Get the required input size for the model"""
        size_map = {
            # DINOv2
            'facebook/dinov2-small': 224,
            'facebook/dinov2-base': 224,
            'facebook/dinov2-large': 224,
            'facebook/dinov2-giant': 224,
            'facebook/dinov2-small-518': 518,
            'facebook/dinov2-base-518': 518,
            'facebook/dinov2-large-518': 518,
            'facebook/dinov2-giant-518': 518,
            
            # CLIP
            'openai/clip-vit-base-patch16': 224,
            'openai/clip-vit-base-patch32': 224,
            'openai/clip-vit-large-patch14': 224,
            'openai/clip-vit-large-patch14-336': 336,
            'laion/CLIP-ViT-H-14-laion2B-s32B-b79K': 224,
            
            # EVA-CLIP
            'EVA02-CLIP-L-14-336': 336,
            'EVA02-CLIP-E-14-plus': 224,
        }
        
        return size_map.get(self.model_path, 224)  # Default to 224
    
    def _build_backbone(self):
        """Build the appropriate backbone based on model name"""
        
        if self.model_name == 'dinov2':
            self._build_dinov2()
        elif self.model_name == 'clip':
            self._build_clip()
        elif self.model_name == 'eva_clip':
            self._build_eva_clip()
        else:
            raise ValueError(f"Unknown spatial model: {self.model_name}")
    
    def _build_dinov2(self):
        """
        DINOv2 - Best for deepfake detection
        Self-supervised, excellent fine-grained features
        Recommended: facebook/dinov2-large or facebook/dinov2-giant
        """
        print("Loading DINOv2 (Recommended for deepfakes)...")
        
        # Load model
        self.backbone = Dinov2Model.from_pretrained(self.model_path)
        
        # DINOv2 output dimensions
        dino_dims = {
            'facebook/dinov2-small': 384,
            'facebook/dinov2-base': 768,
            'facebook/dinov2-large': 1024,
            # 'facebook/dinov2-giant': 1536
        }
        self.backbone_dim = dino_dims.get(self.model_path, 1024)
        
        # Create processor for normalization
        self.processor = AutoImageProcessor.from_pretrained(self.model_path)
        self.use_processor = True
        
    def _build_clip(self):
        """
        CLIP variants
        Good semantic understanding, pretrained on image-text pairs
        """
        print(f"Loading CLIP: {self.model_path}...")
        
        # Load model
        self.backbone = CLIPVisionModel.from_pretrained(self.model_path)
        self.processor = CLIPProcessor.from_pretrained(self.model_path)
        
        # CLIP output dimensions
        clip_dims = {
            'openai/clip-vit-base-patch32': 768,
            'openai/clip-vit-large-patch14': 1024,
            'openai/clip-vit-large-patch14-336': 1024,
            'laion/CLIP-ViT-H-14-laion2B-s32B-b79K': 1024,
        }
        self.backbone_dim = clip_dims.get(self.model_path, 768)
        self.use_processor = False  # We handle normalization ourselves
        
    def _build_eva_clip(self):
        """
        EVA-CLIP - Largest and strongest CLIP variant
        1B parameters, excellent for generalization
        Requires: pip install open_clip_torch
        """
        print("Loading EVA-CLIP (Strongest CLIP variant)...")
        try:
            import open_clip
            
            # Map model names
            eva_models = {
                'EVA02-CLIP-L-14-336': ('EVA02-L-14-336', 'merged2b_s6b_b61k'),
                'EVA02-CLIP-E-14-plus': ('EVA02-E-14-plus', 'laion2b_s9b_b144k'),
            }
            
            if self.model_path in eva_models:
                model_name, pretrained = eva_models[self.model_path]
                self.backbone, _, self.processor = open_clip.create_model_and_transforms(
                    model_name, 
                    pretrained=pretrained
                )
                self.backbone = self.backbone.visual  # Use only vision encoder
            else:
                raise ValueError(f"Unknown EVA-CLIP model: {self.model_path}")
            
            # EVA dimensions
            eva_dims = {
                'EVA02-CLIP-L-14-336': 768,
                'EVA02-CLIP-E-14-plus': 1024,
            }
            self.backbone_dim = eva_dims.get(self.model_path, 1024)
            self.use_processor = True
            
        except ImportError:
            raise ImportError("EVA-CLIP requires: pip install open_clip_torch")
    
    
    def _freeze_backbone(self):
        """Freeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("  ✓ Backbone frozen")
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) - ALREADY normalized with correct params
        """
        B, C, H, W = x.shape
        # Resize if needed
        if self.needs_resize:
            x = nn.functional.interpolate(
                x, 
                size=(self.required_size, self.required_size),
                mode='bilinear',
                align_corners=False
            )
        
        # Extract features based on model type
        if self.model_name == 'dinov2':
            features = self._forward_dinov2(x)
        elif self.model_name == 'clip':
            features = self._forward_clip(x)
        elif self.model_name == 'eva_clip':
            features = self._forward_eva_clip(x)
        
        # Project to target dimension
        features = self.projection(features)
        
        return features
    
    def _forward_dinov2(self, x):
        """DINOv2 forward pass"""
        outputs = self.backbone(pixel_values=x, output_hidden_states=True)
        
        # Use CLS token from last layer
        features = outputs.last_hidden_state[:, 0]  # (B, D)
        
        return features
    
    def _forward_clip(self, x):
        """CLIP forward pass"""
        outputs = self.backbone(pixel_values=x, output_hidden_states=True)
        
        # Use pooled output (CLS token after projection)
        features = outputs.pooler_output  # (B, D)
        
        return features
    
    def _forward_eva_clip(self, x):
        """EVA-CLIP forward pass"""
        # EVA uses open_clip interface
        features = self.backbone(x)  # (B, D)
        return features
    

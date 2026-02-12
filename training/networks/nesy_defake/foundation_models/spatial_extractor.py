"""
Spatial Feature Extractor - Multiple backbone options for deepfake detection
Supports: CLIP variants, DINOv2, EVA-CLIP, GenD (CLIP/DINOv2/PerceptionEncoder via GenD wrapper)
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
    Multi-backbone spatial feature extractor.

    Supported model names (config key: 'name'):
      - 'dinov2'     : DINOv2 (best for fine-grained features, excellent generalization)
      - 'clip'       : CLIP ViT variants (strong semantic understanding)
      - 'eva_clip'   : EVA-CLIP (largest CLIP variant, excellent features)
      - 'gend'       : GenD (uses GenD's pretrained feature_extractor — CLIP, DINOv2, or
                       PerceptionEncoder depending on the checkpoint — strips the
                       classification head and returns raw backbone features)

    GenD config example (config['foundation_models']['spatial']):
        name:            gend
        model_path:      /path/to/gend_checkpoint   # local dir or HF repo id
        output_dim:      1024
        freeze_backbone: true

    The backbone variant (CLIP / DINO / PerceptionEncoder) is determined automatically
    from the GenD checkpoint's config.backbone field, so no extra config key is needed.
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

        # Add resize layer if the model needs a different resolution than 224
        if self.required_size != self.default_input_size:
            print(f"  Adding resize: {self.default_input_size}×{self.default_input_size}"
                  f" → {self.required_size}×{self.required_size}")
            self.needs_resize = True
        else:
            print(f"  Input size matches: {self.required_size}×{self.required_size}")
            self.needs_resize = False

        # Projection layer: align backbone dim to desired output dim
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
        print(f"  Input: {self.default_input_size}×{self.default_input_size}"
              f" → Model: {self.required_size}×{self.required_size}")
        print(f"  Backbone dim: {self.backbone_dim} -> Output dim: {self.output_dim}")
        print(f"  Frozen: {self.freeze_backbone}")

    # ------------------------------------------------------------------
    # Input size resolution
    # ------------------------------------------------------------------

    def _get_required_input_size(self):
        """Return the spatial resolution this model expects."""
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

            # GenD checkpoints — resolution follows the embedded backbone.
            # PerceptionEncoder large-336 needs 336; all others default to 224.
            # Add your checkpoint paths here if they differ from 224.
            'yermandy/GenD_PE_L': 336,   # PerceptionEncoder large-336 variant
        }
        return size_map.get(self.model_path, 224)

    # ------------------------------------------------------------------
    # Backbone construction
    # ------------------------------------------------------------------

    def _build_backbone(self):
        if self.model_name == 'dinov2':
            self._build_dinov2()
        elif self.model_name == 'clip':
            self._build_clip()
        elif self.model_name == 'eva_clip':
            self._build_eva_clip()
        elif self.model_name == 'gend':
            self._build_gend()
        else:
            raise ValueError(f"Unknown spatial model: {self.model_name}")

    def _build_dinov2(self):
        """DINOv2 — self-supervised, excellent fine-grained features."""
        print("Loading DINOv2...")
        self.backbone = Dinov2Model.from_pretrained(self.model_path)
        dino_dims = {
            'facebook/dinov2-small': 384,
            'facebook/dinov2-base': 768,
            'facebook/dinov2-large': 1024,
            'facebook/dinov2-giant': 1536,
        }
        self.backbone_dim = dino_dims.get(self.model_path, 1024)
        self.processor = AutoImageProcessor.from_pretrained(self.model_path)
        self.use_processor = True

    def _build_clip(self):
        """CLIP ViT variants — strong semantic understanding."""
        print(f"Loading CLIP: {self.model_path}...")
        self.backbone = CLIPVisionModel.from_pretrained(self.model_path)
        self.processor = CLIPProcessor.from_pretrained(self.model_path)
        clip_dims = {
            'openai/clip-vit-base-patch32': 768,
            'openai/clip-vit-large-patch14': 1024,
            'openai/clip-vit-large-patch14-336': 1024,
            'laion/CLIP-ViT-H-14-laion2B-s32B-b79K': 1024,
        }
        self.backbone_dim = clip_dims.get(self.model_path, 768)
        self.use_processor = False

    def _build_eva_clip(self):
        """EVA-CLIP — largest CLIP variant. Requires: pip install open_clip_torch"""
        print("Loading EVA-CLIP...")
        try:
            import open_clip
            eva_models = {
                'EVA02-CLIP-L-14-336': ('EVA02-L-14-336', 'merged2b_s6b_b61k'),
                'EVA02-CLIP-E-14-plus': ('EVA02-E-14-plus', 'laion2b_s9b_b144k'),
            }
            if self.model_path not in eva_models:
                raise ValueError(f"Unknown EVA-CLIP model: {self.model_path}")
            model_name, pretrained = eva_models[self.model_path]
            model, _, self.processor = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            self.backbone = model.visual
            eva_dims = {
                'EVA02-CLIP-L-14-336': 768,
                'EVA02-CLIP-E-14-plus': 1024,
            }
            self.backbone_dim = eva_dims.get(self.model_path, 1024)
            self.use_processor = True
        except ImportError:
            raise ImportError("EVA-CLIP requires: pip install open_clip_torch")

    def _build_gend(self):
        """
        GenD — loads a pretrained GenD checkpoint and extracts its feature_extractor
        (one of CLIPEncoder, DINOEncoder, or PerceptionEncoder). The classification
        head is discarded; only the backbone is kept for feature extraction.

        The backbone variant is determined automatically by GenD from its checkpoint
        config, so no additional config is required here.

        Requires: pip install timm  (only if the checkpoint uses PerceptionEncoder)
        """
        print(f"Loading GenD feature extractor from: {self.model_path}...")
        try:
            from networks.nesy_defake.foundation_models.GenD.model import GenD
        except ImportError:
            # Fall back to a relative import if the package is not installed
            from GenD.model import GenD  # type: ignore

        gend_model = GenD.from_pretrained(self.model_path) # , local_files_only=True

        # Keep only the feature extractor — the linear classification head is
        # not needed here; we want raw backbone features for downstream fusion.
        self.backbone = gend_model.feature_extractor
        self.backbone_dim = self.backbone.get_features_dim()

        # GenD encoders expect pre-normalized tensors in forward(). Their
        # preprocess() method is the PIL offline path and is not used here,
        # since normalization is already handled upstream in the data pipeline.
        self.use_processor = False

        # Re-check required resolution from the embedded backbone name in case
        # the checkpoint path wasn't in the size_map above.
        backbone_name = gend_model.config.backbone.lower()
        if '336' in backbone_name:
            self.required_size = 336

        print(f"  GenD backbone: {gend_model.config.backbone}")
        print(f"  GenD backbone dim: {self.backbone_dim}")

    # ------------------------------------------------------------------
    # Freezing
    # ------------------------------------------------------------------

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("  ✓ Backbone frozen")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) — already normalized with the correct stats
        Returns:
            (B, output_dim)
        """
        if self.needs_resize:
            x = nn.functional.interpolate(
                x,
                size=(self.required_size, self.required_size),
                mode='bilinear',
                align_corners=False,
            )

        if self.model_name == 'dinov2':
            features = self._forward_dinov2(x)
        elif self.model_name == 'clip':
            features = self._forward_clip(x)
        elif self.model_name == 'eva_clip':
            features = self._forward_eva_clip(x)
        elif self.model_name == 'gend':
            features = self._forward_gend(x)
        else:
            raise ValueError(f"Unknown spatial model: {self.model_name}")

        return self.projection(features)

    def _forward_dinov2(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=x, output_hidden_states=True)
        return outputs.last_hidden_state[:, 0]  # CLS token, (B, D)

    def _forward_clip(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=x, output_hidden_states=True)
        return outputs.pooler_output  # (B, D)

    def _forward_eva_clip(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # open_clip visual encoder, (B, D)

    def _forward_gend(self, x: torch.Tensor) -> torch.Tensor:
        """
        All three GenD encoder types (CLIPEncoder, DINOEncoder, PerceptionEncoder)
        return (B, D) directly from their forward(), so this is a single unified call.
        """
        return self.backbone(x)  # (B, D)
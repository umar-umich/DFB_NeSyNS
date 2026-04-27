"""
networks/nesy_defake/foundation_models/spatial_feature_extractor.py
====================================================================
Spatial Feature Extractor — GenD-style Training Regime

WHAT CHANGED AND WHY
---------------------
Previous version problems:
  1. _build_gend() loaded GenD's FINE-TUNED checkpoint (yermandy/GenD_PE_L),
     then extracted its PerceptionEncoder. Those backbone weights were already
     adapted by GenD's supervised training — stacking a new projection head on
     top of an already-opinionated backbone means the new head fights the old
     signal. Result: slow convergence, poor generalization.

  2. The extractor had its own internal self.projection (Linear→LN→GELU) AND
     the detector added another spatial_proj on top. Two stacked projection
     heads, the inner one frozen. Redundant and noisy.

  3. freeze_backbone=True froze EVERYTHING, so nothing learned.
     freeze_backbone=False unfroze EVERYTHING, so the backbone drifted from
     ImageNet/LAION pretraining — catastrophic for generalization.
     Neither option matched the GenD training regime.

GenD's actual training regime (what makes it generalize):
  - Backbone (CLIP/DINO/PE): FROZEN except LayerNorms
  - LayerNorms inside the backbone: TRAINABLE (adapts normalization statistics
    to the deepfake domain without moving the feature manifold)
  - Linear classification head: TRAINABLE
  This is why GenD generalizes — the feature manifold stays on the pretrained
  surface, only the normalization scale/shift adapts.

What this file does instead:
  - Loads RAW pretrained CLIP (openai/clip-vit-large-patch14) directly,
    NOT through a fine-tuned GenD checkpoint
  - Freezes all backbone parameters
  - Selectively UNFREEZES all LayerNorm weight+bias inside the backbone
  - Removes the internal self.projection — the detector's spatial_proj is
    the only projection head, avoiding the double-projection problem
  - Exposes get_trainable_params() for the optimizer to build correct
    per-module param groups

Supported backbone names (config key: name):
  'clip'    — raw CLIP ViT (recommended, matches GenD's best-generalizing variant)
  'dinov3'  — Meta DINOv3, successor to DINOv2; stronger dense features (timm)
  'pe'      — Meta Perception Encoder (PE-Core / PE-Spatial), loaded via timm
  'gend'    — KEPT for compatibility, but now loads raw CLIP from the GenD
              config's backbone field rather than the fine-tuned head weights
              NOTE: if you set name=gend, set model_path to the raw CLIP HF id
              e.g. openai/clip-vit-large-patch14, NOT a GenD checkpoint path.

Config example (in foundation_models.spatial):
    name:            clip                          # raw backbone, GenD-style training
    model_path:      openai/clip-vit-large-patch14 # raw HuggingFace id
    output_dim:      1024                          # backbone_dim — no internal projection
    freeze_backbone: true                          # always true; LayerNorms still train
    train_layernorms: true                         # GenD-style: only LNs train in backbone
    normalization:
      mean: [0.481, 0.458, 0.408]
      std:  [0.269, 0.261, 0.276]
"""

import logging
from typing import Iterator, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_layernorm_params(module: nn.Module) -> List[nn.Parameter]:
    """
    Walk the module tree and collect all parameters that belong to a
    LayerNorm (or RMSNorm / GroupNorm) layer. These are the only backbone
    parameters that will be trained under the GenD regime.
    """
    ln_types = (nn.LayerNorm, nn.GroupNorm, nn.RMSNorm
                if hasattr(nn, 'RMSNorm') else nn.LayerNorm)
    params = []
    for mod in module.modules():
        if isinstance(mod, ln_types):
            for p in mod.parameters(recurse=False):
                if p.requires_grad:
                    params.append(p)
    return params


def _freeze_all_except_layernorms(module: nn.Module) -> Tuple[int, int]:
    """
    Freeze every parameter in `module` then selectively unfreeze
    LayerNorm / GroupNorm / RMSNorm weight and bias.

    Returns (total_params, trainable_params) for logging.
    """
    # Step 1: freeze everything
    for p in module.parameters():
        p.requires_grad = False

    # Step 2: unfreeze LayerNorm-family layers only
    ln_types = (nn.LayerNorm, nn.GroupNorm)
    if hasattr(nn, 'RMSNorm'):
        ln_types = ln_types + (nn.RMSNorm,)

    unfrozen = 0
    for mod in module.modules():
        if isinstance(mod, ln_types):
            for p in mod.parameters(recurse=False):
                p.requires_grad = True
                unfrozen += p.numel()

    total = sum(p.numel() for p in module.parameters())
    return total, unfrozen


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SpatialFeatureExtractor(nn.Module):
    """
    Spatial feature extractor with GenD-style training regime.

    Key design decisions:
      1. Loads a RAW pretrained backbone (CLIP, DINOv2) — not a downstream
         fine-tuned checkpoint.
      2. Freezes the backbone entirely then selectively unfreezes all
         LayerNorms inside it (GenD's generalization secret).
      3. Does NOT add an internal projection head. The detector's
         spatial_proj is the only projection, eliminating the double-
         projection problem from the previous version.
      4. Exposes get_trainable_params() so the optimizer can apply the
         correct (lower) learning rate to backbone LayerNorms vs the
         (higher) learning rate to the detector's projection head.

    Forward input:  (B, C, H, W) — pre-normalized frames
    Forward output: (B, output_dim) — raw backbone CLS / pooler features
    """

    def __init__(self, config: dict):
        super().__init__()

        spatial_cfg = config['foundation_models']['spatial']
        self.model_name      = spatial_cfg.get('name', 'clip')
        self.model_path      = spatial_cfg['model_path']
        self.output_dim      = spatial_cfg['output_dim']
        self.freeze_backbone = spatial_cfg.get('freeze_backbone', True)
        self.train_layernorms = spatial_cfg.get('train_layernorms', True)

        # ------------------------------------------------------------------
        # Build backbone (raw pretrained weights only)
        # ------------------------------------------------------------------
        self._build_backbone()

        # Sanity-check: output_dim should match backbone_dim.
        # The detector's spatial_proj handles the mapping if they differ,
        # but we warn loudly here because a mismatch usually means the
        # YAML output_dim is wrong.
        if self.backbone_dim != self.output_dim:
            logger.warning(
                f"[SpatialExtractor] backbone_dim={self.backbone_dim} != "
                f"output_dim={self.output_dim}. The detector's spatial_proj "
                f"will map {self.backbone_dim}→{self.output_dim}. "
                f"If this is intentional, ignore. Otherwise fix the YAML."
            )

        # ------------------------------------------------------------------
        # Apply GenD-style freezing
        # ------------------------------------------------------------------
        if self.freeze_backbone:
            if self.train_layernorms:
                total, unfrozen = _freeze_all_except_layernorms(self.backbone)
                logger.info(
                    f"[SpatialExtractor] GenD-style freeze: "
                    f"{unfrozen:,} / {total:,} backbone params trainable "
                    f"(LayerNorms only)"
                )
            else:
                # Hard freeze — nothing in backbone trains (not recommended)
                for p in self.backbone.parameters():
                    p.requires_grad = False
                logger.info(
                    f"[SpatialExtractor] Hard freeze: all backbone params frozen. "
                    f"Only detector projection head will train."
                )
        else:
            # Full fine-tune — not recommended for generalization
            logger.warning(
                "[SpatialExtractor] freeze_backbone=False: full backbone fine-tune. "
                "This will hurt cross-dataset generalization. "
                "Consider freeze_backbone=true + train_layernorms=true instead."
            )

        # ------------------------------------------------------------------
        # Resize layer (only added when model needs != 224)
        # ------------------------------------------------------------------
        self.required_size = self._get_required_input_size()
        self.needs_resize   = (self.required_size != 224)
        if self.needs_resize:
            logger.info(
                f"[SpatialExtractor] Will resize 224→{self.required_size} in forward()"
            )

        logger.info(
            f"[SpatialExtractor] Ready — backbone={self.model_name}, "
            f"path={self.model_path}, backbone_dim={self.backbone_dim}, "
            f"output_dim={self.output_dim}, frozen={self.freeze_backbone}, "
            f"layernorm_train={self.train_layernorms}"
        )

    # ------------------------------------------------------------------
    # Input size map
    # ------------------------------------------------------------------

    def _get_required_input_size(self) -> int:
        size_map = {
            # CLIP family
            'openai/clip-vit-base-patch16':           224,
            'openai/clip-vit-base-patch32':           224,
            'openai/clip-vit-large-patch14':          224,
            'openai/clip-vit-large-patch14-336':      336,
            'laion/CLIP-ViT-H-14-laion2B-s32B-b79K':  224,
            # DINOv3
            'facebook/dinov3-vits16-pretrain-lvd1689m':      224,
            'facebook/dinov3-vitb16-pretrain-lvd1689m':      224,
            'facebook/dinov3-vitl16-pretrain-lvd1689m':      224,
            'facebook/dinov3-vith16plus-pretrain-lvd1689m':  224,
            # Perception Encoder (PE)
            'facebook/PE-Core-B16-224':               224,
            'facebook/PE-Core-L14-336':               336,
            'facebook/PE-Core-G14-448':               448,
            'facebook/PE-Spatial-L14-448':            448,
            'facebook/PE-Spatial-G14-448':            448,
        }
        if self.model_path in size_map:
            return size_map[self.model_path]
        # Fallback: PE / CLIP path conventions end in -224 / -336 / -448.
        # Parse the trailing token so new variants don't need a map edit.
        for suffix in (448, 384, 336, 256, 224):
            if self.model_path.endswith(f'-{suffix}'):
                return suffix
        return 224

    # ------------------------------------------------------------------
    # Backbone builders — raw pretrained weights ONLY
    # ------------------------------------------------------------------

    def _build_backbone(self):
        name = self.model_name
        dispatch = {
            'clip':    self._build_clip,
            'dinov3':  self._build_dinov3,
            'pe':      self._build_pe,
        }
        if name not in dispatch:
            raise ValueError(
                f"[SpatialExtractor] Unknown backbone name: '{name}'. "
                f"Choose from: {sorted(dispatch.keys())}"
            )
        dispatch[name]()

    def _build_clip(self):
        """
        Load raw CLIP vision encoder from HuggingFace.
        Only the vision model is kept; text encoder and projection are discarded.
        No fine-tuning history — pure pretrained ImageNet/LAION features.
        """
        from transformers import CLIPVisionModel

        logger.info(f"[SpatialExtractor] Loading raw CLIP from: {self.model_path}")
        self.backbone = CLIPVisionModel.from_pretrained(self.model_path)

        clip_dims = {
            'openai/clip-vit-base-patch16':           768,
            'openai/clip-vit-base-patch32':           768,
            'openai/clip-vit-large-patch14':          1024,
            'openai/clip-vit-large-patch14-336':      1024,
            'laion/CLIP-ViT-H-14-laion2B-s32B-b79K': 1280,
        }
        self.backbone_dim  = clip_dims.get(self.model_path, 1024)
        self._forward_fn   = self._forward_clip


    def _build_dinov3(self):
        """
        Load raw DINOv3 via timm.

        NOTE: HuggingFace transformers < 4.45 does not recognise the
        'dinov3_vit' architecture (ValueError on AutoConfig). timm
        (>= 1.0.25) has first-class DINOv3 support and handles the raw
        Meta checkpoints directly.

        You can specify either a timm model name (e.g.
        vit_large_patch16_dinov3) or keep the familiar HF-style path
        (facebook/dinov3-vitl16-pretrain-lvd1689m) — the map below
        translates it.
        """
        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "[SpatialExtractor] DINOv3 backbone requires timm (>=1.0.25). "
                "Install with:  pip install -U timm"
            ) from e

        hf_to_timm = {
            'facebook/dinov3-vits16-pretrain-lvd1689m':     'vit_small_patch16_dinov3',
            'facebook/dinov3-vitb16-pretrain-lvd1689m':     'vit_base_patch16_dinov3',
            'facebook/dinov3-vitl16-pretrain-lvd1689m':     'vit_large_patch16_dinov3',
            'facebook/dinov3-vith16plus-pretrain-lvd1689m': 'vit_huge_plus_patch16_dinov3',
        }
        timm_name = hf_to_timm.get(self.model_path, self.model_path)
        logger.info(f"[SpatialExtractor] Loading DINOv3 via timm: {timm_name}")
        # num_classes=0 drops the classification head → model(x) returns
        # the globally-pooled CLS feature directly (B, num_features).
        self.backbone = timm.create_model(
            timm_name, pretrained=True, num_classes=0,
        )

        dims = {
            'facebook/dinov3-vits16-pretrain-lvd1689m':     384,
            'facebook/dinov3-vitb16-pretrain-lvd1689m':     768,
            'facebook/dinov3-vitl16-pretrain-lvd1689m':     1024,
            'facebook/dinov3-vith16plus-pretrain-lvd1689m': 1280,
        }
        self.backbone_dim = dims.get(
            self.model_path, int(getattr(self.backbone, 'num_features', 1024)))
        self._forward_fn = self._forward_dinov3

    def _build_pe(self):
        """
        Load Meta Perception Encoder (PE-Core / PE-Spatial) via timm.

        NOTE: The HF PE repos (facebook/PE-Core-*, facebook/PE-Spatial-*)
        ship raw Meta .pt checkpoints — there is no config.json, so
        HuggingFace's AutoModel.from_pretrained cannot load them. timm
        (>= 1.0.15) has first-class PE support and handles the checkpoint
        conversion automatically.

        Either specify a timm model name directly (e.g. vit_pe_core_large_
        patch14_336) or keep the familiar HF id (facebook/PE-Core-L14-336)
        — the HF→timm name map below translates it.
        """
        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "[SpatialExtractor] PE backbone requires timm (>=1.0.15). "
                "Install with:  pip install -U timm"
            ) from e

        hf_to_timm = {
            'facebook/PE-Core-B16-224':    'vit_pe_core_base_patch16_224',
            'facebook/PE-Core-L14-336':    'vit_pe_core_large_patch14_336',
            'facebook/PE-Core-G14-448':    'vit_pe_core_gigantic_patch14_448',
            'facebook/PE-Spatial-G14-448': 'vit_pe_spatial_gigantic_patch14_448',
            'facebook/PE-Spatial-L14-448': 'vit_pe_spatial_large_patch14_448',
        }
        timm_name = hf_to_timm.get(self.model_path, self.model_path)
        logger.info(f"[SpatialExtractor] Loading PE via timm: {timm_name}")
        # num_classes=0 drops the classification head so model(x) returns
        # the globally-pooled feature directly (B, num_features).
        self.backbone = timm.create_model(
            timm_name, pretrained=True, num_classes=0,
        )

        # Prefer the map; fall back to the model's declared num_features.
        dims = {
            'facebook/PE-Core-B16-224':    768,
            'facebook/PE-Core-L14-336':    1024,
            'facebook/PE-Core-G14-448':    1536,
            'facebook/PE-Spatial-G14-448': 1536,
            'facebook/PE-Spatial-L14-448': 1024,
        }
        self.backbone_dim = dims.get(
            self.model_path, int(getattr(self.backbone, 'num_features', 1024)))
        self._forward_fn = self._forward_pe

    # ------------------------------------------------------------------
    # Trainable parameter helpers
    # ------------------------------------------------------------------

    def get_trainable_params(self) -> List[nn.Parameter]:
        """
        Return only the backbone parameters that should be trained
        (LayerNorms if train_layernorms=True, all if freeze_backbone=False).

        Used by the optimizer builder in train.py to create per-module
        param groups with the correct (lower) learning rate for backbone LNs.

        Note: The detector's spatial_proj parameters are NOT included here
        because they live on the detector, not on this extractor. The
        optimizer builder should add those separately at a higher LR.
        """
        if not self.freeze_backbone:
            # Full fine-tune: return all backbone params
            return [p for p in self.backbone.parameters() if p.requires_grad]
        if self.train_layernorms:
            # GenD regime: only LayerNorm params
            return _collect_layernorm_params(self.backbone)
        # Hard freeze: nothing trains in backbone
        return []


    def unfreeze_layernorms(self) -> int:
        """
        Phase 2 hook: unfreeze all LayerNorm params in the backbone.
        Called by the trainer at the phase transition epoch.
        Returns the number of newly unfrozen parameters.
        """
        _, unfrozen = _freeze_all_except_layernorms(self.backbone)
        self.train_layernorms = True
        logger.info(f"[SpatialExtractor] Phase 2: unfroze {unfrozen:,} LN params")
        return unfrozen
        
    def count_trainable_params(self) -> Tuple[int, int]:
        """Returns (trainable, total) param counts for logging."""
        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.backbone.parameters())
        return trainable, total

    # ------------------------------------------------------------------
    # Forward implementations
    # ------------------------------------------------------------------

    def _forward_clip(self, x: torch.Tensor) -> torch.Tensor:
        """CLIP pooler_output — the projected CLS embedding. Shape: (B, D)"""
        outputs = self.backbone(pixel_values=x)
        return outputs.pooler_output  # (B, backbone_dim)

    def _forward_dinov3(self, x: torch.Tensor) -> torch.Tensor:
        """DINOv3 (timm backbone, num_classes=0) returns pooled CLS: (B, D)."""
        return self.backbone(x)

    def _forward_pe(self, x: torch.Tensor) -> torch.Tensor:
        """PE (timm backbone with num_classes=0) returns pooled CLS: (B, D)."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) — pre-normalized, values in appropriate range
                              for the chosen backbone

        Returns:
            (B, backbone_dim) — raw backbone features, NO internal projection.
            The detector's spatial_proj handles dim mapping.
        """
        # Resize if the backbone expects a different resolution
        if self.needs_resize:
            x = F.interpolate(
                x,
                size=(self.required_size, self.required_size),
                mode='bilinear',
                align_corners=False,
            )

        return self._forward_fn(x)
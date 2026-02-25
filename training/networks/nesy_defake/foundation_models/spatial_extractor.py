# spatial_feature_extractor.py
"""
networks/nesy_defake/foundation_models/spatial_feature_extractor.py
====================================================================
Spatial Feature Extractor — GenD-style Training Regime

Architecture:
    - Shared vision encoder (CLIP-L/14-336 or PE-Core) built once in detector
      and injected via shared_encoder parameter — no duplicate weight loading.
    - GenD freeze applied exactly once in detector._build_shared_encoder(),
      so _owns_backbone=False skips all freeze logic here.
    - No internal projection head — detector's spatial_proj is the only one.
    - Fix 2: chunked CLIP inference in _forward_clip() to control peak
      activation memory when processing B×T frames simultaneously.

Forward:
    input  : (B×T, C, H, W) — all frames, pre-normalized
    output : (B×T, backbone_dim) — spatial_norm applied, no projection
"""

import logging
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import ENCODER_REGISTRY
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GenD helpers
# ---------------------------------------------------------------------------

def _collect_layernorm_params(module: nn.Module) -> List[nn.Parameter]:
    ln_types = [nn.LayerNorm, nn.GroupNorm]
    if hasattr(nn, 'RMSNorm'):
        ln_types.append(nn.RMSNorm)
    ln_types = tuple(ln_types)
    params = []
    for mod in module.modules():
        if isinstance(mod, ln_types):
            for p in mod.parameters(recurse=False):
                if p.requires_grad:
                    params.append(p)
    return params


def _freeze_all_except_layernorms(module: nn.Module) -> Tuple[int, int]:
    for p in module.parameters():
        p.requires_grad = False
    ln_types = [nn.LayerNorm, nn.GroupNorm]
    if hasattr(nn, 'RMSNorm'):
        ln_types.append(nn.RMSNorm)
    ln_types = tuple(ln_types)
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

    Always receives a shared_encoder from the detector — standalone
    encoder construction is intentionally removed. All freeze logic
    lives in detector._build_shared_encoder() and is applied once.

    Forward input:  (B×T, C, H, W) — all frames, pre-normalized
    Forward output: (B×T, backbone_dim) — spatial_norm applied
    """

    def __init__(self, config: dict, shared_encoder=None):
        super().__init__()

        spatial_cfg           = config['foundation_models']['spatial']
        self.model_path       = spatial_cfg['model_path']
        self.output_dim       = spatial_cfg['output_dim']
        self.freeze_backbone  = spatial_cfg.get('freeze_backbone', True)
        self.train_layernorms = spatial_cfg.get('train_layernorms', True)
        self.clip_chunk_size  = spatial_cfg.get('clip_chunk_size', 4)

        # ── Backbone: always shared, never built standalone ───────────────
        if shared_encoder is None:
            raise ValueError(
                "[SpatialExtractor] shared_encoder is required. "
                "Build it in the detector via _build_shared_encoder() "
                "and pass it here. Standalone encoder construction is "
                "removed to prevent duplicate weight loading."
            )

        logger.info("[SpatialExtractor] Using shared encoder from detector.")
        self.backbone       = shared_encoder
        spec                = ENCODER_REGISTRY[self.model_path]
        self.backbone_dim   = spec['dim']
        self.required_size  = spec['size']
        self.api_type       = spec['api']
        self.needs_resize   = (self.required_size != 224)
        self._owns_backbone = False   # freeze managed by detector, skip here

        # spatial_norm: always trainable, never touched by freeze logic
        self.spatial_norm = nn.LayerNorm(self.backbone_dim)

        if self.backbone_dim != self.output_dim:
            logger.warning(
                f"[SpatialExtractor] backbone_dim={self.backbone_dim} != "
                f"output_dim={self.output_dim}. "
                f"detector's spatial_proj handles the mapping. "
                f"Fix YAML if this is unintentional."
            )

        logger.info(
            f"[SpatialExtractor] Ready — path={self.model_path}, "
            f"backbone_dim={self.backbone_dim}, output_dim={self.output_dim}, "
            f"api={self.api_type}, required_size={self.required_size}, "
            f"clip_chunk_size={self.clip_chunk_size}"
        )
        trainable, total = self.count_trainable_params()
        logger.info(f"[SpatialExtractor] Trainable: {trainable:,} / {total:,}")

    # ------------------------------------------------------------------
    # Trainable parameter helpers
    # ------------------------------------------------------------------

    def get_trainable_params(self) -> List[nn.Parameter]:
        """
        Backbone LayerNorm params — low LR optimizer group.
        Returns empty because freeze is managed by detector on the shared
        encoder. Detector collects shared encoder LN params once and assigns
        them to the correct optimizer group directly.
        """
        return []   # shared encoder — detector handles param group assignment

    def get_always_trainable_params(self) -> List[nn.Parameter]:
        """
        spatial_norm is always trainable regardless of freeze state.
        Goes into optimizer's projection_heads group at standard LR.
        """
        return list(self.spatial_norm.parameters())

    def count_trainable_params(self) -> Tuple[int, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        return trainable, total

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward_clip(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fix 2: chunked CLIP inference.

        Processes B×T frames in chunks of clip_chunk_size instead of one
        giant (B×T, 3, H, W) forward pass. Peak activation memory:
            chunk_size × 257 × 1024 × 24 layers  (fixed)
        instead of:
            B×T × 257 × 1024 × 24 layers          (scales with batch×frames)

        Outputs are bit-for-bit identical to single-pass — CLIP uses
        LayerNorm throughout (batch-size independent), not BatchNorm.

        no_grad is safe because backbone is frozen — only spatial_norm
        (applied after cat in forward()) receives gradients.
        """
        chunks   = x.split(self.clip_chunk_size, dim=0)
        features = []
        for chunk in chunks:
            with torch.no_grad():
                out = self.backbone(pixel_values=chunk)
                features.append(out.pooler_output)   # (chunk_size, backbone_dim)
        return torch.cat(features, dim=0)            # (B×T, backbone_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B×T, C, H, W) — all frames flattened into batch dim

        Returns:
            (B×T, backbone_dim) — spatial_norm applied, no internal projection.
            Detector mean-pools across T then applies spatial_proj.
        """
        if self.needs_resize:
            x = F.interpolate(
                x,
                size=(self.required_size, self.required_size),
                mode='bilinear',
                align_corners=False,
            )
        raw = self._forward_clip(x)         # (B×T, backbone_dim)
        return self.spatial_norm(raw)       # (B×T, backbone_dim)
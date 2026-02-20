"""
networks/nesy_defake/foundation_models/temporal_feature_extractor.py
=====================================================================
Temporal Feature Extractor — GenD-style Training Regime

WHAT CHANGED AND WHY
---------------------
Previous version problems:

  1. _freeze_backbone() froze self.backbone.parameters() which includes
     the fc_norm LayerNorm living inside each backbone wrapper.
     fc_norm is freshly initialized — it MUST train. Freezing it meant
     the only new module in the backbone was frozen from step 1.

  2. fc_norm lived inside _VideoMAEBackbone / _PEVideoBackbone / _VJEPA2Backbone,
     meaning it was entangled with pretrained weights from the optimizer's
     perspective. Moving it to TemporalFeatureExtractor itself makes it
     always-trainable regardless of freeze state — no special casing needed.

  3. No get_trainable_params() — the optimizer's param group builder
     (train.py choose_optimizer) couldn't identify which backbone params
     should get the low LR (LayerNorms) vs which should stay frozen.

  4. unfreeze_backbone() flipped ALL backbone params on. In Phase 2 of
     training this would destroy pretrained representations. The correct
     Phase 2 for generalization is: unfreeze LayerNorms only (GenD regime),
     or optionally do a full unfreeze at a very low LR for fine-tuning.
     Both options are now exposed explicitly.

  5. VJEPA2 used attn_implementation="sdpa" as a hard requirement.
     Made conditional on PyTorch >= 2.0 with graceful fallback.

GenD-style regime applied to temporal (same logic as spatial):
  - Backbone (VideoMAE / PE / VJEPA2): FROZEN except LayerNorms
  - LayerNorms inside the backbone: TRAINABLE (adapt normalization
    statistics to deepfake domain without moving the feature manifold)
  - fc_norm (new, on TemporalFeatureExtractor): ALWAYS TRAINABLE
  - Detector's temporal_proj: ALWAYS TRAINABLE (lives on detector)

Config layout (foundation_models.temporal):
    name:             vjepa2                           # or videomae / perception_encoder
    model_path:       facebook/vjepa2-vitl-fpc64-256
    output_dim:       1024
    freeze_backbone:  true    # always recommended
    train_layernorms: true    # GenD regime — only LNs train in backbone
    normalization:
      mean: [0.485, 0.456, 0.406]
      std:  [0.229, 0.224, 0.225]
"""

import logging
import warnings
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers (same as spatial — keep in sync or move to a utils module)
# ---------------------------------------------------------------------------

def _collect_layernorm_params(module: nn.Module) -> List[nn.Parameter]:
    """
    Walk module tree and return all parameters belonging to a
    LayerNorm / GroupNorm / RMSNorm layer.
    """
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
    """
    Freeze every parameter in module, then selectively unfreeze
    LayerNorm / GroupNorm / RMSNorm weight and bias.

    Returns (total_params, trainable_params) for logging.
    """
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
# Backbone wrappers — fc_norm REMOVED from here, moved to TemporalFeatureExtractor
# ---------------------------------------------------------------------------

class _VideoMAEBackbone(nn.Module):
    """
    Wraps HuggingFace VideoMAEModel.

    Forward input : (B, T, C, H, W)  — T=16 expected
    Forward output: (B, hidden_dim)   — mean-pooled patch tokens (NO LayerNorm here)

    NOTE: fc_norm intentionally removed. It lives on TemporalFeatureExtractor
    so it is always trainable independent of freeze state.
    """
    EXPECTED_FRAMES = 16

    def __init__(self, model_path: str):
        super().__init__()
        from transformers import VideoMAEModel
        logger.info(f"[TemporalExtractor] Loading VideoMAE from: {model_path}")
        self.model = VideoMAEModel.from_pretrained(model_path)

        # Resolve actual hidden dim from model config (don't trust YAML blindly)
        self.hidden_dim = self.model.config.hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        outputs = self.model(pixel_values=x, output_hidden_states=False)
        # sequence output: (B, num_patches, hidden_dim)
        self._last_hidden = outputs.last_hidden_state
        return outputs.last_hidden_state.mean(dim=1)   # (B, hidden_dim)


class _PEVideoBackbone(nn.Module):
    """
    Wraps HuggingFace PeVideoEncoder (Perception Encoder, facebook/pe-av-large).

    Forward input : (B, T, C, H, W)  — T=8 recommended
    Forward output: (B, hidden_dim)   — mean-pooled token sequence (NO LayerNorm here)

    Requires: transformers >= 5.1.0
    """
    EXPECTED_FRAMES = 8

    def __init__(self, model_path: str):
        super().__init__()
        from transformers import PeVideoEncoder
        logger.info(f"[TemporalExtractor] Loading PerceptionEncoder from: {model_path}")
        self.model = PeVideoEncoder.from_pretrained(model_path)
        self.hidden_dim = self.model.config.hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values_videos=x)
        # last_hidden_state: (B, seq_len, hidden_dim)
        self._last_hidden = outputs.last_hidden_state
        return outputs.last_hidden_state.mean(dim=1)   # (B, hidden_dim)


class _VJEPA2Backbone(nn.Module):
    """
    Wraps V-JEPA 2 (AutoModel, FAIR/Meta).

    V-JEPA2 is pre-trained via joint-embedding predictive architecture on
    >1M hours of video with 3D-RoPE. Strongest choice when temporal motion
    dynamics are the primary signal (e.g. detecting unnatural facial motion).

    Forward input : (B, T, C, H, W)  — T=8 recommended at inference
    Forward output: (B, hidden_dim)   — mean-pooled encoder tokens (NO LayerNorm here)

    Checkpoints:
        'facebook/vjepa2-vitl-fpc64-256'  → hidden_dim 1024
        'facebook/vjepa2-vith-fpc64-256'  → hidden_dim 1280
        'facebook/vjepa2-vitg-fpc64-256'  → hidden_dim 1408
    """
    EXPECTED_FRAMES = 8

    def __init__(self, model_path: str):
        super().__init__()
        from transformers import AutoModel

        # sdpa (scaled dot-product attention) requires PyTorch >= 2.0.
        # Use it when available for memory efficiency; fall back gracefully.
        use_sdpa = (
            hasattr(torch.nn.functional, 'scaled_dot_product_attention')
            and torch.cuda.is_available()
        )
        kwargs = {}
        if use_sdpa:
            kwargs['attn_implementation'] = 'sdpa'
        else:
            logger.warning(
                "[TemporalExtractor/VJEPA2] PyTorch < 2.0 or no CUDA — "
                "falling back from SDPA to default attention. "
                "Upgrade to PyTorch >= 2.0 for better memory efficiency."
            )

        logger.info(f"[TemporalExtractor] Loading VJEPA2 from: {model_path}")
        self.model = AutoModel.from_pretrained(model_path, **kwargs)
        self.hidden_dim = self.model.config.hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # skip_predictor=True: run encoder only, skip the JEPA predictor head
        outputs = self.model(pixel_values_videos=x, skip_predictor=True)
        # last_hidden_state: (B, num_patches*T, hidden_dim)
        self._last_hidden = outputs.last_hidden_state
        return outputs.last_hidden_state.mean(dim=1)   # (B, hidden_dim)


# ---------------------------------------------------------------------------
# Public extractor
# ---------------------------------------------------------------------------

class TemporalFeatureExtractor(nn.Module):
    """
    Unified temporal feature extractor with GenD-style training regime.

    Architecture:
        backbone  — frozen pretrained video model (VideoMAE / PE / VJEPA2)
                    LayerNorms inside backbone are selectively unfrozen
        fc_norm   — nn.LayerNorm(hidden_dim), ALWAYS trainable
                    Normalizes backbone output before passing to detector's
                    temporal_proj. Equivalent to the fc_norm in VideoMAE's
                    original head — placed here so freeze state never touches it.

    The detector's temporal_proj (Linear → LayerNorm → GELU) is the only
    projection head. There is no internal projection here.

    Forward:
        input  : (B, T, C, H, W)
        output : (B, hidden_dim)   — fc_norm'd backbone features, raw dim
                 Detector's temporal_proj handles the mapping to proj_dim.
    """

    _BACKBONE_REGISTRY = {
        'videomae':           _VideoMAEBackbone,
        'perception_encoder': _PEVideoBackbone,
        'vjepa2':             _VJEPA2Backbone,
    }

    def __init__(self, config: dict):
        super().__init__()

        temporal_cfg         = config['foundation_models']['temporal']
        self.model_name      = temporal_cfg['name'].lower()
        model_path           = temporal_cfg['model_path']
        self.output_dim      = temporal_cfg['output_dim']
        self.freeze_backbone = temporal_cfg.get('freeze_backbone', True)
        self.train_layernorms = temporal_cfg.get('train_layernorms', True)

        if self.model_name not in self._BACKBONE_REGISTRY:
            raise ValueError(
                f"[TemporalExtractor] Unknown model '{self.model_name}'. "
                f"Choose from: {list(self._BACKBONE_REGISTRY.keys())}"
            )

        # ── Build backbone ────────────────────────────────────────────────
        backbone_cls    = self._BACKBONE_REGISTRY[self.model_name]
        self.backbone   = backbone_cls(model_path)
        self.hidden_dim = self.backbone.hidden_dim
        self.expected_frames = self.backbone.EXPECTED_FRAMES

        # Warn if YAML output_dim doesn't match the backbone's actual hidden dim.
        # A mismatch means either the YAML is wrong OR the detector's temporal_proj
        # will silently get the wrong input dim.
        if self.hidden_dim != self.output_dim:
            logger.warning(
                f"[TemporalExtractor] backbone hidden_dim={self.hidden_dim} "
                f"but config output_dim={self.output_dim}. "
                f"The detector's temporal_proj input will be {self.hidden_dim}. "
                f"Update foundation_models.temporal.output_dim in your YAML to "
                f"{self.hidden_dim} to match reality."
            )
            # Correct output_dim to backbone's actual value so the detector
            # can build its proj head with the right input size.
            self.output_dim = self.hidden_dim

        # ── fc_norm lives HERE — always trainable ─────────────────────────
        # This is the key structural fix: fc_norm is on TemporalFeatureExtractor,
        # not inside the backbone wrapper, so _freeze_backbone() can never
        # accidentally freeze it.
        self.fc_norm = nn.LayerNorm(self.hidden_dim)

        # ── Apply GenD-style freezing to backbone ─────────────────────────
        self._apply_freeze()

    # ------------------------------------------------------------------
    # Freeze / unfreeze
    # ------------------------------------------------------------------

    def _apply_freeze(self) -> None:
        """
        Apply the GenD-style freeze regime to self.backbone.
        fc_norm on self is intentionally NOT touched here.
        """
        if not self.freeze_backbone:
            logger.warning(
                "[TemporalExtractor] freeze_backbone=False: full backbone "
                "fine-tune enabled. This degrades cross-dataset generalization. "
                "Consider freeze_backbone=true + train_layernorms=true."
            )
            return   # nothing frozen — full fine-tune

        if self.train_layernorms:
            total, unfrozen = _freeze_all_except_layernorms(self.backbone)
            logger.info(
                f"[TemporalExtractor] GenD-style freeze ({self.model_name}): "
                f"{unfrozen:,} / {total:,} backbone params trainable "
                f"(LayerNorms only)"
            )
        else:
            # Hard freeze — backbone LNs do NOT train
            for p in self.backbone.parameters():
                p.requires_grad = False
            total = sum(p.numel() for p in self.backbone.parameters())
            logger.info(
                f"[TemporalExtractor] Hard freeze ({self.model_name}): "
                f"all {total:,} backbone params frozen. "
                f"Only fc_norm + detector projection head will train."
            )

    def unfreeze_layernorms(self) -> int:
        """
        Unfreeze LayerNorms only — correct Phase 2 behavior.
        Call this at epoch 5 (start of phase2) instead of full unfreeze.
        Returns the number of newly unfrozen parameters.
        """
        _, unfrozen = _freeze_all_except_layernorms(self.backbone)
        logger.info(
            f"[TemporalExtractor] Phase 2: unfreezing backbone LayerNorms "
            f"({unfrozen:,} params)"
        )
        return unfrozen

    def unfreeze_full(self, warn: bool = True) -> None:
        """
        Unfreeze ALL backbone parameters for full fine-tuning.
        Only use this at a very low LR (< 1e-5) to avoid destroying
        pretrained representations.
        """
        if warn:
            logger.warning(
                "[TemporalExtractor] unfreeze_full() called. "
                "Use a very low LR (≤ 1e-5) to avoid representation collapse. "
                "Consider unfreeze_layernorms() instead for better generalization."
            )
        for p in self.backbone.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------
    # Trainable parameter helpers (for optimizer param groups)
    # ------------------------------------------------------------------

    def get_trainable_params(self) -> List[nn.Parameter]:
        """
        Return backbone parameters that should be in the optimizer's
        'backbone_layernorms' param group (low LR).

        fc_norm parameters are NOT included here — they belong in the
        'projection_heads' group (standard LR) since they are new modules.
        The optimizer builder in train.py handles fc_norm separately via
        the always_trainable list.

        Usage in choose_optimizer():
            backbone_ln_params += temporal_extractor.get_trainable_params()
            # fc_norm handled separately:
            always_trainable += list(temporal_extractor.fc_norm.parameters())
        """
        if not self.freeze_backbone:
            # Full fine-tune: all backbone params
            return [p for p in self.backbone.parameters() if p.requires_grad]
        if self.train_layernorms:
            return _collect_layernorm_params(self.backbone)
        return []   # hard freeze: nothing from backbone

    def get_always_trainable_params(self) -> List[nn.Parameter]:
        """
        Return parameters that are ALWAYS trainable regardless of freeze state.
        Currently this is just fc_norm.
        Used by the optimizer builder to put fc_norm in the correct param group.
        """
        return list(self.fc_norm.parameters())

    def count_trainable_params(self) -> Tuple[int, int]:
        """Returns (trainable, total) including fc_norm."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        return trainable, total

    # ------------------------------------------------------------------
    # Frame-count resampling
    # ------------------------------------------------------------------

    def _resample_frames(self, x: torch.Tensor) -> torch.Tensor:
        """
        Uniformly sub-sample or repeat-pad x along the time dimension so
        the backbone always sees exactly self.expected_frames frames.

        Args:
            x: (B, T, C, H, W)
        Returns:
            (B, expected_frames, C, H, W)

        Emits a one-time warning — set clip_size=expected_frames in the
        dataset config to avoid this overhead entirely.
        """
        T = x.shape[1]
        E = self.expected_frames

        if T == E:
            return x

        warnings.warn(
            f"[TemporalExtractor/{self.model_name}] Input has {T} frames but "
            f"backbone expects {E}. Resampling on-the-fly. "
            f"Set clip_size={E} in your dataset config to avoid this overhead.",
            stacklevel=3,
        )

        if T > E:
            # Uniform sub-sample: E evenly-spaced frame indices
            indices = torch.linspace(0, T - 1, E, dtype=torch.long, device=x.device)
            return x[:, indices]
        else:
            # Repeat-pad: tile frames until we have enough, then trim
            repeats = (E + T - 1) // T
            return x.repeat(1, repeats, 1, 1, 1)[:, :E]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor,
            return_frame_features: bool = False):
        """
        Args:
            x: (B, T, C, H, W)
            return_frame_features: if True, also return the backbone's per-token
                hidden states for the temporal consistency loss.

        Returns:
            return_frame_features=False : (B, output_dim)
            return_frame_features=True  : tuple((B, output_dim), (B, N, D))
                where N = number of tokens/patches output by the backbone.
        """
        x   = self._resample_frames(x)      # (B, E, C, H, W)
        raw = self.backbone(x)              # (B, hidden_dim) — also sets backbone._last_hidden
        out = self.fc_norm(raw)             # (B, hidden_dim)

        if return_frame_features:
            # _last_hidden: (B, N, D) — token sequence from the backbone
            # Detach to prevent gradients flowing through the consistency loss
            # back into the backbone (we only want it to affect the projection
            # heads and fusion, not destabilize the frozen backbone LNs)
            frame_feats = self.backbone._last_hidden.detach()
            return out, frame_feats

        return out
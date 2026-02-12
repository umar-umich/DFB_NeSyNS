"""
Temporal Feature Extractor

Supports three backbones, selected via config['foundation_models']['temporal']['model_name']:

  'videomae'            – VideoMAEModel (HF transformers)
                          input key : pixel_values  (B, T, C, H, W)
                          expected T: 16
                          output dim: 768 (base) / 1024 (large)

  'perception_encoder'  – PeVideoEncoder (HF transformers ≥ 5.1)
                          input key : pixel_values_videos  (B, T, C, H, W)
                          expected T: flexible (8–16 recommended)
                          output dim: 1792  (pe-av-large default)

  'vjepa2'              – V-JEPA 2 AutoModel (HF transformers, FAIR / Meta)
                          input key : pixel_values_videos  (B, T, C, H, W)
                          expected T: 8  ← best for motion understanding
                          output dim: 1024 (ViT-L) / 1280 (ViT-H) / 1408 (ViT-g)
                          Strengths : 3D-RoPE, pre-trained on 1 M+ hours of video,
                                      SOTA on motion understanding & action anticipation.
                                      Recommended when motion/temporal dynamics matter.

Frame-count mismatch handling
──────────────────────────────
Each backbone has a preferred frame count (clip_size in your dataset config).
If the incoming clip has a different number of frames, the extractor will
uniformly sub-sample or repeat-pad on the fly so the backbone always sees
exactly `expected_frames` frames.  A one-time warning is printed when this
resampling is triggered so you can tune clip_size in the dataset config.
"""

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Internal backbone wrappers
# ──────────────────────────────────────────────────────────────────────────────

class _VideoMAEBackbone(nn.Module):
    """
    Wraps HuggingFace VideoMAEModel.

    Forward input : pixel_values  (B, T, C, H, W)
    Forward output: (B, D)  — global-average-pooled patch tokens, LayerNorm'd
    """

    EXPECTED_FRAMES = 16

    def __init__(self, model_path: str, output_dim: int):
        super().__init__()
        from transformers import VideoMAEModel
        self.model = VideoMAEModel.from_pretrained(model_path)
        self.fc_norm = nn.LayerNorm(output_dim)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        outputs = self.model(pixel_values=x, output_hidden_states=False)
        # outputs[0]: (B, num_patches, hidden_dim)
        return self.fc_norm(outputs[0].mean(dim=1))


class _PEVideoBackbone(nn.Module):
    """
    Wraps HuggingFace PeVideoEncoder.

    Forward input : pixel_values_videos  (B, T, C, H, W)
    Forward output: (B, D)  — global-average-pooled token sequence, LayerNorm'd

    Requires: transformers >= 5.1.0
    Checkpoint: e.g. 'facebook/pe-av-large'
    """

    EXPECTED_FRAMES = 8   # PE is flexible; 8 is the recommended sweet-spot

    def __init__(self, model_path: str, output_dim: int):
        super().__init__()
        from transformers import PeVideoEncoder
        self.model = PeVideoEncoder.from_pretrained(model_path)
        self.fc_norm = nn.LayerNorm(output_dim)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        outputs = self.model(pixel_values_videos=x)
        # last_hidden_state: (B, seq_len, hidden_dim)
        last_hidden = outputs.last_hidden_state
        return self.fc_norm(last_hidden.mean(dim=1))


class _VJEPA2Backbone(nn.Module):
    """
    Wraps HuggingFace V-JEPA 2 (AutoModel).

    V-JEPA 2 is pre-trained via joint-embedding predictive architecture on
    >1 M hours of internet video with 3D-RoPE for precise spatiotemporal
    position encoding.  It is the strongest available choice for tasks where
    temporal motion dynamics are the primary signal (e.g. detecting unnatural
    facial motion in deepfakes).

    Forward input : pixel_values_videos  (B, T, C, H, W)  — T=8 recommended
    Forward output: (B, D)  — global-average-pooled encoder tokens, LayerNorm'd

    Checkpoint examples (all on HF Hub):
        'facebook/vjepa2-vitl-fpc64-256'   → hidden_dim 1024
        'facebook/vjepa2-vith-fpc64-256'   → hidden_dim 1280
        'facebook/vjepa2-vitg-fpc64-256'   → hidden_dim 1408
        'facebook/vjepa2-vitg-fpc64-384'   → hidden_dim 1408  (higher resolution)

    Note: the 'fpc64' in the checkpoint names refers to the pre-training frame
    count (64), NOT the inference frame count.  At inference you can pass any
    reasonable T; T=8 balances quality and VRAM for a 16-frame deepfake clip.
    """

    EXPECTED_FRAMES = 8

    def __init__(self, model_path: str, output_dim: int):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(
            model_path,
            attn_implementation="sdpa",   # use efficient scaled-dot-product attention
        )
        self.fc_norm = nn.LayerNorm(output_dim)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        # skip_predictor=True: only run the encoder, not the JEPA predictor head
        outputs = self.model(pixel_values_videos=x, skip_predictor=True)
        # last_hidden_state: (B, num_patches*T, hidden_dim)
        return self.fc_norm(outputs.last_hidden_state.mean(dim=1))


# ──────────────────────────────────────────────────────────────────────────────
# Public extractor
# ──────────────────────────────────────────────────────────────────────────────

class TemporalFeatureExtractor(nn.Module):
    """
    Unified temporal feature extractor.

    Config layout expected under config['foundation_models']['temporal']:

        model_name       : 'videomae' | 'perception_encoder' | 'vjepa2'
        model_path       : HuggingFace repo id or local path
        output_dim       : int  – must match the chosen backbone's hidden size
        freeze_backbone  : bool
    """

    _BACKBONE_REGISTRY = {
        "videomae":           _VideoMAEBackbone,
        "perception_encoder": _PEVideoBackbone,
        "vjepa2":             _VJEPA2Backbone,
    }

    def __init__(self, config: dict):
        super().__init__()

        temporal_cfg = config["foundation_models"]["temporal"]
        self.model_name: str = temporal_cfg["name"].lower()
        model_path: str      = temporal_cfg["model_path"]
        self.output_dim: int = temporal_cfg["output_dim"]

        if self.model_name not in self._BACKBONE_REGISTRY:
            raise ValueError(
                f"Unknown temporal model '{self.model_name}'. "
                f"Choose from: {list(self._BACKBONE_REGISTRY.keys())}"
            )

        # Instantiate the selected backbone
        backbone_cls = self._BACKBONE_REGISTRY[self.model_name]
        self.backbone = backbone_cls(model_path, self.output_dim)
        self.expected_frames: int = self.backbone.EXPECTED_FRAMES

        if temporal_cfg.get("freeze_backbone", False):
            self._freeze_backbone()

    # ------------------------------------------------------------------ #
    #  Freeze / unfreeze                                                   #
    # ------------------------------------------------------------------ #

    def _freeze_backbone(self):
        """Freeze all backbone parameters (feature extraction only)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning (call after warm-up epochs)."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    # ------------------------------------------------------------------ #
    #  Frame-count resampling                                              #
    # ------------------------------------------------------------------ #

    def _resample_frames(self, x: torch.Tensor) -> torch.Tensor:
        """
        Uniformly sub-sample or repeat-pad x along the time dimension so that
        the backbone always sees exactly `self.expected_frames` frames.

        Args:
            x: (B, T, C, H, W)
        Returns:
            (B, expected_frames, C, H, W)
        """
        T = x.shape[1]
        E = self.expected_frames

        if T == E:
            return x

        warnings.warn(
            f"[TemporalFeatureExtractor/{self.model_name}] "
            f"Input has {T} frames but backbone expects {E}. "
            f"Resampling on-the-fly. Set clip_size={E} in your dataset config "
            f"to avoid this overhead.",
            stacklevel=3,
        )

        if T > E:
            # Uniform sub-sampling: pick E evenly spaced indices
            indices = torch.linspace(0, T - 1, E, dtype=torch.long, device=x.device)
            return x[:, indices]
        else:
            # Repeat-pad: tile until we have enough, then trim
            repeats = (E + T - 1) // T          # ceiling division
            x_tiled = x.repeat(1, repeats, 1, 1, 1)
            return x_tiled[:, :E]

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W) — batch of video clips, VideoMAE-normalised
               T should equal clip_size from dataset config (ideally == expected_frames)

        Returns:
            features: (B, output_dim)
        """
        x = self._resample_frames(x)         # (B, expected_frames, C, H, W)
        return self.backbone(x)              # (B, output_dim)
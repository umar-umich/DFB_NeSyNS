# frequency_feature_extractor.py
"""
networks/nesy_defake/foundation_models/frequency_feature_extractor.py
=====================================================================
Frequency Feature Extractor — CLIP backbone + Phase Map + GenD-style Training

Architecture:
    - Shared vision encoder injected from detector (same instance as spatial).
      No duplicate weight loading. GenD freeze applied once in detector.
    - SPSL-inspired phase map: phase-only FFT reconstruction encodes
      manipulation artifacts (blending seams, boundary discontinuities).
    - Optional DCT energy map (use_dct_energy: true): highlights 8×8 block
      energy concentrations from GAN artifacts and compression blocking.
      Recommended — helps disentangle compression artifacts from manipulation.
    - PhaseProjection (always trainable): projects 1ch or 2ch frequency
      signal to 3ch for the shared CLIP encoder.
    - freq_norm (always trainable): LayerNorm on encoder output.
    - Fix 2: chunked CLIP inference — phase map and PhaseProjection run on
      full batch (cheap), only the transformer backbone call is chunked.

Forward:
    input  : (B×T, C, H, W) — all frames, pre-normalized
    output : (B×T, hidden_dim) — freq_norm applied, no internal projection
             Detector mean-pools across T then applies frequency_proj.

Removed (legacy, not needed with shared CLIP encoder):
    - spsl / Xception path
    - srm_resnet path
    - SRM filter helpers
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
# Frequency signal computation
# ---------------------------------------------------------------------------

def compute_phase_map(img: torch.Tensor) -> torch.Tensor:
    """
    Phase-only reconstruction (SPSL: phase_without_amplitude).
    Encodes manipulation artifacts invisible in spatial domain.

    Args:  img: (B, C, H, W)
    Returns:    (B, 1, H, W) — real-valued, range approx [-1, 1]
    """
    gray          = img.mean(dim=1, keepdim=True)
    X             = torch.fft.fftn(gray, dim=(-2, -1))
    reconstructed = torch.exp(1j * torch.angle(X))
    return torch.real(torch.fft.ifftn(reconstructed, dim=(-2, -1)))


def compute_dct_energy_map(img: torch.Tensor) -> torch.Tensor:
    """
    Approximate DCT energy map via 8×8 block FFT magnitude.
    Highlights high-frequency energy from GAN artifacts and compression.

    Args:  img: (B, C, H, W)
    Returns:    (B, 1, H, W) — log-scale, normalized to [0, 1]
    """
    B, C, H, W = img.shape
    gray = img.mean(dim=1, keepdim=True)

    ph = (8 - H % 8) % 8
    pw = (8 - W % 8) % 8
    if ph > 0 or pw > 0:
        gray = F.pad(gray, (0, pw, 0, ph))
    _, _, H2, W2 = gray.shape

    blocks     = gray.view(B, 1, H2 // 8, 8, W2 // 8, 8)
    blocks     = blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
    freq       = torch.fft.rfft2(blocks)
    energy     = freq.abs().sum(dim=-1).sum(dim=-1)           # (B, 1, H2//8, W2//8)
    energy_map = energy.repeat_interleave(8, dim=-2).repeat_interleave(8, dim=-1)
    energy_map = energy_map[:, :, :H, :W]

    energy_map = torch.log1p(energy_map)
    mn = energy_map.flatten(2).min(dim=2)[0].unsqueeze(-1).unsqueeze(-1)
    mx = energy_map.flatten(2).max(dim=2)[0].unsqueeze(-1).unsqueeze(-1)
    return (energy_map - mn) / (mx - mn + 1e-8)


# ---------------------------------------------------------------------------
# Phase → 3-channel projection (always trainable)
# ---------------------------------------------------------------------------

class PhaseProjection(nn.Module):
    """
    Projects 1ch (phase only) or 2ch (phase + DCT energy) to 3ch for CLIP.

    Conv2d → BatchNorm2d → GELU
    ~3K–6K params. Always trainable — no pretrained weights to preserve.
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.GELU(),
        )
        nn.init.kaiming_normal_(self.proj[0].weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)   # (B, 3, H, W)


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

class FrequencyFeatureExtractor(nn.Module):
    """
    Frequency feature extractor: phase map → shared CLIP encoder → freq_norm.

    Always receives shared_encoder from detector. Freeze managed by detector.
    PhaseProjection and freq_norm are always trainable.
    """

    def __init__(self, config: dict, shared_encoder=None):
        super().__init__()

        freq_cfg              = config['foundation_models']['frequency']
        self.model_path       = freq_cfg.get('model_path', 'openai/clip-vit-large-patch14-336')
        self.output_dim       = freq_cfg['output_dim']
        self.freeze_backbone  = freq_cfg.get('freeze_backbone', True)
        self.train_layernorms = freq_cfg.get('train_layernorms', True)
        self.use_dct_energy   = freq_cfg.get('use_dct_energy', True)
        self.clip_chunk_size  = freq_cfg.get('clip_chunk_size', 4)
        # force_size: override encoder's native resolution for freq branch
        # Recommended: 224 even when using 336 encoder — phase map is a
        # derived signal, extra resolution adds cost without proportional gain
        self.force_size       = freq_cfg.get('force_size', 224)

        # ── Backbone: always shared ───────────────────────────────────────
        if shared_encoder is None:
            raise ValueError(
                "[FreqExtractor] shared_encoder is required. "
                "Build it in the detector via _build_shared_encoder() "
                "and pass it here. Standalone construction removed to "
                "prevent duplicate weight loading."
            )

        logger.info("[FreqExtractor] Using shared encoder from detector.")
        self.backbone       = shared_encoder
        spec                = ENCODER_REGISTRY[self.model_path]
        self.hidden_dim     = spec['dim']
        self.api_type       = spec['api']
        native_size         = spec['size']
        self._owns_backbone = False   # freeze managed by detector

        # Resize phase projection output if force_size != encoder native size
        self._native_size     = native_size   # store for logging

        self.needs_resize     = True   # always resize to native_size for encoder
        self.encoder_size     = native_size   # what encoder actually needs
        self.phase_input_size = self.force_size   # phase map computed at this size

        # ── Always-trainable modules ──────────────────────────────────────
        in_channels       = 2 if self.use_dct_energy else 1
        self.phase_proj   = PhaseProjection(in_channels=in_channels)
        self.freq_norm    = nn.LayerNorm(self.hidden_dim)

        if self.hidden_dim != self.output_dim:
            logger.warning(
                f"[FreqExtractor] hidden_dim={self.hidden_dim} != "
                f"output_dim={self.output_dim}. Correcting output_dim."
            )
            self.output_dim = self.hidden_dim

        logger.info(
            f"[FreqExtractor] Ready — path={self.model_path}, "
            f"hidden_dim={self.hidden_dim}, api={self.api_type}, "
            f"phase_input_size={self.phase_input_size} → encoder_size={self.encoder_size}, "
            f"dct_energy={self.use_dct_energy}, "
            f"clip_chunk_size={self.clip_chunk_size}, "
            f"PhaseProjection={in_channels}ch → 3ch"
        )
        trainable, total = self.count_trainable_params()
        logger.info(f"[FreqExtractor] Trainable: {trainable:,} / {total:,}")

    # ------------------------------------------------------------------
    # Trainable parameter helpers
    # ------------------------------------------------------------------

    def get_trainable_params(self) -> List[nn.Parameter]:
        """
        Backbone LN params — returns empty because shared encoder's
        LN params are collected once by detector for the optimizer group.
        """
        return []

    def get_always_trainable_params(self) -> List[nn.Parameter]:
        """
        phase_proj + freq_norm — always trainable, standard LR group.
        """
        return list(self.phase_proj.parameters()) + list(self.freq_norm.parameters())

    def count_trainable_params(self) -> Tuple[int, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        return trainable, total

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_freq_signal(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Per-sample zero-mean unit-variance normalization + clamp to [-3, 3].

        Required before PhaseProjection: raw FFT reconstruction spans ±100+,
        causing exploding activations with kaiming-initialized Conv2d weights.
        Per-sample (not batch) because frequency statistics vary per frame.
        """
        mean = x.flatten(1).mean(dim=1).view(-1, 1, 1, 1)
        std  = x.flatten(1).std(dim=1).view(-1, 1, 1, 1)
        return ((x - mean) / (std + eps)).clamp(-3.0, 3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B×T, C, H, W) — all frames flattened, pre-normalized

        Returns:
            (B×T, hidden_dim) — freq_norm applied, no internal projection.
            Detector mean-pools across T then applies frequency_proj.

        Fix 2: phase map + PhaseProjection run on full batch (cheap —
        no transformer layers). Only the CLIP backbone call is chunked.
        """
        # ── Step 1: frequency signals in float32 (FFT unstable at fp16) ──
        with torch.cuda.amp.autocast(enabled=False):
            x_f32 = x.float()
            phase = compute_phase_map(x_f32)                    # (B×T, 1, H, W)
            phase = self._normalize_freq_signal(phase)

            if self.use_dct_energy:
                dct_e   = compute_dct_energy_map(x_f32)         # (B×T, 1, H, W)
                dct_e   = self._normalize_freq_signal(dct_e)
                freq_in = torch.cat([phase, dct_e], dim=1)      # (B×T, 2, H, W)
            else:
                freq_in = phase                                  # (B×T, 1, H, W)

        # ── Step 2: PhaseProjection — full batch, lightweight ─────────────
        freq_in    = freq_in.to(x.dtype)
        clip_input = self.phase_proj(freq_in)                    # (B×T, 3, H, W)
        clip_input = self._normalize_freq_signal(clip_input)     # stabilize for CLIP

        if self.encoder_size != clip_input.shape[-1]:
            clip_input = F.interpolate(
                clip_input,
                size=(self.encoder_size, self.encoder_size),
                mode='bilinear',
                align_corners=False,
            )

        # ── Step 3: Fix 2 — chunked CLIP inference ────────────────────────
        # Peak mem: chunk_size × 257 × 1024 × 24 layers (fixed)
        # vs B×T × 257 × 1024 × 24 layers (unbounded without chunking)
        chunks   = clip_input.split(self.clip_chunk_size, dim=0)
        features = []
        for chunk in chunks:
            with torch.no_grad():   # backbone frozen — no grads needed
                out = self.backbone(pixel_values=chunk)
                features.append(out.pooler_output)               # (chunk_size, hidden_dim)
        raw = torch.cat(features, dim=0)                         # (B×T, hidden_dim)

        # ── Step 4: freq_norm — always trainable, gradients flow here ─────
        return self.freq_norm(raw)                               # (B×T, hidden_dim)
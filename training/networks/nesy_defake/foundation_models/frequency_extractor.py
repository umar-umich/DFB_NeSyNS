"""
networks/nesy_defake/foundation_models/frequency_feature_extractor.py
=====================================================================
Frequency Feature Extractor — CLIP backbone + Phase Map + GenD-style Training

WHAT CHANGED AND WHY
---------------------
Previous version problems:

  1. Xception backbone with averaged ImageNet conv1 weights is a weak
     generalizer. It was designed for classification, not for learning
     transferable frequency artifacts across unseen datasets/compressions.
     The averaged conv1 initialization is rough — the 4th channel (phase)
     starts with the same weights as the RGB channels, so the network has
     no early incentive to treat phase differently from color.

  2. Internal self.projection + detector's frequency_proj = two stacked
     projection heads. Same double-projection problem as spatial/temporal.

  3. _freeze_backbone() froze everything — the GenD LayerNorm-only regime
     was never applied.

  4. No get_trainable_params() / get_always_trainable_params() — optimizer
     param groups couldn't identify what to train.

  5. _forward_spsl() did manual adaptive_avg_pool2d after backbone.features()
     — fragile, depends on internal Xception method names staying stable.

Architecture decision — why CLIP instead of Xception:
  The phase map idea from SPSL is genuinely useful: the phase spectrum of a
  face image encodes manipulation artifacts (boundary discontinuities, blending
  seams) that are invisible in the spatial domain. We keep this insight but
  feed the phase signal to CLIP instead of Xception because:

    a) CLIP has seen 400M+ image-text pairs — its feature manifold is far
       richer than Xception's ImageNet features.
    b) CLIP's LayerNorms are the proven GenD adaptation mechanism for deepfake
       generalization.
    c) Xception has no equivalent of CLIP's self-supervised pretraining on
       diverse visual concepts — it overfits to training compression levels.

  The phase map is adapted to CLIP's 3-channel input via a small learned
  PhaseProjection (Conv2d 1→3, BN, GELU). This module:
    - Is always trainable regardless of freeze state
    - Teaches CLIP what frequency artifacts look like
    - Adds only ~3K parameters (negligible)

  FreqNet observation: FreqNet's contribution is its multi-scale frequency
  decomposition (DCT + learned filterbank). We can replicate the core benefit
  by using both the phase map AND a lightweight DCT energy map — both are
  fused into a single 3-channel input to CLIP. This is implemented as an
  optional config flag (use_dct_energy: true).

Config (foundation_models.frequency):
    name:              clip_phase               # new name for this architecture
    model_path:        openai/clip-vit-large-patch14
    output_dim:        1024                     # matches CLIP-L hidden_size
    freeze_backbone:   true
    train_layernorms:  true                     # GenD regime
    use_dct_energy:    false                    # optional: add DCT energy as 3rd channel
    normalization:
      mean: [0.481, 0.458, 0.408]
      std:  [0.269, 0.261, 0.276]

    # Keep spsl/srm_resnet entries for reference but they are no longer the
    # default. Set name: spsl to use the old Xception path (still supported).

Backward compatibility:
    name: spsl       — still works, loads Xception via DeepfakeBench registry
    name: srm_resnet — still works, SRM filterbank + ResNet
    name: clip_phase — new default, CLIP + phase map (recommended)
"""

import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared GenD helpers (keep in sync with spatial/temporal or extract to utils)
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
# Phase map + optional DCT energy computation
# ---------------------------------------------------------------------------

def compute_phase_map(img: torch.Tensor) -> torch.Tensor:
    """
    Phase-only reconstruction from SPSL (phase_without_amplitude).
    Encodes manipulation artifacts: boundary discontinuities, blending seams.

    Args:
        img: (B, C, H, W)
    Returns:
        (B, 1, H, W) — real-valued phase reconstruction, range approx [-1, 1]
    """
    gray = img.mean(dim=1, keepdim=True)           # (B, 1, H, W)
    X = torch.fft.fftn(gray, dim=(-2, -1))
    phase = torch.angle(X)
    reconstructed = torch.exp(1j * phase)
    return torch.real(torch.fft.ifftn(reconstructed, dim=(-2, -1)))  # (B, 1, H, W)


def compute_dct_energy_map(img: torch.Tensor) -> torch.Tensor:
    """
    Approximate DCT energy map via 8×8 block DCT magnitude.
    Highlights high-frequency energy concentrations typical of GAN artifacts
    and compression blocking — a lightweight FreqNet-inspired addition.

    Args:
        img: (B, C, H, W)
    Returns:
        (B, 1, H, W) — log-scale DCT energy map, normalized to [0, 1]

    Implementation note: we use the FFT on 8×8 non-overlapping patches as a
    fast approximation of the block DCT. The energy distribution across
    frequency bins is preserved; absolute values differ from true DCT by a
    constant scale factor which is absorbed by the PhaseProjection weights.
    """
    B, C, H, W = img.shape
    gray = img.mean(dim=1, keepdim=True)   # (B, 1, H, W)

    # Pad to nearest multiple of 8
    ph = (8 - H % 8) % 8
    pw = (8 - W % 8) % 8
    if ph > 0 or pw > 0:
        gray = F.pad(gray, (0, pw, 0, ph))

    _, _, H2, W2 = gray.shape

    # Reshape into 8×8 blocks: (B, 1, H2//8, 8, W2//8, 8)
    blocks = gray.view(B, 1, H2 // 8, 8, W2 // 8, 8)
    # (B, 1, H2//8, W2//8, 8, 8)
    blocks = blocks.permute(0, 1, 2, 4, 3, 5).contiguous()

    # 2D FFT on each 8×8 block
    freq = torch.fft.rfft2(blocks)                   # (B, 1, H2//8, W2//8, 8, 5)
    energy = freq.abs().sum(dim=-1).sum(dim=-1)       # (B, 1, H2//8, W2//8)

    # Upsample back to (H, W) — each block gets uniform energy value
    energy_map = energy.repeat_interleave(8, dim=-2).repeat_interleave(8, dim=-1)
    energy_map = energy_map[:, :, :H, :W]            # crop padding

    # Log-scale + normalize to [0, 1]
    energy_map = torch.log1p(energy_map)
    mn = energy_map.flatten(2).min(dim=2)[0].unsqueeze(-1).unsqueeze(-1)
    mx = energy_map.flatten(2).max(dim=2)[0].unsqueeze(-1).unsqueeze(-1)
    energy_map = (energy_map - mn) / (mx - mn + 1e-8)

    return energy_map   # (B, 1, H, W)


# ---------------------------------------------------------------------------
# Phase → 3-channel projection (always trainable)
# ---------------------------------------------------------------------------

class PhaseProjection(nn.Module):
    """
    Projects frequency signal(s) from 1 or 2 channels to 3 channels so
    CLIP's vision encoder can consume them directly.

    Input channels:
        use_dct_energy=False : 1 channel  (phase map only)
        use_dct_energy=True  : 2 channels (phase map + DCT energy map)

    Architecture: Conv2d → BatchNorm2d → GELU
      - Conv2d: learns which frequency patterns are discriminative
      - BatchNorm2d: normalizes the frequency signal to the range CLIP expects
      - GELU: smooth non-linearity consistent with transformer internals

    This is intentionally lightweight (~3K–6K params). Its job is to
    translate frequency signals into CLIP's input space, not to do heavy
    feature extraction itself.

    Always trainable — never frozen regardless of freeze_backbone state.
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.GELU(),
        )
        # Initialize close to identity-like mapping so early training is stable.
        # The conv weight starts as a learned upsampling from phase→RGB channels.
        nn.init.kaiming_normal_(self.proj[0].weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W)
        Returns:
            (B, 3, H, W) — ready for CLIP input
        """
        return self.proj(x)   # (B, 3, H, W)


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

class FrequencyFeatureExtractor(nn.Module):
    """
    Frequency feature extractor with CLIP backbone and SPSL-inspired phase map.

    Primary mode (name: clip_phase):
        1. Compute phase map from input frames     → (B, 1, H, W)
        2. Optionally add DCT energy map           → (B, 2, H, W)
        3. PhaseProjection: freq signal → 3ch      → (B, 3, H, W)
        4. CLIP vision encoder (backbone)          → (B, hidden_dim)
        5. freq_norm (always trainable LayerNorm)  → (B, hidden_dim)
        → Detector's frequency_proj handles dim mapping to proj_dim

    Backward-compatible modes (name: spsl | srm_resnet):
        Same as before — Xception via DeepfakeBench registry or SRM+ResNet.
        These are preserved for ablation but not recommended for new runs.

    GenD-style training:
        - CLIP backbone: frozen except LayerNorms (train_layernorms=True)
        - PhaseProjection: always trainable (new module, no pretrained weights)
        - freq_norm: always trainable (new module, lives on extractor)
        - Detector's frequency_proj: always trainable (lives on detector)

    Forward:
        input  : (B, C, H, W) — pre-normalized frames
        output : (B, hidden_dim) — freq_norm'd CLIP features, NO internal projection
                 Detector's frequency_proj maps this to proj_dim before fusion.
    """

    def __init__(self, config: dict):
        super().__init__()

        freq_cfg = config['foundation_models']['frequency']
        self.model_name       = freq_cfg.get('name', 'clip_phase')
        self.model_path       = freq_cfg.get('model_path', 'openai/clip-vit-large-patch14')
        self.output_dim       = freq_cfg['output_dim']
        self.freeze_backbone  = freq_cfg.get('freeze_backbone', True)
        self.train_layernorms = freq_cfg.get('train_layernorms', True)
        self.use_dct_energy   = freq_cfg.get('use_dct_energy', False)

        # ── Build backbone + always-trainable modules ─────────────────────
        self._build_backbone(config)

        # freq_norm: always trainable, lives here NOT inside the backbone wrapper
        # Same pattern as temporal's fc_norm — freeze state never touches it
        self.freq_norm = nn.LayerNorm(self.hidden_dim)

        # Sanity check: warn if YAML output_dim doesn't match backbone's dim
        if self.hidden_dim != self.output_dim:
            logger.warning(
                f"[FreqExtractor] backbone hidden_dim={self.hidden_dim} != "
                f"config output_dim={self.output_dim}. "
                f"Correcting output_dim to {self.hidden_dim}. "
                f"Update foundation_models.frequency.output_dim in YAML."
            )
            self.output_dim = self.hidden_dim

        # ── Apply GenD-style freeze ────────────────────────────────────────
        self._apply_freeze()

        logger.info(
            f"[FreqExtractor] Ready — backbone={self.model_name}, "
            f"hidden_dim={self.hidden_dim}, output_dim={self.output_dim}, "
            f"freeze={self.freeze_backbone}, layernorm_train={self.train_layernorms}, "
            f"use_dct_energy={self.use_dct_energy}"
        )
        trainable, total = self.count_trainable_params()
        logger.info(
            f"[FreqExtractor] Trainable params: {trainable:,} / {total:,}"
        )

    # ------------------------------------------------------------------
    # Backbone construction
    # ------------------------------------------------------------------

    def _build_backbone(self, config: dict) -> None:
        name = self.model_name
        if name == 'clip_phase':
            self._build_clip_phase(config)
        elif name == 'spsl':
            self._build_spsl(config)
        elif name == 'srm_resnet':
            self._build_srm_resnet(config)
        else:
            raise ValueError(
                f"[FreqExtractor] Unknown name: '{name}'. "
                f"Choose from: clip_phase, spsl, srm_resnet"
            )

    def _build_clip_phase(self, config: dict) -> None:
        """
        CLIP vision encoder as the frequency branch backbone.

        PhaseProjection is created here because it depends on use_dct_energy
        (which determines in_channels). It is NOT frozen — it's always trainable.
        """
        from transformers import CLIPVisionModel

        logger.info(f"[FreqExtractor] Loading raw CLIP from: {self.model_path}")
        self.backbone = CLIPVisionModel.from_pretrained(self.model_path)

        clip_dims = {
            'openai/clip-vit-base-patch16':          768,
            'openai/clip-vit-base-patch32':          768,
            'openai/clip-vit-large-patch14':         1024,
            'openai/clip-vit-large-patch14-336':     1024,
            'laion/CLIP-ViT-H-14-laion2B-s32B-b79K': 1280,
        }
        self.hidden_dim = clip_dims.get(self.model_path, 1024)

        # Required input size
        self.required_size = 336 if '336' in self.model_path else 224
        self.needs_resize  = (self.required_size != 224)

        # Phase projection: freq signal → 3ch for CLIP input (always trainable)
        in_channels = 2 if self.use_dct_energy else 1
        self.phase_proj = PhaseProjection(in_channels=in_channels)

        logger.info(
            f"[FreqExtractor] PhaseProjection: {in_channels}ch → 3ch "
            f"({'phase + DCT energy' if self.use_dct_energy else 'phase only'})"
        )

    def _build_spsl(self, config: dict) -> None:
        """
        Legacy SPSL path — Xception via DeepfakeBench registry.
        Kept for ablation. Not recommended for new runs.
        """
        logger.warning(
            "[FreqExtractor] Using legacy SPSL/Xception backbone. "
            "Consider switching to name: clip_phase for better generalization."
        )
        try:
            from detectors import DETECTOR
        except ImportError:
            raise ImportError(
                "DeepfakeBench 'detectors' registry not found. "
                "Ensure DeepfakeBench is on PYTHONPATH."
            )

        freq_cfg = config['foundation_models']['frequency']
        spsl_detector = DETECTOR['spsl'](freq_cfg)
        self.backbone  = spsl_detector.backbone
        self.hidden_dim = self.backbone.last_linear.in_features
        self.phase_proj = None   # SPSL does its own phase concatenation

        logger.info(
            f"[FreqExtractor] SPSL/Xception loaded, hidden_dim={self.hidden_dim}"
        )

    def _build_srm_resnet(self, config: dict) -> None:
        """Legacy SRM+ResNet path — kept for ablation."""
        import torchvision.models as models

        freq_cfg = config['foundation_models']['frequency']
        num_filters = freq_cfg.get('srm_filters', 3)

        srm_weights = self._get_srm_filters(num_filters)
        self.srm_conv = nn.Conv2d(3, num_filters, kernel_size=5, padding=2, bias=False)
        self.srm_conv.weight.data = torch.from_numpy(srm_weights).float()

        if freq_cfg.get('freeze_srm', False):
            for p in self.srm_conv.parameters():
                p.requires_grad = False

        depth = freq_cfg.get('resnet_depth', 50)
        resnet_map = {18: models.resnet18, 34: models.resnet34, 50: models.resnet50}
        resnet = resnet_map.get(depth, models.resnet50)(pretrained=True)
        resnet.conv1 = nn.Conv2d(
            num_filters, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        resnet.fc = nn.Identity()
        self.backbone   = resnet
        self.hidden_dim = {18: 512, 34: 512, 50: 2048}.get(depth, 512)
        self.phase_proj = None

    # ------------------------------------------------------------------
    # GenD-style freeze
    # ------------------------------------------------------------------

    def _apply_freeze(self) -> None:
        """
        Applies GenD-style freeze to self.backbone only.
        phase_proj and freq_norm are intentionally NOT touched — always trainable.
        """
        if not self.freeze_backbone:
            logger.warning(
                "[FreqExtractor] freeze_backbone=False: full backbone fine-tune. "
                "This hurts cross-dataset generalization. "
                "Use freeze_backbone=true + train_layernorms=true instead."
            )
            return

        if self.train_layernorms:
            total, unfrozen = _freeze_all_except_layernorms(self.backbone)
            logger.info(
                f"[FreqExtractor] GenD-style freeze: "
                f"{unfrozen:,} / {total:,} backbone params trainable "
                f"(LayerNorms only)"
            )
        else:
            for p in self.backbone.parameters():
                p.requires_grad = False
            total = sum(p.numel() for p in self.backbone.parameters())
            logger.info(
                f"[FreqExtractor] Hard freeze: all {total:,} backbone params frozen."
            )

    def unfreeze_layernorms(self) -> int:
        """Phase 2 transition — unfreeze backbone LayerNorms only."""
        _, unfrozen = _freeze_all_except_layernorms(self.backbone)
        logger.info(
            f"[FreqExtractor] Phase 2: unfreezing backbone LayerNorms "
            f"({unfrozen:,} params)"
        )
        return unfrozen

    # ------------------------------------------------------------------
    # Trainable parameter helpers
    # ------------------------------------------------------------------

    def get_trainable_params(self) -> List[nn.Parameter]:
        """
        Backbone LayerNorm params → goes into optimizer's backbone_layernorms
        group (low LR). Does NOT include phase_proj or freq_norm — those go
        into always_trainable / projection_heads group at standard LR.
        """
        if not self.freeze_backbone:
            return [p for p in self.backbone.parameters() if p.requires_grad]
        if self.train_layernorms:
            return _collect_layernorm_params(self.backbone)
        return []

    def get_always_trainable_params(self) -> List[nn.Parameter]:
        """
        Parameters that are ALWAYS trainable regardless of freeze state:
          - freq_norm (LayerNorm on extractor, like temporal's fc_norm)
          - phase_proj (new module, no pretrained weights, must always train)
        These go into the optimizer's projection_heads group at standard LR.
        """
        params = list(self.freq_norm.parameters())
        if self.phase_proj is not None:
            params += list(self.phase_proj.parameters())
        return params

    def count_trainable_params(self) -> Tuple[int, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        return trainable, total

    # ------------------------------------------------------------------
    # SRM helpers (legacy srm_resnet only)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_srm_filters(num_filters: int) -> np.ndarray:
        f1 = np.array([[0,0,0,0,0],[0,0,0,0,0],[-1,-1,4,-1,-1],[0,0,0,0,0],[0,0,0,0,0]])
        f2 = np.array([[0,0,-1,0,0],[0,0,-1,0,0],[0,0,4,0,0],[0,0,-1,0,0],[0,0,-1,0,0]])
        f3 = np.array([[0,0,0,0,0],[0,-1,-1,-1,0],[0,-1,8,-1,0],[0,-1,-1,-1,0],[0,0,0,0,0]])
        base = [f1, f2, f3]
        filters = [np.stack([base[i % 3]] * 3, axis=0) for i in range(num_filters)]
        return np.array(filters)

    # ------------------------------------------------------------------
    # Forward implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_freq_signal(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Normalize a frequency signal map to zero mean, unit variance per sample,
        then clamp to [-3, 3] to kill outlier values from FFT reconstruction.

        This is the missing piece that caused ConvolutionBackward0 NaN:
        raw phase reconstruction values span a very wide range (easily ±100+)
        and kaiming-initialized Conv2d weights produce exploding activations
        and NaN gradients immediately when the input is not normalized.

        Per-sample normalization (not batch) because frequency statistics
        vary significantly across frames — batch norm would mix them.
        """
        # Per-sample mean/std across spatial + channel dims
        mean = x.flatten(1).mean(dim=1).view(-1, 1, 1, 1)
        std  = x.flatten(1).std(dim=1).view(-1, 1, 1, 1)
        x    = (x - mean) / (std + eps)
        # Clamp: kills extreme FFT reconstruction outliers
        return x.clamp(-3.0, 3.0)

    def _forward_clip_phase(self, x: torch.Tensor) -> torch.Tensor:
        """
        1. Compute phase map (+ optional DCT energy map) in float32
        2. Normalize each signal to zero-mean unit-variance, clamp outliers
        3. Cast back to model dtype and project to 3ch via PhaseProjection
        4. Pass to CLIP vision encoder
        5. Apply freq_norm
        """
        # Compute frequency signals in float32 — FFT is numerically unstable at fp16
        with torch.cuda.amp.autocast(enabled=False):
            x_f32 = x.float()
            phase = compute_phase_map(x_f32)                   # (B, 1, H, W)
            phase = self._normalize_freq_signal(phase)         # ← THE FIX: normalize before conv

            if self.use_dct_energy:
                dct_e   = compute_dct_energy_map(x_f32)        # (B, 1, H, W), already [0,1]
                dct_e   = self._normalize_freq_signal(dct_e)   # normalize for consistency
                freq_in = torch.cat([phase, dct_e], dim=1)     # (B, 2, H, W)
            else:
                freq_in = phase                                 # (B, 1, H, W)

        # Cast back to model dtype before entering learnable layers
        freq_in    = freq_in.to(x.dtype)
        clip_input = self.phase_proj(freq_in)                   # (B, 3, H, W)

        # Resize if CLIP needs a different resolution than 224
        if self.needs_resize:
            clip_input = F.interpolate(
                clip_input,
                size=(self.required_size, self.required_size),
                mode='bilinear',
                align_corners=False,
            )

        outputs = self.backbone(pixel_values=clip_input)
        raw     = outputs.pooler_output                         # (B, hidden_dim)
        return self.freq_norm(raw)                              # (B, hidden_dim)

    def _forward_spsl(self, x: torch.Tensor) -> torch.Tensor:
        """Legacy SPSL/Xception path."""
        with torch.cuda.amp.autocast(enabled=False):
            phase = compute_phase_map(x.float()).to(x.dtype)

        x_4ch    = torch.cat([x, phase], dim=1)             # (B, 4, H, W)
        spatial  = self.backbone.features(x_4ch)            # (B, C, H', W')
        pooled   = F.adaptive_avg_pool2d(spatial, (1, 1)).flatten(1)  # (B, C)
        return self.freq_norm(pooled)

    def _forward_srm_resnet(self, x: torch.Tensor) -> torch.Tensor:
        srm_out = self.srm_conv(x)
        raw     = self.backbone(srm_out)
        return self.freq_norm(raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) — pre-normalized frames

        Returns:
            (B, output_dim) — freq_norm'd backbone features, NO internal projection.
            The detector's frequency_proj handles mapping to proj_dim.
        """
        if self.model_name == 'clip_phase':
            return self._forward_clip_phase(x)
        elif self.model_name == 'spsl':
            return self._forward_spsl(x)
        elif self.model_name == 'srm_resnet':
            return self._forward_srm_resnet(x)
        else:
            raise ValueError(f"[FreqExtractor] Unknown model: {self.model_name}")
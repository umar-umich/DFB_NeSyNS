"""
networks/nesy_defake/foundation_models/frequency_feature_extractor.py
=====================================================================
Frequency Feature Extractor — F3Net FAD + Adaptive Frequency Enhancement + CLIP

WHAT CHANGED AND WHY (v4 — Adaptive Frequency Enhancement)
------------------------------------------------------------
Previous version (v3, fad_clip single/multi) problems:

  v3 computed DCT → high-band filter → iDCT and fed ONLY the high-frequency
  band to CLIP. Two fatal issues:

  1. The high band alone is a RESIDUAL signal — it's the original image minus
     low+mid frequency content. For face-swap, the discriminative signal is
     the DIFFERENCE in high-frequency content between the forged face region
     and the background. A single high-band image gives CLIP no context for
     where the face boundary is. Result: FF++ AUC=0.68, CDF AUC=0.63.

  2. The input is already ImageNet-normalized (mean≈0, std≈1), so DCT
     coefficients are much smaller than F3Net expected (raw [0,255] pixels).
     The high-band filter (i+j > 112) captures almost no energy from
     normalized input → iDCT output is near-zero → per-sample normalization
     amplifies noise → CLIP sees noise, not frequency artifacts.

The fix — Adaptive Frequency Enhancement:

  Instead of REPLACING the image with a frequency band, we ENHANCE the
  original image with learnable frequency emphasis. The key insight:

    output = original + α * high_freq_residual

  where high_freq_residual = image - DCT_lowpass(image), computed by
  filtering OUT the low frequencies and keeping what remains.

  This way CLIP still sees a recognizable face (pretraining works) but
  with frequency artifacts AMPLIFIED (detection works). The learnable
  emphasis weight α starts small (0.1) and grows as training discovers
  which high-frequency patterns matter.

  We keep F3Net's learnable bandpass filters to define what "high frequency"
  means — the network learns the optimal cutoff for the specific forgery type.

  Additionally, we provide a multi-band mode where separate emphasis weights
  for low/mid/high bands allow the network to selectively amplify different
  frequency ranges.

Architecture:

  Input (B, 3, H, W) — pre-normalized frames
    → FADFrontEnd:
        DCT → learnable bandpass filters → iDCT → frequency bands
        single mode: output = original + α * high_band        (α learnable)
        multi mode:  output = original + Σ αᵢ * bandᵢ         (αᵢ learnable)
    → (B, 3, H, W) — enhanced image, still looks like a face
    → CLIP vision encoder [frozen except LayerNorms]
    → freq_norm [always trainable]
    → (B, hidden_dim)

Config (foundation_models.frequency):
    name:              fad_clip
    model_path:        openai/clip-vit-large-patch14
    output_dim:        1024
    freeze_backbone:   true
    train_layernorms:  true
    fad_mode:          single                 # 'single' or 'multi'
    fad_learnable:     true                   # learnable bandpass filters
    freq_emphasis_init: 0.1                   # initial emphasis weight α
    normalization:
      keep_raw:        true                   # dataset delivers [0,1] pixels
    clip_normalization:                       # applied inside model after FAD
      mean: [0.481, 0.458, 0.408]
      std:  [0.269, 0.261, 0.276]

    # Backward compatible: clip_phase, spsl, srm_resnet
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
# F3Net FAD modules (adapted from Qian et al., ECCV 2020)
# ---------------------------------------------------------------------------

def _dct_matrix(size: int) -> torch.Tensor:
    """
    Construct the DCT-II transformation matrix of given size.

    D[i,j] = sqrt(1/N) * cos((j+0.5)*pi*i/N)   if i == 0
             sqrt(2/N) * cos((j+0.5)*pi*i/N)   if i > 0

    Returns:
        (size, size) float tensor
    """
    mat = torch.zeros(size, size)
    for i in range(size):
        scale = np.sqrt(1.0 / size) if i == 0 else np.sqrt(2.0 / size)
        for j in range(size):
            mat[i, j] = scale * np.cos((j + 0.5) * np.pi * i / size)
    return mat


def _generate_filter_mask(start: float, end: float, size: int) -> torch.Tensor:
    """
    Generate a binary frequency band mask.
    Position (i, j) is included if start <= i + j <= end.
    i+j roughly corresponds to frequency magnitude in DCT space.
    """
    mask = torch.zeros(size, size)
    for i in range(size):
        for j in range(size):
            if start <= i + j <= end:
                mask[i, j] = 1.0
    return mask


def _norm_sigma(x: torch.Tensor) -> torch.Tensor:
    """
    Constrain learnable filter perturbations to [-0.5, 0.5] via sigmoid.

    Combined with base mask in {0, 1}, effective filter range is [-0.5, 1.5]
    BEFORE clamping. The clamp in FrequencyBandFilter.forward() then limits
    to [0, 1] — the filter can only ATTENUATE, never amplify.

    Previous version used range [-1, 1] giving effective filter [-1, 2].
    Values > 1 amplified large DCT coefficients (DC component ~1000+),
    causing iDCT output explosion → NaN in ConvolutionBackward0 at epoch 9.
    """
    return torch.sigmoid(x) - 0.5


class FrequencyBandFilter(nn.Module):
    """
    Single learnable frequency band filter from F3Net.

    Starts as a binary bandpass mask in DCT space and optionally adds a
    learnable perturbation to fine-tune which exact frequencies within
    the band are most discriminative.

    Always trainable when use_learnable=True — never frozen.
    """

    def __init__(self, size: int, band_start: float, band_end: float,
                 use_learnable: bool = True):
        super().__init__()
        self.use_learnable = use_learnable

        # Fixed binary bandpass mask
        self.register_buffer(
            'base', _generate_filter_mask(band_start, band_end, size)
        )

        if self.use_learnable:
            # Small init (0.01): sigmoid(0.01*randn) ≈ 0.5, so perturbation ≈ 0
            # Early training uses nearly the fixed bandpass mask
            self.learnable = nn.Parameter(torch.randn(size, size) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) DCT coefficients
        Returns:
            (B, C, H, W) filtered DCT coefficients
        """
        if self.use_learnable:
            filt = self.base + _norm_sigma(self.learnable)
        else:
            filt = self.base
        # CRITICAL: clamp to [0, 1] — filter must only attenuate, never amplify.
        # DCT DC component can be ~1000+; any filter value > 1 amplifies it,
        # causing iDCT explosion → NaN in CLIP's ConvolutionBackward0.
        filt = filt.clamp(0.0, 1.0)
        return x * filt


class FADFrontEnd(nn.Module):
    """
    Adaptive Frequency Enhancement front-end using F3Net's DCT decomposition.

    Expects RAW [0, 1] pixel input (dataset sets keep_raw: true for freq branch).
    This is critical because F3Net's DCT bandpass filters were designed for
    pixel-range coefficients. A 224×224 image in [0,1] gives a DC coefficient
    of ~15 — with ImageNet normalization it would be ~0.5, a 30× reduction
    that makes the bandpass filters ineffective.

    Pipeline:
        [0, 1] pixels                                          ← dataset delivers this
          → DCT → learnable bandpass filters → iDCT            ← F3Net's decomposition
          → original + α * filtered_band(s)                    ← adaptive emphasis
          → CLIP normalization: (x - mean) / std               ← for CLIP backbone
          → output for CLIP

    Modes:
        'single': output = orig + α * high_band
            Best for face-swap (blending boundary artifacts in high freq).

        'multi': output = orig + α_low * low + α_mid * mid + α_high * high
            Learns per-band emphasis. More flexible for mixed forgery types.

    Always trainable — never frozen regardless of freeze_backbone state.
    """

    def __init__(self, img_size: int = 224, mode: str = 'single',
                 use_learnable: bool = True, emphasis_init: float = 0.1,
                 clip_mean: list = None, clip_std: list = None):
        super().__init__()
        self.mode = mode
        self.img_size = img_size

        # ── CLIP normalization params (applied AFTER emphasis) ────────────
        # FAD works in raw [0,1] pixel space. Normalize for CLIP only at the end.
        if clip_mean is None:
            clip_mean = [0.481, 0.458, 0.408]
        if clip_std is None:
            clip_std = [0.269, 0.261, 0.276]
        self.register_buffer(
            '_clip_mean', torch.tensor(clip_mean).view(1, 3, 1, 1)
        )
        self.register_buffer(
            '_clip_std', torch.tensor(clip_std).view(1, 3, 1, 1)
        )

        # ── DCT matrices ─────────────────────────────────────────────────
        dct = _dct_matrix(img_size)
        self.register_buffer('_DCT', dct)
        self.register_buffer('_DCT_T', dct.t())

        # ── Bandpass filters (F3Net cutoffs) ─────────────────────────────
        self.filter_low = FrequencyBandFilter(
            img_size, 0, img_size / 2.82, use_learnable
        )
        self.filter_mid = FrequencyBandFilter(
            img_size, img_size / 2.82, img_size / 2, use_learnable
        )
        self.filter_high = FrequencyBandFilter(
            img_size, img_size / 2, img_size * 2, use_learnable
        )

        # ── Learnable emphasis weights ───────────────────────────────────
        # In forward(), we apply softplus(emphasis) to ensure positive values.
        # So we initialize the RAW parameter at inverse_softplus(emphasis_init)
        # so that the EFFECTIVE emphasis starts at emphasis_init.
        # inverse_softplus(x) = log(exp(x) - 1)
        raw_init = float(np.log(np.exp(emphasis_init) - 1.0)) if emphasis_init > 0 else -2.0
        if mode == 'single':
            self.emphasis = nn.Parameter(
                torch.full((3,), raw_init)
            )  # (3,) per-channel
        else:
            self.emphasis = nn.Parameter(
                torch.full((3, 3), raw_init)
            )  # (3 bands, 3 channels)

        n_filter_params = sum(
            p.numel() for f in [self.filter_low, self.filter_mid, self.filter_high]
            for p in f.parameters() if p.requires_grad
        )
        logger.info(
            f"[FADFrontEnd] mode={mode}, img_size={img_size}, "
            f"emphasis_init={emphasis_init}, "
            f"learnable_filter_params={n_filter_params:,}"
        )

    def _normalize_for_clip(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CLIP normalization: (x - mean) / std."""
        return (x - self._clip_mean) / self._clip_std

    def _dct_forward(self, x: torch.Tensor) -> torch.Tensor:
        """2D DCT: D @ x @ D^T"""
        return self._DCT @ x @ self._DCT_T

    def _idct_forward(self, x: torch.Tensor) -> torch.Tensor:
        """2D iDCT: D^T @ x @ D"""
        return self._DCT_T @ x @ self._DCT

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) — RAW [0, 1] pixels (NOT normalized)

        Returns:
            (B, 3, H, W) — frequency-enhanced, CLIP-normalized

        Pipeline:
            1. DCT on [0, 1] pixels (proper magnitude, F3Net's design point)
            2. Learnable bandpass filtering
            3. iDCT → filtered bands in [0, 1] space
            4. Additive emphasis: original + softplus(α) * band
            5. Clamp to valid pixel range
            6. Normalize for CLIP: (x - mean) / std
        """
        orig_dtype = x.dtype
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float()

            # Steps 1-3: DCT → bandpass → iDCT
            # Input is [0, 1] raw pixels → DCT coefficients have proper magnitude
            x_freq = self._dct_forward(x)

            low  = self._idct_forward(self.filter_low(x_freq))
            mid  = self._idct_forward(self.filter_mid(x_freq))
            high = self._idct_forward(self.filter_high(x_freq))

            # Step 4: Additive emphasis in pixel space
            # softplus constrains α to be positive (can only BOOST, not invert)
            # and has smooth gradients everywhere (unlike clamp/relu).
            # Max emphasis capped at 2.0 to prevent runaway amplification.
            if self.mode == 'single':
                alpha = torch.clamp(F.softplus(self.emphasis), max=2.0).view(1, 3, 1, 1)
                enhanced = x + alpha * high
            else:
                alpha = torch.clamp(F.softplus(self.emphasis), max=2.0)
                alpha_low  = alpha[0].view(1, 3, 1, 1)
                alpha_mid  = alpha[1].view(1, 3, 1, 1)
                alpha_high = alpha[2].view(1, 3, 1, 1)
                enhanced = x + alpha_low * low + alpha_mid * mid + alpha_high * high

            # Step 5: Clamp to reasonable pixel range before normalization.
            # Without this, large α * high_band can push values far outside [0,1],
            # and after CLIP normalization these become extreme values (±20+)
            # that cause NaN in the patch embedding's ConvolutionBackward0.
            enhanced = enhanced.clamp(-0.5, 1.5)

            # Step 6: Normalize for CLIP backbone
            out = self._normalize_for_clip(enhanced)

        return out.to(orig_dtype)


class BandAttention(nn.Module):
    """
    DEPRECATED — kept for checkpoint compatibility only.
    v4 uses per-band emphasis weights directly in FADFrontEnd instead.
    """

    def __init__(self, num_bands: int = 4, reduction: int = 2):
        super().__init__()
        self.num_bands = num_bands
        self.mlp = nn.Sequential(
            nn.Linear(num_bands * 3, num_bands * 3 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(num_bands * 3 // reduction, num_bands),
        )

    def forward(self, bands: torch.Tensor) -> torch.Tensor:
        B, K, C, H, W = bands.shape
        stats = bands.mean(dim=(-2, -1)).view(B, -1)
        weights = torch.softmax(self.mlp(stats), dim=1).view(B, K, 1, 1, 1)
        return (bands * weights).sum(dim=1)


# ---------------------------------------------------------------------------
# Legacy helpers (kept for backward-compatible modes)
# ---------------------------------------------------------------------------

def compute_phase_map(img: torch.Tensor) -> torch.Tensor:
    """Phase-only reconstruction from SPSL. Used by legacy clip_phase mode."""
    gray = img.mean(dim=1, keepdim=True)
    X = torch.fft.fftn(gray, dim=(-2, -1))
    phase = torch.angle(X)
    reconstructed = torch.exp(1j * phase)
    return torch.real(torch.fft.ifftn(reconstructed, dim=(-2, -1)))


class PhaseProjection(nn.Module):
    """Legacy 1-2ch → 3ch projection. Used by clip_phase mode only."""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.GELU(),
        )
        nn.init.kaiming_normal_(
            self.proj[0].weight, mode='fan_out', nonlinearity='relu'
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def compute_dct_energy_map(img: torch.Tensor) -> torch.Tensor:
    """Approximate DCT energy map. Used by clip_phase + use_dct_energy."""
    B, C, H, W = img.shape
    gray = img.mean(dim=1, keepdim=True)

    ph = (8 - H % 8) % 8
    pw = (8 - W % 8) % 8
    if ph > 0 or pw > 0:
        gray = F.pad(gray, (0, pw, 0, ph))

    _, _, H2, W2 = gray.shape
    blocks = gray.view(B, 1, H2 // 8, 8, W2 // 8, 8)
    blocks = blocks.permute(0, 1, 2, 4, 3, 5).contiguous()

    freq = torch.fft.rfft2(blocks)
    energy = freq.abs().sum(dim=-1).sum(dim=-1)

    energy_map = energy.repeat_interleave(8, dim=-2).repeat_interleave(8, dim=-1)
    energy_map = energy_map[:, :, :H, :W]

    energy_map = torch.log1p(energy_map)
    mn = energy_map.flatten(2).min(dim=2)[0].unsqueeze(-1).unsqueeze(-1)
    mx = energy_map.flatten(2).max(dim=2)[0].unsqueeze(-1).unsqueeze(-1)
    energy_map = (energy_map - mn) / (mx - mn + 1e-8)

    return energy_map


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

class FrequencyFeatureExtractor(nn.Module):
    """
    Frequency feature extractor with pluggable front-ends and CLIP backbone.

    Primary mode (name: fad_clip) [recommended]:
        FADFrontEnd (adaptive frequency enhancement) → CLIP → freq_norm

    How it works:
        FAD decomposes the image into frequency bands via DCT → learnable
        bandpass → iDCT, then ADDS the frequency bands back to the original
        image with learnable emphasis weights:
            output = original + α * high_freq_band
        CLIP sees a recognizable face with frequency artifacts amplified.

    Why this works:
        - CLIP's patch embedding sees natural image statistics (pretraining works)
        - Frequency artifacts are amplified, not isolated (detection works)
        - Learnable emphasis α grows from 0.1 as training finds what matters
        - The spatial branch already gives CLIP the raw image — this branch
          gives CLIP a DIFFERENT view with frequency emphasis

    Legacy modes (clip_phase, spsl, srm_resnet):
        Preserved for ablation comparisons.

    GenD-style training:
        - CLIP backbone: frozen except LayerNorms
        - FADFrontEnd (filters + emphasis): always trainable
        - freq_norm: always trainable
        - Detector's frequency_proj: always trainable (on detector)

    Forward:
        input  : (B, C, H, W) — pre-normalized frames
        output : (B, hidden_dim) — no internal projection
    """

    def __init__(self, config: dict):
        super().__init__()

        freq_cfg = config['foundation_models']['frequency']
        self.model_name       = freq_cfg.get('name', 'fad_clip')
        self.model_path       = freq_cfg.get('model_path', 'openai/clip-vit-large-patch14')
        self.output_dim       = freq_cfg['output_dim']
        self.freeze_backbone  = freq_cfg.get('freeze_backbone', True)
        self.train_layernorms = freq_cfg.get('train_layernorms', True)
        # FAD-specific
        self.fad_mode         = freq_cfg.get('fad_mode', 'single')
        self.fad_learnable    = freq_cfg.get('fad_learnable', True)
        self.emphasis_init    = freq_cfg.get('freq_emphasis_init', 0.1)
        # CLIP normalization — applied AFTER FAD enhancement, before CLIP backbone.
        # The dataset delivers raw [0,1] pixels to the frequency branch (keep_raw: true),
        # so FAD operates in proper pixel space. We normalize for CLIP inside the model.
        clip_norm = freq_cfg.get('clip_normalization', {})
        self.clip_mean = clip_norm.get('mean', [0.481, 0.458, 0.408])
        self.clip_std  = clip_norm.get('std',  [0.269, 0.261, 0.276])
        # Legacy
        self.use_dct_energy   = freq_cfg.get('use_dct_energy', False)

        self._build_backbone(config)
        self.freq_norm = nn.LayerNorm(self.hidden_dim)

        if self.hidden_dim != self.output_dim:
            logger.warning(
                f"[FreqExtractor] backbone hidden_dim={self.hidden_dim} != "
                f"config output_dim={self.output_dim}. "
                f"Correcting output_dim to {self.hidden_dim}."
            )
            self.output_dim = self.hidden_dim

        self._apply_freeze()

        logger.info(
            f"[FreqExtractor] Ready — backbone={self.model_name}, "
            f"hidden_dim={self.hidden_dim}, output_dim={self.output_dim}, "
            f"freeze={self.freeze_backbone}, layernorm_train={self.train_layernorms}"
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
        builders = {
            'fad_clip':   self._build_fad_clip,
            'clip_phase': self._build_clip_phase,
            'spsl':       self._build_spsl,
            'srm_resnet': self._build_srm_resnet,
        }
        if name not in builders:
            raise ValueError(
                f"[FreqExtractor] Unknown name: '{name}'. "
                f"Choose from: {list(builders.keys())}"
            )
        builders[name](config)

    def _build_fad_clip(self, config: dict) -> None:
        """
        NEW: F3Net FAD front-end + CLIP vision encoder.

        FAD decomposes image into frequency bands via DCT → learnable
        bandpass → iDCT. Output stays in image space so CLIP can use
        its pretrained features.
        """
        from transformers import CLIPVisionModel

        logger.info(f"[FreqExtractor] Loading CLIP from: {self.model_path}")
        self.backbone = CLIPVisionModel.from_pretrained(self.model_path)

        clip_dims = {
            'openai/clip-vit-base-patch16':           768,
            'openai/clip-vit-base-patch32':           768,
            'openai/clip-vit-large-patch14':          1024,
            'openai/clip-vit-large-patch14-336':      1024,
            'laion/CLIP-ViT-H-14-laion2B-s32B-b79K': 1280,
        }
        self.hidden_dim = clip_dims.get(self.model_path, 1024)
        self.required_size = 336 if '336' in self.model_path else 224

        self.fad_front_end = FADFrontEnd(
            img_size=self.required_size,
            mode=self.fad_mode,
            use_learnable=self.fad_learnable,
            emphasis_init=self.emphasis_init,
            clip_mean=self.clip_mean,
            clip_std=self.clip_std,
        )
        self.phase_proj = None

        fad_params = sum(p.numel() for p in self.fad_front_end.parameters()
                         if p.requires_grad)
        logger.info(
            f"[FreqExtractor] FADFrontEnd: {fad_params:,} trainable params, "
            f"mode={self.fad_mode}, learnable={self.fad_learnable}"
        )

    def _build_clip_phase(self, config: dict) -> None:
        """Legacy CLIP + phase map. Not recommended: phase maps are OOD for CLIP."""
        from transformers import CLIPVisionModel

        logger.warning(
            "[FreqExtractor] clip_phase: phase maps are outside CLIP's "
            "pretraining distribution. Consider fad_clip instead."
        )
        self.backbone = CLIPVisionModel.from_pretrained(self.model_path)
        clip_dims = {
            'openai/clip-vit-base-patch16':           768,
            'openai/clip-vit-base-patch32':           768,
            'openai/clip-vit-large-patch14':          1024,
            'openai/clip-vit-large-patch14-336':      1024,
            'laion/CLIP-ViT-H-14-laion2B-s32B-b79K': 1280,
        }
        self.hidden_dim = clip_dims.get(self.model_path, 1024)
        self.required_size = 336 if '336' in self.model_path else 224
        self.needs_resize = (self.required_size != 224)

        in_channels = 2 if self.use_dct_energy else 1
        self.phase_proj = PhaseProjection(in_channels=in_channels)
        self.fad_front_end = None

    def _build_spsl(self, config: dict) -> None:
        """Legacy SPSL/Xception via DeepfakeBench."""
        logger.warning("[FreqExtractor] Legacy SPSL. Consider fad_clip.")
        try:
            from detectors import DETECTOR
        except ImportError:
            raise ImportError(
                "DeepfakeBench 'detectors' registry not found."
            )
        freq_cfg = config['foundation_models']['frequency']
        spsl_detector = DETECTOR['spsl'](freq_cfg)
        self.backbone   = spsl_detector.backbone
        self.hidden_dim = self.backbone.last_linear.in_features
        self.phase_proj = None
        self.fad_front_end = None

    def _build_srm_resnet(self, config: dict) -> None:
        """Legacy SRM + ResNet."""
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
        self.fad_front_end = None

    # ------------------------------------------------------------------
    # GenD-style freeze
    # ------------------------------------------------------------------

    def _apply_freeze(self) -> None:
        """
        Freeze backbone only. fad_front_end, phase_proj, freq_norm
        are NEVER touched — always trainable.
        """
        if not self.freeze_backbone:
            logger.warning(
                "[FreqExtractor] freeze_backbone=False: full fine-tune. "
                "Use freeze_backbone=true + train_layernorms=true instead."
            )
            return

        if self.train_layernorms:
            total, unfrozen = _freeze_all_except_layernorms(self.backbone)
            logger.info(
                f"[FreqExtractor] GenD freeze: {unfrozen:,}/{total:,} "
                f"backbone params trainable (LayerNorms)"
            )
        else:
            for p in self.backbone.parameters():
                p.requires_grad = False
            total = sum(p.numel() for p in self.backbone.parameters())
            logger.info(f"[FreqExtractor] Hard freeze: {total:,} params frozen.")

    def unfreeze_layernorms(self) -> int:
        """Phase 2 transition — unfreeze backbone LayerNorms only."""
        _, unfrozen = _freeze_all_except_layernorms(self.backbone)
        logger.info(f"[FreqExtractor] Phase 2: unfroze {unfrozen:,} LN params")
        return unfrozen

    # ------------------------------------------------------------------
    # Trainable parameter helpers
    # ------------------------------------------------------------------

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Backbone LN params → low LR group."""
        if not self.freeze_backbone:
            return [p for p in self.backbone.parameters() if p.requires_grad]
        if self.train_layernorms:
            return _collect_layernorm_params(self.backbone)
        return []

    def get_always_trainable_params(self) -> List[nn.Parameter]:
        """Always-trainable params → standard LR group."""
        params = list(self.freq_norm.parameters())
        if self.fad_front_end is not None:
            params += list(self.fad_front_end.parameters())
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
        f1 = np.array([
            [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [-1, -1, 4, -1, -1],
            [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]
        ])
        f2 = np.array([
            [0, 0, -1, 0, 0], [0, 0, -1, 0, 0], [0, 0, 4, 0, 0],
            [0, 0, -1, 0, 0], [0, 0, -1, 0, 0]
        ])
        f3 = np.array([
            [0, 0, 0, 0, 0], [0, -1, -1, -1, 0], [0, -1, 8, -1, 0],
            [0, -1, -1, -1, 0], [0, 0, 0, 0, 0]
        ])
        base = [f1, f2, f3]
        filters = [np.stack([base[i % 3]] * 3, axis=0) for i in range(num_filters)]
        return np.array(filters)

    # ------------------------------------------------------------------
    # Forward implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_freq_signal(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Per-sample normalize + clamp ±3. Used by legacy clip_phase only."""
        mean = x.flatten(1).mean(dim=1).view(-1, 1, 1, 1)
        std  = x.flatten(1).std(dim=1).view(-1, 1, 1, 1)
        x    = (x - mean) / (std + eps)
        return x.clamp(-3.0, 3.0)

    def _forward_fad_clip(self, x: torch.Tensor) -> torch.Tensor:
        """
        FAD front-end → CLIP → freq_norm.

        Input is resized to match FAD's DCT matrix size (= CLIP's input size)
        BEFORE the DCT, since DCT matrix dimensions must match spatial dims.
        """
        _, _, H, W = x.shape
        if H != self.required_size or W != self.required_size:
            x = F.interpolate(
                x,
                size=(self.required_size, self.required_size),
                mode='bilinear',
                align_corners=False,
            )

        clip_input = self.fad_front_end(x)                      # (B, 3, H, W)
        outputs    = self.backbone(pixel_values=clip_input)
        raw        = outputs.pooler_output                       # (B, hidden_dim)
        return self.freq_norm(raw)

    def _forward_clip_phase(self, x: torch.Tensor) -> torch.Tensor:
        """Legacy CLIP + phase map path."""
        with torch.cuda.amp.autocast(enabled=False):
            x_f32 = x.float()
            phase = compute_phase_map(x_f32)
            phase = self._normalize_freq_signal(phase)
            if self.use_dct_energy:
                dct_e   = compute_dct_energy_map(x_f32)
                dct_e   = self._normalize_freq_signal(dct_e)
                freq_in = torch.cat([phase, dct_e], dim=1)
            else:
                freq_in = phase

        freq_in    = freq_in.to(x.dtype)
        clip_input = self.phase_proj(freq_in)

        if self.needs_resize:
            clip_input = F.interpolate(
                clip_input,
                size=(self.required_size, self.required_size),
                mode='bilinear',
                align_corners=False,
            )

        outputs = self.backbone(pixel_values=clip_input)
        return self.freq_norm(outputs.pooler_output)

    def _forward_spsl(self, x: torch.Tensor) -> torch.Tensor:
        """Legacy SPSL/Xception path."""
        with torch.cuda.amp.autocast(enabled=False):
            phase = compute_phase_map(x.float()).to(x.dtype)

        x_4ch  = torch.cat([x, phase], dim=1)
        spatial = self.backbone.features(x_4ch)
        pooled  = F.adaptive_avg_pool2d(spatial, (1, 1)).flatten(1)
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
            (B, output_dim) — freq_norm'd features, NO internal projection.
        """
        dispatch = {
            'fad_clip':   self._forward_fad_clip,
            'clip_phase': self._forward_clip_phase,
            'spsl':       self._forward_spsl,
            'srm_resnet': self._forward_srm_resnet,
        }
        if self.model_name not in dispatch:
            raise ValueError(f"[FreqExtractor] Unknown model: {self.model_name}")
        return dispatch[self.model_name](x)
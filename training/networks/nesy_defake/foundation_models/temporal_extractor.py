"""
networks/nesy_defake/foundation_models/temporal_feature_extractor.py
=====================================================================
Temporal Feature Extractor — GenD-style Training Regime

BACKBONE SUPPORT:
  videomae_v1   — MCG-NJU VideoMAE (original, HF VideoMAEModel)
                  HF class  : VideoMAEModel (standard, no custom code)
                  Input     : pixel_values (B, T, C, H, W) — HF handles internally
                  Frames    : 16
                  hidden_dim: 768 (base) / 1024 (large)
                  Paths     : 'MCG-NJU/videomae-large', etc.

  videomaev2    — OpenGVLab VideoMAEv2 (CVPR 2023, dual masking)  ← RECOMMENDED
                  HF class  : AutoModel + trust_remote_code=True (custom code)
                  Input     : pixel_values (B, C, T, H, W) — must permute internally
                  Frames    : 16
                  hidden_dim: 1408 (giant) / 1280 (huge) / 1024 (large) / 768 (base)
                  License   : cc-by-nc-4.0
                  Paths     : 'OpenGVLab/VideoMAEv2-giant'  ← current config
                              'OpenGVLab/VideoMAEv2-Huge'
                              'OpenGVLab/VideoMAEv2-Large'
                  KEY DIFFS vs videomae_v1:
                    1. AutoConfig must be loaded separately with trust_remote_code=True,
                       then passed to AutoModel.from_pretrained — custom modeling code
                    2. Dataloader delivers (B, T, C, H, W); model expects (B, C, T, H, W)
                       → _VideoMAEv2Backbone.forward() calls .permute(0,2,1,3,4)
                    3. hidden_dim read from config.hidden_size at runtime, not hardcoded
                    4. Token count for giant, T=16: (16//2) * (224//14)^2 = 2048 tokens
                       _pool_to_frame_features handles reduction to (B, T, D)

  perception_encoder — facebook/pe-av-large (PeVideoEncoder)
                  Frames    : 8
                  hidden_dim: 1792

  vjepa2        — facebook/vjepa2-vitl/h/g
                  Frames    : 8 (recommended)
                  hidden_dim: 1024 (vitl) / 1280 (vith) / 1408 (vitg)

FIXES (carried forward):
  Fix T1 — torch.no_grad() on frozen backbone (saves 2-4GB VRAM on ViT-G)
  Fix T2 — _pool_to_frame_features: (B, N*T, D) → (B, T, D)
  Fix T3 — T < 2 guard + fp32 cast in consistency loss
"""

import logging
import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
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
# Frame-level pooling helper (Fix T2)
# ---------------------------------------------------------------------------

def _pool_to_frame_features(
    token_seq: torch.Tensor,
    num_frames: int,
) -> torch.Tensor:
    """
    Reduce (B, N*T, D) → (B, T, D) by mean-pooling spatial patches per frame.

    For VideoMAEv2-giant with T=16:
      tokens = (16 // tubelet_size=2) * (224 // patch_size=14)^2 = 8 * 256 = 2048
      After pooling: (B, 16, 1408) — consistency matrix is (16, 16), not (2048, 2048)

    Falls back to adaptive_avg_pool1d when total_tokens % num_frames != 0
    (e.g. CLS token prepended in some architectures).
    """
    B, total_tokens, D = token_seq.shape

    if total_tokens % num_frames == 0:
        N = total_tokens // num_frames
        return token_seq.view(B, num_frames, N, D).mean(dim=2)
    else:
        logger.debug(
            f"[TemporalExtractor] token_seq {total_tokens} not divisible by "
            f"num_frames={num_frames} — adaptive pooling fallback."
        )
        pooled = F.adaptive_avg_pool1d(
            token_seq.permute(0, 2, 1),
            output_size=num_frames,
        )
        return pooled.permute(0, 2, 1)


# ---------------------------------------------------------------------------
# Backbone wrappers
# ---------------------------------------------------------------------------

class _VideoMAEv1Backbone(nn.Module):
    """
    Original VideoMAE (MCG-NJU) via HF VideoMAEModel.
    Standard HF class — no custom code, no permute needed.
    Kept for ablations / lower compute budget.
    """
    EXPECTED_FRAMES = 16

    def __init__(self, model_path: str):
        super().__init__()
        from transformers import VideoMAEModel
        logger.info(f"[TemporalExtractor] Loading VideoMAE v1 from: {model_path}")
        self.model = VideoMAEModel.from_pretrained(model_path)
        self.hidden_dim = self.model.config.hidden_size
        self._last_hidden: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W) — HF VideoMAEModel handles this natively
        outputs = self.model(pixel_values=x)
        self._last_hidden = outputs.last_hidden_state    # (B, num_patches, D)
        return outputs.last_hidden_state.mean(dim=1)     # (B, hidden_dim)


class _VideoMAEv2Backbone(nn.Module):
    """
    VideoMAEv2 (OpenGVLab, CVPR 2023) — dual masking on UnlabeledHybrid-1M.

    Three things that differ from VideoMAE v1 and will silently break if missed:

    [1] trust_remote_code=True — the model uses custom modeling_videomaev2.py.
        Load AutoModel directly WITHOUT pre-loading AutoConfig separately:

            model = AutoModel.from_pretrained(path, trust_remote_code=True)
        Pre-loading AutoConfig alone returns a generic PretrainedConfig (base HF fields only).
        Loading AutoModel directly makes it correctly load VideoMAEv2Config from the repo.

    [2] Pixel value channel order — model expects (B, C, T, H, W).
        Our dataloader delivers (B, T, C, H, W) uniformly across all branches.
        This wrapper permutes internally: x.permute(0, 2, 1, 3, 4).contiguous()
        The .contiguous() is required — permute returns a view and some CUDA
        kernels require contiguous memory.

    [3] hidden_dim from config — do NOT hardcode 1408 for giant.
        ViT-G = 1408, ViT-H = 1280, ViT-L = 1024, ViT-B = 768.
        Resolved via _resolve_hidden_dim() which tries known attribute names.

    [4] Return type — the custom forward() returns a raw Tensor, NOT a
        ModelOutput dataclass. There is no .last_hidden_state attribute.
        Shape is (B, num_tokens, hidden_dim). We store it as self._last_hidden
        ourselves and mean-pool to produce (B, hidden_dim).
    """
    EXPECTED_FRAMES = 16

    def __init__(self, model_path: str):
        super().__init__()
        from transformers import AutoModel

        logger.info(f"[TemporalExtractor] Loading VideoMAEv2 from: {model_path}")
        logger.info(
            "[TemporalExtractor/VideoMAEv2] Uses custom modeling code "
            "(trust_remote_code=True). Review OpenGVLab/VideoMAEv2-giant "
            "on HuggingFace before use in production."
        )

        # Load directly — do NOT pre-load AutoConfig separately.
        # When AutoConfig is loaded alone it returns a generic PretrainedConfig
        # (only has base HF fields like return_dict, use_cache etc.) rather than
        # the custom VideoMAEv2Config. Passing that shell config to AutoModel
        # causes it to ignore the actual model architecture fields entirely.
        # Loading AutoModel directly with trust_remote_code=True causes it to
        # correctly load and instantiate VideoMAEv2Config from the repo's
        # configuration_videomaev2.py before building the model.
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        # Resolve hidden_dim by probing the model's final LayerNorm weight.
        # This is architecture-agnostic and immune to custom config naming —
        # the final encoder norm always has shape (hidden_dim,) regardless of
        # what VideoMAEv2Config calls it.
        self.hidden_dim = self._probe_hidden_dim(self.model)
        self._last_hidden: Optional[torch.Tensor] = None

        logger.info(
            f"[TemporalExtractor/VideoMAEv2] hidden_dim={self.hidden_dim}, "
            f"expected_frames={self.EXPECTED_FRAMES}"
        )

    @staticmethod
    def _probe_hidden_dim(model: nn.Module) -> int:
        """
        Determine hidden_dim by inspecting the model's LayerNorm weights.

        Works on any ViT-style model regardless of config attribute naming.
        The final encoder LayerNorm has weight shape (hidden_dim,).
        We collect all LayerNorm weight sizes and return the largest — for a
        ViT encoder the final norm matches the full hidden dim, smaller norms
        belong to intermediate layers (e.g. attention sub-layer norms in some
        architectures are hidden_dim // num_heads which would be smaller).

        Returns the largest LayerNorm weight dimension found in the model.
        For VideoMAEv2-giant this will be 1408. For large: 1024. Base: 768.
        """
        ln_sizes = set()
        for mod in model.modules():
            if isinstance(mod, nn.LayerNorm) and mod.weight is not None:
                ln_sizes.add(mod.weight.shape[0])

        if not ln_sizes:
            raise RuntimeError(
                "[VideoMAEv2] Could not find any LayerNorm layers in model "
                "to probe hidden_dim. Inspect the model architecture manually."
            )

        hidden_dim = max(ln_sizes)
        logger.info(
            f"[VideoMAEv2] hidden_dim={hidden_dim} resolved by probing "
            f"LayerNorm weights. All LN sizes found: {sorted(ln_sizes)}"
        )
        return hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x from dataloader: (B, T, C, H, W)
        # VideoMAEv2 needs:  (B, C, T, H, W)
        x_in = x.permute(0, 2, 1, 3, 4).contiguous()

        # [4] The custom modeling_videomaev2.py forward() returns a raw Tensor
        # NOT a ModelOutput dataclass. Two possible output shapes:
        #   (B, num_tokens, hidden_dim) -- token sequence, pool over dim 1
        #   (B, hidden_dim)             -- already pooled internally
        #
        # Calling .mean(dim=1) on an already-pooled (B, D) tensor collapses
        # it to (B,) -- a scalar per sample -- causing fc_norm to fail with
        # "expected [*, 1408] but got [32]" (where 32 = batch size).
        # Detect by ndim and handle both cases.
        out = self.model(pixel_values=x_in)

        if out.ndim == 3:
            # (B, num_tokens, D) -- pool over token dim
            self._last_hidden = out
            return out.mean(dim=1)              # (B, hidden_dim)
        elif out.ndim == 2:
            # (B, D) -- model already pooled, use directly
            self._last_hidden = out.unsqueeze(1)  # (B, 1, D) for frame pool compat
            return out                            # (B, hidden_dim)
        else:
            raise RuntimeError(
                f"[VideoMAEv2] Unexpected output shape {out.shape}. "
                f"Expected (B, num_tokens, D) or (B, D)."
            )


class _PEVideoBackbone(nn.Module):
    """Perception Encoder (facebook/pe-av-large)."""
    EXPECTED_FRAMES = 8

    def __init__(self, model_path: str):
        super().__init__()
        from transformers import PeVideoEncoder
        logger.info(f"[TemporalExtractor] Loading PerceptionEncoder from: {model_path}")
        self.model = PeVideoEncoder.from_pretrained(model_path)
        self.hidden_dim = self.model.config.hidden_size
        self._last_hidden: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values_videos=x)
        self._last_hidden = outputs.last_hidden_state
        return outputs.last_hidden_state.mean(dim=1)

class _VJEPA2Backbone(nn.Module):
    """
    V-JEPA 2 (facebook/vjepa2-vitl/h/g).

    Tested variants:
        facebook/vjepa2-vitl-fpc64-256   hidden_dim=1024  res=256
        facebook/vjepa2-vitg-fpc64-384   hidden_dim=1408  res=384  ← active

    Known gotchas vs vitl:
      [1] pixel_values kwarg — vitg uses 'pixel_values', not 'pixel_values_videos'
          in some published HF revisions. We probe both and cache the working name.
      [2] skip_predictor kwarg — not present on all variants. Caught and retried.
      [3] hidden_dim — read from config.hidden_size but falls back to LayerNorm
          probe (same strategy as _VideoMAEv2Backbone) for robustness.
      [4] Output shape — some variants return (B, T, N, D) instead of
          (B, N*T, D). Flattened before storing _last_hidden.
    """
    EXPECTED_FRAMES = 16

    def __init__(self, model_path: str):
        super().__init__()
        from transformers import AutoModel

        use_sdpa = (
            hasattr(torch.nn.functional, 'scaled_dot_product_attention')
            and torch.cuda.is_available()
        )
        kwargs = {'attn_implementation': 'sdpa'} if use_sdpa else {}
        if not use_sdpa:
            logger.warning("[TemporalExtractor/VJEPA2] SDPA unavailable, falling back.")

        logger.info(f"[TemporalExtractor] Loading VJEPA2 from: {model_path}")
        self.model = AutoModel.from_pretrained(model_path, **kwargs)

        # [3] Robust hidden_dim: try config first, fall back to LN probe
        cfg_dim = getattr(self.model.config, 'hidden_size', None)
        if cfg_dim is not None:
            self.hidden_dim = cfg_dim
            logger.info(f"[VJEPA2] hidden_dim={self.hidden_dim} from config.hidden_size")
        else:
            self.hidden_dim = self._probe_hidden_dim(self.model)
            logger.info(f"[VJEPA2] hidden_dim={self.hidden_dim} from LayerNorm probe")

        self._last_hidden: Optional[torch.Tensor] = None

        # [1] Probe which pixel kwarg this checkpoint accepts
        self._pixel_kwarg = self._probe_pixel_kwarg()

        logger.info(
            f"[VJEPA2] hidden_dim={self.hidden_dim}, "
            f"pixel_kwarg='{self._pixel_kwarg}', "
            f"expected_frames={self.EXPECTED_FRAMES}"
        )

    @staticmethod
    def _probe_hidden_dim(model: nn.Module) -> int:
        """Same LayerNorm-max probe used in _VideoMAEv2Backbone."""
        ln_sizes = set()
        for mod in model.modules():
            if isinstance(mod, nn.LayerNorm) and mod.weight is not None:
                ln_sizes.add(mod.weight.shape[0])
        if not ln_sizes:
            raise RuntimeError("[VJEPA2] No LayerNorm found — cannot probe hidden_dim.")
        return max(ln_sizes)

    def _probe_pixel_kwarg(self) -> str:
        """
        Determine whether this checkpoint expects 'pixel_values' or
        'pixel_values_videos' by doing a minimal dry-run on CPU with a
        tiny dummy tensor. Caches result so real forward() has no overhead.
        """
        device = next(self.model.parameters()).device
        # Minimal viable input: B=1, T=2, C=3, H=16, W=16
        dummy = torch.zeros(1, 2, 3, 16, 16, device=device)

        for kwarg in ('pixel_values_videos', 'pixel_values'):
            try:
                with torch.no_grad():
                    for skip in (True, False):
                        try:
                            self.model(**{kwarg: dummy}, skip_predictor=skip)
                            logger.info(
                                f"[VJEPA2] pixel_kwarg='{kwarg}', "
                                f"skip_predictor={skip} works."
                            )
                            self._skip_predictor = skip
                            return kwarg
                        except TypeError:
                            continue
            except Exception:
                continue

        # Last resort — vitl default
        logger.warning(
            "[VJEPA2] Could not probe pixel kwarg — defaulting to "
            "'pixel_values_videos' with skip_predictor=True."
        )
        self._skip_predictor = True
        return 'pixel_values_videos'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W) — standard dataloader format

        kwargs = {self._pixel_kwarg: x}
        try:
            outputs = self.model(**kwargs, skip_predictor=self._skip_predictor)
        except TypeError:
            # skip_predictor not accepted by this variant
            outputs = self.model(**kwargs)

        hidden = outputs.last_hidden_state   # (B, N*T, D) or (B, T, N, D)

        # [4] Flatten (B, T, N, D) → (B, T*N, D) if needed
        if hidden.ndim == 4:
            B, T, N, D = hidden.shape
            hidden = hidden.view(B, T * N, D)

        self._last_hidden = hidden
        return hidden.mean(dim=1)   # (B, hidden_dim)


# ---------------------------------------------------------------------------
# Public extractor
# ---------------------------------------------------------------------------

class TemporalFeatureExtractor(nn.Module):
    """
    Unified temporal feature extractor with GenD-style training regime.

    Backbone is always frozen except LayerNorms (GenD regime).
    fc_norm is always trainable — the only connection from frozen backbone
    to the rest of the trainable graph.

    Forward:
        input  : (B, T, C, H, W)
        output : (B, hidden_dim)
        OR       tuple((B, hidden_dim), (B, T_actual, D))
                 when return_frame_features=True
    """

    _BACKBONE_REGISTRY = {
        'videomae_v1':        _VideoMAEv1Backbone,
        'videomaev2':         _VideoMAEv2Backbone,   # ← active
        'perception_encoder': _PEVideoBackbone,
        'vjepa2':             _VJEPA2Backbone,
    }

    def __init__(self, config: dict):
        super().__init__()

        temporal_cfg          = config['foundation_models']['temporal']
        self.model_name       = temporal_cfg['name'].lower()
        model_path            = temporal_cfg['model_path']
        self.output_dim       = temporal_cfg['output_dim']
        self.freeze_backbone  = temporal_cfg.get('freeze_backbone', True)
        self.train_layernorms = temporal_cfg.get('train_layernorms', True)

        if self.model_name not in self._BACKBONE_REGISTRY:
            raise ValueError(
                f"[TemporalExtractor] Unknown model '{self.model_name}'. "
                f"Choose from: {list(self._BACKBONE_REGISTRY.keys())}"
            )

        backbone_cls         = self._BACKBONE_REGISTRY[self.model_name]
        self.backbone        = backbone_cls(model_path)
        self.hidden_dim      = self.backbone.hidden_dim
        self.expected_frames = self.backbone.EXPECTED_FRAMES

        if self.hidden_dim != self.output_dim:
            logger.warning(
                f"[TemporalExtractor] backbone hidden_dim={self.hidden_dim} "
                f"but config output_dim={self.output_dim}. "
                f"Correcting → {self.hidden_dim}. Update YAML to suppress this."
            )
            self.output_dim = self.hidden_dim

        # fc_norm — always trainable, outside backbone, never frozen
        self.fc_norm = nn.LayerNorm(self.hidden_dim)

        self._apply_freeze()

        logger.info(
            f"[TemporalExtractor] Ready — model={self.model_name}, "
            f"hidden_dim={self.hidden_dim}, expected_frames={self.expected_frames}, "
            f"freeze={self.freeze_backbone}, train_lns={self.train_layernorms}"
        )
        trainable, total = self.count_trainable_params()
        logger.info(f"[TemporalExtractor] Trainable: {trainable:,} / {total:,}")

    # ------------------------------------------------------------------
    # Freeze / unfreeze
    # ------------------------------------------------------------------

    def _apply_freeze(self) -> None:
        if not self.freeze_backbone:
            logger.warning("[TemporalExtractor] freeze_backbone=False: full fine-tune.")
            return
        if self.train_layernorms:
            total, unfrozen = _freeze_all_except_layernorms(self.backbone)
            logger.info(
                f"[TemporalExtractor] GenD freeze ({self.model_name}): "
                f"{unfrozen:,} / {total:,} params trainable (LNs only)"
            )
        else:
            for p in self.backbone.parameters():
                p.requires_grad = False
            total = sum(p.numel() for p in self.backbone.parameters())
            logger.info(f"[TemporalExtractor] Hard freeze: {total:,} params frozen.")

    def unfreeze_layernorms(self) -> int:
        _, unfrozen = _freeze_all_except_layernorms(self.backbone)
        logger.info(f"[TemporalExtractor] Phase 2: LNs unfrozen ({unfrozen:,} params)")
        return unfrozen

    def unfreeze_full(self, warn: bool = True) -> None:
        if warn:
            logger.warning("[TemporalExtractor] unfreeze_full(): use LR ≤ 1e-5.")
        for p in self.backbone.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------
    # Trainable parameter helpers
    # ------------------------------------------------------------------

    def get_trainable_params(self) -> List[nn.Parameter]:
        if not self.freeze_backbone:
            return [p for p in self.backbone.parameters() if p.requires_grad]
        if self.train_layernorms:
            return _collect_layernorm_params(self.backbone)
        return []

    def get_always_trainable_params(self) -> List[nn.Parameter]:
        return list(self.fc_norm.parameters())

    def count_trainable_params(self) -> Tuple[int, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        return trainable, total

    # ------------------------------------------------------------------
    # Frame-count resampling
    # ------------------------------------------------------------------

    def _resample_frames(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        E = self.expected_frames
        if T == E:
            return x
        warnings.warn(
            f"[TemporalExtractor/{self.model_name}] Got {T} frames, "
            f"backbone expects {E}. Resampling on-the-fly. "
            f"Set clip_size={E} in dataset config.",
            stacklevel=3,
        )
        if T > E:
            idx = torch.linspace(0, T - 1, E, dtype=torch.long, device=x.device)
            return x[:, idx]
        else:
            reps = (E + T - 1) // T
            return x.repeat(1, reps, 1, 1, 1)[:, :E]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, return_frame_features: bool = False):
        """
        Args:
            x: (B, T, C, H, W)
            return_frame_features: return (B, T_actual, D) frame features
                for temporal consistency loss alongside the main output.

        Returns:
            (B, output_dim)                              — default
            ((B, output_dim), (B, T_actual, D))          — if return_frame_features
        """
        x_resampled = self._resample_frames(x)
        T_actual    = x_resampled.shape[1]

        # Fix T1: no_grad on frozen backbone saves the full autograd graph.
        # For VideoMAEv2-giant (1B params, 2048 tokens) this is ~3-5GB.
        # fc_norm below is outside no_grad — gradients re-enter here.
        if self.freeze_backbone:
            with torch.no_grad():
                raw = self.backbone(x_resampled)    # (B, hidden_dim)
        else:
            raw = self.backbone(x_resampled)

        out = self.fc_norm(raw)    # (B, hidden_dim) — always trainable

        if return_frame_features:
            # Fix T2: reduce (B, N*T, D) → (B, T, D)
            frame_feats = _pool_to_frame_features(
                self.backbone._last_hidden, num_frames=T_actual
            ).detach()
            return out, frame_feats

        return out
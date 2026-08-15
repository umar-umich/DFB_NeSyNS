"""P2a process residual — AEROBLADE-style frozen LDM first-stage cycle.

Ported from the validated pilot: DiCoME experiments/pilot_P2a/ae_operator.py
:: ReconstructionOperator. Construction is unchanged; only the surrounding packaging is
DISCERN's.

The claim (Ricker et al., CVPR 2024): an image produced *through* a latent diffusion model's
decoder already lies close to that decoder's range, so passing it through the autoencoder
again changes it very little. A camera image takes a larger hit. The residual is therefore a
generation-process signal -- "was this ever decoded by an LDM?" -- which is a different
question from the semantic one CLIP answers.

Instrument: `stabilityai/sdxl-vae`. The original plan named SD 2.1's VAE, but Stability no
longer hosts any SD 1.x/2.x repo; SDXL-VAE is the current first-stage AE from the same lab
and the same KL-regularised family (4-channel latent, f=8), so the operator's construction is
unchanged. That is a change of instrument, not of method, and it is recorded in the pilot's
notes.md.

CACHING HAZARD -- read before adding a cache
--------------------------------------------
The residual r_LDM(x) = D(x, x_hat) is a property of one particular image. If DISCERN applies
stochastic spatial/colour augmentation to x while the process branch loads a residual cached
from the *unaugmented* image, the two branches are no longer describing the same observation.
This exact clean-feature/augmented-image mismatch has bitten DISCERN before.

  * Cache process residuals ONLY when the process-branch input is deterministic and exactly
    aligned with the cached sample.
  * NEVER pair cached clean residuals with independently augmented visual inputs without
    explicitly validating that design.
  * For the first integration, reproduce P2a preprocessing exactly before optimising runtime.

`assert_cache_safe()` below makes the rule checkable rather than a comment someone has to
remember; no caching layer is provided in this phase, on purpose.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Statistics extracted from the reconstruction error map. Deliberately small and hand-named:
# this is the operator's entire interface to the evidential head, so it should be readable in
# the paper rather than an opaque pooled vector.
STAT_NAMES = (
    "mse_mean",       # overall reconstruction error
    "mse_std",        # spatial heterogeneity -- a composited region raises this
    "mse_max",        # worst patch
    "mse_p90",        # upper decile, robust version of the max
    "lpips_proxy",    # error weighted toward high spatial frequency
    "center_ratio",   # centre-vs-border error ratio: faces sit centre-frame after cropping
)
K = len(STAT_NAMES)

# CLIP's normalisation, which the datamodule has already applied. The AE expects [-1, 1], so
# the operator must undo one and apply the other; getting this wrong silently feeds the AE
# out-of-range input and the residual becomes meaningless rather than erroring.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class ProcessResidualOperator(nn.Module):
    """Frozen LDM first-stage AE cycle -> K reconstruction-error statistics per sample."""

    def __init__(self, vae_path: str | Path, resolution: int = 256):
        super().__init__()
        from diffusers import AutoencoderKL

        self.resolution = resolution
        self.n_stats = K
        self.stat_names = STAT_NAMES
        self.vae = AutoencoderKL.from_pretrained(str(vae_path))
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

        self.register_buffer("clip_mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer("clip_std", torch.tensor(CLIP_STD).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        """Keep the VAE in eval mode even when the enclosing model switches to train.

        The pilot froze the AE and asserted it; without this override a plain
        `model.train()` would flip the VAE's norm layers back into training statistics and
        the residual would drift over the run for reasons unrelated to the data.
        """
        super().train(mode)
        self.vae.eval()
        return self

    def _to_ae_space(self, images: torch.Tensor) -> torch.Tensor:
        """CLIP-normalised batch -> [-1, 1] at the AE's working resolution."""
        x = images * self.clip_std + self.clip_mean          # -> [0, 1]
        x = x.clamp(0, 1)
        if x.shape[-1] != self.resolution:
            x = F.interpolate(x, size=(self.resolution, self.resolution),
                              mode="bilinear", align_corners=False, antialias=True)
        return x * 2.0 - 1.0                                  # -> [-1, 1]

    @torch.no_grad()
    def reconstruct(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(input, reconstruction) in [-1, 1]. Deterministic: uses the posterior mode.

        Sampling the posterior would add noise unrelated to whether the image came from an
        LDM decoder, which is exactly the signal being measured.
        """
        x = self._to_ae_space(images)
        posterior = self.vae.encode(x).latent_dist
        recon = self.vae.decode(posterior.mode()).sample
        return x, recon

    @torch.no_grad()
    def error_map(self, images: torch.Tensor) -> torch.Tensor:
        x, recon = self.reconstruct(images)
        return (x - recon).pow(2).mean(dim=1, keepdim=True)   # (B, 1, H, W)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) CLIP-normalised -> (B, K) error statistics."""
        err = self.error_map(images)
        b = err.shape[0]
        flat = err.view(b, -1)

        mse_mean = flat.mean(dim=1)
        mse_std = flat.std(dim=1)
        mse_max = flat.max(dim=1).values
        mse_p90 = torch.quantile(flat.float(), 0.90, dim=1).to(flat.dtype)

        # High-frequency emphasis: an LDM decoder's fingerprint lives in fine detail, so a
        # Laplacian-filtered residual separates it from smooth global brightness error.
        lap = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                           device=err.device, dtype=err.dtype).view(1, 1, 3, 3)
        hf = F.conv2d(err, lap, padding=1).abs().view(b, -1).mean(dim=1)

        # Centre vs border: after face cropping the manipulated region is centre-frame, so a
        # composite should raise centre error relative to the untouched border.
        h, w = err.shape[-2:]
        ch, cw = h // 4, w // 4
        centre = err[:, :, ch:h - ch, cw:w - cw].reshape(b, -1).mean(dim=1)
        border_sum = flat.sum(dim=1) - err[:, :, ch:h - ch, cw:w - cw].reshape(b, -1).sum(dim=1)
        border_n = flat.shape[1] - (h - 2 * ch) * (w - 2 * cw)
        border = border_sum / max(1, border_n)
        centre_ratio = centre / (border + 1e-8)

        stats = torch.stack([mse_mean, mse_std, mse_max, mse_p90, hf, centre_ratio], dim=1)
        # log1p: the raw statistics span orders of magnitude across families, and an
        # unsquashed input would let one scale dominate the head's first layer.
        return torch.log1p(stats.clamp_min(0))

    def sanity_check(self) -> dict:
        frozen = all(not p.requires_grad for p in self.vae.parameters())
        return {"vae_params": sum(p.numel() for p in self.vae.parameters()),
                "vae_all_frozen": frozen,
                "vae_training_mode": self.vae.training,
                "n_stats": K}


def assert_cache_safe(process_input_is_deterministic: bool,
                      visual_input_is_augmented: bool) -> None:
    """Refuse the clean-residual / augmented-image pairing.

    Enforced rather than documented because the failure is invisible: training proceeds
    normally and the two branches simply describe different images, which shows up only as
    an unexplained gap between branch agreement in training and at inference.
    """
    if visual_input_is_augmented and not process_input_is_deterministic:
        raise ValueError(
            "process-branch caching is unsafe here: the visual input is stochastically "
            "augmented while the process residual is not deterministically aligned to the "
            "same sample. Either disable augmentation for the process path, recompute the "
            "residual per augmented sample, or explicitly validate the design first.")


def build_operator(vae_path: str | Path | None = None, resolution: int = 256
                   ) -> ProcessResidualOperator:
    """Construct the frozen operator. `vae_path` must point at a local sdxl-vae checkout."""
    if vae_path is None:
        raise ValueError(
            "process branch needs an explicit vae_path (a local stabilityai/sdxl-vae "
            "directory). Data paths are ASK-UMAR, so this is not guessed.")
    return ProcessResidualOperator(vae_path=vae_path, resolution=resolution)

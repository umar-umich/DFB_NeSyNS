"""
networks/nesy_defake/feature_augment.py
=======================================
Feature-space augmentation for the symbolic inputs (Phase-3 Task 1).

During TRAINING ONLY, perturbs the precomputed symbolic features before they
enter the concept / CCV branches:

  * additive zero-mean Gaussian noise, scaled per feature by that feature's
    FF++-train standard deviation (so each dimension is perturbed on its own
    scale), and
  * random per-dimension dropout (zeroing) as feature-level corruption.

At eval (`model.eval()`), or when disabled, it is a deterministic passthrough.

The per-feature std is FROZEN from the FF++-train split (see
scripts/compute_feature_stats.py). When that frozen vector is unavailable the
module falls back to the current batch's per-feature std — during training the
batch is FF++ data, so this stays within the FF++-only calibration rule.

This is a submodule of the detector, so `model.train()/eval()` toggles it
automatically via `self.training`.
"""
import torch
import torch.nn as nn


class FeatureAugment(nn.Module):
    def __init__(self, enabled: bool = False, gauss_std: float = 0.05,
                 feature_dropout: float = 0.1,
                 attrs_std=None, forensic_std=None):
        super().__init__()
        self.enabled = bool(enabled)
        self.gauss_std = float(gauss_std)
        self.feature_dropout = float(feature_dropout)
        # Frozen FF++-train per-feature std (or None → per-batch fallback).
        self.register_buffer(
            'attrs_std',
            None if attrs_std is None
            else torch.as_tensor(attrs_std, dtype=torch.float32))
        self.register_buffer(
            'forensic_std',
            None if forensic_std is None
            else torch.as_tensor(forensic_std, dtype=torch.float32))

    def _std(self, x, kind):
        frozen = getattr(self, f'{kind}_std', None)
        if frozen is not None and frozen.numel() == x.shape[1]:
            return frozen.to(x.device, x.dtype).unsqueeze(0)     # (1, F)
        # Fallback: per-batch std (FF++ data during training).
        return x.detach().std(dim=0, keepdim=True).clamp_min(1e-6)

    def forward(self, x: torch.Tensor, kind: str) -> torch.Tensor:
        """Augment (B, F) features. kind ∈ {'attrs', 'forensic'}.

        No-op unless enabled AND in training mode → eval is deterministic.
        """
        if not (self.enabled and self.training) or x is None:
            return x
        std = self._std(x, kind)
        if self.gauss_std > 0:
            x = x + torch.randn_like(x) * (self.gauss_std * std)
        if self.feature_dropout > 0:
            # per-(sample, dim) corruption; no inverted-dropout rescale — this
            # is input corruption, not a stochastic-depth style regularizer.
            keep = (torch.rand_like(x) >= self.feature_dropout).to(x.dtype)
            x = x * keep
        return x

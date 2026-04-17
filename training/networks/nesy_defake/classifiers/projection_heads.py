"""
Projection-head variants used by the spatial path.

Selected via config['projection_head']['type']:
  - standard   : Linear -> LayerNorm -> GELU (+ optional Dropout)
  - residual   : x + alpha * standard(x), learnable alpha (small-init)
  - bottleneck : Houlsby-style down/up adapter with skip (no alpha)
  - houlsby    : bottleneck + learnable residual scale (Houlsby et al., ICML 2019)
"""

import torch
import torch.nn as nn


def make_standard_projection(in_dim: int, out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    layers = [nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU()]
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class ResidualProjection(nn.Module):
    """output = x + alpha * standard(x). Early training ≈ linear probe."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0,
                 alpha_init: float = 0.1):
        super().__init__()
        self.proj = make_standard_projection(in_dim, out_dim, dropout)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * self.proj(x)


class BottleneckAdapter(nn.Module):
    """Down -> act -> up, skip-connected (Houlsby-style, no alpha)."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0,
                 bottleneck_ratio: int = 4):
        super().__init__()
        neck_dim = in_dim // bottleneck_ratio
        self.down = nn.Linear(in_dim, neck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(neck_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.dropout(self.up(self.act(self.down(x)))))


class HoulsbyAdapter(nn.Module):
    """Bottleneck + learnable residual scale (Houlsby et al., ICML 2019)."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0,
                 bottleneck_ratio: int = 4, alpha_init: float = 0.1):
        super().__init__()
        neck_dim = in_dim // bottleneck_ratio
        self.down = nn.Linear(in_dim, neck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(neck_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adapted = self.dropout(self.up(self.act(self.down(x))))
        return self.norm(x + self.alpha * adapted)


def build_projection_head(config: dict, in_dim: int, out_dim: int) -> nn.Module:
    """Factory. Reads config['projection_head'] and config['projection_dropout.spatial']."""
    ph_cfg = config.get('projection_head', {}) or {}
    ph_type = ph_cfg.get('type', 'standard')
    dropout = config.get('projection_dropout', {}).get('spatial', 0.0)
    alpha_init = ph_cfg.get('alpha_init', 0.1)
    bottleneck_ratio = ph_cfg.get('bottleneck_ratio', 4)

    if ph_type == 'residual':
        return ResidualProjection(in_dim, out_dim, dropout, alpha_init=alpha_init)
    if ph_type == 'bottleneck':
        return BottleneckAdapter(in_dim, out_dim, dropout, bottleneck_ratio=bottleneck_ratio)
    if ph_type == 'houlsby':
        return HoulsbyAdapter(in_dim, out_dim, dropout,
                              bottleneck_ratio=bottleneck_ratio,
                              alpha_init=alpha_init)
    return make_standard_projection(in_dim, out_dim, dropout)

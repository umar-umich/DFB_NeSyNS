"""
networks/nesy_defake/classifiers/sparse_autoencoder.py
======================================================
Module 4: Per-Branch BatchTopK Sparse Autoencoder

Decomposes polysemantic CLIP representations into monosemantic sparse features
that serve as causal variables for Module 3 (DAGMA-DCE causal discovery).

Architecture:
  Spatial CLIP (1024-d) --> SpatialSAE  --> ~128 sparse features --+
                                                                    |-> Z_sae
  Frequency CLIP (1024-d) -> FreqSAE --> ~128 sparse features ----+

Key design decisions:
  1. BatchTopK activation (Bussmann et al., NeurIPS 2024 workshop).
     Directly controls average sparsity without L1 coefficient tuning.
     Allows adaptive per-sample sparsity.
  2. Per-branch SAEs: spatial=appearance, frequency=artifacts.
  3. Unit-norm decoder columns (Anthropic convention).
  4. Dead feature recycling via auxiliary loss.
"""

import logging
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _batch_topk(x: torch.Tensor, k: int) -> torch.Tensor:
    """Select top (k * B) activations across flattened batch."""
    B, D = x.shape
    total_k = min(k * B, B * D)
    flat = x.view(-1)
    _, topk_idx = torch.topk(flat, total_k, sorted=False)
    mask = torch.zeros_like(flat)
    mask.scatter_(0, topk_idx, 1.0)
    return x * mask.view(B, D)


class BatchTopKActivation(nn.Module):
    """Training: batch-level top-k. Inference: learned threshold."""

    def __init__(self, target_k: int, ema_decay: float = 0.99):
        super().__init__()
        self.target_k = target_k
        self.ema_decay = ema_decay
        self.register_buffer("threshold", torch.tensor(0.0))
        self._threshold_initialized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(x)
        if self.training:
            result = _batch_topk(x, self.target_k)
            with torch.no_grad():
                nonzero = result[result > 0]
                if nonzero.numel() > 0:
                    ct = nonzero.min()
                    if not self._threshold_initialized:
                        self.threshold.fill_(ct.item())
                        self._threshold_initialized = True
                    else:
                        self.threshold.mul_(self.ema_decay).add_(ct * (1 - self.ema_decay))
            return result
        else:
            return x * (x > self.threshold).float()


class BranchSparseAutoencoder(nn.Module):
    """
    SAE for one CLIP branch.

    Encoder: x -> W_enc @ (x - b_dec) + b_enc -> BatchTopK -> z
    Decoder: z -> W_dec @ z + b_dec -> x_hat
    W_dec columns are unit-normalized.
    """

    def __init__(self, input_dim=1024, dict_size=4096, target_k=128,
                 dead_feature_window=5, aux_loss_coeff=1/32):
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.target_k = target_k
        self.dead_feature_window = dead_feature_window
        self.aux_loss_coeff = aux_loss_coeff

        self.W_enc = nn.Parameter(torch.empty(input_dim, dict_size))
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.W_dec = nn.Parameter(torch.empty(dict_size, input_dim))
        self.b_dec = nn.Parameter(torch.zeros(input_dim))

        self.activation = BatchTopKActivation(target_k=target_k)

        self.register_buffer("feature_last_active", torch.zeros(dict_size, dtype=torch.long))
        self.register_buffer("step_counter", torch.tensor(0, dtype=torch.long))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8))

    @torch.no_grad()
    def _normalize_decoder(self):
        """Project decoder columns to unit sphere. Call after optimizer step."""
        self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8))

    def encode(self, x):
        x_centered = x - self.b_dec
        pre_act = x_centered @ self.W_enc + self.b_enc
        return self.activation(pre_act)

    def decode(self, z):
        return z @ self.W_dec + self.b_dec

    def forward(self, x):
        # Clamp input to prevent BF16 overflow feeding into encoder
        x = torch.clamp(x, -100.0, 100.0)
        
        z = self.encode(x)
        x_hat = self.decode(z)

        residual = x - x_hat
        recon_loss = (residual ** 2).mean()
        with torch.no_grad():
            x_var = (x - x.mean(dim=0, keepdim=True)).pow(2).mean().clamp(min=1e-6)
        norm_recon = recon_loss / x_var

        aux_loss = torch.zeros(1, device=x.device)
        if self.training:
            self.step_counter += 1
            with torch.no_grad():
                active_mask = (z.sum(dim=0) > 0)
                self.feature_last_active[active_mask] = self.step_counter.item()

            dead_mask = (self.step_counter - self.feature_last_active) > self.dead_feature_window
            n_dead = dead_mask.sum().item()

            if n_dead > 0 and self.aux_loss_coeff > 0:
                # FIX: don't detach residual — let gradients flow
                dead_pre = (residual - self.b_dec) @ self.W_enc[:, dead_mask] + self.b_enc[dead_mask]
                dead_z = F.relu(dead_pre)
                k_dead = min(self.target_k, n_dead)
                if k_dead > 0 and dead_z.numel() > 0:
                    topk_v, topk_i = torch.topk(dead_z, k_dead, dim=1, sorted=False)
                    dead_sparse = torch.zeros_like(dead_z)
                    dead_sparse.scatter_(1, topk_i, topk_v)
                    dead_recon = dead_sparse @ self.W_dec[dead_mask]
                    # Normalize aux loss to same scale as recon_loss
                    aux_loss = (residual - dead_recon).pow(2).mean() / x_var.clamp(min=1e-6)

        total_loss = norm_recon + self.aux_loss_coeff * aux_loss

        with torch.no_grad():
            l0 = (z > 0).float().sum(dim=1).mean()
            n_dead_now = ((self.step_counter - self.feature_last_active) > self.dead_feature_window).sum()
            fvu = recon_loss / x_var

        info = {"recon_loss": recon_loss.detach(), "aux_loss": aux_loss.detach(),
                "l0": l0, "n_dead": n_dead_now, "fvu": fvu}
        return z, total_loss, info


class DualBranchSparseAutoencoder(nn.Module):
    """
    Module 4: Dual-branch SAE wrapper.
    Operates on raw CLIP features BEFORE projection heads.
    """

    def __init__(self, config):
        super().__init__()
        sae_cfg = config.get("sparse_features", {})
        fm_cfg = config.get("foundation_models", {})

        spatial_dim = fm_cfg.get("spatial", {}).get("output_dim", 1024)
        freq_dim = fm_cfg.get("frequency", {}).get("output_dim", 1024)

        sa_cfg = sae_cfg.get("sparse_autoencoder", {})
        expansion = sa_cfg.get("expansion_factor", 4)
        dict_size = sae_cfg.get("dict_size", expansion * spatial_dim)
        target_k = sae_cfg.get("target_k", sa_cfg.get("target_active_features", 128))
        dead_window = sae_cfg.get("dead_feature_window", 5)
        aux_coeff = sae_cfg.get("aux_loss_coeff", 1/32)
        self.normalize_inputs = sae_cfg.get("normalize_inputs", True)

        active = set(config.get("active_branches", ["spatial", "frequency"]))
        self.has_spatial = "spatial" in active
        self.has_frequency = "frequency" in active

        if self.has_spatial:
            self.spatial_sae = BranchSparseAutoencoder(
                spatial_dim, dict_size, target_k, dead_window, aux_coeff)
            logger.info(f"  Spatial SAE: {spatial_dim} -> {dict_size}, k={target_k}")
        if self.has_frequency:
            self.frequency_sae = BranchSparseAutoencoder(
                freq_dim, dict_size, target_k, dead_window, aux_coeff)
            logger.info(f"  Freq SAE: {freq_dim} -> {dict_size}, k={target_k}")

        self.target_k = target_k
        self.dict_size = dict_size
        self.output_dim = dict_size * (int(self.has_spatial) + int(self.has_frequency))

    @torch.no_grad()
    def normalize_decoder_weights(self):
        """Call after each optimizer step."""
        if self.has_spatial:
            self.spatial_sae._normalize_decoder()
        if self.has_frequency:
            self.frequency_sae._normalize_decoder()

    def forward(self, spatial_feat=None, frequency_feat=None):
        device = (spatial_feat if spatial_feat is not None else frequency_feat).device
        total_loss = torch.zeros(1, device=device)
        info = {}
        z_spatial = z_freq = None

        if self.has_spatial and spatial_feat is not None:
            x = (F.layer_norm(spatial_feat.float(), (spatial_feat.shape[-1],))
                .to(spatial_feat.dtype) if self.normalize_inputs else spatial_feat)
            z_spatial, s_loss, s_info = self.spatial_sae(x)
            total_loss = total_loss + s_loss
            info["spatial"] = s_info

        if self.has_frequency and frequency_feat is not None:
            x = (F.layer_norm(frequency_feat.float(), (frequency_feat.shape[-1],))
                .to(frequency_feat.dtype) if self.normalize_inputs else frequency_feat)
            z_freq, f_loss, f_info = self.frequency_sae(x)
            total_loss = total_loss + f_loss
            info["frequency"] = f_info

        return z_spatial, z_freq, total_loss, info

    def get_z_sae(self, z_spatial=None, z_freq=None):
        """Concatenate sparse features into Z_sae for causal module."""
        parts = [p for p in [z_spatial, z_freq] if p is not None]
        if not parts:
            raise RuntimeError("No SAE features available")
        return torch.cat(parts, dim=1)



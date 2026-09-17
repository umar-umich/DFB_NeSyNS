"""Temporal residual head: the Off-manifold residual transform of DISCERN's CLIP views, applied to a temporal feature.

Mirrors DISCERN_Ext exactly (src/model/core_model.py, src/modules/beta_vae_aligned.py, src/losses/aligned_vae_loss.py,
run1_auxedl config):
  f_t  = Linear(d -> 64)(LayerNorm(z))                  temporal feature (the CLIP view's 64-d projection)
  f_c  = beta-VAE(f_t)   64 -> 32 hidden -> latent 32 -> 32 -> 64, xavier init, mean at eval
  f_r  = f_t - f_c;  f_a = f_r - proj_{f_t}(f_r)        geometric view purification (orthogonal residual)
  temporal opinion  = EvidentialHead(LayerNorm(f_t)),   temporal-residual opinion = EvidentialHead(LayerNorm(f_a))
  fused             = Dempster-Shafer of the two
  loss = EDL(fused) + 0.5 * [EDL(temporal) + EDL(residual)] + 0.7 * [1.0 * (1 - cos(f_t.detach(), f_c)) + 2.0 * KL]

Residual OPERATORS (UMAR 2026-09-17: try alternatives to the beta-VAE):
  bvae       the transform above (DISCERN's off-manifold view)
  bvae_real  same, but the VAE/alignment loss sees GENUINE samples only -> a one-class manifold of real temporal features
  mrvae      DISCERN's MRVAEProjector (FiLM multi-rate VAE, beta log-uniform in training, beta=1 at eval); the residual
             view reads [orthogonal residual (64) ; 5-point rate response] = 69-d, as in the P1d pilot
  order      no VAE: f_c = the SAME projection applied to the shuffled-frame feature, so f_a = the part of the ordered
             embedding orthogonal to its order-free counterpart (isolates what frame order adds)
"""
from __future__ import annotations
import sys
import torch, torch.nn as nn
sys.path.insert(0, "/data/umar/Repos/DISCERN_Ext")
from src.modules.beta_vae_aligned import BetaVAEWithAlignment  # noqa: E402
from src.modules.evidential_head import EvidentialHead  # noqa: E402
from src.modules.mrvae_projector import MRVAEProjector, RATE_RESPONSE_K  # noqa: E402
from src.losses.aligned_vae_loss import aligned_vae_loss_func  # noqa: E402
from src.losses.evidential_loss import evidential_loss_discern_ext  # noqa: E402

FEATURE_DIM, LATENT_DIM, BETA_KLD, LAMBDA_ALIGN, LAMBDA_VAE, LAMBDA_AUX, K = 64, 32, 2.0, 1.0, 0.7, 0.5, 2


def purify(f_s: torch.Tensor, f_c: torch.Tensor) -> torch.Tensor:
    """f_r = f_s - f_c; remove its component along f_s (DISCERN _geometric_view_purification)."""
    f_r = f_s - f_c
    coef = (f_r * f_s).sum(1, keepdim=True) / (f_s * f_s).sum(1, keepdim=True).clamp_min(1e-8)
    return f_r - coef * f_s


def ds_pair(a1: torch.Tensor, a2: torch.Tensor) -> torch.Tensor:
    """Dempster-Shafer combination of two Dirichlet alphas (DISCERN _combine_two_opinions)."""
    S1, S2 = a1.sum(1, keepdim=True), a2.sum(1, keepdim=True)
    b1, b2, u1, u2 = (a1 - 1) / S1, (a2 - 1) / S2, K / S1, K / S2
    conflict = (b1.sum(1) * b2.sum(1) - (b1 * b2).sum(1)).view(-1, 1)
    b = (b1 * b2 + b1 * u2 + b2 * u1) / (1 - conflict)
    u = u1 * u2 / (1 - conflict)
    return b * (K / u) + 1


class TemporalResidualHead(nn.Module):
    def __init__(self, in_dim: int, operator: str = "bvae"):
        super().__init__()
        assert operator in ("bvae", "bvae_real", "mrvae", "order"), operator
        self.operator = operator
        self.inorm, self.proj = nn.LayerNorm(in_dim), nn.Linear(in_dim, FEATURE_DIM)
        r_dim = FEATURE_DIM + (RATE_RESPONSE_K if operator == "mrvae" else 0)
        if operator in ("bvae", "bvae_real"):
            self.vae = BetaVAEWithAlignment(FEATURE_DIM, LATENT_DIM)
        elif operator == "mrvae":
            self.vae = MRVAEProjector(FEATURE_DIM, LATENT_DIM)
        self.t_norm, self.r_norm = nn.LayerNorm(FEATURE_DIM), nn.LayerNorm(r_dim)
        self.t_head, self.r_head = EvidentialHead(FEATURE_DIM, K), EvidentialHead(r_dim, K)

    def forward(self, z: torch.Tensor, z_shuffled: torch.Tensor | None = None) -> dict:
        f_t = self.proj(self.inorm(z.float()))
        out = {"f_t": f_t}
        if self.operator == "order":
            f_c = self.proj(self.inorm(z_shuffled.float()))
            f_a = purify(f_t, f_c)
        else:
            _, mu, log_var, f_c = self.vae(f_t)
            f_a = purify(f_t, f_c)
            if self.operator == "mrvae":
                f_a = torch.cat([f_a, self.vae.rate_response(f_t)], dim=1)
            out.update({"f_c": f_c, "mu": mu, "log_var": log_var})
        a_t, a_r = self.t_head(self.t_norm(f_t)) + 1, self.r_head(self.r_norm(f_a)) + 1
        out.update({"t": a_t, "r": a_r, "fused": ds_pair(a_t, a_r)})
        return out

    def loss(self, out: dict, y: torch.Tensor, epoch: int, epochs: int) -> torch.Tensor:
        edl = lambda a: evidential_loss_discern_ext(a - 1, y, K, epoch, epochs).mean()
        total = edl(out["fused"]) + LAMBDA_AUX * (edl(out["t"]) + edl(out["r"]))
        if self.operator == "order":
            return total
        keep = (y == 0) if self.operator == "bvae_real" else torch.ones_like(y, dtype=torch.bool)
        if keep.sum() < 2:
            return total
        vae = aligned_vae_loss_func(out["f_t"][keep], out["f_c"][keep], out["mu"][keep], out["log_var"][keep], BETA_KLD, LAMBDA_ALIGN)
        return total + LAMBDA_VAE * vae


def to_pu(alpha: torch.Tensor):
    a = alpha.double()
    S = a.sum(1)
    return (a[:, 1] / S).cpu().numpy(), (K / S).cpu().numpy()

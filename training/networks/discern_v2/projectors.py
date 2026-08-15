"""Manifold projectors for DISCERN v2, ported from the validated DiCoME pilot code.

One interface, four implementations, selected by config string:

    beta_vae            original DiCoME beta-VAE           -- the manifold CONTROL
    deterministic_ae    P1a                                 -- is the stochastic latent load-bearing?
    beta_tcvae          P1b                                 -- Chen et al. TC decomposition
    mr_vae              P1d                                 -- beta-conditioned, exposes a rate response

Provenance (ported, not reimplemented from memory):
    beta_vae          <- DiCoME src/modules/beta_vae_aligned.py            :: BetaVAEWithAlignment
    deterministic_ae  <- DiCoME experiments/pilot_P1a/ae_operator.py       :: DeterministicAE
    beta_tcvae        <- DiCoME experiments/pilot_P1b/ae_operator.py       :: BetaTCVAE, tc_decomposed_kl
    mr_vae            <- DiCoME experiments/pilot_P1d/ae_operator.py       :: MRVAE, FiLM

P1c (WAE) is deliberately NOT ported: the Phase-1 instructions leave it unported unless
Track A unexpectedly selects it.

The shared contract is DiCoME's 4-tuple, kept verbatim so a projector swap is a config
change and nothing downstream moves:

    forward(semantic_feature) -> (z, mu, log_var, reconstruction)

Two properties let the branch stay projector-agnostic instead of special-casing MR-VAE:
`has_rate_response` and `extra_loss()`. Hardwiring MR-VAE's rate-distortion assumptions into
the branch would defeat the point of the swap -- Track A may well select P1a or P1b, and the
instructions warn that best-standalone != best-specialist for a gated architecture.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

# MR-VAE beta schedule. Spans two decades around DiCoME's shipped beta_kld = 2.0.
BETA_RANGE = (0.1, 10.0)
BETA_GRID = (0.1, 0.32, 1.0, 3.16, 10.0)   # log-spaced, 5 points
RATE_RESPONSE_K = len(BETA_GRID)

HIDDEN_DIM = 32   # the same literal across every DiCoME projector; kept for parity


class ManifoldProjector(nn.Module):
    """Base contract shared by every projector.

    Subclasses implement `forward` returning (z, mu, log_var, reconstruction). Anything
    projector-specific is exposed through the two hooks below so the branch never needs to
    know which projector it holds.
    """

    has_rate_response: bool = False

    def __init__(self, feature_dim: int, latent_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim

    def init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def extra_loss(self, semantic_feature: torch.Tensor, z: torch.Tensor, mu: torch.Tensor,
                   log_var: torch.Tensor, dataset_size: int) -> torch.Tensor | None:
        """Projector-specific loss term, or None when the standard KL path applies.

        Only beta_tcvae overrides this. Returning None means "the caller's usual
        KL/reconstruction objective is correct for me".
        """
        return None

    def rate_distortion_response(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} exposes no rate response; guard on has_rate_response")


class BetaVAEProjector(ManifoldProjector):
    """Original DiCoME beta-VAE. The manifold CONTROL the P1 variants were defined against.

    Ported verbatim from src/modules/beta_vae_aligned.py::BetaVAEWithAlignment. It is here
    so beta-VAE / AE / beta-TCVAE / MR-VAE all sit behind one interface and the comparison
    is not confounded by an interface difference.
    """

    def __init__(self, feature_dim: int, latent_dim: int):
        super().__init__(feature_dim, latent_dim)
        self.encoder_mlp = nn.Sequential(nn.Linear(feature_dim, HIDDEN_DIM), nn.ReLU())
        self.fc_mu = nn.Linear(HIDDEN_DIM, latent_dim)
        self.fc_log_var = nn.Linear(HIDDEN_DIM, latent_dim)
        self.decoder_mlp = nn.Sequential(
            nn.Linear(latent_dim, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, feature_dim))
        self.init_weights()

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std

    def forward(self, semantic_feature: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder_mlp(semantic_feature)
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var, self.decoder_mlp(z)


class DeterministicAEProjector(ManifoldProjector):
    """P1a: matched-capacity deterministic AE.

    Ported from experiments/pilot_P1a/ae_operator.py::DeterministicAE. Architecture matches
    the beta-VAE layer for layer minus fc_log_var and the reparameterisation.

    CRITICAL, carried over from the pilot: the KL weight must be 0.0 with this projector.
    With log_var = 0 and mu = z, DiCoME's KL term evaluates to 0.5 * sum(mu^2) -- an L2
    penalty on the latent, NOT zero. Leaving beta_kld at its default would silently
    substitute weight decay on the latent rather than removing the KL, so the config for
    this projector sets beta_kld: 0.0 and `assert_projector_config` enforces it.
    """

    def __init__(self, feature_dim: int, latent_dim: int):
        super().__init__(feature_dim, latent_dim)
        self.encoder_mlp = nn.Sequential(nn.Linear(feature_dim, HIDDEN_DIM), nn.ReLU())
        self.fc_latent = nn.Linear(HIDDEN_DIM, latent_dim)
        self.decoder_mlp = nn.Sequential(
            nn.Linear(latent_dim, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, feature_dim))
        self.init_weights()

    def forward(self, semantic_feature: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder_mlp(semantic_feature)
        z = self.fc_latent(hidden)
        # log_var is zeros purely to satisfy the tuple contract; with beta_kld = 0.0 it is
        # multiplied out of the loss entirely.
        return z, z, torch.zeros_like(z), self.decoder_mlp(z)


class BetaTCVAEProjector(ManifoldProjector):
    """P1b: beta-TCVAE (Chen et al., NeurIPS 2018).

    Ported from experiments/pilot_P1b/ae_operator.py. Architecture is IDENTICAL to the
    beta-VAE -- same parameter count -- and only the objective changes:

        KL(q(z|x) || p(z)) = MI(x;z) + TC(z) + dimension-wise KL
        beta-VAE   penalises all three by beta
        beta-TCVAE penalises only TC:   MI + beta*TC + DWKL

    The sampled `z` is stashed on the module because TC estimation needs the *same* sample
    that produced the reconstruction, not a fresh draw.
    """

    def __init__(self, feature_dim: int, latent_dim: int):
        super().__init__(feature_dim, latent_dim)
        self.encoder_mlp = nn.Sequential(nn.Linear(feature_dim, HIDDEN_DIM), nn.ReLU())
        self.fc_mu = nn.Linear(HIDDEN_DIM, latent_dim)
        self.fc_log_var = nn.Linear(HIDDEN_DIM, latent_dim)
        self.decoder_mlp = nn.Sequential(
            nn.Linear(latent_dim, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, feature_dim))
        self.last_z: torch.Tensor | None = None
        self.beta_tc: float = 2.0   # set from config; reuses beta_kld rather than adding a knob
        self.init_weights()

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std

    def forward(self, semantic_feature: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder_mlp(semantic_feature)
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        z = self.reparameterize(mu, log_var)
        self.last_z = z
        return z, mu, log_var, self.decoder_mlp(z)

    def extra_loss(self, semantic_feature, z, mu, log_var, dataset_size):
        return tc_decomposed_kl(z, mu, log_var, dataset_size, self.beta_tc)


def _log_gaussian(z: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """Element-wise Gaussian log-density; broadcasts for the (B,B,D) pairwise matrix."""
    return -0.5 * (math.log(2 * math.pi) + log_var + (z - mu).pow(2) / log_var.exp())


def tc_decomposed_kl(z: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor,
                     dataset_size: int, beta_tc: float) -> torch.Tensor:
    """MI + beta_tc * TC + dimension-wise KL, via minibatch-weighted sampling.

    Ported verbatim from experiments/pilot_P1b/ae_operator.py::tc_decomposed_kl.
    """
    batch_size = z.size(0)

    log_q_z_given_x = _log_gaussian(z, mu, log_var).sum(dim=1)          # (B,)
    log_p_z = (-0.5 * (math.log(2 * math.pi) + z.pow(2))).sum(dim=1)    # (B,)

    # mat[i, j, d] = log q(z_i,d | x_j)
    mat = _log_gaussian(z.unsqueeze(1), mu.unsqueeze(0), log_var.unsqueeze(0))

    # Minibatch-weighted sampling constant (Chen et al. eq. S6).
    log_mn = math.log(batch_size * dataset_size)
    log_q_z = torch.logsumexp(mat.sum(dim=2), dim=1) - log_mn
    log_prod_q_z = (torch.logsumexp(mat, dim=1) - log_mn).sum(dim=1)

    mutual_info = (log_q_z_given_x - log_q_z).mean()
    total_correlation = (log_q_z - log_prod_q_z).mean()
    dimension_wise_kl = (log_prod_q_z - log_p_z).mean()

    return mutual_info + beta_tc * total_correlation + dimension_wise_kl


class FiLM(nn.Module):
    """Per-feature scale and shift from log beta (Perez et al., AAAI 2018)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Linear(1, 2 * hidden_dim)

    def forward(self, h: torch.Tensor, log_beta: torch.Tensor) -> torch.Tensor:
        gamma, delta = self.net(log_beta).chunk(2, dim=-1)
        return h * (1.0 + gamma) + delta


class MRVAEProjector(ManifoldProjector):
    """P1d: Multi-Rate VAE. One model spanning a range of beta.

    Ported from experiments/pilot_P1d/ae_operator.py::MRVAE. beta enters via FiLM modulation
    (multiplicative, because the claim is that beta *rescales* how much information
    survives -- a shift alone cannot express that).

    This is the only projector exposing a per-sample rate response R(x): the K-point curve
    of reconstruction distortion as the latent is squeezed harder. A3 tests whether that
    curve carries incremental forensic information; `has_rate_response` lets the branch
    export it without any other projector needing to know it exists.
    """

    has_rate_response = True

    def __init__(self, feature_dim: int, latent_dim: int):
        super().__init__(feature_dim, latent_dim)
        self.encoder_mlp = nn.Sequential(nn.Linear(feature_dim, HIDDEN_DIM), nn.ReLU())
        self.film_enc = FiLM(HIDDEN_DIM)
        self.fc_mu = nn.Linear(HIDDEN_DIM, latent_dim)
        self.fc_log_var = nn.Linear(HIDDEN_DIM, latent_dim)
        self.decoder_in = nn.Sequential(nn.Linear(latent_dim, HIDDEN_DIM), nn.ReLU())
        self.film_dec = FiLM(HIDDEN_DIM)
        self.decoder_out = nn.Linear(HIDDEN_DIM, feature_dim)
        self.last_z: torch.Tensor | None = None
        self.last_beta: torch.Tensor | None = None
        self.init_weights()
        # FiLM starts as identity (gamma = delta = 0) so the model is equivalent to an
        # unconditioned VAE at init and training shapes the response from there.
        for film in (self.film_enc, self.film_dec):
            nn.init.zeros_(film.net.weight)
            nn.init.zeros_(film.net.bias)

    def encode(self, x: torch.Tensor, log_beta: torch.Tensor):
        h = self.film_enc(self.encoder_mlp(x), log_beta)
        return self.fc_mu(h), self.fc_log_var(h)

    def decode(self, z: torch.Tensor, log_beta: torch.Tensor) -> torch.Tensor:
        h = self.film_dec(self.decoder_in(z), log_beta)
        return self.decoder_out(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        return mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)

    def forward_at_beta(self, x: torch.Tensor, beta: float | torch.Tensor):
        if not torch.is_tensor(beta):
            beta = torch.full((x.size(0), 1), float(beta), device=x.device, dtype=x.dtype)
        log_beta = torch.log(beta)
        mu, log_var = self.encode(x, log_beta)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var, self.decode(z, log_beta)

    def sample_beta(self, batch_size: int, device, dtype) -> torch.Tensor:
        # log-uniform: the rate-distortion curve is ~linear in log beta, so uniform
        # sampling would undertrain the low-beta end where the signal is expected to live
        lo, hi = math.log(BETA_RANGE[0]), math.log(BETA_RANGE[1])
        return torch.exp(torch.rand(batch_size, 1, device=device, dtype=dtype) * (hi - lo) + lo)

    def rate_distortion_response(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample cosine distortion at each beta in BETA_GRID -> (B, K).

        Computed with the encoder deterministic (z = mu) so the profile reflects
        rate-distortion behaviour rather than sampling noise.
        """
        was_training = self.training
        self.eval()
        try:
            cols = []
            for beta in BETA_GRID:
                _, _, _, recon = self.forward_at_beta(x, beta)
                cols.append(1.0 - torch.cosine_similarity(x, recon, dim=1))
            response = torch.stack(cols, dim=1)
        finally:
            if was_training:
                self.train()
        return response

    def forward(self, semantic_feature: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Train: one sampled beta per sample. Eval: the midpoint of the grid."""
        if self.training:
            beta = self.sample_beta(semantic_feature.size(0), semantic_feature.device,
                                    semantic_feature.dtype)
        else:
            beta = torch.full((semantic_feature.size(0), 1),
                              float(BETA_GRID[RATE_RESPONSE_K // 2]),
                              device=semantic_feature.device, dtype=semantic_feature.dtype)
        self.last_beta = beta
        z, mu, log_var, recon = self.forward_at_beta(semantic_feature, beta)
        self.last_z = z
        return z, mu, log_var, recon


PROJECTORS = {
    "beta_vae": BetaVAEProjector,
    "deterministic_ae": DeterministicAEProjector,
    "beta_tcvae": BetaTCVAEProjector,
    "mr_vae": MRVAEProjector,
}

# P1c. Left unported per the Phase-1 instructions; named so the error is informative
# rather than a bare KeyError if a config asks for it.
UNPORTED = {"wae": "P1c (WAE) is unported by decision -- port it only if Track A selects it"}


def build_projector(name: str, feature_dim: int, latent_dim: int) -> ManifoldProjector:
    """Config-driven projector construction. Swapping projectors is this one string."""
    if name in UNPORTED:
        raise NotImplementedError(f"projector {name!r}: {UNPORTED[name]}")
    if name not in PROJECTORS:
        raise KeyError(f"unknown projector {name!r}; available: {sorted(PROJECTORS)}")
    return PROJECTORS[name](feature_dim=feature_dim, latent_dim=latent_dim)


def assert_projector_config(name: str, beta_kld: float) -> None:
    """Guard the one config coupling that is silent when wrong.

    deterministic_ae with a non-zero KL weight does not mean "a bit of KL"; with log_var = 0
    the term becomes 0.5 * sum(mu^2), i.e. weight decay on the latent. The pilot set
    beta_kld: 0.0 for exactly this reason, and a run that forgets it is not the P1a
    condition -- it is an unnamed third thing.
    """
    if name == "deterministic_ae" and beta_kld != 0.0:
        raise ValueError(
            "deterministic_ae requires beta_kld = 0.0. With log_var = 0 the KL term reduces "
            f"to 0.5 * sum(mu^2) -- an L2 penalty on the latent, not 'no KL'. Got "
            f"beta_kld = {beta_kld}.")

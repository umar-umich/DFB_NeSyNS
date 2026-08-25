"""Reals-only frozen reference (Specialist 1) — the E1 audit arms C0/C1/C2/C3.

Phase 2 retires the co-trained swappable manifold projector. The reason is the Phase-1 bug:
the projector never received a generative objective, so the only gradient reaching it came
through the evidence head. It therefore learned a *discriminative* transform of the CLIP
feature and `f - P(f)` was never a manifold residual at all. Phase 2 replaces it with a
reference **fit on FF++ authentic training features only and then frozen permanently**, so
`r_ref = f_0 - P(f_0)` means "how far this observation sits from the authentic manifold" by
construction rather than by hope.

The four arms answer different questions and are meant to run in one sweep:

    C0  discriminatively-learned residual transform   the Phase-1 arm, kept as a CONTROL and
                                                      relabelled honestly. Not discarded.
    C1  frozen random-matched projector               capacity floor. Same dims, same
                                                      normalisation, same head, no fitting.
    C2  reals-only PCA / linear reference             the cheapest honest reference.
    C3  reals-only deterministic AE                   nonlinear, frozen after the reals fit.

If C1 (random) matches C0, the Phase-1 branch was capacity-only. That does **not** imply
C2/C3 fail — they test a different hypothesis, which is exactly why all four run together.

The fitting objective is itself a choice
----------------------------------------
DiCoME's aligned-VAE loss is `1 - cos(z.detach(), z_hat)` plus KL: **direction-only,
magnitude-free, detached target**. Under a cosine fit the residual *magnitude* is free to
carry signal; under an MSE fit it is explicitly minimised and carries much less. These
measure different things, so `fit_objective` is a parameter and E1 evaluates both. We borrow
the loss *form* and not the co-training — ours fits on reals only and freezes.

The frozen protocol is enforced, not documented
-----------------------------------------------
`freeze()` sets `requires_grad=False` on every reference parameter and flips
`_frozen`, and `forward` asserts it. `assert_frozen_protocol()` is called from the
construction path. The offline fitter is the only code allowed to run unfrozen, and it says
so explicitly via `fitting()`. A test asserts reference params receive exactly zero gradient
through a training step — that is the guard that would have caught the original bug.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

ARMS = ("C0_discriminative", "C1_random", "C2_linear", "C3_ae")
FIT_OBJECTIVES = ("cosine", "mse")


# ---------------------------------------------------------------------------
# residual descriptors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResidualDescriptors:
    """What the evidence head is allowed to see, plus the raw residual for auditing.

    Magnitude and angle are kept SEPARATE on purpose. Under a cosine fit the reference is
    blind to magnitude, so `norm` carries information the fit never tried to remove; under an
    MSE fit that is no longer true. Collapsing them into one scalar would hide which of the
    two an arm actually depends on, which is the central question of E1.
    """

    residual: torch.Tensor      # (B, D)  r_ref = f_0 - P(f_0)
    norm: torch.Tensor          # (B,)    ||f - f_hat||_2
    angle: torch.Tensor         # (B,)    1 - cos(f, f_hat)
    recon: torch.Tensor         # (B, D)  P(f_0)

    def features(self) -> torch.Tensor:
        """(B, D+2) — the standardized residual with its magnitude and angle appended."""
        return torch.cat([self.residual,
                          self.norm.unsqueeze(1),
                          self.angle.unsqueeze(1)], dim=1)


def describe_residual(f0: torch.Tensor, recon: torch.Tensor) -> ResidualDescriptors:
    r = f0 - recon
    return ResidualDescriptors(
        residual=r,
        norm=r.norm(dim=1),
        angle=1.0 - F.cosine_similarity(f0, recon, dim=1),
        recon=recon,
    )


# ---------------------------------------------------------------------------
# residual calibration — authentic-training statistics
# ---------------------------------------------------------------------------


class ResidualCalibrator(nn.Module):
    """Standardize the residual with statistics fit on FF++ REALS ONLY.

        r_tilde = (r - mu_R) / (sigma_R + eps)

    Why reals-only and why frozen: the evidence head should see "how unusual is this residual
    relative to authentic video", so the reference distribution has to be authentic data. Using
    all training data would fold fake statistics into the yardstick and shrink exactly the
    deviation we are trying to measure. Fitting per-batch instead would make the yardstick move
    with whatever happens to be in the batch, so a whole batch of fakes would look normal.

    Buffers, not parameters — these must never receive gradient.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.register_buffer("mu", torch.zeros(dim))
        self.register_buffer("sigma", torch.ones(dim))
        self.register_buffer("fitted", torch.zeros(1, dtype=torch.bool))

    @torch.no_grad()
    def fit(self, residuals_real: torch.Tensor) -> "ResidualCalibrator":
        """Fit on authentic training residuals only. `residuals_real` is (N, D)."""
        if residuals_real.dim() != 2:
            raise ValueError(f"expected (N, D), got {tuple(residuals_real.shape)}")
        self.mu.copy_(residuals_real.mean(0))
        self.sigma.copy_(residuals_real.std(0))
        self.fitted.fill_(True)
        return self

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        if not bool(self.fitted):
            # Silently passing through would make an unfitted calibrator look like a working
            # one and quietly change what the head sees.
            raise RuntimeError(
                "ResidualCalibrator used before fit(). Fit it on FF++ authentic training "
                "residuals in the offline Stage-I step, then save it with the reference.")
        return (r - self.mu) / (self.sigma + self.eps)


# ---------------------------------------------------------------------------
# reference arms
# ---------------------------------------------------------------------------


class FrozenReference(nn.Module):
    """Base class: a map `f_0 -> P(f_0)` that is fit offline on reals and then frozen.

    Subclasses implement `project`. Everything else — freezing, the protocol assertion, the
    descriptor extraction — is shared so no arm can quietly opt out of the protocol.
    """

    arm: str = "base"

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self._frozen = False
        self._fitting = False

    # -- protocol -----------------------------------------------------------

    def freeze(self) -> "FrozenReference":
        for p in self.parameters():
            p.requires_grad_(False)
            # Drop gradients left over from the offline fit. A frozen module carrying stale
            # .grad tensors is not just untidy: any optimizer that was handed these params
            # would happily step them once, and a "did the reference receive gradient?" audit
            # cannot distinguish leftovers from a live leak.
            p.grad = None
        self._frozen = True
        self.eval()
        return self

    def fitting(self) -> "FrozenReference":
        """Enter the offline reals-only fitting stage — the ONLY unfrozen mode."""
        self._fitting = True
        self._frozen = False
        for p in self.parameters():
            p.requires_grad_(True)
        return self

    def assert_frozen_protocol(self) -> None:
        if not self._frozen:
            raise RuntimeError(
                f"{type(self).__name__} is not frozen. The reals-only reference must be fit "
                f"offline and frozen before any supervised training step; a reference that "
                f"still receives gradient from the classification loss becomes a "
                f"discriminative transform, which is the Phase-1 bug this class exists to "
                f"prevent. Call .freeze() (or .fitting() for the offline stage).")
        live = [n for n, p in self.named_parameters() if p.requires_grad]
        if live:
            raise RuntimeError(f"frozen reference still has trainable params: {live}")

    def train(self, mode: bool = True):
        """A frozen reference stays in eval mode even when the enclosing model trains."""
        super().train(mode)
        if self._frozen:
            for m in self.modules():
                m.training = False
        return self

    # -- interface ----------------------------------------------------------

    def project(self, f0: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, f0: torch.Tensor) -> ResidualDescriptors:
        if not (self._frozen or self._fitting):
            self.assert_frozen_protocol()      # raises with the explanation above
        recon = self.project(f0)
        return describe_residual(f0, recon)

    def reconstruction_loss(self, f0: torch.Tensor, objective: str) -> torch.Tensor:
        """Loss for the OFFLINE reals-only fit. Never called during supervised training."""
        recon = self.project(f0)
        if objective == "cosine":
            # direction-only, detached target — the DiCoME loss form. Magnitude is left free,
            # so the residual norm remains informative.
            return (1.0 - F.cosine_similarity(f0.detach(), recon, dim=1)).mean()
        if objective == "mse":
            # magnitude-aware: explicitly shrinks ||r||, so the norm carries less signal.
            return F.mse_loss(recon, f0.detach())
        raise ValueError(f"unknown fit objective {objective!r}; expected one of {FIT_OBJECTIVES}")


class C0DiscriminativeReference(FrozenReference):
    """C0 — the Phase-1 arm, relabelled honestly as a discriminatively-learned transform.

    Kept as a CONTROL, not discarded: it is the only arm that shows what the bugged
    configuration actually bought. It is an encoder/decoder pair identical in shape to C3, but
    its weights come from a supervised run rather than a reals-only fit, so it is loaded from a
    Phase-1 checkpoint rather than fit here.
    """

    arm = "C0_discriminative"

    def __init__(self, feature_dim: int, latent_dim: int = 32, hidden_dim: int = 32):
        super().__init__(feature_dim)
        self.encoder = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, feature_dim))

    def project(self, f0: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(f0))


class C1RandomReference(FrozenReference):
    """C1 — frozen random projector, matched in dims/normalisation/head. The capacity floor.

    Never fit on anything. If C0 cannot beat this, whatever C0 achieved came from the head's
    capacity to read *any* fixed nonlinear transform, not from the manifold hypothesis.
    """

    arm = "C1_random"

    def __init__(self, feature_dim: int, latent_dim: int = 32, hidden_dim: int = 32,
                 seed: int = 0):
        super().__init__(feature_dim)
        g = torch.Generator().manual_seed(seed)
        self.encoder = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, feature_dim))
        # deterministic init so the floor is reproducible across the sweep
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                with torch.no_grad():
                    m.weight.copy_(torch.empty_like(m.weight).uniform_(-0.1, 0.1, generator=g))
                    m.bias.zero_()
        self.freeze()      # C1 is frozen from birth; there is nothing to fit

    def project(self, f0: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(f0))

    def fitting(self):
        raise RuntimeError(
            "C1 is the random capacity floor and must never be fit — fitting it would make it "
            "a second C2/C3 and destroy the comparison it exists to provide.")


class C2LinearReference(FrozenReference):
    """C2 — reals-only PCA / linear reference. The cheapest honest reference.

    `P(f) = mu + V V^T (f - mu)` with `mu, V` from a PCA of FF++ authentic training features.
    The residual is then the component of the observation lying outside the authentic
    subspace, which is about as direct a reading of "off the real manifold" as exists.

    Stored as buffers: there are no gradients here at all, by construction, which makes this
    the arm least able to smuggle in discriminative learning.
    """

    arm = "C2_linear"

    def __init__(self, feature_dim: int, latent_dim: int = 32):
        super().__init__(feature_dim)
        self.latent_dim = latent_dim
        self.register_buffer("mean", torch.zeros(feature_dim))
        self.register_buffer("components", torch.zeros(latent_dim, feature_dim))
        self.register_buffer("explained", torch.zeros(latent_dim))
        self.register_buffer("fitted", torch.zeros(1, dtype=torch.bool))
        self.freeze()

    @torch.no_grad()
    def fit_pca(self, features_real: np.ndarray | torch.Tensor) -> dict:
        """Fit on FF++ authentic training features only. Returns a small provenance dict."""
        X = (features_real.detach().cpu().numpy()
             if torch.is_tensor(features_real) else np.asarray(features_real))
        if X.ndim != 2:
            raise ValueError(f"expected (N, D), got {X.shape}")
        if X.shape[0] < self.latent_dim:
            raise ValueError(f"need at least latent_dim={self.latent_dim} samples, got {X.shape[0]}")
        mu = X.mean(0)
        Xc = X - mu
        # economy SVD: N can be large, D is 768/1024, so this is cheap and exact
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        V = Vt[: self.latent_dim]
        var = (S ** 2) / max(1, X.shape[0] - 1)
        self.mean.copy_(torch.from_numpy(mu).float())
        self.components.copy_(torch.from_numpy(V).float())
        self.explained.copy_(torch.from_numpy(var[: self.latent_dim]).float())
        self.fitted.fill_(True)
        ratio = float(var[: self.latent_dim].sum() / max(var.sum(), 1e-12))
        return {"n_samples": int(X.shape[0]), "latent_dim": self.latent_dim,
                "explained_variance_ratio": ratio}

    def project(self, f0: torch.Tensor) -> torch.Tensor:
        if not bool(self.fitted):
            raise RuntimeError("C2LinearReference.project() before fit_pca()")
        centred = f0 - self.mean
        return self.mean + centred @ self.components.T @ self.components


class C3AEReference(FrozenReference):
    """C3 — reals-only deterministic autoencoder, frozen after the fit.

    Same shape as C0 so the two differ only in *what the weights were fit to*: C3 on FF++
    authentic features with a reconstruction objective, C0 on the supervised task. That is the
    comparison which isolates the bug.
    """

    arm = "C3_ae"

    def __init__(self, feature_dim: int, latent_dim: int = 32, hidden_dim: int = 32):
        super().__init__(feature_dim)
        self.encoder = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, feature_dim))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def project(self, f0: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(f0))


REFERENCES = {
    "C0_discriminative": C0DiscriminativeReference,
    "C1_random": C1RandomReference,
    "C2_linear": C2LinearReference,
    "C3_ae": C3AEReference,
}


def build_reference(arm: str, feature_dim: int, latent_dim: int = 32,
                    hidden_dim: int | None = None) -> FrozenReference:
    """Construct one reference arm.

    `hidden_dim` is plumbed rather than left at its default because the default (32) was sized
    for DiCoME's 64-d feature. On V1's 1024-d FS-VFM space that same 32 would make the encoder a
    1024 -> 32 bottleneck — a severe compression inherited from a different feature width rather
    than chosen, which would drive the reconstruction error (and therefore r_ref) far more than
    anything about authenticity. Callers pass a width matched to their feature space; the
    default is preserved for the arms already fit at 64-d so their artifacts stay reproducible.
    """
    if arm not in REFERENCES:
        raise KeyError(f"unknown reference arm {arm!r}; available: {sorted(REFERENCES)}")
    kw = {"feature_dim": feature_dim, "latent_dim": latent_dim}
    if hidden_dim is not None and arm != "C2_linear":   # PCA has no hidden layer
        kw["hidden_dim"] = hidden_dim
    return REFERENCES[arm](**kw)


def assert_reference_config(arm: str, fit_objective: str, encoder_frozen: bool) -> None:
    """Guard the config couplings whose failure modes are silent. Called at construction.

    Two of these would produce a plausible-looking but meaningless reference:

    * A reference fit on a *drifting* encoder. If the CLIP encoder is LN/LoRA-tuned during
      training, the feature space the reference was fit in no longer exists and every residual
      is measured against a stale manifold. This is the root lesson of the bug, and Task 0
      exists to decide the encoder; whatever it decides, the reference's input space must be
      frozen.
    * C1 with a fit objective. The random floor must not be fit (see C1RandomReference).
    """
    if fit_objective not in FIT_OBJECTIVES:
        raise ValueError(f"fit_objective must be one of {FIT_OBJECTIVES}, got {fit_objective!r}")
    if not encoder_frozen:
        raise ValueError(
            "the reals-only reference must be fit on a FROZEN encoder feature space. With an "
            "LN/LoRA-tuned encoder the features drift away from the ones the reference was fit "
            "on, so r_ref is measured against a stale manifold and is miscalibrated by "
            "construction. Use a frozen snapshot E_ref for the reference input (see Task 0's "
            "dual-encoder option) even if e_sem uses a tuned encoder.")
    if arm == "C1_random" and fit_objective is not None:
        # not fatal: the fitter skips C1. Recorded so a sweep config reads honestly.
        logger.info("C1_random ignores fit_objective — it is never fit (capacity floor).")

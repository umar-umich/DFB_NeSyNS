"""`H_traj` — the trajectory readout (brief Stage 1, item 3).

    D(x) = [d_l1, ..., d_lL]  ->  standardize  ->  MLP  ->  e_face (B, 2)

Structurally the same object as the phase-2 rate branch: a fixed-length curve from a frozen
mechanism, standardized against authentic training statistics, read by a small head that emits
non-negative evidence. So the EDL conversion, the calibrator and the shape of the head are reused
rather than rewritten (`networks/discern_v2/rate_branch.py`, `reference.py::ResidualCalibrator`,
`dirichlet.py::to_dirichlet`).

Two things this head deliberately does
--------------------------------------
**It reads the curve's shape, not only its level.** Inputs are the L per-layer deltas plus their
L-1 successive differences across depth. The paper's forensic question is *where in the hierarchy*
adaptation departs from the prior, and a linear map over raw deltas can only express overall
magnitude and offset. Supplying the differences makes early-versus-late departure representable
by the head rather than something a reader has to infer from a plot.

**It fixes no polarity.** Nothing here encodes "larger deviation means fake". The direction, and
any non-monotone shape, is learned. That matters because the honest baseline is B1, an ordinary
fine-tuned student: if the trajectory only worked when hardcoded to "more adaptation = fake", it
would be re-deriving the classifier's decision rather than adding information.

Calibration is required, or explicitly declined
----------------------------------------------
`fit_calibrator` standardizes `D(x)` with statistics from **authentic FF++ training faces only**,
for the same reason the reference branch does: the head should see "how unusual is this
trajectory relative to authentic video", and folding fake statistics into the yardstick shrinks
the deviation being measured.

An unfitted calibrator **raises** rather than passing the deltas through
(`reference.py::ResidualCalibrator.forward`), and that guard is kept rather than softened: a
result that silently claimed a yardstick it never had is exactly what it exists to prevent. If
raw deltas are wanted, construct with `use_calibrator=False`, which is recorded in
`describe()` so the choice appears in the artifact instead of being inferred from a missing file.

Note the deltas are already scale-free — a cosine distance in [0, 2] — so standardisation here
is about equalising the LAYERS, whose typical deviation differs by depth, not about fixing units.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..discern_v2.dirichlet import to_dirichlet
from ..discern_v2.reference import ResidualCalibrator


class TrajectoryEvidenceHead(nn.Module):
    """`D(x)` -> evidence. The only trainable part of Stage B."""

    def __init__(self, n_layers: int, hidden_dim: int = 64, use_slopes: bool = True,
                 num_classes: int = 2, dropout: float = 0.0, use_calibrator: bool = True):
        super().__init__()
        if n_layers < 2 and use_slopes:
            raise ValueError("slopes need at least two layers; pass use_slopes=False")
        self.n_layers = int(n_layers)
        self.use_slopes = bool(use_slopes)
        self.num_classes = int(num_classes)
        self.use_calibrator = bool(use_calibrator)
        self.calibrator = ResidualCalibrator(self.n_layers) if use_calibrator else None
        in_dim = self.n_layers + (self.n_layers - 1 if use_slopes else 0)
        self.in_dim = in_dim
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.head = nn.Sequential(*layers)

    @property
    def is_calibrated(self) -> bool:
        return bool(self.calibrator.fitted.item()) if self.use_calibrator else False

    def describe(self) -> dict:
        """What this head is, for the artifact. Calibration state is recorded, not inferred."""
        return {"n_layers": self.n_layers, "use_slopes": self.use_slopes,
                "in_dim": self.in_dim, "use_calibrator": self.use_calibrator,
                "calibrated": self.is_calibrated,
                "readout": "per-layer cosine deviation D(x)"
                           + (" + successive differences" if self.use_slopes else "")}

    @torch.no_grad()
    def fit_calibrator(self, D_real: torch.Tensor) -> "TrajectoryEvidenceHead":
        """Fit on AUTHENTIC training trajectories only, then freeze the statistics."""
        if not self.use_calibrator:
            raise RuntimeError(
                "this head was constructed with use_calibrator=False, so there is nothing to "
                "fit. Rebuild it with use_calibrator=True rather than fitting a discarded "
                "module.")
        if D_real.dim() != 2 or D_real.shape[1] != self.n_layers:
            raise ValueError(f"expected (N, {self.n_layers}), got {tuple(D_real.shape)}")
        self.calibrator.fit(D_real)
        for p in self.calibrator.parameters():
            p.requires_grad_(False)
        return self

    def features(self, D: torch.Tensor) -> torch.Tensor:
        """Standardized deltas, plus their successive differences across depth."""
        if not self.use_calibrator:
            z = D
        elif not self.is_calibrated:
            raise RuntimeError(
                "H_traj's calibrator is unfitted. Fit it on AUTHENTIC FF++ training trajectories "
                "before Stage B (`fit_calibrator`), or construct the head with "
                "use_calibrator=False if raw deltas are intended. Running uncalibrated by "
                "accident would standardise against nothing while the artifact still recorded a "
                "calibrator.")
        else:
            z = self.calibrator(D)
        if not self.use_slopes:
            return z
        return torch.cat([z, z[:, 1:] - z[:, :-1]], dim=1)

    def forward(self, D: torch.Tensor) -> dict:
        if D.shape[-1] != self.n_layers:
            raise ValueError(
                f"D has {D.shape[-1]} layers but this head was built for {self.n_layers}. A "
                f"head applied to a different layer set is reading a different signal.")
        valid = torch.isfinite(D).all(dim=1)
        feature = self.features(torch.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0))
        evidence = F.softplus(self.head(feature))
        state = to_dirichlet(evidence)
        return {
            "evidence": evidence,
            "feature": feature,
            "valid": valid,
            "prob": state.fake_prob(),
            "u": state.vacuity,
            # the scalar Stage 6's ranking loss ranks on; `D` is a vector, so L_pair cannot act
            # on it directly, and the brief is explicit about ranking the trajectory
            # classifier's fake score instead
            "s_traj": state.fake_prob(),
        }


def build_traj_head(n_layers: int, cfg: dict | None = None) -> TrajectoryEvidenceHead:
    cfg = cfg or {}
    return TrajectoryEvidenceHead(
        n_layers=n_layers,
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        use_slopes=bool(cfg.get("use_slopes", True)),
        dropout=float(cfg.get("dropout", 0.0)),
        use_calibrator=bool(cfg.get("use_calibrator", True)))


class DirectEvidenceHead(nn.Module):
    """The DIRECT readout: classify from the student's own adapted feature (rungs B0/B1).

    Kept in this file beside `H_traj` so the two readouts are visibly the same capacity class.
    `B2 - B1` is a readout-only comparison, so if the direct head were a linear probe and the
    trajectory head an MLP, that contrast would partly measure head capacity instead of what the
    trajectory carries. Same hidden width, same activation, same output.
    """

    def __init__(self, feature_dim: int = 1024, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.GELU(),
                                  nn.Linear(hidden_dim, num_classes))

    def forward(self, h: torch.Tensor) -> dict:
        evidence = F.softplus(self.head(h))
        state = to_dirichlet(evidence)
        return {"evidence": evidence, "prob": state.fake_prob(), "u": state.vacuity,
                "valid": torch.isfinite(h).all(dim=1)}


class SpatialEvidenceHead(nn.Module):
    """Rung B5 — a SPATIAL readout over the patch adaptation map.

    Stage-B change only. The frozen preservation student is untouched; only this head trains.

        a_j = sum_l w_l * d_{l,j}^TS          per-patch adaptation, layer weights LEARNED
        h   = pool_j(a_j) over z_j            top-k mean, or attention softmax(g(a_j))
              -> EDL head -> e_face

    Why the layer weights are learned rather than fixed
    ---------------------------------------------------
    The per-layer localization probe measured, against genuine FF++ masks, that layer 20 localizes
    at 2.64x chance while layers 8 and 12 sit AT or BELOW chance (0.97x, 0.89x) — so an equal mean
    over the six layers dilutes 2.64x down to 1.79x. Hard-coding layer 20 would bake that
    measurement in; a learned softmax over six weights lets the head find it, and the fitted
    weights then become reportable evidence rather than an assumption. Six parameters, initialised
    uniform.

    Why this is `faithfulness by architecture`, and why that is not enough
    ---------------------------------------------------------------------
    The mean-pool readout failed deletion AND insertion: no patch was individually load-bearing
    because the decision was a mean over 196 of them. Here the decision depends on WHICH patches
    the adaptation map selects, so deleting the top patches must cost evidence — the faithfulness
    test becomes close to circular.

    That is exactly why localization against genuine masks is the deciding check, not faithfulness.
    If AUPRC drops relative to the mean-pool baseline, then the head has learned to attend to
    whatever helps classification rather than to the manipulated region, and architectural
    faithfulness is hollow. The evaluation must report both, and localization is the half that can
    still fail.
    """

    def __init__(self, n_layers: int, feature_dim: int = 1024, hidden_dim: int = 64,
                 mode: str = "attention", top_k_fraction: float = 0.25,
                 num_classes: int = 2):
        super().__init__()
        if mode not in ("attention", "topk"):
            raise ValueError(f"mode must be 'attention' or 'topk', got {mode!r}")
        self.n_layers = int(n_layers)
        self.mode = mode
        self.top_k_fraction = float(top_k_fraction)
        # softmax over layers, uniform at init: the head starts as the equal-weight mean the
        # probe measured, and has to earn any departure from it
        self.layer_logits = nn.Parameter(torch.zeros(self.n_layers))
        self.gate = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, 1))
        self.head = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.GELU(),
                                  nn.Linear(hidden_dim, num_classes))

    def layer_weights(self) -> torch.Tensor:
        return torch.softmax(self.layer_logits, dim=0)

    def adaptation_map(self, patch_delta: torch.Tensor) -> torch.Tensor:
        """(B, L, N) -> (B, N), the learned-weight combination of the per-layer patch deltas."""
        if patch_delta.shape[1] != self.n_layers:
            raise ValueError(
                f"patch_delta has {patch_delta.shape[1]} layers, head built for {self.n_layers}")
        w = self.layer_weights().view(1, -1, 1)
        return (patch_delta * w).sum(dim=1)

    def forward(self, patch_delta: torch.Tensor, patch_tokens: torch.Tensor) -> dict:
        a = self.adaptation_map(patch_delta)                        # (B, N)
        if self.mode == "attention":
            weights = torch.softmax(self.gate(a.unsqueeze(-1)).squeeze(-1), dim=1)
        else:
            k = max(1, int(round(self.top_k_fraction * a.shape[1])))
            idx = a.topk(k, dim=1).indices
            weights = torch.zeros_like(a).scatter_(1, idx, 1.0 / k)
        pooled = (weights.unsqueeze(-1) * patch_tokens).sum(dim=1)  # (B, C)
        evidence = F.softplus(self.head(pooled))
        state = to_dirichlet(evidence)
        return {"evidence": evidence, "prob": state.fake_prob(), "u": state.vacuity,
                "adaptation_map": a, "attention": weights, "pooled": pooled,
                "valid": torch.isfinite(a).all(dim=1)}

    def describe(self) -> dict:
        return {"kind": "spatial", "n_layers": self.n_layers, "mode": self.mode,
                "top_k_fraction": self.top_k_fraction,
                "layer_weights": [round(float(v), 4) for v in self.layer_weights()]}


def build_spatial_head(n_layers: int, feature_dim: int, cfg: dict | None = None
                       ) -> SpatialEvidenceHead:
    cfg = cfg or {}
    return SpatialEvidenceHead(n_layers=n_layers, feature_dim=feature_dim,
                               hidden_dim=int(cfg.get("hidden_dim", 64)),
                               mode=str(cfg.get("spatial_mode", "attention")),
                               top_k_fraction=float(cfg.get("top_k_fraction", 0.25)))

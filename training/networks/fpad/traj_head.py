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

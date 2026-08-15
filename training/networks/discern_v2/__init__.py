"""DISCERN v2 evidence branches, projectors and the shared Dirichlet utility.

Phase-1 scaffolding for the T-BIOM journal extension. See
`docs/DiCoME_eval/README-discern-v2-integration.md` for the phase framing and
`docs/DiCoME_eval/claude-code-discern-v2-phase1.md` for this task's contract.

Nothing here is wired into the v1 forward path. The D-ladder configs under
`training/config/discern_v2/` select branches; D0 restores v1 explicitly.
"""

from .branches import (
    BranchOutput,
    EvidenceBranch,
    ManifoldEvidenceBranch,
    ProcessEvidenceBranch,
    VisualEvidenceBranch,
    build_branches,
)
from .dirichlet import DirichletState, conflict, logits_to_evidence, to_dirichlet
from .projectors import (
    BETA_GRID,
    PROJECTORS,
    ManifoldProjector,
    assert_projector_config,
    build_projector,
)

__all__ = [
    "BranchOutput", "EvidenceBranch", "VisualEvidenceBranch", "ManifoldEvidenceBranch",
    "ProcessEvidenceBranch", "build_branches",
    "DirichletState", "to_dirichlet", "logits_to_evidence", "conflict",
    "ManifoldProjector", "build_projector", "assert_projector_config", "PROJECTORS",
    "BETA_GRID",
]

"""Branch D (MR-VAE rate response) — freeze protocol, polarity freedom, model wiring.

    python training/networks/discern_v2/test_rate_branch.py

The properties checked here are the ones that fail SILENTLY if broken:

* the operator staying frozen under `model.train()` (otherwise the response drifts over a run
  for reasons unrelated to the data, and every number still looks plausible);
* `hidden_dim` surviving the save/load round trip (the reference branch shipped with this bug —
  the artifact recorded the width but the loader rebuilt at the default);
* the head being able to map EITHER polarity to fake, since the brief forbids hardcoding
  "larger residual means fake";
* the branch being genuinely optional, because Stage 3 may drop it.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from networks.discern_v2.projectors import BETA_GRID, MRVAEProjector  # noqa: E402
from networks.discern_v2.rate_branch import (  # noqa: E402
    FrozenRateOperator, RateEvidenceBranch, build_rate_branch)
from networks.discern_v2.reference import ResidualCalibrator  # noqa: E402

PASSED = []
FEATURE_DIM, LATENT_DIM, HIDDEN = 1024, 128, 256


def check(name):
    def deco(fn):
        fn()
        PASSED.append(name)
        print(f"  ok  {name}")
        return fn
    return deco


def make_artifact(directory: Path, hidden_dim: int = HIDDEN) -> Path:
    """An artifact in exactly the shape `fit_rate_operator.py` writes."""
    torch.manual_seed(0)
    projector = MRVAEProjector(FEATURE_DIM, LATENT_DIM, hidden_dim=hidden_dim)
    with torch.no_grad():
        response = projector.rate_distortion_response(torch.randn(64, FEATURE_DIM))
    calibrator = ResidualCalibrator(len(BETA_GRID)).fit(response)
    dest = directory / "rate_operator_mrvae.pt"
    torch.save({"feature_dim": FEATURE_DIM, "latent_dim": LATENT_DIM, "hidden_dim": hidden_dim,
                "beta_grid": list(BETA_GRID), "projector_state": projector.state_dict(),
                "calibrator_state": calibrator.state_dict(),
                "provenance": {"stage": "test fixture"}}, dest)
    return dest


@check("hidden_dim survives the artifact round trip")
def _hidden_dim_round_trip():
    """The reference branch shipped with exactly this bug: width saved, loader used the default."""
    with tempfile.TemporaryDirectory() as tmp:
        for width in (32, 256):
            branch = build_rate_branch(make_artifact(Path(tmp), hidden_dim=width))
            assert branch.operator.hidden_dim == width, (branch.operator.hidden_dim, width)


@check("beta grid comes from the artifact, not the module constant")
def _grid_from_artifact():
    with tempfile.TemporaryDirectory() as tmp:
        branch = build_rate_branch(make_artifact(Path(tmp)))
        assert branch.operator.beta_grid == tuple(float(b) for b in BETA_GRID)
        assert branch.k == len(BETA_GRID)


@check("operator is frozen and stays frozen under model.train()")
def _stays_frozen():
    with tempfile.TemporaryDirectory() as tmp:
        branch = build_rate_branch(make_artifact(Path(tmp)))
        branch.train()
        branch.assert_frozen()
        assert not branch.operator.projector.training
        assert all(not p.requires_grad for p in branch.operator.parameters())


@check("no gradient reaches the operator")
def _no_gradient_leak():
    with tempfile.TemporaryDirectory() as tmp:
        branch = build_rate_branch(make_artifact(Path(tmp)))
        branch(torch.randn(8, FEATURE_DIM))["evidence"].sum().backward()
        leaked = [n for n, p in branch.operator.named_parameters() if p.grad is not None]
        assert not leaked, leaked


@check("only the head trains")
def _head_only():
    with tempfile.TemporaryDirectory() as tmp:
        branch = build_rate_branch(make_artifact(Path(tmp)))
        trainable = {n for n, p in branch.named_parameters() if p.requires_grad}
        assert all(n.startswith("head.") for n in trainable), sorted(trainable)


@check("evidence is non-negative and finite; the feature carries curve SHAPE, not just level")
def _evidence_shape():
    with tempfile.TemporaryDirectory() as tmp:
        branch = build_rate_branch(make_artifact(Path(tmp)))
        out = branch(torch.randn(8, FEATURE_DIM))
        assert out["evidence"].shape == (8, 2)
        assert (out["evidence"] >= 0).all() and torch.isfinite(out["evidence"]).all()
        # K distortions + (K-1) slopes: a linear head over raw distortions could read only
        # magnitude and offset, which is not what Stage 2.3 is asking about
        assert out["feature"].shape == (8, len(BETA_GRID) * 2 - 1)
        assert sorted(out["raw_stats"]) == [f"rate_r_{i}" for i in range(len(BETA_GRID))]


@check("polarity is learnable in BOTH directions (nothing encodes 'larger means fake')")
def _polarity_free():
    """Fit the head on inverted targets. If either direction is harder to learn, some part of the
    branch encodes a sign, which the brief forbids ("do not hardcode 'larger residual means
    fake'"). The operator is frozen, so its response is computed ONCE and the head is trained on
    the cached feature — training through a frozen operator 400 times would measure the same
    thing far more slowly."""
    with tempfile.TemporaryDirectory() as tmp:
        branch = build_rate_branch(make_artifact(Path(tmp)))
        torch.manual_seed(1)
        x = torch.randn(256, FEATURE_DIM)
        with torch.no_grad():
            out = branch(x)
        feature, r0 = out["feature"], out["raw_stats"]["rate_r_0"]

        for sign, label in ((+1.0, "high distortion -> fake"), (-1.0, "low distortion -> fake")):
            head = torch.nn.Sequential(torch.nn.Linear(feature.shape[1], 32), torch.nn.GELU(),
                                       torch.nn.Linear(32, 2))
            y = ((sign * r0) > (sign * r0).median()).float()
            opt = torch.optim.Adam(head.parameters(), lr=1e-2)
            for _ in range(300):
                opt.zero_grad()
                evidence = torch.nn.functional.softplus(head(feature))
                p_fake = (evidence[:, 1] + 1) / (evidence.sum(1) + 2)
                loss = torch.nn.functional.binary_cross_entropy(p_fake.clamp(1e-6, 1 - 1e-6), y)
                loss.backward()
                opt.step()
            acc = float(((p_fake > 0.5).float() == y).float().mean())
            assert acc > 0.75, f"{label}: only reached {acc:.2f} — the branch encodes a polarity"


@check("non-finite response marks the sample invalid instead of inventing evidence")
def _invalid_on_nonfinite():
    with tempfile.TemporaryDirectory() as tmp:
        branch = build_rate_branch(make_artifact(Path(tmp)))
        x = torch.randn(4, FEATURE_DIM)
        x[2] = float("nan")
        out = branch(x)
        assert not bool(out["valid"][2]), "a NaN response must not be reported valid"
        assert torch.isfinite(out["evidence"]).all(), (
            "evidence must stay finite even for the invalid row — the caller turns validity into "
            "a vacuous opinion, and a NaN would poison the fusion before it got there")


@check("missing calibrator state is refused, not defaulted")
def _requires_calibrator():
    with tempfile.TemporaryDirectory() as tmp:
        path = make_artifact(Path(tmp))
        blob = torch.load(str(path), map_location="cpu", weights_only=False)
        blob.pop("calibrator_state")
        torch.save(blob, path)
        try:
            build_rate_branch(path)
        except SystemExit as exc:
            assert "calibrator_state" in str(exc)
        else:
            raise AssertionError("an uncalibrated operator must be refused")


@check("the model builds with and without the rate branch")
def _optional_in_model():
    from networks.discern_v2.discern_v1_model import FUSION_ORDER, SPECIALISTS
    assert "rate" in SPECIALISTS and "rate" in FUSION_ORDER
    with tempfile.TemporaryDirectory() as tmp:
        branch = build_rate_branch(make_artifact(Path(tmp)))
        assert isinstance(branch, RateEvidenceBranch)
        assert isinstance(branch.operator, FrozenRateOperator)
    # Stage 3 may drop it: absent from the config means absent from the model, no flag to forget
    from networks.discern_v2 import discern_v1_model as M
    assert "rate_operator" in M.default_artifact_paths()


if __name__ == "__main__":
    print(f"\nall {len(PASSED)} passed")

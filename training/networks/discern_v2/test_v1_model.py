"""Tests for the assembled V1 three-branch model (spec §2, §4.2, §9, §14, §17, §19).

    python training/networks/discern_v2/test_v1_model.py

Uses lightweight stand-in branches by default so the wiring is testable without ViT-L, the
4.4 GB FS-VFM checkpoint or the VAE. One end-to-end test builds the real branches when the
artifacts are present, and skips otherwise.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from networks.discern_v2 import discern_v1_model as M  # noqa: E402
from networks.discern_v2.ds_fusion import Opinion  # noqa: E402

PASSED, SKIPPED = [], []
B = 4


class SkipTest(Exception):
    pass


def check(name):
    def deco(fn):
        try:
            fn()
        except SkipTest as exc:
            SKIPPED.append(name)
            print(f"  skip: {name} ({exc})")
        else:
            PASSED.append(name)
            print(f"  ok: {name}")
        return fn
    return deco


# ---------------------------------------------------------------- stand-ins


class StubSemantic(nn.Module):
    """Same contract as SemanticEvidenceBranch, without the 300M-parameter backbone."""

    name = "sem"

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(8, 2)
        self.lora_config_used = {"stub": True}

    def forward(self, x):
        feat = x.flatten(1)[:, :8]
        return {"evidence": F.softplus(self.head(feat)), "feature": feat,
                "raw_feature": feat}

    def assert_lora_only(self):
        return {"lora": 1, "frozen_backbone": 1, "projection": 0, "norm": 0, "head": 1}


class StubBranchOutput:
    def __init__(self, evidence, features):
        self.evidence, self.features, self.diagnostics = evidence, features, {}


class StubReference(nn.Module):
    name = "ref"

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(16, 2)

    def forward(self, z):
        f = z[:, :16]
        return StubBranchOutput(F.softplus(self.head(f)), f)

    def assert_frozen(self):
        return None


class StubProcess(nn.Module):
    name = "proc"

    def __init__(self, fail_first: bool = False):
        super().__init__()
        self.head = nn.Linear(6, 2)
        self.fail_first = fail_first

    def forward(self, images):
        stats = images.flatten(1)[:, :6]
        valid = torch.ones(stats.shape[0], dtype=torch.bool)
        if self.fail_first:
            valid[0] = False
        return {"evidence": F.softplus(self.head(stats)), "feature": stats,
                "valid": valid, "raw_stats": stats}

    def assert_frozen(self):
        return None


class StubFSVFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w", torch.randn(16, 1024) * 0.05)

    def forward(self, x):
        return x.flatten(1)[:, :16] @ self.w

    def assert_frozen(self):
        return None


def model(stage: str = "B", fail_first: bool = False) -> M.DiscernV1Model:
    return M.DiscernV1Model(
        semantic=StubSemantic(), reference=StubReference(),
        process=StubProcess(fail_first=fail_first), fsvfm=StubFSVFM(),
        direct_probe=M.build_direct_probe(1024, 64), stage=stage)


def batch(n: int = B) -> dict:
    return {"spatial_frames": torch.randn(n, 3, 16, 16),
            "fsvfm_frames": torch.rand(n, 3, 16, 16),
            "process_frames": torch.rand(n, 3, 16, 16)}


# ---------------------------------------------------------------- wiring


@check("§2: exactly three opinions reach fusion; e_direct is computed but never fused")
def _three_branches():
    out = model()(batch())
    assert set(out["order"]) == {"sem", "ref", "proc"}, out["order"]
    assert "direct" in out["branches"], "the §4.2 control must still be computed and logged"
    assert "direct" not in out["order"], "the control must NOT enter fusion"
    assert out["branches"]["direct"]["diagnostics"]["control"] is True


@check("§16/§17: fusion yields a valid Dirichlet and finite V/C/A")
def _fusion_outputs():
    out = model()(batch())
    assert out["alpha"].shape == (B, 2)
    assert (out["alpha"] >= 1).all(), "alpha < 1 is not a valid evidential posterior"
    out["fused"].assert_normalized()
    for key in ("V", "C", "A", "prob"):
        v = out[key]
        assert v.shape == (B,), (key, v.shape)
        assert torch.isfinite(v).all(), key
        assert (v >= -1e-6).all() and (v <= 1 + 1e-6).all(), (key, v)


@check("§4.2: the control head sees the SAME z_ref the reference branch sees")
def _shared_features():
    m = model()
    b = batch()
    out = m(b)
    z = m._z_ref(b)
    assert torch.allclose(out["branches"]["direct"]["feature"], z), (
        "the control must read the identical frozen representation, or the comparison is not "
        "controlled")


@check("§14.1: an invalid specialist becomes vacuous and cannot move the fused decision")
def _invalid_is_ignorance():
    m = model(fail_first=True)
    b = batch()
    out = m(b)
    assert not bool(out["branches"]["proc"]["valid"][0])
    assert float(out["opinions"]["proc"].vacuity[0]) == 1.0

    # the fused opinion for that sample must equal fusing without the process branch at all
    without = m.fuse({k: v for k, v in out["branches"].items() if k in ("sem", "ref")})
    assert torch.allclose(out["fused"].belief[0], without["fused"].belief[0], atol=1e-5), (
        "an unavailable specialist must leave the fusion untouched, not shift it toward Real")


@check("§14.2: q = 0 makes a specialist inert; q = 1 restores it")
def _gating():
    m = model()
    b = batch()
    branches = m.branch_outputs(b)
    fusion_set = {k: v for k, v in branches.items() if k != "direct"}
    zeros = torch.zeros(B)
    ones = torch.ones(B)

    off = m.fuse(fusion_set, q={"ref": zeros, "proc": zeros})
    sem_only = m.fuse({"sem": fusion_set["sem"]})
    assert torch.allclose(off["fused"].belief, sem_only["fused"].belief, atol=1e-5), (
        "with both gates closed the fusion must reduce to the anchor alone")

    on = m.fuse(fusion_set, q={"ref": ones, "proc": ones})
    assert not torch.allclose(on["fused"].belief, sem_only["fused"].belief, atol=1e-3), (
        "with gates open the specialists must actually contribute")
    assert float(off["A"].mean()) > float(on["A"].mean()), (
        "A (unsupportedness) must be higher when no specialist is applicable")


@check("Stage B default: no gate means q = 1, not q = 0")
def _stage_b_ungated():
    m = model()
    b = batch()
    branches = m.branch_outputs(b)
    fusion_set = {k: v for k, v in branches.items() if k != "direct"}
    default = m.fuse(fusion_set)
    explicit = m.fuse(fusion_set, q={"ref": torch.ones(B), "proc": torch.ones(B)})
    assert torch.allclose(default["fused"].belief, explicit["fused"].belief, atol=1e-6)


# ---------------------------------------------------------------- staging / §19


@check("§9/§19: leaving Stage B freezes every expert")
def _stage_freezes():
    m = model(stage="B")
    assert any(p.requires_grad for p in m.semantic.parameters())
    m.set_stage("D")
    for name, module in (("semantic", m.semantic), ("reference", m.reference),
                         ("process", m.process), ("direct", m.direct_probe)):
        live = [n for n, p in module.named_parameters() if p.requires_grad]
        assert not live, f"{name} still trainable in stage D: {live[:3]}"


@check("an unknown stage is refused")
def _bad_stage():
    try:
        model(stage="Z")
    except ValueError as exc:
        assert "stage must be one of" in str(exc)
    else:
        raise AssertionError("an unknown stage must raise")


@check("§9: the auxiliary loss is masked on samples a branch could not process")
def _masked_aux_loss():
    m = model(fail_first=True)
    branches = m.branch_outputs(batch())
    labels = torch.tensor([1, 0, 1, 0])
    seen = {}

    def edl(evidence, y):
        seen["n"] = evidence.shape[0]
        return F.cross_entropy(evidence + 1e-6, y)

    losses = m.auxiliary_losses({"proc": branches["proc"]}, labels, edl)
    assert seen["n"] == B - 1, f"the invalid sample must be excluded, saw {seen['n']}"
    assert torch.isfinite(losses["proc"])


@check("§19: the frozen-protocol assertion runs from the model, not a dead check")
def _protocol_assert():
    m = model()
    m.assert_frozen_protocol()

    class Leaky(StubReference):
        def assert_frozen(self):
            raise RuntimeError("reference weights are trainable")

    m.reference = Leaky()
    try:
        m.assert_frozen_protocol()
    except RuntimeError as exc:
        assert "trainable" in str(exc)
    else:
        raise AssertionError("a leaky reference must be caught")


@check("build_v1_model refuses a missing reference artifact instead of dropping the branch")
def _refuses_missing_artifact():
    try:
        M.build_v1_model({"reference": {"enabled": True}, "process": {"enabled": False}})
    except ValueError as exc:
        assert "artifact_path is required" in str(exc)
    else:
        raise AssertionError("a missing Stage-A artifact must be an error, not a silent 2-branch run")


@check("build_v1_model refuses a process branch with no VAE path")
def _refuses_missing_vae():
    try:
        M.build_v1_model({"reference": {"enabled": False}, "process": {"enabled": True}})
    except ValueError as exc:
        assert "vae_path is required" in str(exc)
    else:
        raise AssertionError("a missing VAE path must be an error")


# ---------------------------------------------------------------- real branches


@check("end to end with the REAL frozen FS-VFM + VAE + CLIP-LoRA")
def _real_end_to_end():
    from networks.discern_v2.fsvfm_encoder import DEFAULT_CHECKPOINT
    vae = "/data/umar/Repos/DiCoME/eval_adaptation/data/models/sdxl-vae"
    if not os.path.isfile(DEFAULT_CHECKPOINT):
        raise SkipTest("no FS-VFM checkpoint")
    if not os.path.isdir(vae):
        raise SkipTest("no sdxl-vae")

    artifact = os.environ.get("DISCERN_TEST_REFERENCE_ARTIFACT")
    if not artifact or not os.path.isfile(artifact):
        raise SkipTest("set DISCERN_TEST_REFERENCE_ARTIFACT to a Stage-A artifact")

    from networks.discern_v2.fsvfm_encoder import FrozenFSVFM
    from networks.discern_v2.process_branch import ProcessEvidenceBranch
    from networks.discern_v2.reference_branch import ReferenceEvidenceBranch
    from networks.discern_v2.semantic_branch import SemanticEvidenceBranch

    process = ProcessEvidenceBranch(vae_path=vae, resolution=128)
    process.calibrator.fit(torch.randn(32, process.n_stats) * 0.1 + 0.5)
    m = M.DiscernV1Model(
        semantic=SemanticEvidenceBranch(),
        reference=ReferenceEvidenceBranch(artifact, input_dim=1024),
        process=process, fsvfm=FrozenFSVFM(),
        direct_probe=M.build_direct_probe(1024, 64), stage="B")
    m.assert_frozen_protocol()

    out = m({"spatial_frames": torch.randn(2, 3, 224, 224),
             "fsvfm_frames": torch.rand(2, 3, 224, 224),
             "process_frames": torch.rand(2, 3, 224, 224)})
    out["fused"].assert_normalized()
    assert torch.isfinite(out["prob"]).all()
    Opinion.from_evidence(out["branches"]["sem"]["evidence"]).assert_normalized()
    print(f"      fused p(fake)={[round(float(x), 4) for x in out['prob']]} "
          f"V={[round(float(x), 3) for x in out['V']]}")


if __name__ == "__main__":
    print(f"\n{len(PASSED)} passed, {len(SKIPPED)} skipped")
    for s in SKIPPED:
        print(f"  - skipped: {s}")
    sys.exit(0 if PASSED else 1)

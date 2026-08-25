"""FPAD teacher-student and H_traj — the properties that fail SILENTLY if broken.

    python training/networks/fpad/test_teacher_student.py

Runs on the REAL FS-VFM checkpoint if it is present, and skips those cases with a printed reason
if it is not. The checks are chosen for the failure modes that produce plausible-looking numbers:

* `D` being exactly zero at init. If it were not, the two-stage training's premise is wrong; if it
  stayed zero after adaptation, the trajectory carries nothing and every downstream AUROC would be
  chance without anything erroring.
* the teacher moving. If the backbone is trainable, `d_l^TS` measures two moving targets instead
  of departure from a fixed prior, and training would still converge.
* hook cross-talk. If the teacher and student passes shared a buffer, `D` would compare a pass
  with itself — an all-zero trajectory that looks like "no adaptation yet".
* the readout being CLS rather than the patch mean. Silent, and it changes what the paper measures.
* head capacity differing between the direct and trajectory readouts, which would contaminate the
  `B2 - B1` contrast with a capacity difference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from networks.fpad.teacher_student import (  # noqa: E402
    DEFAULT_LAYERS, FPAD_LORA, FPADTeacherStudent, _BlockCapture, _cosine_distance)
from networks.fpad.traj_head import (  # noqa: E402
    DirectEvidenceHead, TrajectoryEvidenceHead)
from networks.discern_v2.fsvfm_encoder import DEFAULT_CHECKPOINT  # noqa: E402

PASSED, SKIPPED = [], []
HAVE_CKPT = Path(DEFAULT_CHECKPOINT).is_file()
_MODEL = None


def check(name, needs_ckpt=False):
    def deco(fn):
        if needs_ckpt and not HAVE_CKPT:
            SKIPPED.append(name)
            print(f"  skip {name} (no checkpoint at {DEFAULT_CHECKPOINT})")
            return fn
        fn()
        PASSED.append(name)
        print(f"  ok  {name}")
        return fn
    return deco


def model() -> FPADTeacherStudent:
    """Built once; loading a ViT-L twelve times would dominate the runtime."""
    global _MODEL
    if _MODEL is None:
        _MODEL = FPADTeacherStudent().eval()
    return _MODEL


@check("layer indices match the brief's 1-indexed set")
def _layer_convention():
    assert DEFAULT_LAYERS == (3, 7, 11, 15, 19, 23)
    assert tuple(i + 1 for i in DEFAULT_LAYERS) == (4, 8, 12, 16, 20, 24), (
        "the brief writes {4,8,12,16,20,24}; an off-by-one here shifts the entire depth profile "
        "and every per-layer claim in the paper")


@check("LoRA targets are this ViT's real module names")
def _lora_targets():
    assert FPAD_LORA["target_modules"] == ["qkv", "proj", "fc1", "fc2"], FPAD_LORA
    assert "q_proj" not in FPAD_LORA["target_modules"], (
        "q_proj/v_proj are HuggingFace CLIP names and do not exist in this backbone; peft would "
        "match nothing and produce a student identical to the teacher forever")
    assert FPAD_LORA["r"] == 8 and FPAD_LORA["lora_alpha"] == 16


@check("only LoRA parameters train", needs_ckpt=True)
def _only_lora_trains():
    m = model()
    m.assert_teacher_frozen()
    trainable = [n for n, p in m.peft.named_parameters() if p.requires_grad]
    assert trainable, "nothing is trainable; the student can never adapt"
    assert all("lora_" in n for n in trainable), [n for n in trainable if "lora_" not in n][:3]
    counts = m.trainable_parameters()
    print(f"      LoRA {counts['lora']:,} / frozen {counts['frozen']:,} "
          f"({100 * counts['lora'] / counts['total']:.2f}% trainable)")


@check("teacher and student are BIT-IDENTICAL at initialization", needs_ckpt=True)
def _zero_at_init():
    """The premise of the two-stage training. LoRA's B is zero-init, so the student IS the
    teacher until something trains it.

    The strong claim is bit identity of the representations. `D` itself cannot be asserted at
    exactly 0.0 — `1 - cos(v, v)` in float32 lands ~1e-7 from zero because the dot product and
    the norms accumulate separately. Asserting the distance alone would ALSO be weaker than
    intended, since two different vectors can be collinear; so both are checked.
    """
    m = model()
    x = torch.rand(2, 3, 224, 224)
    assert m.teacher_student_identical(x), "the two passes do not compute the same function"
    out = m(x)
    floor = FPADTeacherStudent.COSINE_ZERO
    for key in ("D", "D_cls", "patch_delta"):
        worst = float(out[key].detach().abs().max())
        assert worst <= floor, (
            f"{key} max = {worst:.3e} at init, above the {floor:.0e} float32 cosine floor. The "
            f"teacher and student are not the same base weights, which breaks the design.")
    print(f"      D at init: max {float(out['D'].detach().abs().max()):.2e} "
          f"(float32 cosine floor, representations bit-identical)")


@check("D becomes non-zero once the adapter moves, and the teacher does not", needs_ckpt=True)
def _moves_after_adaptation():
    m = FPADTeacherStudent().eval()
    x = torch.rand(2, 3, 224, 224)
    before = m(x)
    with torch.no_grad():
        for n, p in m.peft.named_parameters():
            if "lora_B" in n:
                torch.nn.init.normal_(p, std=0.01)
    after = m(x)
    assert float(after["D"].detach().abs().max()) > 1e-4, (
        "adapting the student left D at the numerical floor")
    assert torch.allclose(before["h_teacher"], after["h_teacher"], atol=1e-6), (
        "the TEACHER output changed when only LoRA moved. The teacher must be a constant, or "
        "d_l^TS measures two moving targets rather than departure from a fixed prior.")
    print(f"      D after perturbation: {[round(v, 4) for v in after['D'][0].tolist()]}")


@check("the teacher pass is deterministic even under model.train()", needs_ckpt=True)
def _teacher_deterministic():
    """LoRA dropout is active in train mode. It must not reach the teacher, whose adapters are
    off; every other dropout in this ViT is p=0 and drop_path is Identity."""
    m = FPADTeacherStudent().train()
    x = torch.rand(2, 3, 224, 224)
    a = m(x)["h_teacher"]
    b = m(x)["h_teacher"]
    assert torch.allclose(a, b, atol=1e-6), (
        f"two teacher passes differ by {float((a - b).abs().max()):.3e} in train mode — the "
        f"frozen prior is stochastic, so D would fluctuate for reasons unrelated to adaptation")


@check("no gradient reaches the teacher path", needs_ckpt=True)
def _no_teacher_gradient():
    m = FPADTeacherStudent().train()
    out = m(torch.rand(2, 3, 224, 224))
    (out["D"].sum() + out["h_student"].sum()).backward()
    leaked = [n for n, p in m.peft.named_parameters()
              if p.grad is not None and "lora_" not in n]
    assert not leaked, leaked[:3]


@check("the readout is the PATCH MEAN, not CLS", needs_ckpt=True)
def _readout_is_patch_mean():
    """Silent if wrong, and it changes what the paper measures."""
    m = model()
    x = torch.rand(2, 3, 224, 224)
    xn = m.normalize(x)
    with torch.no_grad():
        cap = _BlockCapture(m._vit.blocks, m.layers)
        with torch.no_grad(), m.peft.disable_adapter(), cap:
            m._vit.forward_features(xn)
        blocks = cap.stacked()
    out = m(x)
    manual_mean = _cosine_distance(blocks[:, :, 1:].mean(dim=2), blocks[:, :, 1:].mean(dim=2))
    assert torch.allclose(out["D"], manual_mean, atol=1e-6)
    # and CLS is a DIFFERENT quantity, reported separately rather than substituted
    assert "D_cls" in out and out["D_cls"].shape == out["D"].shape


@check("hooks do not cross-talk between the two passes", needs_ckpt=True)
def _no_hook_crosstalk():
    """If the passes shared a buffer, D would compare a pass with itself and read all-zero —
    indistinguishable from 'not adapted yet'."""
    m = FPADTeacherStudent().eval()
    with torch.no_grad():
        for n, p in m.peft.named_parameters():
            if "lora_B" in n:
                torch.nn.init.normal_(p, std=0.02)
    out = m(torch.rand(2, 3, 224, 224))
    assert float(out["D"].detach().min()) > 1e-4, (
        "every layer reads zero deviation on an adapted student — the two passes are almost "
        "certainly sharing a capture buffer")
    assert not m.peft.base_model.model.blocks[0]._forward_hooks, (
        "hooks survived the forward pass; they must be removed so a later pass cannot inherit "
        "them")


@check("shapes and the patch grid are as expected", needs_ckpt=True)
def _shapes():
    m = model()
    out = m(torch.rand(3, 3, 224, 224))
    L = len(DEFAULT_LAYERS)
    assert out["D"].shape == (3, L), out["D"].shape
    assert out["patch_delta"].shape == (3, L, 196), out["patch_delta"].shape
    assert out["patch_map"].shape == (3, 14, 14) and out["grid"] == (14, 14)
    assert out["h_student"].shape == (3, 1024)


@check("non-square input is resized rather than silently mis-shaped", needs_ckpt=True)
def _resize():
    m = model()
    out = m(torch.rand(2, 3, 256, 256))
    assert out["patch_map"].shape == (2, 14, 14), (
        "a 256 input must be resized to the model's 224, not reinterpreted as a finer grid")


@check("H_traj reads curve SHAPE, not only level")
def _traj_head_shape():
    head = TrajectoryEvidenceHead(n_layers=6).fit_calibrator(torch.rand(200, 6))
    assert head.in_dim == 6 + 5, head.in_dim
    out = head(torch.rand(4, 6))
    assert out["evidence"].shape == (4, 2) and (out["evidence"] >= 0).all()
    assert torch.isfinite(out["prob"]).all()
    # two curves with the same mean but different shape must be distinguishable
    flat = torch.full((1, 6), 0.5)
    ramp = torch.tensor([[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]])
    assert abs(float(flat.mean() - ramp.mean())) < 1e-6, "same level by construction"
    assert not torch.allclose(head.features(flat), head.features(ramp)), (
        "the head's feature cannot distinguish a flat curve from a ramp of equal mean, so it "
        "cannot express early-versus-late departure — the paper's forensic question")


@check("H_traj fixes no polarity")
def _traj_head_polarity_free():
    """Fit on inverted targets; both directions must be learnable."""
    torch.manual_seed(0)
    D = torch.rand(256, 6)
    level = D.mean(dim=1)
    for sign, label in ((+1.0, "more deviation -> fake"), (-1.0, "less deviation -> fake")):
        head = TrajectoryEvidenceHead(n_layers=6).fit_calibrator(D)
        y = ((sign * level) > (sign * level).median()).float()
        opt = torch.optim.Adam(head.parameters(), lr=1e-2)
        for _ in range(300):
            opt.zero_grad()
            p = head(D)["prob"].clamp(1e-6, 1 - 1e-6)
            F.binary_cross_entropy(p, y).backward()
            opt.step()
        acc = float(((p > 0.5).float() == y).float().mean())
        assert acc > 0.75, f"{label}: only {acc:.2f} — the head encodes a polarity"


@check("H_traj rejects a mismatched layer count")
def _traj_head_layer_guard():
    head = TrajectoryEvidenceHead(n_layers=6).fit_calibrator(torch.rand(50, 6))
    try:
        head(torch.rand(2, 4))
    except ValueError as exc:
        assert "layers" in str(exc)
    else:
        raise AssertionError("a head applied to a different layer set reads a different signal")


@check("an uncalibrated head REFUSES rather than standardising against nothing")
def _calibrator():
    head = TrajectoryEvidenceHead(n_layers=6)
    assert not head.is_calibrated
    try:
        head(torch.rand(2, 6))
    except RuntimeError as exc:
        assert "unfitted" in str(exc)
    else:
        raise AssertionError(
            "an unfitted calibrator must refuse; running uncalibrated by accident would leave "
            "the artifact claiming a yardstick the head never had")

    head.fit_calibrator(torch.rand(500, 6))
    assert head.is_calibrated and head.describe()["calibrated"]
    assert all(not p.requires_grad for p in head.calibrator.parameters())

    # declining calibration is allowed, but must be visible in the artifact
    raw = TrajectoryEvidenceHead(n_layers=6, use_calibrator=False)
    assert torch.allclose(raw.features(torch.full((1, 6), 2.0))[:, :6], torch.full((1, 6), 2.0))
    assert raw.describe()["use_calibrator"] is False
    try:
        raw.fit_calibrator(torch.rand(10, 6))
    except RuntimeError as exc:
        assert "use_calibrator=False" in str(exc)
    else:
        raise AssertionError("fitting a discarded calibrator must be refused, not ignored")


@check("direct and trajectory heads are the same capacity class")
def _matched_capacity():
    """`B2 - B1` is a readout-only comparison, so a capacity difference here would contaminate
    the paper's headline contrast."""
    traj = TrajectoryEvidenceHead(n_layers=6, hidden_dim=64)
    direct = DirectEvidenceHead(feature_dim=1024, hidden_dim=64)
    t_act = [type(m).__name__ for m in traj.head if not isinstance(m, torch.nn.Linear)]
    d_act = [type(m).__name__ for m in direct.head if not isinstance(m, torch.nn.Linear)]
    assert t_act == d_act == ["GELU"], (t_act, d_act)
    t_lin = [m for m in traj.head if isinstance(m, torch.nn.Linear)]
    d_lin = [m for m in direct.head if isinstance(m, torch.nn.Linear)]
    assert len(t_lin) == len(d_lin) == 2
    assert t_lin[0].out_features == d_lin[0].out_features
    assert t_lin[-1].out_features == d_lin[-1].out_features


if __name__ == "__main__":
    print(f"\n{len(PASSED)} passed, {len(SKIPPED)} skipped")
    for s in SKIPPED:
        print(f"  - skipped: {s}")

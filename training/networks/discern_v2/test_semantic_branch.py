"""Tests for Branch A — CLIP-LoRA semantic evidence (V1 spec §3, §10, §19).

    python training/networks/discern_v2/test_semantic_branch.py

Tests that build CLIP need the local HF cache; they skip with a message when it is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from networks.discern_v2 import semantic_branch as S  # noqa: E402
from networks.discern_v2.ds_fusion import Opinion  # noqa: E402

PASSED, SKIPPED = [], []


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


_cached: dict[bool, S.SemanticEvidenceBranch] = {}


def branch(enable_lora: bool = True) -> S.SemanticEvidenceBranch:
    if enable_lora not in _cached:
        try:
            _cached[enable_lora] = S.SemanticEvidenceBranch(enable_lora=enable_lora)
        except Exception as exc:  # noqa: BLE001 — missing HF cache or peft
            raise SkipTest(f"cannot build CLIP: {type(exc).__name__}: {str(exc)[:80]}")
    return _cached[enable_lora]


# ---------------------------------------------------------------- ported constants


@check("§3: the ported LoRA settings match what DiCoME's config actually declares")
def _lora_constants():
    assert S.DICOME_LORA["target_modules"] == ["q_proj", "v_proj"]
    assert S.DICOME_LORA["r"] == 8
    assert S.DICOME_LORA["lora_alpha"] == 16
    assert S.DICOME_LORA["lora_dropout"] == 0.1
    assert S.DICOME_LORA["bias"] == "none"
    assert S.DICOME_BACKBONE == "openai/clip-vit-large-patch14"
    assert S.DICOME_FEATURE_DIM == 64


@check("§10: the augmentation ports DiCoME's order and ranges, blur/jitter unconditional")
def _augmentation():
    a = S.DICOME_AUGMENTATION
    assert a["order"] == ["random_horizontal_flip", "random_affine", "gaussian_blur",
                          "color_jitter"]
    assert a["gaussian_blur"]["sigma"] == (0.1, 2.0)
    assert a["color_jitter"]["brightness"] == 0.2 and a["color_jitter"]["contrast"] == 0.2
    try:
        transform = S.dicome_train_transform()
    except ImportError as exc:
        raise SkipTest(str(exc))
    names = [type(t).__name__ for t in transform.transforms]
    assert names == ["RandomHorizontalFlip", "RandomAffine", "GaussianBlur", "ColorJitter"], names


@check("§10: bias/norm parameters land in the zero-weight-decay group")
def _param_groups():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
    groups = S.dicome_param_groups(model, weight_decay=0.01)
    assert groups[0]["weight_decay"] == 0.01 and groups[1]["weight_decay"] == 0.0
    n_no_decay = sum(p.numel() for p in groups[1]["params"])
    # Only the two biases (4 + 4). The LayerNorm sits in a Sequential, so its parameters are
    # named "1.weight"/"1.bias": the gain matches neither "bias" nor "norm" and DOES receive
    # weight decay. That is the ported rule's real behaviour, asserted so it stays visible.
    assert n_no_decay == 8, n_no_decay
    assert sum(p.numel() for p in groups[0]["params"]) == 20, (
        "Linear.weight (16) + the unmatched LayerNorm gain (4)")


@check("§10: a norm parameter the name rule misses is reported, not silently decayed")
def _param_groups_warn(capture=[]):
    import logging

    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))

    class Grab(logging.Handler):
        def emit(self, record):
            capture.append(record.getMessage())

    log = logging.getLogger(S.__name__)
    handler = Grab()
    log.addHandler(handler)
    try:
        S.dicome_param_groups(model)
    finally:
        log.removeHandler(handler)
    assert any("normalisation parameter" in m for m in capture), capture

    # our own branch names it `self.norm`, so nothing is missed there
    capture.clear()
    named = torch.nn.Module()
    named.norm = torch.nn.LayerNorm(4)
    log.addHandler(handler)
    try:
        S.dicome_param_groups(named)
    finally:
        log.removeHandler(handler)
    assert not capture, capture


@check("frozen parameters are excluded from the optimizer groups entirely")
def _param_groups_skip_frozen():
    model = torch.nn.Linear(4, 4)
    model.weight.requires_grad_(False)
    groups = S.dicome_param_groups(model)
    assert sum(p.numel() for p in groups[0]["params"]) == 0
    assert sum(p.numel() for p in groups[1]["params"]) == 4, "only the bias remains trainable"


# ---------------------------------------------------------------- needs CLIP


@check("§19: only LoRA adapters are trainable inside CLIP")
def _lora_only():
    b = branch()
    counts = b.assert_lora_only()
    assert counts["lora"] > 0, counts
    assert counts["frozen_backbone"] > 100_000_000, counts     # ViT-L/14 vision tower
    assert counts["lora"] < 0.01 * counts["frozen_backbone"], (
        f"LoRA should be a small fraction of the backbone: {counts}")


@check("§3: output is non-negative evidence (B, 2) and a valid opinion")
def _evidence():
    b = branch()
    out = b(torch.randn(2, 3, 224, 224))
    e = out["evidence"]
    assert e.shape == (2, 2), e.shape
    assert (e >= 0).all(), "softplus evidence must be non-negative"
    Opinion.from_evidence(e).assert_normalized()
    assert out["feature"].shape == (2, S.DICOME_FEATURE_DIM)
    assert out["raw_feature"].shape == (2, 1024)


@check("§19: a training step moves LoRA + head, and leaves base CLIP byte-identical")
def _gradient_flow():
    b = branch()
    base_before = {n: p.detach().clone() for n, p in b.vision_model.named_parameters()
                   if "lora_" not in n}
    lora_before = {n: p.detach().clone() for n, p in b.vision_model.named_parameters()
                   if "lora_" in n}

    opt = torch.optim.AdamW(S.dicome_param_groups(b), lr=1e-3)
    out = b(torch.randn(2, 3, 224, 224))
    # any loss that depends on the evidence; the point is where gradient may flow
    loss = torch.nn.functional.cross_entropy(out["evidence"] + 1e-6, torch.tensor([0, 1]))
    loss.backward()
    opt.step()

    base_moved = [n for n, p in b.vision_model.named_parameters()
                  if "lora_" not in n and not torch.equal(p.detach(), base_before[n])]
    assert not base_moved, f"base CLIP parameters moved: {base_moved[:3]}"
    base_grad = [n for n, p in b.vision_model.named_parameters()
                 if "lora_" not in n and p.grad is not None and p.grad.abs().sum() > 0]
    assert not base_grad, f"base CLIP received gradient: {base_grad[:3]}"

    lora_moved = [n for n, p in b.vision_model.named_parameters()
                  if "lora_" in n and not torch.equal(p.detach(), lora_before[n])]
    assert lora_moved, "no LoRA parameter moved — the anchor did not learn"
    assert b.head.fc2.weight.grad is not None, "the evidence head must receive gradient"


@check("the no-LoRA control freezes CLIP entirely (a §24 arm, not a V1 configuration)")
def _no_lora_control():
    b = branch(enable_lora=False)
    assert not any(p.requires_grad for p in b.vision_model.parameters())
    counts = b.assert_lora_only()
    assert counts["lora"] == 0 and counts["head"] > 0


if __name__ == "__main__":
    print(f"\n{len(PASSED)} passed, {len(SKIPPED)} skipped")
    for s in SKIPPED:
        print(f"  - skipped: {s}")
    sys.exit(0 if PASSED else 1)

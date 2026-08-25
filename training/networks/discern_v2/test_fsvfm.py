"""Tests for the frozen FS-VFM reference encoder (V1 spec §4, §6, §19).

Run: python training/networks/discern_v2/test_fsvfm.py

The weight-loading tests need `weights/FS-VFM/checkpoint-599.pth` (4.4 GB, not in git). They
skip with a clear message when it is absent rather than failing, so the suite stays runnable on
a machine without the artifact; the tests that do not need weights always run.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from networks.discern_v2 import fsvfm_encoder as F  # noqa: E402

PASSED, SKIPPED = [], []


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


class SkipTest(Exception):
    pass


def require_weights() -> Path:
    if not os.path.isfile(F.DEFAULT_CHECKPOINT):
        raise SkipTest(f"no checkpoint at {F.DEFAULT_CHECKPOINT}")
    return F.DEFAULT_CHECKPOINT


_cached: dict[str, F.FrozenFSVFM] = {}


def encoder(pooling: str = "global_pool") -> F.FrozenFSVFM:
    """One instance per pooling mode — loading ViT-L four times is 4x30 s for no extra coverage."""
    require_weights()
    if pooling not in _cached:
        _cached[pooling] = F.FrozenFSVFM(pooling=pooling)
    return _cached[pooling]


# ---------------------------------------------------------------- no weights needed


@check("normalization comes from the shipped file and is NOT ImageNet")
def _norm_from_file():
    if not os.path.isfile(F.DEFAULT_MEAN_STD):
        raise SkipTest("pretrain_ds_mean_std.txt absent")
    mean, std = F.load_normalization()
    assert mean == F.EXPECTED_MEAN, mean
    assert std == F.EXPECTED_STD, std
    imagenet = (0.485, 0.456, 0.406)
    assert max(abs(a - b) for a, b in zip(mean, imagenet)) > 0.01, (
        "FS-VFM statistics must differ from ImageNet's; §4 turns on exactly this")


@check("a bad normalization file is refused, not averaged")
def _norm_disagreement(tmp=Path("/tmp/discern_bad_mean_std.txt")):
    tmp.write_text(
        '{"mean": [0.5, 0.4, 0.3], "std": [0.2, 0.2, 0.2]}\n'
        '{"mean": [0.1, 0.1, 0.1], "std": [0.2, 0.2, 0.2]}\n')
    try:
        F.load_normalization(tmp)
    except ValueError as exc:
        assert "more than one normalisation" in str(exc)
    else:
        raise AssertionError("two different normalisations in one file must be refused")
    finally:
        tmp.unlink(missing_ok=True)


@check("an unknown pooling mode is refused")
def _bad_pooling():
    try:
        F.FrozenFSVFM(pooling="mean_patch_but_spelled_wrong")
    except ValueError as exc:
        assert "pooling must be one of" in str(exc)
    else:
        raise AssertionError("unknown pooling must raise")


@check("a missing checkpoint explains where to get the PRETRAINED artifact")
def _missing_ckpt():
    try:
        F.FrozenFSVFM(checkpoint="/nonexistent/checkpoint-599.pth")  # unreadable parent
    except FileNotFoundError as exc:
        msg = str(exc)
        assert "NOT a fine-tuned downstream model" in msg
        assert "checkpoint-599.pth" in msg
    else:
        raise AssertionError("a missing checkpoint must raise with instructions")


# ---------------------------------------------------------------- weights needed


@check("every encoder tensor loads; only fc_norm/head are missing")
def _complete_load():
    enc = encoder()
    missing = enc.load_report["missing_keys"]
    assert all(k.startswith(("fc_norm.", "head.")) for k in missing), missing
    assert enc.load_report["epoch"] == 599, enc.load_report["epoch"]
    assert not enc.load_report["dropped_shape_mismatch"], enc.load_report


@check("a partial encoder load is refused rather than silently randomised")
def _partial_load_refused(tmp=Path("/tmp/discern_partial_fsvfm.pth")):
    require_weights()
    blob = torch.load(str(F.DEFAULT_CHECKPOINT), map_location="cpu", weights_only=False)
    state = {k: v for k, v in blob["model"].items() if not k.startswith("blocks.3.")}
    torch.save({"model": state}, tmp)
    try:
        F.FrozenFSVFM(checkpoint=tmp)
    except RuntimeError as exc:
        assert "Refusing to run a partly-random" in str(exc), str(exc)
    else:
        raise AssertionError("a checkpoint missing encoder blocks must be refused")
    finally:
        tmp.unlink(missing_ok=True)


@check("output is (B, 1024) and deterministic across calls")
def _shape_and_determinism():
    enc = encoder()
    x = torch.randn(2, 3, 224, 224)
    a, b = enc(x), enc(x)
    assert a.shape == (2, F.FEATURE_DIM), a.shape
    assert torch.equal(a, b), "frozen encoder must be deterministic (no dropout/drop-path)"


@check("cls pooling is a different representation from global_pool")
def _pooling_differs():
    x = torch.randn(2, 3, 224, 224)
    g, c = encoder("global_pool")(x), encoder("cls")(x)
    assert g.shape == c.shape == (2, F.FEATURE_DIM)
    assert not torch.allclose(g, c), (
        "the two pooling rules must differ, or the parity comparison between them is vacuous")


@check("§19: FS-VFM receives ZERO gradient through a full training step")
def _zero_gradient():
    enc = encoder()
    head = torch.nn.Linear(F.FEATURE_DIM, 2)
    opt = torch.optim.SGD(head.parameters(), lr=0.1)
    before = {n: p.detach().clone() for n, p in enc.named_parameters()}

    x = torch.randn(4, 3, 224, 224)
    loss = torch.nn.functional.cross_entropy(head(enc(x)), torch.tensor([0, 1, 0, 1]))
    loss.backward()
    opt.step()

    got_grad = [n for n, p in enc.named_parameters() if p.grad is not None]
    assert not got_grad, f"FS-VFM parameters received gradient: {got_grad[:5]}"
    moved = [n for n, p in enc.named_parameters() if not torch.equal(p.detach(), before[n])]
    assert not moved, f"FS-VFM parameters moved: {moved[:5]}"
    assert head.weight.grad is not None, "the trainable head must still learn"


@check("train() cannot take the encoder out of eval")
def _stays_eval():
    enc = encoder()
    enc.train(True)
    assert not enc.training, "FS-VFM must stay in eval mode"
    assert not enc.model.training, "the inner ViT must stay in eval mode"
    enc.assert_frozen()


@check("fingerprint is stable and provenance records the artifact")
def _fingerprint():
    enc = encoder()
    assert enc.fingerprint() == enc.fingerprint()
    prov = enc.provenance()
    assert prov["feature_dim"] == F.FEATURE_DIM
    assert prov["epoch"] == 599
    assert prov["mean"] == list(F.EXPECTED_MEAN)


if __name__ == "__main__":
    print(f"\n{len(PASSED)} passed, {len(SKIPPED)} skipped")
    if SKIPPED:
        print("skipped (need weights/FS-VFM/checkpoint-599.pth):")
        for s in SKIPPED:
            print(f"  - {s}")
    sys.exit(0 if PASSED else 1)

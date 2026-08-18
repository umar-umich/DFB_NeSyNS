#!/usr/bin/env python3
"""Regression test: --resume must be able to load the checkpoints we write.

Why this exists
---------------
`Trainer.save_resume_state` writes optimizer/scheduler state and a `best_metrics` dict
whose values include numpy scalars. PyTorch 2.6 changed the default of `torch.load`'s
`weights_only` argument from False to True, and the safe unpickler permits only tensors —
so from that release onward `load_resume_state` raised

    _pickle.UnpicklingError: Weights only load failed ...
    Unsupported global: GLOBAL numpy.core.multiarray.scalar

and `--resume` failed outright, on a checkpoint this repo had just written itself.

That cost a real migration: D3(P1d) was stopped at an epoch boundary to be moved to a
faster GPU, and could not be resumed, so 5 epochs had to be retrained. The failure mode is
nasty because nothing is wrong with the checkpoint — only with how it is read — and it
surfaces at resume time, which is exactly when you have already thrown away the process
you were hoping to continue.

    python tests/test_resume_checkpoint_load.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch


def _resume_shaped_checkpoint() -> dict:
    """The parts of a real resume checkpoint that matter for deserialisation.

    `best_metrics` holds numpy scalars because the metric helpers return numpy types; that
    is what the safe unpickler rejects, so it must be present for this test to mean
    anything.
    """
    return {
        "model": {"w": torch.zeros(2, 2)},
        "optimizer": {"state": {}, "param_groups": [{"lr": 1e-4}]},
        "scheduler": None,
        "scaler": None,
        "epoch": 5,
        "extra": {"early_stop_counter": 3},
        "best_metrics": {
            "FaceForensics++": {"auc": np.float64(0.9391), "acc": np.float32(0.8976)},
            "avg": {"auc": np.float64(0.9089)},
        },
    }


def test_safe_unpickler_rejects_a_resume_checkpoint():
    """Pin the upstream behaviour this fix exists for. If PyTorch ever relaxes it, this
    test failing is the signal to revisit the workaround rather than keep it forever."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "last_checkpoint.pth"
        torch.save(_resume_shaped_checkpoint(), p)
        try:
            torch.load(p, map_location="cpu", weights_only=True)
        except Exception as e:
            assert "numpy" in str(e) or "Unsupported global" in str(e), str(e)[:200]
            print("  ok: weights_only=True still rejects a resume-shaped checkpoint")
            return
        print("  note: weights_only=True now loads this — upstream may have relaxed; "
              "the explicit weights_only=False remains correct but is no longer required")


def test_resume_checkpoint_round_trips():
    """What load_resume_state actually needs: every key back, with usable values."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "last_checkpoint.pth"
        original = _resume_shaped_checkpoint()
        torch.save(original, p)

        ck = torch.load(p, map_location="cpu", weights_only=False)

        for k in ("model", "optimizer", "epoch", "extra", "best_metrics"):
            assert k in ck, f"missing {k!r}"
        assert int(ck["epoch"]) == 5
        assert ck["extra"]["early_stop_counter"] == 3
        assert abs(float(ck["best_metrics"]["avg"]["auc"]) - 0.9089) < 1e-9
        assert torch.equal(ck["model"]["w"], original["model"]["w"])
        print("  ok: resume checkpoint round-trips with weights_only=False")


def test_trainer_uses_weights_only_false():
    """Guard the call site itself, so a well-meaning cleanup cannot silently re-break
    --resume. Checked by source inspection because constructing a Trainer requires a full
    model, dataloaders and an optimizer."""
    src = (Path(__file__).resolve().parents[1]
           / "training/trainer/trainer.py").read_text()
    start = src.index("def load_resume_state")
    body = src[start:start + 2500]
    assert "weights_only=False" in body, (
        "load_resume_state must pass weights_only=False; without it --resume cannot read "
        "the checkpoints this trainer writes (see this file's docstring)")
    print("  ok: load_resume_state passes weights_only=False")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"resume checkpoint load ({len(tests)} tests)\n")
    for t in tests:
        t()
    print(f"\nall {len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

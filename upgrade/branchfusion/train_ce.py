#!/usr/bin/env python3
"""The CE counterpart of a branch: same trunk, same budget, logits and cross-entropy instead of
evidence and EDL.

    CUDA_VISIBLE_DEVICES=1 <temporal_vjepa2 python> upgrade/branchfusion/train_ce.py \
        --model vjepa2 --lora --augment B --seed 42

WHY THIS EXISTS. DiCoME's Table 2(b) row (D) replaces the evidential loss with cross-entropy and
reports a drop. Our existing CE reference is DISCERN_Ext's arm D -- three DIFFERENT views on a
different trunk -- and it does not drop: it ties at 0.8639 OOD AUROC against the evidential arm's
0.8633. That comparison cannot settle anything about THIS system, because nothing is held fixed
between the two sides. This file holds everything fixed except the head and the loss.

WHAT CHANGES, and nothing else:
  head      `softplus(Linear(...))` emitting evidence  ->  the same Linear emitting LOGITS
  loss      `evidential_loss_discern_ext` (EDL + KL annealing)  ->  cross-entropy
  opinion   p = alpha_fake / S, u = K / S  ->  p = softmax(logits)[1], and u is UNDEFINED

The last line is the point of the experiment, not a limitation of it: a CE branch has no vacuity,
so the only uncertainty it can offer downstream is the decision margin. Selection, the clip
sampler, the arm-B augmentation, the LoRA budget, the epoch count and the development protocol are
untouched, so a difference between the two arms is the loss and the readout.

VACUITY IS EXPORTED AS 1 - |2p - 1|, the margin rescaled to [0, 1], so a CE branch can enter the
same fusion machinery as an evidential one. That is NOT a vacuity and must never be read as one:
it is a confidence transform with no Dirichlet behind it, and the export records it in a
`uncertainty_is_margin` flag so no downstream table can quietly treat the two as the same object.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
HANDOFF = ROOT / "tbiom/handoff/PECORE_L14_TemporalBranch/code"
sys.path[:0] = [str(HANDOFF), str(HANDOFF / "tbiom"), str(HANDOFF / "tbiom/temporal"),
                str(ROOT / "upgrade/branchfusion"), str(ROOT / "analysis/tbiom/temporal")]

import numpy as np                                                  # noqa: E402
import pandas as pd                                                 # noqa: E402
import train_branch as TB                                           # noqa: E402
import train_om_pecore as OM                                        # noqa: E402


class _Identity:
    """`train_branch` holds `torch.nn.functional` as its own module-level `F` and ends
    `Branch.forward` with `F.softplus(...)`. Swapping THAT reference turns the evidential head
    into a logit head for the base trunks without touching torch globally and without copying
    the forward pass, which would then drift from Dr Kutub's as his file changes."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def softplus(self, x, *a, **k):
        return x


def install_ce() -> None:
    """Logits instead of evidence, cross-entropy instead of EDL. Nothing else moves."""
    TB.F = _Identity(TB.F)
    OM.CE_MODE = True

    def loss(ev, y, K, ep, epochs):
        # `ev` is now logits. The signature keeps `ep`/`epochs` because the trainer passes them
        # for EDL's KL annealing; cross-entropy has no schedule and ignores them, which is the
        # honest difference between the two objectives rather than a harness detail.
        return F.cross_entropy(ev.float(), y, reduction="none")

    TB.evidential_loss_discern_ext = loss

    def opinions(model, ds, dev, process=None, workers=4, shuffle=False):
        """The evidential readout rebuilt for logits.

        This is a REPLACEMENT, not a wrapper: `train_branch.opinions` forms `alpha = out + 1` and
        divides, which is meaningless once `out` is a logit, and no post-hoc correction can undo
        it. Everything else -- the loader, the clip sampling, the processing hook, the per-video
        row -- is the same so the two arms are scored over identical videos in identical order.

        `u` is the decision MARGIN rescaled to [0, 1], not a vacuity: a cross-entropy model has no
        Dirichlet and therefore no epistemic quantity. It is written into the same column only so
        a CE branch can pass through the same fusion code, and `uncertainty_is_margin` in the run
        record says so.
        """
        loader = torch.utils.data.DataLoader(
            TB.Clips(ds, model.preprocess, model.T, shuffle), batch_size=TB.BATCH,
            num_workers=workers, worker_init_fn=lambda _: torch.set_num_threads(1))
        out, items = [], loader.dataset.items
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for px, y, i in loader:
                px = px.to(dev)
                if process:
                    B_, T_ = px.shape[:2]
                    px = TB.apply_processing(px.flatten(0, 1).float(), process,
                                             model.mean, model.std).view(B_, T_, *px.shape[2:])
                p = torch.softmax(model(px).float(), dim=1)[:, 1].double().cpu().numpy()
                for k, ii in enumerate(i.tolist()):
                    out.append((items[ii][2], int(y[k]), float(p[k]),
                                float(1.0 - abs(2.0 * p[k] - 1.0))))
        return pd.DataFrame(out, columns=["v", "y", "p", "u"])

    TB.opinions = opinions


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=0, help="PE-Core block tap, as in train_om_pecore")
    ap.add_argument("--temporal", default="mixer", choices=["mixer", "seq"])
    ap.add_argument("--smoke", type=int, default=0, metavar="N",
                    help="end-to-end rehearsal: N training clips, 1 epoch, 1 test export; the "
                         "DEVELOPMENT sets stay whole because the trainer asserts full coverage")
    a, rest = ap.parse_known_args()
    OM.redirect_outputs()
    OM.install_opinion_retry()
    if a.block:
        if a.temporal == "seq":
            OM.install_tpos()
        OM.install(None, a.block, a.temporal, keep_plain=False)
    install_ce()
    if a.smoke:
        TB.EPOCHS = 1
        orig_init, keep = TB.Clips.__init__, set(TB.DEV)

        def cinit(self, ds, *args, **kw):
            orig_init(self, ds, *args, **kw)
            if ds not in keep and len(self.items) > a.smoke:
                step = max(1, len(self.items) // a.smoke)
                self.items = self.items[::step][:a.smoke]

        TB.Clips.__init__ = cinit
        TB.TESTS = TB.TESTS[:1]
    sys.argv = [sys.argv[0]] + rest + [
        "--run-suffix", "_ce" + (f"_b{a.block}" if a.block else "")
        + ("_seq" if a.temporal == "seq" else "")]
    print(f"CE arm: logits + cross-entropy; argv {' '.join(sys.argv[1:])}", flush=True)
    TB.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

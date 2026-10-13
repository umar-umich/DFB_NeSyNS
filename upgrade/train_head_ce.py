#!/usr/bin/env python3
"""App's CROSS-ENTROPY head -- the last piece of the CE arm.

    <discern_ext python> upgrade/train_head_ce.py --backbone pe_core_lLN --seed 0

`train_head.py` trains the evidential Appearance head: Linear 256, GELU, dropout 0.1, Linear 2,
softplus evidence, EDL loss. This trains the same trunk-and-head with the softplus removed and
the loss replaced by cross-entropy, so the CE arm of the framework has an Appearance branch built
the same way the other four CE branches were.

Everything else -- the cached features, the epoch count, the optimiser, the schedule, the
selection rule -- is `train_head`'s, imported rather than copied, so the two arms differ in the
head and the objective and in nothing else.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(ROOT / "upgrade")]

import train_head as TH                                          # noqa: E402


class CEHead(torch.nn.Module):
    """`train_head.Head` without the softplus: the same two layers emitting LOGITS."""

    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.fc1 = torch.nn.Linear(d, 256)
        self.drop = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(256, 2)

    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


def main() -> int:
    TH.Head = CEHead
    # The trainer calls this as `loss(ev, y, 2, ep, EPOCHS)` and reduces it itself; cross-entropy
    # has no KL schedule, so `ep` and `epochs` are ignored -- which is the real difference between
    # the objectives rather than a detail of the harness.
    TH.evidential_loss_discern_ext = lambda ev, y, K, ep, epochs: F.cross_entropy(ev.float(), y)
    # `video_p` forms `alpha = ev + 1` and takes `alpha_1 / sum(alpha)`. For logits that is
    # meaningless, so it is REPLACED -- wrapping it could not undo arithmetic already done. The
    # frame-mean reduction below it is kept exactly, because that is the Appearance view's own
    # reduction and changing it would make the two arms incomparable for a second reason.
    import numpy as np

    @torch.no_grad()
    def video_p(head, D, dev, bs=8192):
        head.eval()
        out = []
        for i in range(0, len(D["x"]), bs):
            logits = head(torch.from_numpy(D["x"][i:i + bs]).to(dev))
            out.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
        p = np.concatenate(out)
        n = len(D["order"])
        cnt = np.bincount(D["inv"], minlength=n).astype(float)
        return np.bincount(D["inv"], weights=p, minlength=n) / cnt

    TH.video_p = video_p

    TH.OUT = ROOT / "analysis/upgrade/heads_ce"
    TH.OUT.mkdir(parents=True, exist_ok=True)
    print("App CE head: logits + cross-entropy, same trunk and schedule", flush=True)
    return TH.main()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Train the authenticity teacher FPNet needs, split-safe, to unblock arm D.

    python analysis/tbiom/train_authenticity_teacher.py

WHY WE TRAIN IT RATHER THAN DOWNLOAD IT. FreqBlender's `main.py:50` loads
`net/resnet/checkpoints/resnet34.tar` into `Detector` = resnet34 with a 2-class head. That file
is on Baidu Pan behind an account wall: the share link redirects to /share/init and the `?pwd=`
only pre-fills the extraction code. There is no GitHub release and no mirror.

Training our own is not a downgrade, it is the better option, and the authors sanction it
(README line 92: "Or, you can choose to simply train a weight yourself"). A released checkpoint
has unknown training exposure, and the brief is explicit that such weights are usable only for
a labelled implementation diagnostic, never in a clean OOD experiment. A teacher trained here
on FF++ TRAIN ONLY is split-safe by construction and can enter the formal comparison.

WHAT MUST BE DECLARED. Arm D's generator then depends on an auxiliary model we trained. The
brief requires accounting for all auxiliary training data and forbids calling the pipeline
"real-only" because the final detector uses SBI donors. This teacher sees FF++ train reals AND
fakes, and that is recorded here rather than buried.

CLASS ORDER IS LOAD-BEARING. The public loss reads `softmax(1)[:, 1]` as the FAKE probability.
Our project convention is also 1 = fake, so the mapping is direct -- but it is asserted below,
because a silent reversal would invert every FPNet mask while training looked healthy.
"""
from __future__ import annotations

import argparse, json, sys, os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# The config's h5 and split paths are relative to the DISCERN_Ext root, so resolve from there
# rather than rewriting every path in the config and risking a divergence from the training run.
sys.path.insert(0, "/data/umar/Repos/DISCERN_Ext")
os.chdir("/data/umar/Repos/DISCERN_Ext")

OUT = Path("/data/umar/Repos/FreqBlender/net/resnet/checkpoints")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--limit", type=int, default=0, help="cap train batches for a smoke run")
    args = ap.parse_args()

    from torchvision.models import resnet34
    from src.config import load_config
    from src.dataset.deepfake import H5DeepfakeDataModule

    cfg = load_config("/data/umar/Repos/DISCERN_Ext/eval_adaptation/configs/run1_auxedl.yaml")
    cfg.training_arm = None                       # the teacher sees the plain h5 path
    dm = H5DeepfakeDataModule(config=cfg, vfm_model_name=cfg.backbone)
    dm.setup("fit")

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    net = resnet34(weights="IMAGENET1K_V1")
    net.fc = nn.Linear(net.fc.in_features, 2)     # 2-class head, index 1 = fake
    net = net.to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()

    tl = DataLoader(dm.train_dataset, batch_size=args.batch, shuffle=True,
                    num_workers=4, drop_last=True)
    # SHUFFLED on purpose. The val split is ordered by manipulation family, so any truncated
    # pass reads one class only -- AUROC comes back nan and the class-order assertion fires on a
    # smoke run that is actually fine. Shuffling costs nothing here (the metric is order-free)
    # and makes a partial evaluation representative.
    vl = DataLoader(dm.val_dataset, batch_size=args.batch, shuffle=True, num_workers=4)

    best, hist = 0.0, []
    for ep in range(args.epochs):
        net.train()
        for i, b in enumerate(tl):
            if args.limit and i >= args.limit:
                break
            x, y = b["image"].to(dev), b["label"].to(dev).long()
            opt.zero_grad()
            loss = lossf(net(x), y)
            loss.backward(); opt.step()
            if i % 200 == 0:
                print(f"  ep{ep} step {i} loss {loss.item():.4f}", flush=True)
        net.eval()
        P, Y = [], []
        with torch.no_grad():
            for j, b in enumerate(vl):
                if args.limit and j >= max(4, args.limit // 4):
                    break
                x = b["image"].to(dev)
                P.append(torch.softmax(net(x), 1)[:, 1].cpu().numpy())
                Y.append(b["label"].numpy())
        P, Y = np.concatenate(P), np.concatenate(Y)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(Y, P) if len(np.unique(Y)) == 2 else float("nan")
        # The assertion that matters: index 1 must score HIGHER on fakes. If this trips, the
        # class order is reversed and every downstream FPNet mask would be inverted.
        sep = float(P[Y == 1].mean() - P[Y == 0].mean())
        print(f"  epoch {ep}: val AUROC {auc:.4f} | mean p1(fake) - p1(real) = {sep:+.4f}",
              flush=True)
        hist.append({"epoch": ep, "auroc": float(auc), "sep": sep})
        if auc > best:
            best = auc
            OUT.mkdir(parents=True, exist_ok=True)
            torch.save({"model": net.state_dict(), "val_auroc": float(auc),
                        "class_order": "index 1 = fake",
                        "trained_on": "FFpp_train (real + 4 manipulations), split-safe",
                        "provenance": "trained locally; Baidu-hosted official weights "
                                      "unreachable (account wall)"},
                       OUT / "resnet34.tar")
    if not np.isfinite(sep):
        print("  class-order check SKIPPED: validation subset held one class only "
              "(expected on a --limit smoke run, never on a full run)")
    else:
        assert sep > 0, ("index 1 does not score higher on fakes -- class order is reversed "
                         "and every FPNet mask built on this teacher would be inverted")
    print(f"\nbest val AUROC {best:.4f} -> {OUT/'resnet34.tar'}")
    Path("/data/umar/Repos/DFB_NeSyNS/logs/tbiom/sbi").mkdir(parents=True, exist_ok=True)
    Path("/data/umar/Repos/DFB_NeSyNS/logs/tbiom/sbi/teacher.json").write_text(
        json.dumps({"history": hist, "best_auroc": best}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

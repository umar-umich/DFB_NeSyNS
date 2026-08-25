#!/usr/bin/env python3
"""Fit a cheap logistic probe over the MR-VAE rate response, for the Stage 1 membership test.

    python analysis/tbiom/fit_rate_probe.py \
        --features cache/discern_v2/fsvfm_frozen/FaceForensics++_train/features.npz \
        --operator configs/discern_v2/rate/rate_operator_mrvae.pt \
        --out configs/discern_v2/rate/rate_probe.json

Stage 1 has to score each candidate expert on DF40-Dev, which needs a prediction, but the expert's
real EDL head is Stage 2 work — only for experts that PASSED Stage 1. That is circular unless the
membership test uses something cheaper.

A logistic probe on the standardized rate response resolves it, and the direction of the
approximation is what makes it safe: it is a **lower bound** on what a trained EDL head would
reach, so an expert showing no complementarity here would not acquire it from a bigger head. The
converse does not hold, which is why a probe that DOES show complementarity would justify training
the real head rather than settling the question.

Fit on FF++ c23 train only — the same gradient-training budget as everything else in this project.
Standardization statistics come from the AUTHENTIC rows only, matching how every other branch in
this codebase standardizes: the head should see "how unusual is this response relative to authentic
video", and folding fake statistics into the yardstick shrinks the deviation being measured.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(REPO / "analysis" / "discern_v2"))

import fit_reference as FR  # noqa: E402
from networks.discern_v2.rate_branch import FrozenRateOperator  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--operator", type=Path, required=True)
    ap.add_argument("--feature-key", default="f0")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-samples", type=int, default=40000)
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    X = FR.load_matrix(args.features, args.feature_key).astype(np.float32)
    labels = np.load(args.features.parent / "labels.npy")
    if len(X) > args.max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=args.max_samples, replace=False)
        X, labels = X[idx], labels[idx]
    print(f"  {len(X)} FF++ train features ({int((labels==0).sum())} real / "
          f"{int((labels==1).sum())} fake)")

    op = FrozenRateOperator(args.operator)
    op.assert_frozen()
    with torch.no_grad():
        R = op(torch.from_numpy(X)).numpy()
    print(f"  rate response {R.shape}, beta grid {op.beta_grid}")

    # authentic-only standardization, as every other branch here does
    mu = R[labels == 0].mean(axis=0)
    sigma = R[labels == 0].std(axis=0) + 1e-8
    Z = (R - mu) / sigma

    model = LogisticRegression(max_iter=2000).fit(Z, labels)
    p = model.predict_proba(Z)[:, 1]
    auc = float(roc_auc_score(labels, p))
    print(f"  probe train AUROC {auc:.4f}")
    if auc < 0.55:
        print("  NOTE this is at or near chance ON ITS OWN TRAINING DATA. The probe is a lower "
              "bound, so Stage 1 should be read expecting little; it is still run, because the "
              "membership decision is complementarity and not this number.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "mu": mu.tolist(), "sigma": sigma.tolist(),
        "coef": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]),
        "train_auroc": auc, "n_train": int(len(X)),
        "operator": str(args.operator), "features": str(args.features),
        "standardization": "authentic FF++ train rows only",
        "role": "LOWER BOUND stand-in for a trained EDL head, for the Stage 1 membership test",
    }, indent=2))
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

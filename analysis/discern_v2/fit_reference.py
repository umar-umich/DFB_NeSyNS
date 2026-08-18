#!/usr/bin/env python3
"""Stage I — fit the reals-only reference offline, then freeze it permanently.

This is the ONLY place a reference is allowed to receive gradient. It runs before any
supervised training, consumes **FF++ authentic training features only**, and writes a frozen
artifact (weights + residual calibration + provenance) that Stage II loads read-only.

Why offline rather than a warmup phase inside training: a warmup still leaves the reference
attached to the optimizer, and one misconfigured flag later puts the classification loss back
on it — which is exactly how Phase 1 failed. An artifact on disk cannot be co-trained by
accident.

Inputs
------
A cached CLIP feature matrix for FF++ authentic **training** frames, from a FROZEN encoder.
Supply it as an .npz/.npy of shape (N, D) via --features, together with a boolean/int label
vector (--labels) so this script can enforce reals-only itself rather than trusting the caller.

Outputs, per (arm, objective), under --out:
    reference_<arm>_<objective>.pt     state_dict + calibrator buffers + provenance
    fit_report.json                    losses, residual statistics, PCA explained variance

    python analysis/discern_v2/fit_reference.py \
        --features cache/ffpp_train_clip.npz --labels cache/ffpp_train_labels.npy \
        --arms C1_random C2_linear C3_ae --objectives cosine mse

🔴 UMAR-RUNS. Nothing here launches training; it fits a small head on cached features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training"))

from networks.discern_v2 import reference as R  # noqa: E402


def load_matrix(path: Path, key: str | None = None) -> np.ndarray:
    if path.suffix == ".npz":
        z = np.load(path, allow_pickle=False)
        if key and key in z:
            return z[key]
        # single-array npz, or fall back to the first 2-D entry
        for k in z.files:
            if z[k].ndim == 2:
                return z[k]
        raise KeyError(f"{path} has no 2-D array; keys={z.files}")
    return np.load(path, allow_pickle=False)


def reals_only(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Keep label == 0. Enforced here rather than trusting the caller.

    A reference accidentally fit on fakes stops being a model of authenticity, and nothing
    downstream would reveal it — the residuals simply shrink on the manipulations we most
    wanted them to flag.
    """
    if len(features) != len(labels):
        raise ValueError(f"features {len(features)} vs labels {len(labels)}")
    mask = np.asarray(labels).reshape(-1) == 0
    n_real = int(mask.sum())
    if n_real == 0:
        raise ValueError("no authentic (label == 0) samples found — refusing to fit")
    if n_real == len(mask):
        print("  note: every sample is labelled authentic; assuming a pre-filtered reals matrix")
    print(f"  reals-only filter: {n_real}/{len(mask)} samples retained")
    return features[mask]


def fit_one(arm: str, objective: str, X: np.ndarray, latent_dim: int, epochs: int,
            lr: float, batch: int, device: str) -> tuple[R.FrozenReference, R.ResidualCalibrator, dict]:
    D = X.shape[1]
    ref = R.build_reference(arm, D, latent_dim).to(device)
    prov: dict = {"arm": arm, "objective": objective, "n_real": int(X.shape[0]),
                  "feature_dim": int(D), "latent_dim": latent_dim}
    Xt = torch.from_numpy(X).float().to(device)

    if arm == "C1_random":
        print(f"  {arm}: capacity floor — not fit by design")
        prov["fit"] = "none (capacity floor)"
    elif arm == "C2_linear":
        prov["pca"] = ref.fit_pca(Xt)
        print(f"  {arm}: PCA explained variance ratio "
              f"{prov['pca']['explained_variance_ratio']:.4f}")
        prov["fit"] = "pca (closed form)"
    else:
        ref.fitting()
        opt = torch.optim.Adam(ref.parameters(), lr=lr)
        hist = []
        n = Xt.shape[0]
        for ep in range(epochs):
            perm = torch.randperm(n, device=device)
            tot = 0.0
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                opt.zero_grad()
                loss = ref.reconstruction_loss(Xt[idx], objective)
                loss.backward()
                opt.step()
                tot += float(loss) * len(idx)
            hist.append(tot / n)
            if ep % max(1, epochs // 5) == 0 or ep == epochs - 1:
                print(f"    epoch {ep:3d}  {objective} loss {hist[-1]:.6f}")
        prov["fit"] = f"adam lr={lr} epochs={epochs}"
        prov["loss_history"] = hist

    ref.freeze()

    # residual calibration on the same authentic features, then frozen with the reference
    with torch.no_grad():
        desc = ref(Xt)
    cal = R.ResidualCalibrator(D).to(device).fit(desc.residual)
    prov["residual_stats_real"] = {
        "norm_mean": float(desc.norm.mean()), "norm_std": float(desc.norm.std()),
        "angle_mean": float(desc.angle.mean()), "angle_std": float(desc.angle.std()),
    }
    print(f"  {arm}/{objective}: authentic residual norm "
          f"{prov['residual_stats_real']['norm_mean']:.4f} "
          f"+- {prov['residual_stats_real']['norm_std']:.4f}, angle "
          f"{prov['residual_stats_real']['angle_mean']:.4f}")
    return ref, cal, prov


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True,
                    help="(N, D) cached CLIP features for FF++ TRAIN frames, frozen encoder")
    ap.add_argument("--labels", type=Path, default=None,
                    help="(N,) labels; 0 = authentic. Omit only if --features is already reals-only")
    ap.add_argument("--feature-key", default=None, help="key inside an .npz")
    ap.add_argument("--arms", nargs="+", default=["C1_random", "C2_linear", "C3_ae"],
                    choices=list(R.ARMS))
    ap.add_argument("--objectives", nargs="+", default=["cosine", "mse"],
                    choices=list(R.FIT_OBJECTIVES))
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path,
                    default=Path("configs/discern_v2/reference"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    X = load_matrix(args.features, args.feature_key).astype(np.float32)
    print(f"loaded features {X.shape} from {args.features}")
    if args.labels is not None:
        X = reals_only(X, load_matrix(args.labels))
    else:
        print("  no --labels given; treating the matrix as already reals-only")

    report = {"features": str(args.features), "n_real": int(X.shape[0]),
              "feature_dim": int(X.shape[1]), "latent_dim": args.latent_dim, "arms": {}}

    for arm in args.arms:
        # C1/C2 have no objective; fit once and record it under "n/a" rather than duplicating
        objectives = ["n/a"] if arm in ("C1_random", "C2_linear") else args.objectives
        for obj in objectives:
            print(f"\n=== {arm} / {obj} ===")
            ref, cal, prov = fit_one(arm, "cosine" if obj == "n/a" else obj, X,
                                     args.latent_dim, args.epochs, args.lr, args.batch,
                                     args.device)
            dest = args.out / f"reference_{arm}_{obj.replace('/', '')}.pt"
            torch.save({"arm": arm, "objective": obj, "feature_dim": int(X.shape[1]),
                        "latent_dim": args.latent_dim,
                        "reference_state": ref.state_dict(),
                        "calibrator_state": cal.state_dict(),
                        "provenance": prov}, dest)
            print(f"  saved {dest}")
            report["arms"][f"{arm}/{obj}"] = prov

    (args.out / "fit_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {args.out}/fit_report.json")
    print("\nStage I complete. These artifacts are FROZEN — Stage II loads them read-only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

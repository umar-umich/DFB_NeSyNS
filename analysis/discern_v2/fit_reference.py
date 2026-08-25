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
Build it with `analysis/discern_v2/cache_encoder_features.py --encoder frozen --split train`;
that writes `features.npz` (key `f0`), `labels.npy`, and the `manifest.json` this script reads
to *verify* the encoder was frozen and to stamp the encoder fingerprint into the artifact.
Without a manifest the fit is refused unless --allow-unprovenanced is passed, because
"fit on a frozen encoder" would otherwise be an unverified assertion.

Labels are supplied separately (--labels) so this script can enforce reals-only itself rather
than trusting the caller.

Outputs, per (arm, objective), under --out:
    reference_<arm>_<objective>.pt     state_dict + calibrator buffers + provenance
    fit_report.json                    losses, residual statistics, PCA explained variance

    python analysis/discern_v2/fit_reference.py \
        --features cache/discern_v2/clip_frozen/FaceForensics++_train/features.npz \
        --feature-key f0 \
        --labels cache/discern_v2/clip_frozen/FaceForensics++_train/labels.npy \
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


def read_cache_manifest(features_path: Path) -> dict | None:
    """Find the `manifest.json` that `cache_encoder_features.py` wrote beside this cache.

    Searched two levels up: the cache layout is `<out>/<dataset>_<split>/features.npz` with
    the manifest at `<out>/manifest.json`.
    """
    for parent in (features_path.parent, features_path.parent.parent):
        candidate = parent / "manifest.json"
        if candidate.is_file():
            return json.loads(candidate.read_text())
    return None


def cache_slice(manifest: dict, features_path: Path) -> dict:
    """The manifest entry for THIS features file.

    One cache directory holds several slices, so the split and the partial flag are read from
    the entry whose path matches the file being fitted — a top-level value would describe
    whichever slice happened to be cached last.
    """
    want = str(features_path.parent)
    for key, entry in manifest.get("datasets", {}).items():
        if str(entry.get("path", "")).rstrip("/") == want.rstrip("/"):
            return {"slice": key, **entry}
    return {"slice": "unlisted"}


def assert_frozen_feature_space(manifest: dict | None, features_path: Path,
                                allow_unprovenanced: bool) -> dict:
    """Refuse to fit on a feature space that is not provably frozen.

    `assert_reference_config` already refuses a non-frozen encoder at construction, but it is
    told `encoder_frozen` by its caller — so it can only enforce what someone asserts. This
    checks the cache's own manifest, which records the encoder state that actually produced
    the features, and carries the fingerprint into the artifact so Stage II can prove it is
    reading the same encoder rather than assuming it. A reference fit on a drifting encoder
    fails silently: the residuals stay finite and plausible while measuring against a stale
    manifold, which is the Phase-1 bug.
    """
    if manifest is None:
        if not allow_unprovenanced:
            raise SystemExit(
                "no manifest.json beside --features, so the encoder state that produced these "
                "features is unknown and 'fit on a frozen encoder' cannot be verified. Build "
                "the cache with analysis/discern_v2/cache_encoder_features.py --encoder frozen, "
                "or pass --allow-unprovenanced for a synthetic/smoke matrix (recorded as such "
                "in the artifact).")
        print("  WARNING: unprovenanced features — artifact marked encoder_provenance=unknown")
        return {"encoder_provenance": "unknown"}

    if not manifest.get("encoder_frozen", False):
        raise SystemExit(
            f"the cache at {manifest.get('config')} was built with encoder_mode="
            f"{manifest.get('encoder_mode')!r}. The reals-only reference must be fit on a "
            f"FROZEN encoder feature space — with a tuned encoder the features drift away "
            f"from the ones the reference was fit on and every residual is measured against a "
            f"stale manifold. Task 0's dual-encoder option exists precisely so e_sem may be "
            f"tuned while the reference input stays frozen.")
    slice_info = cache_slice(manifest, features_path)
    if slice_info.get("partial") or (slice_info["slice"] == "unlisted"
                                     and manifest.get("partial")):
        print("  WARNING: this slice was cached with --max-batches — it is a smoke subset, not "
              "the full authentic training split")
    if slice_info.get("split") not in (None, "train"):
        print(f"  WARNING: fitting on the {slice_info['split']!r} split. Stage I is specified "
              f"on FF++ authentic TRAIN features; a reference fit on a test split has seen the "
              f"data every later audit reports on.")
    print(f"  feature space verified frozen; encoder fingerprint "
          f"{manifest['fingerprint']['all'][:16]}…")
    return {
        "encoder_provenance": "verified",
        "encoder_mode": manifest.get("encoder_mode"),
        "encoder_fingerprint": manifest.get("fingerprint"),
        "encoder_config": manifest.get("config"),
        "cache_slice": slice_info.get("slice"),
        "cache_split": slice_info.get("split"),
        "cache_feature": manifest.get("feature"),
        "cache_partial": bool(slice_info.get("partial", manifest.get("partial"))),
        "cache_git_commit": manifest.get("git_commit"),
    }


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
            lr: float, batch: int, device: str, encoder: dict | None = None,
            hidden_dim: int | None = None
            ) -> tuple[R.FrozenReference, R.ResidualCalibrator, dict]:
    D = X.shape[1]
    # the guard runs on the construction path, not merely in a docstring: encoder_frozen comes
    # from the cache manifest, so a tuned-encoder cache cannot reach a fit.
    R.assert_reference_config(arm, objective, encoder_frozen=True)
    ref = R.build_reference(arm, D, latent_dim, hidden_dim).to(device)
    prov: dict = {"arm": arm, "objective": objective, "n_real": int(X.shape[0]),
                  "feature_dim": int(D), "latent_dim": latent_dim, "hidden_dim": hidden_dim,
                  "n_reference_params": sum(p.numel() for p in ref.parameters()),
                  "encoder": encoder or {"encoder_provenance": "unknown"}}
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
    ap.add_argument("--arms", nargs="+", default=["C3_ae"], choices=list(R.ARMS),
                    help="V1 Stage A fits C3_ae only (spec §4.1). C1_random / C2_linear are the "
                         "§24 iterate candidates (FS-VFM alone vs +PCA vs +deterministic AE)")
    ap.add_argument("--objectives", nargs="+", default=["cosine"],
                    choices=list(R.FIT_OBJECTIVES),
                    help="V1 Stage A uses cosine: L_ref = 1 - cos(z_ref.detach(), z_hat)")
    ap.add_argument("--latent-dim", type=int, default=128,
                    help="V1 default for the 1024-d FS-VFM space (8x bottleneck). 🟡 ASK-UMAR "
                         "before changing — it sets how much of the authentic manifold the "
                         "reference can represent, and therefore what r_ref measures")
    ap.add_argument("--hidden-dim", type=int, default=256,
                    help="AE hidden width. The module default (32) was sized for DiCoME's 64-d "
                         "feature and would be a 1024->32 bottleneck here")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path,
                    default=Path("configs/discern_v2/reference"))
    ap.add_argument("--allow-unprovenanced", action="store_true",
                    help="fit on features with no cache manifest (synthetic/smoke only); the "
                         "artifact records encoder_provenance=unknown")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    X = load_matrix(args.features, args.feature_key).astype(np.float32)
    print(f"loaded features {X.shape} from {args.features}")
    encoder = assert_frozen_feature_space(read_cache_manifest(args.features), args.features,
                                          args.allow_unprovenanced)
    if args.labels is not None:
        X = reals_only(X, load_matrix(args.labels))
    else:
        print("  no --labels given; treating the matrix as already reals-only")

    report = {"features": str(args.features), "n_real": int(X.shape[0]),
              "feature_dim": int(X.shape[1]), "latent_dim": args.latent_dim,
              "encoder": encoder, "arms": {}}

    for arm in args.arms:
        # C1/C2 have no objective; fit once and record it under "n/a" rather than duplicating
        objectives = ["n/a"] if arm in ("C1_random", "C2_linear") else args.objectives
        for obj in objectives:
            print(f"\n=== {arm} / {obj} ===")
            ref, cal, prov = fit_one(arm, "cosine" if obj == "n/a" else obj, X,
                                     args.latent_dim, args.epochs, args.lr, args.batch,
                                     args.device, encoder, args.hidden_dim)
            dest = args.out / f"reference_{arm}_{obj.replace('/', '')}.pt"
            torch.save({"arm": arm, "objective": obj, "feature_dim": int(X.shape[1]),
                        # hidden_dim travels WITH the weights: the branch rebuilds the module
                        # before loading them, and rebuilding at a different width fails the
                        # load outright — better than loading silently, but it still blocks
                        # Stage B until the width is recorded here.
                        "latent_dim": args.latent_dim, "hidden_dim": args.hidden_dim,
                        "encoder": encoder,
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

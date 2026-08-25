#!/usr/bin/env python3
"""Stage 2.1 — fit and FREEZE the MR-VAE rate operator on authentic FF++ features.

    🔴 UMAR-RUNS (GPU, ~5 min):

    python analysis/discern_v2/phase2/fit_rate_operator.py \
        --features cache/discern_v2/fsvfm_frozen/FaceForensics++_train/features.npz \
        --out configs/discern_v2/rate

Why a fit rather than a checkpoint load
----------------------------------------
The brief says to attach "the Phase-1 mechanism-valid MR-VAE and its frozen checkpoint". Two
trained MR-VAEs exist —

    /data/umar/Repos/DiCoME/runs/pilots/P1d/checkpoints/dicome-best-epoch=01-*.ckpt
    logs/train/nesy_defake_d3_p1d_2026-08-16-18-33-53/best_avg.pth

— and neither can be attached, because the MR-VAE is a **feature-space** operator: the first lives
in DiCoME's LoRA-CLIP 64-d space, the second in the D-ladder backbone's, and V1's Branch-A space
does not exist until Stage 4 trains its LoRA. Loading either would run the operator on coordinates
it was never fit to, and it would not error — it would just produce a response curve that means
nothing.

So the operator is re-fit here on the **frozen FS-VFM embedding**, which exists now, never changes,
and is independent of Stage 4. Recorded as a deviation in `phase2/REPO_MAP.md` item 5 rather than
presented as the brief's literal instruction.

Fit on authentic features only
-------------------------------
`R(x)` is meant to answer "how compressible is this face representation under a model of *bona
fide* faces". Training the MR-VAE on fakes too would let it learn to reconstruct forgeries equally
well, which is precisely the contrast the response is supposed to expose. This mirrors `P_R` and
the process branch's calibrator, both of which are authentic-only for the same reason.

The rate schedule is the operator's, not a choice made here
------------------------------------------------------------
Training samples beta log-uniformly over `BETA_RANGE = (0.1, 10.0)`; the response is read at
`BETA_GRID = (0.1, 0.32, 1.0, 3.16, 10.0)` (`projectors.py:38-40`). Both travel into the artifact,
and `FrozenRateOperator` reads the grid back from there — the response is only interpretable at
rates the FiLM conditioning actually saw.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "analysis" / "discern_v2"))
sys.path.insert(0, str(REPO / "training"))

import fit_reference as FR  # noqa: E402
from networks.discern_v2.projectors import BETA_GRID, BETA_RANGE, MRVAEProjector  # noqa: E402
from networks.discern_v2.reference import ResidualCalibrator  # noqa: E402


def elbo(x: torch.Tensor, recon: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor,
         beta: torch.Tensor) -> torch.Tensor:
    """Per-sample beta-weighted ELBO, with beta varying WITHIN the batch.

    Each sample carries its own beta, so one pass covers the whole rate range and the FiLM
    conditioning learns the curve rather than one operating point. Averaging a single beta over
    the batch would train an ordinary VAE with a noisy KL weight and leave the conditioning
    network at its identity initialisation.
    """
    recon_loss = (1.0 - F.cosine_similarity(x, recon, dim=1))
    kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=1)
    return (recon_loss + beta.squeeze(1) * kl).mean()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--features", type=Path, nargs="+", required=True)
    ap.add_argument("--feature-key", default="f0")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--latent-dim", type=int, default=128)
    ap.add_argument("--hidden-dim", type=int, default=256,
                    help="the module default (32) was sized for DiCoME's 64-d feature; on a "
                         "1024-d encoder embedding it dominates the response")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    mats, encoders, composition = [], [], {}
    for path in args.features:
        X = FR.load_matrix(path, args.feature_key).astype(np.float32)
        encoders.append(FR.assert_frozen_feature_space(
            FR.read_cache_manifest(path), path, allow_unprovenanced=False))
        labels_path = path.parent / "labels.npy"
        if labels_path.is_file():
            X = FR.reals_only(X, np.load(labels_path))
        mats.append(X)
        composition[str(path)] = int(len(X))
        print(f"  {path}: {X.shape}")
    fingerprints = {e.get("encoder_fingerprint", {}).get("all") for e in encoders}
    if len(fingerprints) > 1:
        raise SystemExit(f"feature caches come from different encoders: {fingerprints}")

    X = np.concatenate(mats, axis=0)
    Xt = torch.from_numpy(X).float().to(args.device)
    print(f"\nauthentic population: {X.shape[0]} features, dim {X.shape[1]}")
    print(f"beta: log-uniform over {BETA_RANGE}; response read at {BETA_GRID}")

    projector = MRVAEProjector(X.shape[1], args.latent_dim,
                               hidden_dim=args.hidden_dim).to(args.device)
    opt = torch.optim.Adam(projector.parameters(), lr=args.lr)
    projector.train()
    history = []
    n = len(Xt)
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=args.device)
        total = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            batch = Xt[idx]
            beta = projector.sample_beta(len(idx), batch.device, batch.dtype)
            _, mu, log_var, recon = projector.forward_at_beta(batch, beta)
            opt.zero_grad()
            loss = elbo(batch, recon, mu, log_var, beta)
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        history.append(total / n)
        if ep % max(1, args.epochs // 5) == 0 or ep == args.epochs - 1:
            print(f"    epoch {ep:3d}  beta-ELBO {history[-1]:.6f}")

    projector.eval()
    for p in projector.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        response = projector.rate_distortion_response(Xt)
    calibrator = ResidualCalibrator(len(BETA_GRID)).to(args.device).fit(response)

    per_beta = {f"beta_{b}": {"mean": float(response[:, i].mean()),
                              "std": float(response[:, i].std())}
                for i, b in enumerate(BETA_GRID)}
    print("\nauthentic rate response (the yardstick the head standardizes against):")
    for name, stat in per_beta.items():
        print(f"  {name:12s} {stat['mean']:.5f} +- {stat['std']:.5f}")
    monotone = all(per_beta[f"beta_{BETA_GRID[i]}"]["mean"]
                   <= per_beta[f"beta_{BETA_GRID[i + 1]}"]["mean"]
                   for i in range(len(BETA_GRID) - 1))
    print(f"  mean distortion is monotone in beta: {monotone}"
          + ("" if monotone else "  <- unexpected: a rate-distortion curve should rise with "
                                 "beta. Inspect before trusting the response."))

    provenance = {
        "stage": "phase2 Stage 2.1", "features": [str(p) for p in args.features],
        "composition": composition, "n_real": int(len(X)), "feature_dim": int(X.shape[1]),
        "latent_dim": args.latent_dim, "hidden_dim": args.hidden_dim,
        "beta_range": list(BETA_RANGE), "beta_grid": list(BETA_GRID),
        "fit": f"adam lr={args.lr} epochs={args.epochs} batch={args.batch} seed={args.seed}",
        "loss": "beta-weighted ELBO, cosine reconstruction, per-sample log-uniform beta",
        "loss_history": history, "encoder": encoders[0],
        "authentic_response": per_beta, "monotone_in_beta": bool(monotone),
        "n_operator_params": sum(p.numel() for p in projector.parameters()),
        "deviation": "re-fit on frozen FS-VFM features rather than loading a Phase-1 checkpoint; "
                     "the existing checkpoints live in feature spaces V1 does not have. See "
                     "phase2/REPO_MAP.md item 5.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / "rate_operator_mrvae.pt"
    torch.save({"feature_dim": int(X.shape[1]), "latent_dim": args.latent_dim,
                "hidden_dim": args.hidden_dim, "beta_grid": list(BETA_GRID),
                "projector_state": projector.state_dict(),
                "calibrator_state": calibrator.state_dict(),
                "encoder": encoders[0], "provenance": provenance}, dest)
    (args.out / "fit_report.json").write_text(json.dumps(provenance, indent=2, default=str))
    print(f"\nsaved {dest}")
    print("FROZEN. Stage 4 loads this read-only — the operator never trains with the detector.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

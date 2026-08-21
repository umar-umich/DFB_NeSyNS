#!/usr/bin/env python3
"""Stage 1.2 — refit and freeze `P_R` on a DIVERSE real population.

    🔴 UMAR-RUNS (GPU, ~3 min):

    python analysis/discern_v2/phase2/fit_diverse_reference.py \
        --features cache/discern_v2/fsvfm_frozen/FaceForensics++_train/features.npz \
                   cache/discern_v2/fsvfm_frozen/FFHQ-recrop_real/features.npz \
        --out configs/discern_v2/reference_diverse

V1 fit `P_R` on FF++ train reals alone (23,039 features, `configs/discern_v2/reference/
fit_report.json`), so its residual measured "unlike FF++ authentic" rather than "unlike
authentic" — the §20 audit flagged the reference branch on 6 of 6 sources. This refits the same
architecture on a broader authentic population and freezes it permanently before any detector
training.

Three things this does that a plain re-run of `fit_reference.py` would not
--------------------------------------------------------------------------
* **Accepts several feature caches and records the mix.** The population composition is the whole
  point of the stage, so it is written into the artifact rather than left implicit in a command
  line that nobody keeps.
* **Refuses to mix feature spaces.** Every cache must come from the same frozen encoder with the
  same pooling. Concatenating FS-VFM `global_pool` with FS-VFM `cls` features would fit a
  reference over a coordinate system that no branch ever produces, and nothing downstream would
  error.
* **Balances the sources when asked.** Concatenating 23,039 FF++ frames with 8,750 FFHQ images
  leaves the reference 72% FF++, so "diverse" would overstate what changed. `--balance` caps
  every source at the smallest one's size and records both the raw and used counts.

Residual normalization statistics are fit on the SAME population, in the same call, by the same
code path (`fit_reference.fit_one` calibrates on the features it was handed). The brief asks for
that explicitly, and separating the two operations is how they drift.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "analysis" / "discern_v2"))
sys.path.insert(0, str(REPO / "training"))

import fit_reference as FR  # noqa: E402


def load_one(path: Path, feature_key: str | None) -> tuple[np.ndarray, dict]:
    """Reals-only features from one cache, with its encoder provenance."""
    X = FR.load_matrix(path, feature_key).astype(np.float32)
    manifest = FR.read_cache_manifest(path)
    encoder = FR.assert_frozen_feature_space(manifest, path, allow_unprovenanced=False)
    labels_path = path.parent / "labels.npy"
    if labels_path.is_file():
        X = FR.reals_only(X, np.load(labels_path))
    else:
        print(f"  {path.parent.name}: no labels.npy beside the cache; treating as pre-filtered "
              f"reals. If this cache contains fakes they are now in the bona-fide manifold.")
    return X, encoder


def assert_same_feature_space(encoders: list[dict], paths: list[Path]) -> None:
    keys = [(e.get("encoder_fingerprint", {}).get("all"), e.get("encoder_mode")) for e in encoders]
    if len(set(keys)) > 1:
        detail = "\n".join(f"    {p.parent.name}: fingerprint={k[0]} mode={k[1]}"
                           for p, k in zip(paths, keys))
        raise SystemExit(
            "the feature caches come from different encoders:\n" + detail +
            "\nConcatenating them would fit P_R over a coordinate system no branch produces, and "
            "nothing downstream would error — the residual would just stop meaning anything.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--features", type=Path, nargs="+", required=True)
    ap.add_argument("--feature-key", default="f0")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arm", default="C3_ae", choices=list(FR.R.ARMS),
                    help="the brief says start with the deterministic AE; C1/C2 are controls")
    ap.add_argument("--objective", default="cosine", choices=("cosine", "mse"))
    ap.add_argument("--latent-dim", type=int, default=128)   # V1's fitted values, kept
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--balance", action="store_true",
                    help="cap every source at the smallest one, so the fit is genuinely diverse "
                         "rather than dominated by whichever cache is biggest")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    mats, encoders, composition = [], [], {}
    for path in args.features:
        X, encoder = load_one(path, args.feature_key)
        mats.append(X)
        encoders.append(encoder)
        composition[str(path)] = {"n_real_available": int(len(X))}
        print(f"  {path}: {X.shape}")
    assert_same_feature_space(encoders, args.features)

    if args.balance and len(mats) > 1:
        cap = min(len(m) for m in mats)
        print(f"\nbalancing every source to {cap} features (the smallest cache)")
        mats = [m if len(m) == cap else m[rng.choice(len(m), size=cap, replace=False)]
                for m in mats]
    for path, m in zip(args.features, mats):
        composition[str(path)]["n_real_used"] = int(len(m))

    X = np.concatenate(mats, axis=0)
    fractions = {p: c["n_real_used"] / len(X) for p, c in composition.items()}
    print(f"\nreference population: {X.shape[0]} authentic features, dim {X.shape[1]}")
    for p, f in fractions.items():
        print(f"  {f:6.1%}  {Path(p).parent.name}")
    dominant = max(fractions.values())
    if dominant > 0.8 and len(mats) > 1:
        print(f"\n  ⚠️  one source is {dominant:.0%} of the population. Calling this reference "
              f"'diverse' would overstate what changed — use --balance, or report the mix.")

    args.out.mkdir(parents=True, exist_ok=True)
    encoder = encoders[0]
    ref, cal, prov = FR.fit_one(args.arm, args.objective, X, args.latent_dim, args.epochs,
                                args.lr, args.batch, args.device, encoder, args.hidden_dim)
    prov["population"] = {"composition": composition, "fractions": fractions,
                          "balanced": bool(args.balance), "seed": args.seed,
                          "n_total": int(len(X))}

    dest = args.out / f"reference_{args.arm}_{args.objective}.pt"
    torch.save({"arm": args.arm, "objective": args.objective,
                "feature_dim": int(X.shape[1]), "latent_dim": args.latent_dim,
                "hidden_dim": args.hidden_dim, "encoder": encoder,
                "reference_state": ref.state_dict(), "calibrator_state": cal.state_dict(),
                "provenance": prov}, dest)
    (args.out / "fit_report.json").write_text(json.dumps(
        {"stage": "phase2 Stage 1.2 — diverse-real refit",
         "supersedes": "configs/discern_v2/reference/reference_C3_ae_cosine.pt (FF++ reals only)",
         "features": [str(p) for p in args.features], "n_real": int(X.shape[0]),
         "feature_dim": int(X.shape[1]), "encoder": encoder,
         "arms": {f"{args.arm}/{args.objective}": prov}}, indent=2, default=str))

    print(f"\nsaved {dest}")
    print("FROZEN. Stage 4 loads this read-only; refitting after training invalidates the run.")
    print("\nNext (Stage 1.3): re-run the domain audit on the refit branch —")
    print("  it must be audited on ref_residual_norm, ref_angle, p_ref AND u_ref, because V1's "
          "head partly laundered the provenance signal that the raw residual carried.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

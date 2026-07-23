"""Pilot S — one more spectral angle: train the probe IN-DOMAIN on DF40.

The FF++-trained probe collapsed on DF40 (chance). This separates two hypotheses:
  H1  the spectral signal is not in the DISCERN bank for DF40 faces at all
  H2  the signal IS there, but does not cross the FF++ -> DF40 gap

Tests, on the res-matched DF40 GAN banks (stargan, starganv2):
  within-family   train/test split INSIDE one family (in-distribution ceiling)
  cross-family    train stargan -> test starganv2, and vice versa (DF40-internal transfer)

If within-family is at chance too, H1 holds and spectral is dead here. If
within-family separates but cross-family/FF++-transfer does not, H2 holds — the
signal exists but is family-specific, which still fails the CEC bar (certify
unseen families).

Run:
    /data/umar/miniconda3/envs/GenD/bin/python cec/pilots/pilot_s/probe_indomain_df40.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

BANK = REPO / "results" / "pilot_s" / "df40_forensic"
OUT = REPO / "results" / "pilot_s"
DCT_DIMS = [20, 21, 22, 23, 24, 25]
SEED = 0


def load_family(fam):
    X, y = [], []
    for name, lab in (("fake", 1), ("real", 0)):
        feats = torch.load(BANK / f"{fam}_{name}.pt", map_location="cpu",
                           weights_only=False)["features"].float().numpy()
        X.append(feats)
        y.append(np.full(len(feats), lab))
    return np.vstack(X), np.concatenate(y)


def fit_eval(Xtr, ytr, Xte, yte, dims):
    sc = StandardScaler().fit(Xtr[:, dims])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr[:, dims]), ytr)
    s = clf.predict_proba(sc.transform(Xte[:, dims]))[:, 1]
    return float(roc_auc_score(yte, s))


def within_family(fam, dims):
    """50/50 random split inside a family (in-distribution ceiling)."""
    X, y = load_family(fam)
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(y))
    half = len(idx) // 2
    tr, te = idx[:half], idx[half:]
    return fit_eval(X[tr], y[tr], X[te], y[te], dims)


def cross_family(train_fam, test_fam, dims):
    Xtr, ytr = load_family(train_fam)
    Xte, yte = load_family(test_fam)
    return fit_eval(Xtr, ytr, Xte, yte, dims)


def main():
    fams = ["stargan", "starganv2"]
    result = {"within": {}, "cross": {}}
    print(f"{'test':<28} {'full-83d':>10} {'DCT-6d':>9}")
    for fam in fams:
        f = within_family(fam, list(range(83)))
        d = within_family(fam, DCT_DIMS)
        result["within"][fam] = {"full": f, "dct": d}
        print(f"within {fam:<21} {f:>10.3f} {d:>9.3f}")
    for a, b in [("stargan", "starganv2"), ("starganv2", "stargan")]:
        f = cross_family(a, b, list(range(83)))
        d = cross_family(a, b, DCT_DIMS)
        result["cross"][f"{a}->{b}"] = {"full": f, "dct": d}
        print(f"cross {a}->{b:<15} {f:>10.3f} {d:>9.3f}")

    (OUT / "probe_indomain_df40.json").write_text(json.dumps(result, indent=2))
    print(f"\n[written] {OUT / 'probe_indomain_df40.json'}")
    print("H1 (signal absent) if within-family ~chance; H2 (family-specific) if within "
          "separates but cross/transfer does not. Umar's read.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

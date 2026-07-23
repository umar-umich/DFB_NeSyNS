"""Pilot S spectral gate — the real test: FF++-trained probe -> DF40 transfer.

Trains the DISCERN spectral probe on FF++ (features already precomputed) and
tests it on the res-matched DF40 GAN families (banks from
extract_df40_forensic.py). This is the CEC use case: train on available data,
certify on UNSEEN generative families.

Two arms, mirroring Pilot S:
  (i)  AUC(fake vs real) on DF40 — does the domain-matched probe transfer?
  (ii) intervention response — apply notch/checkerboard to DF40 fakes, re-extract
       the bank, and check the probe score DROPS (removing the cited spectral cue
       should lower p_fake). This needs the extractor, so it runs in dfb_nesy.

The probe is trained on the DCT-only 6-d spectral subset (the fair frequency
analogue) and, for reference, the full 83-d bank.

Run (dfb_nesy, for arm ii's re-extraction):
    /data/umar/miniconda3/envs/dfb_nesy/bin/python \
        cec/pilots/pilot_s/probe_transfer_df40.py
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
sys.path.insert(0, str(REPO / "preprocessing"))

PP = Path("/data/umar/Datasets/preprocessed/FaceForensics++")
DF40_BANK = REPO / "results" / "pilot_s" / "df40_forensic"
OUT = REPO / "results" / "pilot_s"
DCT_DIMS = [20, 21, 22, 23, 24, 25]
FAKE_METHODS = ["Deepfakes", "FaceSwap"]
SEED = 0


# ------------------------------------------------------------------ FF++ train set
def _ff_feat_dir(rel):
    return PP / rel / "c23" / "forensic_features"


def load_ffpp():
    X, y = [], []
    for method in FAKE_METHODS:
        for pt in sorted(_ff_feat_dir(f"manipulated_sequences/{method}").glob("*.pt")):
            X.append(torch.load(pt, map_location="cpu", weights_only=False)["features"].float().numpy().mean(0))
            y.append(1)
    reals = sorted(_ff_feat_dir("original_sequences/youtube").glob("*.pt"))
    np.random.RandomState(SEED).shuffle(reals)
    for pt in reals[: len(y)]:
        X.append(torch.load(pt, map_location="cpu", weights_only=False)["features"].float().numpy().mean(0))
        y.append(0)
    return np.array(X), np.array(y)


# ------------------------------------------------------------------ DF40 test set
def load_df40_family(family):
    """Per-frame feature rows + labels for a DF40 family bank."""
    X, y = [], []
    for label_name, label in (("fake", 1), ("real", 0)):
        pt = DF40_BANK / f"{family}_{label_name}.pt"
        feats = torch.load(pt, map_location="cpu", weights_only=False)["features"].float().numpy()
        X.append(feats)
        y.append(np.full(len(feats), label))
    return np.vstack(X), np.concatenate(y)


def fit_probe(Xtr, ytr, dims):
    scaler = StandardScaler().fit(Xtr[:, dims])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(scaler.transform(Xtr[:, dims]), ytr)
    return scaler, clf


def auc_on(scaler, clf, X, y, dims):
    s = clf.predict_proba(scaler.transform(X[:, dims]))[:, 1]
    return float(roc_auc_score(y, s))


def main():
    Xtr, ytr = load_ffpp()
    print(f"train (FF++): {len(ytr)} videos ({int(ytr.sum())} fake / {int((1-ytr).sum())} real)")

    families = [p.stem.rsplit("_", 1)[0] for p in sorted(DF40_BANK.glob("*_fake.pt"))]
    if not families:
        print(f"no DF40 banks in {DF40_BANK}; run extract_df40_forensic.py first.")
        return 1

    result = {"train": "ffpp", "seed": SEED, "auc": {}}
    print(f"\n{'family':<14} {'full-83d AUC':>13} {'DCT-6d AUC':>12}")
    for feat_name, dims in (("full", list(range(83))), ("dct", DCT_DIMS)):
        scaler, clf = fit_probe(Xtr, ytr, dims)
        for fam in families:
            Xte, yte = load_df40_family(fam)
            auc = auc_on(scaler, clf, Xte, yte, dims)
            result["auc"].setdefault(fam, {})[feat_name] = auc
    for fam in families:
        print(f"{fam:<14} {result['auc'][fam]['full']:>13.3f} {result['auc'][fam]['dct']:>12.3f}")

    (OUT / "probe_transfer_df40.json").write_text(json.dumps(result, indent=2))
    print(f"\n[written] {OUT / 'probe_transfer_df40.json'}")
    print("Gate arm (i): does the FF++-trained spectral probe transfer to unseen DF40 "
          "GAN families? Umar's call on these numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

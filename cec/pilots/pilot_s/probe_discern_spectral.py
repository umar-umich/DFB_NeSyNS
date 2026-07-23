"""Pilot S fallback: linear probe on the DISCERN spectral bank (FF++ first read).

Pilot S failed with off-the-shelf frequency detectors (FreqNet/NPR don't transfer
to face crops). Pre-committed fallback (CURRENT_STATE ledger 3, rule-4 amendment):
a linear probe on the DISCERN spectral bank, which is DOMAIN-MATCHED — computed on
our own FF++ crops.

This is the FF++ first read (zero new extraction, zero heavy deps): does a linear
probe on the precomputed forensic bank separate fake from real? Two feature sets:
  full   the 83-d forensic bank
  dct    the 6 DCT/spectral dims only (ff_dct_hf_*, ff_dct_ratio_*) — the true
         "spectral" subset, the fair analogue of a frequency detector

Split is grouped by SOURCE IDENTITY (the id before '_'), so no identity appears in
both train and test — otherwise the probe could separate on identity, not artifact.

NOTE: this tests FF++ only (the bank is not precomputed for DF40; that extraction
needs mediapipe+insightface, not installed — a separate 🔴 decision). It answers
"can a domain-matched spectral probe separate at all?", not the DF40 question.

Run:
    /data/umar/miniconda3/envs/GenD/bin/python cec/pilots/pilot_s/probe_discern_spectral.py
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

PP = Path("/data/umar/Datasets/preprocessed/FaceForensics++")
FF_DIR = PP / "forensic_features"  # unused; features live per-method
SEED = 0
OUT = REPO / "results" / "pilot_s"

FAKE_METHODS = ["Deepfakes", "FaceSwap"]
DCT_DIMS = [20, 21, 22, 23, 24, 25]  # ff_dct_hf_{skin,eye,mouth,nose}, ff_dct_ratio_{eye,mouth}_skin


def _feat_dir(rel):
    return PP / rel / "c23" / "forensic_features"


def load_bank(pt_path):
    """Per-video mean feature vector (mean over the video's frames)."""
    o = torch.load(pt_path, map_location="cpu", weights_only=False)
    feats = o["features"].float().numpy()  # [n_frames, 83]
    return feats.mean(axis=0)              # [83]


def gather():
    """(X, y, groups) — per-video mean features, label, source-identity group."""
    X, y, groups = [], [], []
    # fakes
    for method in FAKE_METHODS:
        for pt in sorted(_feat_dir(f"manipulated_sequences/{method}").glob("*.pt")):
            X.append(load_bank(pt))
            y.append(1)
            groups.append(pt.stem.split("_")[0])  # target identity
    # reals (youtube) — subsample to balance against the fake count
    real_pts = sorted(_feat_dir("original_sequences/youtube").glob("*.pt"))
    rng = np.random.RandomState(SEED)
    rng.shuffle(real_pts)
    for pt in real_pts[: len(y)]:
        X.append(load_bank(pt))
        y.append(0)
        groups.append(pt.stem)
    return np.array(X), np.array(y), np.array(groups)


def grouped_split(groups, y, frac=0.5):
    """Split by identity: no source id in both train and test."""
    rng = np.random.RandomState(SEED)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    n_test = int(len(uniq) * frac)
    test_ids = set(uniq[:n_test])
    test = np.array([g in test_ids for g in groups])
    return ~test, test


def evaluate(X, y, tr, te, dims=None):
    Xt = X[:, dims] if dims else X
    scaler = StandardScaler().fit(Xt[tr])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(scaler.transform(Xt[tr]), y[tr])
    scores = clf.predict_proba(scaler.transform(Xt[te]))[:, 1]
    return float(roc_auc_score(y[te], scores))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X, y, groups = gather()
    tr, te = grouped_split(groups, y)
    print(f"videos: {len(y)} ({int(y.sum())} fake / {int((1 - y).sum())} real); "
          f"train {tr.sum()} / test {te.sum()} (identity-grouped)")

    auc_full = evaluate(X, y, tr, te)
    auc_dct = evaluate(X, y, tr, te, dims=DCT_DIMS)
    print(f"\nDISCERN spectral-bank probe (FF++, identity-grouped test):")
    print(f"  full 83-d forensic bank : AUC {auc_full:.3f}")
    print(f"  DCT-only 6-d spectral   : AUC {auc_dct:.3f}")
    print(f"\nreference (Pilot S): effort/CLIP 0.980 on FF++; FreqNet 0.510; NPR 0.511")

    result = {"n_videos": len(y), "n_train": int(tr.sum()), "n_test": int(te.sum()),
              "auc_full_bank": auc_full, "auc_dct_only": auc_dct,
              "dct_dims": DCT_DIMS, "split": "identity-grouped", "seed": SEED,
              "scope": "FF++ only; DF40 extraction (mediapipe+insightface) is a separate decision"}
    (OUT / "discern_spectral_probe.json").write_text(json.dumps(result, indent=2))
    print(f"\n[written] {OUT / 'discern_spectral_probe.json'}")
    print("Interpretation is Umar's: this is the FF++ first read, not the DF40 gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

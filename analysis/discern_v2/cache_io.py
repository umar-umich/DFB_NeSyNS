#!/usr/bin/env python3
"""Reading the encoder caches — one definition, shared by Task 0 and E1.

`cache_encoder_features.py` writes `<cache>/<source>_<split>/{features.npz, labels.npy,
metadata.csv}` plus one `<cache>/manifest.json`. Task 0 compares two caches; E1 fits
references on one and audits residuals across several. Both need the same three things —
loading, video-level pooling, and train-only standardisation — and if each script defined them
the two would drift apart, which is the failure `dirichlet.py` was centralised to prevent.

Why not `common.py`: that module reads DiCoME's `sample_features.npz` exports, keyed on a
`family` column these caches do not have, and resolves paths through a DiCoME checkout. The
threshold and fold-hygiene rules are the same; the data contract is not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Cache:
    """One (source, split) slice of an encoder cache, with its provenance attached.

    The manifest travels with the arrays rather than being read separately, so no analysis can
    report a number without also having the encoder state that produced it.
    """

    features: np.ndarray          # (N, D) spatial_raw
    labels: np.ndarray            # (N,) 0 = authentic, 1 = manipulated
    meta: pd.DataFrame            # key / label / method / video_id / source / split
    manifest: dict
    source: str
    split: str

    @property
    def encoder_mode(self) -> str:
        return self.manifest.get("encoder_mode", "unknown")

    @property
    def fingerprint(self) -> str:
        return self.manifest.get("fingerprint", {}).get("all", "unknown")

    @property
    def n_real(self) -> int:
        return int((self.labels == 0).sum())

    def reals(self) -> np.ndarray:
        return self.features[self.labels == 0]

    def fakes(self) -> np.ndarray:
        return self.features[self.labels == 1]

    def __repr__(self) -> str:  # keeps log lines readable
        return (f"Cache({self.source}/{self.split}, n={len(self.labels)}, "
                f"D={self.features.shape[1]}, encoder={self.encoder_mode})")


def read_manifest(cache_dir: Path) -> dict:
    path = Path(cache_dir) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found — every analysis must be able to name the encoder state that "
            f"produced its features. Build the cache with cache_encoder_features.py.")
    return json.loads(path.read_text())


def load_cache(cache_dir: Path, source: str, split: str) -> Cache:
    cache_dir = Path(cache_dir)
    manifest = read_manifest(cache_dir)
    d = cache_dir / f"{source}_{split}"
    npz = d / "features.npz"
    if not npz.is_file():
        available = sorted(p.name for p in cache_dir.iterdir() if p.is_dir())
        raise FileNotFoundError(
            f"no cache for {source}/{split} at {npz}.\n"
            f"  present in {cache_dir}: {available}\n"
            f"  🔴 UMAR-RUNS: python analysis/discern_v2/cache_encoder_features.py "
            f"--config <detector.yaml> --encoder {manifest.get('encoder_mode', 'frozen')} "
            f"--datasets {source} --split {split} --out {cache_dir}")
    with np.load(npz, allow_pickle=False) as z:
        features, labels = z["f0"], z["labels"]
    meta = pd.read_csv(d / "metadata.csv")
    if len(meta) != len(labels):
        raise ValueError(f"{source}/{split}: metadata {len(meta)} vs labels {len(labels)}")
    if not np.array_equal(meta["label"].to_numpy(), labels):
        raise ValueError(
            f"{source}/{split}: metadata.csv label column disagrees with labels.npy — the "
            f"cache is inconsistent and any per-sample join would silently mislabel samples")
    return Cache(features=features, labels=labels, meta=meta, manifest=manifest,
                 source=source, split=split)


def assert_same_encoder(*caches: Cache) -> None:
    """All caches came from one encoder state.

    Required wherever residuals from different sources are compared (E1's domain-detector
    audit): if the OOD caches came from a different encoder than the FF++ cache the reference
    was fit on, a shifted residual distribution measures encoder drift, and the audit would
    report "dataset detector" or "clean forensic signal" with equal confidence either way.
    """
    prints = {c.fingerprint for c in caches}
    if len(prints) > 1:
        detail = ", ".join(f"{c.source}/{c.split}={c.fingerprint[:12]}" for c in caches)
        raise ValueError(
            f"caches come from different encoder states ({detail}). Residuals across them are "
            f"not comparable — rebuild every source with one encoder.")


def assert_different_encoder(a: Cache, b: Cache) -> None:
    """Task 0's two arms must actually differ.

    If a run's LN weights never moved (or the wrong checkpoint was passed), the two arms are
    the same encoder and the measured delta is exactly zero — which would read as "LN tuning
    contributes nothing", the strongest possible version of one of the two answers, from a
    configuration error.
    """
    if a.fingerprint == b.fingerprint:
        raise ValueError(
            f"the frozen and tuned caches have identical encoder fingerprints "
            f"({a.fingerprint[:16]}…). Either the checkpoint's LayerNorms never moved or the "
            f"tuned cache was built without --checkpoint. A zero delta from this is a config "
            f"error, not a Task-0 result.")


def video_level(meta: pd.DataFrame, scores: dict[str, np.ndarray]) -> pd.DataFrame:
    """Mean-pool frame scores to video level — DISCERN's aggregation, as in `common.py`.

    Grouped on (source, method, video_id), not `video_id` alone: FF++ reuses the same numeric
    video ids across manipulations, and ids collide across datasets, so pooling on the id
    alone would merge a real and its manipulations into one row.
    """
    df = meta.copy()
    for k, v in scores.items():
        if len(v) != len(df):
            raise ValueError(f"score {k!r} has {len(v)} rows, metadata has {len(df)}")
        df[k] = v
    agg = {k: (k, "mean") for k in scores}
    agg["label"] = ("label", "max")   # a video is fake if its frames are
    out = df.groupby(["source", "method", "video_id"], as_index=False).agg(**agg)
    return out


def train_standardiser(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(mu, sigma) from training features only.

    Returned rather than applied so the caller cannot fit on a concatenation of train and OOD
    — the same reason `common.fold_scaler` takes both splits at once. Per-fold hygiene:
    normalisation statistics never see the data they will be evaluated on.
    """
    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0) + 1e-6
    return mu, sigma


def apply_standardiser(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (x - mu) / sigma


def join_on_key(a: Cache, b: Cache) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Align two caches of the same source/split by frame path.

    Used to measure how far a tuned encoder moved the space for the *same* frames. Both caches
    are built with shuffle=False from the same JSON, so the orders normally match — but they
    are joined on `key` rather than assumed aligned, because a differing `frame_num` between
    runs silently changes which frames were sampled and a positional comparison would then be
    comparing different images.
    """
    if (a.source, a.split) != (b.source, b.split):
        raise ValueError(f"cannot join {a.source}/{a.split} with {b.source}/{b.split}")
    ia = pd.Index(a.meta["key"])
    ib = pd.Index(b.meta["key"])
    common = ia.intersection(ib)
    if len(common) == 0:
        raise ValueError(f"{a.source}/{a.split}: the two caches share no frame keys")
    pos_a = ia.get_indexer(common)
    pos_b = ib.get_indexer(common)
    meta = a.meta.iloc[pos_a].reset_index(drop=True)
    return a.features[pos_a], b.features[pos_b], meta

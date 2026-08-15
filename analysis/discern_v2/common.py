#!/usr/bin/env python3
"""Shared loading, threshold discipline and fold hygiene for the DISCERN v2 Track A analyses.

Every Track A script (A1, A2, A3) reads the same per-sample tables and obeys the same two
rules from the Phase-1 instructions, so they are implemented once here rather than
re-derived per script:

**Threshold discipline.** No operating threshold may be chosen per OOD source — doing so
leaks the target distribution and weakens the zero-shot claim (worst on CDFv3 / DF40 /
Deepfake-Eval-2024). `frozen_threshold()` derives ONE threshold from the permitted
train/val protocol only and every OOD source is scored with it. Threshold-free analyses are
provided alongside so a conclusion never rests on the threshold alone.

**Per-fold hygiene.** In leave-one-generator-out evaluation, scalers/calibrators must be fit
on the fold's *training* generators only. Fitting globally before splitting leaks the
held-out generator into the normalisation and silently inflates the diagnostic.
`fold_scaler()` exists so no script is tempted to call a global `fit_transform`.

Data contract
-------------
Reads the per-sample exports produced by DiCoME's analysis package:

    <DICOME>/experiments/pilot_<PILOT>/analysis/features/<SOURCE>/
        sample_features.npz   p_fused/p_semantic/p_artifact, u_*, evidence_*, feat_*,
                              and for P1d additionally rate_response (B,K) + beta_grid (K,)
        metadata.csv          key, label, family, method, video_id, source, eval_source

Those exports currently exist for P0-DS, P2a, P3a, P4 only. P1a-P1d must be exported before
A1/A3 can run -- see `missing_exports()` and the command it prints. That is a 🔴 UMAR-RUNS
step; nothing here launches it.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DISCERN = Path(__file__).resolve().parents[2]


def _find_dicome() -> Path:
    """Locate the DiCoME repo that holds the per-sample exports.

    Deriving it as `DISCERN.parent / "DiCoME"` alone is fragile: inside a git worktree
    `DISCERN` resolves under `.claude/worktrees/`, so the sibling lookup silently points at a
    directory that does not exist and every analysis reports "no export" instead of failing
    loudly. Resolution order: explicit `DICOME_ROOT` env var, then the sibling of the repo,
    then the sibling of the worktree's main checkout.
    """
    env = os.environ.get("DICOME_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    candidates = [DISCERN.parent / "DiCoME"]
    # <repo>/.claude/worktrees/<name> -> main checkout is parents[2] of the worktree
    for up in (3, 4):
        if len(DISCERN.parents) > up:
            candidates.append(DISCERN.parents[up - 1].parent / "DiCoME")
    for c in candidates:
        if (c / "experiments").is_dir():
            return c
    return candidates[0]


DICOME = _find_dicome()
PILOT_FEATURES = DICOME / "experiments" / "pilot_{pilot}" / "analysis" / "features" / "{source}"
OUT_ROOT = DISCERN / "analysis" / "discern_v2"

# The Phase-1 branch set. P0 is the visual/CLIP baseline every delta is measured against.
P0 = "P0-DS"
PILOTS = ["P0-DS", "P1a", "P1b", "P1c", "P1d", "P2a", "P3a", "P4"]

# Rows the instructions call decision-relevant: the four where P0 is anti-correlated
# (worse than chance) plus the one commercial reenactment product in DF40.
INVERSION_ROWS = ["danet-cdf", "mcnet-cdf", "tpsm-cdf", "facevid2vid-cdf"]
HEYGEN_ROW = "heygen"

SEED = 42


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def features_dir(pilot: str, source: str) -> Path:
    return Path(str(PILOT_FEATURES).format(pilot=pilot, source=source))


def has_export(pilot: str, source: str) -> bool:
    return (features_dir(pilot, source) / "sample_features.npz").exists()


def missing_exports(pilots: list[str], sources: list[str]) -> list[tuple[str, str]]:
    return [(p, s) for p in pilots for s in sources if not has_export(p, s)]


def export_command(pilot: str, sources: list[str]) -> str:
    return (f"python experiments/common/analysis/export_features.py {pilot} "
            f"--source {' '.join(sources)}")


def load(pilot: str, source: str) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Raw arrays + aligned metadata for one (pilot, source)."""
    d = features_dir(pilot, source)
    npz = d / "sample_features.npz"
    if not npz.exists():
        raise FileNotFoundError(
            f"No export for {pilot}/{source} at {npz}.\n"
            f"  This is a UMAR-RUNS step. In {DICOME}:\n"
            f"    {export_command(pilot, [source])}")
    with np.load(npz, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    meta = pd.read_csv(d / "metadata.csv")
    if len(meta) != len(next(iter(arrays.values()))):
        raise ValueError(f"{pilot}/{source}: metadata/array length mismatch")
    return arrays, meta


def video_level(meta: pd.DataFrame, values: dict[str, np.ndarray]) -> pd.DataFrame:
    """Mean-pool frame scores to video level -- DISCERN's own aggregation.

    All Track A complementarity/oracle work is specified at video level, so scripts should
    call this rather than pooling ad hoc.
    """
    df = meta.copy()
    for k, v in values.items():
        df[k] = v
    agg = {k: (k, "mean") for k in values}
    agg.update(label=("label", "first"), family=("family", "first"),
               method=("method", "first"), source=("source", "first"))
    return df.groupby("video_id", as_index=False).agg(**agg)


# ---------------------------------------------------------------------------
# threshold discipline
# ---------------------------------------------------------------------------


def frozen_threshold(pilot: str, protocol_source: str = "FFpp",
                     fallback: float = 0.5) -> tuple[float, str]:
    """ONE decision threshold, derived only from the permitted protocol source.

    Returns (threshold, provenance). The threshold is the equal-error point on the
    protocol source (FF++, i.e. the training distribution), NOT on any OOD source. It is then
    frozen for every OOD dataset and generator.

    If the protocol export is unavailable the function returns 0.5 with provenance saying so,
    rather than silently tuning something — the instructions require that a missing protocol
    threshold be reported, with threshold-free analyses carrying the conclusion instead.
    """
    if not has_export(pilot, protocol_source):
        return fallback, (f"fixed {fallback} (no {protocol_source} export for {pilot}; "
                          f"report threshold-free analyses alongside)")
    from sklearn.metrics import roc_curve

    arrays, meta = load(pilot, protocol_source)
    v = video_level(meta, {"p": arrays["p_fused"]})
    fpr, tpr, thr = roc_curve(v["label"].values, v["p"].values)
    t = float(thr[np.nanargmin(np.abs(fpr - (1 - tpr)))])
    return t, f"EER on {protocol_source} (protocol source), frozen for all OOD sources"


# ---------------------------------------------------------------------------
# fold hygiene
# ---------------------------------------------------------------------------


def logo_folds(groups: np.ndarray, min_per_group: int = 20) -> list[tuple[str, np.ndarray]]:
    """Leave-one-generator-out folds: (held_out_generator, test_mask).

    Generators with fewer than `min_per_group` samples are skipped as held-out folds (they
    stay in training) — a fold of five videos produces a meaningless per-fold number.
    """
    out = []
    for g in sorted(pd.unique(groups)):
        mask = groups == g
        if mask.sum() >= min_per_group:
            out.append((str(g), mask))
    return out


def logo_folds_two_class(methods: np.ndarray, labels: np.ndarray, min_per_group: int = 20,
                         real_method: str = "real", seed: int = SEED,
                         video_ids: np.ndarray | None = None
                         ) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Leave-one-generator-out folds that are actually two-class.

    Returns (held_out_generator, train_mask, test_mask).

    `logo_folds` alone is not usable for any supervised probe here: a DF40/CDFv3 generator
    group contains only fakes, so holding one out yields a single-class test set and every
    fold gets skipped -- silently producing "no result" rather than an error. The reals are
    a shared pool that belongs to no generator, so they are split once, deterministically,
    into disjoint train/test halves and the same split is reused by every fold.

    Guarantees:
      * no real video appears in both the train and test side of a fold;
      * the held-out generator's fakes never appear in training;
      * all other generators' fakes are available for training.

    Class imbalance within a fold (e.g. 18 fakes vs ~416 reals) is expected -- that is why
    every metric built on these folds is a *balanced* one.

    Pass `video_ids` for frame-level data. The real split is then made over whole videos, so
    two frames of the same real video can never land on opposite sides of a fold -- at frame
    level a naive row-wise split leaks near-duplicate frames into the test half and inflates
    every probe. Video-level callers can omit it.
    """
    is_real = methods == real_method
    if not is_real.any():  # fall back to label if the method column has no explicit reals
        is_real = labels == 0

    rng = np.random.default_rng(seed)
    real_test = np.zeros(len(methods), dtype=bool)
    if video_ids is None:
        shuffled = rng.permutation(np.flatnonzero(is_real))
        real_test[shuffled[:len(shuffled) // 2]] = True
    else:
        vids = pd.unique(video_ids[is_real])
        chosen = set(rng.permutation(vids)[:len(vids) // 2].tolist())
        real_test = is_real & np.array([v in chosen for v in video_ids])
    real_train = is_real & ~real_test

    out = []
    for g in sorted(pd.unique(methods[~is_real])):
        gm = (methods == g) & ~is_real
        if gm.sum() < min_per_group:
            continue
        test = gm | real_test
        train = (~is_real & ~gm) | real_train
        if len(np.unique(labels[test])) < 2 or len(np.unique(labels[train])) < 2:
            continue
        out.append((str(g), train, test))
    return out


def fold_scaler(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardise using TRAIN-fold statistics only.

    Deliberately takes both splits and returns both, so a caller cannot accidentally fit on
    the concatenation. Fitting globally before splitting would leak the held-out generator
    into the normalisation and inflate every LOGO number.
    """
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(x_train)
    return sc.transform(x_train), sc.transform(x_test)


def out_dir(name: str) -> Path:
    d = OUT_ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def todo(msg: str) -> str:
    """Marker for a cell that must be filled by a real run. Never invent numbers."""
    return f"TODO(run): {msg}"

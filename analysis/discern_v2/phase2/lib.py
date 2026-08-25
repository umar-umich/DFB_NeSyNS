#!/usr/bin/env python3
"""Shared machinery for the Phase-2 (v2 brief) stages.

Everything here is used by more than one stage, so it is implemented once rather than
re-derived per script. Four things in particular are single-sourced because a divergence between
two copies would be invisible in the results:

**The applicability target.** Stage 5 defines applicability as counterfactual marginal utility
under *the fusion operator that will be used downstream*. Stage 0.3 is required to use the same
definition so its "realizable recovery" number predicts what Stage 5 will actually learn.
`marginal_utility()` is that one definition, parameterised by the operator.

**Label-free gate inputs.** The target uses the label; the inputs must not. `assert_label_free()`
re-checks this from the column names rather than trusting the call site, because a leak here
silently invalidates every go/no-go the phase produces.

**Grouped cross-fitting.** Random k-fold over frames leaks: sibling frames of one video, and
sibling videos from one generator, are not independent. Folds are always grouped, and the group
key is explicit at every call site.

**Video aggregation.** Mean frame p(fake) per video, label = max. One implementation, so a
"video AUROC" always means the same thing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "training") not in sys.path:
    sys.path.insert(0, str(REPO / "training"))

from networks.discern_v2.ds_fusion import (  # noqa: E402
    Opinion, apply_validity, discount, ds_combine, js_divergence)

NOISE_FLOOR = 0.01                       # brief ground rule 3 / V1 §22
ANCHOR = "sem"
SPECIALISTS = ("ref", "proc", "rate")    # `rate` is Stage 2's MR-VAE branch (P1d)
DF40_SPLIT_FILE = REPO / "configs/discern_v2/df40_split.json"

# Any column whose value depends on the ground-truth label, on the specialist's measured utility,
# or on the identity of the dataset/generator. None of these may be a gate input.
FORBIDDEN_GATE_INPUTS = (
    "label", "y", "target", "t_", "phi", "dloss", "utility", "correct", "error",
    "dataset", "source", "method", "generator", "family", "manipulation", "key", "video",
)


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------

def load_samples(parquets: list[Path]) -> pd.DataFrame:
    """Concatenate per-sample exports. Requires pyarrow — use the `dfb_nesy` env."""
    frames = []
    for p in parquets:
        df = pd.read_parquet(p)
        df["_export"] = str(p)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    if out.duplicated(subset=["dataset", "key"]).any():
        n = int(out.duplicated(subset=["dataset", "key"]).sum())
        raise SystemExit(
            f"{n} (dataset, key) pairs appear in more than one export. Two evaluations of the "
            f"same frames were concatenated; every AUROC below would weight those frames twice.")
    return out


def load_df40_split() -> dict:
    if not DF40_SPLIT_FILE.is_file():
        raise SystemExit(f"{DF40_SPLIT_FILE} missing — run Stage 0.1 before any DF40 analysis.")
    return json.loads(DF40_SPLIT_FILE.read_text())


def assert_no_holdout(methods) -> None:
    """The seal, enforced rather than described (brief Stage 0.1)."""
    holdout = set(load_df40_split()["holdout"])
    leaked = sorted(set(map(str, methods)) & holdout)
    if leaked:
        raise SystemExit(
            f"DF40-Holdout methods reached a pre-Stage-8 analysis: {leaked}. The split is sealed; "
            f"reading these here would destroy the only zero-shot claim Stage 8 can make.")


# --------------------------------------------------------------------------------------
# opinions and fusion
# --------------------------------------------------------------------------------------

def branches_present(df: pd.DataFrame) -> list[str]:
    return [b for b in (ANCHOR, *SPECIALISTS) if f"e_{b}_real" in df.columns]


def opinions_of(df: pd.DataFrame, names: list[str] | None = None) -> dict[str, Opinion]:
    """Rebuild each branch's opinion from the exported evidence, validity applied.

    Rebuilt from `e_b_*` rather than read from `p_b` so every configuration below goes through
    the same code the model runs; a toggle then differs from the deployed system only in which
    opinions enter and with what q.
    """
    out: dict[str, Opinion] = {}
    for name in (names or branches_present(df)):
        evidence = torch.tensor(df[[f"e_{name}_real", f"e_{name}_fake"]].to_numpy(),
                                dtype=torch.float32)
        valid = torch.tensor(
            df.get(f"valid_{name}", pd.Series(True, index=df.index)).to_numpy().astype(float),
            dtype=torch.float32)
        out[name] = apply_validity(Opinion.from_evidence(evidence), valid)
    return out


def combine(opinions: list[Opinion], operator: str = "ds") -> Opinion:
    """Dispatch to the fusion operator named. Stage 6 compares `ds` against `ccf`."""
    if operator == "ds":
        fused, _ = ds_combine(opinions)
        return fused
    if operator == "ccf":
        from networks.discern_v2.ccf_fusion import ccf_combine
        return ccf_combine(opinions)
    raise ValueError(f"unknown fusion operator {operator!r}")


def fuse(opinions: dict[str, Opinion], names: list[str],
         q: dict[str, torch.Tensor] | None = None, operator: str = "ds") -> Opinion:
    """Fuse `names`, discounting specialists by q where given. The anchor is never discounted."""
    parts = []
    for name in names:
        op = opinions[name]
        if q is not None and name != ANCHOR and name in q:
            op = discount(op, q[name])
        parts.append(op)
    return combine(parts, operator)


# --------------------------------------------------------------------------------------
# the applicability target (Stage 5, reused by Stage 0.3)
# --------------------------------------------------------------------------------------

def _cross_entropy(p_fake: torch.Tensor, y: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    p = p_fake.clamp(eps, 1.0 - eps)
    return -(y * torch.log(p) + (1.0 - y) * torch.log(1.0 - p))


def marginal_utility(opinions: dict[str, Opinion], labels: torch.Tensor,
                     specialists: list[str], operator: str = "ds") -> dict[str, torch.Tensor]:
    """Per-sample counterfactual marginal utility `phi_b` (brief Stage 5).

    `U(S) = -CE(p_fusion(anchor + S), y)`, and `phi_b` is b's exact Shapley value in the
    cooperative game over the specialist set, with the anchor always present:

        phi_b = sum_{S subset of specialists\\{b}} w(|S|) [U(S + b) - U(S)]
        w(s)  = s! (n-1-s)! / n!

    For two specialists this is exactly the brief's
    `0.5[U(A+ref) - U(A)] + 0.5[U(A+ref+rate) - U(A+rate)]`; written generally so Stage 3 can
    drop a specialist without the formula becoming wrong. n <= 3 here, so enumerating all
    subsets costs 8 fusions and no approximation is needed.

    This is a TRAINING TARGET ONLY. It uses `labels`; gate inputs never do.
    """
    import itertools
    from math import factorial

    n = len(specialists)
    if n == 0:
        return {}
    # cache U(S) for every subset, so each fusion is computed once
    utility: dict[frozenset, torch.Tensor] = {}
    for size in range(n + 1):
        for subset in itertools.combinations(specialists, size):
            key = frozenset(subset)
            p = fuse(opinions, [ANCHOR, *subset], operator=operator).fake_prob()
            utility[key] = -_cross_entropy(p, labels)

    phi: dict[str, torch.Tensor] = {}
    for b in specialists:
        others = [s for s in specialists if s != b]
        total = torch.zeros_like(labels, dtype=torch.float32)
        for size in range(n):
            weight = factorial(size) * factorial(n - 1 - size) / factorial(n)
            for subset in itertools.combinations(others, size):
                key = frozenset(subset)
                total = total + weight * (utility[key | {b}] - utility[key])
        phi[b] = total
    return phi


def applicability_target(phi: torch.Tensor, delta: float = 0.0) -> np.ndarray:
    """`t_b = 1[phi_b > delta]` (brief Stage 5, delta = 0)."""
    return (phi.numpy() > delta).astype(np.int64)


# --------------------------------------------------------------------------------------
# gate features and fitting
# --------------------------------------------------------------------------------------

def assert_label_free(columns) -> None:
    offenders = sorted({c for c in columns
                        if any(tok in str(c).lower() for tok in FORBIDDEN_GATE_INPUTS)})
    if offenders:
        raise SystemExit(
            f"gate inputs {offenders} are label-, utility- or identity-derived. Applicability "
            f"must be inferable at inference time from evidence alone; training on any of these "
            f"produces a gate that cannot be deployed and a recovery number that means nothing.")


def gate_features(df: pd.DataFrame, specialist: str) -> pd.DataFrame:
    """The specialist's own evidence and confidence, plus its disagreement with the anchor.

    Deliberately compact (brief Stage 0.3: "its p, its u, compact branch diagnostics"). A wide
    feature set would let the gate memorise a generator fingerprint and inflate the ceiling this
    stage exists to measure.
    """
    p_b = df[f"p_{specialist}"].to_numpy()
    u_b = df[f"u_{specialist}"].to_numpy()
    p_a = df["p_sem"].to_numpy()
    u_a = df["u_sem"].to_numpy()
    feats = {
        "p_spec": p_b,
        "u_spec": u_b,
        "spec_margin": np.abs(p_b - 0.5),
        "p_anchor": p_a,
        "u_anchor": u_a,
        "anchor_margin": np.abs(p_a - 0.5),
        "abs_disagree": np.abs(p_b - p_a),
        "js_disagree": js_divergence(
            torch.tensor(np.stack([1 - p_b, p_b], 1), dtype=torch.float32),
            torch.tensor(np.stack([1 - p_a, p_a], 1), dtype=torch.float32)).numpy(),
    }
    extra = {"ref": ["ref_residual_norm", "ref_angle"],
             "proc": ["proc_mse_mean", "proc_center_ratio"],
             "rate": [c for c in df.columns if c.startswith("rate_r_")]}
    for col in extra.get(specialist, []):
        if col in df.columns:
            feats[col] = df[col].to_numpy()
    out = pd.DataFrame(feats, index=df.index)
    assert_label_free(out.columns)
    return out


def fit_logistic(X_train: np.ndarray, y_train: np.ndarray, X_apply: np.ndarray) -> np.ndarray:
    """Standardise on the training fold only, then a small logistic. Returns q on X_apply.

    The scaler is fit inside the fold on purpose: fitting it globally before splitting leaks the
    held-out group into the normalisation and silently inflates the diagnostic.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if len(np.unique(y_train)) < 2:
        # the target is constant on this fold — "always applicable" or "never"; say so with a
        # constant q rather than crashing or, worse, fitting a degenerate model
        return np.full(len(X_apply), float(y_train[0]) if len(y_train) else 0.5)
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(max_iter=2000, C=1.0).fit(scaler.transform(X_train), y_train)
    return model.predict_proba(scaler.transform(X_apply))[:, 1]


def cross_fit(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Leave-one-group-out out-of-fold q, so no sample scores its own gate.

    Grouped, never random: frames of one video and videos of one generator are not independent,
    and a random split lets the gate see siblings of what it scores.
    """
    q = np.full(len(y), np.nan)
    for g in np.unique(groups):
        held = groups == g
        q[held] = fit_logistic(X[~held], y[~held], X[held])
    if np.isnan(q).any():
        raise SystemExit("cross-fitting left samples unscored — check the group key")
    return q


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------

def video_auroc(prob: np.ndarray, labels: np.ndarray, videos: np.ndarray) -> float:
    """Mean frame p(fake) per video, label = max. One definition, used everywhere."""
    from sklearn.metrics import roc_auc_score

    agg = pd.DataFrame({"p": prob, "y": labels, "v": videos}).groupby(
        "v", as_index=False).agg(p=("p", "mean"), y=("y", "max"))
    if agg["y"].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(agg["y"], agg["p"]))


def verdict(delta: float, better: str, worse: str) -> str:
    if not np.isfinite(delta):
        return "not computable"
    if abs(delta) < NOISE_FLOOR:
        return f"inside the {NOISE_FLOOR} noise band — a confirming seed is required"
    return f"{better} better" if delta > 0 else f"{worse} better"

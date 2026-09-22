#!/usr/bin/env python3
"""Shared scoring core for the grouped-fusion study: one arm, one table, every endpoint.

Kept separate from the drivers so the confirmation stage, the development matrix and the exposed suites all compute
the same quantities the same way. Every arm is fitted and scored identically; only `ops.fuse` differs.
"""
from __future__ import annotations
import itertools, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent / "temporal")]
import ops  # noqa: E402
import exp1_fusion_audit as A  # noqa: E402
from safeguard_ablation import rc_metrics  # noqa: E402
from health import eer  # noqa: E402

GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
WGRID = [w for w in itertools.product(GRID, repeat=3) if max(w) == 1.0]
ONE = (1.0, 1.0, 1.0)


def views_of(t, w=ONE):
    """Per-view opinions from a scored table, discounted by evidence scaling."""
    out = {}
    for i, v in enumerate(ops.VIEWS):
        o, _bad = ops.opinion_from_pu(t[f"p_{v}"].to_numpy(), t[f"u_{v}"].to_numpy())
        out[v] = ops.discount(o, w[i])
    return out


def fused(t, arm, w=ONE, stats=None):
    return ops.fuse(views_of(t, w), arm, stats)


def fit_w(Fd, arm, cl=None):
    """Same grid, same objective, same folds as every other arm. Never imports another arm's weights."""
    cl = A.cells(Fd) if cl is None else cl
    return max(WGRID, key=lambda w: A.v31(ops.p_of(fused(Fd, arm, w)), cl)[0])


def endpoints(t, arm, w, cl=None, tau=None):
    """Every endpoint the pre-registration names, for one arm on one table."""
    cl = A.cells(t) if cl is None else cl
    st = {}
    o = fused(t, arm, w, st)
    p = ops.p_of(o)
    y = t.y.to_numpy()
    tot, ty = A.v31(p, cl)
    ff = np.where(t.source.to_numpy() == "FFpp_val")[0] if "source" in t else np.array([], int)
    if tau is None:
        tau = float(eer(y[ff], p[ff])[1]) if len(ff) else 0.5
    pred = (p >= tau).astype(int)
    err = pred != y
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    ent = -(pc * np.log(pc) + (1 - pc) * np.log(1 - pc))
    r = {"v31": tot, **ty, "tau": tau, "auroc": A.fast_auc(p[y == 0], p[y == 1]),
         "fpr": float((pred[y == 0] == 1).mean()) if (y == 0).any() else float("nan"),
         "tpr": float((pred[y == 1] == 1).mean()) if (y == 1).any() else float("nan"),
         "nll": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
         "brier": float(np.mean((p - y) ** 2)), "ece": ece(p, y),
         "u_mean": float(o[:, 2].mean()), "u_zero_frac": float((o[:, 2] <= 1e-12).mean()),
         "total_conflict": int(st.get("total_conflict", 0)), "n": int(len(t))}
    for nm, sig in (("vacuity", o[:, 2]), ("entropy", ent), ("margin", -np.abs(p - 0.5))):
        m = rc_metrics(sig, err)
        r[f"err_auroc_{nm}"] = float(m["auroc_err"])
        r[f"aurc_{nm}"], r[f"eaurc_{nm}"] = float(m["aurc"]), float(m["eaurc"])
    return r, p, o


def ece(p, y, bins=15):
    e, n = 0.0, len(p)
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.sum():
            e += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return float(e)


def group_boot(groups, n_draw, seed):
    """Paired source-group bootstrap weights. Every linked copy of a source moves with it."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    pos = {g: np.where(groups == g)[0] for g in ug}
    out = []
    for _ in range(n_draw):
        w = np.zeros(len(groups), dtype=int)
        for g in rng.choice(ug, len(ug), replace=True):
            w[pos[g]] += 1
        out.append(w)
    return out

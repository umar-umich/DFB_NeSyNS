#!/usr/bin/env python3
"""The fixed comparison matrix of Section 4, and the frozen uncertainty scores of Section 5."""
from __future__ import annotations
import itertools, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "grouped"), str(HERE.parent), str(HERE.parent / "temporal")]
import ops_pcr6p as P  # noqa: E402
import ops as G  # noqa: E402
import exp1_fusion_audit as A  # noqa: E402
from health import eer  # noqa: E402

VIEWS = ("T", "APP", "OM")
GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
WGRID = [w for w in itertools.product(GRID, repeat=3) if max(w) == 1.0]
ARMS = ["PCR6P-R", "PCR6-R", "PCR6-pairwise-R", "Yager-R", "Averaging-R", "Dempster-R",
        "T+OM (PCR6+)", "T+APP (PCR6+)", "APP+OM (PCR6+)", "T only"]
# Two- and one-view ablations under the proposed operator. For a single view no combination takes place, so the
# `T only` arm is operator-independent; it is carried here so every ablation is fitted and scored identically.
SUBSET = {"T+OM (PCR6+)": ("T", "OM"), "T+APP (PCR6+)": ("T", "APP"),
          "APP+OM (PCR6+)": ("APP", "OM"), "T only": ("T",)}


def fuse(t, arm, w, attribute=False):
    """Fused opinion for one arm. `w` is the (T, APP, OM) reliability triple."""
    if arm in SUBSET:
        vs = SUBSET[arm]
        idx = [VIEWS.index(v) for v in vs]
        m = P.masses_of(t, vs, [w[i] for i in idx])
        if len(vs) == 1:                                   # a single view is its own fused opinion
            f = m[:, 0, :]
            if attribute:
                n = len(f)
                return f, {"K": np.zeros(n), "K_TS": np.zeros(n), "K_AO": np.zeros(n), "K_all": np.zeros(n)}
            return f
        return P.fuse(m, True, attribute)
    m = P.masses_of(t, VIEWS, w)
    if arm == "PCR6P-R":
        return P.fuse(m, True, attribute)
    if arm == "PCR6-R":
        return P.fuse(m, False, attribute)
    if attribute:
        _f, att = P.fuse(m, True, True)          # attribution is a property of the opinions, not of the rule
    if arm == "Dempster-R":
        f = P.fuse_simple(m, "dempster")
    elif arm == "Yager-R":
        f = P.fuse_simple(m, "yager")
    elif arm == "Averaging-R":
        f = G.evidence_average(*[m[:, i, :] for i in range(m.shape[1])])
    elif arm == "PCR6-pairwise-R":               # the historical fold, kept for the labelled comparison row
        f = m[:, 0, :]
        for i in (1, 2):
            f = G.pcr6_2(f, m[:, i, :])
    else:
        raise ValueError(arm)
    return (f, att) if attribute else f


def fit_w(Fd, arm, cl=None):
    cl = A.cells(Fd) if cl is None else cl
    return max(WGRID, key=lambda w: A.v31(P.p_of(fuse(Fd, arm, w)), cl)[0])


def scores(t, arm, w):
    """The frozen uncertainty scores. Higher = more suspect. None of them changes the decision."""
    f, att = fuse(t, arm, w, attribute=True)
    p = P.p_of(f)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return p, f, {"vacuity": f[:, 2], "conflict_K": att["K"],
                  "entropy": -(pc * np.log(pc) + (1 - pc) * np.log(1 - pc)),
                  "margin": -np.abs(p - 0.5)}, att


def tau_of(p, t):
    ff = np.where(t.source.to_numpy() == "FFpp_val")[0]
    return float(eer(t.y.to_numpy()[ff], p[ff])[1]) if len(ff) else 0.5

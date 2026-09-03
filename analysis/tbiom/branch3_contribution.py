#!/usr/bin/env python3
"""Measure Branch 3's actual contribution to the fused opinion, from exports alone.

    python analysis/tbiom/branch3_contribution.py logs/tbiom/stage23/stage5_e08_VALmix.csv

WHY THIS IS POSSIBLE WITHOUT RE-RUNNING THE MODEL. The exporter writes, per view, the evidential
expectation p = alpha_fake / S and the Dirichlet uncertainty u = K / S. Those two invert exactly:

    S = K / u                 alpha_fake = p * S        alpha_real = (1 - p) * S

so each view's full Dirichlet opinion is recoverable, and DS fusion can be replayed offline. If
the replay of all three views reproduces the exported p_fused, the inversion is correct and the
two-view counterfactual (semantic + artifact, Branch 3 dropped) is trustworthy.

THE QUESTION. Branch 3 is rank-informative on some domains but carries almost no evidence mass
(u_process ~ 0.996). A vacuous opinion is the IDENTITY of the Dempster-Shafer orthogonal sum, so
the architectural prediction is that Branch 3 changes the fused output by nearly nothing however
much it knows. This measures that instead of arguing it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from health import auroc  # noqa: E402

K = 2


def opinion(p: np.ndarray, u: np.ndarray) -> np.ndarray:
    """(p, u) -> alpha, shape (n, 2) ordered [real, fake]."""
    S = K / np.clip(u, 1e-12, None)
    return np.stack([(1.0 - p) * S, p * S], axis=1)


def ds_two(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    """One Dempster-Shafer orthogonal sum, matching core_model._combine_two_opinions."""
    out = []
    for a in (a1, a2):
        S = a.sum(axis=1, keepdims=True)
        out.append(((a - 1.0) / S, K / S))
    (b1, u1), (b2, u2) = out
    # conflict = sum of off-diagonal belief products
    conflict = (b1.sum(axis=1) * b2.sum(axis=1)) - (b1 * b2).sum(axis=1)
    denom = (1.0 - conflict)[:, None]
    b = (b1 * b2 + b1 * u2 + b2 * u1) / denom
    u = (u1 * u2) / denom
    return b * (K / u) + 1.0


def ds(*alphas: np.ndarray) -> np.ndarray:
    f = alphas[0]
    for nxt in alphas[1:]:
        f = ds_two(f, nxt)
    return f


def p_of(alpha: np.ndarray) -> np.ndarray:
    return alpha[:, 1] / alpha.sum(axis=1)


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1
                else "logs/tbiom/stage23/stage5_e08_VALmix.csv")
    d = pd.read_csv(path)
    if "p_process" not in d:
        print(f"{path} has no p_process column — not a three-branch export")
        return 1

    sem = opinion(d.p_semantic.to_numpy(), d.u_semantic.to_numpy())
    art = opinion(d.p_artifact.to_numpy(), d.u_artifact.to_numpy())
    prc = opinion(d.p_process.to_numpy(), d.u_process.to_numpy())

    p3 = p_of(ds(sem, art, prc))
    p2 = p_of(ds(sem, art))
    err = np.abs(p3 - d.p_fused.to_numpy()).max()
    print(f"file: {path.name}   frames: {len(d)}")
    print(f"replay check — max |replayed 3-view - exported p_fused| = {err:.2e}"
          f"   {'OK' if err < 1e-3 else 'MISMATCH: inversion is wrong, ignore what follows'}\n")

    d["dom"] = "ALL"
    for n in ("CDFv2val", "DFDCPval", "DFEval24val", "CDFv3", "CDFv2", "DFDCP", "DFEval24"):
        m = d.key.astype(str).str.contains(n, regex=False)
        if m.any():
            d.loc[m, "dom"] = n
    groups = [(g, s) for g, s in d.groupby("dom")] + [("== ALL ==", d)]

    print(f"{'domain':13s} {'n':>7s} {'2-view':>9s} {'3-view':>9s} {'Δ':>9s} "
          f"{'p_process':>10s} {'mean u_proc':>12s}")
    for g, s in groups:
        if s.label.nunique() < 2:
            continue
        i = s.index.to_numpy()
        a2 = auroc(s.label.to_numpy(), p2[i])
        a3 = auroc(s.label.to_numpy(), p3[i])
        print(f"{g:13s} {len(s):>7d} {a2:>9.4f} {a3:>9.4f} {a3-a2:>+9.4f} "
              f"{auroc(s.label.to_numpy(), s.p_process.to_numpy()):>10.4f} "
              f"{s.u_process.mean():>12.4f}")

    print("\nA vacuous opinion (u -> 1) is the DS identity: it leaves the fusion unchanged.")
    print("So Δ is what Branch 3 is WORTH to the framework, whatever its own AUROC says.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

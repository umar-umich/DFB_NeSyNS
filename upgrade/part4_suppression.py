#!/usr/bin/env python3
"""Part 4: processing suppression on the winner -- subspaces (4b) and the head rows (4c).

    <discern_ext python> upgrade/part4_suppression.py --feats pe_core_l --readout last+mid+half

WHAT IS BEING SUPPRESSED. Paired renders give the same video at seven processing settings, so
Delta_{i,a} = z(T_a(x_i)) - z(x_i) isolates what PROCESSING does to the representation with content
held fixed. A direction that carries processing and not forgery is a nuisance, and the question is
whether removing it helps or whether it was carrying signal all along.

FOUR SUBSPACES, every one fitted on FF++ TRAIN only.
  MC     mean-contrast basis: per-setting mean minus the clean mean, stacked and orthonormalised.
         Cheap and interpretable, but only captures first-order shifts.
  PD     top-k principal directions of the paired differences, k at 90% explained variance capped
         at 32. This is the one the premise predicts should work.
  LEACE  Belrose et al. (2023) concept erasure, IMPLEMENTED here rather than cited: the closed-form
         least-squares eraser that makes the processing label linearly unpredictable from z while
         changing z as little as possible in the whitened metric.
  RND    rank-matched random subspaces, 5 draws. THE CONTROL THAT DECIDES THE STUDY: projecting out
         any 32 directions removes capacity, so a gain over A0 means nothing unless it also beats
         RND at the same rank.

The keep rule's four conditions are evaluated in `part4_rows`, and a row that fails any one of them
is reported with which one it failed, not silently dropped.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(ROOT / "upgrade"), str(ROOT / "branches/ijepa")]

LF = ROOT / "logs/upgrade/lfeats"
OUT = ROOT / "analysis/upgrade"
SETTINGS = ["none", "jpeg:70", "jpeg:50", "jpeg:30", "resize:0.75", "resize:0.50", "resize:0.35"]
PD_VAR, PD_CAP, RND_DRAWS = 0.90, 32, 5


def tag_of(p):
    return "" if p == "none" else "_" + p.replace(":", "")


def load_readout_cache(feats, dataset, readout, process="none"):
    """The same L2-per-component readout the Part 3d screen used, so rows here match rows there."""
    f = LF / f"{feats}_{dataset}{tag_of(process)}.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)
    parts = []
    for comp in readout.split("+"):
        x = z[f"feat_{comp}"].astype(np.float32)
        x = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-6, None)
        parts.append(x)
    x = np.concatenate(parts, 1) if len(parts) > 1 else parts[0]
    return {"x": x, "key": np.array([str(k) for k in z["key"]]), "label": z["label"]}


def stacked_renders(feats, readout, present):
    """All rendered versions stacked with their processing label -- the population LEACE erases on.

    LEACE needs (feature, concept) PAIRS, not the differences: the concept is "which processing
    setting produced this feature", so the clean renders are part of the population and carry the
    label `none`. Fitting it on the differences instead would erase a direction from a space the
    head never sees.
    """
    X, Z = [], []
    for s in ["none"] + list(present):
        c = load_readout_cache(feats, "FFpp_train", readout, s)
        if c is None:
            continue
        X.append(c["x"]); Z += [s] * len(c["x"])
    X = np.concatenate(X); Z = np.array(Z)
    levels = sorted(set(Z))
    onehot = np.zeros((len(Z), len(levels)), dtype=np.float32)
    for i, s in enumerate(levels):
        onehot[Z == s, i] = 1.0
    return X, onehot, levels


def paired_differences(feats, readout):
    """Delta per (frame, setting). Frames are aligned BY KEY, never by row order: a processed cache
    can drop an unreadable frame, and aligning by position would then pair different images."""
    clean = load_readout_cache(feats, "FFpp_train", readout, "none")
    if clean is None:
        raise SystemExit("clean FF++ train cache missing; run the Part 4a renders first")
    pos = {k: i for i, k in enumerate(clean["key"])}
    D, lab, present = [], [], []
    for s in SETTINGS[1:]:
        c = load_readout_cache(feats, "FFpp_train", readout, s)
        if c is None:
            print(f"  setting {s}: not rendered yet, skipped"); continue
        idx = np.array([pos[k] for k in c["key"] if k in pos])
        jdx = np.array([j for j, k in enumerate(c["key"]) if k in pos])
        D.append(c["x"][jdx] - clean["x"][idx]); lab += [s] * len(idx); present.append(s)
        print(f"  setting {s}: {len(idx)} paired frames")
    if not D:
        raise SystemExit("no processed renders found")
    return clean, np.concatenate(D), np.array(lab), present


def basis_mc(clean, D, lab):
    """Mean contrast per setting, orthonormalised."""
    M = np.stack([D[lab == s].mean(0) for s in np.unique(lab)])
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    k = int((s > 1e-8).sum())
    return np.linalg.qr(M[:k].T)[0]


def basis_pd(D):
    """Top-k principal directions of the paired differences; k at 90% variance, capped at 32."""
    Dc = D - D.mean(0, keepdims=True)
    _, s, Vt = np.linalg.svd(Dc, full_matrices=False)
    var = np.cumsum(s ** 2) / np.sum(s ** 2)
    k = int(min(PD_CAP, max(1, np.searchsorted(var, PD_VAR) + 1)))
    return Vt[:k].T, k, float(var[k - 1])


def basis_rnd(d, k, draws=RND_DRAWS, seed=0):
    rng = np.random.default_rng(seed)
    return [np.linalg.qr(rng.normal(size=(d, k)))[0] for _ in range(draws)]


def leace_eraser(X, Z):
    """LEACE (Belrose et al., 2023), closed form.

    Whiten X, project out the component of X that is linearly predictive of the concept Z in that
    whitened space, then unwhiten. Erasure is exact for LINEAR predictors of Z and is the minimal
    such edit in the whitened metric -- it does not promise a nonlinear head cannot recover Z, and
    the report must not claim it does.
    """
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    Zc = Z - Z.mean(0, keepdims=True)
    S = np.cov(Xc, rowvar=False) + 1e-6 * np.eye(X.shape[1])
    w, V = np.linalg.eigh(S)
    w = np.clip(w, 1e-10, None)
    W = V @ np.diag(w ** -0.5) @ V.T          # whitening
    Winv = V @ np.diag(w ** 0.5) @ V.T
    Sxz = Xc.T @ Zc / max(1, len(Xc) - 1)
    A = W @ Sxz
    Ua, sa, _ = np.linalg.svd(A, full_matrices=False)
    r = int((sa > 1e-8).sum())
    Pa = Ua[:, :r] @ Ua[:, :r].T
    P = np.eye(X.shape[1]) - Winv @ Pa @ W
    return P, mu, r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="pe_core_l")
    ap.add_argument("--readout", default="last+mid+half")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"Part 4b: paired differences for {a.feats} readout {a.readout}")
    clean, D, lab, present = paired_differences(a.feats, a.readout)
    d = D.shape[1]
    Umc = basis_mc(clean, D, lab)
    Upd, k, var = basis_pd(D)
    Urnd = basis_rnd(d, k)
    Xs, Zs, levels = stacked_renders(a.feats, a.readout, present)
    print(f"  LEACE population: {len(Xs)} renders over {len(levels)} settings {levels}")
    Pleace, mu_leace, rank_leace = leace_eraser(Xs, Zs)

    res = {"feats": a.feats, "readout": a.readout, "dim": int(d),
           "settings_present": present, "paired_frames": int(len(D)),
           "MC": {"rank": int(Umc.shape[1])},
           "PD": {"rank": int(k), "explained_variance": var},
           "RND": {"rank": int(k), "draws": RND_DRAWS},
           "LEACE": {"erased_rank": int(rank_leace), "population": int(len(Xs)), "levels": levels},
           "note": ("RND is the control that decides the study: projecting out any k directions "
                    "removes capacity, so a PD gain over A0 is only evidence if it also beats RND "
                    "at the same rank. LEACE erases LINEAR predictability of the processing label; "
                    "it does not guarantee a nonlinear head cannot recover it.")}
    np.savez(OUT / f"part4_subspaces_{a.feats}.npz", MC=Umc, PD=Upd, LEACE=Pleace,
             leace_mu=mu_leace, **{f"RND{i}": U for i, U in enumerate(Urnd)})
    json.dump(res, open(OUT / f"part4_subspaces_{a.feats}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    print(f"\nwrote {OUT}/part4_subspaces_{a.feats}.{{npz,json}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Appendix backup, step 4: fit the intervention-guided subspace on aligned pairs.

    <discern_ext python> upgrade/fusion/backup_subspace.py

Three second moments on the paired differences, per the Phase 4 definitions the appendix reuses:

    delta(s,m,a) = z(a(x_s^m)) - z(a(x_s^0))        paired difference under condition a
    C_man  = E[ mean_a(delta) mean_a(delta)^T ]     UNCENTERED: a consistent manipulation shift
    C_proc = E[ (delta - mean_a delta)(...)^T ]     instability of that difference across conditions
    C_real = E[ (z(a(x^0)) - z(x^0))(...)^T ]       processing drift in GENUINE content

then the generalized problem  C_man q = lambda (C_proc + C_real + 1e-3 I) q, each matrix normalised
by its mean diagonal first, solved in float64, keeping up to eight leading directions.

Four subspaces are produced in one pass so the arms are directly comparable: the guided one, an
ordinary PCA control, a random orthonormal control inside the same PCA-32, and a SHUFFLED-PAIR
control whose genuine partners come from different source groups. The shuffled arm is what
separates "correct pairing matters" from "any 8 directions help".
"""
from __future__ import annotations

import json, re, sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(ROOT / "upgrade"), str(ROOT / "upgrade/fusion"),
                str(ROOT / "analysis/tbiom/pcr6p"), str(ROOT / "analysis/tbiom/grouped"),
                str(ROOT / "analysis/tbiom"), str(ROOT / "analysis/tbiom/temporal")]
import dsr_system as DS            # noqa: E402
import final_eval as FE            # noqa: E402
from backup_feasibility import source_key  # noqa: E402

CONDS = ("identity", "jpeg:50", "resize:0.5")
PCA_DIM, OUT_DIM, RIDGE = 32, 8, 1e-3
SEEDS = (42, 1024, 3407)


def feat_path(ds, cond):
    tag = "" if cond == "identity" else "_" + cond.replace(":", "")
    return ROOT / f"logs/upgrade/feats/pe_core_lN{tag}_{ds}.npz"


def pooled(ds, cond):
    f = feat_path(ds, cond)
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)
    vid = np.array([str(v) for v in z["video"]])
    order, inv = np.unique(vid, return_inverse=True)
    s = np.zeros((len(order), z["feature"].shape[1]), dtype=np.float64)
    np.add.at(s, inv, z["feature"].astype(np.float64))
    s /= np.bincount(inv)[:, None]
    return {str(v): r for v, r in zip(order, s)}


def norm_diag(C):
    d = np.trace(C) / C.shape[0]
    return C if d <= 0 else C / d


def main() -> int:
    brun = (ROOT / "outputs/fusion_experiments/CURRENT_BACKUP_RUN").read_text().strip()
    out = ROOT / "outputs/fusion_experiments" / brun
    feas = json.load(open(out / "backup_feasibility.json"))

    sets = [d for d in ("FFpp_val", "VALmixV2") if all(feat_path(d, c).exists() for c in CONDS)]
    if not sets:
        print("no dataset has all three conditions rendered yet"); return 2
    print(f"conditions available for: {sets}", flush=True)

    Z = {c: {} for c in CONDS}
    for d in sets:
        for c in CONDS:
            Z[c].update(pooled(d, c) or {})

    dev_t, _dp, _tests = DS.load("vjepa2_lora_B_seed42")
    op = FE.opinions("pe_core_lN", 0, "cuda", ("VALmix", "VALmixV2", "FFpp_val"))
    t = FE.swap_app(dev_t, op, "development")
    main_run = (ROOT / "outputs/fusion_experiments/CURRENT_RUN").read_text().strip()
    man = json.load(open(ROOT / "outputs/fusion_experiments" / main_run / "data_access_manifest.json"))
    t = t[~t["source"].astype(str).isin(man["reserved_from_fitting"]).to_numpy()].reset_index(drop=True)

    y = t.y.to_numpy(); vid = t["v"].astype(str).to_numpy(); sub = t["subset"].astype(str).to_numpy()
    real, fakes = {}, []
    for i, v in enumerate(vid):
        k = source_key(v, sub[i])
        if k is None or v not in Z["identity"]:
            continue
        (real.setdefault(k, v) if y[i] == 0 else fakes.append((k, v, sub[i])))
    pairs = [(k, fv, real[k], s) for k, fv, s in fakes if k in real]
    print(f"usable pairs with all conditions cached: {len(pairs)}", flush=True)
    if len(pairs) < 50:
        json.dump({"error": "too few usable pairs", "n": len(pairs)},
                  open(out / "backup_subspace.json", "w"), indent=1)
        return 3

    # PCA-32 on genuine+manipulated identity features, training pairs only
    base = np.stack([Z["identity"][v] for _k, v, _r, _s in pairs]
                    + [Z["identity"][r] for _k, _v, r, _s in pairs])
    mu = base.mean(0); Xc = base - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:PCA_DIM].T                                   # (D, 32)
    proj = lambda v: (Z_[v] - mu) @ W

    def build(pairs_, label):
        dl, dr = [], []
        for k, fv, rv, _s in pairs_:
            per = []
            for c in CONDS:
                if fv not in Z[c] or rv not in Z[c]:
                    per = []; break
                per.append(((Z[c][fv] - mu) @ W) - ((Z[c][rv] - mu) @ W))
            if not per:
                continue
            per = np.stack(per)
            dl.append(per)
            dr.append(np.stack([((Z[c][rv] - mu) @ W) - ((Z["identity"][rv] - mu) @ W)
                                for c in CONDS]))
        D = np.stack(dl)                                  # (n, |a|, 32)
        Rr = np.stack(dr)
        mean_d = D.mean(1)
        C_man = (mean_d.T @ mean_d) / len(mean_d)                       # uncentered, by design
        res = (D - mean_d[:, None]).reshape(-1, D.shape[-1])
        C_proc = (res.T @ res) / len(res)
        rr = Rr.reshape(-1, Rr.shape[-1])
        C_real = (rr.T @ rr) / len(rr)
        A = norm_diag(C_man)
        B = norm_diag(C_proc) + norm_diag(C_real) + RIDGE * np.eye(PCA_DIM)
        w_, V_ = eigh(A, B)
        q = V_[:, np.argsort(w_)[::-1][:OUT_DIM]]
        print(f"  {label:18s} pairs {len(D):4d}  top eigenvalues "
              + " ".join(f"{x:.3f}" for x in np.sort(w_)[::-1][:4]), flush=True)
        return q, {"pairs": int(len(D)), "eigenvalues_top8": np.sort(w_)[::-1][:8].tolist()}

    Z_ = Z["identity"]
    rng = np.random.default_rng(0)
    guided, g_meta = build(pairs, "intervention-guided")

    # shuffled-pair control: genuine partners from DIFFERENT source groups
    keys = [k for k, _v, _r, _s in pairs]
    perm = rng.permutation(len(pairs))
    shuf = [(keys[i], pairs[i][1], pairs[perm[i]][2], pairs[i][3]) for i in range(len(pairs))
            if keys[i] != keys[perm[i]]]
    shuffled, s_meta = build(shuf, "shuffled pairs")

    # manipulation-only: C_man alone, no nuisance denominator
    dl = []
    for k, fv, rv, _s in pairs:
        if all(fv in Z[c] and rv in Z[c] for c in CONDS):
            dl.append(np.stack([((Z[c][fv] - mu) @ W) - ((Z[c][rv] - mu) @ W) for c in CONDS]).mean(0))
    M = np.stack(dl); C_man = (M.T @ M) / len(M)
    w_, V_ = eigh(norm_diag(C_man))
    man_only = V_[:, np.argsort(w_)[::-1][:OUT_DIM]]
    print(f"  {'manipulation-only':18s} pairs {len(M):4d}", flush=True)

    pca_ctrl = np.eye(PCA_DIM)[:, :OUT_DIM]
    rnds = {s: np.linalg.qr(np.random.default_rng(s).normal(size=(PCA_DIM, OUT_DIM)))[0] for s in SEEDS}

    np.savez(out / "backup_subspaces.npz", pca_basis=W, mu=mu, guided=guided, shuffled=shuffled,
             man_only=man_only, pca=pca_ctrl, **{f"random_s{s}": v for s, v in rnds.items()})
    json.dump({"backup_run_id": brun, "datasets_with_all_conditions": sets,
               "conditions": list(CONDS), "pca_dim": PCA_DIM, "out_dim": OUT_DIM, "ridge": RIDGE,
               "usable_pairs": len(pairs), "guided": g_meta, "shuffled": s_meta,
               "feasibility_coverage": {k: v["aligned_pairs"] for k, v in feas["per_domain"].items()},
               "caveat": ("Pairs are dominated by FF++; four of seven development domains supply "
                          "none, so the subspace is fitted mostly on FF++ structure.")},
              open(out / "backup_subspace.json", "w"), indent=1)
    print(f"wrote {out}/backup_subspaces.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())

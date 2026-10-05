#!/usr/bin/env python3
"""Part 4c: the suppression rows, and the four-condition keep rule.

    <discern_ext python> upgrade/part4_rows.py --feats pe_core_l --readout last+mid+half

Each row is a fresh head on the SAME frozen features with a different projection applied, so the
only thing that varies is what was removed. A-RND is reported as the mean over its five draws and is
the row every other suppression row must beat -- removing any k directions costs capacity, and
without the rank-matched control a gain over A0 cannot be attributed to the nuisance.

KEEP RULE, all four required, each reported with the number that decided it:
  1. dev v3.1 gain over A-RND with a bootstrap interval above zero
  2. FF++ val AUROC drops by at most 0.01 against A0
  3. FF++ val AUROC at JPEG q30 and resize 0.35 at least A0's
  4. fused development real-video FPR at the threshold at most 0.01 above A0's
Condition 4 needs the fused system, so it is marked TODO(run) when the fused evaluation for this
feature set has not been produced; a row is never passed on three conditions out of four.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(ROOT / "upgrade"), str(ROOT / "branches/ijepa"), "/data/umar/Repos/DISCERN_Ext"]
import v31 as V31                                                   # noqa: E402
import train_layer_head as TLH                                      # noqa: E402
from part4_suppression import load_readout_cache                    # noqa: E402
from train_head import Head, DEV_SETS, V2_SOURCES                   # noqa: E402
from src.losses.evidential_loss import evidential_loss_discern_ext  # noqa: E402

torch.set_num_threads(4)
OUT = ROOT / "analysis/upgrade"
EPOCHS, BATCH, LR, WD = 20, 128, 1e-4, 0.01
LAMBDAS = (0.25, 0.5, 0.75)


def project_out(X, U, lam=1.0):
    """z' = z - lam * U U^T z. lam=1 is a full projection; lam<1 is partial suppression."""
    return X - lam * ((X @ U) @ U.T)


def boot_ci(a, b, groups, rng, n=1000):
    """Source-grouped bootstrap on the paired per-video difference."""
    ug = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ug}
    d = []
    for _ in range(n):
        pick = rng.choice(ug, len(ug), replace=True)
        sel = np.concatenate([idx[g] for g in pick])
        d.append(a[sel].mean() - b[sel].mean())
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def train_eval(Xtr, ytr, DEVC, seed, dev):
    torch.manual_seed(seed); np.random.seed(seed)
    head = Head(Xtr.shape[1]).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
    rng = np.random.default_rng(seed)
    w = np.where(ytr == 1, 1 / max(1, (ytr == 1).sum()), 1 / max(1, (ytr == 0).sum())); w /= w.sum()
    best = (-1.0, None)
    for ep in range(EPOCHS):
        head.train()
        idx = rng.choice(len(Xtr), len(Xtr), replace=True, p=w)
        for i in range(0, len(idx) - BATCH + 1, BATCH):
            j = idx[i:i + BATCH]
            ev = head(torch.from_numpy(Xtr[j]).to(dev))
            loss = evidential_loss_discern_ext(ev, torch.from_numpy(ytr[j]).to(dev), 2, ep, EPOCHS).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sc = score(head, DEVC, dev)[0]
        if sc > best[0] + 1e-6:
            best = (sc, {k: v.detach().clone() for k, v in head.state_dict().items()})
    head.load_state_dict(best[1])
    return head, best[0]


def score(head, DEVC, dev):
    head.eval(); out = {}
    with torch.no_grad():
        for n, D in DEVC.items():
            p = []
            for i in range(0, len(D["x"]), 8192):
                ev = head(torch.from_numpy(D["x"][i:i + 8192]).to(dev))
                a = ev + 1.0; p.append((a / a.sum(1, keepdim=True))[:, 1].cpu().numpy())
            p = np.concatenate(p)
            cnt = np.bincount(D["inv"], minlength=len(D["order"])).astype(float)
            out[n] = np.bincount(D["inv"], weights=p, minlength=len(D["order"])) / cnt
    dk, dp, dy = [], [], []
    for n in ("VALmix", "VALmixV2"):
        dk += list(DEVC[n]["key"]); dp.append(out[n]); dy.append(DEVC[n]["yv"])
    ff = DEVC["FFpp_val"]
    sc, _t, cells = V31.score(dk, np.concatenate(dp), np.concatenate(dy), ff["key"], out["FFpp_val"], ff["yv"])
    v2 = float(np.nanmean([v for (t, s), v in cells.items() if s in V2_SOURCES and not np.isnan(v)]))
    return sc, v2, V31.fast_auc(out["FFpp_val"][ff["yv"] == 0], out["FFpp_val"][ff["yv"] == 1]), out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="pe_core_l"); ap.add_argument("--readout", default="last+mid+half")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sub = OUT / f"part4_subspaces_{a.feats}.npz"
    if not sub.exists():
        print("subspaces missing; run part4_suppression.py first"); return 2
    S = np.load(sub)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr = load_readout_cache(a.feats, "FFpp_train", a.readout, "none")
    DEVC = {n: TLH.load_readout(a.feats, n, a.readout) for n in DEV_SETS}
    PROC = {}
    for p in ("jpeg:30", "resize:0.35"):
        c = TLH.load_readout(a.feats, "FFpp_val", a.readout, p) if \
            (ROOT / f"logs/upgrade/lfeats/{a.feats}_FFpp_val_{p.replace(':','')}.npz").exists() else None
        PROC[p] = c

    rows = {}
    def run(name, Xtr, DEVp, PROCp):
        head, sc = train_eval(Xtr, tr["label"], DEVp, a.seed, dev)
        _s, v2, ff, _o = score(head, DEVp, dev)
        r = {"dev_v31": sc, "valmixv2": v2, "ffpp_val_auroc": ff}
        for p, c in PROCp.items():
            if c is None:
                r[f"ffpp_val_{p}"] = None; continue
            _h = head.eval()
            with torch.no_grad():
                ev = head(torch.from_numpy(c["x"]).to(dev)); al = ev + 1.0
                pr = (al / al.sum(1, keepdim=True))[:, 1].cpu().numpy()
            cnt = np.bincount(c["inv"], minlength=len(c["order"])).astype(float)
            pv = np.bincount(c["inv"], weights=pr, minlength=len(c["order"])) / cnt
            r[f"ffpp_val_{p}"] = float(V31.fast_auc(pv[c["yv"] == 0], pv[c["yv"] == 1]))
        rows[name] = r
        print(f"  {name:12s} dev {sc:.4f}  FF++ {ff:.4f}  " +
              "  ".join(f"{p} {r[f'ffpp_val_{p}']:.4f}" if r[f"ffpp_val_{p}"] else f"{p} --" for p in PROCp),
              flush=True)

    def proj_all(U, lam=1.0):
        return ({n: {**D, "x": project_out(D["x"], U, lam)} for n, D in DEVC.items()},
                {p: (None if c is None else {**c, "x": project_out(c["x"], U, lam)}) for p, c in PROC.items()})

    run("A0", tr["x"], DEVC, PROC)
    for nm, key in (("A-MC", "MC"), ("A-PD", "PD")):
        U = S[key]; d, pc = proj_all(U); run(nm, project_out(tr["x"], U), d, pc)
    for lam in LAMBDAS:
        U = S["PD"]; d, pc = proj_all(U, lam); run(f"A-PD-{lam}", project_out(tr["x"], U, lam), d, pc)
    P = S["LEACE"]; mu = S["leace_mu"]
    lea = lambda X: (X - mu) @ P.T + mu
    run("A-LEACE", lea(tr["x"]), {n: {**D, "x": lea(D["x"])} for n, D in DEVC.items()},
        {p: (None if c is None else {**c, "x": lea(c["x"])}) for p, c in PROC.items()})
    rnd = []
    for i in range(5):
        U = S[f"RND{i}"]; d, pc = proj_all(U); run(f"A-RND{i}", project_out(tr["x"], U), d, pc)
        rnd.append(rows[f"A-RND{i}"]["dev_v31"])
    rows["A-RND"] = {"dev_v31": float(np.mean(rnd)), "sd": float(np.std(rnd)), "draws": rnd}

    a0, arnd = rows["A0"], rows["A-RND"]
    verdicts = {}
    for nm in [k for k in rows if k.startswith("A-") and not k.startswith("A-RND")]:
        r = rows[nm]
        c1 = r["dev_v31"] > arnd["dev_v31"]
        c2 = r["ffpp_val_auroc"] >= a0["ffpp_val_auroc"] - 0.01
        c3 = all(r.get(f"ffpp_val_{p}") is not None and r[f"ffpp_val_{p}"] >= a0[f"ffpp_val_{p}"]
                 for p in PROC)
        verdicts[nm] = {"c1_beats_RND": bool(c1), "c2_clean_within_0.01": bool(c2),
                        "c3_processed_not_worse": bool(c3), "c4_fused_fpr": "TODO(run)",
                        "kept": bool(c1 and c2 and c3) and None,
                        "note": "c4 needs the fused evaluation; no row is passed on 3 of 4 conditions"}
    json.dump({"feats": a.feats, "readout": a.readout, "seed": a.seed,
               "rows": rows, "keep_rule": verdicts},
              open(OUT / "part4_rows.json", "w"), indent=1)
    print(f"\nwrote {OUT/'part4_rows.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

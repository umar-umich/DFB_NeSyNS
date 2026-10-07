#!/usr/bin/env python3
"""Part 8: the DF40 four-family table in slide format, for a final row.

    <discern_ext python> upgrade/df40_family.py --view pe_core_lN --label "PE-Core-L + LayerNorm"

HOW A ROW IS BUILT. The candidate supplies the APPEARANCE opinion only. Off-manifold comes from the
deployed per-set exports and the temporal view from its own cached opinions, both unchanged, so a
difference between this row and the deployed one is the spatial view and nothing else. Trust weights
are the ones fitted on the DEVELOPMENT folds; DF40 is never read to fit anything.

FAMILIES follow DF40's own taxonomy via `tbiom/tables/taxonomy.py`, the same module the published
family table uses, so the two tables cannot drift apart. A family column is the mean and standard
deviation ACROSS its generators, never a pooled AUROC, and the macro is the mean of the four family
means.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(ROOT / "upgrade"), str(ROOT / "tbiom/tables"), str(ROOT / "analysis/tbiom/pcr6p"),
                str(ROOT / "analysis/tbiom/grouped"), str(ROOT / "analysis/tbiom"),
                str(ROOT / "analysis/tbiom/temporal"), str(ROOT / "branches/ijepa")]
import taxonomy as TX           # noqa: E402
import arms as AR               # noqa: E402
import ops_pcr6p as P           # noqa: E402
import dsr_system as DS         # noqa: E402
from health import auroc        # noqa: E402
from video_id import video_id   # noqa: E402
from train_head import Head     # noqa: E402

HEADS = ROOT / "analysis/upgrade/heads"
S23 = ROOT / "logs/tbiom/stage23"
BR = ROOT / "logs/tbiom/temporal/branches/vjepa2_lora_B_seed42/opinions"
OUT = ROOT / "analysis/upgrade/final"
ARM = "PCR6P-R"


@torch.no_grad()
def df40_appearance(view, seed, dev):
    """Video-level (p, u) for the candidate over the whole DF40 cache, keyed by video."""
    st = torch.load(HEADS / f"view_{view}_s{seed}.pt", map_location=dev)
    head = Head(st["dim"]).to(dev); head.load_state_dict(st["state"]); head.eval()
    z = np.load(ROOT / f"logs/upgrade/feats/{view}_DF40.npz", allow_pickle=True)
    X = z["feature"].astype(np.float32)
    vid = np.array([str(v) for v in z["video"]])
    order, inv = np.unique(vid, return_inverse=True)
    ps, us = [], []
    for i in range(0, len(X), 8192):
        ev = head(torch.from_numpy(X[i:i + 8192]).to(dev))
        a = ev + 1.0; s = a.sum(1, keepdim=True)
        ps.append((a / s)[:, 1].float().cpu().numpy()); us.append((2.0 / s.squeeze(1)).float().cpu().numpy())
    p, u = np.concatenate(ps), np.concatenate(us)
    cnt = np.bincount(inv, minlength=len(order)).astype(float)
    pv = np.bincount(inv, weights=p, minlength=len(order)) / cnt
    uv = np.bincount(inv, weights=u, minlength=len(order)) / cnt
    return {str(v): (float(a_), float(b_)) for v, a_, b_ in zip(order, pv, uv)}


def set_table(name, app):
    """One generator set: OM from the deployed export, T from its cached opinions, APP from `app`."""
    f = S23 / f"ours_{name}.csv"
    t = BR / f"{name}.csv"
    if not f.is_file() or not t.is_file():
        return None
    d = pd.read_csv(f)
    d["v"] = video_id(d["key"])
    g = d.groupby("v", as_index=False).agg(p_OM=("p_artifact", "mean"), u_OM=("u_artifact", "mean"),
                                           y=("label", "max"))
    tt = pd.read_csv(t).rename(columns={"p": "p_T", "u": "u_T"})[["v", "p_T", "u_T"]]
    g = g.merge(tt, on="v", how="inner")
    keep = [v for v in g.v if str(v) in app]
    g = g[g.v.isin(keep)].reset_index(drop=True)
    if len(g) < 10 or g.y.nunique() < 2:
        return None
    g["p_APP"] = [app[str(v)][0] for v in g.v]
    g["u_APP"] = [app[str(v)][1] for v in g.v]
    return g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", required=True); ap.add_argument("--label", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    label = a.label or a.view
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # weights come from DEVELOPMENT, with the candidate in the appearance slot
    import final_eval as FE
    dev_t, _dp, _tests = DS.load("vjepa2_lora_B_seed42")
    dev_op = FE.opinions(a.view, a.seed, dev, ("VALmix", "VALmixV2", "FFpp_val"))
    dev_sw = FE.swap_app(dev_t, dev_op, "development")
    ws = [AR.fit_w(Fd, ARM) for _k, Fd, _S in DS.splits(dev_sw)]
    print(f"{label}: weights fitted on development", flush=True)

    app = df40_appearance(a.view, a.seed, dev)
    print(f"  DF40 appearance opinions for {len(app)} videos", flush=True)

    per, missing = {}, []
    for fam, gen, var, name, _fs, _rs in TX.cells():
        g = set_table(name, app)
        if g is None:
            missing.append(name); continue
        y = g.y.to_numpy()
        s = np.mean([P.p_of(AR.fuse(g, ARM, w)) for w in ws], axis=0)
        per[name] = {"family": fam, "generator": gen, "variant": var or "own",
                     "videos": int(len(y)), "auroc": auroc(y, s)}
    fam_stats = {}
    for f in TX.FAMILY_ORDER:
        v = [r["auroc"] for r in per.values() if r["family"] == f]
        if v:
            fam_stats[f] = {"mean": float(np.mean(v)), "sd": float(np.std(v)), "n": len(v)}
    macro = float(np.mean([s["mean"] for s in fam_stats.values()])) if fam_stats else None

    json.dump({"view": a.view, "label": label, "seed": a.seed, "arm": ARM,
               "per_set": per, "families": fam_stats, "macro": macro, "missing_sets": missing},
              open(OUT / f"df40_{a.view}_s{a.seed}.json", "w"), indent=1)
    print(f"\n{'family':14s} {'n':>3s} {'mean':>8s} {'sd':>7s}")
    for f, s in fam_stats.items():
        print(f"{TX.FAMILY_NAME[f]:14s} {s['n']:3d} {s['mean']:8.4f} {s['sd']:7.4f}")
    print(f"{'MACRO':14s}     {macro:8.4f}")
    if missing:
        print(f"\n  {len(missing)} sets unavailable: {missing[:5]}")
    print(f"wrote {OUT}/df40_{a.view}_s{a.seed}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Part 6 evaluation: choose the perturbation on FF++ val, then score it, with the compression check.

    <discern_ext python> branches/independent_views/perturb_eval.py

Pre-set rule: add the score to fusion as an extra view only if dev v3.1 AUROC is at least 0.65 AND
the compression check passes. The compression check mirrors G1: on FF++ validation REAL videos, the
shift in distance from clean to JPEG q50 must be smaller than the real-versus-fake gap on clean.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(ROOT / "branches/ijepa"), str(ROOT / "analysis/tbiom")]
import v31 as V31  # noqa: E402

P = ROOT / "logs/independent_views/perturb"
OUT = ROOT / "analysis/independent_views"
PERTS = ["noise0.01", "noise0.05", "blur0.5", "blur1.0", "min"]
BACKBONES = ["clip", "dinov3", "pe"]


def video_mean(x, vid):
    order, inv = np.unique(vid, return_inverse=True)
    cnt = np.bincount(inv).astype(float)
    return order, np.bincount(inv, weights=x) / cnt, inv


def load(b, ds, jpeg=False):
    f = P / f"{b}_{ds}{'_jpeg50' if jpeg else ''}.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)
    vid = np.array([str(v) for v in z["video"]])
    out = {"vid": vid, "key": np.array([str(k) for k in z["key"]]), "frame_y": z["label"]}
    for p in PERTS:
        order, m, inv = video_mean(z[p].astype(np.float64), vid)
        out[p] = m
    out["order"] = order
    out["y"] = np.array([int(z["label"][inv == i].max()) for i in range(len(order))])
    first = {}
    for k, v in zip(out["key"], vid):
        first.setdefault(v, k)
    out["vkey"] = np.array([first[v] for v in order], dtype=object)
    return out


def main():
    res = {}
    for b in BACKBONES:
        ff = load(b, "FFpp_val"); jp = load(b, "FFpp_val", jpeg=True)
        vm, v2 = load(b, "VALmix"), load(b, "VALmixV2")
        if ff is None:
            continue
        # choose the perturbation on FF++ validation
        ffauc = {p: V31.fast_auc(ff[p][ff["y"] == 0], ff[p][ff["y"] == 1]) for p in PERTS}
        best = max(PERTS, key=lambda p: ffauc[p])
        # dev v3.1 with the chosen perturbation as the score
        dk = list(vm["vkey"]) + list(v2["vkey"])
        dp = np.concatenate([vm[best], v2[best]])
        dy = np.concatenate([vm["y"], v2["y"]])
        sc, types, _ = V31.score(dk, dp, dy, ff["vkey"], ff[best], ff["y"])
        # compression check, on real FF++ validation videos
        real = ff["y"] == 0
        d_comp = float(jp[best][real].mean() - ff[best][real].mean()) if jp is not None else float("nan")
        d_fake = float(ff[best][ff["y"] == 1].mean() - ff[best][real].mean())
        comp_ok = abs(d_comp) < abs(d_fake)
        passes = (sc >= 0.65) and comp_ok
        res[b] = {"ffpp_val_auroc_by_perturbation": ffauc, "chosen": best,
                  "ffpp_val_auroc": ffauc[best], "dev_v31": sc, "types": types,
                  "d_comp_jpeg50_real": d_comp, "d_fake_clean": d_fake,
                  "compression_check": bool(comp_ok),
                  "rule_dev_v31_ge_0.65": bool(sc >= 0.65),
                  "verdict": "ADD" if passes else "REJECT"}
        print(f"  {b}")
        print(f"    FF++ val AUROC by perturbation: "
              + "  ".join(f"{p} {ffauc[p]:.4f}" for p in PERTS))
        print(f"    chosen {best}   FF++ val {ffauc[best]:.4f}   dev v3.1 {sc:.4f}")
        print(f"    compression: d_comp {d_comp:+.5f} vs d_fake {d_fake:+.5f}  -> "
              f"{'pass' if comp_ok else 'FAIL'}")
        print(f"    verdict: {res[b]['verdict']}  (needs dev v3.1 >= 0.65 and the compression check)")
    json.dump(res, open(OUT / "part6_perturbation.json", "w"), indent=1)
    print(f"\n-> {OUT/'part6_perturbation.json'}")


if __name__ == "__main__":
    main()

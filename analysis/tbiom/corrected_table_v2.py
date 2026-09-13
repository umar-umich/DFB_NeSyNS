#!/usr/bin/env python3
"""Corrected comparison under evaluation protocol v2 — one table for R0-historical, R0-v2, A/B/C-v2.

    python analysis/tbiom/corrected_table_v2.py --out tbiom/reports/CORRECTED_V2.md

Every run is scored the same way: video-level DS replay; tau = EER on FF++ val for that run's own
selected checkpoint (never carried over). Deployable numbers use tau. Matched-FPR numbers are
retrospective (thresholds read from the target ROC) and are labelled so.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from health import auroc, eer  # noqa: E402
import symbolic_overlay as SO  # noqa: E402
from video_id import video_id  # noqa: E402

S23, S2 = Path("logs/tbiom/stage23"), Path("logs/tbiom/stage2")
OOD = ["CDFv2", "CDFv3", "DFD", "DFDC", "DFDCP", "DFEval24"]
PROC_SETS, CONDS = ["CDFv2", "DFDCP"], [("base", "clean"), ("jpeg40", "jpeg q40"), ("rs06", "resize 0.60")]
BR = SO.BRANCHES
SHA_CACHE = S23 / "checkpoint_sha256.json"
R0_HIST_CKPT = ("/data/umar/Repos/DISCERN_Ext/runs/discern_ext_train/run1_auxedl_seed42/checkpoints/"
                "discern_ext-best-epoch=05-val_auroc_video=0.9955.ckpt")


def videos(path: Path):
    if not path.is_file():
        return None
    d = pd.read_csv(path)
    need = [f"{k}_{b}" for b in BR for k in ("p", "u")]
    if any(c not in d for c in need):
        return None
    d["v"] = video_id(d["key"])
    g = d.groupby("v", as_index=False).agg(**{c: (c, "mean") for c in need}, y=("label", "max"))
    g["s"] = SO.fuse(g)
    return g


def rates(g, tau):
    y, s = g["y"].to_numpy(), g["s"].to_numpy()
    return float((s[y == 0] >= tau).mean()), float((s[y == 1] >= tau).mean())


def tpr_at_fpr(g, fpr):
    y, s = g["y"].to_numpy(), g["s"].to_numpy()
    thr = np.quantile(s[y == 0], 1 - fpr)
    return float((s[y == 1] > thr).mean())


def sha256(path):
    cache = json.loads(SHA_CACHE.read_text()) if SHA_CACHE.is_file() else {}
    if path not in cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""):
                h.update(b)
        cache[path] = h.hexdigest(); SHA_CACHE.write_text(json.dumps(cache, indent=2))
    return cache[path]


def runs():
    out = []
    sel_r0 = S23 / "run1auxedl_selection_v2.json"
    score = None
    if sel_r0.is_file():
        c5 = [c for c in json.loads(sel_r0.read_text())["candidates"] if c["epoch"] == 5]
        score = c5[0]["score"] if c5 else None
    out.append({"name": "R0-historical", "epoch": 5, "ckpt": R0_HIST_CKPT, "tag": "run1auxedl",
                "proc_prefix": "", "selection": "FF++ val AUROC (Phase F)", "score": score, "tied": []})
    for arm, name in [("run1auxedl", "R0-v2"), ("sbiarmA", "A-v2"), ("sbiarmB", "B-v2"), ("sbiarmC", "C-v2")]:
        j, t, pp = S23 / f"{arm}_selection_v2.json", S23 / f"{arm}_v2_tag.txt", S23 / f"{arm}_v2_proc_prefix.txt"
        if not (j.is_file() and t.is_file() and pp.is_file()):
            out.append({"name": name, "missing": True}); continue
        s = json.loads(j.read_text())["selected"]
        out.append({"name": name, "epoch": s["epoch"], "ckpt": s["checkpoint"], "tag": t.read_text().strip(),
                    "proc_prefix": pp.read_text().strip(), "selection": "rule v2 (VALmix macro AUROC)",
                    "score": s["score"], "tied": s.get("tied_with", [])})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("tbiom/reports/CORRECTED_V2.md"))
    args = ap.parse_args()
    R, res = runs(), {}
    for r in R:
        if r.get("missing"):
            continue
        ff = videos(S23 / f"{r['tag']}_FFpp_val.csv")
        if ff is None:
            r["missing"] = True; continue
        _, tau = eer(ff["y"].to_numpy(), ff["s"].to_numpy())
        ood = {}
        for ds in OOD:
            g = videos(S23 / f"{r['tag']}_{ds}.csv")
            if g is None:
                continue
            fp, tp = rates(g, tau)
            ood[ds] = {"auroc": float(auroc(g["y"].to_numpy(), g["s"].to_numpy())), "fpr": fp, "tpr": tp,
                       "n_real": int((g.y == 0).sum()), "n_fake": int((g.y == 1).sum())}
        proc = {}
        for ds in PROC_SETS:
            for c, _ in CONDS:
                g = videos(S2 / f"p_{r['proc_prefix']}{ds}_{c}.csv")
                if g is None:
                    continue
                fp, tp = rates(g, tau)
                proc[(ds, c)] = {"fpr": fp, "tpr": tp, "auroc": float(auroc(g["y"].to_numpy(), g["s"].to_numpy())),
                                 "tpr@fpr05": tpr_at_fpr(g, 0.05), "tpr@fpr10": tpr_at_fpr(g, 0.10)}
        res[r["name"]] = {"run": r, "tau": float(tau), "ood": ood, "proc": proc,
                          "sha": sha256(r["ckpt"])[:16]}

    L = ["# Corrected comparison — evaluation protocol v2", "",
         "One seed per run. Video-level DS replay throughout. τ = EER on FF++ val for the run's own selected "
         "checkpoint. **Deployable** = at τ. **Retrospective** = threshold read from the target ROC.", "",
         "Arms B and C were trained before the SBI provenance and per-worker RNG fixes (protocol v2 §5).", ""]
    if not res:
        L.append("**No run is complete yet.** Waiting on selection and exports."); args.out.write_text("\n".join(L) + "\n")
        print("\n".join(L)); return 0

    L += ["## 1. Checkpoints", "", "| run | selection | epoch | sha256[:16] | VALmix macro AUROC | tied with | τ |",
          "|---|---|---:|---|---:|---|---:|"]
    for r in R:
        n = r["name"]
        if n not in res:
            L.append(f"| {n} | *pending* | | | | | |"); continue
        q = res[n]; sc = f"{r['score']:.6f}" if r.get("score") is not None else "—"
        L.append(f"| {n} | {r['selection']} | {r['epoch']} | `{q['sha']}` | {sc} | {', '.join(r['tied']) or '—'} | {q['tau']:.4f} |")

    L += ["", "## 2. Six OOD sets — deployable", "", "| run | mean AUROC | mean FPR_real | mean TPR | " +
          " | ".join(OOD) + " |", "|---|---:|---:|---:|" + "---:|" * len(OOD)]
    for n, q in res.items():
        o = q["ood"]
        if len(o) < len(OOD):
            L.append(f"| {n} | *incomplete ({len(o)}/6)* | | |" + " |" * len(OOD)); continue
        L.append(f"| {n} | **{np.mean([o[d]['auroc'] for d in OOD]):.4f}** | {np.mean([o[d]['fpr'] for d in OOD]):.3f} | "
                 f"{np.mean([o[d]['tpr'] for d in OOD]):.3f} | " + " | ".join(f"{o[d]['auroc']:.4f}" for d in OOD) + " |")
    L += ["", "FPR_real / TPR at τ per set:", "", "| run | " + " | ".join(OOD) + " |", "|---|" + "---|" * len(OOD)]
    for n, q in res.items():
        o = q["ood"]
        L.append(f"| {n} | " + " | ".join(f"{o[d]['fpr']:.3f} / {o[d]['tpr']:.3f}" if d in o else "—" for d in OOD) + " |")

    L += ["", "## 3. Held-out processing (jpeg q40, resize 0.60)", "",
          "| run | set | condition | FPR_real @τ | TPR @τ | ΔFPR | ΔTPR | AUROC | TPR@FPR .05 *(retro)* | TPR@FPR .10 *(retro)* |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for n, q in res.items():
        for ds in PROC_SETS:
            base = q["proc"].get((ds, "base"))
            for c, nice in CONDS:
                p = q["proc"].get((ds, c))
                if p is None:
                    L.append(f"| {n} | {ds} | {nice} | *pending* | | | | | | |"); continue
                dF = "—" if c == "base" or base is None else f"{p['fpr']-base['fpr']:+.3f}"
                dT = "—" if c == "base" or base is None else f"{p['tpr']-base['tpr']:+.3f}"
                L.append(f"| {n} | {ds} | {nice} | {p['fpr']:.3f} | {p['tpr']:.3f} | {dF} | {dT} | {p['auroc']:.4f} | "
                         f"{p['tpr@fpr05']:.3f} | {p['tpr@fpr10']:.3f} |")

    def mean_auc(n):
        o = res.get(n, {}).get("ood", {})
        return np.mean([o[d]["auroc"] for d in OOD]) if len(o) == len(OOD) else None
    L += ["", "## 4. Differences (single seed; |Δ| < 0.01 is not claimable)", "", "| comparison | ΔAUROC |", "|---|---:|"]
    for a, b in [("R0-v2", "R0-historical"), ("A-v2", "R0-historical"), ("A-v2", "R0-v2"), ("B-v2", "A-v2"), ("C-v2", "B-v2")]:
        x, y = mean_auc(a), mean_auc(b)
        if x is not None and y is not None:
            L.append(f"| {a} − {b} | {x-y:+.4f} |")
    args.out.write_text("\n".join(L) + "\n")
    Path(str(args.out).replace(".md", ".json")).write_text(json.dumps(
        {n: {**{k: v for k, v in q.items() if k != "proc"},
             "proc": {f"{k[0]}|{k[1]}": v for k, v in q["proc"].items()}} for n, q in res.items()}, indent=2, default=str))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())

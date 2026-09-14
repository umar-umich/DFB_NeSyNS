#!/usr/bin/env python3
"""Complete per-dataset comparison under protocol v3: R0-historical and the v3-selected R0/A/B/C.

    python analysis/tbiom/corrected_table_v3.py --out tbiom/reports/CORRECTED_V3.md

Video-level DS replay; τ = EER on FF++ val for each run's own checkpoint. Deployable columns use τ; TPR@FPR
columns are retrospective. Every dataset is reported on its own; means are secondary.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from corrected_table_v2 import videos, sha256, R0_HIST_CKPT  # noqa: E402
from health import auroc, eer  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

S23, S2 = Path("logs/tbiom/stage23"), Path("logs/tbiom/stage2")
SETS = ["CDFv2", "CDFv3", "DFD", "DFDC", "DFDCP", "DFEval24", "FFIW", "KoDF", "KoDF-adv", "IDForge", "DFDM"]
FILE = {"KoDF-adv": "KoDF"}
EXPOSURE = {"CDFv2": "selection-exposed (v3)", "CDFv3": "selection-exposed (v3)", "FFIW": "selection-exposed (v3)",
            "DFD": "development-exposed", "DFDC": "development-exposed", "DFDCP": "development-exposed",
            "DFEval24": "development-exposed", "KoDF": "new", "KoDF-adv": "new", "IDForge": "new",
            "DFDM": "new fakes; genuine = CDFv2 genuine pool"}
BREAKDOWN = {"CDFv3": {"face swap": "FaceSwap", "reenactment": "FaceReenact", "talking face": "TalkingFace"},
             "KoDF": None, "IDForge": None, "DFDM": None}
PROC_SETS, CONDS = ["CDFv2", "DFDCP"], [("base", "clean"), ("jpeg40", "jpeg q40"), ("rs06", "resize 0.60")]


def with_subset(tag, ds):
    f = S23 / f"{tag}_{FILE.get(ds, ds)}.csv"
    g = videos(f)
    if g is None: return None
    g["subset"] = g["v"].str.split("/").str[-2]
    if ds == "KoDF": g = g[~g.subset.str.startswith("adv-")]
    if ds == "KoDF-adv": g = g[g.subset.str.startswith("adv-")]
    return g.reset_index(drop=True)


def metrics(g, tau):
    y, s = g.y.to_numpy(), g.s.to_numpy()
    fpr, tpr = float((s[y == 0] >= tau).mean()), float((s[y == 1] >= tau).mean())
    thr = lambda f: np.quantile(s[y == 0], 1 - f)
    return {"n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum()), "auroc": float(auroc(y, s)),
            "ap": float(average_precision_score(y, s)), "eer": float(eer(y, s)[0]), "fpr": fpr, "tpr": tpr,
            "bacc": 0.5 * (tpr + 1 - fpr), "t05": float((s[y == 1] > thr(.05)).mean()), "t10": float((s[y == 1] > thr(.10)).mean())}


def runs():
    out = [{"name": "R0-historical", "epoch": 5, "ckpt": R0_HIST_CKPT, "tag": "run1auxedl", "pp": "", "score": None}]
    for arm, name in [("run1auxedl", "R0-v3"), ("sbiarmA", "A-v3"), ("sbiarmB", "B-v3"), ("sbiarmC", "C-v3")]:
        j, t, p = (S23 / f"{arm}_selection_v3.json", S23 / f"{arm}_v3_tag.txt", S23 / f"{arm}_v3_proc_prefix.txt")
        if not (j.is_file() and t.is_file() and p.is_file()):
            out.append({"name": name, "missing": True}); continue
        s = json.loads(j.read_text())["selected"]
        out.append({"name": name, "epoch": s["epoch"], "ckpt": s["checkpoint"], "tag": t.read_text().strip(),
                    "pp": p.read_text().strip(), "score": s["score"], "types": s["types"], "tied": s.get("tied_with", [])})
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=Path("tbiom/reports/CORRECTED_V3.md"))
    args = ap.parse_args(); R = [r for r in runs() if not r.get("missing")]; res = {}
    for r in R:
        ff = videos(S23 / f"{r['tag']}_FFpp_val.csv")
        if ff is None: continue
        tau = float(eer(ff.y.to_numpy(), ff.s.to_numpy())[1]); ds_m, sub_m, proc = {}, {}, {}
        for ds in SETS:
            g = with_subset(r["tag"], ds)
            if g is None or g.y.nunique() < 2: continue
            ds_m[ds] = metrics(g, tau)
            if ds in BREAKDOWN:
                real = g[g.y == 0]
                groups = ({k: g[(g.y == 1) & g.subset.str.contains(v)] for k, v in BREAKDOWN[ds].items()}
                          if BREAKDOWN[ds] else {s: g[(g.y == 1) & (g.subset == s)] for s in sorted(g[g.y == 1].subset.unique())})
                sub_m[ds] = {k: {"n": int(len(f)), "auroc": float(auroc(pd.concat([real, f]).y.to_numpy(), pd.concat([real, f]).s.to_numpy()))}
                             for k, f in groups.items() if len(f)}
        for ds in PROC_SETS:
            for c, _ in CONDS:
                g = videos(S2 / f"p_{r['pp']}{ds}_{c}.csv")
                if g is not None: proc[(ds, c)] = metrics(g, tau)
        res[r["name"]] = {"run": r, "tau": tau, "sets": ds_m, "subsets": sub_m, "proc": proc, "sha": sha256(r["ckpt"])[:16]}

    L = ["# Complete OOD comparison — protocol v3", "",
         "One seed per run. Video-level DS replay. τ = EER on FF++ val for each run's own checkpoint. Deployable = at τ; "
         "TPR@FPR = retrospective. Arms B and C predate the SBI provenance / worker-RNG fixes.", "",
         "## 1. Checkpoints", "", "| run | epoch | sha256[:16] | v3 score | swap | reenact | talking | synthesis | in-the-wild | τ |",
         "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for n, q in res.items():
        r = q["run"]; t = r.get("types") or {}
        f = lambda k: f"{t[k]:.4f}" if k in t else "—"
        L.append(f"| {n} | {r['epoch']} | `{q['sha']}` | {r['score']:.4f} | {f('swap')} | {f('reenactment')} | {f('talking_face')} | "
                 f"{f('entire_face_synthesis')} | {f('in_the_wild')} | {q['tau']:.4f} |" if r["score"] is not None else
                 f"| {n} | {r['epoch']} | `{q['sha']}` | — | — | — | — | — | — | {q['tau']:.4f} |")
    for ds in SETS:
        L += ["", f"## {ds} — {EXPOSURE[ds]}", "",
              "| run | real / fake | AUROC | AP | EER | FPR_real @τ | TPR @τ | bal. acc @τ | TPR@FPR .05 *(retro)* | TPR@FPR .10 *(retro)* |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for n, q in res.items():
            m = q["sets"].get(ds)
            L.append(f"| {n} | *pending* | | | | | | | | |" if m is None else
                     f"| {n} | {m['n_real']} / {m['n_fake']} | {m['auroc']:.4f} | {m['ap']:.4f} | {m['eer']:.4f} | {m['fpr']:.3f} | "
                     f"{m['tpr']:.3f} | {m['bacc']:.3f} | {m['t05']:.3f} | {m['t10']:.3f} |")
        if ds in BREAKDOWN:
            keys = sorted({k for q in res.values() for k in q["subsets"].get(ds, {})})
            if keys:
                L += ["", f"{ds} AUROC by subset (fakes of the subset vs the dataset's genuine videos):", "",
                      "| run | " + " | ".join(keys) + " |", "|---|" + "---:|" * len(keys)]
                for n, q in res.items():
                    sm = q["subsets"].get(ds, {})
                    L.append(f"| {n} | " + " | ".join(f"{sm[k]['auroc']:.4f}" if k in sm else "—" for k in keys) + " |")
    L += ["", "## Held-out processing (jpeg q40, resize 0.60)", "",
          "| run | set | condition | FPR_real @τ | TPR @τ | ΔFPR | ΔTPR | AUROC | TPR@FPR .05 *(retro)* | TPR@FPR .10 *(retro)* |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for n, q in res.items():
        for ds in PROC_SETS:
            b = q["proc"].get((ds, "base"))
            for c, nice in CONDS:
                p = q["proc"].get((ds, c))
                if p is None: L.append(f"| {n} | {ds} | {nice} | *pending* | | | | | | |"); continue
                d = (lambda k: "—" if c == "base" or b is None else f"{p[k]-b[k]:+.3f}")
                L.append(f"| {n} | {ds} | {nice} | {p['fpr']:.3f} | {p['tpr']:.3f} | {d('fpr')} | {d('tpr')} | {p['auroc']:.4f} | {p['t05']:.3f} | {p['t10']:.3f} |")
    L += ["", "## Cross-run AUROC by dataset", "", "| dataset | exposure | " + " | ".join(res) + " |", "|---|---|" + "---:|" * len(res)]
    for ds in SETS:
        L.append(f"| {ds} | {EXPOSURE[ds]} | " + " | ".join(f"{q['sets'][ds]['auroc']:.4f}" if ds in q["sets"] else "—" for q in res.values()) + " |")
    args.out.write_text("\n".join(L) + "\n")
    Path(str(args.out).replace(".md", ".json")).write_text(json.dumps(
        {n: {"tau": q["tau"], "sets": q["sets"], "subsets": q["subsets"], "proc": {f"{k[0]}|{k[1]}": v for k, v in q["proc"].items()}}
         for n, q in res.items()}, indent=2))
    print("\n".join(L)); return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""The comparative methods on DF40's INDIVIDUAL generators, sliced from the run already done.

    <discern_ext python> upgrade/branchfusion/df40_baselines.py

The seven GenD-repo baselines were scored on DF40 as ONE pooled dataset, so the per-generator
cross-manipulation table had our framework and DiCoME in it and nothing else. No new inference is
needed: that pooled run holds all 80 subsets, every generator and every real pool, keyed by the
same frame paths as everything else here. Slicing it per generator costs no GPU at all.

THE PAIRING IS THE SAME ONE OUR ROWS USE. A generator is scored against the real pool DF40 ships
for ITS source -- `DF40-cdf-real` for cdf, `DF40-ff-real` for ff, and DeepFaceLab against its own
-- which is what `adapt_backbone.df40_manifests` composes and what the 2026-09-22 campaign used.
Scoring a generator against the wrong source's genuine videos would change the number for a
reason that has nothing to do with the detector.

Video level by the project's rule: mean probability over the frames of one video directory. The
baselines emit no uncertainty beyond `1 - max p`, so only AUROC is reported here.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                               # noqa: E402
import df40_crossmanip as D                                      # noqa: E402

C.thread_caps()

BASE = C.ROOT / "logs/tbiom/baselines/gend_runs_ours"
MODELS = ["GenD_CLIP", "GenD_PE", "GenD_DINO", "FSFM", "FSFM_Adapter", "Effort", "ForAda"]
PRETTY = {"GenD_CLIP": "GenD-CLIP", "GenD_PE": "GenD-PE", "GenD_DINO": "GenD-DINO",
          "FSFM": "FSFM", "FSFM_Adapter": "FSFM-Adapter", "Effort": "Effort", "ForAda": "ForAda"}
REAL = {"cdf": "DF40-cdf-real", "ff": "DF40-ff-real"}


def pooled(model):
    hits = glob.glob(f"{BASE}/{model}--DF40/**/test_predictions.csv", recursive=True)
    if not hits:
        return None
    d = pd.read_csv(hits[0])
    parts = d.files.str.split("/")
    d = d.assign(subset=parts.str[-3], v=d.files.str.rsplit("/", n=1).str[0])
    return d


def video_auroc(d, fake_subset, real_subset):
    sel = d[d.subset.isin([fake_subset, real_subset])]
    if sel.empty:
        return None, 0, 0
    g = sel.groupby("v").agg(y=("labels", "max"), s=("prob_class_1", "mean"))
    y, s = g.y.to_numpy().astype(int), g.s.to_numpy()
    if len(np.unique(y)) < 2:
        return None, int((y == 0).sum()), int((y == 1).sum())
    return C.auroc(y, s), int((y == 0).sum()), int((y == 1).sum())


def main() -> int:
    meta = json.loads((C.ROOT / "logs/tbiom/df40/meta.json").read_text())
    fam = {k: v["family"] for k, v in meta.items()}
    # dataset name -> (fake subset, real subset), the same composition our rows use
    jobs = {}
    for name, m in meta.items():
        g, src = m["method"], m["source"]
        if name.endswith("_cdf") or name.endswith("_ff"):
            h = name.rsplit("_", 1)[1]
            jobs[name] = (f"{g}-{h}", REAL[h])
        else:
            jobs[name] = (g, f"{g}-real")

    res = {}
    for model in MODELS:
        d = pooled(model)
        if d is None:
            print(f"  {model}: no pooled DF40 run", flush=True)
            continue
        subs = set(d.subset.unique())
        row = {}
        for name, (fk, rl) in jobs.items():
            if fk not in subs or rl not in subs:
                continue
            a, nr, nf = video_auroc(d, fk, rl)
            if a is not None:
                row[name] = {"auroc": a, "real": nr, "fake": nf}
        res[PRETTY[model]] = row
        fs = [v["auroc"] for k, v in row.items() if fam.get(k) == "face swap"]
        fr = [v["auroc"] for k, v in row.items() if fam.get(k) == "reenactment"]
        print(f"  {PRETTY[model]:14s} {len(row)} sets   "
              f"face swap {np.mean(fs):.4f}   reenactment {np.mean(fr):.4f}   "
              f"all {np.mean([v['auroc'] for v in row.values()]):.4f}", flush=True)

    (C.OUT / "df40_baselines.json").write_text(json.dumps(res, indent=1))

    # Our rows and DiCoME's, from the table that owns them.
    ours = json.loads((C.OUT / "DF40_CROSSMANIP_ALL.json").read_text())
    names = sorted(set().union(*[set(r) for r in res.values()]) & set(ours))
    L = ["# DF40 cross-manipulation — every method, every generator", "",
         "The comparative methods sliced per generator from the DF40 run already on disk; no new",
         "inference was needed. Each generator is scored against the real pool DF40 ships for its",
         "SOURCE, which is the pairing our own rows use.", "",
         "AUROC only: a cross-entropy baseline emits no uncertainty beyond `1 - max p`, so the",
         "calibration columns of Table 2 have no counterpart here.", ""]
    # THE SUMMARY FIRST. The per-generator tables below are 19 and 22 columns wide, which is an
    # appendix object; the comparison a reader needs is the family means on one screen.
    import itertools
    variants = [("Ours (App+TD+LFS+STT)", "A+V+O+S16ctl"), ("Ours (App+TD+LFS)", "A+V+O")]
    vc = C.OUT / "variant_compare.json"
    extra = json.loads(vc.read_text()) if vc.exists() else {}
    summary = []
    for lbl, key in variants:
        got = {k: v.get(key) for k, v in ours.items() if v.get(key) is not None}
        if got:
            summary.append((lbl, got))
    for k, r in extra.items():                       # the App-less candidates, scored elsewhere
        fa = r.get("df40_family", {})
        if fa:
            summary.append((f"Ours ({r['name']})", None, fa, r.get("df40_all")))
    dic = {k: v.get("dicome_our_training") for k, v in ours.items()
           if v.get("dicome_our_training") is not None}
    if dic:
        summary.append(("DiCoME (our training)", dic))
    for m, r in res.items():
        summary.append((m, {k: v["auroc"] for k, v in r.items()}))

    table = []
    for item in summary:
        if len(item) == 4:
            lbl, _none, fa, allm = item
            table.append((lbl, fa.get("face swap", {}).get("auroc"),
                          fa.get("reenactment", {}).get("auroc"), allm,
                          sum(v.get("sets", 0) for v in fa.values())))
            continue
        lbl, got = item
        fs = [v for k, v in got.items() if fam.get(k) == "face swap"]
        fr = [v for k, v in got.items() if fam.get(k) == "reenactment"]
        table.append((lbl, float(np.mean(fs)) if fs else None,
                      float(np.mean(fr)) if fr else None,
                      float(np.mean(list(got.values()))), len(got)))
    table.sort(key=lambda r: -(r[3] or 0))
    L += ["## Summary — family means over DF40's 41 generator sets", "",
          "| method | face swap | reenactment | all 41 | sets |", "|---|---:|---:|---:|---:|"]
    for lbl, a, b, c, n in table:
        star = "**" if lbl.startswith("Ours") else ""
        L.append(f"| {star}{lbl}{star} | " + " | ".join(
            f"{x:.4f}" if isinstance(x, float) else "—" for x in (a, b, c)) + f" | {n} |")
    L += ["",
          "Reenactment is where the gap opens. DiCoME's published cross-manipulation table does",
          "not report it at all, and it is half of DF40.", ""]

    for f in ("face swap", "reenactment"):
        ks = [n for n in names if fam.get(n) == f]
        if not ks:
            continue
        L += [f"## {f.title()} — {len(ks)} generator sets", "",
              "| method | " + " | ".join(k.replace("DF40_", "") for k in ks) + " | mean |",
              "|---|" + "---:|" * (len(ks) + 1)]
        for lbl, key in (("**Ours (TD+LFS+STT)**", "V+O+S16ctl"),
                         ("**Ours (App+TD+LFS+STT)**", "A+V+O+S16ctl"),
                         ("**Ours (App+TD+LFS)**", "A+V+O")):
            if key not in next(iter(ours.values())):
                continue
            vals = [ours[k].get(key) for k in ks]
            L.append(f"| {lbl} | " + " | ".join(f"{v:.4f}" if v else "—" for v in vals)
                     + f" | **{np.mean([v for v in vals if v]):.4f}** |")
        dvals = [ours[k].get("dicome_our_training") for k in ks]
        if any(dvals):
            L.append("| DiCoME (our training) | "
                     + " | ".join(f"{v:.4f}" if v else "—" for v in dvals)
                     + f" | {np.mean([v for v in dvals if v]):.4f} |")
        for m, row in res.items():
            vals = [row.get(k, {}).get("auroc") for k in ks]
            got = [v for v in vals if v is not None]
            L.append(f"| {m} | " + " | ".join(f"{v:.4f}" if v is not None else "—" for v in vals)
                     + (f" | {np.mean(got):.4f} |" if got else " | — |"))
        L.append("")
    (C.REPORT / "DF40_ALL_METHODS.md").write_text("\n".join(L) + "\n")
    print(f"-> {C.REPORT / 'DF40_ALL_METHODS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

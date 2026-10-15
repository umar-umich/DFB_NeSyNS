#!/usr/bin/env python3
"""THE DF40 table. Our method variants against every comparative method, on every generator.

    <discern_ext python> upgrade/branchfusion/df40_table.py
    <discern_ext python> upgrade/branchfusion/df40_table.py --ours V+O+S16ctl,V+O+P16ctl

This replaces three overlapping files -- DF40_CROSSMANIP (eight generators), DF40_CROSSMANIP_ALL
(twenty-one, our rows only) and DF40_ALL_METHODS (all methods, but our rows were the wrong
variants). Three tables covering the same corpus with different row sets and different column
sets is not thoroughness; a reader cannot tell which one to quote.

ONE FILE, four sections: the summary every method fits on, face swap per generator, reenactment
per generator, and the head-to-head against DiCoME's published cross-manipulation numbers.

Every method is scored on the same videos: each generator against the real pool DF40 ships for
ITS source, aggregated to video level by the project's rule.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                               # noqa: E402
import branches as B                                             # noqa: E402
import opinions as O                                             # noqa: E402
import df40_crossmanip as D                                      # noqa: E402

C.thread_caps()

#: DiCoME Table 1, cross-manipulation half, transcribed from the paper. PUBLISHED, not measured
#: here, and on a census we cannot inspect.
PUBLISHED = D.PUBLISHED
DICOME_EIGHT = D.DICOME_EIGHT
PRETTY = D.PRETTY


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="V+O+S16ctl,V+O+P16ctl",
                    help="comma list of our candidate branch sets")
    ap.add_argument("--out", default="DF40_RESULTS")
    a = ap.parse_args()
    import ops_pcr6p as P

    cands = [tuple(x.split("+")) for x in a.ours.split(",") if x]
    need = sorted(set().union(*cands))
    meta = json.loads((C.ROOT / "logs/tbiom/df40/meta.json").read_text())
    fam = {k: v["family"] for k, v in meta.items()}
    gens = sorted({v["method"] for v in meta.values()})
    sets, _ = D.gens_available(gens)
    print(f"{len(sets)} generator sets, {len(cands)} candidates", flush=True)

    dev = B.dev_table(need, require_all=True)
    tabs = {ds: B.table([ds], need) for ds in sets}
    rows = {}                                   # label -> {set: auroc}
    census = {}
    for g in cands:
        w, _ = O.fit_weights(dev, g, "F8", objective="v31")
        lbl = f"**Ours ({C.show_subset(g)})**"
        rows[lbl] = {}
        for ds in sets:
            t = tabs[ds]
            y = t.y.to_numpy()
            rows[lbl][ds] = C.auroc(y, P.p_of(P.fuse(O.masses(t, g, list(w)), True)))
            census[ds] = (int((y == 0).sum()), int((y == 1).sum()))
        print(f"  {lbl}: {np.mean(list(rows[lbl].values())):.4f}", flush=True)

    # Comparative methods, each from the module that owns its computation.
    prior = json.loads((C.OUT / "DF40_CROSSMANIP_ALL.json").read_text())
    dic = {k: v.get("dicome_our_training") for k, v in prior.items()
           if v.get("dicome_our_training") is not None}
    if dic:
        rows["DiCoME (our training)"] = dic
    repo = C.ROOT / "logs/tbiom/df40/dicome_repo_df40.json"
    if repo.exists():
        r = json.loads(repo.read_text())
        got = {}
        for ds in sets:
            n = ds[len("DF40_"):]
            key = (n.rsplit("_", 1)[0] + "-" + n.rsplit("_", 1)[1]) if "_" in n else n
            if key in r:
                got[ds] = r[key]["auc_video"]
        if got:
            rows["DiCoME (their repo)"] = got
            print(f"  DiCoME (their repo): {len(got)} of {len(sets)} rows", flush=True)
    base = json.loads((C.OUT / "df40_baselines.json").read_text())
    for m, r in base.items():
        rows[m] = {k: v["auroc"] for k, v in r.items()}

    def fam_mean(d, f):
        v = [x for k, x in d.items() if fam.get(k) == f]
        return float(np.mean(v)) if v else None

    def cell(x):
        return f"{x:.4f}" if isinstance(x, float) else "—"

    order = sorted(rows, key=lambda k: -float(np.mean(list(rows[k].values()))))
    L = [f"# DF40 — every method on every generator", "",
         f"{len(sets)} generator sets, {len(rows)} methods. Each generator is scored against the",
         "real pool DF40 ships for ITS source, aggregated to video level by the project's rule,",
         "so every row is on the same videos. AUROC.", "",
         "This is the only DF40 table. It replaces `DF40_CROSSMANIP`, `DF40_CROSSMANIP_ALL` and",
         "`DF40_ALL_METHODS`, which covered the same corpus with different rows and columns.", "",
         "## Summary — family means", "",
         "| method | face swap | reenactment | all sets | covered |", "|---|---:|---:|---:|---:|"]
    for k in order:
        d = rows[k]
        L.append(f"| {k} | {cell(fam_mean(d, 'face swap'))} | {cell(fam_mean(d, 'reenactment'))} "
                 f"| **{np.mean(list(d.values())):.4f}** | {len(d)} |")
    L += ["",
          "Reenactment is half of DF40 and the half DiCoME's published cross-manipulation table",
          "does not report.", ""]

    for f in ("face swap", "reenactment"):
        ks = [s for s in sets if fam.get(s) == f]
        if not ks:
            continue
        ks.sort()
        L += [f"## {f.title()} — {len(ks)} generator sets", "",
              "| method | " + " | ".join(k[len("DF40_"):] for k in ks) + " | mean |",
              "|---|" + "---:|" * (len(ks) + 1)]
        L.append("| *videos (real / fake)* | "
                 + " | ".join(f"{census[k][0]}/{census[k][1]}" for k in ks) + " | |")
        for k in order:
            d = rows[k]
            vals = [d.get(s) for s in ks]
            got = [v for v in vals if isinstance(v, float)]
            L.append(f"| {k} | " + " | ".join(cell(v) for v in vals)
                     + (f" | **{np.mean(got):.4f}** |" if got else " | — |"))
        L.append("")

    # The published head-to-head, in DiCoME's own column order.
    eight = [g for g in DICOME_EIGHT
             if any(f"DF40_{g}_{h}" in sets for h in ("cdf", "ff"))]
    if eight:
        L += ["## Against DiCoME's published cross-manipulation table", "",
              "Column order is DiCoME's. Our rows pool the two source halves, since the paper",
              "reports one number per generator without saying which half it used. The published",
              "rows are transcribed from the paper and were not re-measured.", "",
              "| method | " + " | ".join(PRETTY[g] for g in eight) + " | Avg. |",
              "|---|" + "---:|" * (len(eight) + 1)]
        for k in order:
            d = rows[k]
            vals = []
            for g in eight:
                v = [d[f"DF40_{g}_{h}"] for h in ("cdf", "ff") if f"DF40_{g}_{h}" in d]
                vals.append(float(np.mean(v)) if v else None)
            got = [v for v in vals if v is not None]
            L.append(f"| {k} | " + " | ".join(cell(v) for v in vals)
                     + (f" | **{np.mean(got):.4f}** |" if got else " | — |"))
        for lbl, vals in PUBLISHED.items():
            v = [vals[DICOME_EIGHT.index(g)] for g in eight]
            L.append(f"| {lbl} *(published)* | " + " | ".join(f"{x:.3f}" for x in v)
                     + f" | {np.mean(v):.3f} |")
        L.append("")

    (C.OUT / f"{a.out}.json").write_text(json.dumps(rows, indent=1))
    (C.REPORT / f"{a.out}.md").write_text("\n".join(L) + "\n")
    print(f"-> {C.REPORT / f'{a.out}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

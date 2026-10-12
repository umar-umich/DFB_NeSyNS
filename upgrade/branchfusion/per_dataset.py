#!/usr/bin/env python3
"""Every branch subset scored on EACH test set separately, not just by the ten-set macro.

    OMP_NUM_THREADS=4 <discern_ext python> upgrade/branchfusion/per_dataset.py --all

WHY THIS EXISTS. The locked test reports one macro per subset, and a macro hides the thing that
decides a table: a subset can win overall while losing badly on two sets, and two subsets 0.005 apart
on the macro can disagree by 0.05 on an individual set. Choosing which sets go in a main table, or
which subset to deploy, needs the grid.

WHAT IT IS NOT. `subsets.evaluate_test` scores a LOCKED subset once, and the lock is what makes its
number an honest held-out estimate. This module sweeps EVERY subset over every test set, which is a
full exploratory pass over the test partitions. The grid it produces is a DIAGNOSTIC: it is for
reading structure and for choosing table scope, never for choosing the deployed subset and never to
be reported as a held-out result. The header of every artifact it writes says so, because a wide
grid of test numbers is exactly the kind of object that gets quoted later without its caveat.

The weights and the decision threshold follow the locked convention exactly -- weights fitted on the
whole development pool, threshold from the FF++ validation rows inside development, never from a test
set -- so each row is comparable with the locked rows for the same subset.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                      # noqa: E402
import branches as B                                    # noqa: E402
import opinions as O                                    # noqa: E402
import subsets as S                                     # noqa: E402

for _m in (B, O, S):
    C.assert_own_module(_m)

C.thread_caps()

HEADER = (
    "EXPLORATORY DIAGNOSTIC -- NOT A HELD-OUT RESULT. Every subset here was scored on every test "
    "set, so the best cell in this grid is a maximum over a large search on the test partitions and "
    "is biased upward. Use it to read structure and to choose table scope. The held-out numbers are "
    "the locked rows in test_eval.json, and the deployed subset must be chosen on development."
)


def dataset_census(test_tabs: dict) -> dict:
    """Videos, real/fake balance and prevalence per set -- needed to read any per-set AUROC."""
    out = {}
    for ds, t in test_tabs.items():
        y = t.y.to_numpy()
        out[ds] = {"videos": int(len(y)), "real": int((y == 0).sum()), "fake": int((y == 1).sum()),
                   "prevalence": round(float((y == 1).mean()), 4),
                   "sources": int(len(np.unique(C.groups_of(t)))) if "v" in t.columns else None}
    return out


def oracle_row(test_tabs: dict, items) -> dict:
    """Best SINGLE branch per set, chosen with test labels: the non-deployable reference."""
    per = {}
    for ds, t in test_tabs.items():
        y = t.y.to_numpy()
        pb = {}
        for b in items:
            col = f"p_{b}"
            if col in t.columns and np.isfinite(t[col].to_numpy()).any():
                pb[b] = C.auroc(y, t[col].to_numpy())
        if not pb:
            continue
        best = max(pb, key=pb.get)
        per[ds] = {"auroc": pb[best], "branch": best, "per_branch": pb}
    aus = [per[d]["auroc"] for d in C.TEST_SETS if d in per]
    return {"per_dataset": per, "ten_set_macro": float(np.mean(aus)) if aus else float("nan")}


def sweep(dev, test_tabs, items, sizes=(1, 2, 3, 4, 5, 6), op=S.SCREEN_OP, shard=None,
          grid=None) -> dict:
    """Score every subset on every test set. `shard=(i, n)` takes a strided slice.

    The grid is 2**k - 1 subsets, so a fourteen-branch candidate list is 16383 of them at roughly
    a minute each: a month in one process and a few hours across shards. `subs[i::n]` is used
    rather than a contiguous block because subset COST grows with size, and contiguous blocks would
    hand the last shard every large subset and finish hours after the first.
    """
    subs = C.subsets_of(tuple(items), sizes)
    total = len(subs)
    if shard is not None:
        i, n = shard
        subs = subs[i::n]
        print(f"shard {i + 1}/{n}: {len(subs)} of {total} subsets", flush=True)
    print(f"{len(subs)} subsets x {len(test_tabs)} test sets under {op}", flush=True)
    rows = []
    for i, s in enumerate(subs, 1):
        r = S.evaluate_test(dev, test_tabs, s, op, **({} if grid is None else {"grid": grid}))
        r.pop("locked", None)
        r["exploratory"] = True
        rows.append(r)
        print(f"  [{i:>2}/{len(subs)}] {C.subset_id(s):<14} macro {r['ten_set_macro']:.4f}", flush=True)
    return {"header": HEADER, "operator": op, "items": list(items), "sizes": list(sizes),
            "weight_grid": list(grid) if grid is not None else list(O.WEIGHT_GRID),
            "census": dataset_census(test_tabs), "oracle": oracle_row(test_tabs, items),
            "rows": rows}


def grid(res: dict, metric: str = "auroc") -> pd.DataFrame:
    """subsets x test sets, as a DataFrame ordered by the ten-set macro."""
    sets = [d for d in C.TEST_SETS if d in res["census"]]
    recs = []
    for r in res["rows"]:
        rec = {"subset": r["subset"], "n_branches": r["subset"].count("+") + 1}
        for d in sets:
            rec[d] = r["per_dataset"].get(d, {}).get(metric, float("nan"))
        rec["macro"] = r["ten_set_macro"]
        rec["worst_set"] = min(sets, key=lambda d: rec[d])
        rec["worst"] = min(rec[d] for d in sets)
        rec["spread"] = max(rec[d] for d in sets) - rec["worst"]
        recs.append(rec)
    df = pd.DataFrame(recs).sort_values("macro", ascending=False).reset_index(drop=True)
    o = res["oracle"]["per_dataset"]
    orow = {"subset": "ORACLE (best single per set, uses test labels)", "n_branches": 0,
            **{d: o[d]["auroc"] for d in sets if d in o},
            "macro": res["oracle"]["ten_set_macro"]}
    orow["worst_set"] = min((d for d in sets if d in o), key=lambda d: orow[d])
    orow["worst"] = min(orow[d] for d in sets if d in o)
    orow["spread"] = max(orow[d] for d in sets if d in o) - orow["worst"]
    return pd.concat([df, pd.DataFrame([orow])], ignore_index=True)


def markdown(res: dict, df: pd.DataFrame, top: int = 0) -> str:
    sets = [d for d in C.TEST_SETS if d in res["census"]]
    L = [f"# Per-dataset ablation grid ({res['operator']})", "",
         f"> **{HEADER}**", "",
         "## Dataset census", "",
         "| set | videos | real | fake | prevalence |", "|---|---:|---:|---:|---:|"]
    for d in sets:
        c = res["census"][d]
        L.append(f"| {d} | {c['videos']} | {c['real']} | {c['fake']} | {c['prevalence']:.3f} |")
    L += ["", "## AUROC by subset and test set", "",
          "`worst` is the subset's weakest set, `spread` its best minus its worst -- a subset with a "
          "high macro and a wide spread is carried by a few sets.", "",
          "| subset | " + " | ".join(sets) + " | macro | worst | spread |",
          "|---|" + "---:|" * (len(sets) + 3)]
    d2 = df if not top else pd.concat([df.head(top), df.tail(1)]).drop_duplicates(subset=["subset"])
    for _, r in d2.iterrows():
        cells = " | ".join("--" if not np.isfinite(r[d]) else f"{r[d]:.4f}" for d in sets)
        L.append(f"| {r['subset']} | {cells} | **{r['macro']:.4f}** | "
                 f"{r['worst']:.4f} ({r['worst_set']}) | {r['spread']:.4f} |")
    L += ["", "## Best single branch per set (the oracle's choice)", "",
          "| set | best branch | AUROC | " + " | ".join(res["oracle"]["per_dataset"][sets[0]]["per_branch"]) + " |",
          "|---|---|---:|" + "---:|" * len(res["oracle"]["per_dataset"][sets[0]]["per_branch"])]
    for d in sets:
        o = res["oracle"]["per_dataset"][d]
        pb = " | ".join(f"{o['per_branch'][b]:.4f}" for b in o["per_branch"])
        L.append(f"| {d} | **{o['branch']}** | {o['auroc']:.4f} | {pb} |")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branches", default="A,S,V,O,M,P")
    ap.add_argument("--sizes", default="1,2,3,4,5,6")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--top", type=int, default=0, help="markdown rows to show (0 = every subset)")
    ap.add_argument("--tag", default="", help="suffix for the output files, so a focused grid does "
                                              "not overwrite the full one")
    ap.add_argument("--shard", default="", metavar="i/n",
                    help="compute only every n-th subset starting at i (0-based) and write a "
                         "shard file; the grid and the report are built later by --merge")
    ap.add_argument("--weight-grid", default="locked", choices=["locked", "coarse", "equal"],
                    help="locked = B0's (0,.25,.5,.75,1), the only grid a locked number may use. "
                         "coarse = (0,.5,1). equal = no search at all, every branch at weight 1. "
                         "`fit_weights` is EXHAUSTIVE, so this is what decides whether a subset "
                         "size is reachable: at size 8 locked costs 325,089 fusions, coarse 6,305 "
                         "and equal exactly 1.")
    ap.add_argument("--merge", type=int, default=0, metavar="N",
                    help="read the N shard files for --tag and write the combined grid and report")
    a = ap.parse_args()
    items = [x.strip() for x in a.branches.split(",") if x.strip()]
    sizes = tuple(int(x) for x in a.sizes.split(","))
    sfx = f"_{a.tag}" if a.tag else ""

    if a.merge:
        parts, miss = [], []
        for i in range(a.merge):
            f = C.OUT / f"per_dataset{sfx}_shard{i}.json"
            if not f.exists():
                miss.append(i); continue
            parts.append(json.load(open(f)))
        if miss:
            print(f"shards not finished: {miss}")
            return 1
        res = dict(parts[0])
        seen, rows = set(), []
        for q in parts:                                  # subset id is the key; shards cannot overlap
            for r in q["rows"]:
                k = r.get("subset") or C.subset_id(r.get("items", ()))
                if k not in seen:
                    seen.add(k); rows.append(r)
        res["rows"] = rows
        res.pop("shard", None)
        print(f"merged {len(parts)} shards -> {len(rows)} subsets")
    else:
        dev = B.dev_table(items, require_all=True)
        test_tabs = B.test_tables(items)
        print(f"branches {items} | development {len(dev)} videos | test sets {list(test_tabs)}")
        wgrid = {"locked": O.WEIGHT_GRID, "coarse": (0.0, 0.5, 1.0), "equal": (1.0,)}[a.weight_grid]
        if a.weight_grid != "locked":
            print(f"WEIGHT GRID {a.weight_grid} {wgrid} -- NOT the locked fit; these rows are not "
                  f"comparable with locked numbers and every artifact must say so", flush=True)
        shard = None
        if a.shard:
            i, n = (int(x) for x in a.shard.split("/"))
            shard = (i, n)
        res = sweep(dev, test_tabs, items, sizes, shard=shard, grid=wgrid)
        if shard is not None:
            res["shard"] = list(shard)
            f = C.OUT / f"per_dataset{sfx}_shard{shard[0]}.json"
            json.dump(res, open(f, "w"), indent=1, default=str)
            print(f"-> {f}")
            return 0
    json.dump(res, open(C.OUT / f"per_dataset{sfx}.json", "w"), indent=1, default=str)
    df = grid(res)
    df.to_csv(C.OUT / f"per_dataset_auroc{sfx}.csv", index=False)
    out = C.ROOT / f"reports/branch_fusion/PER_DATASET{sfx.upper()}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown(res, df, a.top))
    print(f"\n-> {C.OUT / f'per_dataset{sfx}.json'}\n"
          f"-> {C.OUT / f'per_dataset_auroc{sfx}.csv'}\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

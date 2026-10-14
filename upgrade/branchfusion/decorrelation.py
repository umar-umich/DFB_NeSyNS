#!/usr/bin/env python3
"""The pairwise correlation structure of every branch, and which triples are actually decorrelated.

    OMP_NUM_THREADS=4 <discern_ext python> upgrade/branchfusion/decorrelation.py --branches A,S,V,O,M,P,Y

THE QUESTION UMAR ASKED. The deployed system wants three branches that are each strong AND mutually
uncorrelated. "Is Y redundant with A" is one cell of that question; the answer needs the whole matrix.

A CORRECTION TO AN EARLIER STATISTIC. I first reported correlation across the TEN DATASET-LEVEL
AUROCs (n = 10). That number is weak and partly an artefact: it is dominated by how hard each dataset
is, so any two competent branches look correlated because both score high on IDForge and low on
DeepfakeEval2024. It cannot see whether two branches fail on the SAME VIDEOS, which is the only thing
that matters for fusion. Everything here is therefore computed PER VIDEO.

TWO MEASURES, because they answer different questions:

  phi    Pearson correlation of the two branches' BINARY ERROR vectors, each branch thresholded at
         its own EER point taken from the FF++ validation rows inside development -- never from a
         test set. This is the fusion-relevant quantity: it asks whether the branches are wrong
         together. Reported overall and split by real/fake, because a pair can decorrelate on fakes
         while agreeing on every false alarm.
  r_p    Pearson correlation of the predicted probabilities. Finer-grained (no thresholding) but it
         rewards agreement on easy videos that no fusion rule needs help with.

WHAT THIS IS NOT ALLOWED TO DO. Section 4 forbids promoting a branch because its errors decorrelate,
or rejecting one because a fitted weight reached zero. So `triples` reports the measured fused macro
beside the correlation, and `predicts_fusion` asks the honest question: across all triples, does low
correlation actually track fusion gain? If it does not, the correlation matrix is description, not a
selection rule, and must be used as such.
"""
from __future__ import annotations

import argparse
import itertools
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


def thresholds(dev: pd.DataFrame, items) -> dict:
    """Each branch's EER point on the FF++ validation rows inside development."""
    from health import eer
    y = dev.y.to_numpy()
    ff = np.where(dev.source.to_numpy() == "FFpp_val")[0] if "source" in dev.columns else np.arange(len(y))
    out = {}
    for b in items:
        p = dev[f"p_{b}"].to_numpy()
        out[b] = float(eer(y[ff], p[ff])[1]) if len(ff) else 0.5
    return out


def _corr(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def matrix(t: pd.DataFrame, items, taus: dict) -> dict:
    """Pairwise phi (binary errors) and r_p (probabilities) on one table of videos."""
    y = t.y.to_numpy()
    err, prob = {}, {}
    for b in items:
        p = t[f"p_{b}"].to_numpy()
        prob[b] = p
        err[b] = ((p >= taus[b]).astype(int) != y).astype(int)
    real, fake = y == 0, y == 1
    out = {"n": int(len(y)), "error_rate": {b: float(err[b].mean()) for b in items},
           "auroc": {b: C.auroc(y, prob[b]) for b in items}, "phi": {}, "r_p": {}}
    for a, b in itertools.combinations(items, 2):
        k = f"{a}-{b}"
        out["phi"][k] = {"all": _corr(err[a], err[b]),
                         "real": _corr(err[a][real], err[b][real]),
                         "fake": _corr(err[a][fake], err[b][fake])}
        out["r_p"][k] = _corr(prob[a], prob[b])
    return out


def pooled_test(items, taus: dict) -> dict:
    """One matrix over all ten test sets stacked, and one per set."""
    tabs = B.test_tables(items)
    per = {ds: matrix(t, items, taus) for ds, t in tabs.items()}
    stack = pd.concat([tabs[d] for d in C.TEST_SETS if d in tabs], ignore_index=True)
    return {"pooled": matrix(stack, items, taus), "per_dataset": per}


def triples(dev, items, mat: dict, op=S.SCREEN_OP) -> list:
    """Every triple: its branches' strength, its WORST pairwise phi, and its measured fused macro."""
    rows = []
    for tri in itertools.combinations(items, 3):
        ph = [mat["phi"][f"{a}-{b}"]["all"] for a, b in itertools.combinations(tri, 2)]
        r = S.screen_subset(dev, tri, op)
        singles = [mat["auroc"][b] for b in tri]
        rows.append({"subset": C.subset_id(tri),
                     "max_phi": float(np.nanmax(ph)), "mean_phi": float(np.nanmean(ph)),
                     "pairs": {f"{a}-{b}": mat["phi"][f"{a}-{b}"]["all"]
                               for a, b in itertools.combinations(tri, 2)},
                     "min_single_auroc": float(np.min(singles)),
                     "mean_single_auroc": float(np.mean(singles)),
                     "fused_macro": r["v31_macro"], "fused_pooled_auroc": r["oof_auroc"],
                     "gain_over_best_single": r["v31_macro"] - max(
                         S.screen_subset(dev, (b,), op)["v31_macro"] for b in tri)})
    return sorted(rows, key=lambda d: -d["fused_macro"])


def predicts_fusion(rows: list) -> dict:
    """Does low error correlation actually track fusion gain? Asked, not assumed."""
    mp = np.array([r["max_phi"] for r in rows], float)
    gain = np.array([r["gain_over_best_single"] for r in rows], float)
    macro = np.array([r["fused_macro"] for r in rows], float)
    ok = np.isfinite(mp) & np.isfinite(gain)
    from scipy.stats import spearmanr
    rho_gain = float(spearmanr(mp[ok], gain[ok]).statistic) if ok.sum() > 2 else float("nan")
    rho_macro = float(spearmanr(mp[ok], macro[ok]).statistic) if ok.sum() > 2 else float("nan")
    return {"n_triples": int(ok.sum()),
            "spearman_maxphi_vs_gain": rho_gain,
            "spearman_maxphi_vs_fused_macro": rho_macro,
            "reading": (
                "A NEGATIVE spearman means more-decorrelated triples fuse better, which is the "
                "assumption behind picking branches by decorrelation. A value near zero means the "
                "correlation matrix does NOT predict fusion on this data and must be read as "
                "description only -- section 4 forbids promoting a branch because its errors "
                "decorrelate, and this is the check that keeps that rule honest.")}


def render(mat: dict, items, title: str) -> str:
    L = [f"### {title}  (n = {mat['n']} videos)", "",
         "| branch | AUROC | error rate |", "|---|---:|---:|"]
    for b in items:
        L.append(f"| {b} | {mat['auroc'][b]:.4f} | {mat['error_rate'][b]:.4f} |")
    L += ["", "**phi — correlation of BINARY ERRORS (lower = more complementary)**", "",
          "| | " + " | ".join(items) + " |", "|---|" + "---:|" * len(items)]
    for a in items:
        cells = []
        for b in items:
            if a == b:
                cells.append("—")
            else:
                k = f"{a}-{b}" if f"{a}-{b}" in mat["phi"] else f"{b}-{a}"
                cells.append(f"{mat['phi'][k]['all']:+.3f}")
        L.append(f"| **{a}** | " + " | ".join(cells) + " |")
    L += ["", "**r_p — correlation of predicted probabilities**", "",
          "| | " + " | ".join(items) + " |", "|---|" + "---:|" * len(items)]
    for a in items:
        cells = []
        for b in items:
            if a == b:
                cells.append("—")
            else:
                k = f"{a}-{b}" if f"{a}-{b}" in mat["r_p"] else f"{b}-{a}"
                cells.append(f"{mat['r_p'][k]:+.3f}")
        L.append(f"| **{a}** | " + " | ".join(cells) + " |")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branches", default="A,S,V,O,M,P")
    ap.add_argument("--no-triples", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    items = [x.strip() for x in a.branches.split(",") if x.strip()]
    dev = B.dev_table(items, require_all=True)
    taus = thresholds(dev, items)
    print(f"branches {items} | development {len(dev)} videos")
    print("EER thresholds from FF++ val: " + "  ".join(f"{b} {taus[b]:.3f}" for b in items))

    res = {"items": items, "thresholds": taus,
           "development": matrix(dev, items, taus), "test": pooled_test(items, taus)}
    md = ["# Branch correlation structure", "",
          "> phi is computed PER VIDEO on binary errors, each branch thresholded at its own EER "
          "point from the FF++ validation rows inside development. An earlier version of this "
          "analysis correlated the ten dataset-level AUROCs instead; that statistic is dominated by "
          "how hard each dataset is and cannot see whether two branches fail on the same videos.", "",
          render(res["development"], items, "Development pool"), "",
          render(res["test"]["pooled"], items, "All ten test sets pooled"), ""]

    if not a.no_triples:
        rows = triples(dev, items, res["development"])
        res["triples"] = rows
        res["predicts_fusion"] = predicts_fusion(rows)
        md += ["### Every triple: strength, worst pairwise phi, and measured fusion", "",
               "| triple | fused macro | gain over best single | worst phi | mean phi | weakest branch |",
               "|---|---:|---:|---:|---:|---:|"]
        for r in rows:
            md.append(f"| {r['subset']} | {r['fused_macro']:.4f} | {r['gain_over_best_single']:+.4f} "
                      f"| {r['max_phi']:+.3f} | {r['mean_phi']:+.3f} | {r['min_single_auroc']:.4f} |")
        pf = res["predicts_fusion"]
        md += ["", f"**Spearman(worst phi, gain over best single) = {pf['spearman_maxphi_vs_gain']:+.3f}**  "
                   f"over {pf['n_triples']} triples; against the fused macro "
                   f"{pf['spearman_maxphi_vs_fused_macro']:+.3f}.", "", pf["reading"], ""]
        print(f"\nSpearman(max phi, gain over best single) = {pf['spearman_maxphi_vs_gain']:+.4f}")
        print(f"Spearman(max phi, fused macro)           = {pf['spearman_maxphi_vs_fused_macro']:+.4f}")
        print("\ntop triples by fused macro:")
        for r in rows[:8]:
            print(f"  {r['subset']:<10} macro {r['fused_macro']:.4f}  worst phi {r['max_phi']:+.3f}  "
                  f"pairs " + " ".join(f"{k}{v:+.2f}" for k, v in r["pairs"].items()))

    sfx = f"_{a.tag}" if a.tag else ""
    json.dump(res, open(C.OUT / f"decorrelation{sfx}.json", "w"), indent=1, default=str)
    out = C.REPORT / f"CORRELATION{sfx.upper()}.md"
    out.write_text("\n".join(md) + "\n")
    print(f"\n-> {C.OUT / f'decorrelation{sfx}.json'}\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Does Y16 come apart from A under degradation, or duplicate it there too?

    <discern_ext python> upgrade/branchfusion/robustness.py

Y16 is a near-duplicate of A on clean data (phi +0.678 development, +0.669 test, gate 0.60). The
case for keeping it anyway is that it is built differently -- off-manifold residual, block-16 tap,
beta-VAE in the training loop -- and may therefore hold where A does not. That case is testable and
this is the test: the same two heads, over features re-extracted from jpeg:40 crops, on the same
development videos.

WHAT WOULD SUPPORT KEEPING IT. phi(Y16, A) falling under degradation: the two views separate where
robustness matters, even though they agree when the input is clean. A second, weaker signal is Y16
losing less AUROC than A does.

WHAT WOULD NOT. phi staying near +0.67. Then Y16 duplicates A under attack as well as on clean
data, and the argument for its row is novelty alone -- a legitimate reason to keep a branch, but a
different one, and it should be stated as that rather than as robustness.

Thresholds come from the CLEAN development pool and are reused for the degraded one, because a
branch rethresholded on the degraded data would be scored at an operating point it could not have
chosen in deployment.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                               # noqa: E402
import branches as B                                             # noqa: E402
import decorrelation as D                                        # noqa: E402

OUT = C.ROOT / "reports/branch_fusion/ROBUSTNESS_Y16.md"
#: A's head, replayed over whichever feature cache the prefix names.
A_HEAD = "view_pe_core_lLN_s0.pt"


def pool(frame):
    """Concatenate the development sets into one table of (v, y, p)."""
    return pd.concat(frame, ignore_index=True).drop_duplicates("v")


def auroc(y, p) -> float:
    y = np.asarray(y); p = np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([neg, pos]), kind="mergesort")
    ranks = np.empty(len(order), float); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def a_opinions(prefix: str) -> pd.DataFrame | None:
    rows = []
    for ds in C.DEV_SETS:
        try:
            rows.append(B._head_opinions(prefix, A_HEAD, ds)[["v", "y", "p"]])
        except FileNotFoundError:
            return None
    return pool(rows)


def y16_opinions(branch_dir: Path) -> pd.DataFrame | None:
    rows = []
    for ds in C.DEV_SETS:
        f = branch_dir / f"{ds}.csv"
        if not f.exists():
            return None
        rows.append(pd.read_csv(f)[["v", "y", "p"]])
    return pool(rows)


def phi(a: pd.DataFrame, y: pd.DataFrame, taus=None):
    t = a.rename(columns={"p": "p_A"}).merge(
        y[["v", "p"]].rename(columns={"p": "p_Y16"}), on="v", how="inner")
    taus = taus or D.thresholds(t, ["A", "Y16"])
    return float(D.matrix(t, ["A", "Y16"], taus)["phi"]["A-Y16"]["all"]), taus, len(t)


#: (label in the report, feature-cache suffix, branch-directory suffix). The suffix is what
#: `extract_adapted` builds -- `process.replace(":", "")` -- so the dot in resize:0.6 survives.
CONDITIONS = [("jpeg:40", "jpeg40"), ("resize:0.6", "resize0.6")]


def main() -> int:
    clean_a = a_opinions("pe_core_lLN")
    clean_y = y16_opinions(C.BRANCH_OUT / "om_eil_Y16")
    if clean_a is None or clean_y is None:
        print("not ready: clean opinions missing")
        return 1
    p_clean, taus, n_c = phi(clean_a, clean_y)

    L = ["# Y16 against A under degradation", "",
         "Development pool. Thresholds are the CLEAN EER points, reused on every degraded pool: a",
         "branch rethresholded after the attack would be scored at an operating point it could not",
         "have chosen in deployment.", "",
         f"| condition | A AUROC | Y16 AUROC | dA | dY16 | phi(Y16, A) | d phi |",
         "|---|---:|---:|---:|---:|---:|---:|",
         f"| clean (n={n_c}) | {auroc(clean_a.y, clean_a.p):.4f} | "
         f"{auroc(clean_y.y, clean_y.p):.4f} | -- | -- | {p_clean:+.3f} | -- |"]

    seen = 0
    for name, sfx in CONDITIONS:
        deg_a = a_opinions(f"pe_core_lLN_{sfx}")
        deg_y = y16_opinions(C.BRANCH_OUT / f"om_eil_Y16_{sfx}")
        if deg_a is None or deg_y is None:
            L.append(f"| {name} | not yet extracted | | | | | |")
            continue
        seen += 1
        p_deg, _, n_d = phi(deg_a, deg_y, taus)
        a_c, a_d = auroc(clean_a.y, clean_a.p), auroc(deg_a.y, deg_a.p)
        y_c, y_d = auroc(clean_y.y, clean_y.p), auroc(deg_y.y, deg_y.p)
        L.append(f"| {name} (n={n_d}) | {a_d:.4f} | {y_d:.4f} | {a_d - a_c:+.4f} | "
                 f"{y_d - y_c:+.4f} | {p_deg:+.3f} | {p_deg - p_clean:+.3f} |")

    L.append("")
    if not seen:
        L.append("No degraded condition has finished extracting yet.")
    else:
        L += ["**How to read this.** A fall in phi is the result that matters: it means Y16 and A",
              "stop failing on the same videos exactly where robustness is at stake, which is the",
              "evidence for giving Y16 a row rather than treating it as a duplicate of A. A fall in",
              "Y16's own AUROC larger than A's is NOT a contradiction and must not be written as a",
              "robustness win: Y16 is differently wrong, not less wrong, and for a fusion row that",
              "is the property that counts. The two claims are separate and the table reports both."]
    OUT.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

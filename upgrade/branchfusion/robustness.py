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


def main() -> int:
    clean_a = a_opinions("pe_core_lLN")
    clean_y = y16_opinions(C.BRANCH_OUT / "om_eil_Y16")
    deg_a = a_opinions("pe_core_lLN_jpeg40")
    deg_y = y16_opinions(C.BRANCH_OUT / "om_eil_Y16j40")
    missing = [n for n, v in (("A clean", clean_a), ("Y16 clean", clean_y),
                              ("A jpeg40", deg_a), ("Y16 jpeg40", deg_y)) if v is None]
    if missing:
        print(f"not ready: {', '.join(missing)}")
        return 1

    p_clean, taus, n_c = phi(clean_a, clean_y)
    p_deg, _, n_d = phi(deg_a, deg_y, taus)                      # CLEAN thresholds, reused

    rows = [("A", auroc(clean_a.y, clean_a.p), auroc(deg_a.y, deg_a.p)),
            ("Y16", auroc(clean_y.y, clean_y.p), auroc(deg_y.y, deg_y.p))]

    L = ["# Y16 against A under jpeg:40", "",
         "Development pool. Thresholds are the CLEAN EER points, reused on the degraded data: a",
         "branch rethresholded after the attack would be scored at an operating point it could not",
         "have chosen in deployment.", "",
         f"| branch | AUROC clean (n={n_c}) | AUROC jpeg:40 (n={n_d}) | change |",
         "|---|---:|---:|---:|"]
    for name, c, d in rows:
        L.append(f"| {name} | {c:.4f} | {d:.4f} | {d - c:+.4f} |")
    L += ["", "**Per-video error phi, Y16 against A**", "",
          "| condition | phi |", "|---|---:|",
          f"| clean | {p_clean:+.3f} |", f"| jpeg:40 | {p_deg:+.3f} |",
          f"| change | {p_deg - p_clean:+.3f} |", ""]
    moved = p_deg - p_clean
    if moved <= -0.05:
        L.append(f"Y16 and A SEPARATE under compression: phi falls {abs(moved):.3f}, so the two "
                 f"views stop failing on the same videos exactly where robustness is at stake. "
                 f"That is evidence for the row, not just a reason to hope for one.")
    elif moved >= 0.05:
        L.append(f"Y16 and A converge FURTHER under compression (phi {moved:+.3f}). Whatever Y16 "
                 f"adds, it is not independence under this attack.")
    else:
        L.append(f"phi barely moves ({moved:+.3f}). Y16 duplicates A under compression as well as "
                 f"on clean data, so the case for keeping it rests on novelty of construction, "
                 f"which is a real reason but a different one -- it should be written as that.")
    OUT.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""The exposed suites: ten test sets and all 41 DF40 generator sets, every arm of the fixed matrix.

    OMP_NUM_THREADS=8 python analysis/tbiom/grouped/exposed.py

The recipe is frozen by `tbiom/configs/grouped_fusion_preregister.yaml` before these are scored. Both DF40 means the
brief asks for are reported -- family-balanced (the mean of the four family x source cell means) and generator-macro
(the plain mean over all 41 sets) -- with their weights stated, and never one family alone.

One check here is decisive for the novelty question and is reported before any headline: whether grouped-R, under the
discount its own fit chooses, is numerically identical to the two-view T+OM fusion. It is, whenever the fit sets the
appearance weight to zero.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent / "temporal")]
import ops, score as S  # noqa: E402
import exp1_fusion_audit as A  # noqa: E402
import dsr_system as D  # noqa: E402
import df40_analysis as X  # noqa: E402

R, LOG = Path("tbiom/reports"), Path("logs/tbiom/df40")
ARMS = ops.ALL_ARMS
TESTS = ["CDFv2", "CDFv3", "DFD", "DFDC", "DFDCP", "DFEval24", "FFIW", "KoDF", "IDForge", "DFDM"]
CELLS = (("face swap", "cdf"), ("face swap", "ff"), ("reenactment", "cdf"), ("reenactment", "ff"))


def main():
    t0 = time.time()
    dev, _dproc, tests = D.load("vjepa2_lora_B_seed42")
    meta = json.load(open(LOG / "meta.json"))
    d40 = {n: X.ours_table(n) for n in meta if (X.S23 / f"ours_{n}.csv").exists()}
    print(f"{len(d40)} DF40 sets ({time.time() - t0:.0f}s)", flush=True)
    out = {"tests": {}, "df40": {}, "w": {}, "identity": {}}
    for k, Fd, _Sd in D.splits(dev):
        clF = A.cells(Fd)
        for arm in ARMS:
            w = S.fit_w(Fd, arm, clF)
            out["w"].setdefault(arm, []).append(list(w))
            for tn, T in tests.items():
                y = T.y.to_numpy(); p = ops.p_of(S.fused(T, arm, w))
                out["tests"].setdefault(arm, {}).setdefault(tn, []).append(A.fast_auc(p[y == 0], p[y == 1]))
            for n, t in d40.items():
                y = t.y.to_numpy(); p = ops.p_of(S.fused(t, arm, w))
                out["df40"].setdefault(arm, {}).setdefault(n, []).append(A.fast_auc(p[y == 0], p[y == 1]))
            if arm == "grouped-R":
                same = [bool(np.allclose(ops.p_of(S.fused(t, "grouped-R", w)), ops.p_of(S.fused(t, "T+OM", w))))
                        for t in d40.values()]
                out["identity"][f"split{k}"] = {"w": list(w), "identical_to_T+OM_on_all_DF40": all(same)}
        print(f"  split {k} done ({time.time() - t0:.0f}s)", flush=True)
    json.dump(out, open(R / "EXP_GROUPED_EXPOSED.json", "w"), indent=1, default=float)
    write(out, meta, time.time() - t0)


def write(out, meta, secs):
    m = lambda arm, tn: float(np.mean(out["tests"][arm][tn]))
    g = lambda arm, n: float(np.mean(out["df40"][arm][n]))
    cell = lambda arm, fam, src: float(np.mean([g(arm, n) for n in out["df40"][arm]
                                                if meta[n]["family"] == fam and meta[n]["source"] == src]))
    macro = lambda arm: float(np.mean([g(arm, n) for n in out["df40"][arm]]))
    balanced = lambda arm: float(np.mean([cell(arm, f, s) for f, s in CELLS]))

    nid = sum(v["identical_to_T+OM_on_all_DF40"] for v in out["identity"].values())
    L = ["# Grouped fusion on the exposed suites (ten test sets and DF40)", "",
         __doc__.split("\n\n", 2)[2].strip(), "",
         f"Wall clock {secs / 60:.1f} minutes, CPU. Detector seed 42.", "",
         "## 0. The identity check, before anything else", "",
         "| split | fitted discount (T, APP, OM) | grouped-R identical to T+OM on all 41 DF40 sets |",
         "|---|---|---|"]
    for k, v in out["identity"].items():
        L.append(f"| {k} | ({', '.join(f'{x:g}' for x in v['w'])}) | "
                 f"{'**yes**' if v['identical_to_T+OM_on_all_DF40'] else 'no'} |")
    L += ["", f"**In {nid} of 5 folds the fitted discount sets the appearance weight to zero, and grouped-R is then "
          "bit-for-bit the two-view Dempster fusion of the temporal and off-manifold views.** This is not a numerical "
          "coincidence: an evidence-scaled opinion with r = 0 carries no evidence, its canonical weights are both 1, "
          "and the minimum of 1 and w_OM is w_OM, so the cautious inner step returns the off-manifold opinion "
          "unchanged and the outer Dempster step is the only thing left. Under its own fitted discount the operator "
          "is therefore, in most folds, not a new combination rule at all — it is the decision to drop the "
          "appearance view. Any claim built on the fitted-discount numbers has to answer that first, which is why "
          "T+OM is carried as an arm in every table below.", "",
          "## 1. Ten test sets, video AUROC", "",
          "| arm | " + " | ".join(TESTS) + " | mean |", "|---" * (len(TESTS) + 2) + "|"]
    for a in ARMS:
        row = [m(a, t) for t in TESTS]
        tag = "**" if a == "grouped-R" else ""
        note = " *(baseline)*" if a in ("Dempster-R", "AVG-R") else ""
        L.append(f"| {tag}{a}{tag}{note} | " + " | ".join(f"{x:.4f}" for x in row) + f" | {np.mean(row):.4f} |")

    L += ["", "## 2. DF40, both means the brief requires", "",
          "Family-balanced = the mean of the four family x source cell means, each cell weighted 1/4. "
          "Generator-macro = the plain mean over all 41 generator sets, each set weighted 1/41, which weights the "
          "Celeb-DF-sourced cells more because they contain more generators.", "",
          "| arm | swap CDF | swap FF++ | reenact CDF | reenact FF++ | family-balanced | generator-macro |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for a in ARMS:
        tag = "**" if a == "grouped-R" else ""
        note = " *(baseline)*" if a in ("Dempster-R", "AVG-R") else ""
        L.append(f"| {tag}{a}{tag}{note} | " + " | ".join(f"{cell(a, f, s):.4f}" for f, s in CELLS)
                 + f" | {balanced(a):.4f} | {macro(a):.4f} |")

    L += ["", "## 3. grouped-R against the arms that matter, per DF40 family", "",
          "| contrast | swap CDF | swap FF++ | reenact CDF | reenact FF++ | family-balanced | generator-macro |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for b in ("Dempster-R", "AVG-R", "T+OM", "CAUT-R"):
        L.append(f"| grouped-R − {b} | " + " | ".join(f"{cell('grouped-R', f, s) - cell(b, f, s):+.4f}" for f, s in CELLS)
                 + f" | {balanced('grouped-R') - balanced(b):+.4f} | {macro('grouped-R') - macro(b):+.4f} |")
    L += ["", "The row against T+OM is the one to read first: where it is zero, the grouped operator is doing "
          "nothing that dropping the appearance view does not already do.", "",
          "## 4. Fitted discounts", "", "| arm | " + " | ".join(f"split {i}" for i in range(5)) + " |",
          "|---" * 6 + "|"]
    for a in ARMS:
        L.append(f"| {a} | " + " | ".join("(" + ", ".join(f"{x:g}" for x in w) + ")" for w in out["w"][a]) + " |")
    L += ["", "Both suites are exposed: CDFv2 and DeepfakeEval2024 informed development, and DF40 informed the "
          "choice of this operator. Nothing here is a confirmation set (`GROUPED_REFERENCE_MANIFEST.md` §5).", ""]
    (R / "EXP_GROUPED_EXPOSED.md").write_text("\n".join(L) + "\n")
    print("wrote EXP_GROUPED_EXPOSED.md")


if __name__ == "__main__":
    main()

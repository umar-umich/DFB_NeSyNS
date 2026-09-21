#!/usr/bin/env python3
"""Stage 3 pre-check: does the Stage-1B winner (arm E spatial views) beat the DEPLOYED spatial views?

    OMP_NUM_THREADS=8 python analysis/tbiom/stage3_precheck.py

Stage 1B compared arm E with arm B, both from the sbi_arm family. The deployed reference takes its Appearance and
Off-manifold views from the warm+gate detector (epoch 19). Same frozen temporal branch, same 5 source-grouped folds,
DS-R and the logistic comparator refitted per arm, paired source-group bootstrap. Development only.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "temporal"), str(HERE / "freqblender")]
import exp1_fusion_audit as A  # noqa: E402
import dsr_system as D  # noqa: E402
import finish_line as F  # noqa: E402
import exp5_analysis as E  # noqa: E402

PROC, RULES = ("jpeg40", "rs06"), ("DS-R", "LR")


def main():
    arms = {"deployed (warm+gate e19)": E.load_arm("warmgate_e19", "warmgate_sel", "warmgate"),
            "arm E (sbiarmE_e02)": E.load_arm("sbiarmE_e02", "fbE_sel", "fbE")}
    names = list(arms); rng = np.random.default_rng(20260921); rows = []
    for k, Fd, Sd in D.splits(arms[names[0]][0]):
        vs, gS = Sd.v.to_numpy(), A.groups_for(Sd); cl = A.cells(Sd); sc, row = {}, {"arms": {}}
        for name, (dev, dproc, tests) in arms.items():
            fd = dev.set_index("v").loc[Fd.v].reset_index(); fd["source"] = Fd.source.to_numpy()
            sd = dev.set_index("v").loc[vs].reset_index(); sd["source"] = Sd.source.to_numpy()
            rules, _, _ = F.rules_for(fd, E.VIEWS); r = {}
            for rule, f in rules.items():
                s = f(sd); tot, ty = A.v31(s, A.cells(sd)); tau = F.tau_of(s, sd)
                r[f"{rule}|clean"] = {"v31": tot, **ty, "fpr": E.rates(s, sd, tau)[0], "tpr": E.rates(s, sd, tau)[1]}
                sc[f"{name}|{rule}|clean"] = s
                for c, P in dproc.items():
                    pt = P.set_index("v").loc[vs].reset_index(); pt["source"] = Sd.source.to_numpy()
                    sp = f(pt); r[f"{rule}|{c}"] = {"v31": A.v31(sp, A.cells(pt))[0]}; sc[f"{name}|{rule}|{c}"] = sp
            row["arms"][name] = r
        ug = np.unique(gS); pos = {g: np.where(gS == g)[0] for g in ug}; row["boot"] = {}
        for rule in RULES:
            b = {"clean": [], "processed": [], "reenactment": []}
            for _ in range(1000):
                w = np.zeros(len(Sd), dtype=int)
                for g in rng.choice(ug, len(ug), replace=True):
                    w[pos[g]] += 1
                t1, y1 = A.v31(sc[f"{names[1]}|{rule}|clean"], cl, w); t0, y0 = A.v31(sc[f"{names[0]}|{rule}|clean"], cl, w)
                b["clean"].append(t1 - t0); b["reenactment"].append(y1["reenactment"] - y0["reenactment"])
                b["processed"].append(np.mean([A.v31(sc[f"{names[1]}|{rule}|{c}"], cl, w)[0] - A.v31(sc[f"{names[0]}|{rule}|{c}"], cl, w)[0] for c in PROC]))
            row["boot"][rule] = {q: [float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for q, v in b.items()}
        rows.append(row); print(f"split {k} done", flush=True)
    json.dump(rows, open("tbiom/reports/STAGE3_PRECHECK.json", "w"), indent=1, default=float)
    m = lambda n, key, f="v31": float(np.mean([r["arms"][n][key][f] for r in rows]))
    L = ["# Stage 3 pre-check — arm E spatial views against the deployed spatial views", "", __doc__.split("\n\n", 1)[1].split("\n\nPROC")[0].strip(), "",
         "| arm | rule | clean | swap | reenactment | in-the-wild | JPEG q40 | resize 0.60 | FPR own τ | TPR own τ |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n in names:
        for rule in RULES:
            c = f"{rule}|clean"
            L.append(f"| {n} | {rule} | {m(n, c):.4f} | {m(n, c, 'swap'):.4f} | {m(n, c, 'reenactment'):.4f} | {m(n, c, 'in_the_wild'):.4f} | "
                     f"{m(n, f'{rule}|jpeg40'):.4f} | {m(n, f'{rule}|rs06'):.4f} | {m(n, c, 'fpr'):.3f} | {m(n, c, 'tpr'):.3f} |")
    L += ["", "| rule | quantity | mean Δ (E − deployed) | splits CI above 0 | splits CI below 0 |", "|---|---|---:|---:|---:|"]
    for rule in RULES:
        for q in ("clean", "processed", "reenactment"):
            v = [r["boot"][rule][q] for r in rows]
            L.append(f"| {rule} | {q} | {np.mean([x[0] for x in v]):+.4f} | {sum(x[1] > 0 for x in v)}/5 | {sum(x[2] < 0 for x in v)}/5 |")
    Path("tbiom/reports/STAGE3_PRECHECK.md").write_text("\n".join(L) + "\n"); print("wrote tbiom/reports/STAGE3_PRECHECK.md")


if __name__ == "__main__":
    main()

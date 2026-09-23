#!/usr/bin/env python3
"""Section 7: the exposed suites, scored after the recipe is frozen.

    OMP_NUM_THREADS=8 python analysis/tbiom/pcr6p/exposed.py

Ten test sets, the processed conditions, and all 41 DF40 generator sets, for every arm of the fixed matrix and every
detector configuration. Labelled exposed benchmark evaluation: CDFv2 and DeepfakeEval2024 informed development and
DF40 has informed operator design in three studies now, so none of this confirms anything.

Both DF40 means are reported with their weights: family-balanced gives each of the four family x source cells weight
1/4; generator-macro gives each of the 41 sets weight 1/41, which favours the Celeb-DF-sourced cells because they
contain more generators.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "grouped"), str(HERE.parent), str(HERE.parent / "temporal")]
import arms as AR, ops_pcr6p as P  # noqa: E402
import score as S  # noqa: E402
import exp1_fusion_audit as A  # noqa: E402
import dsr_system as D  # noqa: E402
import confirm as C  # noqa: E402
from safeguard_ablation import rc_metrics  # noqa: E402

R, LOG = Path("tbiom/reports"), Path("logs/tbiom/df40")
CONF = {"42": ("ours", "warmgate_e19", "vjepa2_lora_B_seed42"),
        "T1337": ("ours", "warmgate_e19", "vjepa2_lora_B_seed1337"),
        "armB42": ("armB42", "sbiarmB_e08", "vjepa2_lora_B_seed42"),
        "armB1337": ("armB1337", "sbiarmBs1337_e03", "vjepa2_lora_B_seed1337")}
TESTS = ["CDFv2", "CDFv3", "DFD", "DFDC", "DFDCP", "DFEval24", "FFIW", "KoDF", "IDForge", "DFDM"]
CELLS = (("face swap", "cdf"), ("face swap", "ff"), ("reenactment", "cdf"), ("reenactment", "ff"))
SIMPLE = ("vacuity", "conflict_K", "entropy", "margin")


def main():
    t0 = time.time()
    meta = json.load(open(LOG / "meta.json"))
    out = {}
    for tag, (pref, devtag, br) in CONF.items():
        dev = C.dev_table(devtag, br)
        tests = D.load(br)[2] if pref == "ours" else {}
        d40 = {n: C.table(n, pref, br) for n in meta if (C.X.S23 / f"{pref}_{n}.csv").exists()}
        res = {"tests": {}, "df40": {}, "w": {}, "unc": {}}
        for k, Fd, _Sd in D.splits(dev):
            clF = A.cells(Fd)
            for arm in AR.ARMS:
                w = AR.fit_w(Fd, arm, clF)
                res["w"].setdefault(arm, []).append(list(w))
                # the frozen threshold rule: EER on FF++ val, taken from the fitting fold and applied unchanged
                tau = AR.tau_of(P.p_of(AR.fuse(Fd, arm, w)), Fd)
                for tn, T in tests.items():
                    y = T.y.to_numpy()
                    p, f, sc, att = AR.scores(T, arm, w)
                    pred = (p >= tau).astype(int); err = pred != y
                    pc = np.clip(p, 1e-6, 1 - 1e-6)
                    r = {"auroc": A.fast_auc(p[y == 0], p[y == 1]),
                         "fpr": float((pred[y == 0] == 1).mean()) if (y == 0).any() else np.nan,
                         "tpr": float((pred[y == 1] == 1).mean()) if (y == 1).any() else np.nan,
                         "ece": S.ece(p, y), "brier": float(np.mean((p - y) ** 2)),
                         "nll": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))}
                    for nm, sig in sc.items():
                        m = rc_metrics(sig, err)
                        r[f"err_auroc_{nm}"] = float(m["auroc_err"]); r[f"eaurc_{nm}"] = float(m["eaurc"])
                        r[f"risk90_{nm}"] = float(m["risk@90"]); r[f"risk80_{nm}"] = float(m["risk@80"])
                    res["tests"].setdefault(arm, {}).setdefault(tn, []).append(r)
                for n, t in d40.items():
                    y = t.y.to_numpy(); p = P.p_of(AR.fuse(t, arm, w))
                    res["df40"].setdefault(arm, {}).setdefault(n, []).append(A.fast_auc(p[y == 0], p[y == 1]))
            print(f"  {tag} split {k} ({time.time() - t0:.0f}s)", flush=True)
        out[tag] = res
    json.dump(out, open(R / "EXP_PCR6P_EXPOSED.json", "w"), indent=1, default=float)
    write(out, meta, time.time() - t0)


def write(out, meta, secs):
    L = ["# PCR6+ on the exposed suites", "", __doc__.split("\n\n", 2)[2].strip(), "",
         f"Wall clock {secs / 60:.1f} minutes, CPU.", ""]
    for tag, res in out.items():
        g = lambda arm, n: float(np.mean(res["df40"][arm][n]))
        cell = lambda arm, fam, src: float(np.mean([g(arm, n) for n in res["df40"][arm]
                                                    if meta[n]["family"] == fam and meta[n]["source"] == src]))
        macro = lambda arm: float(np.mean([g(arm, n) for n in res["df40"][arm]]))
        bal = lambda arm: float(np.mean([cell(arm, f, s) for f, s in CELLS]))
        L += [f"## Configuration `{tag}`", ""]
        if res["tests"]:
            t = lambda arm, tn, key: float(np.mean([r[key] for r in res["tests"][arm][tn]]))
            L += ["### Ten test sets, video AUROC", "",
                  "| arm | " + " | ".join(TESTS) + " | mean |", "|---" * (len(TESTS) + 2) + "|"]
            for a in AR.ARMS:
                row = [t(a, tn, "auroc") for tn in TESTS]
                tg = "**" if a == "PCR6P-R" else ""
                L.append(f"| {tg}{a}{tg} | " + " | ".join(f"{x:.4f}" for x in row) + f" | {np.mean(row):.4f} |")
            L += ["", "### PCR6+-R per test set, at the frozen threshold", "",
                  "| test set | AUROC | FPR | TPR | ECE | NLL | Brier | E-AURC (vacuity) | err-AUROC vacuity | "
                  "err-AUROC K | err-AUROC entropy | risk@90 | risk@80 |", "|---" * 13 + "|"]
            for tn in TESTS:
                L.append(f"| {tn} | " + " | ".join(
                    f"{t('PCR6P-R', tn, k):.4f}" for k in ("auroc", "fpr", "tpr", "ece", "nll", "brier",
                                                           "eaurc_vacuity", "err_auroc_vacuity", "err_auroc_conflict_K",
                                                           "err_auroc_entropy", "risk90_vacuity", "risk80_vacuity")) + " |")
        L += ["", "### DF40, both required means", "",
              "| arm | swap CDF | swap FF++ | reenact CDF | reenact FF++ | family-balanced (1/4 each) | "
              "generator-macro (1/41 each) |", "|---|---:|---:|---:|---:|---:|---:|"]
        for a in AR.ARMS:
            tg = "**" if a == "PCR6P-R" else ""
            note = " *(baseline)*" if a in ("Dempster-R", "Averaging-R") else ""
            L.append(f"| {tg}{a}{tg}{note} | " + " | ".join(f"{cell(a, f, s):.4f}" for f, s in CELLS)
                     + f" | {bal(a):.4f} | {macro(a):.4f} |")
        L += ["", "### DF40 per generator, PCR6+-R against the baselines", "",
              "| generator | family | source | **PCR6+-R** | PCR6-R | Dempster-R | Averaging-R |",
              "|---|---|---|---:|---:|---:|---:|"]
        for n in sorted(res["df40"]["PCR6P-R"], key=lambda x: (meta[x]["family"], meta[x]["source"], meta[x]["method"])):
            L.append(f"| {meta[n]['method']} | {meta[n]['family']} | {meta[n]['source']} | "
                     f"**{g('PCR6P-R', n):.4f}** | {g('PCR6-R', n):.4f} | {g('Dempster-R', n):.4f} | "
                     f"{g('Averaging-R', n):.4f} |")
        L.append("")
    (R / "EXP_PCR6P_EXPOSED.md").write_text("\n".join(L) + "\n")
    print("wrote EXP_PCR6P_EXPOSED.md")


if __name__ == "__main__":
    main()

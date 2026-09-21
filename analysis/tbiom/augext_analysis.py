#!/usr/bin/env python3
"""Stage 1B (executable brief): extended processing (arm E) vs arm B, fused with the frozen temporal branch.

    OMP_NUM_THREADS=4 python analysis/tbiom/augext_analysis.py --extended sbiarmE_eNN

Criteria frozen in tbiom/docs/stage1b-augmentation-preregistration.md before arm E was trained. Uses the same loader,
folds, fusion refits and bootstrap as the Experiment 5 comparison; only the arms and the acceptance rule differ.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "temporal"), str(HERE / "freqblender")]
import exp1_fusion_audit as A  # noqa: E402
import dsr_system as D  # noqa: E402
import finish_line as F  # noqa: E402
import exp5_analysis as E  # noqa: E402

R, PROC, RULES = Path("tbiom/reports"), ("jpeg40", "rs06"), ("DS-R", "LR")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--extended", required=True); a = ap.parse_args()
    arms = {"B (existing mixture)": E.load_arm("sbiarmB_e08", "fbB_sel", "fbB"),
            "E (adds resize→JPEG)": E.load_arm(a.extended, "fbE_sel", "fbE")}
    names = list(arms); rng = np.random.default_rng(20260920); rows = []
    for k, Fd, Sd in D.splits(arms[names[0]][0]):
        vs, gS = Sd.v.to_numpy(), A.groups_for(Sd); cl = A.cells(Sd)
        row = {"split": k, "arms": {}, "boot": {}}; scores = {}
        tau_ctrl = {}
        for name, (dev, dproc, tests) in arms.items():
            fd = dev.set_index("v").loc[Fd.v].reset_index(); fd["source"] = Fd.source.to_numpy()
            sd = dev.set_index("v").loc[vs].reset_index(); sd["source"] = Sd.source.to_numpy()
            rules, params, w = F.rules_for(fd, E.VIEWS); r = {"w": params["w"]}
            for rule, f in rules.items():
                s = f(sd); tau = F.tau_of(s, sd)
                if name == names[0]:
                    tau_ctrl[rule] = tau
                tot, ty = A.v31(s, A.cells(sd)); r[f"{rule}|clean"] = {"v31": tot, **ty}
                for tn, tv in (("own", tau), ("drop-in", tau_ctrl[rule])):
                    r[f"{rule}|clean"][f"fpr_{tn}"], r[f"{rule}|clean"][f"tpr_{tn}"] = E.rates(s, sd, tv)
                scores[f"{name}|{rule}|clean"] = s
                for c, P in dproc.items():
                    pt = P.set_index("v").loc[vs].reset_index(); pt["source"] = Sd.source.to_numpy()
                    sp = f(pt); t2, ty2 = A.v31(sp, A.cells(pt)); r[f"{rule}|{c}"] = {"v31": t2, **ty2}
                    scores[f"{name}|{rule}|{c}"] = sp
                r[f"{rule}|tests"] = {tn: A.fast_auc(f(T)[T.y.to_numpy() == 0], f(T)[T.y.to_numpy() == 1]) for tn, T in tests.items()}
            row["arms"][name] = r
        ug = np.unique(gS); pos = {g: np.where(gS == g)[0] for g in ug}
        for rule in RULES:
            dp, dc, dr = [], [], []
            for _ in range(E.N_BOOT):
                wts = np.zeros(len(Sd), dtype=int)
                for g in rng.choice(ug, len(ug), replace=True):
                    wts[pos[g]] += 1
                dp.append(np.mean([A.v31(scores[f"{names[1]}|{rule}|{c}"], cl, wts)[0] - A.v31(scores[f"{names[0]}|{rule}|{c}"], cl, wts)[0] for c in PROC]))
                t1, y1 = A.v31(scores[f"{names[1]}|{rule}|clean"], cl, wts); t0, y0 = A.v31(scores[f"{names[0]}|{rule}|clean"], cl, wts)
                dc.append(t1 - t0); dr.append(y1["reenactment"] - y0["reenactment"])
            q = lambda d: [float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
            row["boot"][rule] = {"processed": q(dp), "clean": q(dc), "reenactment": q(dr)}
        rows.append(row)
        print(f"split {k}: " + " | ".join(f"{n} DS-R clean {row['arms'][n]['DS-R|clean']['v31']:.4f}" for n in names), flush=True)
    json.dump(rows, open(R / "EXP_AUGMENTATION_EXTENSION.json", "w"), indent=1, default=float)

    m = lambda n, key, f="v31": float(np.mean([r["arms"][n][key][f] for r in rows]))
    L = ["# Stage 1B — matched augmentation extension (composed resize → JPEG)", "",
         "Pre-registration: `tbiom/docs/stage1b-augmentation-preregistration.md`. Manifest: "
         "`tbiom/reports/REFERENCE_MANIFEST.md`. Arm B = `sbi_armB_seed42` at its development-selected epoch 8; arm E = "
         f"`sbi_armE_seed42` (config identical except `training_arm: E`) at its development-selected checkpoint `{a.extended}`. "
         "Both spatial detectors are fused with the frozen reference temporal branch; DS-R and the logistic comparator "
         "are refitted per arm on the same source-grouped folds. This tests whether composition helps beyond individual "
         "transforms; it does not test the ceiling of FF++ generalisation.", "",
         "## 1. Development (selection folds, mean of 5 splits)", "",
         "| arm | rule | clean v3.1 | swap | reenactment | in-the-wild | JPEG q40 | resize 0.60 | FPR own τ | TPR own τ | FPR drop-in τ | TPR drop-in τ |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n in names:
        for rule in RULES:
            c = f"{rule}|clean"
            L.append(f"| {n} | {rule} | {m(n, c):.4f} | {m(n, c, 'swap'):.4f} | {m(n, c, 'reenactment'):.4f} | "
                     f"{m(n, c, 'in_the_wild'):.4f} | {m(n, f'{rule}|jpeg40'):.4f} | {m(n, f'{rule}|rs06'):.4f} | "
                     f"{m(n, c, 'fpr_own'):.3f} | {m(n, c, 'tpr_own'):.3f} | {m(n, c, 'fpr_drop-in'):.3f} | {m(n, c, 'tpr_drop-in'):.3f} |")
    L += ["", "## 2. Verdict against the frozen criteria (Δ = E − B, paired source-group bootstrap)", ""]
    for rule in RULES:
        b = [r["boot"][rule] for r in rows]
        gp = np.mean([x["processed"][0] for x in b]); gp_pos = sum(x["processed"][1] > 0 for x in b)
        gc = m(names[1], f"{rule}|clean") - m(names[0], f"{rule}|clean")
        ren = np.mean([x["reenactment"][0] for x in b]); ren_neg = sum(x["reenactment"][2] < 0 for x in b)
        fpr = m(names[1], f"{rule}|clean", "fpr_own") - m(names[0], f"{rule}|clean", "fpr_own")
        ok = gp >= 0.005 and gp_pos >= 3 and gc >= -0.005 and not (ren < 0 and ren_neg >= 3) and fpr <= 0.02
        L += [f"**{rule}** — processed Δ {gp:+.4f} (CI > 0 in {gp_pos}/5; need ≥ +0.005 and ≥ 3/5); clean Δ {gc:+.4f} "
              f"(need ≥ −0.005); reenactment Δ {ren:+.4f} (CI upper < 0 in {ren_neg}/5; ≥ 3/5 rejects); clean FPR Δ "
              f"{fpr:+.3f} at own τ (need ≤ +0.02) → **{'ACCEPT' if ok else 'NOT ACCEPTED'}**.", ""]
    tl = list(rows[0]["arms"][names[0]]["DS-R|tests"])
    L += ["## 3. Test sets — video AUROC, read after the verdict", "",
          "| test | " + " | ".join(f"{n} {rule}" for n in names for rule in RULES) + " |", "|---|" + "---:|" * 4]
    for t in tl:
        L.append(f"| {t} | " + " | ".join(f"{np.mean([r['arms'][n][f'{rule}|tests'][t] for r in rows]):.4f}" for n in names for rule in RULES) + " |")
    L += ["", "No test set, CDFv3 included, entered the verdict. One seed per arm."]
    (R / "EXP_AUGMENTATION_EXTENSION.md").write_text("\n".join(L) + "\n"); print("wrote", R / "EXP_AUGMENTATION_EXTENSION.md")


if __name__ == "__main__":
    main()

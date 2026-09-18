#!/usr/bin/env python3
"""Experiment 5, part 2: Spatial-B vs Spatial-B-Freq at the fused three-view level (cached outputs, CPU).

    OMP_NUM_THREADS=4 python analysis/tbiom/freqblender/exp5_analysis.py [--freq-tag warmgateBFreq_last]

Criteria, arms, selection and disqualifier were frozen before the FreqBlender arm was trained:
`tbiom/docs/exp5-freqblender-preregistration.md`. Nothing here chooses them after the fact.

Both arms are fused with the SAME frozen temporal branch (the reference B-augmented V-JEPA 2 LoRA checkpoint) and the
SAME source-grouped folds; DS-R and the logistic comparator are refitted separately per arm on the fitting folds.
Development decides; the test tables are printed after the verdict and never enter it.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "temporal"))
import exp1_fusion_audit as A  # noqa: E402
import dsr_system as D  # noqa: E402
import finish_line as F  # noqa: E402
from fusion_combos import attach  # noqa: E402
from health import eer  # noqa: E402

S23, S2, BR, R = Path("logs/tbiom/stage23"), Path("logs/tbiom/stage2"), Path("logs/tbiom/temporal/branches"), Path("tbiom/reports")
DEV, PROC, VIEWS = ("VALmix", "VALmixV2", "FFpp_val"), ("jpeg40", "rs06"), ["T", "APP", "OM"]
TEMPORAL = "vjepa2_lora_B_seed42"
N_BOOT = 1000


def load_arm(dev_tag, sel_tag, proc_tag, branch=TEMPORAL):
    ob = BR / branch / "opinions"
    dev = attach(A.dev_tables({n: S23 / f"{dev_tag}_{n}.csv" for n in DEV}, [1.0]), [ob / f"{n}.csv" for n in DEV])
    dproc = {c: attach(A.dev_tables({n: S2 / f"p_dev_{dev_tag}_{n}_{c}.csv" for n in DEV}, [1.0]),
                       [ob / f"{n}_{c}.csv" for n in DEV]) for c in PROC}
    tests = {}
    for t in A.TEST:
        d = A.load_frames(S23 / f"{sel_tag}_{t}.csv")
        keep = ~d.subset.str.contains("IP_LAP") & (d.subset != "fake-audio-driven") & ~d.subset.str.startswith("adv-")
        tests[t] = attach(A.video_table(d[keep].reset_index(drop=True), [1.0]), [ob / f"{t}.csv"])
    c3 = tests["CDFv3"]
    tests["CDFv3 reenactment"] = pd.concat([c3[c3.y == 0], c3[(c3.y == 1) & c3.subset.str.contains("FaceReenact|TalkingFace")]],
                                           ignore_index=True)
    for t in ("CDFv2", "DFDCP"):
        for c in PROC:
            tests[f"{t} {c}"] = attach(A.video_table(A.load_frames(S2 / f"p_v3_{proc_tag}_{t}_{c}.csv"), [1.0]),
                                       [ob / f"{t}_{c}.csv"])
    return dev, dproc, tests


def rates(score, t, tau):
    y = t.y.to_numpy(); pred = (score >= tau).astype(int)
    return float((pred[y == 0] == 1).mean()), float((pred[y == 1] == 1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", default="warmgateB_last"); ap.add_argument("--control-sel", default="warmgateB_sel")
    ap.add_argument("--control-proc", default="warmgateB")
    ap.add_argument("--freq", default="warmgateBFreq_last"); ap.add_argument("--freq-sel", default="warmgateBFreq_sel")
    ap.add_argument("--freq-proc", default="warmgateBFreq")
    a = ap.parse_args()
    arms = {"Spatial-B": load_arm(a.control, a.control_sel, a.control_proc),
            "Spatial-B-Freq": load_arm(a.freq, a.freq_sel, a.freq_proc)}
    rng = np.random.default_rng(20260917)
    rows = []
    base_dev = arms["Spatial-B"][0]
    for k, Fd, Sd in D.splits(base_dev):
        vs, gS = Sd.v.to_numpy(), A.groups_for(Sd)
        row = {"split": k, "arms": {}, "scores": {}}
        for name, (dev, dproc, tests) in arms.items():
            # the same videos, in the same order, for every arm
            fd = dev.set_index("v").loc[Fd.v].reset_index(); fd["source"] = Fd.source.to_numpy()
            sd = dev.set_index("v").loc[vs].reset_index(); sd["source"] = Sd.source.to_numpy()
            rules, params, w = F.rules_for(fd, VIEWS)
            cl = A.cells(sd)
            r = {"w": params["w"], "clean": {}, "proc": {}, "tau": {}, "rates": {}, "tests": {}}
            for rule, f in rules.items():
                s = f(sd); row["scores"][f"{name}|{rule}"] = s
                tot, types = A.v31(s, cl)
                tau = F.tau_of(s, sd); r["tau"][rule] = tau
                r["clean"][rule] = {"v31": tot, **types}
                r["rates"][f"{rule}|clean"] = rates(s, sd, tau)
                for c, P in dproc.items():
                    pt = P.set_index("v").loc[vs].reset_index(); pt["source"] = Sd.source.to_numpy()
                    sp = f(pt); t2, ty2 = A.v31(sp, A.cells(pt))
                    r["proc"][f"{rule}|{c}"] = {"v31": t2, **ty2}
                    r["rates"][f"{rule}|{c}"] = rates(sp, pt, tau)
                for tn, T in tests.items():
                    y = T.y.to_numpy(); st = f(T); r["tests"][f"{rule}|{tn}"] = A.fast_auc(st[y == 0], st[y == 1])
            row["arms"][name] = r
        # paired source-grouped bootstrap of Delta = Freq - B, per rule, on v3.1 total and on reenactment
        cl = A.cells(Sd); ug = np.unique(gS); pos = {g: np.where(gS == g)[0] for g in ug}
        row["boot"] = {}
        for rule in ("DS-R", "LR"):
            sb, sf = row["scores"][f"Spatial-B|{rule}"], row["scores"][f"Spatial-B-Freq|{rule}"]
            d_tot, d_ren = [], []
            for _ in range(N_BOOT):
                wts = np.zeros(len(Sd), dtype=int)
                for g in rng.choice(ug, len(ug), replace=True):
                    wts[pos[g]] += 1
                tf, yf = A.v31(sf, cl, wts); tb, yb = A.v31(sb, cl, wts)
                d_tot.append(tf - tb); d_ren.append(yf["reenactment"] - yb["reenactment"])
            row["boot"][rule] = {"total": {"mean": float(np.mean(d_tot)), "ci": [float(np.percentile(d_tot, 2.5)), float(np.percentile(d_tot, 97.5))]},
                                 "reenactment": {"mean": float(np.mean(d_ren)), "ci": [float(np.percentile(d_ren, 2.5)), float(np.percentile(d_ren, 97.5))]}}
        del row["scores"]
        rows.append(row)
        print(f"split {k}: " + " | ".join(f"{n} DS-R {row['arms'][n]['clean']['DS-R']['v31']:.4f} LR {row['arms'][n]['clean']['LR']['v31']:.4f}"
                                          for n in arms), flush=True)
    json.dump(rows, open(R / "EXP_FREQBLENDER.json", "w"), indent=1, default=float)
    write(rows, arms, a)


def write(rows, arms, a):
    names = list(arms)
    RULES = ("DS-R", "LR")

    def dev(arm, rule, field, cond="clean"):
        """Mean over splits of one development number. cond: 'clean', 'jpeg40' or 'rs06'."""
        if cond == "clean":
            return float(np.mean([r["arms"][arm]["clean"][rule][field] for r in rows]))
        return float(np.mean([r["arms"][arm]["proc"][f"{rule}|{cond}"][field] for r in rows]))

    def rate(arm, rule, which, cond="clean"):
        i = 0 if which == "fpr" else 1
        return float(np.mean([r["arms"][arm]["rates"][f"{rule}|{cond}"][i] for r in rows]))

    L = ["# Experiment 5 — FreqBlender spatial study (local reproduction)", "",
         "Two matched spatial arms with the temporal branch frozen at the reference B-augmented V-JEPA 2 LoRA "
         "checkpoint. Arms, generator provenance, checkpoint selection, acceptance criteria and the reenactment "
         "disqualifier were **frozen before the FreqBlender arm was trained**: "
         "`tbiom/docs/exp5-freqblender-preregistration.md`. DS-R is the primary fusion, logistic stacking on matched "
         "inputs the mandatory comparator; both are refitted separately per arm on the same source-grouped folds.", "",
         "**What a verdict here does and does not change.** The arms are the `sbi_arm*` family, because that recipe "
         "trains the Appearance and Off-manifold views; the deployed reference takes those views from the warm-start "
         "+ gated detector, where they are frozen. So this experiment answers whether FreqBlender helps spatial "
         "views that are being trained — the question the brief poses — and an adoption verdict would be a finding "
         "about spatial training rather than an automatic re-pointing of the deployed reference, which would need its "
         "own matched comparison. Fixed before the numbers were read.", "",
         "**This is a local reproduction of FreqBlender, not the released FreqBlender.** The released FPNet weights are "
         "behind a Baidu Pan account wall, so FPNet was trained here from the upstream implementation at its pinned "
         "revision, with a locally trained split-safe authenticity teacher; the facial-fidelity teacher is the official "
         "one shipped in the repository. Deviations are listed in the pre-registration and in "
         "`analysis/tbiom/freqblender/fpnet.py`.", "", F.CITE, "",
         "## 1. Development — the whole picture in one table (selection folds, mean of 5 splits)", "",
         "| arm | rule | clean v3.1 | swap | reenactment | in-the-wild | JPEG q40 | resize 0.60 | FPR clean | TPR clean |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n in names:
        for rule in RULES:
            L.append(f"| {n} | {rule} | {dev(n, rule, 'v31'):.4f} | {dev(n, rule, 'swap'):.4f} | "
                     f"{dev(n, rule, 'reenactment'):.4f} | {dev(n, rule, 'in_the_wild'):.4f} | "
                     f"{dev(n, rule, 'v31', 'jpeg40'):.4f} | {dev(n, rule, 'v31', 'rs06'):.4f} | "
                     f"{rate(n, rule, 'fpr'):.3f} | {rate(n, rule, 'tpr'):.3f} |")
    L += ["", "## 2. Δ = Spatial-B-Freq − Spatial-B (paired source-grouped bootstrap, 1000 draws per split)", "",
          "| rule | quantity | mean Δ | splits with CI upper bound < 0 | splits with CI lower bound > 0 | CI (split 0) |",
          "|---|---|---:|---:|---:|---|"]
    boot = {}
    for rule in RULES:
        for q in ("total", "reenactment"):
            d = [r["boot"][rule][q]["mean"] for r in rows]
            neg = sum(r["boot"][rule][q]["ci"][1] < 0 for r in rows)
            pos = sum(r["boot"][rule][q]["ci"][0] > 0 for r in rows)
            ci = rows[0]["boot"][rule][q]["ci"]
            boot[f"{rule}|{q}"] = (float(np.mean(d)), neg, pos)
            L.append(f"| {rule} | {q} | {np.mean(d):+.4f} | {neg}/5 | {pos}/5 | [{ci[0]:+.4f}, {ci[1]:+.4f}] |")

    C, Fq = "Spatial-B", "Spatial-B-Freq"
    L += ["", "## 3. Verdict against the frozen criteria", ""]
    adopt, gain_proc_any = {}, False
    for rule in RULES:
        mean_ren, neg_ren, _ = boot[f"{rule}|reenactment"]
        disq = bool(mean_ren < 0 and neg_ren >= 3)
        g_clean = dev(Fq, rule, "v31") - dev(C, rule, "v31")
        g_proc = float(np.mean([dev(Fq, rule, "v31", c) - dev(C, rule, "v31", c) for c in PROC]))
        d_fpr = rate(Fq, rule, "fpr") - rate(C, rule, "fpr")
        d_swap = dev(Fq, rule, "swap") - dev(C, rule, "swap")
        d_ren = dev(Fq, rule, "reenactment") - dev(C, rule, "reenactment")
        swap_only = d_swap > 0 and d_ren < 0
        ok = bool((not disq) and (not swap_only) and (g_clean >= 0.005 or g_proc >= 0.005)
                  and g_clean >= -0.005 and d_fpr <= 0.02)
        adopt[rule] = ok
        gain_proc_any = gain_proc_any or g_proc >= 0.005
        L += [f"**{rule}**", "",
              f"- Disqualifier (fused development reenactment regression beyond the noise band): mean Δ {mean_ren:+.4f}, "
              f"CI upper bound < 0 in {neg_ren}/5 splits → **{'DISQUALIFIED' if disq else 'not disqualified'}**. "
              f"The precedent is arm C, where synthetic pseudo-fakes collapsed fused development reenactment to 0.527.",
              f"- Swaps-alone check: clean swap Δ {d_swap:+.4f} with clean reenactment Δ {d_ren:+.4f} → "
              f"**{'fails — a swap gain bought with a reenactment drop does not qualify' if swap_only else 'passes'}**.",
              f"- Development gain: clean Δ {g_clean:+.4f}, processed Δ {g_proc:+.4f} (either ≥ +0.005 required, clean "
              f"loss ≤ 0.005); clean FPR Δ {d_fpr:+.3f} (≤ +0.02 required).",
              f"- **Inclusion: {'ADOPT' if ok else 'NOT ADOPTED'}.**", ""]
    L += [f"Compression robustness is claimed only if the processed-development criterion is met; it is "
          f"{'met' if gain_proc_any else 'not met'} here.", "",
          "## 4. Test sets — video AUROC, read AFTER the verdict above", "",
          "| test | " + " | ".join(f"{n} {rule}" for n in names for rule in RULES) + " |",
          "|---|" + "---:|" * (2 * len(names))]
    tl = list(A.TEST) + ["CDFv3 reenactment", "CDFv2 jpeg40", "CDFv2 rs06", "DFDCP jpeg40", "DFDCP rs06"]
    for tn in tl:
        L.append(f"| {tn} | " + " | ".join(f"{np.mean([r['arms'][n]['tests'][f'{rule}|{tn}'] for r in rows]):.4f}"
                                           for n in names for rule in RULES) + " |")
    L.append("| **10-set mean** | " + " | ".join(
        f"**{np.mean([np.mean([r['arms'][n]['tests'][f'{rule}|{t}'] for r in rows]) for t in A.TEST]):.4f}**"
        for n in names for rule in RULES) + " |")
    L += ["", "CDFv3 test reenactment is reported here, after the development verdict, and was never a selection "
          "criterion. DS-R weights per split and every raw number are in `EXP_FREQBLENDER.json`.", "", F.FOOT]
    (R / "EXP_FREQBLENDER.md").write_text("\n".join(L) + "\n"); print("wrote", R / "EXP_FREQBLENDER.md")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stage 6 on development, and the Stage 7 mechanism inspection.

    OMP_NUM_THREADS=8 python analysis/tbiom/grouped/development.py

The fixed matrix of thirteen arms, each fitted and scored on the same five source-grouped folds with the same grid
and objective, twice: once with every discount held at one, which isolates the grouping itself, and once with the
discount fitted per arm per fold. Paired source-group bootstrap against BOTH mandatory baselines, Dempster-R and
AVG-R, on the same draws.

The mechanism inspection asks the question the operator is supposed to answer: when the two CLIP heads agree and are
wrong while the temporal view is right, which rule follows the temporal view? Labels and generator identity are used
for analysis only and never enter inference.
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
import finish_line as F  # noqa: E402
from health import eer  # noqa: E402

R, N_BOOT = Path("tbiom/reports"), 2000
ARMS = ops.ALL_ARMS
BASELINES = ("Dempster-R", "AVG-R")
PROC = ("jpeg40", "rs06")


def mechanism(Sd, w_by_arm):
    """Counts and rescue behaviour on the selection fold, at each view's own fitting-calibrated threshold."""
    v = S.views_of(Sd)
    y = Sd.y.to_numpy()
    ff = np.where(Sd.source.to_numpy() == "FFpp_val")[0]
    p = {k: ops.p_of(o) for k, o in v.items()}
    tau = {k: float(eer(y[ff], p[k][ff])[1]) for k in p}
    ok = {k: (p[k] >= tau[k]).astype(int) == y for k in p}
    spatial_agree = (p["APP"] >= tau["APP"]) == (p["OM"] >= tau["OM"])
    cases = {"both CLIP heads wrong together, T right": spatial_agree & ~ok["APP"] & ~ok["OM"] & ok["T"],
             "T wrong, CLIP heads agree and right": spatial_agree & ok["APP"] & ok["OM"] & ~ok["T"],
             "all three right": ok["T"] & ok["APP"] & ok["OM"],
             "all three wrong": ~ok["T"] & ~ok["APP"] & ~ok["OM"]}
    out = {}
    for nm, m in cases.items():
        row = {"n": int(m.sum())}
        for arm in ("grouped-R", "Dempster-R", "AVG-R", "CAUT-R"):
            pf = ops.p_of(S.fused(Sd, arm, w_by_arm[arm]))
            t = float(eer(y[ff], pf[ff])[1])
            row[arm] = float((((pf >= t).astype(int) == y)[m]).mean()) if m.sum() else float("nan")
        out[nm] = row
    return out


def main():
    t0 = time.time()
    dev, dproc, _tests = D.load("vjepa2_lora_B_seed42")
    rows = []
    for k, Fd, Sd in D.splits(dev):
        clF, clS = A.cells(Fd), A.cells(Sd)
        row = {"split": k, "fitted": {}, "unit": {}, "w": {}, "boot": {}, "scores": {}}
        for mode in ("unit", "fitted"):
            for arm in ARMS:
                w = S.ONE if mode == "unit" else S.fit_w(Fd, arm, clF)
                r, p, _o = S.endpoints(Sd, arm, w, clS)
                for c, P in dproc.items():
                    pt = P.set_index("v").loc[Sd.v].reset_index()
                    pt["source"] = Sd.source.to_numpy()
                    r[c] = A.v31(ops.p_of(S.fused(pt, arm, w)), A.cells(pt))[0]
                row[mode][arm] = r
                if mode == "fitted":
                    row["w"][arm] = list(w)
                    row["scores"][arm] = p
        g = A.groups_for(Sd)
        draws = S.group_boot(g, N_BOOT, 20260922 + k)
        cache = {}

        def v31w(arm, i):
            key = (arm, i)
            if key not in cache:
                cache[key] = A.v31(row["scores"][arm], clS, draws[i])[0]
            return cache[key]

        for arm in ARMS:
            row["boot"][arm] = {}
            for b in BASELINES:
                d = [v31w(arm, i) - v31w(b, i) for i in range(N_BOOT)]
                row["boot"][arm][b] = {"delta": float(np.mean(d)),
                                       "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
        row["mechanism"] = mechanism(Sd, row["w"])
        row.pop("scores")
        rows.append(row)
        print(f"  split {k} done ({time.time() - t0:.0f}s)", flush=True)
    json.dump(rows, open(R / "EXP_GROUPED_DEVELOPMENT.json", "w"), indent=1, default=float)
    write(rows, time.time() - t0)


def write(rows, secs):
    m = lambda arm, key, mode="fitted": float(np.mean([r[mode][arm][key] for r in rows]))
    L = ["# Grouped fusion on development (Stage 6)", "", __doc__.split("\n\n", 2)[2].strip(), "",
         f"Wall clock {secs / 60:.1f} minutes, CPU, cached opinions. Pre-registration: "
         "`tbiom/configs/grouped_fusion_preregister.yaml`. Five folds, {N} paired bootstrap draws per contrast."
         .replace("{N}", str(N_BOOT)), "",
         "## 1. Fitted discount (the deployed setting)", "",
         "| arm | v3.1 clean | swap | reenactment | in-the-wild | JPEG q40 | resize 0.60 | FPR | TPR | NLL | Brier | "
         "ECE | vacuity err-AUROC | E-AURC (vacuity) |", "|---" * 14 + "|"]
    for a in ARMS:
        tag = "**" if a == "grouped-R" else ""
        note = " *(baseline)*" if a in BASELINES else ""
        L.append(f"| {tag}{a}{tag}{note} | {m(a, 'v31'):.4f} | {m(a, 'swap'):.4f} | {m(a, 'reenactment'):.4f} | "
                 f"{m(a, 'in_the_wild'):.4f} | {m(a, 'jpeg40'):.4f} | {m(a, 'rs06'):.4f} | {m(a, 'fpr'):.3f} | "
                 f"{m(a, 'tpr'):.3f} | {m(a, 'nll'):.4f} | {m(a, 'brier'):.4f} | {m(a, 'ece'):.4f} | "
                 f"{m(a, 'err_auroc_vacuity'):.4f} | {m(a, 'eaurc_vacuity'):.5f} |")

    L += ["", "## 2. All discounts held at one — the grouping on its own", "",
          "| arm | v3.1 clean | reenactment | JPEG q40 | vacuity err-AUROC |", "|---|---:|---:|---:|---:|"]
    for a in ARMS:
        L.append(f"| {a} | {m(a, 'v31', 'unit'):.4f} | {m(a, 'reenactment', 'unit'):.4f} | "
                 f"{m(a, 'jpeg40', 'unit'):.4f} | {m(a, 'err_auroc_vacuity', 'unit'):.4f} |")

    L += ["", "## 3. Paired bootstrap against both mandatory baselines", "",
          "Source groups resampled with every linked clean and processed copy together; the same draws for every arm.", "",
          "| arm | Δ vs Dempster-R (95% CI) | splits CI>0 | Δ vs AVG-R (95% CI) | splits CI>0 |",
          "|---|---:|---:|---:|---:|"]
    for a in ARMS:
        cells = []
        for b in BASELINES:
            d = [r["boot"][a][b] for r in rows]
            lo = float(np.mean([x["ci"][0] for x in d])); hi = float(np.mean([x["ci"][1] for x in d]))
            cells += [f"{np.mean([x['delta'] for x in d]):+.4f} [{lo:+.4f}, {hi:+.4f}]",
                      f"{sum(x['ci'][0] > 0 for x in d)}/5"]
        L.append(f"| {a} | " + " | ".join(cells) + " |")

    # pre-registered criteria, applied
    base = {b: m(b, "v31") for b in BASELINES}
    L += ["", "## 4. Pre-registered criteria, applied to every arm", "",
          "| arm | clean gain vs both baselines >= 0.002 | both CIs above 0 | safeguards | vacuity guard | verdict |",
          "|---|---|---|---|---|---|"]
    dref = m("Dempster-R", "err_auroc_vacuity")
    verdicts = {}
    for a in ARMS:
        gain = min(m(a, "v31") - base[b] for b in BASELINES)
        cis = all(sum(r["boot"][a][b]["ci"][0] > 0 for r in rows) >= 3 for b in BASELINES)
        safe = all(m(a, k) - max(m(b, k) for b in BASELINES) >= -0.005
                   for k in ("v31", "jpeg40", "rs06", "reenactment"))
        vac = m(a, "err_auroc_vacuity") - dref >= -0.005
        ok = gain >= 0.002 and cis and safe and vac
        verdicts[a] = {"gain": gain, "cis": cis, "safe": safe, "vacuity": vac, "pass": ok}
        L.append(f"| {a} | {gain:+.4f} {'yes' if gain >= 0.002 else 'no'} | {'yes' if cis else 'no'} | "
                 f"{'ok' if safe else 'breached'} | {m(a, 'err_auroc_vacuity') - dref:+.4f} "
                 f"{'ok' if vac else '**disqualified**'} | {'**PASS**' if ok else 'no'} |")

    L += ["", "## 5. Mechanism — who follows the temporal view when both CLIP heads are wrong together", "",
          "Accuracy within each case, on the selection folds, each rule at its own FF++-val EER threshold.", "",
          "| case | videos | **grouped-R** | Dempster-R | AVG-R | CAUT-R |", "|---|---:|---:|---:|---:|---:|"]
    for nm in rows[0]["mechanism"]:
        g = lambda a: float(np.mean([r["mechanism"][nm][a] for r in rows]))
        n = float(np.mean([r["mechanism"][nm]["n"] for r in rows]))
        L.append(f"| {nm} | {n:.0f} | **{g('grouped-R'):.4f}** | {g('Dempster-R'):.4f} | {g('AVG-R'):.4f} | "
                 f"{g('CAUT-R'):.4f} |")
    mk = "both CLIP heads wrong together, T right"
    gg, dd = (float(np.mean([r["mechanism"][mk][a] for r in rows])) for a in ("grouped-R", "Dempster-R"))
    L += ["", ("The first row is the mechanism the operator is built for, and grouped-R does rescue more of it than "
               f"Dempster-R ({gg:.4f} against {dd:.4f})." if gg > dd else
               "The first row is the mechanism the operator is built for, and grouped-R does NOT rescue more of it "
               f"than Dempster-R ({gg:.4f} against {dd:.4f}) — the mechanism claim is not supported on development."),
          "", "## 6. Fitted discounts per fold", "", "| arm | " + " | ".join(f"split {r['split']}" for r in rows) + " |",
          "|---" * (len(rows) + 1) + "|"]
    for a in ARMS:
        L.append(f"| {a} | " + " | ".join("(" + ", ".join(f"{x:g}" for x in r["w"][a]) + ")" for r in rows) + " |")
    L += ["", "Weights are (T, APP, OM) and are refitted per arm per fold; no arm inherits another's weights. "
          "Development results are exploratory in the sense the pre-registration records: they select, they do not "
          "confirm.", ""]
    (R / "EXP_GROUPED_DEVELOPMENT.md").write_text("\n".join(L) + "\n")
    json.dump(verdicts, open(R / "EXP_GROUPED_DEVELOPMENT_VERDICTS.json", "w"), indent=1, default=float)
    print("wrote EXP_GROUPED_DEVELOPMENT.md")


if __name__ == "__main__":
    main()

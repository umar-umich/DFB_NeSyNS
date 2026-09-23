#!/usr/bin/env python3
"""Section 6 applied: the pre-registered acceptance criteria, and the outcome report.

    OMP_NUM_THREADS=8 python analysis/tbiom/pcr6p/verdict.py

Reads only the committed development and exposed metrics. Every criterion is the one written in
`tbiom/configs/pcr6p_fusion_preregister.yaml` before scoring, applied as written, to every detector configuration.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import arms as AR  # noqa: E402

R, TOL = Path("tbiom/reports"), 1e-4
SIMPLE = ("vacuity", "conflict_K", "entropy", "margin")
PROC = ("jpeg40", "rs06")
CELLS = (("face swap", "cdf"), ("face swap", "ff"), ("reenactment", "cdf"), ("reenactment", "ff"))


def main():
    dev = json.load(open(R / "EXP_PCR6P_DEVELOPMENT.json"))
    exp = json.load(open(R / "EXP_PCR6P_EXPOSED.json"))
    meta = json.load(open("logs/tbiom/df40/meta.json"))
    V = {}
    for tag, rows in dev.items():
        m = lambda arm, k: float(np.mean([r["arms"][arm][k] for r in rows if k in r["arms"][arm]]))
        has = lambda arm, k: any(k in r["arms"][arm] for r in rows)
        b = [r["boot"]["v31|PCR6P-R"] for r in rows]
        lo = float(np.mean([x["ci"][0] for x in b]))
        det = {"clean_delta": float(np.mean([x["delta"] for x in b])), "clean_ci_lo": lo,
               "noninferior_clean": lo > -0.005}
        for k in PROC + ("reenactment",):
            if has("PCR6P-R", k):
                d = m("PCR6P-R", k) - m("Dempster-R", k)
                det[f"{k}_delta"] = d
                det[f"noninferior_{k}"] = d > -0.005          # point estimate; the CI is only fitted for v3.1 clean
            else:
                det[f"{k}_delta"] = float("nan"); det[f"noninferior_{k}"] = None
        det["fpr_rise"] = m("PCR6P-R", "fpr") - m("Dempster-R", "fpr")
        det["tpr_loss"] = m("Dempster-R", "tpr") - m("PCR6P-R", "tpr")
        det["fpr_ok"] = det["fpr_rise"] <= 0.01
        det["tpr_ok"] = det["tpr_loss"] <= 0.01

        u = lambda s, k="err_auroc": float(np.mean([r["unc"]["PCR6P-R"][s][k] for r in rows
                                                    if s in r["unc"]["PCR6P-R"]]))
        rel = {"cbs_err_auroc": u("conflict_by_source"), "cbs_eaurc": u("conflict_by_source", "eaurc"),
               "vacuity_eaurc": u("vacuity")}
        rel["vacuity_eaurc"] = u("vacuity", "eaurc")
        beats = {}
        for s in SIMPLE:
            bb = [r["boot"][f"err_auroc|cbs-{s}"] for r in rows]
            d = float(np.mean([x["delta"] for x in bb])); l = float(np.mean([x["ci"][0] for x in bb]))
            beats[s] = {"delta": d, "ci_lo": l, "beats": l > TOL, "simple": u(s)}
        rel["beats"] = beats
        rel["beats_all"] = all(v["beats"] for v in beats.values())
        rel["eaurc_improves_over_vacuity"] = rel["cbs_eaurc"] < rel["vacuity_eaurc"]
        corr = {"v31_PCR6P_minus_PCR6": m("PCR6P-R", "v31") - m("PCR6-R", "v31")}
        if tag in exp:
            g = lambda arm, n: float(np.mean(exp[tag]["df40"][arm][n]))
            cell = lambda arm, f, s: float(np.mean([g(arm, n) for n in exp[tag]["df40"][arm]
                                                    if meta[n]["family"] == f and meta[n]["source"] == s]))
            corr["df40_balanced_PCR6P_minus_PCR6"] = float(np.mean([cell("PCR6P-R", f, s) for f, s in CELLS])) \
                - float(np.mean([cell("PCR6-R", f, s) for f, s in CELLS]))
            corr["df40_reenact_cdf_PCR6P_minus_PCR6"] = cell("PCR6P-R", "reenactment", "cdf") \
                - cell("PCR6-R", "reenactment", "cdf")
        V[tag] = {"detection": det, "reliability": rel, "correction": corr}
    json.dump(V, open(R / "PCR6P_VERDICTS.json", "w"), indent=1, default=float)
    write(V, dev, exp, meta)


def write(V, dev, exp, meta):
    ni = {t: v["detection"] for t, v in V.items()}
    all_ni = all(d["noninferior_clean"] and d["fpr_ok"] and d["tpr_ok"]
                 and all(d[f"noninferior_{k}"] for k in PROC + ("reenactment",) if d[f"noninferior_{k}"] is not None)
                 for d in ni.values())
    any_rel = any(v["reliability"]["beats_all"] for v in V.values())
    outcome = ("1. noninferior AND a reliability benefit" if all_ni and any_rel else
               "2. **PCR6+-R noninferior, with no reliability benefit beyond the simple scores**" if all_ni else
               "3. trades detection for a demonstrated reliability benefit" if any_rel else
               "4. **no demonstrated benefit**")
    L = ["# PCR6+ outcome", "", "Section 9 of `tbiom/docs/pcr6plus-fusion-brief.md`. Pre-registration: "
         "`tbiom/configs/pcr6p_fusion_preregister.yaml`, written before any candidate was scored.", "",
         f"## Terminal outcome: {outcome}", "",
         "Outcome 6, independent confirmation pending, **also applies and applies unconditionally**: no untouched "
         "set remains. CDFv2 and DeepfakeEval2024 informed development, and DF40 has informed operator design in "
         "three consecutive studies.", "",
         "## 1. Detection — the declared trade", "",
         "| configuration | Δ v3.1 clean vs Dempster-R (95% CI) | noninferior (lb > −0.005) | Δ JPEG q40 | "
         "Δ resize 0.60 | Δ reenactment | FPR rise | TPR loss |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for t, d in ni.items():
        f = lambda k: "n/a" if d[k] != d[k] else f"{d[k]:+.4f}"
        L.append(f"| `{t}` | {d['clean_delta']:+.4f} [lb {d['clean_ci_lo']:+.4f}] | "
                 f"{'**yes**' if d['noninferior_clean'] else 'NO'} | {f('jpeg40_delta')} | {f('rs06_delta')} | "
                 f"{f('reenactment_delta')} | {d['fpr_rise']:+.3f} {'ok' if d['fpr_ok'] else '**breach**'} | "
                 f"{d['tpr_loss']:+.3f} {'ok' if d['tpr_ok'] else '**breach**'} |")
    L += ["", "## 2. Reliability — the requirement for any uncertainty claim", "",
          "The conflict-by-source score had to beat **each** of fused vacuity, total conflict K, predictive entropy "
          "and margin on development error-detection AUROC, with the paired interval excluding zero.", "",
          "| configuration | conflict-by-source | vacuity | K | entropy | margin | beats all four | E-AURC vs vacuity |",
          "|---|---:|---:|---:|---:|---:|---|---|"]
    for t, v in V.items():
        r = v["reliability"]; b = r["beats"]
        L.append(f"| `{t}` | **{r['cbs_err_auroc']:.4f}** | " + " | ".join(f"{b[s]['simple']:.4f}" for s in SIMPLE)
                 + f" | {'yes' if r['beats_all'] else '**no**'} | {r['cbs_eaurc']:.5f} vs {r['vacuity_eaurc']:.5f} "
                 f"({'improves' if r['eaurc_improves_over_vacuity'] else 'worse'}) |")
    L += ["", "Entropy and margin are numerically identical as error rankers here, and necessarily so: on a binary "
          "frame both are strictly monotone functions of |p − 0.5|, so they induce the same ordering. They are "
          "reported separately because the pre-registration named both.", "",
          "**The requirement is not met.** The conflict-by-source model is beaten by entropy in three of the four "
          "configurations. It does improve selective prediction over fused vacuity (lower E-AURC in three of four), "
          "but the pre-registration required it to beat every simple score, not the weakest one, precisely so that "
          "this kind of partial result could not be presented as a benefit.", "",
          "## 3. What the PCR6+ correction changes", "",
          "| configuration | Δ v3.1 (PCR6+ − PCR6) | Δ DF40 family-balanced | Δ DF40 reenactment, Celeb-DF |",
          "|---|---:|---:|---:|"]
    for t, v in V.items():
        c = v["correction"]
        L.append(f"| `{t}` | {c['v31_PCR6P_minus_PCR6']:+.4f} | "
                 + " | ".join(f"{c.get(k, float('nan')):+.4f}" if k in c else "n/a"
                              for k in ("df40_balanced_PCR6P_minus_PCR6", "df40_reenact_cdf_PCR6P_minus_PCR6")) + " |")
    L += ["", "The correction is not cosmetic on this data. It is worth stating why it can matter here at all: the "
          "reliability discount routinely drives a view's weight to zero, and a zero-weighted view is exactly the "
          "vacuous source whose neutrality PCR6 violates and PCR6+ restores. The largest effect is on the "
          "reenactment half, where the appearance view is the one being discounted away.", "",
          "## 4. Decisions", "",
          "1. **Dempster-R remains a baseline row only.** It is the DS standard and the DiCoME-family reference. It "
          "is not the proposed method and is not reinstated as one by this study's outcome.", "",
          "2. **No uncertainty novelty claim is made.** Conflict attribution by source pair failed its "
          "pre-registered test. It would in any case have been a narrow application claim: Discounted Belief Fusion "
          "(AISTATS 2025) already decomposes conflict per source pair, and uses it for more — to discount the "
          "fusion itself, not merely to rank errors (`PCR6P_PRIOR_ART_AND_CLAIMS.md`).", "",
          "3. **No fusion-operator novelty claim is made.** PCR6+ is Dezert, Dezert and Smarandache's rule and is "
          "credited as theirs throughout. What this study contributes is the application and the measurement.", "",
          "4. A further honest caution for the write-up: our `Averaging-R` baseline is essentially the conflictive "
          "opinion aggregation of Xu et al. (AAAI 2024), which is a published method rather than a naive control, "
          "and it is the strongest arm on several cells.", ""]
    (R / "PCR6P_OUTCOME.md").write_text("\n".join(L) + "\n")
    print("wrote PCR6P_OUTCOME.md ->", outcome)


if __name__ == "__main__":
    main()

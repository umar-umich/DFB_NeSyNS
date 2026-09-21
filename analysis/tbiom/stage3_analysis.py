#!/usr/bin/env python3
"""Stage 3: both routes UMAR chose, evaluated on the same folds and the same bootstrap draws (cached, CPU).

    OMP_NUM_THREADS=8 python analysis/tbiom/stage3_analysis.py

Pre-registration: tbiom/docs/stage3-composition-preregistration.md.
  Route 2 -- composition on the deployed recipe: UB (warm+gate, views unfrozen, arm B) vs UE (same, arm E), and UE
             against the deployed checkpoint, which is the adoption decision.
  Route 3 -- arm E adopted as an override: E seeds 42 / 1337 / 7 each against the deployed checkpoint, per seed.
Every arm's Appearance and Off-manifold views are fused with the frozen reference temporal branch; DS-R and the
logistic comparator are refitted per arm on the same 5 source-grouped folds. Development only; no test set is used.
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

LOG, R = Path("logs/tbiom/stage3"), Path("tbiom/reports")
PROC, RULES, N_BOOT = ("jpeg40", "rs06"), ("DS-R", "LR"), 1000


def sel(tag):
    return (LOG / f"{tag}_DONE").read_text().strip()


def arms():
    return {"deployed (warm+gate e19, views frozen)": ("warmgate_e19", "warmgate_sel", "warmgate"),
            "UB (warm+gate, views unfrozen, arm B)": (sel("wgUB"), "wgUB_sel", "wgUB"),
            "UE (warm+gate, views unfrozen, arm E)": (sel("wgUE"), "wgUE_sel", "wgUE"),
            "E seed 42 (sbi_armE)": ("sbiarmE_e02", "fbE_sel", "fbE"),
            "E seed 1337": (sel("sbiarmEs1337"), "sbiarmEs1337_sel", "sbiarmEs1337"),
            "E seed 7": (sel("sbiarmEs7"), "sbiarmEs7_sel", "sbiarmEs7")}


PAIRS = {"route 2 — composition on the deployed recipe (UE − UB)": ("UE (warm+gate, views unfrozen, arm E)", "UB (warm+gate, views unfrozen, arm B)"),
         "route 2 — ADOPTION (UE − deployed)": ("UE (warm+gate, views unfrozen, arm E)", "deployed (warm+gate e19, views frozen)"),
         "route 2 — unfreezing alone (UB − deployed)": ("UB (warm+gate, views unfrozen, arm B)", "deployed (warm+gate e19, views frozen)"),
         "route 3 — E seed 42 − deployed": ("E seed 42 (sbi_armE)", "deployed (warm+gate e19, views frozen)"),
         "route 3 — E seed 1337 − deployed": ("E seed 1337", "deployed (warm+gate e19, views frozen)"),
         "route 3 — E seed 7 − deployed": ("E seed 7", "deployed (warm+gate e19, views frozen)")}


def main():
    spec = arms(); data = {n: E.load_arm(*t) for n, t in spec.items()}
    base = data["deployed (warm+gate e19, views frozen)"][0]; rng = np.random.default_rng(20260921); rows = []
    for k, Fd, Sd in D.splits(base):
        vs, gS, cl = Sd.v.to_numpy(), A.groups_for(Sd), A.cells(Sd); sc, row = {}, {"arms": {}, "pairs": {}}
        for name, (dev, dproc, tests) in data.items():
            fd = dev.set_index("v").loc[Fd.v].reset_index(); fd["source"] = Fd.source.to_numpy()
            sd = dev.set_index("v").loc[vs].reset_index(); sd["source"] = Sd.source.to_numpy()
            rules, params, _ = F.rules_for(fd, E.VIEWS); r = {"w": params["w"]}
            for rule, f in rules.items():
                s = f(sd); tot, ty = A.v31(s, cl); tau = F.tau_of(s, sd); fpr, tpr = E.rates(s, sd, tau)
                r[f"{rule}|clean"] = {"v31": tot, **ty, "fpr": fpr, "tpr": tpr}; sc[(name, rule, "clean")] = s
                for c, P in dproc.items():
                    pt = P.set_index("v").loc[vs].reset_index(); pt["source"] = Sd.source.to_numpy()
                    sp = f(pt); r[f"{rule}|{c}"] = {"v31": A.v31(sp, cl)[0]}; sc[(name, rule, c)] = sp
            row["arms"][name] = r
        ug = np.unique(gS); pos = {g: np.where(gS == g)[0] for g in ug}
        draws = []
        for _ in range(N_BOOT):
            w = np.zeros(len(Sd), dtype=int)
            for g in rng.choice(ug, len(ug), replace=True):
                w[pos[g]] += 1
            draws.append(w)
        cache = {}
        def v(name, rule, c, i):                                            # v3.1 total and types on draw i, memoised
            key = (name, rule, c, i)
            if key not in cache:
                cache[key] = A.v31(sc[(name, rule, c)], cl, draws[i])
            return cache[key]
        for pname, (a, b) in PAIRS.items():
            row["pairs"][pname] = {}
            for rule in RULES:
                dc, dp, dr = [], [], []
                for i in range(N_BOOT):
                    ta, ya = v(a, rule, "clean", i); tb, yb = v(b, rule, "clean", i)
                    dc.append(ta - tb); dr.append(ya["reenactment"] - yb["reenactment"])
                    dp.append(np.mean([v(a, rule, c, i)[0] - v(b, rule, c, i)[0] for c in PROC]))
                q = lambda d: [float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
                row["pairs"][pname][rule] = {"clean": q(dc), "processed": q(dp), "reenactment": q(dr),
                                             "fpr": row["arms"][a][f"{rule}|clean"]["fpr"] - row["arms"][b][f"{rule}|clean"]["fpr"]}
        rows.append(row); print(f"split {k} done", flush=True)
    json.dump(rows, open(R / "EXP_COMBINE_CONFIRM.json", "w"), indent=1, default=float)
    write(rows, spec)


def write(rows, spec):
    m = lambda n, key, f="v31": float(np.mean([r["arms"][n][key][f] for r in rows]))
    L = ["# Stage 3 — combine and confirm (both routes)", "", __doc__.split("\n\n", 1)[1].strip(), "",
         "Selected checkpoints: " + "; ".join(f"{n} → `{t[0]}`" for n, t in spec.items()) + ".", "",
         "## 1. Development, selection folds (mean of 5 splits)", "",
         "| arm | rule | clean | swap | reenactment | in-the-wild | JPEG q40 | resize 0.60 | FPR own τ | TPR own τ |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n in spec:
        for rule in RULES:
            c = f"{rule}|clean"
            L.append(f"| {n} | {rule} | {m(n, c):.4f} | {m(n, c, 'swap'):.4f} | {m(n, c, 'reenactment'):.4f} | "
                     f"{m(n, c, 'in_the_wild'):.4f} | {m(n, f'{rule}|jpeg40'):.4f} | {m(n, f'{rule}|rs06'):.4f} | "
                     f"{m(n, c, 'fpr'):.3f} | {m(n, c, 'tpr'):.3f} |")
    L += ["", "## 2. Paired comparisons (source-grouped bootstrap, 1000 draws per split, the same draws for every pair)", "",
          "Criteria (pre-registered, applied to every pair for comparability; binding only for the route-2 adoption row): "
          "processed Δ ≥ +0.005 with CI > 0 in ≥ 3/5 splits; clean Δ ≥ −0.005; reenactment not significantly negative "
          "(CI < 0 in ≥ 3/5 rejects); clean FPR Δ ≤ +0.02 at each arm's own τ.", "",
          "| comparison | rule | clean Δ | processed Δ (CI > 0) | reenactment Δ (CI < 0) | FPR Δ | meets criteria |", "|---|---|---:|---:|---:|---:|---|"]
    verdict = {}
    for pname in rows[0]["pairs"]:
        for rule in RULES:
            b = [r["pairs"][pname][rule] for r in rows]
            c = np.mean([x["clean"][0] for x in b]); p = np.mean([x["processed"][0] for x in b]); pp = sum(x["processed"][1] > 0 for x in b)
            re_ = np.mean([x["reenactment"][0] for x in b]); rn = sum(x["reenactment"][2] < 0 for x in b); fp = np.mean([x["fpr"] for x in b])
            ok = p >= 0.005 and pp >= 3 and c >= -0.005 and not (re_ < 0 and rn >= 3) and fp <= 0.02
            verdict[(pname, rule)] = ok
            L.append(f"| {pname} | {rule} | {c:+.4f} | {p:+.4f} ({pp}/5) | {re_:+.4f} ({rn}/5) | {fp:+.3f} | {'**yes**' if ok else 'no'} |")
    adopt = verdict[("route 2 — ADOPTION (UE − deployed)", "DS-R")]
    seeds = [pn for pn in rows[0]["pairs"] if pn.startswith("route 3")]
    sp = lambda q: [float(np.mean([r["pairs"][pn]["DS-R"][q][0] for r in rows])) for pn in seeds]
    L += ["", "## 3. Verdicts", "",
          f"**Route 2 (pre-registered adoption, DS-R): UE {'REPLACES' if adopt else 'does NOT replace'} the deployed spatial views.**"
          + (" Per the pre-registration it must now be confirmed across training seeds (1337, 7) before it is reported as adopted."
             if adopt else " The deployed system stays unchanged under the pre-registered rule."), "",
          "**Route 3 (override, reported as measured):** arm E − deployed under DS-R across seeds 42 / 1337 / 7 — "
          f"clean {', '.join(f'{x:+.4f}' for x in sp('clean'))} (spread {max(sp('clean')) - min(sp('clean')):.4f}); "
          f"processed {', '.join(f'{x:+.4f}' for x in sp('processed'))}; "
          f"reenactment {', '.join(f'{x:+.4f}' for x in sp('reenactment'))}. "
          "This route overrides the frozen criteria at the authors' request and is labelled as such wherever it is used.",
          "", "One seed per route-2 arm; three seeds for route 3. Development only — no test set entered any row."]
    (R / "EXP_COMBINE_CONFIRM.md").write_text("\n".join(L) + "\n"); print("wrote", R / "EXP_COMBINE_CONFIRM.md")


if __name__ == "__main__":
    main()

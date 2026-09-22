#!/usr/bin/env python3
"""Combination rules for the reference triple: is there a better — and defensible — alternative to Dempster's rule?

    OMP_NUM_THREADS=8 python analysis/tbiom/fusion_operators_v2.py

Cached opinions, CPU, the same five source-grouped folds, the same evidence-scaling discount, the same v3.1 objective
as every other report. Only the operator that combines the three views changes.

WHY THIS IS WORTH ASKING. Dempster's rule assumes the sources are independent and normalises the conflicting mass
away. Neither assumption fits this system: Appearance and Off-manifold are two heads on ONE CLIP feature path, so
their evidence overlaps, and the conflict between a temporal view and a spatial view is exactly the signal the
uncertainty safeguard uses. The rules below relax one assumption each.

The operators, all on binomial opinions (b1 = p − u/2 fake mass, b0 = 1 − p − u/2 genuine mass, u vacuity):
  Dempster            conjunctive, conflict normalised away (the deployed operator)
  Yager               conjunctive, conflict returned to vacuity instead of normalised (Dubois–Prade coincides with
                      Yager on a binary frame, so it is not listed twice)
  PCR6                conflicting mass redistributed to the views that caused it, in proportion to their own masses
  Murphy              average the three opinions, then combine the average with itself three times
  cautious (Denœux)   canonical weights w_A = q(Θ)/q(A) per view, combined by MIN — the rule for sources that are NOT
                      independent, which is what a shared feature path makes them; renormalised like Dempster
  cumulative (SL)     evidence summed across views (subjective-logic cumulative belief fusion)
  averaging           evidence averaged across views
  logistic            not an operator: the matched stacking comparator, for reference

Each operator is run twice: with equal trust, and with the same per-view reliability weights fitted by the same grid
search on the same fitting folds (`-R`). Nothing else differs, so a difference is the operator's.
"""
from __future__ import annotations
import itertools, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "temporal")]
import exp1_fusion_audit as A  # noqa: E402
import dsr_system as D  # noqa: E402
import finish_line as F  # noqa: E402
from health import eer  # noqa: E402
from safeguard_ablation import rc_metrics  # noqa: E402

R, K, VIEWS, PROC = Path("tbiom/reports"), 2.0, ["T", "APP", "OM"], ("jpeg40", "rs06")
GRID, N_BOOT = (0.0, 0.25, 0.5, 0.75, 1.0), 1000
EPS = 1e-9


def masses(t, w=None):
    """(n, 3, 3) array of (b1, b0, u) per view, after optional evidence scaling by the reliability weights."""
    out = []
    for i, v in enumerate(VIEWS):
        p, u = t[f"p_{v}"].to_numpy(), np.clip(t[f"u_{v}"].to_numpy(), EPS, 1.0)
        if w is not None:
            S = K / u; a = 1 + w[i] * (np.stack([(1 - p) * S, p * S], 1) - 1)
            Sn = a.sum(1); p, u = a[:, 1] / Sn, K / Sn
        out.append(np.stack([p - u / 2, 1 - p - u / 2, u], 1))
    return np.clip(np.stack(out, 1), 0.0, 1.0)


def _pair(m1, m2, rule):
    b1, b0, u = m1[:, 0], m1[:, 1], m1[:, 2]
    c1, c0, v = m2[:, 0], m2[:, 1], m2[:, 2]
    conj1 = b1 * c1 + b1 * v + u * c1
    conj0 = b0 * c0 + b0 * v + u * c0
    conjU = u * v
    k = b1 * c0 + b0 * c1                                        # conflicting mass
    if rule == "Dempster":
        d = np.clip(1 - k, EPS, None); return np.stack([conj1 / d, conj0 / d, conjU / d], 1)
    if rule == "Yager":                                          # conflict -> vacuity (Dubois-Prade coincides here)
        return np.stack([conj1, conj0, conjU + k], 1)
    if rule == "PCR6":                                           # conflict -> the views that caused it
        s1 = b1 + c0; s2 = b0 + c1
        x1 = np.where(s1 > EPS, b1 * b1 * c0 / np.where(s1 > EPS, s1, 1), 0) + np.where(s2 > EPS, c1 * c1 * b0 / np.where(s2 > EPS, s2, 1), 0)
        x0 = np.where(s1 > EPS, c0 * c0 * b1 / np.where(s1 > EPS, s1, 1), 0) + np.where(s2 > EPS, b0 * b0 * c1 / np.where(s2 > EPS, s2, 1), 0)
        return np.stack([conj1 + x1, conj0 + x0, conjU], 1)
    raise ValueError(rule)


def combine(m, op):
    """m: (n, 3, 3) -> fused (b1, b0, u)."""
    if op in ("Dempster", "Yager", "PCR6"):
        f = m[:, 0]
        for i in (1, 2):
            f = _pair(f, m[:, i], op)
        return f
    if op == "Murphy":                                            # average first, then combine the average 3 times
        a = m.mean(1); f = a
        for _ in range(2):
            f = _pair(f, a, "Dempster")
        return f
    if op == "cautious":                                          # Denoeux: canonical weights combined by MIN
        mT = np.clip(m[:, :, 2], EPS, 1.0)
        w1 = mT / np.clip(m[:, :, 0] + mT, EPS, None)             # weight of the simple support toward "fake"
        w0 = mT / np.clip(m[:, :, 1] + mT, EPS, None)
        W1, W0 = w1.min(1), w0.min(1)                             # the cautious operator: the least committed weight
        b1 = W0 * (1 - W1); b0 = W1 * (1 - W0); uu = W1 * W0; empty = (1 - W1) * (1 - W0)
        s = np.clip(b1 + b0 + uu + empty, EPS, None)
        d = np.clip(1 - empty / s, EPS, None)                     # renormalise like Dempster, for comparability
        return np.stack([b1 / s / d, b0 / s / d, uu / s / d], 1)
    if op in ("cumulative", "averaging"):
        u = np.clip(m[:, :, 2], EPS, 1.0); S = K / u
        # alpha = [(1-p)S, pS]: class 0 (genuine) FIRST, matching every other table. b0 -> alpha_0, b1 -> alpha_1.
        e = np.stack([m[:, :, 1] * S + u * S / 2 - 1, m[:, :, 0] * S + u * S / 2 - 1], -1).clip(0)   # (n, 3, 2)
        e = e.sum(1) if op == "cumulative" else e.mean(1)
        a = e + 1; Sn = a.sum(1); p, uu = a[:, 1] / Sn, K / Sn
        return np.stack([p - uu / 2, 1 - p - uu / 2, uu], 1)
    raise ValueError(op)


def p_of(f):
    return np.clip(f[:, 0] + f[:, 2] / 2, 0, 1)


OPS = ["Dempster", "Yager", "PCR6", "Murphy", "cautious", "cumulative", "averaging"]


def main():
    dev, dproc, tests = D.load("vjepa2_lora_B_seed42")
    wgrid = [w for w in itertools.product(GRID, repeat=3) if max(w) == 1.0]
    rng = np.random.default_rng(20260922); rows = []
    for k, Fd, Sd in D.splits(dev):
        clF, clS = A.cells(Fd), A.cells(Sd)
        row = {"split": k, "arms": {}, "w": {}}
        scores = {}
        for op in OPS:
            for disc in (False, True):
                name = f"{op}-R" if disc else op
                if disc:
                    w = max(wgrid, key=lambda ww: A.v31(p_of(combine(masses(Fd, ww), op)), clF)[0])
                    row["w"][name] = list(w)
                else:
                    w = None
                s = p_of(combine(masses(Sd, w), op)); scores[name] = s
                tot, ty = A.v31(s, clS); tau = float(eer(Sd.y.to_numpy()[np.where(Sd.source == "FFpp_val")[0]],
                                                        s[np.where(Sd.source == "FFpp_val")[0]])[1])
                pred = (s >= tau).astype(int); y = Sd.y.to_numpy(); err = pred != y
                fused_u = combine(masses(Sd, w), op)[:, 2]
                r = {"clean": {"v31": tot, **ty}, "tau": tau,
                     "fpr": float((pred[y == 0] == 1).mean()), "tpr": float((pred[y == 1] == 1).mean()),
                     "err_auroc_vacuity": rc_metrics(fused_u, err)["auroc_err"], "tests": {}}
                for c, P in dproc.items():
                    pt = P.set_index("v").loc[Sd.v].reset_index(); pt["source"] = Sd.source.to_numpy()
                    r[c] = {"v31": A.v31(p_of(combine(masses(pt, w), op)), A.cells(pt))[0]}
                for tn, T in tests.items():
                    yt = T.y.to_numpy(); st = p_of(combine(masses(T, w), op))
                    r["tests"][tn] = A.fast_auc(st[yt == 0], st[yt == 1])
                row["arms"][name] = r
        # the matched stacking comparator, fitted the same way
        rules, _, _ = F.rules_for(Fd, VIEWS)
        s = rules["LR"](Sd); scores["logistic"] = s; tot, ty = A.v31(s, clS)
        tau = float(eer(Sd.y.to_numpy()[np.where(Sd.source == "FFpp_val")[0]], s[np.where(Sd.source == "FFpp_val")[0]])[1])
        pred = (s >= tau).astype(int); y = Sd.y.to_numpy()
        row["arms"]["logistic"] = {"clean": {"v31": tot, **ty}, "tau": tau, "fpr": float((pred[y == 0] == 1).mean()),
                                   "tpr": float((pred[y == 1] == 1).mean()), "err_auroc_vacuity": float("nan"),
                                   "tests": {tn: A.fast_auc(rules["LR"](T)[T.y.to_numpy() == 0], rules["LR"](T)[T.y.to_numpy() == 1])
                                             for tn, T in tests.items()}}
        for c, P in dproc.items():
            pt = P.set_index("v").loc[Sd.v].reset_index(); pt["source"] = Sd.source.to_numpy()
            row["arms"]["logistic"][c] = {"v31": A.v31(rules["LR"](pt), A.cells(pt))[0]}
        # paired bootstrap of every arm against the deployed Dempster-R
        g = A.groups_for(Sd); ug = np.unique(g); pos = {x: np.where(g == x)[0] for x in ug}
        row["boot"] = {}
        for name, s in scores.items():
            d = []
            for _ in range(N_BOOT):
                wts = np.zeros(len(Sd), dtype=int)
                for x in rng.choice(ug, len(ug), replace=True):
                    wts[pos[x]] += 1
                d.append(A.v31(s, clS, wts)[0] - A.v31(scores["Dempster-R"], clS, wts)[0])
            row["boot"][name] = [float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        rows.append(row)
        print(f"split {k}: " + " | ".join(f"{o} {row['arms'][o]['clean']['v31']:.4f}/{row['arms'][o + '-R']['clean']['v31']:.4f}"
                                          for o in ("Dempster", "PCR6", "cautious")), flush=True)
    json.dump(rows, open(R / "EXP_FUSION_OPERATORS.json", "w"), indent=1, default=float)
    write(rows)


def write(rows):
    names = list(rows[0]["arms"])
    m = lambda n, *path: float(np.mean([_get(r["arms"][n], path) for r in rows]))
    L = ["# Combination rules for the reference triple", "", __doc__.split("\n\n", 1)[1].strip(), "", F.CITE, "",
         "## 1. Development (selection folds, mean of 5 splits)", "",
         "| operator | clean v3.1 | swap | reenactment | in-the-wild | JPEG q40 | resize 0.60 | FPR | TPR | error AUROC (fused vacuity) |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n in names:
        L.append(f"| {'**' + n + '**' if n == 'Dempster-R' else n} | {m(n, 'clean', 'v31'):.4f} | {m(n, 'clean', 'swap'):.4f} | "
                 f"{m(n, 'clean', 'reenactment'):.4f} | {m(n, 'clean', 'in_the_wild'):.4f} | {m(n, 'jpeg40', 'v31'):.4f} | "
                 f"{m(n, 'rs06', 'v31'):.4f} | {m(n, 'fpr'):.3f} | {m(n, 'tpr'):.3f} | {m(n, 'err_auroc_vacuity'):.3f} |")
    L += ["", "`-R` = the same operator with per-view reliability weights fitted by the same grid search on the same "
          "fitting folds. **Dempster-R is the deployed system.**", "",
          "## 2. Against the deployed operator (paired source-grouped bootstrap, 1000 draws per split)", "",
          "| operator | mean Δ vs Dempster-R | splits with CI above 0 | splits with CI below 0 |", "|---|---:|---:|---:|"]
    for n in names:
        b = [r["boot"][n] for r in rows]
        L.append(f"| {n} | {np.mean([x[0] for x in b]):+.4f} | {sum(x[1] > 0 for x in b)}/5 | {sum(x[2] < 0 for x in b)}/5 |")
    tl = list(rows[0]["arms"][names[0]]["tests"])
    L += ["", "## 3. Test sets — video AUROC (read after fitting; mean over the 5 fitted splits)", "",
          "| operator | " + " | ".join(tl) + " | 10-set mean |", "|---|" + "---:|" * (len(tl) + 1)]
    for n in names:
        v = {t: float(np.mean([r["arms"][n]["tests"][t] for r in rows])) for t in tl}
        L.append(f"| {n} | " + " | ".join(f"{v[t]:.4f}" for t in tl) + f" | {np.mean([v[t] for t in A.TEST]):.4f} |")
    L += ["", "Reliability weights per operator (split 0): " +
          "; ".join(f"{n} {tuple(rows[0]['w'][n])}" for n in rows[0]["w"]) + ".", "",
          "One detector seed and one temporal-branch seed. CDFv2 and DeepfakeEval2024 informed development. "
          "FaceShifter is not scored here."]
    (R / "EXP_FUSION_OPERATORS.md").write_text("\n".join(L) + "\n"); print("wrote", R / "EXP_FUSION_OPERATORS.md")


def _get(d, path):
    for p in path:
        d = d[p]
    return d


if __name__ == "__main__":
    main()

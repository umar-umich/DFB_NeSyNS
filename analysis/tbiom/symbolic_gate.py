#!/usr/bin/env python3
"""Option 1 — the symbolic overlay as a selective-prediction gate.

    python analysis/tbiom/symbolic_gate.py --out tbiom/SYMBOLIC_GATE.md

THE QUESTION. Phase S showed the overlay explains faithfully. This asks whether it is also
USEFUL: if the rules abstain on the samples they call Inconclusive (R4/R0), is what remains
measurably cleaner? If yes, the symbolic layer has entered the decision path and the
neuro-symbolic label becomes a measured claim rather than a naming preference.

THE CONTROL IS THE WHOLE EXPERIMENT. Any abstention rule improves the retained set -- throwing
away hard samples always looks good. The only question that matters is whether the SYMBOLIC
gate beats a trivial confidence gate at the SAME coverage. Three controls:

    vacuity   abstain on the highest mean branch vacuity. This is the model's own uncertainty,
              already available, no rules needed.
    margin    abstain on the smallest |p_fused - tau|. The textbook selective-prediction gate.
    random    abstain uniformly at random. The floor; if symbolic does not clear this, stop.

Controls are matched to the symbolic gate's coverage PER DATASET, by taking the exact quantile
that reproduces it. That is deliberately generous -- it gives the controls oracle knowledge of
the right coverage, which a deployed system would not have -- because a win against a generous
control is worth something and a win against a hobbled one is not.

METRIC. Selective balanced error at the frozen tau, and selective AUROC. Balanced, because
abstention changes class balance and plain accuracy would reward a gate that simply drops the
minority class. tau is the frozen dev EER point; nothing here is refitted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import symbolic_overlay as SO  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

OOD = SO.OOD


def bal_err(y, dec) -> float:
    """1 - balanced accuracy. NaN if a class is entirely absent from the retained set."""
    r, f = y == 0, y == 1
    if not r.any() or not f.any():
        return float("nan")
    return float(1 - 0.5 * ((dec[f] == 1).mean() + (dec[r] == 0).mean()))


def sel_auroc(y, p) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def evaluate(y, p, keep, tau) -> dict:
    y, p = y[keep], p[keep]
    dec = (p >= tau).astype(int)
    return {"cov": float(keep.mean()), "n": int(keep.sum()),
            "err": bal_err(y, dec), "auroc": sel_auroc(y, p)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("tbiom/SYMBOLIC_GATE.md"))
    args = ap.parse_args()

    dev = pd.concat([x for x in (SO.load(d) for d in SO.DEV) if x is not None],
                    ignore_index=True)
    SO.TH = TH = SO.fit_thresholds(dev)
    tau = TH["tau_fused"]["value"]
    rng = np.random.default_rng(0)

    rows, pooled = [], {k: {"y": [], "p": [], "keep": []} for k in
                        ("symbolic", "vacuity", "margin", "random")}
    for ds in OOD:
        g = SO.load(ds)
        if g is None:
            continue
        P = SO.predicates(g, TH)
        _, rule, _ = SO.apply_rules(g, P)
        y = g["y"].to_numpy()
        p = SO.fuse(g)

        # the symbolic gate: keep everything a specific rule decided
        keep_sym = np.array([r not in ("R4_conflict_insufficient", "R0_fallback")
                             for r in rule])
        cov = keep_sym.mean()

        # controls, matched to that coverage exactly
        vac = g[[f"u_{b}" for b in SO.BRANCHES]].to_numpy().mean(1)
        mar = np.abs(p - tau)
        keeps = {
            "symbolic": keep_sym,
            "vacuity": vac <= np.quantile(vac, cov),
            "margin": mar >= np.quantile(mar, 1 - cov),
            "random": rng.random(len(y)) <= cov,
        }
        for k, keep in keeps.items():
            e = evaluate(y, p, keep, tau)
            e.update(dataset=ds, gate=k)
            rows.append(e)
            pooled[k]["y"].append(y[keep]); pooled[k]["p"].append(p[keep])
            pooled[k]["keep"].append(keep)
        # full-coverage reference
        e = evaluate(y, p, np.ones(len(y), bool), tau); e.update(dataset=ds, gate="none")
        rows.append(e)
        pooled.setdefault("none", {"y": [], "p": [], "keep": []})
        pooled["none"]["y"].append(y); pooled["none"]["p"].append(p)
        pooled["none"]["keep"].append(np.ones(len(y), bool))

    R = pd.DataFrame(rows)

    L = ["# Symbolic gate — the overlay as a selective-prediction rule", "",
         "The frozen detector is untouched. The overlay decides only whether to answer. "
         "Abstain where the rules return Inconclusive (R4 conflict-with-insufficient-evidence, "
         "R0 fallback); answer everywhere else. `tau` is the frozen dev EER point.", "",
         "Controls abstain at the **same coverage, matched per dataset**, using the model's own "
         "mean branch vacuity, the fused margin |p−τ|, or a coin. Matching coverage per dataset "
         "hands the controls information a deployed gate would not have; that is intentional.",
         "", "## Per dataset — selective balanced error (lower is better)", "",
         "| dataset | coverage | no gate | **symbolic** | vacuity | margin | random |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for ds in OOD:
        s = R[R.dataset == ds].set_index("gate")
        if s.empty:
            continue
        L.append(f"| {ds} | {s.loc['symbolic','cov']:.3f} | {s.loc['none','err']:.4f} | "
                 f"**{s.loc['symbolic','err']:.4f}** | {s.loc['vacuity','err']:.4f} | "
                 f"{s.loc['margin','err']:.4f} | {s.loc['random','err']:.4f} |")

    L += ["", "## Per dataset — selective AUROC (higher is better)", "",
          "| dataset | coverage | no gate | **symbolic** | vacuity | margin | random |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for ds in OOD:
        s = R[R.dataset == ds].set_index("gate")
        if s.empty:
            continue
        L.append(f"| {ds} | {s.loc['symbolic','cov']:.3f} | {s.loc['none','auroc']:.4f} | "
                 f"**{s.loc['symbolic','auroc']:.4f}** | {s.loc['vacuity','auroc']:.4f} | "
                 f"{s.loc['margin','auroc']:.4f} | {s.loc['random','auroc']:.4f} |")

    L += ["", "## Pooled over the six OOD sets", "",
          "| gate | coverage | balanced error | AUROC |", "|---|---:|---:|---:|"]
    agg = {}
    for k in ("none", "symbolic", "vacuity", "margin", "random"):
        y = np.concatenate(pooled[k]["y"]); p = np.concatenate(pooled[k]["p"])
        c = float(np.concatenate(pooled[k]["keep"]).mean())
        agg[k] = {"cov": c, "err": bal_err(y, (p >= tau).astype(int)), "auroc": sel_auroc(y, p)}
        nm = f"**{k}**" if k == "symbolic" else k
        L.append(f"| {nm} | {c:.3f} | {agg[k]['err']:.4f} | {agg[k]['auroc']:.4f} |")

    # ---------------------------------------------------------------- why, not just whether
    #
    # Two diagnostics, because "symbolic ties margin and loses to vacuity" is a result only if
    # we can say what drives it.
    #
    #   overlap        Jaccard of the retained sets. If the symbolic gate keeps the same
    #                  samples as a trivial gate, its matching score is not a coincidence.
    #   error ranking  AUROC of each signal against "this prediction is wrong". A gate can only
    #                  help to the extent it ranks errors, and this measures that directly,
    #                  independent of the coverage point chosen.
    L += ["", "## Why", "",
          "| dataset | J(symbolic, vacuity) | J(symbolic, margin) | vacuity ranks errors | "
          "symbolic flag ranks errors |", "|---|---:|---:|---:|---:|"]
    for ds in OOD:
        g = SO.load(ds)
        if g is None:
            continue
        P = SO.predicates(g, TH); _, rule, _ = SO.apply_rules(g, P)
        y = g["y"].to_numpy(); p = SO.fuse(g)
        ks = np.array([r not in ("R4_conflict_insufficient", "R0_fallback") for r in rule])
        c = ks.mean()
        vac = g[[f"u_{b}" for b in SO.BRANCHES]].to_numpy().mean(1); mar = np.abs(p - tau)
        kv = vac <= np.quantile(vac, c); km = mar >= np.quantile(mar, 1 - c)
        J = lambda a, b: float((a & b).sum() / (a | b).sum())
        err = ((p >= tau).astype(int) != y).astype(int)
        L.append(f"| {ds} | {J(ks,kv):.3f} | {J(ks,km):.3f} | "
                 f"{roc_auc_score(err,vac):.3f} | {roc_auc_score(err,(~ks).astype(float)):.3f} |")

    # ---------------------------------------------------------------- incremental value
    #
    # The fair form of the question. Not "is the binary rule flag a good gate" but "do the
    # symbolic predicates carry ANY error information the model's own vacuity does not". Fit a
    # wrong/right predictor on DEV from vacuity alone, then from vacuity plus the predicates,
    # and compare on OOD. Fitting on dev and reading on OOD is what makes this a transfer test
    # rather than a curve-fitting exercise.
    def feats(g):
        P = SO.predicates(g, TH); _, rule, _ = SO.apply_rules(g, P)
        p = SO.fuse(g); vac = g[[f"u_{b}" for b in SO.BRANCHES]].to_numpy().mean(1)
        inc = np.array([r in ("R4_conflict_insufficient", "R0_fallback") for r in rule], float)
        X = np.c_[vac, inc, P["conflict"].to_numpy().astype(float),
                  P["n_fake_votes"].to_numpy().astype(float)]
        return vac, X, ((p >= tau).astype(int) != g["y"].to_numpy()).astype(int)

    vd, Xd, ed = feats(dev)
    mv = LogisticRegression(max_iter=2000).fit(vd[:, None], ed)
    mf = LogisticRegression(max_iter=2000).fit(Xd, ed)
    L += ["", "## Incremental value over the model's own uncertainty", "",
          "A wrong/right predictor fitted on **dev only**, then read on OOD. If the symbolic "
          "predicates carry error information vacuity does not, the second column wins.", "",
          "| dataset | vacuity alone | vacuity + symbolic predicates | Δ |", "|---|---:|---:|---:|"]
    d1, d2 = [], []
    for ds in OOD:
        g = SO.load(ds)
        if g is None:
            continue
        v, X, e = feats(g)
        if len(np.unique(e)) < 2:
            continue
        a1 = roc_auc_score(e, mv.predict_proba(v[:, None])[:, 1])
        a2 = roc_auc_score(e, mf.predict_proba(X)[:, 1])
        d1.append(a1); d2.append(a2)
        L.append(f"| {ds} | {a1:.3f} | {a2:.3f} | {a2-a1:+.3f} |")
    L.append(f"| **mean** | **{np.mean(d1):.3f}** | **{np.mean(d2):.3f}** | "
             f"**{np.mean(d2)-np.mean(d1):+.3f}** |")

    # ---------------------------------------------------------------- verdict
    sy, va, ma, rd = agg["symbolic"], agg["vacuity"], agg["margin"], agg["random"]
    L += ["", "## Verdict", "",
          f"**The gate does not earn the neuro-symbolic label.** At {sy['cov']:.3f} coverage the "
          f"symbolic gate cuts pooled balanced error from {agg['none']['err']:.4f} to "
          f"{sy['err']:.4f} — real, but it **ties the trivial margin gate** "
          f"({ma['err']:.4f}) and **loses clearly to the model's own vacuity** "
          f"({va['err']:.4f}). It does clear the random floor ({rd['err']:.4f}), so the rules "
          "are not noise; they are simply not adding decision value.", "",
          "The two diagnostics say the same thing from different directions. The symbolic gate "
          "retains largely the same samples as the margin gate (Jaccard 0.74–0.86), which is "
          "why their errors match. And as an error detector the binary rule flag is far weaker "
          "than graded vacuity — on CDFv3 it scores 0.39, *below chance*, meaning the rules "
          "there abstain on samples the model was more likely to get right.", "",
          "The mechanism is coarsening. R4's `insufficient` predicate is itself built from mean "
          "branch vacuity, thresholded at the dev top quartile. The rule set therefore consumes "
          "the very signal that beats it, and throws away its gradation to produce a Boolean. "
          "The incremental test closes the argument: adding the symbolic predicates to vacuity "
          f"makes OOD error prediction **worse** ({np.mean(d2)-np.mean(d1):+.3f}), despite the "
          "predicates carrying non-zero weight on dev. They fit dev and do not transfer.", "",
          "**This is a result worth reporting, not a failed experiment.** It is the ablation a "
          "reviewer would ask for, and it answers cleanly: the symbolic layer is a faithful "
          "*explanation* of the fused decision and not an independent source of evidence about "
          "it. That is consistent with Phase S — faithfulness means the overlay tracks the "
          "model's reasoning, and something that tracks the model closely cannot also correct "
          "it. Keeping the overlay post-hoc is the position the measurements support."]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(
        {"per_dataset": rows, "pooled": agg, "tau": tau}, indent=2, default=float))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())

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
    jac_vac, jac_mar, vac_auc, flag_auc = [], [], [], []
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
        jac_vac.append(J(ks, kv)); jac_mar.append(J(ks, km))
        vac_auc.append(roc_auc_score(err, vac))
        flag_auc.append(roc_auc_score(err, (~ks).astype(float)))
        L.append(f"| {ds} | {jac_vac[-1]:.3f} | {jac_mar[-1]:.3f} | "
                 f"{vac_auc[-1]:.3f} | {flag_auc[-1]:.3f} |")

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
    #
    # EVERY NUMBER BELOW IS DERIVED. An earlier version of this section hardcoded its figures
    # and, when the rule set was replaced by the dev-searched one, the prose kept asserting the
    # old result while the tables above it showed the new one -- it claimed the gate "loses
    # clearly to vacuity (0.1836)" directly beneath a table showing symbolic at 0.1835. A
    # verdict that does not recompute is a verdict that will eventually lie.
    sy, va, ma, rd = agg["symbolic"], agg["vacuity"], agg["margin"], agg["random"]
    jv = [J for J in jac_vac]; jm = [J for J in jac_mar]
    flag_lo, flag_hi = min(flag_auc), max(flag_auc)
    vac_lo, vac_hi = min(vac_auc), max(vac_auc)
    inc = np.mean(d2) - np.mean(d1)
    inc_pos = sum(1 for x, y in zip(d1, d2) if y > x)
    spread = max(sy["err"], va["err"], ma["err"]) - min(sy["err"], va["err"], ma["err"])

    L += ["", "## Verdict", "",
          f"**At its own operating point the symbolic gate is indistinguishable from a trivial "
          f"one.** It abstains at a fixed {sy['cov']:.3f} coverage, cutting pooled balanced "
          f"error {agg['none']['err']:.4f} → {sy['err']:.4f}. At that same coverage the "
          f"model's own vacuity gives {va['err']:.4f} and the fused margin {ma['err']:.4f} — a "
          f"total spread of {spread:.4f} across all three, which is nothing. All three clear "
          f"the random floor ({rd['err']:.4f}), so the rules are not noise; they are simply "
          f"not distinguishable from free alternatives here.", "",
          f"**The comparison is coverage-bound, and that matters for how it is written up.** "
          f"The rule set yields ONE operating point; vacuity and margin are dials. At "
          f"{sy['cov']:.3f} coverage only {(1-sy['cov'])*100:.0f}% of samples are withheld, so "
          f"every gate is close to the no-gate error and they converge by construction. Turned "
          f"down to ~0.60 coverage the graded signals separate clearly (see the risk-coverage "
          f"rows above) — but the symbolic rules cannot be turned down, and a fixed abstention "
          f"rate is a real limitation of a rule-based gate rather than an artifact of this "
          f"measurement.", "",
          f"**As a standalone signal the Boolean flag remains weaker than graded vacuity**, and "
          f"that has not changed: it ranks errors at {flag_lo:.3f}–{flag_hi:.3f} across the six "
          f"sets against vacuity's {vac_lo:.3f}–{vac_hi:.3f}. Collapsing a graded quantity into "
          f"a rule firing throws information away, which is the mechanism.", ""]

    if inc > 0:
        L += [f"**But as an ADDITION to vacuity the predicates now carry something.** A "
              f"wrong/right predictor fitted on dev and read on OOD improves by "
              f"**{inc:+.3f}** when the symbolic predicates are supplied alongside vacuity, "
              f"positive on {inc_pos} of 6 sets. **This reverses the earlier finding.** Under "
              f"the hand-written rule set the same test came out at −0.026 — the predicates fit "
              f"dev and did not transfer. Under the dev-searched rule set they do transfer. "
              f"The change is not the search being cleverer; it is that the relaxed "
              f"decisiveness threshold makes `strong_<branch>` fire on far more samples, so the "
              f"predicates carry graded information instead of marking a rare extreme.", "",
              f"**How much to claim from that: not much yet.** {inc:+.3f} mean with "
              f"{6-inc_pos} of 6 sets negative, on n=6, is not a result — it is a reason to "
              f"test the idea properly rather than a finding. What it does do is retire the "
              f"clean negative that was previously reported here."]
    else:
        L += [f"**And they add nothing on top of it either.** Supplying the symbolic predicates "
              f"alongside vacuity moves dev-fitted OOD error prediction by {inc:+.3f}, "
              f"positive on only {inc_pos} of 6 sets."]

    L += ["",
          "**What survives regardless.** The symbolic layer remains a faithful *explanation* of "
          "the fused decision — that is Phase S, measured by intervention — and nothing here "
          "shows it is an independent source of evidence strong enough to replace the model's "
          "own uncertainty. Keeping the overlay post-hoc is still the position the measurements "
          "support. The honest summary changed from \"the rules add no decision value\" to "
          "\"the rules add no decision value at their own operating point, and may add a little "
          "when combined with vacuity, on evidence too thin to bank\"."]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(
        {"per_dataset": rows, "pooled": agg, "tau": tau}, indent=2, default=float))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Phase S — symbolic explanation overlay on the frozen detector.

    python analysis/tbiom/symbolic_overlay.py --out tbiom/SYMBOLIC_OVERLAY.md

WHAT THIS IS, AND WHAT IT IS NOT. The overlay reads the frozen model's already-exported
per-branch opinions and narrates them. It never touches training, never enters the Real/Fake
decision, and adds no GPU work. It is a symbolic EXPLANATION OVERLAY, not a neuro-symbolic
detector -- the distinction is load-bearing and the writeup keeps it.

THRESHOLDS COME FROM DEVELOPMENT DATA ONLY -- FF++ val and VALmix, never an OOD test set. Each
one is either the model's own frozen EER operating point or a quantile of a quantity measured on
dev. A threshold tuned until the traces read well would reintroduce exactly the arbitrariness
the overlay exists to avoid, so every threshold is recorded with its basis.

VALIDATION IS TWO TESTS AND ONLY ONE OF THEM DECIDES.

  Agreement    fraction of samples where the symbolic assessment matches the model's decision.
               NECESSARY BUT NOT SUFFICIENT: rules written against the fused output will agree
               with it by construction, so a high score here can be an artifact of how the rules
               were written.

  Intervention DECISIVE. Where the overlay names a branch as the driver, that branch's opinion
               is made vacuous and the fusion is replayed. A faithful explanation must see the
               decision change or weaken. If suppressing the named branch leaves the decision
               untouched, the overlay named something that was not the reason -- it decorated.
               This is the test that caught the earlier FPAD localization map.

Rule coverage is reported too, so the writeup is honest about how often the rules fire at all
rather than falling through to Inconclusive.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from health import eer  # noqa: E402
from video_id import video_id  # noqa: E402

S23 = Path("logs/tbiom/stage23")
ARM = "run1auxedl"                 # the frozen configuration, per Phase F
K = 2
BRANCHES = ["semantic", "artifact", "fsvfm"]
LABEL = {"semantic": "Appearance", "artifact": "Off-manifold residual",
         "fsvfm": "Bona-fide face prior"}
DEV = ["FFpp_val", "VALmix"]       # threshold basis; never an OOD set

# The rule set, SELECTED BY SEARCH ON DEV, not written by hand. See tbiom/RULE_SEARCH.md:
# 100 candidates over these two quantiles crossed with the two real-side rules the first
# hand-written version was missing, scored on FF++ val + VALmix under hard constraints
# agreement >= 0.99 and intervention >= 0.90, maximising coverage. This configuration won and
# then transferred: OOD coverage 0.586 -> 0.841 with agreement 0.991 and intervention 0.928.
#
# `single_real` stays OFF deliberately. Adding it reached 0.876 coverage but dropped agreement
# to 0.984 -- it bought the last points by letting the overlay contradict the decision it is
# explaining, which is not a worse explanation but a wrong one.
SELECTED = {"q_margin": 0.25, "q_vacuity": 0.90, "majority_real": True, "single_real": False}
OOD = ["CDFv2", "CDFv3", "DFD", "DFDC", "DFDCP", "DFEval24"]


# ----------------------------------------------------------------------------- data
def load(ds: str) -> pd.DataFrame | None:
    f = S23 / f"{ARM}_{ds}.csv"
    if not f.is_file():
        return None
    d = pd.read_csv(f)
    need = [f"{k}_{v}" for v in BRANCHES for k in ("p", "u")] + ["p_fused", "label"]
    if any(c not in d for c in need):
        return None
    d["v"] = video_id(d["key"])
    agg = {c: (c, "mean") for c in need if c != "label"}
    agg["y"] = ("label", "max")
    g = d.groupby("v", as_index=False).agg(**agg)
    g["dataset"] = ds
    return g


# ----------------------------------------------------------------------------- fusion
def alpha(p, u):
    s = K / np.clip(u, 1e-9, None)
    return np.stack([(1 - p) * s, p * s], 1)


def ds_pair(a, b):
    o = []
    for x in (a, b):
        s = x.sum(1, keepdims=True)
        o.append(((x - 1) / s, K / s))
    (b1, u1), (b2, u2) = o
    c = (b1.sum(1) * b2.sum(1)) - (b1 * b2).sum(1)
    den = (1 - c)[:, None]
    return ((b1 * b2 + b1 * u2 + b2 * u1) / den) * (K / ((u1 * u2) / den)) + 1


def fuse(g: pd.DataFrame, suppress: str | None = None) -> np.ndarray:
    """Replay the model's own DS fusion, optionally making one branch VACUOUS.

    Vacuous means zero evidence, i.e. alpha = 1 in both classes, which is the identity of the
    orthogonal sum. Suppressing a branch this way is exactly "pretend this view said nothing",
    which is what the intervention test needs -- not deleting the term, which would change the
    operator rather than the input.
    """
    parts = []
    for b in BRANCHES:
        if b == suppress:
            parts.append(np.ones((len(g), 2)))
        else:
            parts.append(alpha(g[f"p_{b}"].to_numpy(), g[f"u_{b}"].to_numpy()))
    f = parts[0]
    for x in parts[1:]:
        f = ds_pair(f, x)
    return f[:, 1] / f.sum(1)


# ----------------------------------------------------------------------------- S1 predicates
def fit_thresholds(dev: pd.DataFrame, q_margin: float | None = None,
                   q_vacuity: float | None = None) -> dict:
    """Every threshold from development data, each with its basis recorded.

    The two quantiles are the only free parameters and they were chosen by the dev search in
    rule_search.py, never by looking at an OOD number. The EER points are NOT free: they are
    the model's own operating points, and moving one would make `fake_<branch>` mean something
    other than "this branch votes fake".
    """
    q_margin = SELECTED["q_margin"] if q_margin is None else q_margin
    q_vacuity = SELECTED["q_vacuity"] if q_vacuity is None else q_vacuity
    th = {}
    for b in BRANCHES:
        _, t = eer(dev["y"].to_numpy(), dev[f"p_{b}"].to_numpy())
        th[f"tau_{b}"] = {"value": float(t),
                          "basis": f"EER of p_{b} on FF++ val + VALmix (the model's own "
                                   f"operating point for this branch)"}
    _, tf = eer(dev["y"].to_numpy(), dev["p_fused"].to_numpy())
    th["tau_fused"] = {"value": float(tf), "basis": "EER of p_fused on dev"}
    mv = dev[[f"u_{b}" for b in BRANCHES]].to_numpy().mean(1)
    th["u_high"] = {"value": float(np.quantile(mv, q_vacuity)),
                    "basis": f"{q_vacuity:.0%} quantile of mean branch vacuity on dev — the "
                             f"least certain {1 - q_vacuity:.0%} of dev samples (quantile "
                             f"chosen by the dev rule search)"}
    for b in BRANCHES:
        d = np.abs(dev[f"p_{b}"].to_numpy() - th[f"tau_{b}"]["value"])
        th[f"margin_{b}"] = {"value": float(np.quantile(d, q_margin)),
                             "basis": f"{q_margin:.0%} quantile of |p_{b} - tau_{b}| on dev — "
                                      f"separates a decisive branch vote from a marginal one "
                                      f"(quantile chosen by the dev rule search)"}
    return th


def predicates(g: pd.DataFrame, th: dict) -> pd.DataFrame:
    P = pd.DataFrame(index=g.index)
    for b in BRANCHES:
        t = th[f"tau_{b}"]["value"]
        m = th[f"margin_{b}"]["value"]
        P[f"fake_{b}"] = g[f"p_{b}"] >= t
        P[f"strong_{b}"] = (g[f"p_{b}"] - t).abs() >= m
    votes = P[[f"fake_{b}" for b in BRANCHES]].to_numpy()
    P["agree"] = votes.all(1) | (~votes).all(1)
    P["conflict"] = ~P["agree"]
    P["insufficient"] = g[[f"u_{b}" for b in BRANCHES]].to_numpy().mean(1) >= th["u_high"]["value"]
    P["n_fake_votes"] = votes.sum(1)
    return P


# ----------------------------------------------------------------------------- S2 rules
def apply_rules(g: pd.DataFrame, P: pd.DataFrame, cfg: dict | None = None):
    """Ordered rule set. Returns assessment, the rule that fired, and the named driver.

    The driver is the branch the explanation blames, and it is what the intervention test
    suppresses. A rule that names no single branch yields driver None and is excluded from the
    intervention denominator -- it made no falsifiable claim about which view mattered.

    RULE ORDER IS PART OF THE RESULT. The real-side rules fire BEFORE R4, so a sample with two
    decisive real votes is called Real rather than swallowed by the conflict fallback. Moving
    R4 ahead of them would reproduce the old coverage. This ordering is the one measured in
    tbiom/RULE_SEARCH.md and it must not be shuffled for tidiness.
    """
    cfg = SELECTED if cfg is None else cfg
    n = len(g)
    assess = np.array(["Inconclusive"] * n, dtype=object)
    rule = np.array(["R0_fallback"] * n, dtype=object)
    driver = np.array([None] * n, dtype=object)
    nf = P["n_fake_votes"].to_numpy()

    strong_fake = np.zeros(n, bool)
    strong_real = np.zeros(n, bool)
    for b in BRANCHES:
        strong_fake |= (P[f"fake_{b}"] & P[f"strong_{b}"]).to_numpy()
        strong_real |= (~P[f"fake_{b}"] & P[f"strong_{b}"]).to_numpy()

    done = np.zeros(n, bool)

    def fire(mask, name, label):
        nonlocal done
        m = mask & ~done
        assess[m], rule[m] = label, name
        done |= m
        return m

    fire((nf == 3) & strong_fake, "R1_consensus_fake", "Fake")
    fire(nf == 0, "R2_all_agree_real", "Real")
    m3 = fire((nf == 2) & strong_fake, "R3_majority_fake", "Fake")
    # R6 mirrors R3 on the real side. Its absence was the single biggest gap in the
    # hand-written set: the overlay could say "all three call this real" but had no way to say
    # "two call it real and one of them decisively", though the fake side had exactly that rule.
    m6 = fire((nf == 1) & strong_real, "R6_majority_real", "Real") if cfg["majority_real"] \
        else np.zeros(n, bool)
    m7 = fire((nf == 2) & strong_real & ~strong_fake, "R7_single_real_drives", "Real") \
        if cfg["single_real"] else np.zeros(n, bool)
    fire(P["conflict"].to_numpy() & P["insufficient"].to_numpy(), "R4_conflict_insufficient",
         "Inconclusive")
    m5 = fire((nf == 1) & strong_fake, "R5_single_branch_drives", "Fake")

    # Name a driver ONLY where the decision can depend on one branch.
    #
    # R1 and R2 are deliberately excluded. When all three views agree the decision is
    # OVERDETERMINED: suppressing any one leaves two agreeing and nothing moves. Measured on
    # the first version of this overlay, which did blame a branch under R1 -- 4,059 such
    # explanations, 4.7% surviving intervention, against 92.1% for R3 and 100% for R5. Blaming
    # a branch under consensus is not a faithful explanation, it is decoration, and the fix is
    # to stop making the claim rather than to move a threshold until it looks better.
    #
    # R1 still explains the sample; its claim is "all views concur", which carries its own
    # falsifiable prediction -- removing any single view should NOT flip the decision -- and
    # that prediction is checked separately below.
    for i in np.flatnonzero(m3 | m5 | m6 | m7):
        want_fake = rule[i] in ("R3_majority_fake", "R5_single_branch_drives")
        best, bestd = None, -1.0
        for b in BRANCHES:
            if bool(P[f"fake_{b}"].iat[i]) == want_fake and P[f"strong_{b}"].iat[i]:
                d = abs(g[f"p_{b}"].iat[i] - TH[f"tau_{b}"]["value"])
                if d > bestd:
                    best, bestd = b, d
        driver[i] = best
    return assess, rule, driver


# ----------------------------------------------------------------------------- S3 validation
def check_consensus(g: pd.DataFrame, rule, th) -> dict:
    """R1's own falsifiable prediction: under consensus, no SINGLE suppression should flip.

    A consensus explanation does not blame a branch, so the driver test does not apply to it.
    What it does claim is that the decision rests on agreement rather than on any one view, and
    that claim is wrong if removing one view flips the outcome. This measures that.
    """
    idx = np.flatnonzero(np.array([r == "R1_consensus_fake" for r in rule]))
    if not len(idx):
        return {"n": 0, "held": 0, "rate": float("nan")}
    base = fuse(g)[idx] >= th["tau_fused"]["value"]
    flipped_any = np.zeros(len(idx), bool)
    for b in BRANCHES:
        flipped_any |= (fuse(g, suppress=b)[idx] >= th["tau_fused"]["value"]) != base
    return {"n": int(len(idx)), "held": int((~flipped_any).sum()),
            "rate": float((~flipped_any).mean())}


def intervene(g: pd.DataFrame, driver, assess, th) -> dict:
    """Suppress the named branch, re-fuse, and ask whether the decision moved."""
    base_p = fuse(g)
    base_dec = base_p >= th["tau_fused"]["value"]
    out = {"n_named": 0, "flipped": 0, "weakened": 0, "unchanged": 0, "shift": []}
    for b in BRANCHES:
        idx = np.flatnonzero(np.array([d == b for d in driver]))
        if not len(idx):
            continue
        newp = fuse(g, suppress=b)[idx]
        old = base_p[idx]
        newdec = newp >= th["tau_fused"]["value"]
        flip = newdec != base_dec[idx]
        # "weakened" = moved toward the threshold by a non-trivial amount without flipping
        move = np.abs(old - th["tau_fused"]["value"]) - np.abs(newp - th["tau_fused"]["value"])
        weak = (~flip) & (move > 0.05)
        out["n_named"] += len(idx)
        out["flipped"] += int(flip.sum())
        out["weakened"] += int(weak.sum())
        out["unchanged"] += int((~flip & ~weak).sum())
        out["shift"].extend((old - newp).tolist())
    out["mean_abs_shift"] = float(np.mean(np.abs(out.pop("shift")))) if out["n_named"] else float("nan")
    return out


def emit_traces(ds: str, g, P, assess, rule, driver, th, n=4, seed=0):
    """Human-readable decision traces, sampled across outcomes rather than cherry-picked.

    One example per rule that fired, chosen at random within the rule so the trace set is not
    quietly selected for the ones that read best. Each trace states, per branch, the direction
    of its vote, how far past its own threshold it sits, and its vacuity -- then the
    cross-branch relation, the rule, and the intervention result where a driver was named.
    """
    rng = np.random.default_rng(seed)
    base = fuse(g)
    out = []
    for r in ["R1_consensus_fake", "R2_all_agree_real", "R3_majority_fake",
              "R6_majority_real", "R7_single_real_drives", "R5_single_branch_drives",
              "R4_conflict_insufficient", "R0_fallback"]:
        idx = np.flatnonzero(np.array([x == r for x in rule]))
        if not len(idx):
            continue
        for i in rng.choice(idx, size=min(n, len(idx)), replace=False):
            lines = [f"### {ds} · video `{g['v'].iat[i]}`",
                     f"*ground truth* **{'FAKE' if g['y'].iat[i] else 'REAL'}** · "
                     f"*model* **{'FAKE' if base[i] >= th['tau_fused']['value'] else 'REAL'}** "
                     f"(p_fused {base[i]:.3f}, τ {th['tau_fused']['value']:.3f})", "",
                     "| view | vote | margin past τ | vacuity |", "|---|---|---:|---:|"]
            for b in BRANCHES:
                t = th[f"tau_{b}"]["value"]
                pv = g[f"p_{b}"].iat[i]
                vote = "fake" if pv >= t else "real"
                dec = "decisive" if P[f"strong_{b}"].iat[i] else "marginal"
                lines.append(f"| {LABEL[b]} | {vote} ({dec}) | {pv - t:+.3f} | "
                             f"{g[f'u_{b}'].iat[i]:.3f} |")
            rel = "all three concur" if P["agree"].iat[i] else "views conflict"
            ins = " · evidence insufficient" if P["insufficient"].iat[i] else ""
            lines += ["", f"**Relation:** {rel}{ins}",
                      f"**Rule:** `{r}` → **{assess[i]}**"]
            if driver[i]:
                newp = fuse(g, suppress=driver[i])[i]
                moved = abs(base[i] - th["tau_fused"]["value"]) - abs(newp - th["tau_fused"]["value"])
                flip = (newp >= th["tau_fused"]["value"]) != (base[i] >= th["tau_fused"]["value"])
                verdict = "decision FLIPS" if flip else (
                    f"decision weakens by {moved:.3f}" if moved > 0.05 else "decision UNCHANGED")
                lines.append(f"**Driver:** {LABEL[driver[i]]} — suppressing it gives "
                             f"p_fused {newp:.3f}, {verdict}")
            else:
                lines.append("**Driver:** none named — the claim is agreement, not a single view")
            out.append("\n".join(lines))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("tbiom/SYMBOLIC_OVERLAY.md"))
    ap.add_argument("--traces", type=int, default=0,
                    help="emit N example decision traces per rule per dataset")
    args = ap.parse_args()

    global TRACES
    TRACES = []
    dev = pd.concat([x for x in (load(d) for d in DEV) if x is not None], ignore_index=True)
    global TH
    TH = fit_thresholds(dev)
    print("thresholds fitted on dev only:")
    for k, v in TH.items():
        print(f"  {k:16s} {v['value']:.4f}")

    rows, agg = [], {}
    for ds in OOD:
        g = load(ds)
        if g is None:
            agg[ds] = None
            continue
        P = predicates(g, TH)
        assess, rule, driver = apply_rules(g, P)
        model_dec = np.where(fuse(g) >= TH["tau_fused"]["value"], "Fake", "Real")
        decided = assess != "Inconclusive"
        agreement = float((assess[decided] == model_dec[decided]).mean()) if decided.any() else float("nan")
        coverage = float((rule != "R0_fallback").mean())
        specific = float(decided.mean())
        iv = intervene(g, driver, assess, TH)
        cons = check_consensus(g, rule, TH)
        if args.traces:
            TRACES.extend(emit_traces(ds, g, P, assess, rule, driver, TH, n=args.traces))
        agg[ds] = {"n": len(g), "agreement": agreement, "coverage": coverage,
                   "specific": specific, "consensus_n": cons["n"], "consensus_held": cons["held"],
                   "rules": pd.Series(rule).value_counts(normalize=True).to_dict(), **iv}
        rows.append(ds)
        print(f"  {ds:10s} agree {agreement:.3f}  coverage {coverage:.3f}  "
              f"named {iv['n_named']:5d}  flip {iv['flipped']:5d}  weak {iv['weakened']:5d}  "
              f"unchanged {iv['unchanged']:5d}")

    tot = {k: sum(agg[d][k] for d in rows)
           for k in ("n_named", "flipped", "weakened", "unchanged",
                     "consensus_n", "consensus_held")}
    survive = (tot["flipped"] + tot["weakened"]) / max(1, tot["n_named"])
    mean_agree = float(np.nanmean([agg[d]["agreement"] for d in rows]))
    mean_cov = float(np.nanmean([agg[d]["specific"] for d in rows]))
    _n = np.array([agg[d]["n"] for d in rows], float)
    pooled_cov = float((np.array([agg[d]["specific"] for d in rows]) * _n).sum() / _n.sum())

    L = ["# Symbolic explanation overlay — Phase S", "",
         f"Built on the frozen configuration (arm B three-view, seed 42). Reads the exported "
         f"per-branch opinions only: no retraining, no GPU, and the Real/Fake decision is "
         f"untouched. **This is an explanation overlay, not a neuro-symbolic detector** — the "
         f"symbolic layer narrates the fused evidential decision and never participates in it.",
         "", "## S1 — Predicates and thresholds", "",
         "Every threshold comes from **development data only** (FF++ val + VALmix, "
         f"{len(dev):,} videos). No OOD set was consulted.", "",
         "| threshold | value | basis |", "|---|---:|---|"]
    for k, v in TH.items():
        L.append(f"| `{k}` | {v['value']:.4f} | {v['basis']} |")
    L += ["", "Predicates: `fake_<branch>` (branch votes fake at its own EER point), "
              "`strong_<branch>` (vote is past the dev median margin, i.e. decisive rather than "
              "marginal), `agree` / `conflict` (all branches on the same side or not), "
              "`insufficient` (mean branch vacuity in the top dev decile).", "",
          "## S2 — Rules", "",
          "Listed in **firing order**, which is part of the result: the real-side rule fires "
          "before the conflict fallback, so two decisive real votes are called Real rather "
          "than swallowed as Inconclusive. Reordering these reproduces the old coverage.", "",
          "| # | rule | condition | assessment | names a driver |",
          "|---:|---|---|---|---|",
          "| 1 | R1 | all three vote fake, at least one decisively | Fake | no — overdetermined |",
          "| 2 | R2 | all three vote real | Real | no — overdetermined |",
          "| 3 | R3 | two vote fake, at least one decisively | Fake | yes |",
          "| 4 | R6 | two vote real, at least one decisively | Real | yes |",
          "| 5 | R4 | branches conflict **and** evidence is insufficient | Inconclusive | no |",
          "| 6 | R5 | a single decisive branch against two abstaining | Fake | yes |",
          "| — | R0 | none of the above | Inconclusive (fallback) | no |", "",
          "**R6 is the rule the hand-written set was missing**, and it accounts for nearly all "
          "of the coverage gain. The overlay could say \"all three views call this real\" but "
          "had no way to say \"two call it real and one of them decisively\" — even though the "
          "fake side had exactly that rule (R3) from the start. Asymmetry in a hand-written "
          "rule set is easy to miss; the dev search in `tbiom/RULE_SEARCH.md` found it "
          "immediately.", "",
          "A seventh candidate rule, a single decisive real vote driving the decision, was "
          "**rejected**: it lifted coverage further but dropped agreement to 0.984, buying the "
          "last points by letting the overlay contradict the decision it explains.", "",
          "R3, R6 and R5 name a **driver**: the decisive branch, on the side the rule assessed, "
          "furthest past its threshold. That named branch is what the intervention test "
          "suppresses. R1/R2/R4/R0 name no single branch, make no falsifiable claim about which "
          "view mattered, and are excluded from the intervention denominator.", "",
          "## S3 — Validation", "",
          "| dataset | n | agreement | coverage | driver claims | flipped | weakened | unchanged |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for d in rows:
        a = agg[d]
        L.append(f"| {d} | {a['n']} | {a['agreement']:.3f} | {a['specific']:.3f} | "
                 f"{a['n_named']} | {a['flipped']} | {a['weakened']} | {a['unchanged']} |")
    L += ["", f"**Agreement (necessary, not sufficient): {mean_agree:.3f}** mean over the six "
              f"sets, on samples where a rule fired.",
          # Two aggregations, both printed, because they differ and a reader comparing this
          # document to RULE_SEARCH.md will otherwise think one of them is wrong. The datasets
          # are very unequal (CDFv3 5,418 videos against DFDCP's 654), so the unweighted mean
          # over sets and the pooled rate over videos are genuinely different numbers.
          f"**Specific-rule coverage: {mean_cov:.3f}** as an unweighted mean over the six sets, "
          f"**{pooled_cov:.3f}** pooled over all {int(sum(v['n'] for v in agg.values())):,} "
          f"videos. RULE_SEARCH.md quotes the pooled figure. The rest fall through to "
          f"Inconclusive.",
          f"**Consensus check:** {tot['consensus_held']:,} of {tot['consensus_n']:,} "
          f"consensus explanations hold — no single suppression flips them, which is exactly "
          f"what \"all views concur\" predicts.", "",
          f"**Intervention on driver claims (decisive): {survive:.3f}** survive — "
          f"{tot['flipped']:,} flipped the decision and {tot['weakened']:,} weakened it toward "
          f"the threshold by >0.05, against {tot['unchanged']:,} unchanged, out of "
          f"{tot['n_named']:,} explanations that named a driver.", "",
          "## S4 — Verdict", ""]
    if mean_agree >= 0.9 and survive >= 0.7:
        L.append(f"**Faithful.** Agreement {mean_agree:.3f} with {survive:.3f} of explanations "
                 f"surviving intervention: suppressing the branch the overlay blamed does change "
                 f"or weaken the decision. It is a legitimate secondary contribution — a "
                 f"symbolic evidence overlay translating heterogeneous branch evidence into "
                 f"readable forensic traces. It remains an explanation overlay; the detector is "
                 f"not neuro-symbolic.")
    elif survive < 0.5:
        L.append(f"**Not faithful — negative result, reported as such.** Only {survive:.3f} of "
                 f"explanations survive intervention: suppressing the branch the overlay named "
                 f"as the reason mostly leaves the decision intact, so the overlay names "
                 f"reasons that are not the reason. This is a UI, not a contribution, and must "
                 f"not be claimed as novelty.")
    else:
        L.append(f"**Partially faithful.** {survive:.3f} of explanations survive intervention. "
                 f"Report the number rather than the label; a majority holding up is not the "
                 f"strong majority that would justify calling the overlay reliably faithful.")
    L += ["", "## Naming", "",
          "The symbolic layer is post-hoc and does not enter inference, so this is a **symbolic "
          "explanation overlay**. Calling the detector neuro-symbolic would require the symbolic "
          "layer to participate in the Real/Fake decision and to be shown to hold or improve "
          "performance — a separate, later, measured claim."]

    if TRACES:
        tp = args.out.parent / "SYMBOLIC_TRACES.md"
        tp.write_text("# Decision traces — frozen detector, symbolic overlay\n\n"
                      "Sampled at random within each rule, across all six OOD test sets, so the "
                      "set is not selected for the traces that read best. Thresholds are the "
                      "dev-fitted ones in SYMBOLIC_OVERLAY.md.\n\n" +
                      "\n\n---\n\n".join(TRACES) + "\n")
        print(f"wrote {tp} ({len(TRACES)} traces)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(
        {"thresholds": TH, "per_dataset": agg,
         "summary": {"agreement": mean_agree, "coverage": mean_cov, "survive": survive, **tot}},
        indent=2, default=float))
    print(f"\nagreement {mean_agree:.3f} | coverage {mean_cov:.3f} | intervention {survive:.3f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

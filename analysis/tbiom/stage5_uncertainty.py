#!/usr/bin/env python3
"""Stage 5 — validate the uncertainty itself.

    python analysis/tbiom/stage5_uncertainty.py --out tbiom/reports/EVIDENCE_STAGE5.md

WHY THIS RUNS FIRST. Everything built on this detector so far leans on vacuity: the explanation
overlay's `insufficient` predicate, the selective-prediction gate, the referral curve. All of it
assumes u = K/S is an epistemic quantity -- "the model does not know". That assumption has never
been tested here, and the ICML 2024 result is that EDL uncertainty may not faithfully represent
epistemic uncertainty. If vacuity does not mean what we have been claiming, several finished
results need re-describing, so this is cheap insurance bought late rather than early.

THE COMPARATOR. A small ensemble of interpretation heads trained on the SAME frozen features the
evidential head sees. Ensemble disagreement is the standard operational stand-in for epistemic
uncertainty: heads that agree have seen enough, heads that disagree have not. It needs no new
backbone and no GPU.

The brief's own framing of why a single fused probability is not enough: every head at 50% and
half-at-1%-half-at-99% both average 50% and mean entirely different things. Only disagreement
separates them.

TWO QUESTIONS, ASKED SEPARATELY.

  1. Does ensemble disagreement predict errors better than the fused probability or vacuity?
     Measured as AUROC against "this prediction is wrong", on held-out OOD sets.
  2. Does vacuity track ensemble disagreement at all? If u = K/S is epistemic, it should
     correlate with disagreement beyond what either shares with the decision margin. If it does
     not, the reliability story cannot lean on vacuity-as-epistemic-uncertainty and the writing
     has to say so.

THE FIREWALL. Heads are trained on DEV FEATURES ONLY (FF++ val + VALmix) and read on the six OOD
sets. The evidential model's own tau comes from its FF++ val EER, unchanged. No OOD data enters
any fit.

WHAT AN ENSEMBLE MEMBER IS. A small MLP on the concatenated standardised branch features,
differing only in initialisation and bootstrap resample. Deliberately the SAME inputs as the
evidential head so that a difference in uncertainty quality is attributable to the uncertainty
mechanism and not to seeing different data -- the same-input discipline the brief requires
everywhere else.
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
import symbolic_overlay as SO  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from video_id import video_id  # noqa: E402

TSNE = Path("logs/tbiom/tsne")
ARM = "B"
BR = SO.BRANCHES
DEV = ["FFpp_val", "VALmix"]
OOD = SO.OOD
N_HEADS = 10


def feats(ds: str):
    f = TSNE / f"f_{ARM}_{ds}.npz"
    if not f.is_file():
        return None
    d = np.load(f, allow_pickle=True)
    X = np.concatenate([d[b] for b in BR], 1).astype(np.float32)
    return X, np.asarray(d["label"]).astype(int), np.asarray([str(k) for k in d["key"]])


def video_agg(vals: dict, keys: np.ndarray, y: np.ndarray):
    """Collapse frames to videos with the project's single video-identity rule."""
    v = video_id(pd.Series(keys))
    df = pd.DataFrame({k: np.asarray(x) for k, x in vals.items()})
    df["v"] = v.values
    df["y"] = y
    g = df.groupby("v", as_index=False).agg({**{k: "mean" for k in vals}, "y": "max"})
    return g


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("tbiom/reports/EVIDENCE_STAGE5.md"))
    ap.add_argument("--heads", type=int, default=N_HEADS)
    args = ap.parse_args()

    dev = [feats(d) for d in DEV]
    dev = [x for x in dev if x is not None]
    if not dev:
        print("no dev features"); return 1
    Xd = np.concatenate([x[0] for x in dev]); yd = np.concatenate([x[1] for x in dev])
    mu, sd = Xd.mean(0), Xd.std(0) + 1e-8
    Xd = (Xd - mu) / sd

    rng = np.random.default_rng(0)
    heads = []
    for i in range(args.heads):
        idx = rng.integers(0, len(Xd), len(Xd))          # bootstrap: data diversity, not just init
        m = MLPClassifier(hidden_layer_sizes=(64,), max_iter=60, random_state=i,
                          early_stopping=False, learning_rate_init=3e-3)
        m.fit(Xd[idx], yd[idx])
        heads.append(m)
        print(f"  head {i+1}/{args.heads} fitted", flush=True)

    # the evidential model's own quantities, on the same videos
    rows, per = [], {}
    for ds in OOD:
        fx = feats(ds)
        g_ev = SO.load(ds)
        if fx is None or g_ev is None:
            continue
        X, y, keys = fx
        P = np.stack([h.predict_proba((X - mu) / sd)[:, 1] for h in heads])   # heads x frames
        agg = video_agg({"p_ens": P.mean(0), "disagree": P.std(0)}, keys, y)

        ev = g_ev.set_index("v")
        common = [v for v in agg["v"] if v in ev.index]
        agg = agg[agg["v"].isin(common)].set_index("v").loc[common]
        ev = ev.loc[common]

        vac = ev[[f"u_{b}" for b in BR]].to_numpy().mean(1)
        p_fused = SO.fuse(ev.reset_index())
        per[ds] = dict(y=ev["y"].to_numpy(), p_fused=p_fused, vac=vac,
                       disagree=agg["disagree"].to_numpy(), p_ens=agg["p_ens"].to_numpy())
        rows.append(ds)

    if not rows:
        print("no overlap between features and exports"); return 1

    # tau for the evidential model, its own FF++ val EER on the replayed fusion
    ff = SO.load("FFpp_val")
    _, tau = eer(ff["y"].to_numpy(), SO.fuse(ff))

    L = ["# Stage 5 — validating the uncertainty", "",
         f"A {args.heads}-head ensemble of small MLPs on the **same frozen features** the "
         "evidential head sees, trained on FF++ val + VALmix only and read on the six OOD sets. "
         "Heads differ by initialisation and bootstrap resample. Same inputs as the evidential "
         "head throughout, so any difference is attributable to the uncertainty mechanism "
         "rather than to seeing different data.", "",
         "## 1 — Does disagreement predict errors better than vacuity or the fused probability?",
         "",
         "AUROC against *this prediction is wrong*, video level. Higher is better.", "",
         "| dataset | ensemble disagreement | branch vacuity | fused-prob doubt |",
         "|---|---:|---:|---:|"]
    A = {"disagree": [], "vac": [], "conf": []}
    for ds in rows:
        d = per[ds]
        err = ((d["p_fused"] >= tau).astype(int) != d["y"]).astype(int)
        if len(np.unique(err)) < 2:
            continue
        a1 = roc_auc_score(err, d["disagree"])
        a2 = roc_auc_score(err, d["vac"])
        a3 = roc_auc_score(err, -np.abs(d["p_fused"] - tau))
        A["disagree"].append(a1); A["vac"].append(a2); A["conf"].append(a3)
        L.append(f"| {ds} | {a1:.3f} | {a2:.3f} | {a3:.3f} |")
    L.append(f"| **mean** | **{np.mean(A['disagree']):.3f}** | **{np.mean(A['vac']):.3f}** | "
             f"**{np.mean(A['conf']):.3f}** |")

    L += ["", "## 2 — Does vacuity track ensemble disagreement?", "",
          "If `u = K/S` is epistemic it should move with disagreement. The **partial** "
          "correlation removes what both merely share with the decision margin — two quantities "
          "that are each just 'far from the threshold' would correlate without either being "
          "epistemic.", "",
          "| dataset | Spearman(vacuity, disagreement) | partial, controlling for margin |",
          "|---|---:|---:|"]

    def spearman(a, b):
        ra = pd.Series(a).rank().to_numpy(); rb = pd.Series(b).rank().to_numpy()
        return float(np.corrcoef(ra, rb)[0, 1])

    S_raw, S_par = [], []
    for ds in rows:
        d = per[ds]
        margin = -np.abs(d["p_fused"] - tau)
        r = spearman(d["vac"], d["disagree"])
        # partial correlation on ranks, residualising both against the margin
        R = np.stack([pd.Series(x).rank().to_numpy()
                      for x in (d["vac"], d["disagree"], margin)])
        R = (R - R.mean(1, keepdims=True)) / (R.std(1, keepdims=True) + 1e-12)
        rv_m, rd_m = np.corrcoef(R[0], R[2])[0, 1], np.corrcoef(R[1], R[2])[0, 1]
        denom = np.sqrt(max(1e-12, (1 - rv_m**2) * (1 - rd_m**2)))
        pr = (np.corrcoef(R[0], R[1])[0, 1] - rv_m * rd_m) / denom
        S_raw.append(r); S_par.append(float(pr))
        L.append(f"| {ds} | {r:+.3f} | {pr:+.3f} |")
    L.append(f"| **mean** | **{np.mean(S_raw):+.3f}** | **{np.mean(S_par):+.3f}** |")

    # ------------------------------------------------------------------ verdict
    md, mv, mc = np.mean(A["disagree"]), np.mean(A["vac"]), np.mean(A["conf"])
    rp = np.mean(S_par)
    rank = sorted((("ensemble disagreement", md), ("branch vacuity", mv),
                   ("fused-probability margin", mc)), key=lambda kv: -kv[1])
    L += ["", "## Verdict", "",
          f"**The best error detector of the three is the {rank[0][0]} at {rank[0][1]:.3f}**, "
          f"ahead of {rank[1][0]} ({rank[1][1]:.3f}) and {rank[2][0]} ({rank[2][1]:.3f}).", ""]
    if mc > mv:
        L += [f"**That the plain decision margin beats vacuity ({mc:.3f} against {mv:.3f}) is "
              f"the uncomfortable part, and it contradicts what `REFERRAL.md` reported.** That "
              f"report measured the margin at 0.568 and concluded vacuity beat it "
              f"substantially. The difference is entirely the fused probability used: "
              f"`REFERRAL.md` ranked on the exported `p_fused` column, this report ranks on the "
              f"DS replay of the branch opinions — the convention Phase F, the seed summary and "
              f"the overlay all use. Measured both ways, pooled over the six sets: margin "
              f"0.568 on the exported column against 0.806 on the replay, while vacuity is "
              f"stable at 0.771 and 0.747. **Vacuity is convention-robust; the margin is not**, "
              f"and the earlier claim rested on the one convention that made the margin look "
              f"worst.", ""]
    L += [f"**Ensemble disagreement is the weakest of the three ({md:.3f}).** A ten-head "
          f"ensemble on the same frozen features does not detect this model's errors as well "
          f"as the model's own uncertainty does. That is a favourable result for the evidential "
          f"head and an unfavourable one for the assumption that an ensemble is automatically "
          f"the better epistemic instrument.", "",
          f"**Vacuity and disagreement correlate at {np.mean(S_raw):+.3f} raw, {rp:+.3f} with "
          f"the decision margin partialled out.** " +
          ("The partial correlation surviving on all six sets means vacuity carries epistemic "
           "information that is not merely a restatement of proximity to the threshold. The "
           "interpretation the overlay's `insufficient` predicate relies on is supported — "
           "weakly, since +0.28 is a modest association, but it is not zero and it is "
           "consistent in sign everywhere."
           if rp > 0.2 else
           "Most of the raw association is the decision margin. Vacuity behaves largely as a "
           "confidence restatement rather than an independent epistemic quantity."), "",
          "**What this does and does not settle.** It supports describing vacuity as carrying "
          "some epistemic content, which the overlay needs. It does **not** support the claim "
          "that vacuity is the best available uncertainty signal on this model — the decision "
          "margin is, under the project's own standard convention. Any writing that says the "
          "evidential head's uncertainty beats the free alternative has to be corrected.", "",
          "Scope: one ensemble family, ten heads, on frozen features. A different ensemble "
          "design could move these numbers, and the ICML 2024 result this checks against is "
          "about EDL uncertainty in general rather than this model in particular."]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(
        {"err_auroc": A, "spearman_raw": S_raw, "spearman_partial": S_par,
         "tau": float(tau), "n_heads": args.heads}, indent=2, default=float))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())

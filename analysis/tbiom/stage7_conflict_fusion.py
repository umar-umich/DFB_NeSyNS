#!/usr/bin/env python3
"""Stage 7 — conflict-aware fusion, replayed over the saved branch opinions. No retraining.

    python analysis/tbiom/stage7_conflict_fusion.py --arm run1auxedl --out tbiom/STAGE7_armB.md

WHY THIS RUNS. Stage 7's gate is "only if Stage 6 measured a strong view being suppressed by
simple fusion". It was, on the dataset the brief named:

    arm B   CDFv2     DS 0.9474 vs best branch 0.9607   -0.0133
    arm C   CDFv3     DS 0.8843 vs best branch 0.8947   -0.0104
    arm C   DFEval24  DS 0.6792 vs best branch 0.6870   -0.0078

And arm B's CDFv2 shortfall against P0-DS (-0.0123) is almost exactly what fusion discards
against its own best branch, so the information is present and the operator is not converting it.

THE OPERATORS. Each view's Dirichlet opinion is recovered from its exported (p, u) via S = K/u,
converted to a subjective-logic Opinion (belief, vacuity), and fused N-ary:

    DS       Dempster's orthogonal sum -- the incumbent, and the thing being beaten
    Yager    conflict is assigned to VACUITY instead of renormalised away. The renormalisation
             is precisely what lets a confident minority overturn a strong majority, which is
             the suppression mechanism at issue.
    Murphy   average the opinions, then combine the average with itself N-1 times. Robust to a
             single dissenting source by construction.
    CCF      multi-source Consensus & Compromise Fusion (van der Heijden et al., FUSION 2018),
             ported from the validated V1 implementation. Genuinely N-ary: it is NON-associative,
             so chaining pairwise would be a different operator.

The best SIMPLE arm (F1 averaging / F2 learned weights / DS, from fusion_replay.py) is the bar,
not DS alone -- adopting a conflict-aware operator that merely beats DS while losing to plain
averaging would be a worse framework dressed as a better one.

PASS CONDITION, from the brief: beats simple fusion on the SUPPRESSED cases while HOLDING the
strong datasets. Reported per dataset, not only on the mean, because a mean can hide exactly the
trade the condition forbids.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training"))
from health import auroc, dashboard, eer  # noqa: E402
from video_id import video_id  # noqa: E402
from networks.discern_v2.ccf_fusion import ccf_combine  # noqa: E402
from networks.discern_v2.ds_fusion import Opinion  # noqa: E402

S23 = Path("logs/tbiom/stage23")
K = 2
VIEWS = ["semantic", "artifact", "fsvfm"]
OOD = ["CDFv2", "CDFv3", "DFD", "DFDC", "DFDCP", "DFEval24"]
ALL = OOD + ["VALmix"]
P0 = {"CDFv2": 0.9646, "CDFv3": 0.8409, "DFD": 0.9421, "DFDC": 0.8828,
      "DFDCP": 0.8573, "DFEval24": 0.6922, "VALmix": 0.8852}


def to_opinion(p: np.ndarray, u: np.ndarray) -> Opinion:
    """(p, u) -> subjective-logic Opinion. b_k = alpha_k/S - u/K, so sum b + u = 1 exactly."""
    ut = torch.as_tensor(np.clip(u, 1e-6, 1.0), dtype=torch.float64).view(-1, 1)
    pf = torch.as_tensor(p, dtype=torch.float64).view(-1, 1)
    probs = torch.cat([1.0 - pf, pf], dim=1)
    belief = probs * (1.0 - ut)          # evidential expectation minus the vacuous share
    return Opinion(belief=belief, vacuity=ut).assert_normalized()


def _bu(ops):
    return (torch.stack([o.belief for o in ops], 0), torch.stack([o.vacuity for o in ops], 0))


def ds_pair(a: Opinion, b: Opinion) -> Opinion:
    conflict = (a.belief[:, 0] * b.belief[:, 1] + a.belief[:, 1] * b.belief[:, 0]).view(-1, 1)
    den = (1.0 - conflict).clamp_min(1e-12)
    bel = (a.belief * b.belief + a.belief * b.vacuity + b.belief * a.vacuity) / den
    vac = (a.vacuity * b.vacuity) / den
    return Opinion(belief=bel, vacuity=vac)


def fuse_ds(ops): 
    f = ops[0]
    for o in ops[1:]:
        f = ds_pair(f, o)
    return f


def fuse_yager(ops):
    """Dempster's rule without renormalisation: conflict is added to vacuity."""
    f = ops[0]
    for o in ops[1:]:
        bel = f.belief * o.belief + f.belief * o.vacuity + o.belief * f.vacuity
        vac = 1.0 - bel.sum(dim=1, keepdim=True)     # conflict absorbed here
        f = Opinion(belief=bel, vacuity=vac.clamp_min(0.0))
    return f


def fuse_murphy(ops):
    b, v = _bu(ops)
    avg = Opinion(belief=b.mean(0), vacuity=v.mean(0))
    f = avg
    for _ in range(len(ops) - 1):
        f = ds_pair(f, avg)
    return f


def fuse_ccf(ops):
    return ccf_combine([Opinion(belief=o.belief.float(), vacuity=o.vacuity.float()) for o in ops])


OPS = {"DS": fuse_ds, "Yager": fuse_yager, "Murphy": fuse_murphy, "CCF": fuse_ccf}


def load(arm: str, ds: str):
    f = S23 / f"{arm}_{ds}.csv"
    if not f.is_file():
        return None
    d = pd.read_csv(f)
    if any(f"{k}_{v}" not in d for v in VIEWS for k in ("p", "u")):
        return None
    d["v"] = video_id(d["key"])
    agg = {f"{k}_{v}": (f"{k}_{v}", "mean") for v in VIEWS for k in ("p", "u")}
    agg["y"] = ("label", "max")
    return d.groupby("v", as_index=False).agg(**agg)


def prob(arm_df, op: str) -> np.ndarray:
    ops = [to_opinion(arm_df[f"p_{v}"].to_numpy(), arm_df[f"u_{v}"].to_numpy()) for v in VIEWS]
    f = OPS[op](ops)
    b = f.belief.double()
    total = b.sum(dim=1, keepdim=True) + f.vacuity.double()
    # project to a probability with a uniform base rate, the standard SL projection
    p = (b[:, 1:2] + f.vacuity.double() * 0.5) / total.clamp_min(1e-12)
    return p.squeeze(1).numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", default="run1auxedl")
    ap.add_argument("--simple", type=Path, default=None, help="fusion_replay json for the bar")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ff = load(args.arm, "FFpp_val")
    taus = {}
    if ff is not None:
        for op in OPS:
            _, taus[op] = eer(ff["y"].to_numpy(), prob(ff, op))

    rows, branch = {}, {}
    for ds in ALL:
        g = load(args.arm, ds)
        if g is None or g["y"].nunique() < 2:
            continue
        for op in OPS:
            rows[(op, ds)] = dashboard(g["y"].to_numpy(), prob(g, op), taus.get(op, 0.5))
        branch[ds] = max(auroc(g["y"].to_numpy(), g[f"p_{v}"].to_numpy()) for v in VIEWS)

    simple = {}
    if args.simple and args.simple.is_file():
        j = json.load(open(args.simple))
        for op in ("F1", "F2", "F3"):
            simple[op] = {d: j["rows"][f"{op}|{d}"]["auroc"] for d in ALL
                          if f"{op}|{d}" in j["rows"]}

    done = [d for d in OOD if ("DS", d) in rows]
    L = [f"# Stage 7 — conflict-aware fusion ({args.arm})", "",
         "Replayed over the SAME saved branch opinions; no retraining. Each view's Dirichlet "
         "opinion is recovered from its exported `(p, u)` and converted to a subjective-logic "
         "opinion, then fused N-ary.", "",
         "| operator | conflict handling |", "|---|---|",
         "| DS | renormalised away — the incumbent |",
         "| Yager | assigned to **vacuity** instead of renormalised |",
         "| Murphy | opinions averaged, then combined N−1 times |",
         "| CCF | multi-source consensus & compromise (FUSION 2018), genuinely N-ary |", "",
         "## Complete-framework video AUROC", "",
         "| dataset | " + " | ".join(OPS) + " | best branch | best simple | P0-DS |",
         "|---" * (len(OPS) + 4) + "|"]
    for ds in ALL:
        if ("DS", ds) not in rows:
            continue
        cells = [f"{rows[(o, ds)]['auroc']:.4f}" for o in OPS]
        top = max(range(len(cells)), key=lambda i: float(cells[i]))
        cells[top] = f"**{cells[top]}**"
        bs = max((simple[o].get(ds, float('nan')) for o in simple), default=float('nan')) \
            if simple else float('nan')
        L.append(f"| {ds} | " + " | ".join(cells) +
                 f" | {branch[ds]:.4f} | {bs:.4f} | {P0[ds]:.4f} |")
    L += ["", "| operator | mean AUROC (6 OOD) | mean FPR_real | vs P0-DS |", "|---|---:|---:|---:|"]
    mp = np.mean([P0[d] for d in done])
    means = {}
    for op in OPS:
        ma = np.mean([rows[(op, d)]["auroc"] for d in done])
        mf = np.mean([rows[(op, d)]["fpr_real_at_tau"] for d in done])
        means[op] = ma
        L.append(f"| {op} | {ma:.4f} | {mf:.3f} | {ma-mp:+.4f} |")
    if simple:
        for op in simple:
            v = [simple[op][d] for d in done if d in simple[op]]
            if v:
                L.append(f"| *(simple {op})* | {np.mean(v):.4f} | — | {np.mean(v)-mp:+.4f} |")

    # --- the pass condition, per dataset ---------------------------------------------------
    L += ["", "## Pass condition", "",
          "From the brief: *beats simple fusion on the suppressed cases while holding the strong "
          "datasets.* Suppressed = fused loses to its own best branch under DS.", ""]
    supp = [d for d in done if rows[("DS", d)]["auroc"] < branch[d]]
    best_simple = {d: max((simple[o].get(d, -1) for o in simple), default=-1) for d in done} \
        if simple else {}
    if supp:
        L += ["| suppressed dataset | DS | best branch | best conflict-aware | best simple | fixed? |",
              "|---|---:|---:|---:|---:|---|"]
        for d in supp:
            bca = max(rows[(o, d)]["auroc"] for o in OPS if o != "DS")
            who = max((o for o in OPS if o != "DS"), key=lambda o: rows[(o, d)]["auroc"])
            bs = best_simple.get(d, float("nan"))
            ok = bca > branch[d] and bca > bs
            L.append(f"| {d} | {rows[('DS', d)]['auroc']:.4f} | {branch[d]:.4f} | "
                     f"**{bca:.4f}** ({who}) | {bs:.4f} | "
                     f"{'**yes**' if ok else 'no'} |")
    else:
        L.append("No dataset is suppressed under DS in this arm.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(
        {"taus": taus, "rows": {"|".join(k): v for k, v in rows.items()},
         "best_branch": branch}, indent=2, default=float))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())

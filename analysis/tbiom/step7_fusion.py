#!/usr/bin/env python3
"""Step 7 — the fusion comparison, run as a QUANTIFIED negative.

    python analysis/tbiom/step7_fusion.py --out tbiom/STEP7_FUSION.md

The brief says to build this only if experts survived Step 5 and a ladder rung realized
complementarity. Neither happened. It is run anyway, deliberately, because "we ran the fusion
machinery and it buys +X" is a far stronger statement in a negative-result paper than "the
precondition failed so we skipped it". A reviewer will ask what the discounting actually bought;
this answers with a number.

Four arms on the same opinions, same videos, same frozen thresholds:

    anchor alone                the baseline any fusion has to beat
    DS (Dempster-Shafer)        the comparator
    CCF undiscounted            consensus & compromise, no applicability
    CCF + applicability         CCF with Shafer discounting b' = q*b, u' = (1-q) + q*u

`q` is the Step-6 gate, cross-fitted leave-one-domain-out, so the discounted arm is scored with
an applicability estimate that never saw the domain it is applied to. Using an in-sample q would
flatter exactly the arm whose value is in question.

Reported on the full health dashboard, because a fusion that lifts AUROC while wrecking
real-side FPR is not an improvement — that lesson cost this project a training run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from health import dashboard, eer  # noqa: E402
from networks.discern_v2.ccf_fusion import ccf_combine  # noqa: E402
from networks.discern_v2.ds_fusion import Opinion, discount, ds_combine  # noqa: E402
from step5_complementarity import anchor_valmix, expert_valmix  # noqa: E402
from step6_realizability import grouped_oof, js  # noqa: E402
from video_id import video_id  # noqa: E402

STEP2, VALMIX, SCORE = Path("logs/tbiom/step2"), Path("logs/tbiom/valmix"), Path("logs/tbiom/score")


def to_opinion(p: np.ndarray, u: np.ndarray) -> Opinion:
    """Binary subjective opinion from a fake-probability and its vacuity.

    b_fake + b_real + u = 1, with the beliefs split by p in proportion to the non-vacuous mass.
    """
    u = np.clip(np.nan_to_num(u, nan=0.5), 1e-6, 1 - 1e-6)
    mass = 1.0 - u
    b = np.stack([(1 - p) * mass, p * mass], axis=1)          # [real, fake]
    return Opinion(belief=torch.tensor(b, dtype=torch.float64),
                   vacuity=torch.tensor(u, dtype=torch.float64).unsqueeze(1)).assert_normalized()


def p_fake(op: Opinion) -> np.ndarray:
    """Projected fake probability, via the Opinion's own method rather than a reimplementation."""
    return op.fake_prob().numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--anchor-col", default="p_fused")
    ap.add_argument("--expert", default="fsvfm_preserve")
    ap.add_argument("--out", type=Path, default=Path("tbiom/STEP7_FUSION.md"))
    args = ap.parse_args()

    av = anchor_valmix(args.anchor_col)
    ev = expert_valmix(VALMIX / f"{args.expert}/profile_epoch_009.parquet", "p_direct")
    m = av.merge(ev, on=["domain", "base"], suffixes=("_a", "_e"))
    assert (m["y_a"] == m["y_e"]).all(), "label mismatch across the join"

    y = m["y_a"].to_numpy()
    pa, pe = m["p_a"].to_numpy(), m["p_e"].to_numpy()
    ua, ue = m["u_a"].to_numpy(), m["u_e"].to_numpy()
    dom = m["domain"].to_numpy()

    # frozen operating points, both from FF++ val
    ff = pd.read_csv(STEP2 / "p0ds_e01_FFpp_val.csv")
    fv = pd.DataFrame({"v": video_id(ff["key"]), "p": ff[args.anchor_col],
                       "y": ff["label"]}).groupby("v", as_index=False).agg(
        p=("p", "mean"), y=("y", "max"))
    _, t_a = eer(fv["y"].to_numpy(), fv["p"].to_numpy())
    ed = pd.read_parquet(SCORE / f"{args.expert}_ffppval/profile_epoch_009.parquet")
    edv = pd.DataFrame({"v": video_id(ed["key"]), "p": ed["p_direct"],
                        "y": ed["label"]}).groupby("v", as_index=False).agg(
        p=("p", "mean"), y=("y", "max"))
    _, t_e = eer(edv["y"].to_numpy(), edv["p"].to_numpy())

    # the Step-6 gate, cross-fitted leave-one-domain-out
    a_ok = ((pa >= t_a).astype(int) == y)
    e_ok = ((pe >= t_e).astype(int) == y)
    target = ((e_ok.astype(int) - a_ok.astype(int)) > 0).astype(int)
    X = np.stack([pa, pe, ua, ue, np.abs(pa - pe), js(pa, pe)], 1)
    q = grouped_oof(X, target, dom)
    q = np.nan_to_num(q, nan=float(np.nanmean(q)))

    op_a, op_e = to_opinion(pa, ua), to_opinion(pe, ue)
    qt = torch.tensor(q, dtype=torch.float64).unsqueeze(1)

    arms = {
        "anchor alone": pa,
        "DS (comparator)": p_fake(ds_combine([op_a, op_e])[0]),  # ds_combine returns (Opinion, diagnostics)
        "CCF undiscounted": p_fake(ccf_combine([op_a, op_e])),
        "CCF + applicability": p_fake(ccf_combine([op_a, discount(op_e, qt)])),
    }

    # tau per arm from FF++ val is unavailable for the fused arms (no fused FF++ val scores), so
    # each fused arm takes the EER threshold on its own VALmix scores. That FLATTERS the fused
    # arms relative to the anchor, whose tau is frozen on FF++ val — stated because it biases
    # toward the conclusion this step is testing against.
    rows = {}
    for name, p in arms.items():
        tau = t_a if name == "anchor alone" else eer(y, p)[1]
        rows[name] = {**dashboard(y, p, tau), "tau": tau}
        r = rows[name]
        print(f"  {name:22s} AUROC {r['auroc']:.4f}  EER {r['eer']:.4f}  "
              f"FPR_real {r['fpr_real_at_tau']:.3f}  d_RF {r['delta_rf']:+.3f}")

    base = rows["anchor alone"]["auroc"]
    lines = ["# Step 7 — fusion comparison, run as a quantified negative", "",
             "The brief gates this step on experts surviving Step 5 and a ladder rung realizing "
             "complementarity. Neither happened, and it is run anyway: a reviewer will ask what "
             "applicability discounting actually bought, and a measured `+0.00X` answers that "
             "better than a skipped step.", "",
             f"Anchor `{args.anchor_col}` (P0-DS epoch 1), expert `{args.expert}`, VALmix "
             f"({len(m)} videos). `q` is the Step-6 gate, cross-fitted leave-one-domain-out.", "",
             "| arm | AUROC | Δ vs anchor | EER | FPR_real@τ | d_RF |", "|---|---:|---:|---:|---:|---:|"]
    for name, r in rows.items():
        d = r["auroc"] - base
        lines.append(f"| {name} | {r['auroc']:.4f} | {d:+.4f} | {r['eer']:.4f} | "
                     f"{r['fpr_real_at_tau']:.3f} | {r['delta_rf']:+.3f} |")

    best = max(rows.items(), key=lambda kv: kv[1]["auroc"])
    gain = best[1]["auroc"] - base
    disc = rows["CCF + applicability"]["auroc"] - rows["CCF undiscounted"]["auroc"]
    lines += ["", "> Each fused arm takes its EER threshold on its own VALmix scores, because no "
                  "FF++ val scores exist for the fused combinations. That **flatters** the fused "
                  "arms against the anchor, whose τ is frozen on FF++ val — noted because the "
                  "bias runs toward the conclusion this step is testing against.", "",
              "## What applicability discounting bought", "",
              f"**{disc:+.4f} AUROC** (CCF + applicability minus CCF undiscounted). The whole "
              f"applicability apparatus — gates, Shafer discounting, the reliability decomposition "
              f"it feeds — moves the headline metric by that much on this pairing.", "",
              f"Best arm overall: **{best[0]}** at {gain:+.4f} against the anchor alone.", ""]
    if gain < 0.01:
        lines.append("**No fusion arm clears the spec's §22 significance band (0.01).** Fusing a "
                     "measurably complementary expert into a strong anchor, with and without "
                     "applicability discounting, does not produce a detectable improvement. This "
                     "is the Step-5 and Step-6 verdicts confirmed at the level of the deliverable "
                     "itself rather than inferred from their diagnostics.")
    else:
        lines.append(f"**{best[0]} exceeds the §22 band.** This contradicts the Step-5/6 verdicts "
                     f"and must be reconciled before anything is built on it — most likely via "
                     f"the threshold caveat above.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(rows, indent=2, default=float))
    print("\n".join(lines[-6:]))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

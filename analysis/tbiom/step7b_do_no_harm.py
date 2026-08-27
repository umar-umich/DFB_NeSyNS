#!/usr/bin/env python3
"""Which experts can join WITHOUT hurting the fused system?

    python analysis/tbiom/step7b_do_no_harm.py --out tbiom/STEP7B_DO_NO_HARM.md

A different inclusion bar from Step 5's. Step 5 asked whether an expert's complementarity is
RECOVERABLE by a label-free gate — a demanding test, and everything failed it. This asks the
weaker, deployment-relevant question: does adding this expert LEAVE THE SYSTEM NO WORSE?

Both bars are legitimate and they answer different questions. An expert that neither helps nor
harms is defensible in a framework paper as an available branch; it is not defensible as evidence
that applicability gating works. Kept separate here so neither claim borrows the other's support.

Judged on BOTH axes, because a fusion that lifts AUROC while raising real-side FPR is not
harmless — that distinction cost this project a training run.
"""
from __future__ import annotations

import argparse
import itertools
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
from networks.discern_v2.ds_fusion import Opinion, ds_combine  # noqa: E402
from step5_complementarity import anchor_valmix, expert_valmix  # noqa: E402
from step7_fusion import p_fake, to_opinion  # noqa: E402
from video_id import video_id  # noqa: E402

STEP2, VALMIX, SCORE = Path("logs/tbiom/step2"), Path("logs/tbiom/valmix"), Path("logs/tbiom/score")
EXPERTS = {
    "fsvfm_preserve": (VALMIX / "fsvfm_preserve/profile_epoch_009.parquet", "p_direct"),
    "fsvfm_ordinary": (VALMIX / "fsvfm_ordinary/profile_epoch_009.parquet", "p_direct"),
    "mrvae_rate":     (VALMIX / "fsvfm_preserve_rate/profile_epoch_009.parquet", "p_rate"),
}
NOISE = 0.01     # the spec's §22 band


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--anchor-col", default="p_fused")
    ap.add_argument("--out", type=Path, default=Path("tbiom/STEP7B_DO_NO_HARM.md"))
    args = ap.parse_args()

    av = anchor_valmix(args.anchor_col)
    tables = {}
    for name, (path, col) in EXPERTS.items():
        if path.is_file():
            tables[name] = expert_valmix(path, col)

    # one merged frame so every arm is scored on exactly the same videos
    m = av.copy()
    for name, ev in tables.items():
        m = m.merge(ev.rename(columns={"p": f"p_{name}", "u": f"u_{name}", "y": f"y_{name}"}),
                    on=["domain", "base"])
    for name in tables:
        assert (m["y"] == m[f"y_{name}"]).all(), f"label mismatch for {name}"

    ff = pd.read_csv(STEP2 / "p0ds_e01_FFpp_val.csv")
    fv = pd.DataFrame({"v": video_id(ff["key"]), "p": ff[args.anchor_col],
                       "y": ff["label"]}).groupby("v", as_index=False).agg(
        p=("p", "mean"), y=("y", "max"))
    _, t_a = eer(fv["y"].to_numpy(), fv["p"].to_numpy())

    y = m["y"].to_numpy()
    op_a = to_opinion(m["p"].to_numpy(), m["u"].to_numpy())
    ops = {n: to_opinion(m[f"p_{n}"].to_numpy(), m[f"u_{n}"].to_numpy()) for n in tables}

    base = dashboard(y, m["p"].to_numpy(), t_a)
    rows = {"anchor alone": {**base, "tau": t_a}}
    for r in range(1, len(ops) + 1):
        for combo in itertools.combinations(sorted(ops), r):
            for op_name, fn in (("CCF", ccf_combine),
                                ("DS", lambda o: ds_combine(o)[0])):
                p = p_fake(fn([op_a] + [ops[c] for c in combo]))
                tau = eer(y, p)[1]
                rows[f"{op_name}: anchor + {' + '.join(combo)}"] = {
                    **dashboard(y, p, tau), "tau": tau}

    lines = ["# Which experts can join without hurting the system?", "",
             "A weaker bar than Step 5's. Step 5 asked whether complementarity is RECOVERABLE by "
             "a label-free gate and everything failed. This asks whether an expert leaves the "
             "system no worse — a deployment question, not an evidence-of-gating question. Both "
             "are kept separate so neither claim borrows the other's support.", "",
             f"Anchor: **P0-DS epoch 1, `{args.anchor_col}`**, VALmix ({len(m)} videos). "
             f"§22 noise band = {NOISE:.2f}.", "",
             "| arm | AUROC | Δ AUROC | FPR_real@τ | Δ FPR | d_RF | verdict |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for name, r in rows.items():
        dA = r["auroc"] - base["auroc"]
        dF = r["fpr_real_at_tau"] - base["fpr_real_at_tau"]
        if name == "anchor alone":
            v = "—"
        elif dA < -NOISE or dF > 0.03:
            v = "**HARMS**"
        elif dA > NOISE:
            v = "**helps**"
        else:
            v = "harmless"
        lines.append(f"| {name} | {r['auroc']:.4f} | {dA:+.4f} | {r['fpr_real_at_tau']:.3f} | "
                     f"{dF:+.3f} | {r['delta_rf']:+.3f} | {v} |")

    harmless = [n for n, r in rows.items() if n != "anchor alone"
                and (r["auroc"] - base["auroc"]) >= -NOISE
                and (r["fpr_real_at_tau"] - base["fpr_real_at_tau"]) <= 0.03]
    best = max((n for n in rows if n != "anchor alone"), key=lambda n: rows[n]["auroc"])
    lines += ["", "## Reading", "",
              f"{len(harmless)} of {len(rows) - 1} arms leave the system no worse on both axes.",
              "",
              f"Best by AUROC: **{best}** at {rows[best]['auroc']:.4f} "
              f"({rows[best]['auroc'] - base['auroc']:+.4f} vs anchor alone, FPR_real "
              f"{rows[best]['fpr_real_at_tau']:.3f} vs {base['fpr_real_at_tau']:.3f}).", "",
              "**What may and may not be claimed.** A gain inside the ±0.01 band is not a "
              "demonstrated improvement, so these arms support 'this expert can be carried "
              "without cost' and NOT 'this expert improves detection'. The framework paper can "
              "include a harmless branch as an available component; it cannot cite it as evidence "
              "that applicability fusion works, which is what Steps 5-7 tested and rejected.", "",
              "> Fused arms take their EER on their own VALmix scores because no FF++ val scores "
              "exist for fused combinations, while the anchor's τ is frozen on FF++ val. That "
              "flatters the fused arms; they are judged against it anyway."]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(rows, indent=2, default=float))
    print("\n".join(lines[8:8 + len(rows) + 2]))
    print("\n".join(lines[-8:-4]))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

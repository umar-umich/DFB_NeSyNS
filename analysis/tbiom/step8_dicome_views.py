#!/usr/bin/env python3
"""Step 8 — nested vs unified fusion over DiCoME's OWN two views.

    python analysis/tbiom/step8_dicome_views.py --out tbiom/STEP8_DICOME_VIEWS.md

The brief gates the DiCoME fork on external experts surviving complementarity against DiCoME.
None did. But the step's OTHER half does not depend on that at all:

    "First compare treating DiCoME's fused output as one anchor opinion against exposing its
     internal branch opinions for one unified outer fusion, and prefer the unified formulation
     if it performs similarly or better."

That question is live and untested, and Step 1 gave a concrete reason to ask it: the ARTIFACT
view alone beats DiCoME's fused output on Celeb-DF-v2 (0.9778 vs 0.9731), DFD and DFDC, and
Step 3 found it with far better zero-shot real-side FPR (0.169 vs 0.247). A fusion whose output
is worse than one of its inputs is mishandling them.

So this step asks two things nothing has asked:

1. **Are semantic and artifact complementary to each other?** Every membership test so far
   measured EXTERNAL experts against a DiCoME anchor. The two internal views have never been put
   through the same gate, even though they are exactly a two-expert system.
2. **Nested or unified?** DiCoME computes DS(semantic, artifact) internally and we would fuse on
   top of that. Exposing both views to one outer operator is the alternative the brief prefers if
   it performs similarly or better.
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
from stage1_membership import realizable_gate, rescue_harm  # noqa: E402
from step6_realizability import grouped_oof, js  # noqa: E402
from step7_fusion import p_fake, to_opinion  # noqa: E402
from video_id import video_id  # noqa: E402

STEP1, STEP2 = Path("logs/tbiom/step1"), Path("logs/tbiom/step2")
DOMAINS = ["CDFv2val", "DFDCPval", "DFEval24val"]


def load(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    v = video_id(d["key"])
    cols = {c: (c, "mean") for c in
            ("p_semantic", "u_semantic", "p_artifact", "u_artifact", "p_fused", "u_fused")
            if c in d}
    g = pd.DataFrame({"v": v, **{c: d[c] for c in cols}, "label": d["label"]}).groupby(
        "v", as_index=False).agg(**{**cols, "y": ("label", "max")})
    return g


def domain_of(v: pd.Series) -> pd.Series:
    parts = v.str.split("/")
    return parts.str[1].str.split("-").str[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("tbiom/STEP8_DICOME_VIEWS.md"))
    args = ap.parse_args()

    vm = load(STEP2 / "p0ds_e01_VALmix.csv")
    vm["domain"] = domain_of(vm["v"])
    ffv = load(STEP2 / "p0ds_e01_FFpp_val.csv")

    taus = {}
    for c in ("p_semantic", "p_artifact", "p_fused"):
        _, taus[c] = eer(ffv["y"].to_numpy(), ffv[c].to_numpy())

    y = vm["y"].to_numpy()
    dom = vm["domain"].to_numpy()
    ps, pa_ = vm["p_semantic"].to_numpy(), vm["p_artifact"].to_numpy()
    us, ua_ = vm["u_semantic"].to_numpy(), vm["u_artifact"].to_numpy()

    # --- 1. are the two internal views complementary to each other? -------------------------
    s_ok = ((ps >= taus["p_semantic"]).astype(int) == y)
    a_ok = ((pa_ >= taus["p_artifact"]).astype(int) == y)
    pooled = rescue_harm(s_ok, a_ok)                     # semantic as anchor, artifact as expert
    per_dom = {}
    for d in DOMAINS:
        i = dom == d
        if i.sum() > 20:
            per_dom[d] = rescue_harm(s_ok[i], a_ok[i])
    feats = np.stack([pa_, np.abs(pa_ - 0.5), ua_], 1)
    gate = realizable_gate(ps, pa_, feats, y, dom, taus["p_semantic"], taus["p_artifact"])

    # --- 2. nested vs unified ----------------------------------------------------------------
    op_s, op_a = to_opinion(ps, us), to_opinion(pa_, ua_)
    target = ((a_ok.astype(int) - s_ok.astype(int)) > 0).astype(int)
    X = np.stack([ps, pa_, us, ua_, np.abs(ps - pa_), js(ps, pa_)], 1)
    q = np.nan_to_num(grouped_oof(X, target, dom), nan=0.5)
    qt = torch.tensor(q, dtype=torch.float64).unsqueeze(1)

    arms = {
        "semantic alone": ps,
        "artifact alone": pa_,
        "NESTED — DiCoME's own DS(sem, art)": vm["p_fused"].to_numpy(),
        "UNIFIED — outer DS(sem, art)": p_fake(ds_combine([op_s, op_a])[0]),
        "UNIFIED — outer CCF(sem, art)": p_fake(ccf_combine([op_s, op_a])),
        "UNIFIED — CCF + applicability": p_fake(ccf_combine([op_s, discount(op_a, qt)])),
    }
    rows = {}
    for name, p in arms.items():
        tau = taus.get({"semantic alone": "p_semantic", "artifact alone": "p_artifact",
                        "NESTED — DiCoME's own DS(sem, art)": "p_fused"}.get(name, ""),
                       eer(y, p)[1])
        rows[name] = {**dashboard(y, p, tau), "tau": tau}
        r = rows[name]
        print(f"  {name:38s} AUROC {r['auroc']:.4f}  FPR_real {r['fpr_real_at_tau']:.3f}  "
              f"d_RF {r['delta_rf']:+.3f}")

    margin = (pooled["p_expert_right_given_anchor_wrong"]
              - pooled["p_expert_wrong_given_anchor_right"])
    rec = gate.get("recovered_fraction")
    nested = rows["NESTED — DiCoME's own DS(sem, art)"]["auroc"]
    best_uni = max((n for n in rows if n.startswith("UNIFIED")), key=lambda n: rows[n]["auroc"])

    lines = ["# Step 8 — nested vs unified fusion over DiCoME's own two views", "",
             "The DiCoME fork is gated on external experts surviving complementarity against "
             "DiCoME, and none did. But the step's other half — nested versus unified fusion — "
             "does not depend on that, and Step 1 gave a concrete reason to ask it: the artifact "
             "view ALONE beats DiCoME's fused output on Celeb-DF-v2, DFD and DFDC. A fusion whose "
             "output is worse than one of its inputs is mishandling them.", "",
             "## 1. Are the two internal views complementary to each other?", "",
             "Every membership test so far measured EXTERNAL experts against a DiCoME anchor. The "
             "two internal views are themselves a two-expert system and have never been put "
             "through the same gate. Semantic as anchor, artifact as expert, VALmix:", "",
             "| P(art right \\| sem wrong) | P(art wrong \\| sem right) | margin | ceiling | realizable recovery |",
             "|---:|---:|---:|---:|---:|",
             f"| {pooled['p_expert_right_given_anchor_wrong']:.3f} | "
             f"{pooled['p_expert_wrong_given_anchor_right']:.3f} | {margin:+.3f} | "
             f"{gate['ceiling_accuracy']:.3f} | {'n/a' if rec is None else f'{rec:.1%}'} |", "",
             "| domain | sem acc | art acc | margin |", "|---|---:|---:|---:|"]
    for d, f in per_dom.items():
        mg = (f["p_expert_right_given_anchor_wrong"] - f["p_expert_wrong_given_anchor_right"])
        lines.append(f"| {d} | {f['anchor_accuracy']:.3f} | {f['expert_accuracy']:.3f} | {mg:+.3f} |")

    lines += ["", "## 2. Nested vs unified", "",
              "| arm | AUROC | Δ vs nested | FPR_real@τ | d_RF |", "|---|---:|---:|---:|---:|"]
    for name, r in rows.items():
        lines.append(f"| {name} | {r['auroc']:.4f} | {r['auroc'] - nested:+.4f} | "
                     f"{r['fpr_real_at_tau']:.3f} | {r['delta_rf']:+.3f} |")

    lines += ["", "## Verdict", ""]
    d_uni = rows[best_uni]["auroc"] - nested
    d_art = rows["artifact alone"]["auroc"] - nested
    lines.append(
        f"Best unified arm is **{best_uni}** at {d_uni:+.4f} against DiCoME's own nested DS. "
        f"Artifact alone is {d_art:+.4f} against it.")
    lines.append("")
    if d_uni > 0.005 or d_art > 0.005:
        lines.append(
            "**The nested formulation is not the right default.** Exposing DiCoME's internal "
            "views to one outer operator — or simply taking the artifact view — beats fusing on "
            "top of its own DS output. Any later integration should consume the internal "
            "opinions, which is what the brief said to prefer if it performed similarly or "
            "better.")
    else:
        lines.append(
            "**Unified fusion does not beat the nested formulation on this split**, so DiCoME's "
            "own DS output is an acceptable single anchor opinion here. The Step-1 observation "
            "that artifact beats fused on several TEST sets does not reproduce as an advantage on "
            "VALmix, which is the honest place to check it — and that discrepancy is itself worth "
            "carrying, since Step 1's was a test-set observation and this is a development one.")
    if rec is not None and rec >= 0.25 and margin > 0.05:
        lines += ["", "**Note the internal pair DOES pass the membership gate that every external "
                      "expert failed.** The complementarity the architecture needs is already "
                      "inside DiCoME, between its own two views, rather than in any bolted-on "
                      "specialist. That reframes the fusion question rather than closing it."]
    else:
        lines += ["", f"The internal pair does not clear the membership bar either "
                      f"(margin {margin:+.3f}, recovery "
                      f"{'n/a' if rec is None else f'{rec:.1%}'} against 25%), so the negative "
                      f"result extends to DiCoME's own decomposition — it is not that we picked "
                      f"the wrong experts."]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(
        {"pooled": pooled, "gate": gate, "arms": rows}, indent=2, default=float))
    print("\n".join(lines[-6:]))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

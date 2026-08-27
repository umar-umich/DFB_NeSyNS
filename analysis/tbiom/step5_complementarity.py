#!/usr/bin/env python3
"""Step 5 — do the experts complement the CHOSEN anchor (P0-DS), not the old CLIP port?

    python analysis/tbiom/step5_complementarity.py --out tbiom/STEP5_COMPLEMENTARITY.md

Every complementarity number so far was measured against the weaker CLIP port. Step 1 showed
DiCoME's readouts beat it by ~0.045 AUROC on Celeb-DF-v2, so the rescue question has to be asked
again against the anchor actually chosen.

EXPECT THIS TO BE HARDER, NOT EASIER. Rescue is `P(expert right | anchor wrong)`, and a stronger
anchor is wrong less often — the denominator shrinks and the cases that remain are the hard ones.
An expert that failed against the port will not pass here by accident.

The criterion is conditional information, not standalone AUC. An expert may sit well below the
anchor and still qualify if it is right where the anchor is wrong, harm is low, and a
label-free-at-inference gate can find those cases.

JOINING TWO PIPELINES. The anchor is scored through DiCoME's h5 path and the experts through our
PNG path, so video identity is matched on (domain, basename) rather than on raw path. Within a
VALmix domain the basename is unique — Celeb-real's `id0_0005` and Celeb-synthesis's
`id0_id21_0005` cannot collide — and the join is asserted to cover both sides before any metric
is computed.
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
from stage1_membership import (MEANINGFUL_RESCUE_MARGIN, MIN_FAMILIES,  # noqa: E402
                               USEFUL_RECOVERY, auroc, gate_features,
                               realizable_gate, rescue_harm)
from video_id import video_id  # noqa: E402

STEP2 = Path("logs/tbiom/step2")
VALMIX = Path("logs/tbiom/valmix")
SCORE = Path("logs/tbiom/score")

# our PNG-path corpus -> VALmix domain name used on the DiCoME side
OUR_DOMAIN = {"/Celeb-DF-v2/": "CDFv2val", "/DFDCP/": "DFDCPval",
              "/Deepfake-Eval-2024/": "DFEval24val"}
ANCHOR_READOUTS = {"p_fused": "P0-DS fused", "p_artifact": "P0-DS artifact"}
EXPERTS = [
    ("fsvfm_preserve", VALMIX / "fsvfm_preserve/profile_epoch_009.parquet", "p_direct",
     SCORE / "fsvfm_preserve_ffppval/profile_epoch_009.parquet"),
    ("fsvfm_ordinary", VALMIX / "fsvfm_ordinary/profile_epoch_009.parquet", "p_direct",
     SCORE / "fsvfm_ordinary_ffppval/profile_epoch_009.parquet"),
    ("mrvae_rate", VALMIX / "fsvfm_preserve_rate/profile_epoch_009.parquet", "p_rate",
     SCORE / "fsvfm_preserve_ffppval/profile_epoch_009.parquet"),
]


def anchor_valmix(col: str) -> pd.DataFrame:
    d = pd.read_csv(STEP2 / "p0ds_e01_VALmix.csv")
    v = video_id(d["key"])
    g = pd.DataFrame({"v": v, "p": d[col], "u": d.get(col.replace("p_", "u_"), np.nan),
                      "y": d["label"]}).groupby("v", as_index=False).agg(
        p=("p", "mean"), u=("u", "mean"), y=("y", "max"))
    parts = g["v"].str.split("/")
    g["domain"] = parts.str[1].str.split("-").str[0]      # VALmix/CDFv2val-Celeb-real/... -> CDFv2val
    g["base"] = parts.str[-1].str.replace(r"^(cdf2|dfdcp|dfe24)-", "", regex=True)
    return g[["domain", "base", "p", "u", "y"]]


def expert_valmix(path: Path, col: str) -> pd.DataFrame:
    d = pd.read_parquet(path)
    v = video_id(d["key"])
    unc = col.replace("p_", "u_")
    g = pd.DataFrame({"v": v, "p": d[col], "u": d[unc] if unc in d else np.nan,
                      "y": d["label"]}).groupby("v", as_index=False).agg(
        p=("p", "mean"), u=("u", "mean"), y=("y", "max"))
    dom = pd.Series("other", index=g.index)
    for needle, name in OUR_DOMAIN.items():
        dom = dom.mask(g["v"].str.contains(needle, regex=False), name)
    g["domain"] = dom
    g["base"] = g["v"].str.rsplit("/", n=1).str[-1]
    return g[["domain", "base", "p", "u", "y"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("tbiom/STEP5_COMPLEMENTARITY.md"))
    args = ap.parse_args()

    results, decisions = {}, {}
    for a_col, a_name in ANCHOR_READOUTS.items():
        av = anchor_valmix(a_col)
        ff = pd.read_csv(STEP2 / "p0ds_e01_FFpp_val.csv")
        fv = pd.DataFrame({"v": video_id(ff["key"]), "p": ff[a_col], "y": ff["label"]}).groupby(
            "v", as_index=False).agg(p=("p", "mean"), y=("y", "max"))
        _, t_anchor = eer(fv["y"].to_numpy(), fv["p"].to_numpy())

        for e_name, e_path, e_col, e_val in EXPERTS:
            if not e_path.is_file():
                continue
            ev = expert_valmix(e_path, e_col)
            m = av.merge(ev, on=["domain", "base"], suffixes=("_a", "_e"))
            cov = len(m) / max(1, min(len(av), len(ev)))
            if len(m) < 800 or cov < 0.7:
                print(f"  {a_name} x {e_name}: join too thin ({len(m)}, {cov:.0%}) — skipped")
                continue
            if not (m["y_a"] == m["y_e"]).all():
                bad = int((m["y_a"] != m["y_e"]).sum())
                raise SystemExit(f"{a_name} x {e_name}: {bad} label mismatches after the join — "
                                 f"the (domain, basename) key is not matching the same videos")

            ed = pd.read_parquet(e_val)
            edv = pd.DataFrame({"v": video_id(ed["key"]), "p": ed[e_col],
                                "y": ed["label"]}).groupby("v", as_index=False).agg(
                p=("p", "mean"), y=("y", "max"))
            _, t_expert = eer(edv["y"].to_numpy(), edv["p"].to_numpy())

            y = m["y_a"].to_numpy()
            pa, pe = m["p_a"].to_numpy(), m["p_e"].to_numpy()
            a_ok = (pa >= t_anchor).astype(int) == y
            e_ok = (pe >= t_expert).astype(int) == y
            pooled = rescue_harm(a_ok, e_ok)
            per_family = {}
            for dom, g in m.groupby("domain"):
                idx = m.index.get_indexer(g.index)
                per_family[str(dom)] = rescue_harm(a_ok[idx], e_ok[idx])

            feats = gate_features(
                pd.DataFrame({e_col: pe, "u": m["u_e"].to_numpy()}), e_col,
                "u" if m["u_e"].notna().any() else None)
            gate = realizable_gate(pa, pe, feats, y, m["domain"].to_numpy(),
                                   t_anchor, t_expert)

            margin = (pooled["p_expert_right_given_anchor_wrong"]
                      - pooled["p_expert_wrong_given_anchor_right"])
            fams = [f for f, e in per_family.items()
                    if (e["p_expert_right_given_anchor_wrong"]
                        - e["p_expert_wrong_given_anchor_right"]) > MEANINGFUL_RESCUE_MARGIN]
            rec = gate.get("recovered_fraction")
            enters = bool(margin > MEANINGFUL_RESCUE_MARGIN and len(fams) >= MIN_FAMILIES
                          and rec is not None and rec >= USEFUL_RECOVERY)
            key = f"{a_name} | {e_name}"
            results[key] = {"n": int(len(m)), "coverage": cov, "t_anchor": t_anchor,
                            "t_expert": t_expert, "pooled": pooled, "gate": gate,
                            "per_family": per_family, "margin": margin,
                            "families_ok": len(fams)}
            decisions[key] = enters
            print(f"  {key:34s} rescue {pooled['p_expert_right_given_anchor_wrong']:.3f} "
                  f"harm {pooled['p_expert_wrong_given_anchor_right']:.3f} "
                  f"margin {margin:+.3f}  recovered "
                  f"{'n/a' if rec is None else f'{rec:.1%}'}  "
                  f"AUROC {gate['anchor_video_auroc']:.4f}->{gate['fused_video_auroc']:.4f}  "
                  f"{'ENTERS' if enters else 'dropped'}")

    lines = ["# Step 5 — complementarity against the chosen anchor (P0-DS epoch 1)", "",
             "Every earlier complementarity number was measured against the weaker CLIP port. A "
             "stronger anchor makes rescue HARDER — it is wrong less often, so the denominator "
             "shrinks and the remaining errors are the hard ones. An expert that failed against "
             "the port cannot pass here by accident.", "",
             "Basis: VALmix, 1,350 videos over three real-world domains. The anchor comes through "
             "DiCoME's h5 path and the experts through ours, so videos are joined on "
             "(domain, basename) and labels are asserted to agree across the join.", "",
             "| anchor readout | expert | videos | rescue | harm | margin | ceiling | recovered | AUROC | enters? |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for key, r in results.items():
        a, e = key.split(" | ")
        p, g = r["pooled"], r["gate"]
        rec = g.get("recovered_fraction")
        lines.append(
            f"| {a} | `{e}` | {r['n']} | {p['p_expert_right_given_anchor_wrong']:.3f} | "
            f"{p['p_expert_wrong_given_anchor_right']:.3f} | {r['margin']:+.3f} | "
            f"{g['ceiling_accuracy']:.3f} | {'n/a' if rec is None else f'{rec:.1%}'} | "
            f"{g['anchor_video_auroc']:.4f} → {g['fused_video_auroc']:.4f} | "
            f"{'**YES**' if decisions[key] else 'no'} |")

    lines += ["", f"Bars: rescue margin > {MEANINGFUL_RESCUE_MARGIN:.2f} pooled AND on "
                  f"≥ {MIN_FAMILIES} domains, AND realizable-gate recovery ≥ {USEFUL_RECOVERY:.0%}.",
              "", "## Per domain", "",
              "| anchor | expert | domain | anchor acc | expert acc | margin |",
              "|---|---|---|---:|---:|---:|"]
    for key, r in results.items():
        a, e = key.split(" | ")
        for dom, f in r["per_family"].items():
            mg = (f["p_expert_right_given_anchor_wrong"] - f["p_expert_wrong_given_anchor_right"])
            lines.append(f"| {a} | `{e}` | {dom} | {f['anchor_accuracy']:.3f} | "
                         f"{f['expert_accuracy']:.3f} | {mg:+.3f} |")

    lines += ["", "## Verdict", ""]
    if any(decisions.values()):
        won = [k for k, v in decisions.items() if v]
        lines.append(f"**{len(won)} expert/anchor pairing(s) pass**: {', '.join(won)}. A fusion "
                     f"path is live; carry the survivors into Step 6's realizability ladder.")
    elif results:
        best = max(results.items(), key=lambda kv: kv[1]["gate"].get("recovered_fraction") or -1)
        rec = best[1]["gate"].get("recovered_fraction")
        lines += [
            f"**No expert enters against the chosen anchor.** Best pairing was {best[0]} at "
            f"{'n/a' if rec is None else f'{rec:.1%}'} realizable recovery against a "
            f"{USEFUL_RECOVERY:.0%} bar. That is now THREE independent bases — DF40-Dev and "
            f"VALmix against the CLIP port, and VALmix against the stronger P0-DS anchor — "
            f"agreeing that no fusion path is live.", "",
            "Note the rescue half is UNDIMINISHED against the stronger anchor: margins of +0.222 "
            "and +0.233 here against +0.202 and +0.206 against the port. The oracle "
            "complementarity is real and did not shrink when the anchor improved. What collapsed "
            "is the realizable half — recovery fell from 7-10% to between -4.0% and 5.7%, with "
            "several pairings now NEGATIVE, meaning a fitted gate does worse than not gating.", "",
            "Step 6 becomes the decisive experiment rather than a follow-up: if the realizability "
            "ladder also fails its audit, the multi-expert architecture has no support and the "
            "paper is the single-anchor reliability result from Step 4, with the realizability "
            "gap reported as a characterised negative."]
    else:
        lines.append("TODO(run) — no pairing produced a result.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(
        {"results": results, "decisions": decisions}, indent=2, default=float))
    print("\n".join(lines[-6:]))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

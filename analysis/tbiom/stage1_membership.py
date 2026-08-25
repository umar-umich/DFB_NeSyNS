#!/usr/bin/env python3
"""Stage 1 — BRANCH MEMBERSHIP GATE. Which experts enter T-BIOM at all.

    🔴 UMAR-RUNS (CPU, seconds):

    python analysis/tbiom/stage1_membership.py \
        --anchor  clip   logs/tbiom/score/clip_df40dev/per_sample.parquet \
        --expert  fsvfm  logs/tbiom/score/fsvfm_preserve_df40dev/profile.parquet p_direct \
        --expert  mrvae  logs/tbiom/score/mrvae_df40dev/rate.parquet p_rate \
        --threshold-source logs/tbiom/score/clip_ffppval/per_sample.parquet \
        --out tbiom

Decides membership on complementarity with the CLIP anchor, BEFORE any fusion training. It is the
hard gate: an expert that shows neither rescue nor recoverable complementarity is dropped from the
architecture entirely. Three branches for symmetry is not a requirement.

The three numbers, in the brief's corrected forms
-------------------------------------------------
**Rescue as conditional probabilities, both directions.** `P(expert right | anchor wrong)` against
`P(expert wrong | anchor right)`, per family and pooled. The first must clearly exceed the second,
or the expert harms about as often as it helps. Reporting only the first is how a harmful expert
looks useful.

**The ceiling as `1 - P(both wrong)`, NOT an oracle AUROC.** A label-aware per-sample selector
produces a ranking that is a function of the labels, so its "AUROC" is an artificial number that
approaches 1 and makes every realizable recovery look negligible. The honest ceiling is the
fraction of samples at least one of the two gets right — an accuracy, which is what a perfect
router could actually deliver.

**Realizable-gate recovery, which is the go signal.** A cheap cross-fit logistic gate on the
expert's own evidence and confidence only, reported as the actual fused AUC and as the fraction of
the ceiling-minus-anchor gain it recovers. A large ceiling with near-zero realizable recovery means
the applicability signal is not visible in the features, and the reasoner would be speculative.

Threshold discipline
--------------------
Rescue and harm are decisions, so they need an operating threshold. ONE threshold is derived from
the permitted protocol source (FF++ val) and frozen across every DF40 family — never tuned per
family, which would leak the target distribution into the decision. `--threshold-source` supplies
it; without one the script refuses rather than silently using 0.5, because a mis-set threshold
moves every conditional probability in this file.

Fresh both sides
----------------
The brief is explicit, and this project was bitten twice: never inherit the anchor's failure rows
from a P0-DS, B1 or V1 table. The anchor's per-row results must be recomputed in the same harness
as the experts'. `--anchor` therefore takes a freshly scored export, and the script records its
provenance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "analysis" / "discern_v2" / "phase2"))

NOISE = 0.01
MEANINGFUL_RESCUE_MARGIN = 0.05   # P(right|anchor wrong) must beat P(wrong|anchor right) by this
USEFUL_RECOVERY = 0.25            # fraction of the ceiling-minus-anchor gain a gate must recover
MIN_FAMILIES = 2


def video_frame(df: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    """Aggregate to video level once, so every number below is on the same unit."""
    return df.groupby(["dataset", "video_id"], as_index=False).agg(
        p=(prob_col, "mean"), y=("label", "max"))


def eer_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Equal-error-rate threshold on the protocol source. One number, frozen everywhere."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y, p)
    i = int(np.nanargmin(np.abs((1 - tpr) - fpr)))
    return float(thr[i])


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def rescue_harm(anchor_ok: np.ndarray, expert_ok: np.ndarray) -> dict:
    """The two conditional probabilities, and the honest ceiling."""
    a_wrong, a_right = ~anchor_ok, anchor_ok
    return {
        "n": int(len(anchor_ok)),
        "anchor_accuracy": float(anchor_ok.mean()),
        "expert_accuracy": float(expert_ok.mean()),
        "n_anchor_wrong": int(a_wrong.sum()),
        "n_anchor_right": int(a_right.sum()),
        # the two directions, never one without the other
        "p_expert_right_given_anchor_wrong": (float(expert_ok[a_wrong].mean())
                                              if a_wrong.any() else float("nan")),
        "p_expert_wrong_given_anchor_right": (float((~expert_ok)[a_right].mean())
                                              if a_right.any() else float("nan")),
        # 1 - P(both wrong): the fraction at least one gets right. An ACCURACY, not an AUROC.
        "ceiling_union_correct": float((anchor_ok | expert_ok).mean()),
        "both_wrong": float((~anchor_ok & ~expert_ok).mean()),
    }


def gate_features(expert: pd.DataFrame, prob_col: str, unc_col: str | None) -> np.ndarray:
    """The expert's OWN evidence and confidence. Deliberately label-free and compact."""
    p = expert[prob_col].to_numpy()
    cols = [p, np.abs(p - 0.5)]
    if unc_col and unc_col in expert:
        cols.append(expert[unc_col].to_numpy())
    return np.stack(cols, axis=1)


def realizable_gate(anchor_p: np.ndarray, expert_p: np.ndarray, feats: np.ndarray,
                    y: np.ndarray, groups: np.ndarray, threshold: float) -> dict:
    """Cross-fit logistic gate, then the ACTUAL fused AUC it delivers.

    Target: would routing to the expert have been better on this sample? Grouped
    leave-one-family-out, so no sample scores its own gate and no family's siblings train it.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    anchor_ok = (anchor_p >= threshold).astype(int) == y
    expert_ok = (expert_p >= threshold).astype(int) == y
    target = (expert_ok & ~anchor_ok).astype(int)     # the expert helps exactly here

    q = np.full(len(y), np.nan)
    for g in np.unique(groups):
        held = groups == g
        if len(np.unique(target[~held])) < 2:
            q[held] = float(target[~held].mean()) if (~held).any() else 0.5
            continue
        scaler = StandardScaler().fit(feats[~held])
        model = LogisticRegression(max_iter=2000).fit(scaler.transform(feats[~held]),
                                                      target[~held])
        q[held] = model.predict_proba(scaler.transform(feats[held]))[:, 1]

    # the realizable fused score: q-weighted blend, which is what a deployed gate would give
    fused = (1 - q) * anchor_p + q * expert_p
    anchor_auc, fused_auc = auroc(y, anchor_p), auroc(y, fused)
    ceiling_acc = float((anchor_ok | expert_ok).mean())
    anchor_acc = float(anchor_ok.mean())
    fused_ok = (fused >= threshold).astype(int) == y
    headroom = ceiling_acc - anchor_acc
    return {
        "gate_auroc_vs_target": auroc(target, q),
        "target_positive_rate": float(target.mean()),
        "mean_q": float(np.mean(q)),
        "anchor_video_auroc": anchor_auc,
        "fused_video_auroc": fused_auc,
        "auroc_gain": fused_auc - anchor_auc,
        "anchor_accuracy": anchor_acc,
        "fused_accuracy": float(fused_ok.mean()),
        "ceiling_accuracy": ceiling_acc,
        "accuracy_headroom": headroom,
        "recovered_fraction": (float((fused_ok.mean() - anchor_acc) / headroom)
                               if headroom > 1e-9 else None),
        "no_headroom": bool(headroom <= 1e-9),
    }


def decide(name: str, pooled: dict, gate: dict, per_family: dict) -> dict:
    margin = (pooled["p_expert_right_given_anchor_wrong"]
              - pooled["p_expert_wrong_given_anchor_right"])
    families_with_rescue = [f for f, e in per_family.items()
                            if (e["p_expert_right_given_anchor_wrong"]
                                - e["p_expert_wrong_given_anchor_right"]) > MEANINGFUL_RESCUE_MARGIN]
    rescue_ok = margin > MEANINGFUL_RESCUE_MARGIN and len(families_with_rescue) >= MIN_FAMILIES
    rec = gate.get("recovered_fraction")
    recovery_ok = rec is not None and rec >= USEFUL_RECOVERY
    enters = bool(rescue_ok and recovery_ok)
    if enters:
        why = (f"rescue margin {margin:+.3f} on {len(families_with_rescue)} families, and the "
               f"realizable gate recovers {rec:.0%} of the ceiling-minus-anchor headroom")
    else:
        bits = []
        if not rescue_ok:
            bits.append(f"rescue margin {margin:+.3f} "
                        f"(P(right|anchor wrong) {pooled['p_expert_right_given_anchor_wrong']:.3f} "
                        f"vs P(wrong|anchor right) "
                        f"{pooled['p_expert_wrong_given_anchor_right']:.3f}) on "
                        f"{len(families_with_rescue)} families")
        if not recovery_ok:
            bits.append("no method-level headroom to recover" if gate.get("no_headroom")
                        else f"the realizable gate recovers only "
                             f"{rec:.0%} of the headroom" if rec is not None else "gate not computable")
        why = "; ".join(bits)
    return {"expert": name, "enters": enters, "justification": why,
            "rescue_margin": margin, "families_with_rescue": sorted(families_with_rescue),
            "recovered_fraction": rec}


def render(payload: dict) -> str:
    lines = [
        "# Stage 1 — branch membership gate",
        "",
        f"Anchor `{payload['anchor']['name']}` scored freshly in this harness "
        f"({payload['anchor']['n_videos']} DF40-Dev videos, "
        f"{payload['anchor']['n_families']} methods). Operating threshold "
        f"**{payload['threshold']:.4f}**, EER on `{payload['threshold_source']}`, frozen across "
        f"every family.",
        "",
        "> Rescue rows are recomputed against THIS anchor. They are not inherited from a P0-DS, "
        "B1 or V1 table — that mistake was made twice in this project, and a rescue claim is only "
        "as valid as the baseline it is measured against.",
        "",
        "## Decision", "",
        "| expert | enters? | why |", "|---|---|---|",
    ]
    for name, d in payload["decisions"].items():
        lines.append(f"| `{name}` | {'**YES**' if d['enters'] else 'no'} | {d['justification']} |")

    lines += ["", "## The three numbers", "",
              "| expert | P(right \\| anchor wrong) | P(wrong \\| anchor right) | margin | "
              "ceiling 1−P(both wrong) | anchor acc | realizable fused acc | recovered |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, e in payload["experts"].items():
        p, g = e["pooled"], e["gate"]
        rec = g.get("recovered_fraction")
        lines.append(
            f"| `{name}` | {p['p_expert_right_given_anchor_wrong']:.3f} | "
            f"{p['p_expert_wrong_given_anchor_right']:.3f} | "
            f"{p['p_expert_right_given_anchor_wrong'] - p['p_expert_wrong_given_anchor_right']:+.3f} | "
            f"{p['ceiling_union_correct']:.3f} | {g['anchor_accuracy']:.3f} | "
            f"{g['fused_accuracy']:.3f} | "
            f"{'no headroom' if g.get('no_headroom') else (f'{rec:.0%}' if rec is not None else '—')} |")
    lines += ["", "The ceiling is an ACCURACY — the fraction at least one of anchor and expert "
                  "gets right. A label-aware per-sample selector's AUROC would sit near 1 and make "
                  "every realizable recovery look negligible, which is why the brief forbids it.",
              "", "## Video AUROC, anchor vs realizable fusion", "",
              "| expert | anchor AUROC | fused AUROC | gain | gate AUROC vs target | mean q |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, e in payload["experts"].items():
        g = e["gate"]
        lines.append(f"| `{name}` | {g['anchor_video_auroc']:.4f} | {g['fused_video_auroc']:.4f} | "
                     f"{g['auroc_gain']:+.4f} | {g['gate_auroc_vs_target']:.4f} | "
                     f"{g['mean_q']:.3f} |")

    lines += ["", "## Per family (DF40-Dev method)", ""]
    for name, e in payload["experts"].items():
        lines += [f"### `{name}`", "",
                  "| method | anchor acc | expert acc | P(right\\|wrong) | P(wrong\\|right) | "
                  "margin | ceiling |", "|---|---:|---:|---:|---:|---:|---:|"]
        for fam, f in sorted(e["per_family"].items()):
            m = (f["p_expert_right_given_anchor_wrong"] - f["p_expert_wrong_given_anchor_right"])
            lines.append(f"| {fam} | {f['anchor_accuracy']:.3f} | {f['expert_accuracy']:.3f} | "
                         f"{f['p_expert_right_given_anchor_wrong']:.3f} | "
                         f"{f['p_expert_wrong_given_anchor_right']:.3f} | {m:+.3f} | "
                         f"{f['ceiling_union_correct']:.3f} |")
        lines.append("")

    entering = [n for n, d in payload["decisions"].items() if d["enters"]]
    lines += ["## What happens next", ""]
    if entering:
        lines.append(f"Stage 2 trains the anchor and the surviving expert head(s): "
                     f"**{', '.join(entering)}**. Experts that did not pass are removed from the "
                     f"architecture, not disabled by a flag.")
        if len(entering) == 1:
            lines.append("")
            lines.append("With ONE surviving specialist, Stage 3's applicability target reduces to "
                         "the brief's single-specialist form `phi_b = U(A+b) - U(A)`; the "
                         "anchor-conditioned Shapley form is not needed.")
    else:
        lines.append("**No expert passed.** Go to the reliability-centered fallback: do not build "
                     "applicability machinery over experts that carry no recoverable signal. Note "
                     "the fallback changes the reliability model's shape — with only the anchor "
                     "there is no inter-branch conflict `C` and no `U_sup`, so the risk model "
                     "becomes `g(V_sem, M_sem)`. Manufacturing C or U_sup from one branch would "
                     "be wrong.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--anchor", nargs=2, required=True, metavar=("NAME", "PARQUET"))
    ap.add_argument("--anchor-col", default="p_sem")
    ap.add_argument("--expert", nargs=3, action="append", required=True,
                    metavar=("NAME", "PARQUET", "PROB_COL"))
    ap.add_argument("--threshold-source", type=Path, required=True,
                    help="a scored export from the PERMITTED protocol (FF++ val). One EER "
                         "threshold is derived here and frozen across every family.")
    ap.add_argument("--threshold-col", default=None,
                    help="probability column in the threshold source (default: --anchor-col)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    a_name, a_path = args.anchor
    anchor_df = pd.read_parquet(a_path)
    if args.anchor_col not in anchor_df:
        raise SystemExit(f"anchor export has no `{args.anchor_col}`; columns: "
                         f"{sorted(anchor_df.columns)[:12]}")
    anchor_v = video_frame(anchor_df, args.anchor_col)

    tsrc = pd.read_parquet(args.threshold_source)
    tcol = args.threshold_col or args.anchor_col
    if tcol not in tsrc:
        raise SystemExit(f"threshold source has no `{tcol}`")
    tv = video_frame(tsrc, tcol)
    threshold = eer_threshold(tv["y"].to_numpy(), tv["p"].to_numpy())
    print(f"frozen operating threshold {threshold:.4f} (EER on {args.threshold_source})")

    experts, decisions = {}, {}
    for name, path, col in args.expert:
        edf = pd.read_parquet(path)
        if col not in edf:
            raise SystemExit(f"expert `{name}` export has no `{col}`; columns: "
                             f"{sorted(edf.columns)[:12]}")
        ev = video_frame(edf, col)
        merged = anchor_v.merge(ev, on=["dataset", "video_id"], suffixes=("_a", "_e"))
        if len(merged) < 0.9 * min(len(anchor_v), len(ev)):
            raise SystemExit(
                f"`{name}`: only {len(merged)} videos are common to the anchor "
                f"({len(anchor_v)}) and the expert ({len(ev)}). Complementarity must be computed "
                f"on the SAME videos or it mixes a sampling difference into the comparison.")
        if not (merged["y_a"] == merged["y_e"]).all():
            raise SystemExit(f"`{name}`: label mismatch between the two exports on the same videos")

        y = merged["y_a"].to_numpy()
        pa, pe = merged["p_a"].to_numpy(), merged["p_e"].to_numpy()
        a_ok = (pa >= threshold).astype(int) == y
        e_ok = (pe >= threshold).astype(int) == y

        pooled = rescue_harm(a_ok, e_ok)
        per_family = {}
        for fam, g in merged.groupby("dataset"):
            idx = merged.index.get_indexer(g.index)
            per_family[str(fam)] = rescue_harm(a_ok[idx], e_ok[idx])

        unc = {"p_direct": "u_direct", "p_traj": "u_traj", "p_spatial": "u_spatial",
               "p_rate": "u_rate", "p_sem": "u_sem"}.get(col)
        feats = gate_features(
            edf.groupby(["dataset", "video_id"], as_index=False).agg(
                **{col: (col, "mean"), **({unc: (unc, "mean")} if unc and unc in edf else {})}
            ).merge(merged[["dataset", "video_id"]], on=["dataset", "video_id"]), col, unc)
        gate = realizable_gate(pa, pe, feats, y, merged["dataset"].to_numpy(), threshold)

        experts[name] = {"pooled": pooled, "per_family": per_family, "gate": gate,
                         "parquet": str(path), "prob_col": col}
        decisions[name] = decide(name, pooled, gate, per_family)
        d = decisions[name]
        print(f"\n  {name}: {'ENTERS' if d['enters'] else 'dropped'} — {d['justification']}")
        print(f"    P(right|anchor wrong) {pooled['p_expert_right_given_anchor_wrong']:.3f} · "
              f"P(wrong|anchor right) {pooled['p_expert_wrong_given_anchor_right']:.3f} · "
              f"ceiling {pooled['ceiling_union_correct']:.3f} · "
              f"anchor acc {gate['anchor_accuracy']:.3f} -> fused {gate['fused_accuracy']:.3f}")

    payload = {
        "anchor": {"name": a_name, "parquet": str(a_path), "column": args.anchor_col,
                   "n_videos": int(len(anchor_v)),
                   "n_families": int(anchor_v["dataset"].nunique())},
        "threshold": threshold, "threshold_source": str(args.threshold_source),
        "thresholds_policy": {"rescue_margin": MEANINGFUL_RESCUE_MARGIN,
                              "useful_recovery": USEFUL_RECOVERY,
                              "min_families": MIN_FAMILIES},
        "experts": experts, "decisions": decisions,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "stage1_membership.json").write_text(json.dumps(payload, indent=2, default=str))
    (args.out / "STAGE1_MEMBERSHIP.md").write_text(render(payload))
    print(f"\nwrote {args.out}/STAGE1_MEMBERSHIP.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

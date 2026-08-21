#!/usr/bin/env python3
"""Stage 0.3 + 0.4 — oracle headroom, realizable-gate recovery, rescue and harm.

    🔴 UMAR-RUNS (CPU, minutes):

    python analysis/discern_v2/phase2/complementarity.py \
        --parquet logs/v1/eval/epoch_007_gated_diverse/per_sample_epoch_7.parquet \
                  logs/v1/eval/epoch_007_df40dev/per_sample_epoch_7.parquet \
        --out phase2/stage0

Two numbers, and the brief is explicit that the first must never be reported without the second:

  **oracle**      what is there to win if routing were perfect        (an unrealizable ceiling)
  **recovery**    how much of it an observable gate actually gets     (the decision signal)

A large oracle with near-zero recovery is a legitimate negative result — "applicability is not
inferable from the observable evidence here" — and it is the finding that would stop the
multi-specialist build at Stage 3.

Three oracles, because the denominator decides the answer
---------------------------------------------------------
The brief asks for a per-sample ideal selector. Taken literally that oracle consults the label
once per frame, so its AUROC sits near 1.0 and *every* realizable gate recovers a negligible
fraction of it — the ratio would say "no" regardless of what the gate learned. Reporting only
that would make the go/no-go a foregone conclusion rather than a measurement, so all three
denominators are computed:

  `oracle_sample`   the brief's literal per-frame ideal selector           (loosest ceiling)
  `oracle_video`    one ideal choice per video                            (a video-level router)
  `oracle_method`   the best single branch for that method, fixed         (tightest, and the one
                                                                           a deployed router could
                                                                           actually reach)

`oracle_method` is the honest denominator for a deployment claim; the other two are reported
beside it so nobody has to take that judgement on faith.

Two gate protocols, because the gap between them is the finding
---------------------------------------------------------------
  `insample`   leave-one-method-out cross-fit **on the evaluation pool itself**. The gate has
               seen sibling generators, so this is a feasibility ceiling, not a deployable number.
  `protocol`   fit on FF++ only, frozen, applied to sources it has never seen. This is what a
               deployed system would have.

Phase 1 already established that these two diverge (A2a vs A2b); collapsing them into one number
would hide exactly the thing Stage 3 needs to weigh.
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

import lib  # noqa: E402

PROTOCOL_SOURCE = "FaceForensics++"     # ground rule 2: calibration sees FF++ labels only


# --------------------------------------------------------------------------------------
# oracles
# --------------------------------------------------------------------------------------

def oracle_probabilities(df: pd.DataFrame, branches: list[str]) -> dict[str, np.ndarray]:
    """The three ceilings. Each returns a per-frame p(fake); all three consult the label."""
    y = df["label"].to_numpy().astype(float)
    P = np.stack([df[f"p_{b}"].to_numpy() for b in branches], axis=1)     # (N, B)
    loss = np.abs(P - y[:, None])                                        # distance to truth

    out = {}
    out["oracle_sample"] = P[np.arange(len(P)), loss.argmin(axis=1)]

    frame = pd.DataFrame({"v": df["video_id"].to_numpy()})
    for i, b in enumerate(branches):
        frame[f"loss_{b}"] = loss[:, i]
    per_video = frame.groupby("v")[[f"loss_{b}" for b in branches]].mean()
    best_for_video = per_video.to_numpy().argmin(axis=1)
    choice = pd.Series(best_for_video, index=per_video.index).reindex(frame["v"]).to_numpy()
    out["oracle_video"] = P[np.arange(len(P)), choice]

    best_overall = int(loss.mean(axis=0).argmin())
    out["oracle_method"] = P[:, best_overall]
    out["_oracle_method_branch"] = branches[best_overall]
    return out


# --------------------------------------------------------------------------------------
# the realizable gate
# --------------------------------------------------------------------------------------

def realizable_q(df: pd.DataFrame, specialists: list[str], protocol: str,
                 protocol_df: pd.DataFrame | None, operator: str,
                 group_key: str) -> dict[str, np.ndarray]:
    """Out-of-fold (or frozen-protocol) applicability per specialist.

    The target is Stage 5's counterfactual marginal utility, computed under `operator` — the same
    fusion the deployed system will use — so this number predicts what Stage 5 will learn rather
    than approximating it.
    """
    if protocol == "insample":
        ops = lib.opinions_of(df, [lib.ANCHOR, *specialists])
        y = torch.tensor(df["label"].to_numpy(), dtype=torch.float32)
        phi = lib.marginal_utility(ops, y, specialists, operator=operator)
        groups = df[group_key].to_numpy()
        return {b: lib.cross_fit(lib.gate_features(df, b).to_numpy(),
                                 lib.applicability_target(phi[b]), groups)
                for b in specialists}

    if protocol_df is None or protocol_df.empty:
        raise SystemExit(
            f"the `protocol` arm needs {PROTOCOL_SOURCE} rows to fit on, and none were found in "
            f"the exports given. Pass a parquet containing them, or run only --protocol insample "
            f"and label the result a ceiling.")
    ops = lib.opinions_of(protocol_df, [lib.ANCHOR, *specialists])
    y = torch.tensor(protocol_df["label"].to_numpy(), dtype=torch.float32)
    phi = lib.marginal_utility(ops, y, specialists, operator=operator)
    return {b: lib.fit_logistic(lib.gate_features(protocol_df, b).to_numpy(),
                                lib.applicability_target(phi[b]),
                                lib.gate_features(df, b).to_numpy())
            for b in specialists}


# --------------------------------------------------------------------------------------
# per-method analysis
# --------------------------------------------------------------------------------------

def analyse_method(df: pd.DataFrame, specialists: list[str], q: dict[str, np.ndarray],
                   operator: str) -> dict:
    labels = df["label"].to_numpy()
    videos = df["video_id"].to_numpy()
    branches = [lib.ANCHOR, *specialists]
    ops = lib.opinions_of(df, branches)

    scores = {b: lib.video_auroc(df[f"p_{b}"].to_numpy(), labels, videos) for b in branches}
    scores["anchor"] = scores[lib.ANCHOR]
    scores["plain_fusion"] = lib.video_auroc(
        lib.fuse(ops, branches, operator=operator).fake_prob().numpy(), labels, videos)

    q_t = {b: torch.tensor(v, dtype=torch.float32) for b, v in q.items()}
    scores["gated_fusion"] = lib.video_auroc(
        lib.fuse(ops, branches, q_t, operator=operator).fake_prob().numpy(), labels, videos)

    oracles = oracle_probabilities(df, branches)
    best_branch = oracles.pop("_oracle_method_branch")
    for name, prob in oracles.items():
        scores[name] = lib.video_auroc(prob, labels, videos)

    anchor = scores["anchor"]
    recovery = {}
    for name in ("oracle_sample", "oracle_video", "oracle_method"):
        headroom = scores[name] - anchor
        gain = scores["gated_fusion"] - anchor
        # Zero headroom is a RESULT, not a missing value: it says the anchor is already the best
        # available expert on this method, so there is nothing for a router to recover and the
        # ratio is undefined rather than small. Reporting it as nan would let a reader mistake
        # "no headroom exists" for "the number could not be computed".
        no_headroom = abs(headroom) <= 1e-9
        recovery[name] = {
            "headroom": headroom,
            "gate_gain": gain,
            "no_headroom": bool(no_headroom),
            "recovered_fraction": None if no_headroom else gain / headroom,
        }

    # Stage 0.4 — rescue and harm, per specialist, relative to the anchor on THIS method
    rescue_harm = {}
    for b in specialists:
        delta = scores[b] - anchor
        rescue_harm[b] = {
            "specialist_auroc": scores[b],
            "delta_vs_anchor": delta,
            # "rescue" only means something where the anchor is failing; "harm" only where it is
            # working. Labelling a delta on a mid-range anchor as either would be a category error.
            "regime": ("anchor inverted (< 0.5)" if anchor < 0.5 else
                       "anchor strong (>= 0.8)" if anchor >= 0.8 else "anchor middling"),
            "rescue": delta if anchor < 0.5 else None,
            "harm": min(delta, 0.0) if anchor >= 0.8 else None,
            "mean_q": float(np.mean(q[b])) if b in q else None,
            "verdict": lib.verdict(delta, b, "anchor"),
        }

    # The delta that actually matters operationally. A specialist scoring below the anchor
    # standalone is not the same as the SYSTEM being worse for its presence: the gate exists
    # precisely to admit a weak specialist only where it helps. Both are reported, because the
    # per-specialist column above answers "is this branch good?" and this one answers "is the
    # deployed system better or worse than the anchor alone?" — and Stage 3 needs the second.
    system = {
        "gated_minus_anchor": scores["gated_fusion"] - anchor,
        "plain_minus_anchor": scores["plain_fusion"] - anchor,
        "gated_minus_plain": scores["gated_fusion"] - scores["plain_fusion"],
        "verdict": lib.verdict(scores["gated_fusion"] - anchor, "gated fusion", "anchor alone"),
    }
    return {"n_frames": int(len(df)), "n_videos": int(pd.unique(videos).size),
            "scores": scores, "recovery": recovery, "rescue_harm": rescue_harm,
            "system": system, "oracle_method_branch": best_branch}


# --------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------

def render(results: dict, meta: dict) -> str:
    specialists = meta["specialists"]
    lines = [
        "# Stage 0 — current V1 on DF40-Dev: rescue, harm, complementarity",
        "",
        f"Fusion operator `{meta['operator']}` · gate protocol `{meta['protocol']}` · "
        f"{meta['n_frames']} frames over {len(results)} methods.",
        f"Exports: " + ", ".join(f"`{p}`" for p in meta["parquets"]),
        "",
        "> **Contribution diagnostics on a frozen model, not ablations.** Every configuration "
        "below reuses heads trained with all branches present, so a specialist's contribution "
        "here means *what it adds on top of heads trained alongside it*, not what a system built "
        "without it would score.",
        "",
    ]
    if meta["protocol"] == "insample":
        lines += [
            "> ⚠️ **`insample` is a ceiling, not a deployable result.** The gate was cross-fit on "
            "the evaluation pool itself (leave-one-method-out), so it has seen sibling "
            "generators. The `protocol` arm — fit on FF++ only and frozen — is the number a "
            "deployed system would have.",
            "",
        ]

    lines += ["## Branch AUROC (video level)", "",
              "| method | anchor `sem` | " + " | ".join(f"`{s}`" for s in specialists) +
              " | plain fusion | gated fusion |", "|---" * (4 + len(specialists)) + "|"]
    for m, r in sorted(results.items()):
        s = r["scores"]
        cells = [f"{s.get(b, float('nan')):.4f}" for b in specialists]
        lines.append(f"| {m} | {s['anchor']:.4f} | " + " | ".join(cells) +
                     f" | {s['plain_fusion']:.4f} | {s['gated_fusion']:.4f} |")

    lines += ["", "## Oracle headroom and realizable recovery", "",
              "The recovered fraction is the decision signal. `oracle_method` is the honest "
              "denominator for a deployment claim — a gate can plausibly learn to prefer one "
              "branch on a family; it cannot learn to consult the label per frame.", "",
              "| method | anchor | gated | headroom (sample / video / method) | "
              "recovered fraction (sample / video / method) |",
              "|---|---:|---:|---|---|"]
    for m, r in sorted(results.items()):
        rec = r["recovery"]
        head = " / ".join(f"{rec[k]['headroom']:+.3f}"
                          for k in ("oracle_sample", "oracle_video", "oracle_method"))
        frac = " / ".join(
            ("none to recover" if rec[k]["no_headroom"]
             else f"{rec[k]['recovered_fraction']:+.2f}")
            for k in ("oracle_sample", "oracle_video", "oracle_method"))
        lines.append(f"| {m} | {r['scores']['anchor']:.4f} | {r['scores']['gated_fusion']:.4f} "
                     f"| {head} | {frac} |")

    lines += ["", "## Rescue and harm (Stage 0.4)", "",
              "Rescue is only defined where the anchor is failing and harm only where it is "
              "working; a delta measured on a middling anchor is neither, and is left blank "
              "rather than counted as evidence.", "",
              "| method | anchor regime | specialist | AUROC | delta vs anchor | rescue | harm | "
              "mean q | verdict |", "|---|---|---|---:|---:|---:|---:|---:|---|"]
    system_rows = []
    for m, r in sorted(results.items()):
        for b, rh in r["rescue_harm"].items():
            def fmt(v):
                return "—" if v is None else f"{v:+.4f}"
            mq = "—" if rh["mean_q"] is None else f"{rh['mean_q']:.3f}"
            lines.append(f"| {m} | {rh['regime']} | `{b}` | {rh['specialist_auroc']:.4f} | "
                         f"{rh['delta_vs_anchor']:+.4f} | {fmt(rh['rescue'])} | "
                         f"{fmt(rh['harm'])} | {mq} | {rh['verdict']} |")

    lines += ["", "### System-level effect (the operational question)", "",
              "| method | anchor | plain fusion − anchor | gated fusion − anchor | "
              "gated − plain | verdict |", "|---|---:|---:|---:|---:|---|"]
    for m, r in sorted(results.items()):
        sy = r["system"]
        lines.append(f"| {m} | {r['scores']['anchor']:.4f} | {sy['plain_minus_anchor']:+.4f} | "
                     f"{sy['gated_minus_anchor']:+.4f} | {sy['gated_minus_plain']:+.4f} | "
                     f"{sy['verdict']} |")

    # pooled read-off, computed rather than asserted
    inverted = {m: r for m, r in results.items() if r["scores"]["anchor"] < 0.5}
    strong = {m: r for m, r in results.items() if r["scores"]["anchor"] >= 0.8}
    lines += ["", "## Pooled read-off", ""]
    lines.append(f"* {len(inverted)} of {len(results)} methods have an **inverted anchor** "
                 f"(AUROC < 0.5): {', '.join(sorted(inverted)) or 'none'}.")
    for b in specialists:
        n_rescue = sum(1 for r in inverted.values()
                       if (r["rescue_harm"][b]["rescue"] or 0) > lib.NOISE_FLOOR)
        n_below = sum(1 for r in strong.values()
                      if (r["rescue_harm"][b]["harm"] or 0) < -lib.NOISE_FLOOR)
        lines.append(f"* `{b}` standalone: rescues {n_rescue}/{len(inverted)} inverted-anchor "
                     f"methods; scores below the anchor on {n_below}/{len(strong)} "
                     f"strong-anchor methods (beyond the {lib.NOISE_FLOOR} noise band).")
    sys_help = sum(1 for r in results.values()
                   if r["system"]["gated_minus_anchor"] > lib.NOISE_FLOOR)
    sys_hurt = sum(1 for r in results.values()
                   if r["system"]["gated_minus_anchor"] < -lib.NOISE_FLOOR)
    lines.append(
        f"* **Deployed system vs anchor alone**: better on {sys_help}, worse on {sys_hurt}, "
        f"indistinguishable on {len(results) - sys_help - sys_hurt} of {len(results)} methods. "
        f"This, not the standalone column above, is what the gate is answerable for — a "
        f"specialist may score badly alone and still be harmless once the gate can shut it off.")
    hurt_strong = [m for m, r in strong.items()
                   if r["system"]["gated_minus_anchor"] < -lib.NOISE_FLOOR]
    if hurt_strong:
        lines.append(f"* Methods where the anchor was already strong and the system made it "
                     f"**worse**: {', '.join(sorted(hurt_strong))}. These are the harm cases the "
                     f"brief's design goal forbids.")
    no_head = [m for m, r in results.items() if r["recovery"]["oracle_method"]["no_headroom"]]
    fractions = [r["recovery"]["oracle_method"]["recovered_fraction"] for r in results.values()
                 if r["recovery"]["oracle_method"]["recovered_fraction"] is not None]
    lines.append(
        f"* **{len(no_head)} of {len(results)} methods have no method-level headroom at all** — "
        f"the anchor is already the best single branch there, so no router of any quality could "
        f"improve on it by choosing differently"
        + (f": {', '.join(sorted(no_head))}." if no_head else "."))
    if fractions:
        lines.append(f"* Where headroom does exist ({len(fractions)} of {len(results)} methods), the median "
                     f"recovered fraction is **{float(np.median(fractions)):+.2f}**.")
    else:
        lines.append("* No method has method-level headroom, so the recovered fraction is "
                     "undefined everywhere. On this evidence there is nothing for an "
                     "applicability gate to exploit at the method level.")
    lines += ["", "Stage 3 consumes these numbers. Nothing is decided here.", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parquet", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--specialists", nargs="+", default=None,
                    help="default: every specialist present in the export")
    ap.add_argument("--operator", choices=("ds", "ccf"), default="ds",
                    help="the fusion the applicability target is computed under (Stage 5)")
    ap.add_argument("--protocol", choices=("insample", "protocol", "both"), default="both")
    ap.add_argument("--group-key", default="dataset",
                    help="cross-fitting group; `dataset` = leave-one-method-out for DF40 exports")
    ap.add_argument("--allow-holdout", action="store_true",
                    help="override the DF40 seal — only ever legitimate at Stage 8")
    args = ap.parse_args()

    df = lib.load_samples(args.parquet)
    protocol_df = df[df["dataset"] == PROTOCOL_SOURCE].copy()
    pool = df[df["dataset"] != PROTOCOL_SOURCE].copy()
    if pool.empty:
        raise SystemExit("the exports contain only the protocol source; nothing to evaluate on")
    if not args.allow_holdout:
        lib.assert_no_holdout(pool["dataset"].unique())

    present = lib.branches_present(pool)
    specialists = args.specialists or [b for b in present if b != lib.ANCHOR]
    missing = [b for b in specialists if b not in present]
    if missing:
        raise SystemExit(f"specialists {missing} are not in the export (present: {present})")
    print(f"anchor `{lib.ANCHOR}` + specialists {specialists}; "
          f"{len(pool)} evaluation frames, {len(protocol_df)} {PROTOCOL_SOURCE} frames")

    args.out.mkdir(parents=True, exist_ok=True)
    protocols = ["insample", "protocol"] if args.protocol == "both" else [args.protocol]
    written = []
    for protocol in protocols:
        try:
            q = realizable_q(pool, specialists, protocol, protocol_df, args.operator,
                             args.group_key)
        except SystemExit as exc:
            print(f"\n[{protocol}] skipped: {exc}")
            continue
        results = {}
        for method, group in pool.groupby("dataset"):
            idx = pool.index.get_indexer(group.index)
            q_m = {b: v[idx] for b, v in q.items()}
            results[str(method)] = analyse_method(group.reset_index(drop=True), specialists,
                                                  {b: v for b, v in q_m.items()}, args.operator)
            r = results[str(method)]
            rec = r["recovery"]["oracle_method"]
            frac = ("no headroom" if rec["no_headroom"]
                    else f"{rec['recovered_fraction']:+.2f}")
            print(f"  {method:22s} anchor {r['scores']['anchor']:.4f}  "
                  f"gated {r['scores']['gated_fusion']:.4f}  recovered(method) {frac}")
        meta = {"parquets": [str(p) for p in args.parquet], "operator": args.operator,
                "protocol": protocol, "specialists": specialists, "n_frames": int(len(pool)),
                "protocol_source": PROTOCOL_SOURCE, "group_key": args.group_key,
                "protocol_rows": int(len(protocol_df))}
        (args.out / f"stage0_{protocol}.json").write_text(
            json.dumps({"meta": meta, "results": results}, indent=2, default=str))
        doc = args.out / f"STAGE0_DF40DEV_{protocol}.md"
        doc.write_text(render(results, meta))
        written.append(doc)
        print(f"[{protocol}] wrote {doc}")

    if not written:
        raise SystemExit("no protocol arm produced results")
    print("\nStage 0 always completes and is never itself a stop — these numbers feed Stage 3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

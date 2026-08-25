#!/usr/bin/env python3
"""Stage 8 — the final untouched evaluation table.

    🔴 UMAR-RUNS (CPU, minutes):

    python analysis/discern_v2/phase2/stage8_final.py \
        --parquet logs/phase2/eval/final/per_sample_epoch_NNN.parquet \
                  logs/phase2/eval/final_df40holdout/per_sample_epoch_NNN.parquet \
        --stage567 logs/phase2/stage567/epoch_NNN/stage567.pt \
        --baseline docs/DiCoME_eval/reproduced_baseline.json \
        --out phase2

Everything is frozen by the time this runs. The script computes the four comparison columns from
the per-sample export and writes `phase2/STAGE8_FINAL.md`.

The two bookkeeping rules, enforced rather than remembered
----------------------------------------------------------
The brief says both must survive into the paper, so neither is left to a reader's memory:

1. **A source whose validation split calibrated the gates or the defer policy is `calibrated`, not
   `zero-shot`.** Read from the `stage567.pt` artifact's own `sources_no_longer_zero_shot` field,
   which `stage567.py` computed by intersecting each calibration split with that source's test
   split. V1's Celeb-DF-v2 gated rows were calibrated for exactly this reason.
2. **DF40-Dev methods can never be presented as zero-shot evidence.** They drove the Stage 3
   architecture decision. Read from `configs/discern_v2/df40_split.json`; every Dev row is
   labelled `dev (drove decisions)` and only Holdout rows carry the zero-shot claim.

Four columns, kept distinct
---------------------------
| column | what it is |
|---|---|
| `this_system` | the gated, applicability-discounted fusion under the primary arm |
| `plain_ungated` | the same experts fused with q = 1 — no applicability layer |
| `anchor_only` | Branch A alone; the number the specialists must beat to exist |
| `baseline` | the reproduced external baseline, from its own file |

They are never averaged and never merged. A cell with no run behind it is written `TODO(run)`,
never filled by inference from a neighbouring column.
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

TODO = "TODO(run)"
CONVENTIONAL = ("FaceForensics++", "Celeb-DF-v1", "Celeb-DF-v2", "Celeb-DF-v3",
                "DeepFakeDetection", "DFDCP", "DFDC", "UADFV")
WILD = ("Deepfake-Eval-2024",)


def provenance_of(source: str, not_zero_shot: list[str], split: dict) -> str:
    """How this row may be described. One place, so no table can disagree with another."""
    if not_zero_shot == ["__UNKNOWN__"]:
        if source in split.get("dev", {}):
            return "DF40-Dev (drove decisions — never zero-shot)"
        if source in split.get("holdout", {}):
            return "DF40-Holdout (sealed; see DF40_SPLIT.md for the exact claim)"
        return "UNVERIFIED (no frozen artifact supplied)"
    if any(source in entry for entry in not_zero_shot):
        return "calibrated (NOT zero-shot)"
    if source in split.get("dev", {}):
        return "DF40-Dev (drove decisions — never zero-shot)"
    if source in split.get("holdout", {}):
        return "DF40-Holdout (sealed; see DF40_SPLIT.md for the exact claim)"
    if source in CONVENTIONAL or source in WILD:
        return "zero-shot"
    return "unclassified — classify before publishing"


def columns_for(group: pd.DataFrame, arm: str) -> dict:
    """The four provenance-distinct columns, each computed or left as TODO(run)."""
    labels = group["label"].to_numpy()
    videos = group["video_id"].to_numpy()

    def auroc(col: str) -> float | str:
        return lib.video_auroc(group[col].to_numpy(), labels, videos) if col in group else TODO

    # `this_system` prefers the arm-specific column written by a phase-2 eval, then the generic
    # gated column a V1-style eval writes. Never falls back to the ungated column: a table that
    # silently reported plain fusion as "this system" would misdescribe the contribution.
    system = TODO
    for candidate in (f"prob_{arm}", "prob_gated"):
        if candidate in group:
            system = lib.video_auroc(group[candidate].to_numpy(), labels, videos)
            break
    return {
        "this_system": system,
        "plain_ungated": auroc("prob_fused"),
        "anchor_only": auroc("p_sem"),
        "n_videos": int(pd.unique(videos).size),
        "n_frames": int(len(group)),
        "coverage": (float((group["decision"] != 2).mean()) if "decision" in group else TODO),
    }


def fmt(value) -> str:
    if isinstance(value, str):
        return value
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return TODO
    return f"{value:.4f}"


def render(rows: dict, meta: dict, baseline: dict) -> str:
    lines = [
        "# Stage 8 — final evaluation (everything frozen)",
        "",
        f"Arm `{meta['arm']}` · operator `{meta['operator']}` · checkpoint epoch "
        f"{meta['epoch']} · calibrated on {', '.join(meta['calibration_sources'] or ['?'])}.",
        "",
        "> **The four columns are provenance-distinct and are never averaged together.** "
        "`this_system` is the gated, applicability-discounted fusion; `plain_ungated` is the same "
        "experts at q = 1; `anchor_only` is Branch A alone; `baseline` is the reproduced external "
        "baseline read from its own file. An unrun cell is `TODO(run)` — never filled in from a "
        "neighbouring column.",
        "",
    ]
    if meta["not_zero_shot"] == ["__UNKNOWN__"]:
        lines += ["> ⚠️ **Provenance UNVERIFIED.** No frozen `stage567.pt` was supplied, so which "
                  "sources were used for calibration is unknown. The `zero-shot` labels below are "
                  "**assumptions, not records** — regenerate this table with `--stage567` before "
                  "any of them is repeated in writing.", ""]
    elif meta["not_zero_shot"]:
        lines += [f"> ⚠️ **Not zero-shot:** {', '.join(meta['not_zero_shot'])}. These sources' "
                  f"test videos calibrated the gates and the defer policy. Every row below is "
                  f"labelled, and these must be reported as calibrated.", ""]

    for title, keys in (("Conventional benchmarks", CONVENTIONAL),
                        ("In the wild", WILD)):
        present = [k for k in keys if k in rows]
        if not present:
            continue
        lines += [f"## {title}", "",
                  "| source | provenance | this system | plain ungated | anchor only | "
                  "baseline | coverage | videos |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for key in present:
            r = rows[key]
            lines.append(
                f"| {key} | {r['provenance']} | {fmt(r['this_system'])} | "
                f"{fmt(r['plain_ungated'])} | {fmt(r['anchor_only'])} | "
                f"{fmt(baseline.get(key, TODO))} | {fmt(r['coverage'])} | {r['n_videos']} |")
        lines.append("")

    df40 = {k: v for k, v in rows.items() if "DF40" in v["provenance"]}
    if df40:
        holdout = {k: v for k, v in df40.items() if "Holdout" in v["provenance"]}
        dev = {k: v for k, v in df40.items() if "Dev" in v["provenance"]}
        for title, part, note in (
            ("DF40-Holdout — the only DF40 rows that carry a zero-shot claim", holdout,
             "Sealed before Phase 2 and never read by any configuration, gate, threshold or "
             "promotion decision. Not \"never seen\": a Phase-1 per-generator table computed with "
             "a different operator family covered most of the corpus. See `phase2/DF40_SPLIT.md` "
             "for the exact wording that survives scrutiny."),
            ("DF40-Dev — reported for completeness, NOT as zero-shot evidence", dev,
             "These methods drove the Stage 3 architecture decision. They may be shown, but never "
             "as evidence of generalisation for this model."),
        ):
            if not part:
                continue
            lines += [f"## {title}", "", note, "",
                      "| method | family | this system | plain ungated | anchor only | videos |",
                      "|---|---|---:|---:|---:|---:|"]
            for key in sorted(part):
                r = part[key]
                lines.append(f"| {key} | {r.get('family', '—')} | {fmt(r['this_system'])} | "
                             f"{fmt(r['plain_ungated'])} | {fmt(r['anchor_only'])} | "
                             f"{r['n_videos']} |")
            lines.append("")
            fams: dict[str, list[float]] = {}
            for r in part.values():
                if isinstance(r["this_system"], float) and np.isfinite(r["this_system"]):
                    fams.setdefault(r.get("family", "—"), []).append(r["this_system"])
            if fams:
                lines += ["| family | methods | median this-system AUROC |", "|---|---:|---:|"]
                for family, values in sorted(fams.items()):
                    lines.append(f"| {family} | {len(values)} | {np.median(values):.4f} |")
                lines.append("")

    scored = [r for r in rows.values()
              if isinstance(r["this_system"], float) and np.isfinite(r["this_system"])
              and isinstance(r["anchor_only"], float) and np.isfinite(r["anchor_only"])]
    lines += ["## Read-off", ""]
    if scored:
        better = sum(1 for r in scored if r["this_system"] - r["anchor_only"] > lib.NOISE_FLOOR)
        worse = sum(1 for r in scored if r["this_system"] - r["anchor_only"] < -lib.NOISE_FLOOR)
        lines += [
            f"* System vs anchor alone, over {len(scored)} scored sources: better on "
            f"**{better}**, worse on **{worse}**, indistinguishable on "
            f"**{len(scored) - better - worse}** (±{lib.NOISE_FLOOR}).",
            f"* Median delta: **{np.median([r['this_system'] - r['anchor_only'] for r in scored]):+.4f}**.",
        ]
        zero_shot = [r for r in scored if r["provenance"] == "zero-shot"]
        if zero_shot:
            lines.append(
                f"* Restricted to the **{len(zero_shot)} genuinely zero-shot** sources, median "
                f"delta is "
                f"**{np.median([r['this_system'] - r['anchor_only'] for r in zero_shot]):+.4f}**. "
                f"This is the number a generalisation claim rests on.")
    else:
        lines.append(f"* No source has both a system and an anchor score. Everything is {TODO}.")
    lines += ["",
              f"* Anything inside ±{lib.NOISE_FLOOR} requires a confirming seed before it "
              f"supports a promotion, removal or architecture change (brief ground rule 3).", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parquet", type=Path, nargs="+", required=True)
    ap.add_argument("--stage567", type=Path, default=None,
                    help="the frozen artifact; its own record of which sources stopped being "
                         "zero-shot is what labels the rows")
    ap.add_argument("--baseline", type=Path, default=None,
                    help="JSON of {source: video_auroc} for the reproduced external baseline; "
                         "sources absent from it stay TODO(run)")
    ap.add_argument("--arm", default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    df = lib.load_samples(args.parquet)
    split = lib.load_df40_split()

    arm, operator, epoch, not_zero_shot, calibration = args.arm, "?", "?", [], None
    if args.stage567 and args.stage567.is_file():
        blob = torch.load(str(args.stage567), map_location="cpu", weights_only=False)
        arm = arm or blob.get("primary_arm")
        operator = "ccf" if str(arm).endswith("ccf") else "ds"
        epoch = blob.get("epoch", "?")
        not_zero_shot = list(blob.get("sources_no_longer_zero_shot") or [])
        calibration = blob.get("calibration_sources")
    else:
        print("  !! no --stage567 given, so NOTHING is known about which sources were "
              "calibrated. Every conventional source will be labelled `zero-shot`, which is the "
              "exact mislabelling rule 1 exists to prevent. Pass the frozen artifact.")
        not_zero_shot = ["__UNKNOWN__"]
    if arm is None:
        raise SystemExit(
            "no arm given and no --stage567 to read one from. The arm decides which frozen policy "
            "produced the `this_system` column, so it cannot be inferred.")

    baseline = json.loads(args.baseline.read_text()) if args.baseline and \
        args.baseline.is_file() else {}
    if not baseline:
        print(f"  no baseline file: every baseline cell will read {TODO}")

    families = {**{m: v.get("family") for m, v in split.get("dev", {}).items()},
                **{m: v.get("family") for m, v in split.get("holdout", {}).items()}}

    rows = {}
    for source, group in df.groupby("dataset"):
        source = str(source)
        rows[source] = {**columns_for(group, arm),
                        "provenance": provenance_of(source, not_zero_shot, split),
                        "family": families.get(source)}
        r = rows[source]
        print(f"  {source:24s} {r['provenance']:42s} system {fmt(r['this_system']):>9s}  "
              f"ungated {fmt(r['plain_ungated']):>9s}  anchor {fmt(r['anchor_only']):>9s}")

    unclassified = [s for s, r in rows.items() if "unclassified" in r["provenance"]]
    if unclassified:
        print(f"\n  !! unclassified sources {unclassified} — classify them before publishing; a "
              f"row with no provenance label will be read as zero-shot by default")

    meta = {"arm": arm, "operator": operator, "epoch": epoch,
            "not_zero_shot": not_zero_shot, "calibration_sources": calibration,
            "parquets": [str(p) for p in args.parquet], "n_frames": int(len(df))}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "stage8_final.json").write_text(
        json.dumps({"meta": meta, "rows": rows, "baseline": baseline}, indent=2, default=str))
    (args.out / "STAGE8_FINAL.md").write_text(render(rows, meta, baseline))
    print(f"\nwrote {args.out}/STAGE8_FINAL.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

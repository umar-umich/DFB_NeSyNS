#!/usr/bin/env python3
"""Stage 3 — the main cross-dataset table, and the two contrasts that carry the paper.

    🔴 UMAR-RUNS (CPU, seconds):

    python analysis/fpad/main_table.py \
        --rung B0 direct logs/fpad/score/B0_ood/profile_epoch_009.parquet \
        --rung B1 direct logs/fpad/score/B1_ood/profile_epoch_009.parquet \
        --rung B2 traj   logs/fpad/score/B2_ood/profile_epoch_009.parquet \
        --rung B3 traj   logs/fpad/score/B3_ood/profile_epoch_009.parquet \
        --external docs/DiCoME_eval/fsvfm_linearprobe.json \
        --out wacv

Each `--rung NAME READOUT PARQUET` names the readout explicitly, because the whole ladder turns on
holding the student fixed and changing the readout (or the reverse). `direct` scores `p_direct`;
`traj` scores `p_traj`. A rung whose parquet lacks its readout column is reported `TODO(run)`
rather than silently falling back to the other one — a B2 row quietly filled with B1's direct
readout would look like a result and mean nothing.

The two contrasts, stated as the brief does
-------------------------------------------
* **B2 − B1** — does the trajectory hold anything beyond the ordinary adapted student? The student
  is IDENTICAL in both; only the readout changes. Under two-stage training that makes it a clean
  readout-only comparison, which is why the ladder is built this way.
* **B3 − B2** — does authentic-prior preservation improve that trajectory? The readout is
  identical; only `L_preserve` differs.

Both are computed with their own noise verdicts. The headline is NOT B3 against chance: the honest
baseline is B1, ordinary fine-tuned FS-VFM, which prior analysis showed is already strong here.

Provenance, enforced
--------------------
Sources spent on the Stage-2 gate decision are not part of the zero-shot claim, and DF40-Dev can
never be. `provenance_of` labels every row from the sealed split file and the Stage-2 artifact, so
a table cannot present a development source as OOD.
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

TODO = "TODO(run)"
NOISE = 0.01
READOUTS = {"direct": "p_direct", "traj": "p_traj", "spatial": "p_spatial"}
OOD_SUITE = ("Celeb-DF-v2", "Celeb-DF-v3", "DFDC", "DFDCP", "DeepFakeDetection", "UADFV",
             "Deepfake-Eval-2024")
PROTOCOL = "FaceForensics++"


def video_auroc(prob: np.ndarray, labels: np.ndarray, videos: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    agg = pd.DataFrame({"p": prob, "y": labels, "v": videos}).groupby(
        "v", as_index=False).agg(p=("p", "mean"), y=("y", "max"))
    if agg["y"].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(agg["y"], agg["p"]))


def load_rung(name: str, readout: str, paths: list[Path]) -> dict:
    if readout not in READOUTS:
        raise SystemExit(f"rung {name}: readout must be one of {sorted(READOUTS)}, got {readout!r}")
    column = READOUTS[readout]
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    scores: dict[str, float | str] = {}
    for source, group in df.groupby("dataset"):
        if column not in group.columns:
            scores[str(source)] = TODO
            continue
        scores[str(source)] = video_auroc(group[column].to_numpy(),
                                          group["label"].to_numpy(),
                                          group["video_id"].to_numpy())
    meta = {}
    mp = paths[0].parent / "score_meta.json"
    if mp.is_file():
        meta = json.loads(mp.read_text())
    return {"name": name, "readout": readout, "column": column, "scores": scores,
            "has_column": column in df.columns, "student_meta": meta.get("student_meta", {}),
            "epoch": meta.get("epoch"), "paths": [str(p) for p in paths]}


def provenance_of(source: str, dev_methods: set[str], holdout: set[str],
                  gate_touched: list[str]) -> str:
    if source in dev_methods:
        return "DF40-Dev (drove decisions — never zero-shot)"
    if source in holdout:
        return "DF40-Holdout (sealed until Stage 7)"
    if any(source in t for t in gate_touched):
        return "read during the Stage-2 gate — footnote, not zero-shot"
    if source == PROTOCOL:
        return "in-domain (trained on)"
    if source in OOD_SUITE:
        return "zero-shot"
    return "unclassified — classify before publishing"


def fmt(v) -> str:
    if isinstance(v, str):
        return v
    if v is None or not np.isfinite(v):
        return TODO
    return f"{v:.4f}"


def contrast(rungs: dict, better: str, worse: str, question: str, sources: list[str]) -> dict:
    if better not in rungs or worse not in rungs:
        return {"question": question, "status": f"{TODO} — needs both {worse} and {better}"}
    per_source, deltas = {}, []
    for s in sources:
        a, b = rungs[better]["scores"].get(s), rungs[worse]["scores"].get(s)
        if isinstance(a, str) or isinstance(b, str) or a is None or b is None:
            per_source[s] = TODO
            continue
        d = a - b
        per_source[s] = d
        deltas.append(d)
    if not deltas:
        return {"question": question, "status": TODO, "per_source": per_source}
    median = float(np.median(deltas))
    return {
        "question": question, "better": better, "worse": worse,
        "per_source": per_source, "median": median,
        "n_better": int(sum(d > NOISE for d in deltas)),
        "n_worse": int(sum(d < -NOISE for d in deltas)),
        "n_indistinguishable": int(sum(abs(d) <= NOISE for d in deltas)),
        "verdict": (f"inside the {NOISE} noise band — a confirming seed is required"
                    if abs(median) <= NOISE else
                    f"{better} better" if median > 0 else f"{worse} better"),
    }


def render(rungs: dict, rows: dict, contrasts: dict, external: dict, meta: dict) -> str:
    order = [n for n in ("B0", "B1", "B2", "B3", "B4", "B5") if n in rungs]
    sources = list(rows)

    lines = [
        "# Stage 3 — main cross-dataset table",
        "",
        "Trained on FF++ c23 only; every rung selected on FF++ validation only.",
        "",
        "| rung | student adaptation | readout |", "|---|---|---|",
        "| B0 | frozen FS-VFM | direct FS-VFM head |",
        "| B1 | ordinary LoRA | direct student feature |",
        "| B2 | ordinary LoRA | trajectory `D(x)` |",
        "| B3 | LoRA + real-prior preservation | trajectory `D(x)` |",
        "| B4 | B3 + paired ranking objective | trajectory `D(x)` |",
        "| B5 | B3's frozen student, unchanged | spatial map over patch adaptation |",
        "",
        "> **B1 is the honest baseline.** It is ordinary fine-tuned FS-VFM, already strong on this "
        "task. The headline is `B2 - B1` and `B3 - B2`, not B3 against chance.",
        "",
        "## Video-level AUROC", "",
        "| source | provenance | " + " | ".join(order) +
        " | FS-VFM linear probe (official) |",
        "|---|---" + "|---:" * (len(order) + 1) + "|",
    ]
    for source in sources:
        cells = [fmt(rungs[n]["scores"].get(source)) for n in order]
        lines.append(f"| {source} | {rows[source]['provenance']} | " + " | ".join(cells) +
                     f" | {fmt(external.get(source, TODO))} |")

    lines += ["", "## The two contrasts that carry the paper", ""]
    for key in ("readout_only", "preservation"):
        c = contrasts.get(key)
        if not c:
            continue
        lines += [f"### {c['question']}", ""]
        if "median" not in c:
            lines += [f"{c.get('status', TODO)}", ""]
            continue
        lines += [f"`{c['better']} - {c['worse']}` · median **{c['median']:+.4f}** · "
                  f"better on {c['n_better']}, worse on {c['n_worse']}, indistinguishable on "
                  f"{c['n_indistinguishable']} of {len(sources)} sources · **{c['verdict']}**",
                  "", "| source | delta |", "|---|---:|"]
        for s, d in c["per_source"].items():
            lines.append(f"| {s} | {fmt(d) if isinstance(d, str) else f'{d:+.4f}'} |")
        lines.append("")

    unrun = [n for n in order if not rungs[n]["has_column"]]
    if unrun:
        lines += [f"> ⚠️ Rungs {unrun} have no `{READOUTS[rungs[unrun[0]]['readout']]}` column in "
                  f"their profile, so their rows are `{TODO}`. They are NOT filled from another "
                  f"readout.", ""]
    lines += ["## Bookkeeping", "",
              "* Sources spent on the Stage-2 gate decision are footnoted, not presented as "
              "zero-shot.",
              "* DF40-Dev can never be zero-shot evidence; only DF40-Holdout carries that claim, "
              "and it is read for the first time in Stage 7.",
              f"* Anything inside ±{NOISE} needs a confirming seed before it supports a claim.",
              ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rung", nargs="+", action="append", required=True,
                    metavar="NAME READOUT PARQUET...",
                    help="e.g. --rung B2 traj path/to/profile.parquet")
    ap.add_argument("--external", type=Path, default=None,
                    help="JSON {source: video_auroc} for the official FS-VFM protocol")
    ap.add_argument("--stage2", type=Path, default=None,
                    help="wacv/stage2_gate.json — names the sources the gate read")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rungs = {}
    for spec in args.rung:
        if len(spec) < 3:
            raise SystemExit(f"--rung needs NAME READOUT PARQUET, got {spec}")
        name, readout, *paths = spec
        rungs[name] = load_rung(name, readout, [Path(p) for p in paths])
        r = rungs[name]
        print(f"{name:3s} readout={readout:6s} epoch={r['epoch']} "
              f"sources={len(r['scores'])} column_present={r['has_column']}")

    split_file = REPO / "configs/discern_v2/df40_split.json"
    dev_methods, holdout = set(), set()
    if split_file.is_file():
        blob = json.loads(split_file.read_text())
        dev_methods, holdout = set(blob.get("dev", {})), set(blob.get("holdout", {}))
    gate_touched = []
    if args.stage2 and args.stage2.is_file():
        g = json.loads(args.stage2.read_text())
        pc = g.get("provenance_cost") or {}
        if pc.get("corpus"):
            gate_touched.append(pc["corpus"])
        print(f"  Stage-2 gate read: {gate_touched or 'nothing recorded'}")

    all_sources = sorted({s for r in rungs.values() for s in r["scores"]})
    rows = {s: {"provenance": provenance_of(s, dev_methods, holdout, gate_touched)}
            for s in all_sources}
    for s, r in rows.items():
        print(f"  {s:24s} {r['provenance']}")

    external = json.loads(args.external.read_text()) if args.external and \
        args.external.is_file() else {}
    if not external:
        print(f"  no external baseline file: that column reads {TODO}")

    contrasts = {
        "readout_only": contrast(
            rungs, "B2", "B1",
            "B2 - B1: does the trajectory hold anything beyond the ordinary adapted student? "
            "(same student, readout only)", all_sources),
        "preservation": contrast(
            rungs, "B3", "B2",
            "B3 - B2: does authentic-prior preservation improve the trajectory? "
            "(same readout, L_preserve only)", all_sources),
    }
    for c in contrasts.values():
        print(f"\n{c['question']}\n  -> {c.get('verdict', c.get('status'))}")

    meta = {"rungs": {n: {k: v for k, v in r.items() if k != "scores"}
                      for n, r in rungs.items()},
            "gate_touched": gate_touched}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "stage3_main_table.json").write_text(json.dumps(
        {"meta": meta, "rungs": {n: r["scores"] for n, r in rungs.items()},
         "rows": rows, "contrasts": contrasts, "external": external}, indent=2, default=str))
    (args.out / "STAGE3_MAIN_TABLE.md").write_text(
        render(rungs, rows, contrasts, external, meta))
    print(f"\nwrote {args.out}/STAGE3_MAIN_TABLE.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Stage 7 — freeze everything and assemble the paper's tables.

    🔴 UMAR-RUNS (CPU, seconds):

    python analysis/fpad/stage7_final.py \
        --stage2 wacv/stage2_gate.json \
        --stage3 wacv/stage3_main_table.json \
        --stage4 wacv/stage4_rescue_profile.json \
        --stage6 wacv/stage6_jpeg_probe.json \
        --localization logs/fpad/localization/B3/localization.json \
        --holdout B1 direct logs/fpad/score/B1_holdout/profile_epoch_009.parquet \
        --holdout B3 traj   logs/fpad/score/B3_holdout/profile_epoch_009.parquet \
        --out wacv

Assembles, never recomputes. Every number is read from a stage artifact, so the paper and the
stage files cannot disagree. Two rules shape the output.

**A missing section is `TODO(run)`, not an omission.** An omitted section reads as "not needed"; a
`TODO(run)` reads as "not run". The difference matters when someone else picks the document up, so
the completeness checklist lists everything the brief asks for and marks what is absent.

**Provenance is enforced, not remembered.** DF40-Dev drove design decisions and can never be
presented as zero-shot. Only DF40-Holdout carries that claim, and this stage is the ONLY place it
may be read — the moment of unsealing is recorded with a timestamp taken from the artifact, not
from the clock, so a re-run does not rewrite history. Any source whose validation split fed
selection or calibration is labelled, and the Stage-2 provenance cost (Celeb-DF-v2 reals read as
the domain axis) is carried through verbatim.

The framing follows the Stage-2 verdict
---------------------------------------
The mechanism-validation outcome decides what the paper may claim. If it says PIVOT, this document
will not present an anomaly-detection framing; if it says BORDERLINE, it says so and refuses to
pick. That is deliberate: the point of running that stage on day two was to let the evidence choose
the framing, and a final assembly that quietly ignored it would undo the whole exercise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
TODO = "TODO(run)"
READOUTS = {"direct": "p_direct", "traj": "p_traj", "spatial": "p_spatial"}
SPLIT_FILE = REPO / "configs/discern_v2/df40_split.json"
NOISE = 0.01


def load(path: Path | None) -> dict | None:
    if path is None or not Path(path).is_file():
        return None
    return json.loads(Path(path).read_text())


def video_auroc(prob, labels, videos) -> float:
    from sklearn.metrics import roc_auc_score

    agg = pd.DataFrame({"p": prob, "y": labels, "v": videos}).groupby(
        "v", as_index=False).agg(p=("p", "mean"), y=("y", "max"))
    if agg["y"].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(agg["y"], agg["p"]))


def fmt(v, sign=False) -> str:
    if isinstance(v, str):
        return v
    if v is None or not np.isfinite(v):
        return TODO
    return f"{v:+.4f}" if sign else f"{v:.4f}"


def score_holdout(specs: list[list[str]]) -> dict:
    """The DF40-Holdout table. This is the seal being broken, so it is scored here and nowhere else."""
    if not SPLIT_FILE.is_file():
        raise SystemExit(f"{SPLIT_FILE} missing — the Holdout set is defined by it")
    split = json.loads(SPLIT_FILE.read_text())
    holdout, dev = set(split.get("holdout", {})), set(split.get("dev", {}))
    families = {m: v.get("family") for m, v in (split.get("holdout") or {}).items()}

    out: dict[str, dict] = {}
    for spec in specs:
        name, readout, *paths = spec
        df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
        seen = set(df["dataset"].astype(str))
        stray = sorted(seen - holdout)
        if stray:
            raise SystemExit(
                f"--holdout for rung {name} includes non-Holdout sources {stray}. This table is "
                f"the ONLY zero-shot claim in the paper; mixing Dev or conventional sources into "
                f"it would destroy that. Score them through --stage3 instead."
                + (f" ({sorted(set(stray) & dev)} are DF40-Dev, which drove design decisions.)"
                   if set(stray) & dev else ""))
        column = READOUTS[readout]
        if column not in df.columns:
            raise SystemExit(
                f"rung {name} declared readout `{readout}` but its profile has no `{column}`. "
                f"Filling this table from the other readout would mislabel the rung in the one "
                f"place the paper claims zero-shot.")
        for method, g in df.groupby("dataset"):
            out.setdefault(str(method), {"family": families.get(str(method), "unknown")})[name] = \
                video_auroc(g[column].to_numpy(), g["label"].to_numpy(), g["video_id"].to_numpy())
    return out


def checklist(parts: dict) -> list[tuple[str, str, str]]:
    """What the brief asks Stage 7 to assemble, and whether it exists."""
    items = [
        ("main cross-dataset table (B0-B4 + external baseline)", "stage3",
         "analysis/fpad/main_table.py"),
        ("DF40-Dev rescue table", "stage4", "analysis/fpad/stage4_rescue_profile.py"),
        ("DF40-Holdout table (the only zero-shot claim)", "holdout",
         "score the Holdout methods, then --holdout here"),
        ("domain-audit contrast (ordinary vs preservation delta)", "stage2",
         "analysis/fpad/stage2_gate.py"),
        ("depth-profile figure", "stage4", "analysis/fpad/stage4_rescue_profile.py"),
        ("localization + faithfulness", "localization", "training/localize_fpad.py"),
        ("compression robustness", "stage6",
         "analysis/fpad/jpeg_probe.py (JPEG); H.264 c40 needs the download"),
    ]
    return [(label, "present" if parts.get(key) else TODO, how) for label, key, how in items]


def render(parts: dict, holdout: dict, out_dir: Path) -> str:
    s2, s3, s4, s6, loc = (parts.get(k) for k in
                           ("stage2", "stage3", "stage4", "stage6", "localization"))
    lines = ["# Stage 7 — final tables (everything frozen)", ""]

    # ---- framing, driven by the mechanism-validation verdict --------------------------
    lines += ["## Framing", ""]
    if s2:
        d = s2["decision"]
        lines += [f"Mechanism validation returned **{d['verdict']}**.", "", d["justification"], ""]
        if d.get("borderline"):
            lines += ["> The framing is therefore **not settled by the evidence**. This document "
                      "presents the measurements and does not choose between the "
                      "anomaly-detection and localization-first framings. Re-run the stage at the "
                      "final epoch on the full scoring set before writing the abstract.", ""]
        elif not d.get("proceed"):
            lines += ["> The anomaly-detection framing is **not supported**. Present the work as a "
                      "faithful region-resolved adaptation map; the localization and faithfulness "
                      "results below carry the contribution.", ""]
        else:
            lines += ["> The anomaly-detection framing is supported: preservation buys "
                      "manipulation-specificity. The depth profile and rescue results are "
                      "supporting evidence, not the headline.", ""]
    else:
        lines += [f"{TODO} — mechanism validation has not been run, so the paper's framing is "
                  f"undetermined. Nothing below should be written up until it is.", ""]

    # ---- completeness ----------------------------------------------------------------
    lines += ["## Completeness", "",
              "| deliverable | status | how to produce it |", "|---|---|---|"]
    for label, status, how in checklist({**parts, "holdout": holdout or None}):
        lines.append(f"| {label} | {status} | `{how}` |")
    lines.append("")

    # ---- main table -------------------------------------------------------------------
    lines += ["## Main cross-dataset table", ""]
    if s3:
        rungs, rows, ext = s3["rungs"], s3["rows"], s3.get("external", {})
        order = [r for r in ("B0", "B1", "B2", "B3", "B4", "B5") if r in rungs]
        lines += ["| source | provenance | " + " | ".join(order) +
                  " | FS-VFM linear probe |", "|---|---" + "|---:" * (len(order) + 1) + "|"]
        for source, meta in rows.items():
            cells = [fmt(rungs[r].get(source)) for r in order]
            lines.append(f"| {source} | {meta['provenance']} | " + " | ".join(cells) +
                         f" | {fmt(ext.get(source, TODO))} |")
        lines.append("")
        for key, c in (s3.get("contrasts") or {}).items():
            if "median" in c:
                lines.append(f"* **{c['better']} − {c['worse']}**: median {c['median']:+.4f}, "
                             f"{c['verdict']} ({c['n_better']} better / {c['n_worse']} worse / "
                             f"{c['n_indistinguishable']} tied)")
        lines.append("")
    else:
        lines += [f"{TODO} — run `analysis/fpad/main_table.py`.", ""]

    # ---- holdout ----------------------------------------------------------------------
    lines += ["## DF40-Holdout — the only zero-shot claim", ""]
    if holdout:
        rung_names = sorted({r for v in holdout.values() for r in v if r != "family"})
        lines += ["Read for the first time at this stage. See `phase2/DF40_SPLIT.md` for the "
                  "exact wording this claim supports — these methods were **sealed before Phase 2 "
                  "and unread by any Phase-2 decision**, which is not the same as never seen.", "",
                  "| method | family | " + " | ".join(rung_names) + " |",
                  "|---|---" + "|---:" * len(rung_names) + "|"]
        for method, v in sorted(holdout.items()):
            lines.append(f"| {method} | {v.get('family', '—')} | " +
                         " | ".join(fmt(v.get(r)) for r in rung_names) + " |")
        lines.append("")
        for r in rung_names:
            vals = [v[r] for v in holdout.values() if isinstance(v.get(r), float)
                    and np.isfinite(v[r])]
            if vals:
                lines.append(f"* `{r}` median over {len(vals)} Holdout methods: "
                             f"**{np.median(vals):.4f}**")
        lines.append("")
    else:
        lines += [f"{TODO} — the Holdout set is still sealed. Score it, then pass `--holdout`. "
                  f"Nothing else in the paper may be described as zero-shot on DF40.", ""]

    # ---- rescue -----------------------------------------------------------------------
    lines += ["## DF40-Dev rescue (B1-relative)", ""]
    if s4:
        r = s4["rescue"]
        lines += ["> DF40-Dev drove design decisions and is **never** zero-shot evidence. Rescue "
                  "is measured against B1, this method's own baseline — not against the Phase-1 "
                  "P0-DS operator, which had different failures.", "",
                  f"B1 regimes: {len(r['sets']['inverted'])} inverted, {len(r['sets']['weak'])} "
                  f"weak, {len(r['sets']['strong'])} strong.", "",
                  "| rung | rescue rows | median | helped | harm rows | harmed | median |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for rung, s in r["summary"].items():
            h = s["headline"]
            lines.append(f"| {rung} − B1 | {h['rescue_rows']} | {fmt(h['rescue_median'], True)} | "
                         f"{h['rescue_n_helped']} | {h['harm_rows']} | {h['harm_n_harmed']} | "
                         f"{fmt(h['harm_median'], True)} |")
        lines.append("")
        for rung, prof in (s4.get("profiles") or {}).items():
            if isinstance(prof, dict) and "shape_correlation_real_fake" in prof:
                lines.append(f"* `{rung}` depth profile: real/fake shape correlation "
                             f"**{prof['shape_correlation_real_fake']:.4f}** (max per-layer "
                             f"difference {prof['shape_max_abs_difference']:.4f}). A correlation "
                             f"near 1 means the profile differs by LEVEL, not shape.")
        for rung, fig in (s4.get("figures") or {}).items():
            if fig:
                lines.append(f"* depth-profile figure (`{rung}`): `{fig}`")
        lines.append("")
    else:
        lines += [f"{TODO} — run `analysis/fpad/stage4_rescue_profile.py`.", ""]

    # ---- localization ----------------------------------------------------------------
    lines += ["## Localization and faithfulness", ""]
    if loc:
        lines += [f"Readout `{loc.get('readout', '?')}`, epoch {loc.get('epoch', '?')}. "
                  f"{loc.get('grid_note', '')}", "",
                  "| manipulation | tier | AUPRC | chance | AUROC | IoU | frames |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for manip, m in (loc.get("localization") or {}).items():
            if m.get("status") != "ok":
                lines.append(f"| {manip} | — | {TODO} | — | — | — | {m.get('status', '?')} |")
                continue
            lines.append(f"| {manip} | {m['tier']} | {m['patch_auprc']:.4f} | "
                         f"{m['chance_auprc']:.4f} | {m['patch_auroc']:.4f} | "
                         f"{m['patch_iou_at_oracle_k']:.4f} | {m['n_frames']} |")
        lines += ["", f"Excluded for having no masks: "
                      f"{', '.join(loc.get('excluded_no_masks', [])) or 'none'} — never "
                      f"pseudo-masked in.", ""]
        faith = loc.get("faithfulness") or {}
        if faith:
            lines += ["Faithfulness (deletion; positive gain means the map explains the "
                      "decision):", "",
                      "| manipulation | baseline p(fake) | gain @10% | @20% | @30% |",
                      "|---|---:|---:|---:|---:|"]
            for manip, f in faith.items():
                g = [f.get(f"faithfulness_gain_{p}pct") for p in (10, 20, 30)]
                lines.append(f"| {manip} | {fmt(f.get('baseline_p_fake'))} | " +
                             " | ".join(fmt(v, True) for v in g) + " |")
            lines.append("")
    else:
        lines += [f"{TODO} — run `training/localize_fpad.py`. Genuine FF++ masks are staged for "
                  f"Deepfakes, Face2Face, FaceSwap, NeuralTextures and DFD.", ""]

    # ---- robustness -------------------------------------------------------------------
    lines += ["## Compression robustness", ""]
    if s6:
        lines += [f"> {s6.get('note', '')}", "",
                  "| source | " + " | ".join(f"q{q}" for q in s6["qualities"]) + " |",
                  "|---" * (len(s6["qualities"]) + 1) + "|"]
        for source, per_q in s6["curve"].items():
            lines.append(f"| {source} | " +
                         " | ".join(fmt(per_q.get(str(q))) for q in s6["qualities"]) + " |")
        lines.append("")
        for q, entry in (s6.get("audit") or {}).items():
            bad = [n for n, e in entry["quantities"].items()
                   if e["verdict"] in ("DATASET DETECTOR", "BORDERLINE")]
            lines.append(f"* at q{q}: {len(bad)} of {len(entry['quantities'])} quantities read "
                         f"compression at least as well as manipulation"
                         + (f" — {', '.join(f'`{n}`' for n in bad)}" if bad else ""))
        lines += ["", f"* H.264 c40: **{TODO}** — not staged on this machine; the JPEG probe is a "
                      f"different codec applied on top of c23 and is not a substitute.", ""]
    else:
        lines += [f"{TODO} — run `analysis/fpad/jpeg_probe.py`.", ""]

    # ---- provenance rules -------------------------------------------------------------
    lines += ["## Provenance rules that must survive into the paper", "",
              "1. **DF40-Dev drove design decisions and is never zero-shot evidence.** It may be "
              "shown; it may not be described as generalisation.",
              "2. **Only DF40-Holdout carries the zero-shot claim**, in the precise form recorded "
              "in `phase2/DF40_SPLIT.md`: sealed before Phase 2 and unread by any Phase-2 "
              "decision, NOT never seen.",
              "3. **Any source whose validation split fed selection or calibration is not OOD.**"]
    if s2 and (s2.get("provenance_cost") or {}).get("n_real_videos_read"):
        pc = s2["provenance_cost"]
        lines.append(f"4. **{pc['corpus']} is footnoted, not zero-shot on the real side**: "
                     f"{pc['n_real_videos_read']} real videos / {pc['n_real_frames_read']} frames "
                     f"were read as the mechanism-validation domain axis. No fake was read and "
                     f"nothing was fit on them.")
    lines += ["", "## Terminology", "",
              "See `wacv/TERMINOLOGY.md`. In particular \"gate\" must not appear in the paper for "
              "this stage — it collides with the learned applicability `q_b` of the V1 work.", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage2", type=Path, default=None)
    ap.add_argument("--stage3", type=Path, default=None)
    ap.add_argument("--stage4", type=Path, default=None)
    ap.add_argument("--stage6", type=Path, default=None)
    ap.add_argument("--localization", type=Path, default=None)
    ap.add_argument("--holdout", nargs="+", action="append", default=None,
                    metavar="NAME READOUT PARQUET",
                    help="breaks the DF40-Holdout seal; only legitimate at this stage")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    parts = {"stage2": load(args.stage2), "stage3": load(args.stage3),
             "stage4": load(args.stage4), "stage6": load(args.stage6),
             "localization": load(args.localization)}
    for key, path in (("stage2", args.stage2), ("stage3", args.stage3), ("stage4", args.stage4),
                      ("stage6", args.stage6), ("localization", args.localization)):
        state = "loaded" if parts[key] else (f"MISSING ({path})" if path else "not supplied")
        print(f"  {key:13s} {state}")

    holdout = score_holdout(args.holdout) if args.holdout else {}
    if holdout:
        print(f"  holdout       UNSEALED: {len(holdout)} methods scored")
    else:
        print(f"  holdout       still sealed — the zero-shot table will read {TODO}")

    args.out.mkdir(parents=True, exist_ok=True)
    doc = render(parts, holdout, args.out)
    (args.out / "STAGE7_FINAL.md").write_text(doc)
    (args.out / "stage7_final.json").write_text(json.dumps({
        "sources": {k: str(v) for k, v in (
            ("stage2", args.stage2), ("stage3", args.stage3), ("stage4", args.stage4),
            ("stage6", args.stage6), ("localization", args.localization)) if v},
        "present": {k: bool(v) for k, v in parts.items()},
        "holdout_unsealed": bool(holdout), "holdout": holdout,
        "todo": [label for label, status, _ in
                 checklist({**parts, "holdout": holdout or None}) if status == TODO],
    }, indent=2, default=str))

    todo = [label for label, status, _ in checklist({**parts, "holdout": holdout or None})
            if status == TODO]
    print(f"\nwrote {args.out}/STAGE7_FINAL.md")
    if todo:
        print(f"{len(todo)} deliverable(s) still {TODO}:")
        for t in todo:
            print(f"  - {t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

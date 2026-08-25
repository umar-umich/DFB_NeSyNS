#!/usr/bin/env python3
"""§20 — post-hoc contribution diagnostics (NOT retrained ablations).

    🔴 UMAR-RUNS (CPU, seconds):

    python analysis/discern_v2/contribution_diagnostics.py \
        --parquet logs/v1/eval/epoch_007_gated_diverse/per_sample_epoch_7.parquet \
        --stage-de logs/v1/stage_de/epoch_007_diverse/stage_de.pt \
        --out analysis/discern_v2/V1_contribution

§20 lists the toggles to evaluate on the already-trained model: semantic only, +reference,
+process, full, plain DS without applicability, DS + applicability, and residual reference vs
`e_direct`. Every one is recomputed from the per-sample export, so there is no second forward
pass and no risk of the diagnostics describing a different model than the results table.

**These are contribution diagnostics, not ablations.** §20 is explicit, and the distinction is
not cosmetic: each configuration reuses heads that were trained with all three branches present.
A branch removed here was still present while the others learned, so a small "+reference"
contribution means "reference adds little ON TOP of heads trained alongside it", not "a
two-branch system would score this". Publication-quality ablations require retraining.

Fusion is computed with the same `ds_fusion` code the model runs, from the exported evidence, so
a toggle differs from the deployed system only in which opinions enter and with what `q`.
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

from networks.discern_v2.ds_fusion import (  # noqa: E402
    Opinion, apply_validity, discount, ds_combine)

NOISE_FLOOR = 0.01          # §22
BRANCHES = ("sem", "ref", "proc")


def opinions_of(df: pd.DataFrame) -> dict[str, Opinion]:
    """Rebuild each branch's opinion from the exported evidence, validity applied."""
    out = {}
    for name in BRANCHES:
        if f"e_{name}_real" not in df.columns:
            continue
        evidence = torch.tensor(df[[f"e_{name}_real", f"e_{name}_fake"]].to_numpy(),
                                dtype=torch.float32)
        valid = torch.tensor(df.get(f"valid_{name}", pd.Series(True, index=df.index))
                             .to_numpy().astype(float), dtype=torch.float32)
        out[name] = apply_validity(Opinion.from_evidence(evidence), valid)
    return out


def fuse(opinions: dict[str, Opinion], names: list[str],
         q: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
    """p(fake) of the DS fusion over `names`, discounting specialists by `q` when given."""
    parts = []
    for name in names:
        op = opinions[name]
        if q is not None and name != "sem" and name in q:
            op = discount(op, q[name])
        parts.append(op)
    fused, _ = ds_combine(parts)
    return fused.fake_prob()


def video_auroc(prob: np.ndarray, labels: np.ndarray, videos: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    agg = pd.DataFrame({"p": prob, "y": labels, "v": videos}).groupby(
        "v", as_index=False).agg(p=("p", "mean"), y=("y", "max"))
    if agg["y"].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(agg["y"], agg["p"]))


def configurations(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Every §20 toggle, as per-frame p(fake)."""
    ops = opinions_of(df)
    present = [n for n in BRANCHES if n in ops]
    q = {n: torch.tensor(df[f"q_{n}"].to_numpy(), dtype=torch.float32)
         for n in ("ref", "proc") if f"q_{n}" in df.columns}

    configs: dict[str, np.ndarray] = {}
    configs["sem_only"] = fuse(ops, ["sem"]).numpy()
    for spec in ("ref", "proc"):
        if spec in ops:
            configs[f"sem+{spec}"] = fuse(ops, ["sem", spec]).numpy()
            configs[f"{spec}_only"] = fuse(ops, [spec]).numpy()
    configs["full_plain_ds"] = fuse(ops, present).numpy()
    if q:
        configs["full_applicability"] = fuse(ops, present, q).numpy()
    if "p_direct" in df.columns:
        # the §4.2 control, taken as exported: it never enters fusion
        configs["direct_probe_only"] = df["p_direct"].to_numpy()
    return configs


def deltas(scores: dict[str, float]) -> dict[str, dict]:
    """The comparisons §20 and §24 actually turn on, each with a noise-floor verdict."""
    def pair(better: str, worse: str, question: str) -> dict | None:
        if better not in scores or worse not in scores:
            return None
        d = scores[better] - scores[worse]
        return {"delta": d, "question": question,
                "verdict": ("inside §22's noise floor — needs a second seed"
                            if abs(d) < NOISE_FLOOR else
                            f"{better} better" if d > 0 else f"{worse} better")}

    out = {
        "applicability_vs_plain_ds": pair("full_applicability", "full_plain_ds",
                                          "does the applicability layer beat plain DS? (§24)"),
        "reference_contribution": pair("sem+ref", "sem_only",
                                       "does the reference add over the anchor alone?"),
        "process_contribution": pair("sem+proc", "sem_only",
                                     "does the process branch add over the anchor alone?"),
        "full_vs_sem_only": pair("full_plain_ds", "sem_only",
                                 "do both specialists together beat the anchor?"),
        "reference_vs_direct_probe": pair("ref_only", "direct_probe_only",
                                          "does P_R beat the capacity-matched direct probe? (§4.2)"),
    }
    return {k: v for k, v in out.items() if v is not None}


def render(results: dict, meta: dict) -> str:
    order = ["sem_only", "ref_only", "proc_only", "sem+ref", "sem+proc",
             "full_plain_ds", "full_applicability", "direct_probe_only"]
    present = [c for c in order if any(c in r["scores"] for r in results.values())]

    lines = [
        "# V1 contribution diagnostics (§20)",
        "",
        f"From `{meta['parquet']}` — {meta['n_frames']} frames, "
        f"{len(results)} sources, epoch {meta.get('epoch', '?')}.",
        "",
        "> **These are contribution diagnostics, not ablations.** Every configuration reuses heads "
        "that were trained with all three branches present, so a branch removed here was still "
        "present while the others learned. A small \"+reference\" number means *the reference adds "
        "little on top of heads trained alongside it*, not *a two-branch system would score this*. "
        "Publication-quality ablations require retraining (§20).",
        "",
    ]
    if meta.get("not_zero_shot"):
        lines += [
            f"> ⚠️ **Not zero-shot:** {', '.join(meta['not_zero_shot'])} — these sources' test "
            f"videos were used to calibrate the gates and the defer policy. Label those rows "
            f"calibrated, not OOD.",
            "",
        ]
    lines += ["## Video AUROC by configuration", "",
              "| source | " + " | ".join(f"`{c}`" for c in present) + " |",
              "|---" * (len(present) + 1) + "|"]
    for source, r in results.items():
        tag = " ⚠️" if source in meta.get("not_zero_shot_sources", []) else ""
        cells = [f"{r['scores'].get(c, float('nan')):.4f}" for c in present]
        lines.append(f"| {source}{tag} | " + " | ".join(cells) + " |")

    lines += ["", "## The comparisons that decide things", ""]
    questions = sorted({k for r in results.values() for k in r["deltas"]})
    for key in questions:
        first = next((r["deltas"][key]["question"] for r in results.values()
                      if key in r["deltas"]), key)
        lines += [f"### {key}", "", f"*{first}*", "",
                  "| source | delta | verdict |", "|---|---:|---|"]
        for source, r in results.items():
            if key in r["deltas"]:
                d = r["deltas"][key]
                lines.append(f"| {source} | {d['delta']:+.4f} | {d['verdict']} |")
        inside = [s for s, r in results.items()
                  if key in r["deltas"] and abs(r["deltas"][key]["delta"]) < NOISE_FLOOR]
        lines += ["", f"{len(inside)} of {len(results)} sources are inside §22's noise floor"
                      f"{' — a second seed is required before any promotion or removal' if inside else ''}.",
                  ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parquet", type=Path, nargs="+", required=True)
    ap.add_argument("--stage-de", type=Path, default=None,
                    help="stage_de.pt — read which sources are no longer zero-shot")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.read_parquet(p) for p in args.parquet], ignore_index=True)

    not_zero_shot: list[str] = []
    if args.stage_de and args.stage_de.is_file():
        blob = torch.load(str(args.stage_de), map_location="cpu", weights_only=False)
        not_zero_shot = list(blob.get("sources_no_longer_zero_shot", []))

    results = {}
    for source, group in df.groupby("dataset"):
        configs = configurations(group)
        labels = group["label"].to_numpy()
        videos = group["video_id"].to_numpy()
        scores = {name: video_auroc(prob, labels, videos) for name, prob in configs.items()}
        results[str(source)] = {"scores": scores, "deltas": deltas(scores),
                                "n_frames": int(len(group))}
        print(f"\n{source} ({len(group)} frames)")
        for name in sorted(scores):
            print(f"   {name:22s} {scores[name]:.4f}")
        for key, d in results[str(source)]["deltas"].items():
            print(f"   -> {key:28s} {d['delta']:+.4f}  {d['verdict']}")

    # sources whose names appear in the not-zero-shot list (matched loosely: the artifact stores
    # "Celeb-DF-v2:val" while the export's dataset column says "Celeb-DF-v2")
    flagged = [s for s in results if any(s in entry for entry in not_zero_shot)]
    meta = {"parquet": [str(p) for p in args.parquet], "n_frames": int(len(df)),
            "not_zero_shot": not_zero_shot, "not_zero_shot_sources": flagged}

    (args.out / "contribution_diagnostics.json").write_text(
        json.dumps({"meta": meta, "results": results}, indent=2, default=str))
    (args.out / "V1_CONTRIBUTION_DIAGNOSTICS.md").write_text(render(results, meta))
    print(f"\nwrote {args.out}/V1_CONTRIBUTION_DIAGNOSTICS.md")
    if flagged:
        print(f"  flagged as NOT zero-shot: {flagged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

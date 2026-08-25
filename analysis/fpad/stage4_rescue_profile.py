#!/usr/bin/env python3
"""Stage 4 — DF40-Dev rescue (B1-relative) and the depth profile.

    🔴 UMAR-RUNS (CPU, ~1 min):

    python analysis/fpad/stage4_rescue_profile.py \
        --rung B1 direct logs/fpad/score/B1_df40dev/profile_epoch_009.parquet \
        --rung B2 traj   logs/fpad/score/B2_df40dev/profile_epoch_009.parquet \
        --rung B3 traj   logs/fpad/score/B3_df40dev/profile_epoch_009.parquet \
        --out wacv

Rescue is defined relative to B1, not to P0-DS
----------------------------------------------
The eight inverted rows recorded in Phase-1
(`analysis/discern_v2/A1_complementarity/DF40/inversion_rows_detected.csv`) were measured on
DiCoME's **P0-DS** operator, not on this paper's baseline. They say where the anchor FAMILY failed.
For this paper there is no CLIP anchor at all, so "rescue" has to mean *rescue relative to B1* —
ordinary fine-tuned FS-VFM, this method's own direct baseline. **A row that P0-DS inverted but B1
already handles is not a rescue for us.** (Umar, 2026-08-21.)

So B1's per-row AUROC is computed FIRST, B1's own weak and inverted sets are defined from it, and
the headline is B1 -> B2 -> B3 on that set. The eight P0-DS rows are shown alongside, labelled as
lineage only.

Harm is scored on B1's STRONG rows, because "harm" only means something where the baseline was
already right — the same asymmetry the V1 analysis used.

The depth profile, plotted twice on purpose
-------------------------------------------
At epoch 0 the measured profile was
`[0.0144, 0.0129, 0.0116, 0.0113, 0.0247, 0.6875]` for the ordinary student: layer 24 is ~50x every
earlier layer. A single linear-axis plot of that is not a depth profile, it is a bar at layer 24.

So two panels, and the second is the one that answers the brief's question:

* **raw, log y** — the honest magnitudes, showing how concentrated adaptation is.
* **level-removed** — each sample's profile divided by its own mean, so the curve is SHAPE. The
  brief asks "where in the hierarchy manipulation supervision forces departure from the prior" and
  whether the shape differs between real and fake and across families; that is a question about
  relative departure, which the raw scale hides.

Shape difference is also quantified rather than left to the eye: per-layer real-vs-fake
separability, and the correlation between the real and fake mean shapes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "analysis" / "discern_v2"))

from domain_audit import separability  # noqa: E402  (reused verbatim)

READOUTS = {"direct": "p_direct", "traj": "p_traj", "spatial": "p_spatial"}
NOISE = 0.01
SPLIT_FILE = REPO / "configs/discern_v2/df40_split.json"
# The Phase-1 P0-DS inversions. LINEAGE ONLY — see the module docstring.
P0DS_INVERTED = {
    "sadtalker_ff": 0.2534, "danet_cdf": 0.2855, "tpsm_ff": 0.2903, "danet_ff": 0.4007,
    "mcnet_cdf": 0.4042, "mcnet_ff": 0.4320, "facevid2vid_ff": 0.4796, "tpsm_cdf": 0.4997,
}


def video_auroc(prob, labels, videos) -> float:
    from sklearn.metrics import roc_auc_score

    agg = pd.DataFrame({"p": prob, "y": labels, "v": videos}).groupby(
        "v", as_index=False).agg(p=("p", "mean"), y=("y", "max"))
    if agg["y"].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(agg["y"], agg["p"]))


def layer_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("d_l") and c[3:].isdigit()]
    return sorted(cols, key=lambda c: int(c[3:]))


def per_method_auroc(df: pd.DataFrame, column: str) -> dict[str, float]:
    if column not in df.columns:
        return {}
    return {str(m): video_auroc(g[column].to_numpy(), g["label"].to_numpy(),
                                g["video_id"].to_numpy())
            for m, g in df.groupby("dataset")}


def families() -> dict[str, str]:
    if not SPLIT_FILE.is_file():
        return {}
    blob = json.loads(SPLIT_FILE.read_text())
    out = {}
    for part in ("dev", "holdout"):
        for m, v in (blob.get(part) or {}).items():
            out[m] = v.get("family", "unknown")
    return out


def rescue_table(scores: dict[str, dict[str, float]], weak_th: float,
                 inverted_th: float) -> dict:
    """B1-relative rescue and harm. B1 defines the row sets; B2/B3 are measured against it."""
    if "B1" not in scores:
        raise SystemExit(
            "B1 is required: rescue is defined relative to it, not to P0-DS. Score the B1 rung "
            "on DF40-Dev first.")
    b1 = scores["B1"]
    methods = sorted(b1)
    sets = {
        "inverted": [m for m in methods if np.isfinite(b1[m]) and b1[m] < inverted_th],
        "weak": [m for m in methods if np.isfinite(b1[m]) and inverted_th <= b1[m] < weak_th],
        "strong": [m for m in methods if np.isfinite(b1[m]) and b1[m] >= weak_th],
    }
    rows = {}
    for m in methods:
        entry = {"B1": b1[m],
                 "regime": next(k for k, v in sets.items() if m in v) if any(
                     m in v for v in sets.values()) else "unscored",
                 "p0ds_auroc": P0DS_INVERTED.get(m)}
        for rung in ("B2", "B3", "B4"):
            if rung in scores and m in scores[rung]:
                entry[rung] = scores[rung][m]
                entry[f"{rung}_minus_B1"] = scores[rung][m] - b1[m]
        if "B2" in entry and "B3" in entry:
            entry["B3_minus_B2"] = entry["B3"] - entry["B2"]
        rows[m] = entry

    summary = {}
    present = [r for r in ("B2", "B3", "B4") if r in scores]
    for rung in present:
        deltas = {regime: [rows[m][f"{rung}_minus_B1"] for m in members
                           if f"{rung}_minus_B1" in rows[m]]
                  for regime, members in sets.items()}
        summary[rung] = {
            regime: {
                "n": len(v),
                "median": float(np.median(v)) if v else None,
                "n_helped": int(sum(d > NOISE for d in v)),
                "n_harmed": int(sum(d < -NOISE for d in v)),
            } for regime, v in deltas.items()}
        # the headline: rescue on B1's failing rows, harm on B1's strong rows
        rescue = deltas["inverted"] + deltas["weak"]
        summary[rung]["headline"] = {
            "rescue_rows": len(rescue),
            "rescue_median": float(np.median(rescue)) if rescue else None,
            "rescue_n_helped": int(sum(d > NOISE for d in rescue)),
            "harm_rows": len(deltas["strong"]),
            "harm_n_harmed": int(sum(d < -NOISE for d in deltas["strong"])),
            "harm_median": float(np.median(deltas["strong"])) if deltas["strong"] else None,
        }
    return {"rows": rows, "sets": {k: sorted(v) for k, v in sets.items()}, "summary": summary,
            "thresholds": {"weak": weak_th, "inverted": inverted_th, "noise": NOISE}}


def depth_profile(df: pd.DataFrame, fam: dict[str, str]) -> dict:
    """Mean d_l by depth, for real vs fake and per family, raw and level-removed."""
    cols = layer_columns(df)
    if not cols:
        return {"status": "no d_l* columns"}
    D = df[cols].to_numpy(dtype=np.float64)
    level = D.mean(axis=1, keepdims=True)
    shape = D / np.clip(level, 1e-12, None)          # each sample's profile, level removed
    label = df["label"].to_numpy()

    def stats(mask):
        if mask.sum() < 2:
            return None
        return {"n": int(mask.sum()),
                "raw_mean": D[mask].mean(axis=0).tolist(),
                "shape_mean": shape[mask].mean(axis=0).tolist()}

    out = {"layers": [int(c[3:]) for c in cols],
           "real": stats(label == 0), "fake": stats(label == 1)}
    # per-layer real vs fake separability, and whether the SHAPE differs
    out["per_layer_real_vs_fake"] = {
        c: separability(D[label == 0, i], D[label == 1, i]) for i, c in enumerate(cols)}
    out["shape_per_layer_real_vs_fake"] = {
        c: separability(shape[label == 0, i], shape[label == 1, i]) for i, c in enumerate(cols)}
    if out["real"] and out["fake"]:
        r, f = np.array(out["real"]["shape_mean"]), np.array(out["fake"]["shape_mean"])
        out["shape_correlation_real_fake"] = float(np.corrcoef(r, f)[0, 1])
        out["shape_max_abs_difference"] = float(np.abs(r - f).max())
    out["by_family"] = {}
    df = df.assign(_fam=df["dataset"].map(lambda m: fam.get(str(m), "unknown")))
    for family, g in df.groupby("_fam"):
        idx = df.index.get_indexer(g.index)
        out["by_family"][str(family)] = {
            "n": int(len(g)), "methods": sorted(set(g["dataset"].astype(str))),
            "shape_mean": shape[idx].mean(axis=0).tolist(),
            "raw_mean": D[idx].mean(axis=0).tolist()}
    out["by_method"] = {}
    for method, g in df.groupby("dataset"):
        idx = df.index.get_indexer(g.index)
        out["by_method"][str(method)] = {"n": int(len(g)),
                                          "shape_mean": shape[idx].mean(axis=0).tolist()}
    return out


def plot_profile(prof: dict, out: Path, tag: str) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    if prof.get("status") or not prof.get("real"):
        return None
    layers = prof["layers"]
    x = np.arange(len(layers))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    for name, style in (("real", "-o"), ("fake", "--s")):
        if prof.get(name):
            axes[0].plot(x, prof[name]["raw_mean"], style, label=name)
            axes[1].plot(x, prof[name]["shape_mean"], style, label=name)
    axes[0].set_yscale("log")
    axes[0].set_title("raw mean $d_l^{TS}$ (log y)\nadaptation magnitude by depth")
    axes[1].axhline(1.0, color="grey", lw=0.8, ls=":")
    axes[1].set_title("level-removed SHAPE\n(profile / its own mean)")
    for family, e in sorted(prof.get("by_family", {}).items()):
        axes[2].plot(x, e["shape_mean"], "-o", label=f"{family} (n={e['n']})", alpha=0.85)
    axes[2].axhline(1.0, color="grey", lw=0.8, ls=":")
    axes[2].set_title("shape by DF40 family")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(layers)
        ax.set_xlabel("encoder block (1-indexed)")
        ax.legend(fontsize=7)
    fig.suptitle(f"Depth-resolved adaptation profile — {tag}")
    fig.tight_layout()
    dest = out / f"depth_profile_{tag}.png"
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    return str(dest)


def render(rescue: dict, profiles: dict, figures: dict, meta: dict) -> str:
    lines = [
        "# Stage 4 — DF40-Dev rescue and the depth profile",
        "",
        f"Rungs: {', '.join(meta['rungs'])} · epoch {meta.get('epoch', '?')} · "
        f"{meta['n_methods']} DF40-Dev methods · {meta['n_frames']} frames.",
        "",
        "> **Rescue is B1-relative.** The eight inverted rows from Phase 1 were measured on "
        "DiCoME's P0-DS operator, and this paper has no CLIP anchor. A row P0-DS inverted that B1 "
        "already handles is not a rescue for us, so B1's own AUROC defines the row sets and the "
        "P0-DS numbers are shown as lineage only.",
        "",
        f"B1's regimes at thresholds inverted < {rescue['thresholds']['inverted']}, "
        f"weak < {rescue['thresholds']['weak']}: "
        f"**{len(rescue['sets']['inverted'])} inverted**, "
        f"**{len(rescue['sets']['weak'])} weak**, "
        f"**{len(rescue['sets']['strong'])} strong**.",
        "",
        "## Headline: rescue where B1 fails, harm where B1 is strong", "",
        "| rung | rescue rows | median delta | helped | harm rows | harmed | median delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rung, s in rescue["summary"].items():
        h = s["headline"]
        def f(v):
            return "—" if v is None else f"{v:+.4f}"
        lines.append(f"| {rung} − B1 | {h['rescue_rows']} | {f(h['rescue_median'])} | "
                     f"{h['rescue_n_helped']} | {h['harm_rows']} | {h['harm_n_harmed']} | "
                     f"{f(h['harm_median'])} |")

    lines += ["", "## Per method", "",
              "| method | B1 | B2 | B3 | B2−B1 | B3−B1 | B3−B2 | B1 regime | P0-DS (lineage) |",
              "|---|---:|---:|---:|---:|---:|---:|---|---:|"]
    for m, e in sorted(rescue["rows"].items(), key=lambda kv: kv[1]["B1"]):
        def g(k, sign=False):
            v = e.get(k)
            if v is None or not np.isfinite(v):
                return "—"
            return f"{v:+.4f}" if sign else f"{v:.4f}"
        p0 = e.get("p0ds_auroc")
        lines.append(f"| {m} | {g('B1')} | {g('B2')} | {g('B3')} | {g('B2_minus_B1', True)} | "
                     f"{g('B3_minus_B1', True)} | {g('B3_minus_B2', True)} | {e['regime']} | "
                     f"{'—' if p0 is None else f'{p0:.4f}'} |")

    for tag, prof in profiles.items():
        if prof.get("status"):
            continue
        lines += ["", f"## Depth profile — {tag}", "",
                  "| layer | mean $d_l$ real | mean $d_l$ fake | real-vs-fake sep (raw) | "
                  "sep (shape) |", "|---|---:|---:|---:|---:|"]
        for i, layer in enumerate(prof["layers"]):
            col = f"d_l{layer}"
            r = prof["real"]["raw_mean"][i] if prof.get("real") else float("nan")
            f_ = prof["fake"]["raw_mean"][i] if prof.get("fake") else float("nan")
            lines.append(f"| {layer} | {r:.5f} | {f_:.5f} | "
                         f"{prof['per_layer_real_vs_fake'][col]:.4f} | "
                         f"{prof['shape_per_layer_real_vs_fake'][col]:.4f} |")
        if "shape_correlation_real_fake" in prof:
            lines += ["",
                      f"Real and fake mean SHAPES correlate at "
                      f"**{prof['shape_correlation_real_fake']:.4f}**, with a maximum per-layer "
                      f"difference of **{prof['shape_max_abs_difference']:.4f}**. A correlation "
                      f"near 1 with a small maximum difference means the two classes depart from "
                      f"the prior in the same PATTERN and differ mainly in magnitude — which "
                      f"would make the depth profile a level, not a shape.",
                      ""]
        if figures.get(tag):
            lines += [f"Figure: `{figures[tag]}` — raw magnitudes on a log axis, the "
                      f"level-removed shape, and the shape per DF40 family. The log axis is not "
                      f"cosmetic: the last block's delta is roughly fifty times every earlier "
                      f"layer, so a linear plot shows one bar and no profile.", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rung", nargs="+", action="append", required=True,
                    metavar="NAME READOUT PARQUET")
    ap.add_argument("--weak-threshold", type=float, default=0.70)
    ap.add_argument("--inverted-threshold", type=float, default=0.50)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allow-holdout", action="store_true")
    args = ap.parse_args()

    fam = families()
    holdout = set()
    if SPLIT_FILE.is_file():
        holdout = set(json.loads(SPLIT_FILE.read_text()).get("holdout", {}))

    scores, frames, epoch = {}, {}, None
    for spec in args.rung:
        if len(spec) < 3:
            raise SystemExit(f"--rung needs NAME READOUT PARQUET, got {spec}")
        name, readout, *paths = spec
        df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
        leaked = sorted(set(df["dataset"].astype(str)) & holdout)
        if leaked and not args.allow_holdout:
            raise SystemExit(
                f"{name} was scored on DF40-Holdout methods {leaked}. Stage 4 drives design "
                f"decisions, so reading Holdout here would destroy the only zero-shot claim "
                f"Stage 7 can make.")
        column = READOUTS[readout]
        if column not in df.columns:
            raise SystemExit(
                f"{name} was declared readout `{readout}` but its profile has no `{column}` "
                f"column. Re-score with --traj-head for a trajectory rung; falling back to the "
                f"other readout would mislabel the rung.")
        scores[name] = per_method_auroc(df, column)
        frames[name] = df
        meta_path = Path(paths[0]).parent / "score_meta.json"
        if meta_path.is_file():
            epoch = json.loads(meta_path.read_text()).get("epoch", epoch)
        print(f"{name:3s} readout={readout:6s} {len(scores[name])} methods, {len(df)} frames")

    rescue = rescue_table(scores, args.weak_threshold, args.inverted_threshold)
    print(f"\nB1 regimes: inverted={rescue['sets']['inverted']}")
    print(f"            weak={rescue['sets']['weak']}")
    print(f"            strong={len(rescue['sets']['strong'])} methods")
    for rung, s in rescue["summary"].items():
        h = s["headline"]
        med = "n/a" if h["rescue_median"] is None else f"{h['rescue_median']:+.4f}"
        hmed = "n/a" if h["harm_median"] is None else f"{h['harm_median']:+.4f}"
        print(f"  {rung}-B1: rescue on {h['rescue_rows']} rows, median {med} "
              f"(helped {h['rescue_n_helped']}) · harm on {h['harm_rows']} strong rows, "
              f"median {hmed} (harmed {h['harm_n_harmed']})")

    args.out.mkdir(parents=True, exist_ok=True)
    profiles, figures = {}, {}
    for name, df in frames.items():
        profiles[name] = depth_profile(df, fam)
        figures[name] = plot_profile(profiles[name], args.out, name)
        p = profiles[name]
        if not p.get("status"):
            print(f"  {name} depth profile: shape corr(real,fake) = "
                  f"{p.get('shape_correlation_real_fake', float('nan')):.4f}, "
                  f"max shape diff {p.get('shape_max_abs_difference', float('nan')):.4f}")

    meta = {"rungs": list(scores), "epoch": epoch,
            "n_methods": len(next(iter(scores.values()))),
            "n_frames": int(sum(len(d) for d in frames.values()))}
    (args.out / "stage4_rescue_profile.json").write_text(json.dumps(
        {"meta": meta, "rescue": rescue, "profiles": profiles, "figures": figures},
        indent=2, default=str))
    (args.out / "STAGE4_RESCUE_PROFILE.md").write_text(render(rescue, profiles, figures, meta))
    print(f"\nwrote {args.out}/STAGE4_RESCUE_PROFILE.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

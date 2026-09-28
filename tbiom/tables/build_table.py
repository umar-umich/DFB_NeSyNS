#!/usr/bin/env python3
"""The four-family DF40 results table: every method on this machine, per generator, frame and video.

    python tbiom/tables/build_table.py                 # rebuild analysis/tables/per_generator.csv
    python tbiom/tables/build_table.py --render        # + Markdown and LaTeX from that CSV

TWO STAGES, ON PURPOSE. Stage one writes ONE long-form row per (method, generator, level) into
`analysis/tables/per_generator.csv`; stage two renders every view from that file and nothing else.
So a new method, a new family split, or a different aggregation is a rendering change, never a
re-run of 195 GPU exports -- and the numbers in the paper, the proposal and the appendix are the
same numbers by construction.

WHAT A CELL IS. Each generator is scored on its own against the real set of its own domain; the
family column is then the mean +- standard deviation ACROSS the generators in that family. The
spread is the point: it says whether a family-level number is a property of the family or of one
easy generator inside it.

SCORING. Our rows replay the model's own DS fusion over whatever views the export actually carries
(two for DiCoME, three for ours) -- the branch list is read off the CSV columns rather than assumed,
so a missing view fails loudly instead of being silently dropped. Third-party rows use their own
`prob_class_1`. Video level always means: aggregate frames to videos first, then score, with video
identity from THE rule in `analysis/tbiom/video_id.py`.
"""
from __future__ import annotations

import argparse, glob, sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis/tbiom"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import symbolic_overlay as SO           # noqa: E402
import taxonomy as T                    # noqa: E402
from health import auroc                # noqa: E402
from video_id import video_id           # noqa: E402

S23 = ROOT / "logs/tbiom/stage23"
GEND_RUNS = ROOT / "logs/tbiom/baselines/gend_runs_ours"
OUT_CSV = ROOT / "analysis/tables/per_generator.csv"
FRAMES_ROOT = "/data/umar/Datasets/tbiom_frames/DF40"


@contextmanager
def branches(bs):
    old = list(SO.BRANCHES)
    SO.BRANCHES[:] = bs
    try:
        yield
    finally:
        SO.BRANCHES[:] = old


def views_of(d: pd.DataFrame) -> list[str]:
    """The branches this export actually carries, in the fusion's own order."""
    return [b for b in ("semantic", "artifact", "fsvfm") if f"p_{b}" in d and f"u_{b}" in d]


def score_views(path: Path) -> dict | None:
    """Frame and video AUROC for one of our exports, replaying its own DS fusion."""
    if not path.is_file():
        return None
    d = pd.read_csv(path)
    bs = views_of(d)
    if not bs:
        raise SystemExit(f"{path.name}: no view columns -- refusing to score")
    with branches(bs):
        frame = auroc(d["label"].to_numpy(), SO.fuse(d))
        cols = [f"{k}_{b}" for b in bs for k in ("p", "u")]
        d["v"] = video_id(d["key"])
        g = d.groupby("v", as_index=False).agg(**{c: (c, "mean") for c in cols}, y=("label", "max"))
        video = auroc(g["y"].to_numpy(), SO.fuse(g))
    return {"frame": frame, "video": video, "n_frames": len(d), "n_videos": len(g),
            "views": "+".join(bs)}


def gend_predictions(model: str) -> pd.DataFrame | None:
    """One DF40 run per baseline; the per-generator split happens here, by path, not by re-running."""
    hits = glob.glob(f"{GEND_RUNS}/{model}--DF40/**/test_predictions.csv", recursive=True)
    if not hits:
        return None
    d = pd.read_csv(sorted(hits)[0])
    d["subset"] = d["files"].str.split("/").str[-3]
    return d


def score_gend(pred: pd.DataFrame | None, fake_split: str, real_split: str) -> dict | None:
    if pred is None:
        return None
    d = pred[pred["subset"].isin({fake_split, real_split})]
    if d.empty or d["labels"].nunique() < 2:
        return None
    v = pd.DataFrame({"v": video_id(d["files"]), "y": d["labels"].to_numpy(),
                      "p": d["prob_class_1"].to_numpy()}).groupby("v").agg(y=("y", "max"), p=("p", "mean"))
    return {"frame": auroc(d["labels"].to_numpy(), d["prob_class_1"].to_numpy()),
            "video": auroc(v["y"].to_numpy(), v["p"].to_numpy()),
            "n_frames": len(d), "n_videos": len(v), "views": "prob_class_1"}


def collect() -> pd.DataFrame:
    rows, missing = [], []
    gend_cache = {m: gend_predictions(m) for m, _ in T.BASELINE_ROWS.values()}
    methods = {**T.OURS_ROWS, **T.BASELINE_ROWS}
    for label, (tag, kind) in methods.items():
        for fam, gen, var, name, fake_split, real_split in T.cells():
            r = (score_views(S23 / f"{tag}_{name}.csv") if kind == "views"
                 else score_gend(gend_cache[tag], fake_split, real_split))
            if r is None:
                missing.append(f"{label}/{name}")
                continue
            for lvl in ("frame", "video"):
                rows.append({"method": label, "kind": kind, "family": fam, "generator": gen,
                             "variant": var or "own", "set": name, "level": lvl,
                             "auroc": r[lvl], "n_frames": r["n_frames"], "n_videos": r["n_videos"],
                             "views": r["views"]})
            print(f"  {label:26s} {name:26s} frame {r['frame']:.4f}  video {r['video']:.4f}", flush=True)
    if missing:
        print(f"\n  {len(missing)} cells not yet on disk:")
        for m in missing[:12]:
            print(f"    {m}")
        if len(missing) > 12:
            print(f"    ... and {len(missing)-12} more")
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame, level: str) -> pd.DataFrame:
    """Family column = mean +- sd across the generators in that family."""
    d = df[df["level"] == level]
    g = d.groupby(["method", "family"])["auroc"].agg(["mean", "std", "count"]).reset_index()
    g["std"] = g["std"].fillna(0.0)
    return g


def cell(g: pd.DataFrame, method: str, fam: str, tex: bool = False) -> str:
    r = g[(g["method"] == method) & (g["family"] == fam)]
    if r.empty:
        return "--" if tex else "—"
    m, s, n = r.iloc[0][["mean", "std", "count"]]
    pm = r"$\pm$" if tex else "±"
    return f"{m:.3f} {pm} {s:.3f}" if n > 1 else f"{m:.3f}"


def render(df: pd.DataFrame, out_dir: Path):
    order = [m for m in {**T.OURS_ROWS, **T.BASELINE_ROWS} if m in set(df["method"])]
    fams = [f for f in T.FAMILY_ORDER if f in set(df["family"])]
    for level in ("video", "frame"):
        g = aggregate(df, level)
        cov = df[df["level"] == level].groupby("family")["set"].nunique()
        head = [f"{T.FAMILY_NAME[f]} ({cov.get(f, 0)})" for f in fams]

        L = [f"### DF40 {level}-level AUROC — mean ± sd across the generators in each family", "",
             "| Method | " + " | ".join(head) + " | Macro |", "|---|" + "---:|" * (len(fams) + 1)]
        for m in order:
            vals = [cell(g, m, f) for f in fams]
            fm = g[g["method"] == m]["mean"]
            macro = f"{fm.mean():.3f}" if len(fm) == len(fams) else "—"
            L.append(f"| {m} | " + " | ".join(vals) + f" | {macro} |")
        (out_dir / f"table_{level}.md").write_text("\n".join(L) + "\n")

        X = [r"\begin{tabular}{l" + "c" * (len(fams) + 1) + "}", r"\toprule",
             "Method & " + " & ".join(f"{T.FAMILY_NAME[f]} ({cov.get(f,0)})" for f in fams) + r" & Macro \\",
             r"\midrule"]
        for m in order:
            fm = g[g["method"] == m]["mean"]
            macro = f"{fm.mean():.3f}" if len(fm) == len(fams) else "--"
            name = rf"\textbf{{{m}}}" if "ours" in m else m
            X.append(f"{name} & " + " & ".join(cell(g, m, f, tex=True) for f in fams) + f" & {macro} " + r"\\")
        X += [r"\bottomrule", r"\end{tabular}"]
        (out_dir / f"table_{level}.tex").write_text("\n".join(X) + "\n")
        print(f"  wrote {out_dir}/table_{level}.{{md,tex}}")


def coverage(df: pd.DataFrame, out_dir: Path) -> list[str]:
    """Every method must score the SAME videos in a set. A disagreement means the rows are not
    comparable, and a silently different denominator is exactly how a table lies."""
    d = df[df["level"] == "video"]
    piv = d.pivot_table(index="set", columns="method", values="n_videos", aggfunc="max")
    bad = [s for s, r in piv.iterrows() if r.dropna().nunique() > 1]
    L = ["### Video counts per set (must agree across methods)", "",
         "| set | " + " | ".join(piv.columns) + " |", "|---|" + "---:|" * len(piv.columns)]
    for s, r in piv.iterrows():
        mark = " **!**" if s in bad else ""
        L.append(f"| {s}{mark} | " + " | ".join("—" if pd.isna(v) else f"{int(v)}" for v in r) + " |")
    L += ["", f"**Count mismatches:** {', '.join(bad) if bad else 'none'}", ""]
    (out_dir / "coverage.md").write_text("\n".join(L) + "\n")
    return bad


def appendix(df: pd.DataFrame, out_dir: Path):
    """Per-generator numbers, so a family mean can always be traced to the generators behind it."""
    d = df[df["level"] == "video"].copy()
    d["family"] = pd.Categorical(d["family"], T.FAMILY_ORDER, ordered=True)
    piv = d.pivot_table(index=["family", "generator", "variant"], columns="method",
                        values="auroc", observed=True).sort_index()
    order = [m for m in {**T.OURS_ROWS, **T.BASELINE_ROWS} if m in piv.columns]
    piv = piv[order]
    L = ["### DF40 video-level AUROC per generator", "",
         "`*` marks audio-driven lip-sync, which DF40 files under reenactment and our v3.1 "
         "objective excludes; it is kept here because the taxonomy is DF40's.", "",
         "| family | generator | real | " + " | ".join(order) + " |",
         "|---|---|---|" + "---:|" * len(order)]
    for (fam, gen, var), r in piv.iterrows():
        star = "*" if gen in T.LIPSYNC else ""
        L.append(f"| {T.FAMILY_NAME[fam]} | {gen}{star} | {var} | "
                 + " | ".join("—" if pd.isna(v) else f"{v:.4f}" for v in r) + " |")
    (out_dir / "per_generator.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true", help="also render Markdown and LaTeX")
    ap.add_argument("--render-only", action="store_true", help="render from the existing CSV")
    a = ap.parse_args()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if a.render_only:
        df = pd.read_csv(OUT_CSV)
    else:
        df = collect()
        if df.empty:
            raise SystemExit("no results on disk yet")
        df.to_csv(OUT_CSV, index=False)
        print(f"\n  wrote {OUT_CSV}: {len(df)} rows, "
              f"{df['method'].nunique()} methods x {df['set'].nunique()} sets x 2 levels")
    if a.render or a.render_only:
        render(df, OUT_CSV.parent)
        appendix(df, OUT_CSV.parent)
        bad = coverage(df, OUT_CSV.parent)
        print(f"  wrote {OUT_CSV.parent}/per_generator.md, coverage.md")
        if bad:
            print(f"  !! video-count mismatch in {len(bad)} set(s): {', '.join(bad[:6])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

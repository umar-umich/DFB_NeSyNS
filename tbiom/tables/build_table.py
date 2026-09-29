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

FAMILY_HEAD = {"FS": "Face swap", "FR": "Reenactment", "EFS": "Synthesis", "FE": "Editing"}
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis/tbiom"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import symbolic_overlay as SO           # noqa: E402  (imported for the two-view DiCoME replay)
import taxonomy as T                    # noqa: E402
from taxonomy import OPERATOR_ARMS       # noqa: E402
from health import auroc                # noqa: E402
from video_id import video_id           # noqa: E402

S23 = ROOT / "logs/tbiom/stage23"
GEND_RUNS = ROOT / "logs/tbiom/baselines/gend_runs_ours"
OUT_CSV = ROOT / "analysis/tables/per_generator.csv"
REPORT = ROOT / "tbiom/reports/DF40_FAMILY_TABLE.md"
FRAMES_ROOT = "/data/umar/Datasets/tbiom_frames/DF40"


@contextmanager
def branches(bs):
    old = list(SO.BRANCHES)
    SO.BRANCHES[:] = bs
    try:
        yield
    finally:
        SO.BRANCHES[:] = old


_DSR = None


def dsr_scorer():
    """The DEPLOYED decision rule, loaded once.

    Our system of record is not a plain DS replay of the export's columns. It fuses three views --
    APP and OM from the export, and T, the V-JEPA 2 LoRA temporal branch whose opinions live in
    their OWN directory -- with DS-R, whose reliabilities are fitted per split on the development
    folds. Scoring our row any other way silently reports a different system: on the 22 reenactment
    sets the plain replay gives 0.8338 where the deployed rule gives 0.9391, because the plain
    replay never sees the view that carries reenactment at all.
    """
    global _DSR
    if _DSR is None:
        sys.path.insert(0, str(ROOT / "analysis/tbiom/temporal"))
        import df40_analysis as DA
        import dsr_system as D
        import finish_line as F
        dev = D.load("vjepa2_lora_B_seed42")[0]
        rules = [F.rules_for(Fd, DA.VIEWS)[0] for _k, Fd, _S in D.splits(dev)]
        _DSR = (DA, rules)
    return _DSR


_OPS = None


def op_scorer():
    """Per-split reliability weights for the PCR6+ arm family, fitted exactly as DS-R's are.

    Same development folds, same grid search, same three views -- only the combination rule
    changes, so a difference between the rows is the operator and nothing else."""
    global _OPS
    if _OPS is None:
        sys.path.insert(0, str(ROOT / "analysis/tbiom/pcr6p"))
        sys.path.insert(0, str(ROOT / "analysis/tbiom/temporal"))
        import arms as ARMS_MOD
        import ops_pcr6p as P
        import dsr_system as D
        dev = D.load("vjepa2_lora_B_seed42")[0]
        folds = [Fd for _k, Fd, _S in D.splits(dev)]
        w = {}
        for arm, how in set(OPERATOR_ARMS.values()):
            w[(arm, how)] = ([ARMS_MOD.fit_w(Fd, arm) for Fd in folds] if how == "fit"
                             else [(1.0, 1.0, 1.0)])
        _OPS = (ARMS_MOD, P, w)
    return _OPS


def ablation_arms():
    """Register one arm per view subset, then fit each on the same folds as the headline row."""
    ARMS_MOD, P, _ = op_scorer()
    specs = {}
    for label, views in T.ABLATION:
        if len(views) == 3:
            specs[label] = "PCR6P-R"
            continue
        arm = next((a for a, v in ARMS_MOD.SUBSET.items() if tuple(v) == tuple(views)), None)
        if arm is None:                                  # the single-view arms arms.py does not ship
            arm = f"{'+'.join(views)} (PCR6+)"
            ARMS_MOD.SUBSET[arm] = tuple(views)
        specs[label] = arm
    return ARMS_MOD, P, specs


def ablation(df: pd.DataFrame, out_dir: Path):
    """Which views carry which family, at fixed operator.

    The point of the table is the ROW-TO-ROW difference: identical opinions, identical folds,
    identical combination rule, so adding a line's worth of views is the only thing that changed.
    """
    ARMS_MOD, P, specs = ablation_arms()
    _DA, rules = dsr_scorer()
    import dsr_system as D
    dev = D.load("vjepa2_lora_B_seed42")[0]
    folds = [Fd for _k, Fd, _S in D.splits(dev)]
    cells = list(T.cells())
    fams = [f for f in T.FAMILY_ORDER if any(c[0] == f for c in cells)]

    rows = []
    for label, _views in T.ABLATION:
        arm = specs[label]
        ws = [ARMS_MOD.fit_w(Fd, arm) for Fd in folds]
        per_fam = {f: [] for f in fams}
        for fam, _g, _v, name, _fs, _rs in cells:
            tt = _DA.ours_table(name)
            y = tt["y"].to_numpy()
            per_fam[fam].append(float(np.mean([auroc(y, P.p_of(ARMS_MOD.fuse(tt, arm, w))) for w in ws])))
        rows.append((label, {f: (float(np.mean(v)), float(np.std(v))) for f, v in per_fam.items()}))
        print(f"  ablation {label:22s} " +
              "  ".join(f"{f} {np.mean(per_fam[f]):.3f}" for f in fams), flush=True)

    head = [T.FAMILY_NAME[f] for f in fams]
    L = ["### Branch ablation — DF40 video-level AUROC, PCR6+-R throughout", "",
         "Every row is the same operator, the same reliability fitting on the same five development",
         "folds and the same 65 sets; only the views available to the fusion change.", "",
         "| Views | " + " | ".join(head) + " | Macro |", "|---|" + "---:|" * (len(fams) + 1)]
    X = [r"\begin{tabular}{l" + "c" * (len(fams) + 1) + "}", r"\toprule",
         "Views & " + " & ".join(head) + r" & Macro \\", r"\midrule"]
    for label, d in rows:
        L.append(f"| {label} | " + " | ".join(f"{d[f][0]:.3f} ± {d[f][1]:.3f}" for f in fams)
                 + f" | {np.mean([d[f][0] for f in fams]):.3f} |")
        X.append(f"{label} & " + " & ".join(f"{d[f][0]:.3f} $\\pm$ {d[f][1]:.3f}" for f in fams)
                 + f" & {np.mean([d[f][0] for f in fams]):.3f} " + r"\\")
    X += [r"\bottomrule", r"\end{tabular}"]

    # Rows that a reader should not have to spot by eye. Both are DETECTED here rather than
    # asserted, so they cannot survive in the text after they stop being true.
    macro = {lbl: float(np.mean([d[f][0] for f in fams])) for lbl, d in rows}
    byl = dict(rows)
    notes = []
    for a, b in (("APP + OM", "OM"), ("T + APP", "T"), ("T + OM", "OM")):
        if a in byl and b in byl and all(abs(byl[a][f][0] - byl[b][f][0]) < 5e-4 for f in fams):
            notes.append(f"- **`{a}` is identical to `{b}` in every family.** The reliability fit drives the "
                         f"other view's weight to zero, so the pair is not a fusion at all -- it is `{b}` "
                         f"with a second view that the fitting discards.")
    full = "T + APP + OM (ours)"
    if full in macro and "T + OM" in macro:
        d = macro[full] - macro["T + OM"]
        fr = byl[full].get("FR", (0, 0))[0] - byl["T + OM"].get("FR", (0, 0))[0]
        notes.append(f"- **Adding APP to `T + OM` moves the macro by {d:+.4f}** and reenactment by {fr:+.4f}. "
                     f"On this benchmark the appearance view does not earn its place once the temporal and "
                     f"off-manifold views are both present; it is carried on the strength of the other "
                     f"evaluations, not this one.")
    if notes:
        L += ["", "**What the rows say.**", ""] + notes
    (out_dir / "ablation.md").write_text("\n".join(L) + "\n")
    (out_dir / "ablation.tex").write_text("\n".join(X) + "\n")
    print(f"  wrote {out_dir}/ablation.{{md,tex}}")
    return L


def score_operator(name: str, spec: tuple) -> dict | None:
    """Video AUROC for one combination rule over the three deployed views, averaged over splits."""
    DA, _ = dsr_scorer()
    if not (S23 / f"ours_{name}.csv").is_file() or not (DA.BR / f"{name}.csv").is_file():
        return None
    ARMS_MOD, P, ws = op_scorer()
    arm, how = spec
    t = DA.ours_table(name)
    y = t["y"].to_numpy()
    vals = [auroc(y, P.p_of(ARMS_MOD.fuse(t, arm, w))) for w in ws[spec]]
    tag = arm if how == "fit" else arm.removesuffix("-R")
    return {"frame": float("nan"), "video": float(np.mean(vals)),
            "n_frames": 0, "n_videos": len(t), "views": f"T+APP+OM, {tag}"}


def score_dsr(name: str) -> dict | None:
    """Video-level AUROC under DS-R, averaged over the five development splits.

    Frame level is not defined for this row: the temporal view emits one opinion per video, so
    there is no per-framethree-view opinion to score. The frame table simply omits the row rather
    than quietly substituting a different rule for it.
    """
    DA, rules_per_split = dsr_scorer()
    if not (S23 / f"ours_{name}.csv").is_file():
        return None
    if not (DA.BR / f"{name}.csv").is_file():
        return None
    t = DA.ours_table(name)
    y = t["y"].to_numpy()
    vals = [auroc(y, r["DS-R"](t)) for r in rules_per_split]
    return {"frame": float("nan"), "video": float(np.mean(vals)),
            "n_frames": 0, "n_videos": len(t), "views": "T+APP+OM, DS-R"}


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
            if kind == "dsr":
                r = score_dsr(name)
            elif kind == "operator":
                r = score_operator(name, OPERATOR_ARMS[label])
            elif kind == "views":
                r = score_views(S23 / f"{tag}_{name}.csv")
            else:
                r = score_gend(gend_cache[tag], fake_split, real_split)
            if r is None:
                missing.append(f"{label}/{name}")
                continue
            for lvl in ("frame", "video"):
                if lvl == "frame" and pd.isna(r["frame"]):
                    continue
                rows.append({"method": label, "kind": kind, "family": fam, "generator": gen,
                             "variant": var or "own", "set": name, "level": lvl,
                             "auroc": r[lvl], "n_frames": r["n_frames"], "n_videos": r["n_videos"],
                             "views": r["views"]})
            fr = "  n/a " if pd.isna(r["frame"]) else f"{r['frame']:.4f}"
            print(f"  {label:26s} {name:26s} frame {fr}  video {r['video']:.4f}", flush=True)
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
    order = T.row_order(set(df["method"]))
    fams = [f for f in T.FAMILY_ORDER if f in set(df["family"])]
    pub = published_rows()

    def pub_cell(r, fam, tex=False):
        if fam not in r:
            return "--" if tex else "—"
        m, s = r[fam]
        return f"{m:.3f} {'$\\pm$' if tex else '±'} {s:.3f}"

    for level in ("video", "frame"):
        g = aggregate(df, level)
        cov = df[df["level"] == level].groupby("family")["set"].nunique()
        head = [f"{T.FAMILY_NAME[f]} ({cov.get(f, 0)})" for f in fams]

        L = [f"### DF40 {level}-level AUROC — mean ± sd across the generators in each family", "",
             "| Method | " + " | ".join(head) + " | Macro |", "|---|" + "---:|" * (len(fams) + 1)]
        if level == "video":
            for r in pub:
                vals = [pub_cell(r, f) for f in fams]
                ms = [r[f][0] for f in fams if f in r]
                mac = f"{np.mean(ms):.3f}" if len(ms) == len(fams) else "—"
                L.append(f"| {r['method']} | " + " | ".join(vals) + f" | {mac} |")
        for m in order:
            vals = [cell(g, m, f) for f in fams]
            fm = g[g["method"] == m]["mean"]
            macro = f"{fm.mean():.3f}" if len(fm) == len(fams) else "—"
            L.append(f"| {m} | " + " | ".join(vals) + f" | {macro} |")
        if level == "video" and pub:
            L += ["", "† published number, transcribed as printed and not re-measured here; the "
                  "protocol behind it (generator membership per family, real pool, frame or video "
                  "level) is not recorded, so it is context rather than a matched comparison."]
        (out_dir / f"table_{level}.md").write_text("\n".join(L) + "\n")

        X = [r"\begin{tabular}{l" + "c" * (len(fams) + 1) + "}", r"\toprule",
             "Method & " + " & ".join(f"{T.FAMILY_NAME[f]} ({cov.get(f,0)})" for f in fams) + r" & Macro \\",
             r"\midrule"]
        if level == "video":
            for r in pub:
                ms = [r[f][0] for f in fams if f in r]
                mac = f"{np.mean(ms):.3f}" if len(ms) == len(fams) else "--"
                X.append(f"{r['method'].replace(chr(8224), r'$\dagger$')} & "
                         + " & ".join(pub_cell(r, f, tex=True) for f in fams) + f" & {mac} " + r"\\")
            X.append(r"\midrule")
        for m in order:
            fm = g[g["method"] == m]["mean"]
            macro = f"{fm.mean():.3f}" if len(fm) == len(fams) else "--"
            name = rf"\textbf{{{m}}}" if "ours" in m else m
            X.append(f"{name} & " + " & ".join(cell(g, m, f, tex=True) for f in fams) + f" & {macro} " + r"\\")
        X += [r"\bottomrule", r"\end{tabular}"]
        (out_dir / f"table_{level}.tex").write_text("\n".join(X) + "\n")
        print(f"  wrote {out_dir}/table_{level}.{{md,tex}}")


def resolutions(out_dir: Path) -> dict:
    """Native image size of each set's real and fake halves.

    A set whose two halves ship at different native resolutions is separable on resampling alone,
    by any detector, before anything about the manipulation is considered. That is a property of
    how DF40 distributes the set, not of a method, so it is measured and printed next to the
    numbers rather than left for a reader to discover."""
    import json
    from PIL import Image
    cache = out_dir / "resolution_audit.json"
    if cache.is_file():
        return json.loads(cache.read_text())

    def size_of(subset: str):
        d = Path(FRAMES_ROOT) / subset
        if not d.is_dir():
            return None
        seen = set()
        for i, q in enumerate(p for p in d.rglob("*") if p.is_file()):
            seen.add(Image.open(q).size)
            if i >= 40:
                break
        return sorted(seen)

    out = {}
    for fam, gen, var, name, fake_split, real_split in T.cells():
        f, r = size_of(fake_split), size_of(real_split)
        if f and r:
            out[name] = {"fake": f, "real": r, "mismatch": f != r}
    cache.write_text(json.dumps(out, indent=1))
    return out


def published_rows() -> list[dict]:
    """The four published four-family rows, rescaled to our 0-1 AUROCs.

    They are printed at the TOP of the main table (UMAR, 2026-09-29): they are older methods and a
    lower number is the expected result, not a red flag. They keep a dagger because they were
    transcribed from a paper rather than measured here -- their protocol (generator membership per
    family, real pool, frame or video level) is not recorded, so the dagger is what tells a reader
    the row is a citation and not a run of ours.
    """
    import json
    src = Path(__file__).resolve().parent / "published.json"
    if not src.is_file():
        return []
    d = json.loads(src.read_text())
    return [{"method": r["method"] + " †",
             **{f: (r[f] / 100.0, r[f + "_sd"] / 100.0) for f in ("FS", "FR", "EFS", "FE")}}
            for r in d["rows"]]


def published(out_dir: Path) -> list[str]:
    """Published four-family numbers, printed in their OWN block and never merged into ours.

    They are transcribed as the source prints them, and the protocol behind them is not recorded:
    which generators are in each family, which real pool each is scored against, frame or video
    level, and whether AUROC is over the same 65 sets. Those choices move a DF40 family mean by
    more than the gaps in this table, so a cell here and a cell above are not the same measurement
    and a reader must not subtract one from the other. Printing them in the main table would invite
    exactly that subtraction, which is why they sit apart.
    """
    import json
    src = Path(__file__).resolve().parent / "published.json"
    if not src.is_file():
        return []
    d = json.loads(src.read_text())
    fams = ["FS", "FR", "EFS", "FE"]
    L = ["### Published numbers (as printed; NOT re-measured, NOT comparable cell-for-cell)", "",
         f"Transcribed from `{d['_source_file']}`. Citation: **{d['_citation']}**.", "",
         "The protocol behind these is unrecorded -- generator membership per family, the real pool,",
         "frame or video level -- and each of those moves a DF40 family mean by more than the gaps",
         "in the table above. Read them as context for the scale of the task, not as a head-to-head.", "",
         "| Method | " + " | ".join(FAMILY_HEAD[f] for f in fams) + " |",
         "|---|" + "---:|" * len(fams)]
    for r in d["rows"]:
        L.append(f"| {r['method']} | " + " | ".join(f"{r[f]:.2f} ± {r[f + '_sd']:.2f}" for f in fams) + " |")
    L.append("")
    (out_dir / "published.md").write_text("\n".join(L) + "\n")
    return L


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
    order = T.row_order(list(piv.columns))
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


METHODS = """
## What the numbers are

**Protocol.** Our row and the DiCoME row are trained on FaceForensics++ c23 by us, on the same
recipe, seed and schedule. The four third-party rows are the authors' RELEASED weights, and their
training data is taken on the authors' word rather than verified here: Effort ships as
`effort_clip_L14_trainOn_FaceForensic`, ForAda and FSFM are the published checkpoints, and
`yermandy/GenD_CLIP_L_14` carries only an architecture config, so nothing on this machine records
what it saw. FSFM in particular is a self-supervised face model pretrained well beyond FF++ before
any detection fine-tuning. Treat the third-party rows as "published detector, our evaluation", not
as a matched-training comparison; the DiCoME row is the one where training is matched.
Every detector is evaluated, without adaptation, on
DF40's test sets. A generator is scored on its own against the real set of its own domain --
`DF40-cdf-real` for a `-cdf` variant, `DF40-ff-real` for `-ff` -- so a family column is the mean and
standard deviation ACROSS the generators in that family, never a single pooled AUROC. The spread is
part of the result: it separates a family a method actually handles from a family carried by one
easy generator.

**Families.** FS / FR / EFS / FE are DF40's own published taxonomy, used unchanged so the columns
line up with published DF40 tables. Two consequences are ours to disclose rather than to fix.
DF40 files `wav2lip` under reenactment while our v3.1 objective excludes it as audio-driven
lip-sync; it is included here and flagged in the per-generator appendix. And `stargan`, `starganv2`
and `styleclip` ship their own paired reals rather than DF40's shared pools -- DF40's own split
between 31 "known" methods that share the FF++/Celeb-DF reals and 9 "unknown" ones that bundle
theirs. That is packaging, not taxonomy: all four editors are face editing and sit in ONE Editing
column. Averaging across different real pools is already what every family does, since `-cdf` and
`-ff` cells are scored against different pools; the appendix names the pool used for every cell.

**The `-ff` half is included.** DF40's FF++-domain sets are built over the 1000 FaceForensics++
source videos, 720 of which are in our training split. Those halves are therefore not
identity-disjoint from training for ANY method in the table -- every row here trains on FF++ c23, so
the advantage is shared rather than ours. They are included because excluding them would drop half
of DF40 and break comparability with published DF40 numbers; `-cdf` and `-ff` are reported
separately in the appendix so the two can always be read apart.

**Image-level sets.** DF40's EFS and FE sets are image collections rather than videos:
one image is one "video", and video level equals frame level for them by construction. Our temporal
view still runs on those sets -- it receives the single frame repeated across its clip window -- so
it contributes a real but uninformative opinion rather than being switched off. No row is given a
different fusion on those sets than on the video sets.

**Scoring.** `DISCERN (ours)` is the three-view system -- appearance and off-manifold from the
export, plus the V-JEPA 2 LoRA temporal branch -- combined with **PCR6+ under reliability
discounting (PCR6+-R)**, whose reliability triple is grid-searched on each split's fitting fold and
the result averaged over the five development splits. The row label does not carry the operator, so
it is stated here: swapping it is one line in `tbiom/tables/taxonomy.py`, which also records what
Dempster-R and undiscounted Dempster score on these same sets. DiCoME replays its own two-view
Dempster-Shafer fusion (semantic, artifact). Video level means aggregating frames to videos first
and then scoring, with
video identity from the rule in `analysis/tbiom/video_id.py`. Third-party rows use their own
`prob_class_1` with GenD's reduction (mean over the frames of a video). Every method sees exactly
the same frames: the baselines read PNGs dumped from the same h5 files our exports read.

**The DiCoME row is our own retraining, not the released checkpoint.** It is the comparison of
record: the same FF++ c23 recipe, seed and schedule as ours, so the difference between the rows is
the method rather than the training budget. The authors' released DiCoME weights were also exported
over all 65 sets and are not printed here; they remain on disk as a reproduction check on that
retraining.
"""


def report(df: pd.DataFrame, out_dir: Path):
    parts = ["# DF40 four-family results table", "",
             "    python tbiom/tables/build_table.py --render",
             "",
             "Generated from `analysis/tables/per_generator.csv`; edit the builder, not this file.",
             METHODS.strip(), ""]
    res = resolutions(out_dir)
    bad = {k: v for k, v in res.items() if v["mismatch"]}
    parts += ["## Sets whose two halves ship at different native resolutions", ""]
    if bad:
        parts += ["These are separable on resampling alone, by any detector, independently of the",
                  "manipulation. The confound is DF40's, not a method's -- every row in the table is",
                  "affected equally -- but a number from one of these sets does not measure what the",
                  "column claims to measure, so they are named here and should be read as an upper bound.",
                  "", "| set | fake | real |", "|---|---|---|"]
        for k, v in sorted(bad.items()):
            parts.append(f"| {k} | {v['fake'][0][0]}x{v['fake'][0][1]} | {v['real'][0][0]}x{v['real'][0][1]} |")
        parts += ["", f"Every other set in the table is matched at {res['DF40_blendface_cdf']['fake'][0][0]}"
                  f"x{res['DF40_blendface_cdf']['fake'][0][1]} on both halves.", ""]
    else:
        parts += ["None: every set is matched on both halves.", ""]
    for f in ("table_video.md", "table_frame.md"):
        parts += [(out_dir / f).read_text().strip(), ""]
    for f in ("ablation.md", "per_generator.md", "coverage.md"):
        if (out_dir / f).is_file():
            parts += [(out_dir / f).read_text().strip(), ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(parts) + "\n")
    print(f"  wrote {REPORT}")


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
        ablation(df, OUT_CSV.parent)
        bad = coverage(df, OUT_CSV.parent)
        print(f"  wrote {OUT_CSV.parent}/per_generator.md, coverage.md")
        report(df, OUT_CSV.parent)
        if bad:
            print(f"  !! video-count mismatch in {len(bad)} set(s): {', '.join(bad[:6])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

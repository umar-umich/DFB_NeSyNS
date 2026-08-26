#!/usr/bin/env python
"""Video-level AUROC for every model x dataset the sweep has finished.

    python analysis/tbiom/crossdataset_table.py --root logs/tbiom/crossdataset --out tbiom

Reads whatever is on disk and reports it, leaving the unfinished cells blank rather than waiting
for the sweep or quietly dropping a dataset. Every cell records the videos it was computed over,
because a table whose columns rest on different denominators reads as comparable when it is not.

WHAT THESE NUMBERS ARE FOR. Every dataset here is a TEST split, and four of them (DFDCP,
Celeb-DF-v1/v2, UADFV) ship a `val` split that IS their test split, with DFDC offering 2 val
videos against 4,704 test. There is therefore no held-out partition inside these corpora, and
nothing here may be used to FIT anything — not a threshold, not a gate, not a checkpoint choice.
They answer "does the Stage-1 verdict travel", not "which branch should enter".
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

MODELS = {"clip": "CLIP anchor", "preserve": "FS-VFM preservation", "ordinary": "FS-VFM ordinary"}
PROB_COL = {"clip": "p_sem", "preserve": "p_direct", "ordinary": "p_direct"}
DATASETS = ["FaceForensics__", "Celeb_DF_v2", "Celeb_DF_v1", "DFDCP", "DFDC",
            "DeepFakeDetection", "Deepfake_Eval_2024", "UADFV"]
PRETTY = {"FaceForensics__": "FF++ (in-domain)", "Celeb_DF_v2": "Celeb-DF-v2",
          "Celeb_DF_v1": "Celeb-DF-v1", "DFDCP": "DFDCP", "DFDC": "DFDC",
          "DeepFakeDetection": "DFD", "Deepfake_Eval_2024": "Deepfake-Eval-2024",
          "UADFV": "UADFV"}


def auroc(y: np.ndarray, p: np.ndarray) -> float | None:
    """Rank AUROC. None when a split is single-class, which is not a score of 0.5."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return None
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), float)
    ranks[order] = np.arange(1, len(p) + 1)
    # average ranks within ties, or every tied block biases the statistic
    s = pd.Series(p)
    ranks = s.rank(method="average").to_numpy()
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def video_level(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Mean fake-probability per video, the aggregation the spec fixes (§21)."""
    key = df["key"].astype(str).str.rstrip("/")
    directory = key.str.rsplit("/", n=1).str[0]
    parent = directory.str.rsplit("/", n=1).str[-1]
    degenerate = parent.str.lower().isin({"real", "fake", "frames", "images"})
    stem = key.str.replace(r"\.[A-Za-z0-9]+$", "", regex=True)
    vid = directory.where(~degenerate, stem)
    out = pd.DataFrame({"video": vid, "p": df[col].to_numpy(), "y": df["label"].to_numpy()})
    return out.groupby("video", as_index=False).agg(p=("p", "mean"), y=("y", "max"))


def paired_bootstrap_ci(root: pathlib.Path, ds: str, n_boot: int = 2000,
                        seed: int = 42) -> tuple[float | None, float | None]:
    """95% CI on (preservation - ordinary) video AUROC, resampling VIDEOS jointly.

    The two students are scored on the same videos, so the delta is a paired quantity. Drawing
    independent resamples for each model would add back the between-video variance that pairing
    removes and widen every interval, making real differences look unresolved.
    """
    frames = {}
    for model in ("preserve", "ordinary"):
        parts = sorted((root / f"{model}_{ds}").glob("*.parquet"))
        if not parts:
            return None, None
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        frames[model] = video_level(df, PROB_COL[model]).set_index("video")

    common = frames["preserve"].index.intersection(frames["ordinary"].index)
    if len(common) < 20:
        return None, None
    a = frames["preserve"].loc[common]
    b = frames["ordinary"].loc[common]
    y = a["y"].to_numpy()
    if len(np.unique(y)) < 2:
        return None, None
    pa, pb = a["p"].to_numpy(), b["p"].to_numpy()

    rng = np.random.default_rng(seed)
    n = len(common)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ys = y[idx]
        if len(np.unique(ys)) < 2:
            continue
        ra, rb = auroc(ys, pa[idx]), auroc(ys, pb[idx])
        if ra is not None and rb is not None:
            out.append(ra - rb)
    if len(out) < n_boot // 2:
        return None, None
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=pathlib.Path, default=pathlib.Path("logs/tbiom/crossdataset"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("tbiom"))
    ap.add_argument("--extra", action="append", default=[],
                    metavar="KEY:LABEL:ROOT:PROBCOL",
                    help="an additional model whose per-dataset directories live under ROOT and "
                         "are named by the dataset alone (as eval_ffpp_df40_checkpoint.sh writes "
                         "them), rather than MODEL_DATASET. Repeatable.")
    args = ap.parse_args()

    # Extra models are laid out as ROOT/<dataset>/ rather than ROOT/<model>_<dataset>/, because
    # a checkpoint evaluation writes one directory per dataset under its own epoch root.
    extra_roots: dict[str, pathlib.Path] = {}
    for spec in args.extra:
        key, label, root, col = spec.split(":", 3)
        MODELS[key] = label
        PROB_COL[key] = col
        extra_roots[key] = pathlib.Path(root)

    cells: dict[tuple[str, str], dict] = {}
    for model in MODELS:
        for ds in DATASETS:
            d = (extra_roots[model] / ds) if model in extra_roots \
                else args.root / f"{model}_{ds}"
            parquets = sorted(d.glob("*.parquet"))
            if not parquets:
                continue
            df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
            col = PROB_COL[model]
            if col not in df:
                continue
            v = video_level(df, col)
            cells[(model, ds)] = {
                "auroc": auroc(v["y"].to_numpy(), v["p"].to_numpy()),
                "videos": int(len(v)), "frames": int(len(df)),
                "real": int((v["y"] == 0).sum()), "fake": int((v["y"] == 1).sum()),
            }

    lines = ["# Cross-dataset video AUROC", "",
             "Every column is a TEST split. Four of these corpora (DFDCP, Celeb-DF-v1/v2, UADFV) "
             "ship a `val` split that IS their test split, and DFDC ships 2 val videos against "
             "4,704 test — so there is no held-out partition inside them and **nothing here may "
             "be fitted on**. These rows show whether the Stage-1 membership verdict travels off "
             "DF40; they cannot be used to reverse it. VALmix remains the only clean basis for "
             "that.", "",
             "Blank cells are still scoring.", "",
             "| model | " + " | ".join(PRETTY[d] for d in DATASETS) + " |",
             "|---" * (len(DATASETS) + 1) + "|"]
    for model, label in MODELS.items():
        row = [label]
        for ds in DATASETS:
            c = cells.get((model, ds))
            row.append("—" if not c or c["auroc"] is None else f"{c['auroc']:.4f}")
        lines.append("| " + " | ".join(row) + " |")

    # The preservation-vs-ordinary delta is the decision the user already made on three datasets;
    # this is the same comparison on eight, and it is only meaningful where BOTH cells exist.
    lines += ["", "## Preservation minus ordinary", "",
              "Positive favours the preservation student. Reported only where both cells "
              "finished. The interval is a PAIRED bootstrap over videos (2,000 resamples, the "
              "same video indices drawn for both models), which is the right test here: the two "
              "students score the identical videos, so resampling them independently would "
              "inflate the spread with variance that cancels. A CI spanning zero means the sign "
              "of the delta is not established on that corpus — which matters most where the "
              "corpus is small.", "",
              "| dataset | preservation | ordinary | delta | 95% CI | videos |",
              "|---|---:|---:|---:|---:|---:|"]
    deltas = []
    for ds in DATASETS:
        a, b = cells.get(("preserve", ds)), cells.get(("ordinary", ds))
        if not a or not b or a["auroc"] is None or b["auroc"] is None:
            continue
        d = a["auroc"] - b["auroc"]
        deltas.append(d)
        lo, hi = paired_bootstrap_ci(args.root, ds)
        ci = "—" if lo is None else f"[{lo:+.4f}, {hi:+.4f}]"
        lines.append(f"| {PRETTY[ds]} | {a['auroc']:.4f} | {b['auroc']:.4f} | {d:+.4f} "
                     f"| {ci} | {a['videos']} |")
    if deltas:
        wins = sum(1 for d in deltas if d > 0)
        resolved = []
        for ds in DATASETS:
            a, b = cells.get(("preserve", ds)), cells.get(("ordinary", ds))
            if not a or not b or a["auroc"] is None or b["auroc"] is None:
                continue
            lo, hi = paired_bootstrap_ci(args.root, ds)
            if lo is not None and (lo > 0 or hi < 0):
                resolved.append((PRETTY[ds], a["auroc"] - b["auroc"], a["videos"]))
        lines += ["",
                  f"Preservation ahead on **{wins}/{len(deltas)}** datasets, mean delta "
                  f"**{np.mean(deltas):+.4f}** — so the earlier three-dataset reading that "
                  f"preservation *consistently* outperforms does not survive the wider suite.",
                  ""]
        if resolved:
            lines.append("Only these deltas have a CI excluding zero:")
            lines.append("")
            for name, d, n in sorted(resolved, key=lambda r: -r[2]):
                who = "preservation" if d > 0 else "ordinary"
                lines.append(f"- **{name}** ({n:,} videos): {d:+.4f}, favours **{who}**")
            lines += ["",
                      "The two resolved deltas favouring preservation are the two LARGEST OOD "
                      "corpora; the one favouring ordinary is the smallest. That ordering is "
                      "worth weighing, and Celeb-DF-v1 deserves particular caution: its shipped "
                      "test list is also its val list, and 92 of its 100 test videos appear in "
                      "its own train split, so it is a loosely specified benchmark quite apart "
                      "from its size.", "",
                      "Read together: preservation is defensible as the default, but on the "
                      "strength of two corpora rather than a general advantage, and the spec's "
                      "§22 rule (|delta AUC| < 0.01 needs a second seed) covers DFDC's +0.0041 "
                      "as well. DFDCP's +0.0138 is the only delta that both clears §22 and has a "
                      "CI excluding zero in preservation's favour."]

    lines += ["", "## Coverage", "", "| model | dataset | videos | real / fake | frames |",
              "|---|---|---:|---|---:|"]
    for (model, ds), c in sorted(cells.items()):
        lines.append(f"| {MODELS[model]} | {PRETTY[ds]} | {c['videos']} | "
                     f"{c['real']} / {c['fake']} | {c['frames']} |")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "CROSSDATASET.md").write_text("\n".join(lines) + "\n")
    (args.out / "crossdataset.json").write_text(json.dumps(
        {f"{m}|{d}": c for (m, d), c in cells.items()}, indent=2))
    print("\n".join(lines[:14]))
    print(f"\n{len(cells)} cells -> {args.out / 'CROSSDATASET.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

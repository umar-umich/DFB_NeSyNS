#!/usr/bin/env python3
"""Baseline matrix: our v3.1-selected runs vs our trained DiCoME vs the GenD-repo detectors, frame- and video-level.

    python analysis/tbiom/baselines/build_baseline_matrix.py --out tbiom/reports/BASELINE_MATRIX.md

Every measured row scores the SAME frames per dataset: CDFv2/CDFv3/DFD/DFDC/DFDCP/DFEval24 use our crops (the
baselines read PNGs dumped from our h5 files); FFIW/KoDF/IDForge/DFDM use the GenD-release crops our h5 files were
built from. Scope filters as protocol v3.1: CDFv3 without IP_LAP; KoDF without audio-driven and adversarial subsets.
Our models: DS replay of their own views (frame = per-frame opinion; video = mean opinion per video, then fused —
the corrected-table rule). Baselines: prob_fake per frame; video = mean over frames (GenD's own reduction).
A video is the directory containing the frame (full path). Published numbers are listed separately, never mixed in.
"""
import argparse, glob, json, sys
from contextlib import contextmanager
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import symbolic_overlay as SO  # noqa: E402
from corrected_table_v2 import videos  # noqa: E402
from health import auroc  # noqa: E402
from video_id import video_id  # noqa: E402

S23 = Path("logs/tbiom/stage23"); B = Path("logs/tbiom/baselines")
SETS = ["CDFv2", "CDFv3", "DFD", "DFDC", "DFDCP", "DFEval24", "FFIW", "KoDF", "IDForge", "DFDM"]
OURS = {"R0 (v3.1, ep 6)": "run1auxedl_v2e06", "A (v3.1, ep 5)": "sbiarmA_e05",
        "B (v3.1, ep 8)": "sbiarmB_v2e08", "C (v3.1, ep 2)": "sbiarmC_v3e02"}
DICOME_FILE = {"CDFv2": "CDFv2_base", "DFDCP": "DFDCP_base"}
BASE = ["GenD_CLIP", "GenD_PE", "GenD_DINO", "Effort", "ForAda", "FSFM", "FSFM_Adapter"]
BASE_NAME = {"GenD_CLIP": "GenD-CLIP", "GenD_PE": "GenD-PE", "GenD_DINO": "GenD-DINO", "FSFM_Adapter": "FSFM-Adapter"}
GEND_CROPS = {"FFIW", "KoDF", "IDForge", "DFDM"}


@contextmanager
def branches(bs):
    old = list(SO.BRANCHES); SO.BRANCHES[:] = bs
    try: yield
    finally: SO.BRANCHES[:] = old


def scope(df, subset):
    keep = ~subset.str.contains("IP_LAP") & (subset != "fake-audio-driven") & ~subset.str.startswith("adv-")
    return df[keep.to_numpy()].reset_index(drop=True)


def ours(path):
    if not path.is_file(): return None
    d = pd.read_csv(path); d = scope(d, d["key"].astype(str).str.split("::").str[-1].str.split("/").str[-3])
    fr = auroc(d["label"].to_numpy(), SO.fuse(d))
    tmp = path.with_name(path.stem + ".__scoped.csv"); d.to_csv(tmp, index=False)
    try: g = videos(tmp)
    finally: tmp.unlink()
    return {"frame": fr, "video": auroc(g.y.to_numpy(), g.s.to_numpy()), "videos": len(g)}


def baseline(model, ds):
    root = B / ("gend_runs" if ds in GEND_CROPS else "gend_runs_ours")
    hits = glob.glob(f"{root}/{model}--{ds}/**/test_predictions.csv", recursive=True)
    if not hits: return None
    d = pd.read_csv(hits[0]); d = scope(d, d.files.str.split("/").str[-3])
    d["v"] = d.files.str.rsplit("/", n=1).str[0]
    v = d.groupby("v").agg(y=("labels", "max"), s=("prob_class_1", "mean"))
    return {"frame": auroc(d.labels.to_numpy(), d.prob_class_1.to_numpy()), "video": auroc(v.y.to_numpy(), v.s.to_numpy()),
            "videos": len(v)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=Path("tbiom/reports/BASELINE_MATRIX.md"))
    a = ap.parse_args()
    rows = {}
    for name, tag in OURS.items():
        rows[name] = {ds: ours(S23 / f"{tag}_{ds}.csv") for ds in SETS}
    with branches(["semantic", "artifact"]):
        rows["DiCoME (our training, v3.1, ep 5)"] = {ds: ours(B / "dicome_ours" / f"{DICOME_FILE.get(ds, ds)}.csv") for ds in SETS}
    for m in BASE:
        rows[BASE_NAME.get(m, m)] = {ds: baseline(m, ds) for ds in SETS}

    L = ["# Baseline matrix — frame- and video-level AUROC", "", __doc__.strip().split("\n\n", 1)[1], "",
         "All detectors trained on FF++ c23. One seed per row. — = not yet available.", ""]
    for lvl in ("video", "frame"):
        L += [f"## {lvl.capitalize()}-level AUROC", "", "| method | " + " | ".join(SETS) + " | mean (complete rows) |",
              "|---|" + "---:|" * (len(SETS) + 1)]
        for n, r in rows.items():
            vals = [r[ds][lvl] if r[ds] else None for ds in SETS]
            mean = f"{np.mean(vals):.4f}" if all(v is not None for v in vals) else "—"
            L.append(f"| {n} | " + " | ".join(f"{v:.4f}" if v is not None else "—" for v in vals) + f" | {mean} |")
        L.append("")
    L += ["## Video counts (must agree across rows)", "", "| dataset | " + " | ".join(rows) + " |", "|---|" + "---:|" * len(rows)]
    mism = []
    for ds in SETS:
        cnt = [rows[n][ds]["videos"] if rows[n][ds] else None for n in rows]
        L.append(f"| {ds} | " + " | ".join(str(c) if c is not None else "—" for c in cnt) + " |")
        if len({c for c in cnt if c is not None}) > 1: mism.append(ds)
    L += ["", f"**Count mismatches:** {', '.join(mism) if mism else 'none'}", ""]

    rep = json.load(open(HERE / "reported.json"))
    L += ["## Published numbers (as printed; not re-measured)", ""]
    for r in rep["rows"]:
        L.append(f"- **{r['method']}** ({r['level']}, {rep['sources'][r['source']]}): " +
                 ", ".join(f"{k} {v:.3f}" for k, v in r["auc"].items()))
    a.out.write_text("\n".join(L) + "\n")
    json.dump({n: r for n, r in rows.items()}, open(a.out.with_suffix(".json"), "w"), indent=1)
    print(f"wrote {a.out}; mismatches: {mism or 'none'}")


if __name__ == "__main__":
    main()

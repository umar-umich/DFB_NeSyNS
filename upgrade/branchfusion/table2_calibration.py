#!/usr/bin/env python3
"""Table 2 -- probability quality and uncertainty, per dataset: ECE, Brier, E-AURC.

    <discern_ext python> upgrade/branchfusion/table2_calibration.py

ECE and Brier ask whether a reported probability means what it says. E-AURC asks something the
other two cannot: whether the model knows WHICH of its own answers to distrust. A system can be
perfectly calibrated on average and still have no idea which individual video it is wrong about.

E-AURC, and why it rather than AURC. AURC is the area under the risk-coverage curve -- reject the
most uncertain fraction, measure the error on what remains, integrate. Its LEVEL is set by the
base error rate, so a strong detector scores well on it for reasons that have nothing to do with
its uncertainty. E-AURC subtracts the AURC an ORACLE ranking of the same predictions would reach,
which removes the accuracy term and leaves the quality of the ranking alone. It is the honest way
to compare uncertainty across systems of different accuracy, and it is why the column is E-AURC.

WHAT EACH SYSTEM RANKS BY, which is the asymmetry the table exists to show:
  ours        the fused VACUITY, the Theta mass of the ES-PCR6+ opinion -- an epistemic quantity
  baselines   1 - max p, the softmax confidence, because that is the only uncertainty a
              cross-entropy detector has. This is not a handicap imposed by us; it is what the
              model emits.
A gap in our favour therefore says the Dirichlet is doing work the decision score cannot. A gap
the other way would say it is not, and the table is built to let that come out.
"""
from __future__ import annotations

import glob
import json
import pathlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                               # noqa: E402
import branches as B                                             # noqa: E402
import opinions as O                                             # noqa: E402

BASE = C.ROOT / "logs/tbiom/baselines"
#: Datasets whose baseline runs use the GenD release's OWN crops rather than ours. KoDF left this
#: set on 2026-10-14: its adversarial subsets exist only in our h5, so widening its census meant
#: dumping it from our crops, and a dataset must not mix two crop sources inside one column.
GEND_CROPS = {"FFIW", "IDForge", "DFDM"}
MODELS = ["GenD_CLIP", "GenD_PE", "GenD_DINO", "FSFM", "FSFM_Adapter", "Effort", "ForAda"]
PRETTY = {"GenD_CLIP": "GenD-CLIP", "GenD_PE": "GenD-PE", "GenD_DINO": "GenD-DINO",
          "FSFM_Adapter": "FSFM-Adapter"}
FINAL = ("A", "V", "O", "S16ctl")
NEW_SETS = ["UADFV", "DeepSpeak-1.1", "DeepSpeak-2.0"]
NOTE = """A ROW'S `mean` IS OVER ITS OWN COVERAGE, which is not a comparison. Our row spans twelve
sets, most baselines nine, DiCoME five -- and the three corpora only we cover are not the
easy ones, so our own mean is read off a harder average than anyone else's. The two
matched columns are the comparable figures:

- `mean (all)` -- every set every row in the table resolves. DiCoME's five-set coverage
  sets that width, so it is honest for all nine methods but narrow.
- `mean (wide)` -- every set that every method covering nine or more sets resolves. The
  narrow rows say in progress there rather than shrinking the column for everyone.

Rank methods on a matched column and say which one. The per-row mean belongs in the
per-dataset discussion, not in a ranking."""
TODO = "*in progress*"
OUT = C.REPORT / "TABLE2_CALIBRATION.md"


def ece(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    out = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            out += m.mean() * abs(float(p[m].mean()) - float(y[m].mean()))
    return float(out)


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def aurc(y, p, u, tau):
    """Area under the risk-coverage curve: keep the most CERTAIN first, integrate the error."""
    order = np.argsort(u)
    err = ((p >= tau).astype(int) != y).astype(float)[order]
    cov = np.arange(1, len(err) + 1)
    return float(np.mean(np.cumsum(err) / cov))


def e_aurc(y, p, u, tau):
    """AURC minus the AURC an oracle ranking of the SAME predictions reaches."""
    err = ((p >= tau).astype(int) != y).astype(float)
    return aurc(y, p, u, tau) - aurc(y, p, err, tau)


def baseline_videos(model, ds):
    root = BASE / ("gend_runs" if ds in GEND_CROPS else "gend_runs_ours")
    hits = glob.glob(f"{root}/{model}--{ds}/**/test_predictions.csv", recursive=True)
    if not hits:
        return None
    d = pd.read_csv(hits[0])
    # A SEVENTH hardcoded copy of the scope filter lived here. Same consequence as the others:
    # once BF_SCOPE could widen our rows, this one stayed at v3.1 and Table 2 compared rows over
    # different videos. `branches.in_scope` takes the subset Series directly.
    d = d[B.in_scope(d.files.str.split("/").str[-3]).to_numpy()]
    d = d.assign(v=d.files.str.rsplit("/", n=1).str[0])
    g = d.groupby("v").agg(y=("labels", "max"), s=("prob_class_1", "mean"))
    return g.y.to_numpy(), g.s.to_numpy()


#: DiCoME's own runs. It covers five of our sets and never ran FFIW, KoDF, IDForge or DFDM.
DICOME_RUNS = pathlib.Path("/data/umar/Repos/DiCoME/runs/dicome_trained")


def scope(df, subset):
    """The one scope predicate, for both readers in this file."""
    return df[B.in_scope(subset).to_numpy()].reset_index(drop=True)


def dicome_videos(ds):
    """DiCoME's own run, aggregated by OUR video rule from its FRAME predictions.

    NOT `test_video_predictions.csv`. That file keys on a bare `video_id` -- `test_00000000` --
    and FFIW ships that same id in both `FFIW-fake` and `FFIW-real`, so its rows merge one fake
    and one real video into a single prediction: 3,470 videos collapse to 1,735, with a label
    that belongs to whichever of the pair won the aggregation. The number such a cell produces is
    wrong, not merely on a different census, and it is the same id-collision DiCoME's own
    adaptation notes document.

    The frame file carries the full path, so the video key and the scope filter are the SAME ones
    every other baseline row uses, via `baseline_videos`'s rule below.
    """
    f = DICOME_RUNS / f"dicome_trained_{ds}_full" / "csv_logs/version_0/test_frame_predictions.csv"
    if not f.exists():
        # `_full` is the re-run on our census; CDFv3's original export covers 1,918 videos of our
        # 5,418 and is kept on disk so the v3.1 numbers stay reproducible.
        f = DICOME_RUNS / f"dicome_trained_{ds}" / "csv_logs/version_0/test_frame_predictions.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    d = scope(d, d.files.str.split("/").str[-3])
    d = d.assign(v=d.files.str.rsplit("/", n=1).str[0])
    g = d.groupby("v").agg(y=("labels", "max"), s=("prob_class_1", "mean"))
    return g.y.to_numpy().astype(int), g.s.to_numpy().astype(float)


def main() -> int:
    sys.path[:0] = [str(C.ROOT / "analysis/tbiom/pcr6p")]
    import ops_pcr6p as P
    from health import eer
    dev = B.dev_table(list(FINAL), require_all=True)
    w, _ = O.fit_weights(dev, FINAL, "F8", objective="v31")
    f = P.fuse(O.masses(dev, FINAL, list(w)), True)
    pdev = P.p_of(f)
    ff = np.where(dev.source.to_numpy() == "FFpp_val")[0]
    tau = float(eer(dev.y.to_numpy()[ff], pdev[ff])[1])

    tabs = B.test_tables(list(FINAL))
    # De-duplicated union, NOT a concatenation. NEW_SETS were promoted into C.TEST_SETS on
    # 2026-10-13 once every CORE branch covered them, so `TEST_SETS + NEW_SETS` printed each of
    # the three twice AND averaged them twice -- the reported mean was over fifteen cells, three
    # of them copies, which pulled every "Ours" mean toward the three easiest corpora.
    # Columns come from the union so a set with no coverage still prints and says so; a table
    # that silently drops a column is how a missing number stops being noticed.
    sets = list(dict.fromkeys(list(C.TEST_SETS) + NEW_SETS))
    avail = [d for d in sets if d in tabs]
    rows = {"**Ours (App+TD+LFS+STT, ES-PCR6+)**": {}}
    for ds in avail:
        t = tabs[ds]
        fo = P.fuse(O.masses(t, FINAL, list(w)), True)
        p, u, y = P.p_of(fo), fo[:, 2], t.y.to_numpy()
        rows["**Ours (App+TD+LFS+STT, ES-PCR6+)**"][ds] = {
            "ece": ece(y, p), "brier": brier(y, p), "eaurc": e_aurc(y, p, u, tau)}
        print(f"  ours {ds}", flush=True)
    rows["DiCoME (trained, their repo)"] = {}
    for ds in avail:
        r = dicome_videos(ds)
        if r is None:
            continue
        y, p = r
        u = 1.0 - np.maximum(p, 1.0 - p)
        rows["DiCoME (trained, their repo)"][ds] = {
            "ece": ece(y, p), "brier": brier(y, p), "eaurc": e_aurc(y, p, u, 0.5)}
    print("  DiCoME", flush=True)
    for m in MODELS:
        name = PRETTY.get(m, m)
        rows[name] = {}
        for ds in avail:
            r = baseline_videos(m, ds)
            if r is None:
                continue
            y, p = r
            # the only uncertainty a cross-entropy detector emits
            u = 1.0 - np.maximum(p, 1.0 - p)
            rows[name][ds] = {"ece": ece(y, p), "brier": brier(y, p),
                              "eaurc": e_aurc(y, p, u, 0.5)}
        print(f"  {name}", flush=True)

    L = ["# Table 2 — probability quality and uncertainty", "",
         "Per dataset. **All three lower is better.** ECE and Brier ask whether a reported",
         "probability means what it says; E-AURC asks whether the model knows which of its own",
         "answers to distrust, which the other two cannot see.", "",
         "E-AURC subtracts the AURC an oracle ranking of the same predictions would reach, so the",
         "base error rate drops out and what remains is the quality of the uncertainty ranking",
         "alone — the only way to compare it across systems of different accuracy.", "",
         "Ours ranks by the fused **vacuity**; the baselines rank by **1 − max p**, which is the",
         "only uncertainty a cross-entropy detector emits. That asymmetry is the point of the",
         "column, not a handicap we imposed.", "",
         f"`{TODO}`: not computable yet, and not zero or not applicable. Our row covers every",
         "set. The baselines were never run on the three corpora packed on 2026-10-13 (UADFV,",
         "DeepSpeak-1.1, DeepSpeak-2.0), and DiCoME's public checkpoints cover only the five",
         "sets its paper reports; closing those cells needs a scoring pass, not a re-run.", "",
         *NOTE.split("\n"), ""]
    for key, title in [("ece", "ECE-15 (lower is better)"), ("brier", "Brier (lower is better)"),
                       ("eaurc", "E-AURC (lower is better)")]:
        all_s, wide_s = C.common_supports(rows, sets, key)
        L += [f"## {title}", "",
              f"`mean (all)`: {len(all_s)} sets ({', '.join(all_s) or 'none'}).  ",
              f"`mean (wide)`: {len(wide_s)} sets ({', '.join(wide_s) or 'none'}).", "",
              "| method | " + " | ".join(sets) + " | mean | mean (all) | mean (wide) |",
              "|---|" + "---:|" * (len(sets) + 3)]
        for name, r in rows.items():
            vals = [r.get(d, {}).get(key) for d in sets]
            got = [v for v in vals if v is not None]
            L.append(f"| {name} | " + " | ".join(TODO if v is None else f"{v:.4f}" for v in vals) +
                     (f" | {np.mean(got):.4f} |" if got else f" | {TODO} |") +
                     C.matched_cells(r, all_s, key, TODO) +
                     C.matched_cells(r, wide_s, key, TODO))
        L.append("")
    OUT.write_text("\n".join(L) + "\n")
    json.dump(rows, open(C.ROOT / "analysis/branchfusion/table2_calibration.json", "w"),
              indent=1, default=str)
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""A1 — sample-level complementarity across the branch suite.

The north star for this phase is *not* the highest average AUC; it is finding
representations whose regions of expertise are complementary to the visual baseline P0.
A1 measures that directly, at video level, per source:

  rescue(op)              P(op right | P0 wrong)   -- what the operator buys us
  harm(op)                P(op wrong | P0 right)   -- what it costs us
  error-overlap(i, j)     agreement between two operators on the samples they get wrong
  evidence correlation    do two branches carry the same signal, regardless of decisions

An operator with a mediocre standalone AUC but a large rescue on the rows P0 inverts is
exactly the specialist this architecture is built to exploit; an operator whose errors
overlap P0's completely adds nothing no matter how good it looks alone.

Threshold discipline
--------------------
rescue/harm are decisions, so they need a threshold. Per the Phase-1 instructions we use
ONE threshold from the permitted protocol source, frozen across every OOD source and
generator -- never tuned per source, which would leak the target distribution and break the
zero-shot claim. `common.frozen_threshold` reports its provenance; when no protocol export
exists it falls back to a fixed 0.5 and says so, and the threshold-free block below
(AUROC, score correlation, error-overlap on ranks) carries the conclusion instead of the
decisions. Both are always written, so no reader has to take the threshold on faith.

Decision-relevant rows
----------------------
The instructions call out the inversion rows (danet/mcnet/tpsm/facevid2vid on the cdf side)
and heygen explicitly: those are where P0 is *confidently wrong*, so they are where a
complementary specialist either proves itself or does not. They get their own breakout
table rather than being averaged away into a source-level mean.

    python analysis/discern_v2/a1_complementarity.py
    python analysis/discern_v2/a1_complementarity.py --source DF40 CDFv3
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def available_pilots(source: str) -> tuple[list[str], list[str]]:
    have = [p for p in C.PILOTS if C.has_export(p, source)]
    missing = [p for p in C.PILOTS if p not in have]
    return have, missing


def video_table(pilot: str, source: str, thr: float) -> pd.DataFrame:
    """Video-level score + decision + correctness for one operator."""
    arrays, meta = C.load(pilot, source)
    v = C.video_level(meta, {"p": arrays["p_fused"], "u": arrays["u_fused"]})
    v["pred"] = (v["p"] >= thr).astype(int)
    v["correct"] = (v["pred"] == v["label"]).astype(int)
    return v.rename(columns={"p": f"p_{pilot}", "u": f"u_{pilot}",
                             "pred": f"pred_{pilot}", "correct": f"ok_{pilot}"})


def merge_operators(pilots: list[str], source: str, thr: float) -> pd.DataFrame:
    """One row per video, one column-group per operator, aligned on video_id."""
    base = None
    for p in pilots:
        v = video_table(p, source, thr)
        keep = ["video_id", "label", "family", "method", f"p_{p}", f"u_{p}",
                f"pred_{p}", f"ok_{p}"]
        v = v[keep]
        if base is None:
            base = v
        else:
            base = base.merge(v.drop(columns=["label", "family", "method"]),
                              on="video_id", how="inner")
    if base is None:
        raise RuntimeError("no operators available")
    return base


def rescue_harm(df: pd.DataFrame, pilots: list[str], subset: pd.Series | None = None
                ) -> pd.DataFrame:
    """rescue = P0 wrong & op right; harm = P0 right & op wrong. Fractions of the subset."""
    d = df if subset is None else df[subset]
    if len(d) == 0:
        return pd.DataFrame()
    p0_ok = d[f"ok_{C.P0}"].astype(bool)
    rows = []
    for p in pilots:
        ok = d[f"ok_{p}"].astype(bool)
        rescue = (~p0_ok & ok).mean()
        harm = (p0_ok & ~ok).mean()
        rows.append({
            "operator": p,
            "n": int(len(d)),
            "acc": float(ok.mean()),
            "p0_acc": float(p0_ok.mean()),
            "rescue": float(rescue),
            "harm": float(harm),
            "net": float(rescue - harm),
            # conditional forms are the honest per-regime numbers: a rescue of 0.02 means
            # something very different when P0 is wrong on 4% vs 60% of the subset.
            "rescue_rate_given_p0_wrong": float(ok[~p0_ok].mean()) if (~p0_ok).any() else np.nan,
            "harm_rate_given_p0_right": float((~ok[p0_ok]).mean()) if p0_ok.any() else np.nan,
        })
    return pd.DataFrame(rows)


def error_overlap(df: pd.DataFrame, pilots: list[str]) -> pd.DataFrame:
    """Agreement on the wrong samples: |E_i & E_j| / |E_i | E_j| (Jaccard over error sets).

    Jaccard rather than a raw count so operators with very different error volumes stay
    comparable. 1.0 = identical failure modes (no complementarity available); 0.0 = disjoint
    failures (maximum routing headroom).
    """
    rows = []
    for a, b in combinations(pilots, 2):
        ea = ~df[f"ok_{a}"].astype(bool)
        eb = ~df[f"ok_{b}"].astype(bool)
        union = (ea | eb).sum()
        rows.append({"op_i": a, "op_j": b,
                     "n_err_i": int(ea.sum()), "n_err_j": int(eb.sum()),
                     "n_err_both": int((ea & eb).sum()),
                     "jaccard": float((ea & eb).sum() / union) if union else np.nan,
                     # P(both wrong | either wrong) is the shared-failure floor a router
                     # cannot escape.
                     "shared_given_either": float((ea & eb).sum() / union) if union else np.nan})
    return pd.DataFrame(rows)


def evidence_correlation(df: pd.DataFrame, pilots: list[str]) -> pd.DataFrame:
    """Spearman correlation between operator scores -- threshold-free.

    Two branches can disagree on decisions yet carry the same underlying ranking; this is
    the check that a 'complementary' operator is not just P0 with a different offset.
    """
    from scipy.stats import spearmanr

    rows = []
    for a, b in combinations(pilots, 2):
        rho, _ = spearmanr(df[f"p_{a}"], df[f"p_{b}"])
        rows.append({"op_i": a, "op_j": b, "spearman": float(rho)})
    return pd.DataFrame(rows)


def threshold_free(df: pd.DataFrame, pilots: list[str]) -> pd.DataFrame:
    """Per-operator AUROC at video level -- no threshold involved.

    Reported alongside every rescue/harm table so a conclusion never rests on a threshold
    whose provenance is a fallback.
    """
    from sklearn.metrics import roc_auc_score

    rows = []
    for p in pilots:
        y = df["label"].values
        if len(np.unique(y)) < 2:
            rows.append({"operator": p, "auroc": np.nan, "n": int(len(df))})
            continue
        rows.append({"operator": p, "auroc": float(roc_auc_score(y, df[f"p_{p}"])),
                     "n": int(len(df))})
    return pd.DataFrame(rows)


def per_generator(df: pd.DataFrame, pilots: list[str]) -> pd.DataFrame:
    """Per-generator rescue/harm computed on that generator's FAKES ONLY.

    Deliberately *not* pooled with the shared real set. A DF40 generator row holds ~18
    videos against ~830 reals, so folding the reals in makes every row's rescue/harm a
    restatement of behaviour on the same real pool -- the rows come out near-identical and
    the generator, the thing we are trying to discriminate, contributes almost nothing.
    Fakes-only keeps `acc` interpretable as the detection rate on that generator, which is
    what "P0 is inverted on danet/mcnet/tpsm" actually refers to.

    Behaviour on reals is a single shared property of the operator, so it is reported once
    by `reals_summary()` rather than repeated per row.
    """
    out = []
    for method, sub in df.groupby("method"):
        if method == "real":
            continue
        rows = rescue_harm(df, pilots, subset=(df["method"] == method))
        if rows.empty:
            continue
        rows.insert(0, "method", method)
        rows["n_fake"] = int(len(sub))
        out.append(rows)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def per_generator_auroc(df: pd.DataFrame, pilots: list[str]) -> pd.DataFrame:
    """AUROC per generator, scored against the shared real pool -- threshold-free.

    This is the form the inversion rows were originally identified in: AUROC well below 0.5
    on a generator means the operator ranks those fakes as *more real* than genuine video,
    i.e. it is confidently wrong rather than merely uncertain. Needs the real pool, because
    AUROC is undefined on a single-class subset.
    """
    from sklearn.metrics import roc_auc_score

    reals = df[df["method"] == "real"]
    out = []
    for method, sub in df.groupby("method"):
        if method == "real":
            continue
        d = pd.concat([sub, reals], ignore_index=True)
        y = d["label"].values
        if len(np.unique(y)) < 2:
            continue
        row = {"method": method, "n_fake": int(len(sub)), "n_real": int(len(reals))}
        for p in pilots:
            row[f"auroc_{p}"] = float(roc_auc_score(y, d[f"p_{p}"]))
        out.append(row)
    return pd.DataFrame(out)


def reals_summary(df: pd.DataFrame, pilots: list[str]) -> pd.DataFrame:
    """Operator behaviour on the real pool: specificity and the rescue/harm it contributes.

    Reported once. Real-side behaviour is what a naive per-generator table double-counts.
    """
    reals = df[df["method"] == "real"]
    if reals.empty:
        return pd.DataFrame()
    rows = rescue_harm(reals, pilots)
    rows.insert(0, "subset", "real")
    rows = rows.rename(columns={"acc": "specificity"})
    return rows


def heatmap(mat: pd.DataFrame, title: str, out: Path, fmt: str = "{:.3f}",
            cmap: str = "RdYlBu_r") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1.1 * len(mat.columns) + 2.4, .52 * len(mat) + 1.9))
    data = mat.values.astype(float)
    im = ax.imshow(data, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(mat.columns)), mat.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(mat)), mat.index, fontsize=8)
    finite = data[np.isfinite(data)]
    mid = (finite.max() + finite.min()) / 2 if finite.size else 0
    for i in range(len(mat)):
        for j in range(len(mat.columns)):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7,
                        color="white" if v > mid else "black")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=.030, pad=.02)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", nargs="+", default=["DF40", "CDFv3"])
    args = ap.parse_args()

    out = C.out_dir("A1_complementarity")
    thr, prov = C.frozen_threshold(C.P0)
    summary: dict = {"threshold": thr, "threshold_provenance": prov, "sources": {}}

    print(f"frozen threshold = {thr:.4f}  ({prov})\n")

    for source in args.source:
        have, missing = available_pilots(source)
        if C.P0 not in have:
            print(f"[{source}] skipped — no {C.P0} export (every delta is measured "
                  f"against it):\n    cd {C.DICOME} && {C.export_command(C.P0, [source])}")
            continue
        if missing:
            print(f"[{source}] missing exports (UMAR-RUNS) — omitted from the matrices, "
                  f"NOT silently ignored:")
            for p in missing:
                print(f"    cd {C.DICOME} && {C.export_command(p, [source])}")

        df = merge_operators(have, source, thr)
        sdir = C.out_dir(f"A1_complementarity/{source}")

        overall = rescue_harm(df, have)
        overall.to_csv(sdir / "rescue_harm_overall.csv", index=False)

        tf = threshold_free(df, have)
        tf.to_csv(sdir / "threshold_free_auroc.csv", index=False)

        eo = error_overlap(df, have)
        eo.to_csv(sdir / "error_overlap.csv", index=False)

        ec = evidence_correlation(df, have)
        ec.to_csv(sdir / "evidence_correlation.csv", index=False)

        pg = per_generator(df, have)
        if not pg.empty:
            pg.to_csv(sdir / "rescue_harm_per_generator.csv", index=False)

        pga = per_generator_auroc(df, have)
        if not pga.empty:
            pga.to_csv(sdir / "auroc_per_generator.csv", index=False)
            # inversion rows are those where the visual baseline ranks fakes as more real
            # than genuine video -- identified from the data, not hard-coded from memory.
            inv = pga[pga[f"auroc_{C.P0}"] < 0.5].sort_values(f"auroc_{C.P0}")
            inv.to_csv(sdir / "inversion_rows_detected.csv", index=False)

        rs = reals_summary(df, have)
        if not rs.empty:
            rs.to_csv(sdir / "reals_summary.csv", index=False)

        # --- decision-relevant breakout ------------------------------------------------
        rows_of_interest = [r for r in C.INVERSION_ROWS + [C.HEYGEN_ROW]
                            if (df["method"] == r).any()]
        absent = [r for r in C.INVERSION_ROWS + [C.HEYGEN_ROW]
                  if not (df["method"] == r).any()]
        if rows_of_interest:
            brk = pg[pg["method"].isin(rows_of_interest)] if not pg.empty else pd.DataFrame()
            if not brk.empty:
                brk.to_csv(sdir / "breakout_inversion_and_heygen.csv", index=False)
                piv = brk.pivot_table(index="method", columns="operator", values="rescue")
                heatmap(piv, f"A1 [{source}] rescue on inversion / heygen rows",
                        sdir / "breakout_rescue.png")
                pivh = brk.pivot_table(index="method", columns="operator", values="harm")
                heatmap(pivh, f"A1 [{source}] harm on inversion / heygen rows",
                        sdir / "breakout_harm.png")

        # --- matrices as plots ----------------------------------------------------------
        if not eo.empty:
            m = pd.DataFrame(index=have, columns=have, dtype=float)
            np.fill_diagonal(m.values, 1.0)
            for _, r in eo.iterrows():
                m.loc[r["op_i"], r["op_j"]] = r["jaccard"]
                m.loc[r["op_j"], r["op_i"]] = r["jaccard"]
            heatmap(m, f"A1 [{source}] error overlap (Jaccard over error sets)",
                    sdir / "error_overlap.png")
        if not ec.empty:
            m = pd.DataFrame(index=have, columns=have, dtype=float)
            np.fill_diagonal(m.values, 1.0)
            for _, r in ec.iterrows():
                m.loc[r["op_i"], r["op_j"]] = r["spearman"]
                m.loc[r["op_j"], r["op_i"]] = r["spearman"]
            heatmap(m, f"A1 [{source}] evidence correlation (Spearman)",
                    sdir / "evidence_correlation.png")

        summary["sources"][source] = {
            "n_videos": int(len(df)),
            "operators_present": have,
            "operators_missing": missing,
            "breakout_rows_present": rows_of_interest,
            "breakout_rows_absent": absent,
            "overall": overall.to_dict("records"),
            "threshold_free_auroc": tf.to_dict("records"),
            "reals_summary": rs.to_dict("records") if not rs.empty else [],
            "inversion_rows_detected": (
                pga.loc[pga[f"auroc_{C.P0}"] < 0.5, "method"].tolist()
                if not pga.empty else []),
        }
        print(f"[{source}] {len(df)} videos, operators {have} -> {sdir}")

    (out / "A1_results.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {out}/A1_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""The health dashboard — AUROC is never reported alone again.

    python analysis/tbiom/health.py \
        --scores NAME PARQUET PROB_COL [NAME PARQUET PROB_COL ...] \
        --threshold-source logs/tbiom/score/clip_ffppval_seeded/per_sample_epoch_7.parquet p_sem \
        --out tbiom/SOMETHING.md

WHY THIS EXISTS. The FF++ (+) DF40 run collapsed operationally while AUROC barely moved. On
Celeb-DF-v2 it emitted 0.997 for real and fake alike — zero probability separation, every
unfamiliar real called fake with 99.7% confidence — and still scored 0.75 AUROC, because ranking
survives on noise inside a saturated region. A table of AUROC would have called that a moderate
regression. The operational metrics called it what it was.

The five columns, and what each one catches:

    AUROC       ranking quality. Necessary, and by itself misleading under saturation.
    EER         threshold-independent operating quality. Moves when ranking degrades in the
                region a deployed system would actually threshold on.
    FPR_real@t  THE COLLAPSE HEADLINE. Fraction of REAL videos called fake at a threshold frozen
                once on development data. This is the number that spikes when a model stops
                recognising unfamiliar reals.
    d_RF        probability separation, E[p_F | fake] - E[p_F | real]. Goes to 0.000 under
                saturation while AUROC stays comfortably above chance.
    mean p_R    mean p(fake) on REAL videos, the calibration diagnostic. In the collapse it read
                0.205 on the training corpus's reals and 0.575-0.997 everywhere else.

THE THRESHOLD IS FROZEN ONCE, on the permitted development source, and applied unchanged to every
domain. A per-domain threshold would re-fit the operating point to each test set and hide exactly
the failure this dashboard exists to expose.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))


# The video-identity rule lives in ONE place. A wrong grouping does not raise — it silently
# regroups frames and changes every AUROC below it, which has already reversed two conclusions in
# this project. See analysis/tbiom/video_id.py and its regression check.
from video_id import to_video_level, video_id as consistent_video_id  # noqa: E402


def video_level(df: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    return to_video_level(df, prob_col)


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def eer(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Equal error rate and the threshold that achieves it."""
    from sklearn.metrics import roc_curve
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    fpr, tpr, thr = roc_curve(y, p)
    i = int(np.nanargmin(np.abs((1 - tpr) - fpr)))
    return float((fpr[i] + (1 - tpr[i])) / 2), float(thr[i])


def dashboard(y: np.ndarray, p: np.ndarray, tau: float) -> dict:
    real, fake = p[y == 0], p[y == 1]
    e, _ = eer(y, p)
    return {
        "n_videos": int(len(y)), "n_real": int(len(real)), "n_fake": int(len(fake)),
        "auroc": auroc(y, p),
        "eer": e,
        # the collapse headline: reals called fake at the frozen operating point
        "fpr_real_at_tau": float((real >= tau).mean()) if len(real) else float("nan"),
        "tpr_fake_at_tau": float((fake >= tau).mean()) if len(fake) else float("nan"),
        "delta_rf": float(fake.mean() - real.mean()) if len(real) and len(fake) else float("nan"),
        "mean_p_on_real": float(real.mean()) if len(real) else float("nan"),
        "mean_p_on_fake": float(fake.mean()) if len(fake) else float("nan"),
    }


def load(parquet: Path, prob_col: str) -> pd.DataFrame:
    paths = sorted(parquet.glob("*.parquet")) if parquet.is_dir() else [parquet]
    if not paths:
        raise SystemExit(f"no parquet under {parquet}")
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    if prob_col not in df:
        raise SystemExit(f"{parquet} has no `{prob_col}`; columns: {sorted(df.columns)[:12]}")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scores", nargs=3, action="append", required=True,
                    metavar=("NAME", "PARQUET_OR_DIR", "PROB_COL"))
    ap.add_argument("--threshold-source", nargs=2, required=True,
                    metavar=("PARQUET_OR_DIR", "PROB_COL"),
                    help="the PERMITTED development source (FF++ val). Its EER fixes tau, which "
                         "is then frozen across every row. Required: a per-domain threshold "
                         "would re-fit the operating point to each test set and hide the very "
                         "collapse this dashboard exists to expose.")
    ap.add_argument("--title", default="Health dashboard")
    ap.add_argument("--note", default=None, help="one line of context for the report header")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tsrc, tcol = args.threshold_source
    tdf = load(Path(tsrc), tcol)
    tv = video_level(tdf, tcol)
    _, tau = eer(tv["y"].to_numpy(), tv["p"].to_numpy())
    print(f"frozen threshold tau = {tau:.4f} (EER on {tsrc})")

    rows = {}
    for name, path, col in args.scores:
        v = video_level(load(Path(path), col), col)
        rows[name] = {**dashboard(v["y"].to_numpy(), v["p"].to_numpy(), tau),
                      "parquet": str(path), "prob_col": col}
        r = rows[name]
        print(f"  {name:28s} AUROC {r['auroc']:.4f}  EER {r['eer']:.4f}  "
              f"FPR_real@t {r['fpr_real_at_tau']:.3f}  d_RF {r['delta_rf']:+.3f}  "
              f"meanP_real {r['mean_p_on_real']:.3f}")

    lines = [f"# {args.title}", ""]
    if args.note:
        lines += [args.note, ""]
    lines += [
        f"Operating threshold **tau = {tau:.4f}**, the EER on `{tsrc}` (`{tcol}`), frozen across "
        f"every row. A per-domain threshold would re-fit the operating point to each test set and "
        f"hide the collapse this table exists to expose.", "",
        "`FPR_real@t` is the headline: the fraction of REAL videos called fake at that frozen "
        "point. `d_RF` is probability separation; it reads 0.000 under saturation while AUROC can "
        "still look moderate.", "",
        "| model / domain | videos | real/fake | AUROC | EER | **FPR_real@t** | d_RF | mean p on real |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for name, r in rows.items():
        lines.append(
            f"| {name} | {r['n_videos']} | {r['n_real']}/{r['n_fake']} | {r['auroc']:.4f} | "
            f"{r['eer']:.4f} | **{r['fpr_real_at_tau']:.3f}** | {r['delta_rf']:+.3f} | "
            f"{r['mean_p_on_real']:.3f} |")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(
        {"tau": tau, "threshold_source": tsrc, "rows": rows}, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

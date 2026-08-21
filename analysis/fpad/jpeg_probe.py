#!/usr/bin/env python3
"""Stage 6 — the JPEG compression probe.

    🔴 UMAR-RUNS (CPU, seconds, after scoring each quality):

    python analysis/fpad/jpeg_probe.py --readout traj \
        --profile 100 logs/fpad/score/B3_ffppval/profile_epoch_009.parquet \
        --profile  85 logs/fpad/score/B3_ffppval_q85/profile_epoch_009.parquet \
        --profile  50 logs/fpad/score/B3_ffppval_q50/profile_epoch_009.parquet \
        --profile  20 logs/fpad/score/B3_ffppval_q20/profile_epoch_009.parquet \
        --out wacv

Why this exists, and what it is NOT
-----------------------------------
The brief's compression-robustness check wants H.264 c40, which is not staged on this machine and
was left as `TODO(run)`. This is the substitute that needs no download: re-encode the frames at a
descending JPEG quality ladder and measure what happens.

It must be labelled precisely. Our frames are PNG crops **already derived from H.264 c23 video**,
so this applies a SECOND, different codec on top. It is a JPEG probe, not a c40 measurement, and
the two rows in the paper are not interchangeable.

The question it actually answers
--------------------------------
"Is your signal just reading compression?" has two readings, and only one of them is about
robustness:

1. **Robustness** — does AUROC survive compression? That is the degradation curve.
2. **Confounding** — does the delta respond to compression the way it responds to manipulation?
   That is the sharper question, and AUROC degradation does not answer it: a signal can degrade
   gracefully while still being a compression detector.

So the headline is a **compression audit** in the same currency as the §20 domain audit, with
compression substituted for corpus as the nuisance axis:

    compression separability   D(uncompressed real, compressed real)     pure compression
    forensic separability      D(compressed real, compressed fake)       compression cancels
    gap                        forensic - compression

`audit_quantity` is imported from `discern_v2/domain_audit.py` verbatim, so the verdicts here mean
exactly what they mean in the domain audit and the two tables can sit side by side. A negative gap
says the delta reads compression better than it reads manipulation — which is the finding the
reviewer is asking about, and the degradation curve alone would not have shown it.
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

from domain_audit import audit_quantity, separability  # noqa: E402  (reused verbatim)

READOUTS = {"direct": "p_direct", "traj": "p_traj"}
UNCOMPRESSED = 100          # the label used for the no-JPEG baseline
NOISE = 0.01


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


def align_on_key(base: pd.DataFrame, other: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict both tables to the frames they share, in the same order.

    The compression audit compares the SAME frame with and without JPEG. Comparing whichever rows
    each run happened to score would mix a compression effect with a sampling difference, and with
    `--max-batches` in play the two runs need not cover the same frames at all.
    """
    b = base.set_index("key")
    o = other.set_index("key")
    shared = b.index.intersection(o.index)
    fraction = len(shared) / max(1, min(len(b), len(o)))
    if len(shared) < 200 or fraction < 0.5:
        raise SystemExit(
            f"only {len(shared)} frames are common to the baseline ({len(b)}) and the compressed "
            f"run ({len(o)}) — {fraction:.0%} of the smaller set. The audit compares the SAME "
            f"frame with and without JPEG, so a partial overlap measures sampling as well as "
            f"compression. The usual cause is an unseeded scoring run: `abstract_dataset` "
            f"shuffles with the global RNG, so pass the same --seed to every score_fpad.py call.")
    return b.loc[shared].reset_index(), o.loc[shared].reset_index()


def compression_audit(base: pd.DataFrame, comp: pd.DataFrame, readout: str) -> dict:
    """§20's three-way test with compression as the nuisance axis instead of corpus."""
    b, c = align_on_key(base, comp)
    column = READOUTS[readout]
    quantities = {}
    names = layer_columns(b) + ["d_mean"] + ([column] if column in b.columns else [])
    for name in names:
        if name not in b.columns or name not in c.columns:
            continue
        real = b["label"] == 0
        quantities[name] = audit_quantity(
            b.loc[real, name].to_numpy(),         # uncompressed real
            c.loc[real, name].to_numpy(),         # compressed real   -> pure compression
            c.loc[c["label"] == 1, name].to_numpy())  # compressed fake -> compression cancels
        quantities[name]["forensic_uncompressed"] = separability(
            b.loc[real, name].to_numpy(), b.loc[b["label"] == 1, name].to_numpy())
    return {"n_shared_frames": int(len(b)), "quantities": quantities}


def render(payload: dict) -> str:
    curve, audit = payload["curve"], payload["audit"]
    qualities = payload["qualities"]
    base_q = payload["baseline_quality"]

    lines = [
        "# Stage 6 — JPEG compression probe",
        "",
        f"Rung readout `{payload['readout']}` · epoch {payload.get('epoch', '?')} · "
        f"qualities {qualities}.",
        "",
        "> ⚠️ **This is a JPEG probe, not an H.264 c40 measurement.** Our frames are PNG crops "
        "already derived from c23 video, so this applies a second, different codec on top. The "
        "c40 row remains `TODO(run)` pending the download; the two are not interchangeable and "
        "must not be merged in the paper.",
        "",
        "## Robustness — does AUROC survive compression?",
        "",
        "| source | " + " | ".join(f"q{q}" for q in qualities) + " | drop (best→worst) |",
        "|---" * (len(qualities) + 2) + "|",
    ]
    for source, per_q in curve.items():
        cells = []
        for q in qualities:
            v = per_q.get(str(q))
            cells.append("—" if v is None or not np.isfinite(v) else f"{v:.4f}")
        vals = [per_q.get(str(q)) for q in qualities
                if per_q.get(str(q)) is not None and np.isfinite(per_q.get(str(q)))]
        drop = f"{(vals[0] - vals[-1]):+.4f}" if len(vals) >= 2 else "—"
        lines.append(f"| {source} | " + " | ".join(cells) + f" | {drop} |")

    lines += ["", "## Confounding — does the delta READ compression?", "",
              "The sharper question, in the same currency as the §20 domain audit: compression "
              "separability is `D(uncompressed real, compressed real)`, forensic separability is "
              "`D(compressed real, compressed fake)`, and the gap is forensic − compression. A "
              "negative gap means the quantity tells compression apart better than it tells "
              "manipulation apart — which AUROC degradation alone would not reveal.", ""]
    for q, entry in audit.items():
        lines += [f"### at quality {q} ({entry['n_shared_frames']} frames shared with the "
                  f"baseline)", "",
                  "| quantity | compression sep | forensic sep | gap | verdict | "
                  "forensic uncompressed |", "|---|---:|---:|---:|---|---:|"]
        for name, e in entry["quantities"].items():
            lines.append(
                f"| `{name}` | {e['domain_separability']:.4f} | "
                f"{e['forensic_separability']:.4f} | {e['forensic_minus_domain']:+.4f} | "
                f"{e['verdict']} | {e['forensic_uncompressed']:.4f} |")
        lines.append("")
        bad = [n for n, e in entry["quantities"].items()
               if e["verdict"] in ("DATASET DETECTOR", "BORDERLINE")]
        if bad:
            lines.append(f"⚠️ {len(bad)} of {len(entry['quantities'])} quantities read "
                         f"compression at least as well as manipulation at this quality: "
                         f"{', '.join(f'`{n}`' for n in bad)}.")
        else:
            lines.append("All quantities separate manipulation better than they separate "
                         "compression at this quality.")
        lines.append("")

    if payload.get("figure"):
        lines += [f"Figure: `{payload['figure']}`", ""]
    lines += ["## Bookkeeping", "",
              f"* The baseline column is quality {base_q} = no JPEG applied.",
              "* The audit compares the SAME frames with and without JPEG, intersected on `key`; "
              "a compressed run that covered different frames would confound compression with "
              "sampling.",
              f"* Differences under ±{NOISE} need a confirming seed before they support a claim.",
              ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profile", nargs=2, action="append", required=True,
                    metavar=("QUALITY", "PARQUET"),
                    help=f"repeat per quality; use {UNCOMPRESSED} for the uncompressed baseline")
    ap.add_argument("--readout", choices=sorted(READOUTS), required=True)
    ap.add_argument("--audit-at", type=int, nargs="+", default=None,
                    help="qualities to run the compression audit at (default: the lowest)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tables, epoch = {}, None
    for quality, path in args.profile:
        q = int(quality)
        df = pd.read_parquet(path)
        meta_path = Path(path).parent / "score_meta.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            epoch = meta.get("epoch", epoch)
            recorded = meta.get("jpeg_quality")
            expected = None if q == UNCOMPRESSED else q
            if recorded != expected:
                raise SystemExit(
                    f"{path} records jpeg_quality={recorded} but was passed as quality {q}. "
                    f"Mislabelling a quality would put the degradation curve in the wrong order "
                    f"and invert its conclusion.")
        tables[q] = df
        print(f"  q{q:3d}: {len(df)} frames from {path}")

    if UNCOMPRESSED not in tables:
        raise SystemExit(
            f"no uncompressed baseline: pass one profile as quality {UNCOMPRESSED}. Degradation "
            f"has to be measured against something.")
    qualities = sorted(tables, reverse=True)
    column = READOUTS[args.readout]

    curve: dict[str, dict[str, float]] = {}
    for q in qualities:
        df = tables[q]
        if column not in df.columns:
            raise SystemExit(
                f"q{q} profile has no `{column}` column for readout `{args.readout}`; re-score "
                f"with the matching readout rather than substituting the other one.")
        for source, g in df.groupby("dataset"):
            curve.setdefault(str(source), {})[str(q)] = video_auroc(
                g[column].to_numpy(), g["label"].to_numpy(), g["video_id"].to_numpy())
    for source, per_q in curve.items():
        vals = [per_q[str(q)] for q in qualities if np.isfinite(per_q.get(str(q), np.nan))]
        print(f"  {source:22s} " + " ".join(f"q{q}={per_q[str(q)]:.4f}" for q in qualities
                                            if str(q) in per_q)
              + (f"  drop {vals[0] - vals[-1]:+.4f}" if len(vals) >= 2 else ""))

    audit_at = args.audit_at or [min(q for q in qualities if q != UNCOMPRESSED)]
    audit = {}
    for q in audit_at:
        if q not in tables:
            raise SystemExit(f"no profile at quality {q} to audit")
        audit[str(q)] = compression_audit(tables[UNCOMPRESSED], tables[q], args.readout)
        print(f"\n  compression audit at q{q} "
              f"({audit[str(q)]['n_shared_frames']} shared frames):")
        for name, e in audit[str(q)]["quantities"].items():
            print(f"    {name:10s} compression {e['domain_separability']:.4f} · forensic "
                  f"{e['forensic_separability']:.4f} · gap {e['forensic_minus_domain']:+.4f} · "
                  f"{e['verdict']}")

    figure = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        args.out.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4.4))
        xs = [q for q in qualities]
        for source, per_q in curve.items():
            ys = [per_q.get(str(q)) for q in xs]
            ax.plot(range(len(xs)), ys, "-o", label=source)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([("none" if q == UNCOMPRESSED else f"q{q}") for q in xs])
        ax.set_xlabel("JPEG quality applied on top of c23")
        ax.set_ylabel("video AUROC")
        ax.set_title(f"JPEG compression probe — readout `{args.readout}`")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        dest = args.out / f"jpeg_probe_{args.readout}.png"
        fig.savefig(dest, dpi=150)
        plt.close(fig)
        figure = str(dest)
    except ImportError:
        pass

    payload = {"readout": args.readout, "epoch": epoch, "qualities": qualities,
               "baseline_quality": UNCOMPRESSED, "curve": curve, "audit": audit,
               "figure": figure,
               "note": "JPEG applied on top of c23-derived PNG crops; NOT H.264 c40"}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "stage6_jpeg_probe.json").write_text(json.dumps(payload, indent=2, default=str))
    (args.out / "STAGE6_JPEG_PROBE.md").write_text(render(payload))
    print(f"\nwrote {args.out}/STAGE6_JPEG_PROBE.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

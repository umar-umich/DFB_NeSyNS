#!/usr/bin/env python3
"""Stage 2.3 — does R(x) carry family-specific STRUCTURE, or is it a monotone magnitude?

    🔴 UMAR-RUNS (CPU, ~5 min):

    python analysis/discern_v2/phase2/rate_response_analysis.py \
        --parquet logs/v1/eval/<run>_df40dev/per_sample_epoch_N.parquet \
        --out phase2/stage2

The brief flags this analysis as one that "was flagged twice before and, per the handoff, was
never run", and makes P1d's place in the architecture conditional on it: the branch enters only
if the response shows conditional structure, not on faith.

The question, made falsifiable
-------------------------------
`R(x) = [r_beta_1, ..., r_beta_K]` is decomposed into two orthogonal parts:

    level  = mean_k r_k                      how compressible this sample is overall
    shape  = r - level                       how that varies ACROSS rates (K-1 free dimensions)

If R(x) is "a monotone magnitude", `level` carries everything and `shape` is noise. So every probe
is run three times — on `level` alone, on `shape` alone, and on both — and what matters is the
**increment of shape over level**. A probe on the full vector that beats chance proves nothing on
its own: it may only be re-reading the level.

Discipline, ported from A3 rather than re-derived
--------------------------------------------------
* **Leave-one-generator-out.** Training and testing a probe on the same DF40 generators measures
  memorisation of generator fingerprints, not transferable structure. Supervised probes are
  scored on held-out generators only.
* **Per-fold scalers.** Standardisation is fit inside each fold. Fitting it globally before
  splitting leaks the held-out generator into the normalisation.
* The embedding (t-SNE) is descriptive and uses everything, and is labelled as such.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lib  # noqa: E402

CHANCE = 0.5


def response_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols = sorted((c for c in df.columns if c.startswith("rate_r_")),
                  key=lambda c: int(c.rsplit("_", 1)[1]))
    if len(cols) < 2:
        raise SystemExit(
            "the export has fewer than two `rate_r_*` columns, so there is no rate RESPONSE to "
            "analyse — only a single distortion. Re-run eval_v1.py with the rate branch enabled.")
    return df[cols].to_numpy(dtype=np.float64), cols


def decompose(R: np.ndarray) -> dict[str, np.ndarray]:
    """Split the response into overall level and the across-rate shape."""
    level = R.mean(axis=1, keepdims=True)
    return {"level": level, "shape": R - level, "full": R}


def probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    """Leave-one-group-out AUROC of a logistic probe, scaler fit inside each fold."""
    from sklearn.metrics import roc_auc_score

    oof = np.full(len(y), np.nan)
    for g in np.unique(groups):
        held = groups == g
        if len(np.unique(y[~held])) < 2:
            continue
        oof[held] = lib.fit_logistic(X[~held], y[~held], X[held])
    scored = ~np.isnan(oof)
    if scored.sum() < 10 or len(np.unique(y[scored])) < 2:
        return {"auroc": float("nan"), "n_scored": int(scored.sum())}
    return {"auroc": float(roc_auc_score(y[scored], oof[scored])),
            "n_scored": int(scored.sum())}


def family_separability(R: np.ndarray, families: np.ndarray) -> dict:
    """Fisher ratio and silhouette over the response vectors — descriptive, not a probe.

    Reported on `full` and on `shape` separately: a high Fisher ratio on `full` that vanishes on
    `shape` means the families differ in overall compressibility, which is a magnitude finding,
    not the family-specific structure the brief is asking about.
    """
    from sklearn.metrics import silhouette_score

    out = {}
    for name, X in (("full", R), ("shape", decompose(R)["shape"])):
        labels, counts = np.unique(families, return_counts=True)
        keep = np.isin(families, labels[counts >= 10])
        if keep.sum() < 20 or len(np.unique(families[keep])) < 2:
            out[name] = {"silhouette": float("nan"), "fisher_ratio": float("nan")}
            continue
        Xk, fk = X[keep], families[keep]
        grand = Xk.mean(axis=0)
        between = sum((fk == f).sum() * np.sum((Xk[fk == f].mean(axis=0) - grand) ** 2)
                      for f in np.unique(fk))
        within = sum(np.sum((Xk[fk == f] - Xk[fk == f].mean(axis=0)) ** 2)
                     for f in np.unique(fk))
        out[name] = {
            "silhouette": float(silhouette_score(Xk, fk)) if len(Xk) < 20000 else
            float(silhouette_score(Xk[:20000], fk[:20000])),
            "fisher_ratio": float(between / within) if within > 0 else float("inf"),
            "n_families": int(len(np.unique(fk))),
        }
    return out


def embed_plot(R: np.ndarray, df: pd.DataFrame, out: Path, seed: int = 42,
               max_points: int = 6000) -> str | None:
    """t-SNE of the response vectors, coloured by family and by label. Descriptive."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError as exc:
        print(f"  skipping the embedding: {exc}")
        return None

    rng = np.random.default_rng(seed)
    idx = (rng.choice(len(R), size=max_points, replace=False)
           if len(R) > max_points else np.arange(len(R)))
    emb = TSNE(n_components=2, init="pca", perplexity=30, random_state=seed).fit_transform(R[idx])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    families = df["dataset"].to_numpy()[idx]
    for family in sorted(set(families)):
        m = families == family
        axes[0].scatter(emb[m, 0], emb[m, 1], s=3, alpha=0.5, label=family)
    axes[0].set_title("R(x) by method")
    if len(set(families)) <= 20:
        axes[0].legend(markerscale=3, fontsize=6, loc="best")
    labels = df["label"].to_numpy()[idx]
    for value, name in ((0, "real"), (1, "fake")):
        m = labels == value
        axes[1].scatter(emb[m, 0], emb[m, 1], s=3, alpha=0.5, label=name)
    axes[1].set_title("R(x) by label")
    axes[1].legend(markerscale=3)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("t-SNE of the multi-rate response (descriptive; all generators)")
    fig.tight_layout()
    dest = out / "rate_response_tsne.png"
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    return str(dest)


def component_plot(R: np.ndarray, df: pd.DataFrame, cols: list[str], out: Path) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    methods = sorted(set(df["dataset"]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(cols))
    for method in methods[:15]:
        m = (df["dataset"] == method).to_numpy()
        axes[0].plot(x, R[m].mean(axis=0), marker="o", label=method, alpha=0.8)
        axes[1].plot(x, decompose(R[m])["shape"].mean(axis=0), marker="o", alpha=0.8)
    axes[0].set_title("mean R(x) per method (level + shape)")
    axes[1].set_title("mean SHAPE per method (level removed)\nflat lines here = monotone magnitude")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("rate_r_", "beta ") for c in cols])
        ax.set_xlabel("rate index")
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    dest = out / "rate_response_components.png"
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    return str(dest)


def render(results: dict, meta: dict) -> str:
    q = results["probes"]
    sep = results["family_separability"]
    inc_forgery = q["forgery"]["shape_increment"]
    inc_error = q["anchor_error"]["shape_increment"]

    lines = [
        "# Stage 2.3 — rate-response structure",
        "",
        f"`{meta['parquet']}` — {meta['n']} frames, {meta['n_methods']} methods, "
        f"K = {meta['k']} rates.",
        "",
        "The question is whether `R(x)` carries family-specific **shape** or is a monotone "
        "**magnitude**. Every probe is run on the level alone, the shape alone, and both, and "
        "the number that answers the question is the *increment of shape over level* — a probe "
        "on the full vector may simply be re-reading the level.",
        "",
        "## Probes (leave-one-generator-out, per-fold scalers)", "",
        "| question | level only | shape only | full | shape increment |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in (("forgery", "real vs fake"),
                       ("anchor_error", "does the anchor get this wrong?")):
        r = q[key]
        lines.append(f"| {label} | {r['level']['auroc']:.4f} | {r['shape']['auroc']:.4f} | "
                     f"{r['full']['auroc']:.4f} | **{r['shape_increment']:+.4f}** |")

    lines += ["", "## Family structure (descriptive, all generators)", "",
              "| quantity | silhouette | Fisher ratio | families |", "|---|---:|---:|---:|"]
    for name in ("full", "shape"):
        s = sep[name]
        lines.append(f"| `{name}` | {s['silhouette']:.4f} | {s['fisher_ratio']:.4f} | "
                     f"{s.get('n_families', 0)} |")
    lines += ["", "A high Fisher ratio on `full` that collapses on `shape` means the families "
                  "differ in overall compressibility — a magnitude finding, not the structure "
                  "the brief asks about.", ""]

    lines += ["## Read-off", ""]
    verdicts = []
    if inc_forgery > lib.NOISE_FLOOR:
        verdicts.append(f"shape adds {inc_forgery:+.4f} AUROC to real/fake over level alone")
    if inc_error > lib.NOISE_FLOOR:
        verdicts.append(f"shape adds {inc_error:+.4f} AUROC to predicting anchor errors")
    if sep["shape"]["fisher_ratio"] > sep["full"]["fisher_ratio"] * 0.25:
        verdicts.append("families remain separable after the level is removed")
    if verdicts:
        lines.append("**R(x) carries conditional structure.** " + "; ".join(verdicts) + ".")
        lines.append("")
        lines.append("On this evidence P1d has a basis for entering the architecture — subject "
                     "to Stage 3, which asks the separate question of whether a realizable gate "
                     "can exploit it.")
    else:
        lines.append("**R(x) behaves as a monotone magnitude on this data.** Removing the level "
                     "leaves no probe above the noise band and no family separability worth the "
                     "name.")
        lines.append("")
        lines.append("The brief's condition is explicit: \"P1d enters the final architecture "
                     "only if this analysis shows conditional structure, not on faith.\" That "
                     "condition is not met, and the K-point curve should be replaced by its "
                     "scalar level — or the branch dropped at Stage 3.")
    lines += ["", f"Figures: {', '.join(f'`{p}`' for p in results['figures'] if p) or 'none'}", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parquet", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-holdout", action="store_true")
    args = ap.parse_args()

    df = lib.load_samples(args.parquet).reset_index(drop=True)
    if not args.allow_holdout:
        lib.assert_no_holdout(df["dataset"].unique())
    R, cols = response_matrix(df)
    parts = decompose(R)
    groups = df["dataset"].to_numpy()
    args.out.mkdir(parents=True, exist_ok=True)

    targets = {
        "forgery": df["label"].to_numpy().astype(int),
        # "is the anchor wrong here?" — the complementarity question, asked of R(x) directly
        "anchor_error": ((df["p_sem"].to_numpy() > 0.5).astype(int)
                         != df["label"].to_numpy().astype(int)).astype(int),
    }
    probes = {}
    for name, y in targets.items():
        entry = {which: probe(parts[which], y, groups) for which in ("level", "shape", "full")}
        entry["shape_increment"] = entry["full"]["auroc"] - entry["level"]["auroc"]
        probes[name] = entry
        print(f"{name}: level {entry['level']['auroc']:.4f}  shape {entry['shape']['auroc']:.4f}  "
              f"full {entry['full']['auroc']:.4f}  increment {entry['shape_increment']:+.4f}")

    sep = family_separability(R, groups)
    print(f"family separability: full Fisher {sep['full']['fisher_ratio']:.4f}, "
          f"shape Fisher {sep['shape']['fisher_ratio']:.4f}")

    figures = [component_plot(R, df, cols, args.out), embed_plot(R, df, args.out, args.seed)]
    results = {"probes": probes, "family_separability": sep, "figures": figures}
    meta = {"parquet": [str(p) for p in args.parquet], "n": int(len(df)),
            "n_methods": int(df["dataset"].nunique()), "k": len(cols), "columns": cols}
    (args.out / "stage2_rate_response.json").write_text(
        json.dumps({"meta": meta, **results}, indent=2, default=str))
    doc = args.out / "STAGE2_MRVAE_rate_response.md"
    doc.write_text(render(results, {**meta, "parquet": args.parquet[0]}))
    print(f"\nwrote {doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""§20 — the domain-detector distribution audit, plus t-SNE views of the evidence space.

    🔴 UMAR-RUNS (CPU, minutes):

    python analysis/discern_v2/domain_audit.py \
        --parquet logs/v1/eval/epoch_001/per_sample_epoch_1.parquet \
        --out analysis/discern_v2/V1_domain_audit

The question (§20)
------------------
A branch that scores well may be exploiting dataset acquisition or compression differences rather
than forensic signal. For the reference and the process branch:

    compare  p(r | R_FF++)   p(r | R_OOD)   p(r | F_OOD)
    test     D(R_FF++, R_OOD)  <<  D(R_OOD, F_OOD)  ?

If authentic footage from an unseen source moves the residual as much as a manipulation does, the
branch is reading the dataset and its headline AUROC is suspect. §20 makes this a required
diagnostic, not a build gate: failures are flagged for the iterate stage, not blocked.

Two measurement choices, both deliberate
----------------------------------------
* **Separability is reported as AUROC**, not as a raw distance. Residual magnitude, angle and the
  six process statistics live on different scales, so a Wasserstein distance of 0.3 means
  something different for each and the three-way comparison would not be commensurable. AUROC is
  bounded, unit-free and directly readable as "how well does this quantity tell the two groups
  apart". The standardised mean shift is reported alongside for effect size and direction.
* **The FF++ side must be the TEST split.** Stage A fits `P_R` on FF++ *train* reals, so those
  frames are in-sample for the reference: it reconstructs them better than any unseen real, the
  residual is systematically smaller, and every arm would look like a dataset detector for a
  reason that has nothing to do with domain. The audit refuses to run if the FF++ rows it is
  handed came from the train split.

An uninformative branch is not a passing branch
-----------------------------------------------
If a residual separates fakes at chance AND separates datasets at chance, the "gap" between the
two numbers is healthy-looking and meaningless. That case is reported as UNINFORMATIVE rather
than as a pass — it is a different finding from "reads the dataset", and conflating them would
hide a branch that contributes nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PROTOCOL = "FaceForensics++"

# The residual descriptors §20 names, per branch. Kept explicit so the report says which quantity
# each verdict is about rather than pooling them into one opaque score.
BRANCH_QUANTITIES = {
    "ref": ["ref_residual_norm", "ref_angle", "p_ref", "u_ref"],
    "proc": ["proc_mse_mean", "proc_mse_std", "proc_mse_max", "proc_mse_p90",
             "proc_lpips_proxy", "proc_center_ratio", "p_proc", "u_proc"],
    "sem": ["p_sem", "u_sem"],
}
FORENSIC_CHANCE = 0.55        # below this a quantity does not separate fakes at all
DOMAIN_CHANCE = 0.55          # below this it does not separate datasets either


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Separability, oriented so 0.5 is chance and >0.5 means group 1 scores higher.

    Reported un-flipped: a value well BELOW 0.5 is just as much separability as one above, and
    folding it to max(a, 1-a) would hide the direction, which is what tells a shrinking residual
    apart from a growing one (AEROBLADE's signed response).
    """
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2 or len(y) < 4:
        return float("nan")
    return float(roc_auc_score(y, s))


def separability(a: np.ndarray, b: np.ndarray) -> float:
    """AUROC of `b` scoring higher than `a`, folded to [0.5, 1] for magnitude comparisons."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    y = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
    s = np.concatenate([a, b])
    value = auroc(y, s)
    return float(max(value, 1.0 - value)) if np.isfinite(value) else float("nan")


def standardised_shift(reference: np.ndarray, other: np.ndarray) -> float:
    if len(reference) < 2 or len(other) < 1:
        return float("nan")
    return float((other.mean() - reference.mean()) / (reference.std() + 1e-12))


def audit_quantity(ffpp_real: np.ndarray, ood_real: np.ndarray,
                   ood_fake: np.ndarray) -> dict:
    """§20's three-way comparison for one scalar quantity."""
    domain = separability(ffpp_real, ood_real)          # R_FF++ vs R_OOD  — pure domain
    forensic = separability(ood_real, ood_fake)         # R_OOD vs F_OOD   — domain cancels
    gap = forensic - domain

    if not np.isfinite(domain) or not np.isfinite(forensic):
        verdict = "NOT COMPUTABLE"
    elif forensic < FORENSIC_CHANCE and domain < DOMAIN_CHANCE:
        verdict = "UNINFORMATIVE"
    elif gap <= 0:
        verdict = "DATASET DETECTOR"
    elif gap <= 0.05:
        verdict = "BORDERLINE"
    else:
        verdict = "pass"
    return {
        "domain_separability": domain,
        "forensic_separability": forensic,
        "forensic_minus_domain": gap,
        "shift_ood_real": standardised_shift(ffpp_real, ood_real),
        "shift_ood_fake": standardised_shift(ffpp_real, ood_fake),
        "verdict": verdict,
        "n_ffpp_real": int(len(ffpp_real)), "n_ood_real": int(len(ood_real)),
        "n_ood_fake": int(len(ood_fake)),
    }


def run_audit(df: pd.DataFrame) -> dict:
    protocol = df[df["dataset"] == PROTOCOL]
    if protocol.empty:
        raise SystemExit(
            f"no {PROTOCOL} rows in the export — the audit needs the protocol source as its "
            f"authentic reference distribution. Re-run eval_v1.py including {PROTOCOL}.")
    ffpp_real = protocol[protocol["label"] == 0]

    results: dict[str, dict] = {}
    for branch, quantities in BRANCH_QUANTITIES.items():
        present = [q for q in quantities if q in df.columns]
        if not present:
            continue
        per_source: dict[str, dict] = {}
        for source in sorted(df["dataset"].unique()):
            if source == PROTOCOL:
                continue
            ood = df[df["dataset"] == source]
            rows = {q: audit_quantity(ffpp_real[q].to_numpy(),
                                      ood[ood["label"] == 0][q].to_numpy(),
                                      ood[ood["label"] == 1][q].to_numpy())
                    for q in present}
            worst = min((r for r in rows.values() if np.isfinite(r["forensic_minus_domain"])),
                        key=lambda r: r["forensic_minus_domain"], default=None)
            per_source[source] = {
                "quantities": rows,
                "verdict": (worst["verdict"] if worst else "NOT COMPUTABLE"),
                "worst_quantity": (min((q for q in rows
                                        if np.isfinite(rows[q]["forensic_minus_domain"])),
                                       key=lambda q: rows[q]["forensic_minus_domain"],
                                       default=None)),
            }
        results[branch] = per_source
    return results


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def _grid(n: int, per_row: int = 3):
    rows = int(np.ceil(n / per_row))
    return rows, min(n, per_row)


def plot_distributions(df: pd.DataFrame, quantity: str, out: Path, title: str) -> None:
    """One figure, one panel per OOD source: p(r|R_FF++) vs p(r|R_OOD) vs p(r|F_OOD)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sources = [s for s in sorted(df["dataset"].unique()) if s != PROTOCOL]
    if not sources:
        return
    ffpp_real = df[(df["dataset"] == PROTOCOL) & (df["label"] == 0)][quantity].to_numpy()
    rows, cols = _grid(len(sources))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.4 * rows), squeeze=False)
    for ax, source in zip(axes.flat, sources):
        ood = df[df["dataset"] == source]
        r_ood = ood[ood["label"] == 0][quantity].to_numpy()
        f_ood = ood[ood["label"] == 1][quantity].to_numpy()
        pooled = np.concatenate([x for x in (ffpp_real, r_ood, f_ood) if len(x)])
        lo, hi = np.percentile(pooled, [0.5, 99.5])
        bins = np.linspace(lo, hi, 50)
        for values, label, colour in ((ffpp_real, "R FF++ (test)", "#4C72B0"),
                                      (r_ood, f"R {source}", "#55A868"),
                                      (f_ood, f"F {source}", "#C44E52")):
            if len(values):
                ax.hist(values, bins=bins, density=True, alpha=0.5, label=label, color=colour)
        ax.set_title(source, fontsize=10)
        ax.set_xlabel(quantity)
        ax.legend(fontsize=7)
    for ax in axes.flat[len(sources):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def evidence_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """The per-sample evidence/reliability vector t-SNE is computed on.

    Deliberately the *evidence space*, not raw backbone features: this figure exists to show how
    the branches' opinions and reliability signals arrange samples, which is the space the fusion
    and the gate actually operate in. Raw-feature t-SNE would describe the encoders instead.
    """
    columns = [c for c in df.columns
               if c.startswith(("p_", "u_", "e_", "ref_", "proc_")) or c in ("V", "C", "A")]
    columns = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    x = df[columns].to_numpy(dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = (x - x.mean(0)) / (x.std(0) + 1e-8)
    return x, columns


def plot_tsne(df: pd.DataFrame, out: Path, per_source: int = 1500, seed: int = 42) -> dict:
    """One figure: a panel per dataset coloured by label, plus two pooled panels.

    The pooled panels are the point of the exercise. Coloured by DATASET, visible clustering by
    source is the same claim the numeric audit makes — the evidence space is arranged by
    provenance. Coloured by LABEL, separation means it is arranged by authenticity. Which of the
    two dominates is the whole question.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    rng = np.random.default_rng(seed)
    parts = []
    for source in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == source]
        if len(sub) > per_source:
            # stratified by label so a source with 90% fakes does not lose its reals
            keep = []
            for label, group in sub.groupby("label"):
                n = max(1, int(round(per_source * len(group) / len(sub))))
                keep.append(group.iloc[rng.choice(len(group), size=min(n, len(group)),
                                                  replace=False)])
            sub = pd.concat(keep)
        parts.append(sub)
    sample = pd.concat(parts, ignore_index=True)

    x, columns = evidence_matrix(sample)
    perplexity = float(min(30, max(5, len(sample) / 100)))
    embedded = TSNE(n_components=2, init="pca", random_state=seed,
                    perplexity=perplexity).fit_transform(x)
    sample = sample.assign(tsne_x=embedded[:, 0], tsne_y=embedded[:, 1])

    sources = sorted(sample["dataset"].unique())
    n_panels = len(sources) + 2
    rows, cols = _grid(n_panels)
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.0 * rows), squeeze=False)
    # Indexed explicitly rather than by consuming an iterator: `zip(flat, sources)` pulls one
    # extra axis before it stops, which leaves an empty framed panel nobody turns off.
    panels = list(axes.flat)

    ax = panels[0]
    palette = plt.cm.tab10(np.linspace(0, 1, max(len(sources), 2)))
    for colour, source in zip(palette, sources):
        s_df = sample[sample["dataset"] == source]
        ax.scatter(s_df["tsne_x"], s_df["tsne_y"], s=3, alpha=0.5, color=colour, label=source)
    ax.set_title("all sources — coloured by DATASET\n(clustering here = provenance structure)",
                 fontsize=9)
    ax.legend(fontsize=6, markerscale=2)

    ax = panels[1]
    for label, colour, name in ((0, "#4C72B0", "real"), (1, "#C44E52", "fake")):
        s_df = sample[sample["label"] == label]
        ax.scatter(s_df["tsne_x"], s_df["tsne_y"], s=3, alpha=0.5, color=colour, label=name)
    ax.set_title("all sources — coloured by LABEL\n(clustering here = forensic structure)",
                 fontsize=9)
    ax.legend(fontsize=7, markerscale=2)

    # per-source panels share the pooled embedding's axes, so positions are comparable panel to
    # panel rather than each being its own arbitrary layout
    xlim = (sample["tsne_x"].min(), sample["tsne_x"].max())
    ylim = (sample["tsne_y"].min(), sample["tsne_y"].max())
    for ax, source in zip(panels[2:], sources):
        s_df = sample[sample["dataset"] == source]
        ax.scatter(sample["tsne_x"], sample["tsne_y"], s=2, alpha=0.08, color="#999999")
        for label, colour, name in ((0, "#4C72B0", "real"), (1, "#C44E52", "fake")):
            t = s_df[s_df["label"] == label]
            ax.scatter(t["tsne_x"], t["tsne_y"], s=4, alpha=0.7, color=colour, label=name)
        ax.set_title(f"{source} (n={len(s_df)})", fontsize=9)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.legend(fontsize=6, markerscale=2)

    for ax in panels:
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in panels[n_panels:]:
        ax.axis("off")

    fig.suptitle("t-SNE of the per-sample evidence space "
                 f"({len(columns)} standardised quantities, {len(sample)} points)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return {"n_points": int(len(sample)), "n_columns": len(columns), "columns": columns,
            "perplexity": perplexity}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def render(results: dict, meta: dict) -> str:
    lines = [
        "# V1_BRANCH_DIAGNOSTICS — §20 domain-detector distribution audit",
        "",
        f"Source: `{meta['parquet']}` (epoch {meta.get('epoch', '?')}), "
        f"{meta['n_rows']} frames across {len(meta['datasets'])} sources.",
        "",
        "**Reading this table.** `forensic` separates reals from fakes *within* an OOD source, so "
        "domain shift is common to both classes and cancels. `domain` separates FF++ **test** "
        "reals from OOD reals, where no manipulation is involved at all. §20's test is "
        "`D(R_FF++, R_OOD) << D(R_OOD, F_OOD)`; a branch whose domain number meets its forensic "
        "number is reading the dataset, not the forgery.",
        "",
        "Both are AUROC-based separability folded to [0.5, 1], so quantities on different scales "
        "stay comparable. §20 makes this a required diagnostic, not a gate: failures are flagged "
        "for the iterate stage.",
        "",
    ]
    for branch, per_source in results.items():
        lines += [f"## Branch `{branch}`", "",
                  "| source | verdict | worst quantity | forensic | domain | gap |",
                  "|---|---|---|---|---:|---:|"]
        for source, entry in per_source.items():
            worst = entry["worst_quantity"]
            row = entry["quantities"].get(worst, {}) if worst else {}
            lines.append(
                f"| {source} | **{entry['verdict']}** | `{worst}` | "
                f"{row.get('forensic_separability', float('nan')):.4f} | "
                f"{row.get('domain_separability', float('nan')):.4f} | "
                f"{row.get('forensic_minus_domain', float('nan')):+.4f} |")
        lines.append("")
        lines.append("<details><summary>every quantity</summary>")
        lines.append("")
        lines.append("| source | quantity | forensic | domain | gap | shift R_OOD (σ) | "
                     "shift F_OOD (σ) | verdict |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---|")
        for source, entry in per_source.items():
            for name, r in entry["quantities"].items():
                lines.append(
                    f"| {source} | `{name}` | {r['forensic_separability']:.4f} | "
                    f"{r['domain_separability']:.4f} | {r['forensic_minus_domain']:+.4f} | "
                    f"{r['shift_ood_real']:+.2f} | {r['shift_ood_fake']:+.2f} | {r['verdict']} |")
        lines += ["", "</details>", ""]
    lines += [
        "## Figures",
        "",
        "- `distributions_<quantity>.png` — one panel per source, showing `p(r | R_FF++)`, "
        "`p(r | R_OOD)` and `p(r | F_OOD)` together. This is §20's plot.",
        "- `tsne_evidence_space.png` — the per-sample evidence space. The two pooled panels are "
        "the ones to read: clustering by DATASET means the evidence space is arranged by "
        "provenance, clustering by LABEL means it is arranged by authenticity.",
        "",
        "The t-SNE is computed on the evidence and reliability quantities (per-branch p/u, "
        "residual diagnostics, process statistics, V/C/A) rather than on raw backbone features, "
        "because that is the space fusion and the applicability gate actually operate in.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parquet", type=Path, nargs="+", required=True,
                    help="per-sample export(s) from training/eval_v1.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tsne-per-source", type=int, default=1500)
    ap.add_argument("--skip-tsne", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.read_parquet(p) for p in args.parquet], ignore_index=True)
    datasets = sorted(df["dataset"].unique())
    print(f"loaded {len(df)} frames across {len(datasets)} sources: {datasets}")

    if PROTOCOL not in datasets:
        raise SystemExit(f"{PROTOCOL} must be present — it supplies p(r | R_FF++)")
    # The FF++ rows must be the TEST split: Stage A fit P_R on FF++ train, so train reals are
    # in-sample for the reference and would make every branch look like a dataset detector.
    keys = df[df["dataset"] == PROTOCOL]["key"].astype(str)
    if keys.str.contains("/train/").any():
        raise SystemExit("the FF++ rows look like training frames; the audit needs the test split")

    results = run_audit(df)
    meta = {"parquet": [str(p) for p in args.parquet], "n_rows": int(len(df)),
            "datasets": datasets}

    figures = {}
    for quantity in ("ref_residual_norm", "ref_angle", "proc_mse_mean", "p_ref", "p_proc"):
        if quantity in df.columns:
            path = args.out / f"distributions_{quantity}.png"
            plot_distributions(df, quantity, path, f"§20 residual distributions — {quantity}")
            figures[quantity] = str(path)
            print(f"  wrote {path}")

    if not args.skip_tsne:
        print("  computing t-SNE (this is the slow part)")
        info = plot_tsne(df, args.out / "tsne_evidence_space.png",
                         per_source=args.tsne_per_source, seed=args.seed)
        figures["tsne"] = info
        print(f"  wrote {args.out}/tsne_evidence_space.png ({info['n_points']} points)")

    (args.out / "domain_audit.json").write_text(
        json.dumps({"meta": meta, "figures": figures, "results": results},
                   indent=2, default=str))
    (args.out / "V1_BRANCH_DIAGNOSTICS.md").write_text(render(results, meta))
    print(f"\nwrote {args.out}/V1_BRANCH_DIAGNOSTICS.md")

    for branch, per_source in results.items():
        flagged = [s for s, e in per_source.items()
                   if e["verdict"] in ("DATASET DETECTOR", "BORDERLINE")]
        print(f"  {branch}: {len(flagged)} flagged source(s)" +
              (f" — {flagged}" if flagged else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

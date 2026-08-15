#!/usr/bin/env python3
"""A3 — does P1d's rate response R(x) carry incremental forensic/applicability information?

Runs FIRST because it is the cheapest analysis and gates the most: it decides whether P1d
keeps privileged status going into D1.

Three questions, per the Phase-1 instructions. Family clustering is explicitly NOT the only
thing that keeps P1d alive:

  Q1 forgery separability      real/fake linear probe from R(x)
  Q2 family structure          silhouette / Fisher ratio over FS/FR/EFS/FE  (descriptive)
  Q3 complementarity           P0-error prediction from R(x), + rescue/harm separation

**Revised gate**: P1d survives if R(x) adds incremental signal on *any* of the three, even
with no clean family clusters — it may separate real/fake, or P0-correct/P0-error, without
naming the family. Only if it adds nothing on all three does P1d lose privileged status.

Two disciplines are enforced rather than assumed:

* **Held-out generator evaluation for the supervised probes (Q1, Q3).** Training and testing
  a probe on the same DF40 generators measures memorisation of generator fingerprints, not
  transferable signal. Q2's silhouette/Fisher and the UMAP are descriptive and use the full
  set.
* **Per-fold scalers.** Standardisation is fit on each fold's training generators only
  (`common.fold_scaler`). Fitting globally before splitting leaks the held-out generator.

"Incremental" is measured explicitly: every probe is reported both standalone and as a
delta over a baseline built from the evidence already available without R(x). A probe that
merely re-encodes p_fused is not incremental information.

    python analysis/discern_v2/a3_rate_response.py
    python analysis/discern_v2/a3_rate_response.py --source DF40 --min-per-group 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

RATE_PILOT = "P1d"          # the only pilot whose projector exposes a rate response
FAMILIES = ["FS", "FR", "EFS", "FE"]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def load_rate_response(source: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """R(x) (N,K), beta grid (K,), metadata. Fails loudly if the export lacks the response."""
    arrays, meta = C.load(RATE_PILOT, source)
    if "rate_response" not in arrays:
        raise KeyError(
            f"{RATE_PILOT}/{source} export has no 'rate_response'. The exporter only captures "
            f"it when the projector exposes rate_distortion_response(). Re-export with the "
            f"MR-VAE projector:\n    {C.export_command(RATE_PILOT, [source])}")
    betas = arrays.get("beta_grid", np.array([0.1, 0.32, 1.0, 3.16, 10.0], dtype=np.float32))
    return arrays["rate_response"], betas, meta


def slope_features(R: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """Augment the raw K-point response with its log-log slope and curvature.

    The response is a curve, so its *shape* is the hypothesis ("a real face decays
    gracefully; a manipulated one falls off a cliff"). Handing the probe only the K raw
    points makes it rediscover that shape from scratch on a small sample.
    """
    x = np.log(betas)
    x = x - x.mean()
    y = np.log(np.clip(R, 1e-8, None))
    yc = y - y.mean(axis=1, keepdims=True)
    slope = (yc @ x) / (x @ x)
    curv = np.polyfit(x, y.T, 2)[0] if len(x) >= 3 else np.zeros(len(R))
    return np.column_stack([R, slope, curv])


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------


def logo_probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
               real_labels: np.ndarray, min_per_group: int,
               video_ids: np.ndarray | None = None) -> pd.DataFrame:
    """Logistic probe under leave-one-generator-out, scalers fit per fold.

    `groups` are generator IDs and `real_labels` is the real/fake label used only to build
    the folds -- a generator group is fakes-only, so the shared real pool must be split into
    disjoint halves or every fold is single-class and the probe silently returns nothing.
    `y` is the probe target, which is not necessarily the real/fake label (Q3 predicts P0
    error, and the rescue-vs-harm probe predicts which of the two a sample is).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    rows = []
    for held, tr, te in C.logo_folds_two_class(groups, real_labels, min_per_group,
                                               video_ids=video_ids):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        Xtr, Xte = C.fold_scaler(X[tr], X[te])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, y[tr])
        auc = roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1])
        rows.append({"held_out": held, "n_test": int(te.sum()), "auroc": float(auc)})
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {"probe": label, "n_folds": 0, "mean_auroc": None, "median_auroc": None}
    return {"probe": label, "n_folds": len(df),
            "mean_auroc": float(df["auroc"].mean()),
            "median_auroc": float(df["auroc"].median()),
            "min_auroc": float(df["auroc"].min()),
            "max_auroc": float(df["auroc"].max())}


# ---------------------------------------------------------------------------
# Q2 descriptive structure
# ---------------------------------------------------------------------------


def family_structure(X: np.ndarray, fam: np.ndarray) -> dict:
    """Silhouette + Fisher ratio over families. Descriptive: full set, no folds."""
    from sklearn.metrics import silhouette_score

    keep = np.isin(fam, FAMILIES)
    X, fam = X[keep], fam[keep]
    present = [f for f in FAMILIES if (fam == f).sum() > 1]
    if len(present) < 2:
        return {"silhouette": None, "fisher_ratio": None, "families": present}

    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    sil = float(silhouette_score(Xs, fam))

    # Fisher ratio: between-family scatter over within-family scatter (trace form).
    mu = Xs.mean(0)
    between = sum((fam == f).sum() * ((Xs[fam == f].mean(0) - mu) ** 2).sum() for f in present)
    within = sum(((Xs[fam == f] - Xs[fam == f].mean(0)) ** 2).sum() for f in present)
    return {"silhouette": sil,
            "fisher_ratio": float(between / within) if within > 0 else None,
            "families": present,
            "n": int(len(fam))}


# ---------------------------------------------------------------------------
# plots (descriptive)
# ---------------------------------------------------------------------------


def plot_curves(R: np.ndarray, betas: np.ndarray, fam: np.ndarray, label: np.ndarray,
                out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"FS": "#2a78d6", "FR": "#eb6834", "EFS": "#1baf7a", "FE": "#eda100"}
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    m = label == 0
    if m.sum():
        ax.plot(betas, np.median(R[m], 0), color="#8a8880", lw=2.2, marker="o", label=f"real (n={m.sum()})")
        ax.fill_between(betas, *np.percentile(R[m], [25, 75], axis=0), color="#8a8880", alpha=.15, lw=0)
    for f in FAMILIES:
        mm = (fam == f) & (label == 1)
        if mm.sum() < 10:
            continue
        ax.plot(betas, np.median(R[mm], 0), color=colors[f], lw=2.0, marker="o", label=f"{f} (n={mm.sum()})")
        ax.fill_between(betas, *np.percentile(R[mm], [25, 75], axis=0), color=colors[f], alpha=.15, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel("rate penalty  beta  (log)")
    ax.set_ylabel("distortion  1 - cos(f_s, recon)")
    ax.set_title("A3 — P1d rate response by family (median, IQR band)")
    ax.legend(fontsize=8)
    ax.grid(True, color="#e4e3df", lw=.6)
    ax.set_axisbelow(True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


def plot_embedding(X: np.ndarray, fam: np.ndarray, label: np.ndarray, out: Path) -> str:
    """UMAP if available, else PCA. Returns which was used, for the verdict text."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    try:
        import umap
        xy = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=C.SEED).fit_transform(Xs)
        method = "UMAP"
    except ImportError:
        from sklearn.decomposition import PCA
        xy = PCA(n_components=2, random_state=C.SEED).fit_transform(Xs)
        method = "PCA (umap-learn not installed)"

    colors = {"FS": "#2a78d6", "FR": "#eb6834", "EFS": "#1baf7a", "FE": "#eda100"}
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    m = label == 0
    ax.scatter(xy[m, 0], xy[m, 1], s=5, c="#8a8880", alpha=.35, lw=0, label="real")
    for f in FAMILIES:
        mm = (fam == f) & (label == 1)
        if mm.sum():
            ax.scatter(xy[mm, 0], xy[mm, 1], s=5, c=colors[f], alpha=.5, lw=0, label=f)
    ax.set_title(f"A3 — rate-response embedding ({method})\ndescriptive only; read with the metrics")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=3, fontsize=8)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    return method


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="DF40")
    ap.add_argument("--min-per-group", type=int, default=20)
    args = ap.parse_args()

    out = C.out_dir("A3_rate_response")

    missing = C.missing_exports([RATE_PILOT, C.P0], [args.source])
    if missing:
        print("A3 cannot run — required per-sample exports are missing (UMAR-RUNS):")
        for p, s in missing:
            print(f"  {p}/{s}:  cd {C.DICOME} && {C.export_command(p, [s])}")
        print("\nNote: P1d's export must come from the MR-VAE projector so that "
              "'rate_response' is captured.")
        return 1

    R, betas, meta = load_rate_response(args.source)
    X = slope_features(R, betas)
    label = meta["label"].values
    fam = meta["family"].values
    gen = meta["method"].values
    vid = meta["video_id"].values  # frame-level data: reals are split by whole video

    # P0's per-sample correctness, aligned on key (both use the same frozen cohort).
    p0_arr, p0_meta = C.load(C.P0, args.source)
    thr, prov = C.frozen_threshold(C.P0)
    p0 = pd.DataFrame({"key": p0_meta["key"], "p0_p": p0_arr["p_fused"],
                       "p0_label": p0_meta["label"]})
    j = pd.DataFrame({"key": meta["key"]}).merge(p0, on="key", how="left")
    if j["p0_p"].isna().any():
        raise RuntimeError(f"{int(j['p0_p'].isna().sum())} samples missing a P0 score — "
                           f"the two exports are not on the same cohort")
    p0_correct = ((j["p0_p"].values >= thr).astype(int) == j["p0_label"].values)

    res: dict = {"source": args.source, "pilot": RATE_PILOT,
                 "n_samples": int(len(meta)), "beta_grid": betas.tolist(),
                 "threshold": thr, "threshold_provenance": prov,
                 "min_per_group": args.min_per_group}

    # ---- Q1 forgery separability -------------------------------------------------
    q1_R = logo_probe(X, label.astype(int), gen, label, args.min_per_group, vid)
    # Incrementality baseline: what a probe gets from P1d's own fused score alone.
    p1d_arr, _ = C.load(RATE_PILOT, args.source)
    base_feat = p1d_arr["p_fused"].reshape(-1, 1)
    q1_base = logo_probe(base_feat, label.astype(int), gen, label, args.min_per_group, vid)
    q1_both = logo_probe(np.column_stack([X, base_feat]), label.astype(int), gen,
                         label, args.min_per_group, vid)
    res["Q1_forgery_separability"] = {
        "R_only": summarise(q1_R, "R(x)"),
        "score_only_baseline": summarise(q1_base, "p_fused"),
        "R_plus_score": summarise(q1_both, "R(x) + p_fused"),
        "incremental_auroc": (None if q1_both.empty or q1_base.empty else
                              float(q1_both["auroc"].mean() - q1_base["auroc"].mean())),
    }
    q1_R.to_csv(out / "Q1_folds_R_only.csv", index=False)
    q1_both.to_csv(out / "Q1_folds_R_plus_score.csv", index=False)

    # ---- Q2 family structure (descriptive) ---------------------------------------
    res["Q2_family_structure"] = family_structure(X[label == 1], fam[label == 1])

    # ---- Q3 P0-error predictiveness ----------------------------------------------
    err = (~p0_correct).astype(int)
    q3_R = logo_probe(X, err, gen, label, args.min_per_group, vid)
    q3_base = logo_probe(base_feat, err, gen, label, args.min_per_group, vid)
    res["Q3_p0_error_prediction"] = {
        "R_only": summarise(q3_R, "R(x) -> P0 error"),
        "score_only_baseline": summarise(q3_base, "p_fused -> P0 error"),
        "incremental_auroc": (None if q3_R.empty or q3_base.empty else
                              float(q3_R["auroc"].mean() - q3_base["auroc"].mean())),
        "p0_error_rate": float(err.mean()),
    }
    q3_R.to_csv(out / "Q3_folds_R_only.csv", index=False)

    # rescue/harm separation: does R(x) separate samples P1d rescues from those it harms?
    p1d_correct = ((p1d_arr["p_fused"] >= thr).astype(int) == label)
    rescue = (~p0_correct) & p1d_correct
    harm = p0_correct & (~p1d_correct)
    sep = {}
    if rescue.sum() >= 10 and harm.sum() >= 10:
        sub = np.concatenate([X[rescue], X[harm]])
        ysub = np.r_[np.ones(rescue.sum()), np.zeros(harm.sum())]
        gsub = np.concatenate([gen[rescue], gen[harm]])
        lsub = np.concatenate([label[rescue], label[harm]])
        vsub = np.concatenate([vid[rescue], vid[harm]])
        rh = logo_probe(sub, ysub.astype(int), gsub, lsub, min_per_group=10, video_ids=vsub)
        sep = summarise(rh, "R(x) -> rescue vs harm")
        rh.to_csv(out / "Q3_folds_rescue_vs_harm.csv", index=False)
    res["Q3_rescue_harm_separation"] = {"n_rescue": int(rescue.sum()),
                                        "n_harm": int(harm.sum()), **sep}

    # ---- plots --------------------------------------------------------------------
    plot_curves(R, betas, fam, label, out / "curves_by_family.png")
    res["embedding_method"] = plot_embedding(X, fam, label, out / "embedding.png")

    (out / "A3_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\nwrote {out}")
    print("\nNext: write the verdict paragraph in A3_VERDICT.md from these statistics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

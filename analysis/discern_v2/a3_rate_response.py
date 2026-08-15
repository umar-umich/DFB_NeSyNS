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


def fakes_only_probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                     video_ids: np.ndarray, min_per_group: int,
                     video_level: bool = True) -> pd.DataFrame:
    """Leave-one-generator-out probe over FAKES ONLY.

    No real pool is involved, so the folds are simply "this generator" vs "every other
    generator" and the two-class requirement falls on the target `y` (P0 error) rather than
    on real/fake. This is the coherent form of the applicability question: *will P0 miss
    this generator's forgeries?*

    Aggregated to video level by default. The exports are frame-level and frames of one
    video almost always share their outcome, so a frame-level AUROC counts the same
    evidence dozens of times and reports an effective sample size far larger than the real
    one. Video level is the honest unit here.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    if video_level:
        df = pd.DataFrame(X)
        feat_cols = list(df.columns)
        df["_y"], df["_g"], df["_v"] = y, groups, video_ids
        agg = df.groupby("_v", as_index=False).agg(
            {**{c: "mean" for c in feat_cols}, "_y": "max", "_g": "first"})
        X, y = agg[feat_cols].values, agg["_y"].values.astype(int)
        groups = agg["_g"].values
        # "_y": max -> a video counts as a P0 error if any of its frames is one. Videos are
        # overwhelmingly all-or-nothing, so this is a tie-break, not a redefinition.

    rows = []
    for g in sorted(pd.unique(groups)):
        te = groups == g
        tr = ~te
        if te.sum() < min_per_group:
            continue
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            # P0 is either right on all of this generator or wrong on all of it; AUROC is
            # undefined. Skipped and counted, never silently folded into the mean.
            continue
        Xtr, Xte = C.fold_scaler(X[tr], X[te])
        clf = LogisticRegression(max_iter=2000).fit(Xtr, y[tr])
        rows.append({"held_out": str(g), "n_test": int(te.sum()),
                     "auroc": float(roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1]))})
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
    ap.add_argument("--min-per-group", type=int, default=20,
                    help="minimum FRAMES per held-out generator (frame-level probes)")
    ap.add_argument("--min-videos-per-group", type=int, default=8,
                    help=("minimum VIDEOS per held-out generator (video-level probes). "
                          "DF40 generators hold ~18 videos each, so reusing the frame-level "
                          "threshold of 20 would silently discard almost every fold."))
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
                 "min_per_group": args.min_per_group,
                 "min_videos_per_group": args.min_videos_per_group}

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
    #
    # WITHIN CLASS, deliberately. A pooled real+fake "P0 error" target is incoherent: the
    # error means opposite things on either side. A real P0 gets wrong is a false positive
    # (it looked FAKE); a fake P0 gets wrong is a miss (it looked REAL). Pooling inverts the
    # probe, because the train folds are fake-majority (~40 generators x 18) while the test
    # folds are real-majority (18 fakes vs ~416 reals) -- so it learns one direction and is
    # scored on the other. That yielded a systematic AUROC of ~0.17, which reads as
    # "anti-predictive" but is a sign error. The slope deltas confirm the flip directly:
    # -0.0022 for reals, +0.0077 for fakes.
    #
    # The applicability question that matters is fakes-only: will P0 miss this generator's
    # forgeries? Reals are a single pool with no generator to hold out, so no leave-one-
    # generator-out real-side number exists; it is reported as absent rather than
    # manufactured from a different split.
    err = (~p0_correct).astype(int)
    res["Q3_p0_error_prediction"] = {
        "p0_error_rate_overall": float(err.mean()),
        "p0_error_rate_reals": float(err[label == 0].mean()) if (label == 0).any() else None,
        "p0_error_rate_fakes": float(err[label == 1].mean()) if (label == 1).any() else None,
        "note": ("evaluated within class at video level; a pooled target inverts because a "
                 "real-side error is a false positive and a fake-side error is a miss"),
    }

    fake = label == 1
    q3_R = fakes_only_probe(X[fake], err[fake], gen[fake], vid[fake],
                            args.min_videos_per_group)
    q3_base = fakes_only_probe(base_feat[fake], err[fake], gen[fake], vid[fake],
                               args.min_videos_per_group)
    res["Q3_p0_error_prediction"]["fakes_only"] = {
        "R_only": summarise(q3_R, "R(x) -> P0 miss (fakes, video level)"),
        "score_only_baseline": summarise(q3_base, "p_fused -> P0 miss (fakes, video level)"),
        "incremental_auroc": (None if q3_R.empty or q3_base.empty else
                              float(q3_R["auroc"].mean() - q3_base["auroc"].mean())),
    }
    res["Q3_p0_error_prediction"]["reals_only"] = C.todo(
        "no leave-one-generator-out split exists for reals (single pool); use a "
        "video-grouped CV if a real-side number is wanted")
    q3_R.to_csv(out / "Q3_folds_R_only_fakes.csv", index=False)

    # rescue/harm separation: does R(x) separate samples P1d rescues from those it harms?
    # Fakes only, for the same reason as Q3 above -- a rescue on a real and a rescue on a
    # fake are opposite movements in score space, and pooling them mixes the directions.
    p1d_correct = ((p1d_arr["p_fused"] >= thr).astype(int) == label)
    rescue = (~p0_correct) & p1d_correct & fake
    harm = p0_correct & (~p1d_correct) & fake
    sep = {}
    if rescue.sum() >= 10 and harm.sum() >= 10:
        sub = np.concatenate([X[rescue], X[harm]])
        ysub = np.r_[np.ones(rescue.sum()), np.zeros(harm.sum())].astype(int)
        gsub = np.concatenate([gen[rescue], gen[harm]])
        vsub = np.concatenate([vid[rescue], vid[harm]])
        mpg = max(4, args.min_videos_per_group // 2)
        rh = fakes_only_probe(sub, ysub, gsub, vsub, min_per_group=mpg)
        sep = summarise(rh, "R(x) -> rescue vs harm (fakes, video level)")
        rh.to_csv(out / "Q3_folds_rescue_vs_harm.csv", index=False)
        # Same incrementality discipline as Q1/Q3: a standalone AUROC here proves nothing,
        # because p_fused alone separates rescue from harm almost by construction (a rescue
        # is where P1d is right and P0 wrong). Only the delta over that baseline is evidence
        # that the *response curve* carries something the score does not.
        sub_base = np.concatenate([base_feat[rescue], base_feat[harm]])
        rh_base = fakes_only_probe(sub_base, ysub, gsub, vsub, min_per_group=mpg)
        sep["score_only_baseline"] = summarise(rh_base, "p_fused -> rescue vs harm")
        sep["incremental_auroc"] = (
            None if rh.empty or rh_base.empty
            else float(rh["auroc"].mean() - rh_base["auroc"].mean()))
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

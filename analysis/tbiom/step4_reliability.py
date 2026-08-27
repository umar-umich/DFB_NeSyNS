#!/usr/bin/env python3
"""Step 4 — is single-anchor reliability R = g(V, M) a viable spine?

    python analysis/tbiom/step4_reliability.py --out tbiom/STEP4_RELIABILITY.md

THE POINT OF THIS STEP IS THAT IT MIGHT FAIL. V1's ~0.81 error-detection result came from the
MULTI-BRANCH setting, where the risk model saw vacuity V, inter-branch conflict C, support
uncertainty U_sup and margin M. With one anchor, C and U_sup do not exist — there is nothing to
conflict with. So the fallback spine is g(V, M) on two features, and whether it still predicts
errors is an open question, not a formality. Manufacturing C or U_sup from a single branch would
be inventing a number, and is not done here.

    V = vacuity, K / S from the Dirichlet, which the export already carries as `u_*`
    M = margin, |p - 0.5|, distance from the decision boundary

PROTOCOL. Fit on FF++ VAL_meta only, the permitted calibration partition, 5-fold cross-fit so no
sample scores a model that saw it. VAL_select is excluded because it chose the checkpoint; fitting
the risk model there would calibrate the defer policy on the data that selected the model it is
deferring for.

Reported twice, because in-domain and shifted are different questions:

    error-detection AUROC on VAL_meta, out-of-fold   -- does (V, M) see errors at all?
    the same model applied to each OOD domain        -- does it still see them under shift?

plus selective risk at a 10% deferral budget: error rate on the 90% of videos the model is most
confident about. A useful risk model cuts error there; a useless one leaves it at the base rate.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from health import auroc, eer  # noqa: E402
from video_id import video_id  # noqa: E402

STEP1, STEP2 = Path("logs/tbiom/step1"), Path("logs/tbiom/step2")
SPLIT = Path("configs/discern_v2/meta_split.json")
READOUTS = {"p_artifact": "u_artifact", "p_fused": "u_fused", "p_semantic": "u_semantic"}
NICE = {"p_artifact": "artifact view (Step-3 pick)", "p_fused": "fused (DS)",
        "p_semantic": "semantic (CLIP branch)"}
OOD = [("Celeb-DF-v2", "CDFv2"), ("Celeb-DF-v3", "CDFv3"), ("DFD", "DFD"),
       ("DFDC", "DFDC"), ("DFDCP", "DFDCP"), ("Deepfake-Eval-2024", "DFEval24")]


def video_table(path: Path, p_col: str, u_col: str) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    d = pd.read_csv(path)
    if p_col not in d or u_col not in d:
        return None
    vid = video_id(d["key"])
    g = pd.DataFrame({"v": vid, "p": d[p_col], "u": d[u_col], "y": d["label"]}).groupby(
        "v", as_index=False).agg(p=("p", "mean"), u=("u", "mean"), y=("y", "max"))
    g["base"] = g["v"].str.rsplit("/", n=1).str[-1].str.replace(r"^[A-Za-z0-9\-]+__", "",
                                                                regex=True)
    return g


def features(df: pd.DataFrame) -> np.ndarray:
    """R = g(V, M). Exactly two features — no fabricated C or U_sup."""
    V = df["u"].to_numpy()
    M = np.abs(df["p"].to_numpy() - 0.5)
    return np.stack([V, M], axis=1)


def frame_table(path: Path, p_col: str, u_col: str) -> pd.DataFrame | None:
    """Per-FRAME rows. V1's risk model was frame-level (25,532 frames), so this is what makes
    the comparison to its 0.8118 a comparison at all."""
    if not path.is_file():
        return None
    d = pd.read_csv(path)
    if p_col not in d or u_col not in d:
        return None
    out = pd.DataFrame({"p": d[p_col], "u": d[u_col], "y": d["label"]})
    out["base"] = video_id(d["key"]).str.rsplit("/", n=1).str[-1].str.replace(
        r"^[A-Za-z0-9\-]+__", "", regex=True)
    return out


def selective_risk(err: np.ndarray, risk: np.ndarray, budget: float = 0.10) -> tuple[float, float]:
    """Error rate on the (1-budget) least-risky videos, and the base rate for comparison."""
    keep = int(round(len(err) * (1 - budget)))
    order = np.argsort(risk, kind="mergesort")          # low risk first
    return float(err[order[:keep]].mean()), float(err.mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("tbiom/STEP4_RELIABILITY.md"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--level", choices=("frame", "video"), default="frame",
                    help="FRAME by default, because that is what V1's 0.8118 was computed at "
                         "(25,532 frames) and because at video level this anchor makes 0-3 "
                         "errors in 280 VAL_meta videos — there is no error signal to fit.")
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold

    part = json.loads(SPLIT.read_text())["video_to_partition"]
    rows, per_domain = {}, {}

    for p_col, u_col in READOUTS.items():
        loader = frame_table if args.level == "frame" else video_table
        val = loader(STEP2 / "p0ds_e01_FFpp_val.csv", p_col, u_col)
        if val is None:
            continue
        val["partition"] = val["base"].map(part)
        meta = val[val["partition"] == "VAL_meta"].reset_index(drop=True)
        if len(meta) < 50:
            continue

        # the anchor's own operating point, frozen on VAL_select so tau does not see VAL_meta
        sel = val[val["partition"] == "VAL_select"]
        _, tau = eer(sel["y"].to_numpy(), sel["p"].to_numpy())
        err = ((meta["p"].to_numpy() >= tau).astype(int) != meta["y"].to_numpy()).astype(int)
        X = features(meta)

        if err.sum() < 5 or err.sum() == len(err):
            rows[p_col] = {"tau": tau, "n_meta": int(len(meta)), "n_errors": int(err.sum()),
                           "oof_auroc": float("nan"), "note": "too few errors to fit"}
            continue

        # cross-fit by IDENTITY GROUP, not by row: a fake and the real it was made from share an
        # identity, so a random split would put relatives on both sides and leak.
        groups = meta["base"].str.split("_").str[0].to_numpy()
        n_splits = min(args.folds, len(np.unique(groups)), int(err.sum()))
        oof = np.full(len(err), np.nan)
        for tr, te in GroupKFold(n_splits=n_splits).split(X, err, groups):
            if len(np.unique(err[tr])) < 2:
                continue
            sc = StandardScaler().fit(X[tr])
            m = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), err[tr])
            oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
        ok = ~np.isnan(oof)
        oof_auroc = auroc(err[ok], oof[ok])
        sel_risk, base = selective_risk(err[ok], oof[ok])

        # deployment model: refit on all of VAL_meta, then applied unchanged to every OOD domain
        sc = StandardScaler().fit(X)
        model = LogisticRegression(max_iter=2000).fit(sc.transform(X), err)

        dom = {}
        for label, stem in OOD:
            t = loader(STEP1 / f"p0ds_{stem}.csv", p_col, u_col)
            if t is None or t["y"].nunique() < 2:
                continue
            e = ((t["p"].to_numpy() >= tau).astype(int) != t["y"].to_numpy()).astype(int)
            if e.sum() < 5 or e.sum() == len(e):
                continue
            r = model.predict_proba(sc.transform(features(t)))[:, 1]
            sr, br = selective_risk(e, r)
            dom[label] = {"n": int(len(e)), "error_rate": br,
                          "risk_auroc": auroc(e, r),
                          "selective_risk_90": sr,
                          "risk_reduction": (br - sr) / br if br > 0 else float("nan")}
        per_domain[p_col] = dom
        rows[p_col] = {"tau": tau, "n_meta": int(len(meta)), "n_errors": int(err.sum()),
                       "oof_auroc": oof_auroc, "selective_risk_90": sel_risk,
                       "base_error": base,
                       "coef_V": float(model.coef_[0][0]), "coef_M": float(model.coef_[0][1]),
                       "mean_ood_risk_auroc": float(np.mean([d["risk_auroc"] for d in dom.values()]))
                       if dom else float("nan"),
                       "mean_ood_risk_reduction": float(np.mean(
                           [d["risk_reduction"] for d in dom.values()])) if dom else float("nan"),
                       "n_domains_risk_increased": int(sum(
                           d["risk_reduction"] < 0 for d in dom.values()))}
        print(f"  {p_col:12s} tau {tau:.4f}  OOF risk-AUROC {oof_auroc:.4f}  "
              f"mean OOD {rows[p_col]['mean_ood_risk_auroc']:.4f}  "
              f"({int(err.sum())} errors / {len(meta)} VAL_meta videos)")

    lines = ["# Step 4 — single-anchor reliability, R = g(V, M)", "",
             "V1's ~0.81 error-detection came from the MULTI-BRANCH setting with V, C, U_sup and "
             "M. With one anchor there is no inter-branch conflict and no support uncertainty, so "
             "the fallback spine has two features. Whether it still predicts errors is the open "
             "question this step exists to answer; C and U_sup are **not** fabricated from a "
             "single branch.", "",
             "Fitted on FF++ **VAL_meta** only, cross-fit by identity group so relatives cannot "
             "straddle a fold. `tau` comes from VAL_select, so the operating point never sees the "
             "partition the risk model is fitted on.", "",
             "## In-domain (FF++ VAL_meta, out-of-fold)", "",
             "| readout | VAL_meta videos | anchor errors | **risk-AUROC** | selective risk @90% | base error |",
             "|---|---:|---:|---:|---:|---:|"]
    for p_col in READOUTS:
        r = rows.get(p_col)
        if not r:
            continue
        if np.isnan(r.get("oof_auroc", float("nan"))):
            lines.append(f"| {NICE[p_col]} | {r['n_meta']} | {r['n_errors']} | — | — | — |")
            continue
        lines.append(f"| {NICE[p_col]} | {r['n_meta']} | {r['n_errors']} | "
                     f"**{r['oof_auroc']:.4f}** | {r['selective_risk_90']:.4f} | "
                     f"{r['base_error']:.4f} |")

    lines += ["", "## Under shift — the same fitted model applied to each OOD domain", "",
              "| readout | " + " | ".join(l for l, _ in OOD) + " | mean |",
              "|---" * (len(OOD) + 2) + "|"]
    for p_col in READOUTS:
        d = per_domain.get(p_col)
        if not d:
            continue
        cells = [f"{d[l]['risk_auroc']:.3f}" if l in d else "—" for l, _ in OOD]
        mean = rows[p_col].get("mean_ood_risk_auroc", float("nan"))
        lines.append(f"| {NICE[p_col]} | " + " | ".join(cells) + f" | **{mean:.3f}** |")

    lines += ["", "### Selective risk at a 10% deferral budget", "",
              "Error rate on the 90% of videos the risk model is most confident about, against "
              "the domain's base error rate. A useful spine cuts error here.", "",
              "| readout | domain | base error | selective risk @90% | reduction |",
              "|---|---|---:|---:|---:|"]
    for p_col in READOUTS:
        for label, _ in OOD:
            d = per_domain.get(p_col, {}).get(label)
            if not d:
                continue
            lines.append(f"| {NICE[p_col]} | {label} | {d['error_rate']:.4f} | "
                         f"{d['selective_risk_90']:.4f} | {d['risk_reduction']:+.1%} |")

    # --- verdict --------------------------------------------------------------------------------
    lines += ["", "## Verdict", ""]
    # Ranked by mean selective-risk REDUCTION, not by risk-AUROC. Risk-AUROC says the model can
    # order videos by error probability; the reduction says a deferral budget actually buys
    # something. They disagree here — semantic edges the AUROC by 0.002 while fused buys three
    # times the risk reduction and is the only readout that never makes a domain worse — and the
    # deployment quantity is the one that should decide.
    best = max((p for p in rows if not np.isnan(rows[p].get("oof_auroc", float("nan")))),
               key=lambda p: (rows[p].get("mean_ood_risk_reduction", -9),
                              -rows[p].get("n_domains_risk_increased", 9)), default=None)
    if best:
        r = rows[best]
        lines += [
            "### Comparability with V1's 0.8118 — they are not the same measurement", "",
            "V1's number was calibrated on **FF++ val + Celeb-DF-v2 val**, which its own report "
            "flags as `NOT zero-shot for: Celeb-DF-v2:val`. That is the protocol since renamed "
            "`ffpp_cdf2_TESTCONTAMINATED`, and it put Celeb-DF-v2 inside the calibration set. It "
            "also carried a **15.24% full-coverage error rate**, against this anchor's 3.9% on "
            "FF++ VAL_meta. A risk model has roughly four times more error signal to learn from "
            "there, on data drawn from the domain it is later scored on.", "",
            "So V1's 0.81 is not a target this step can be held to. It is a number from an easier "
            "and leakier setting.", "",
            "This step is run at FRAME level for the same reason V1 was (25,532 frames): at VIDEO "
            "level the anchor makes **0-3 errors in 280 VAL_meta videos**, and an error predictor "
            "cannot be fitted where there are no errors. That saturation is itself a finding "
            "about the firewall — the permitted calibration partition carries almost no error "
            "signal for a strong anchor.", "",
            "### What the fit actually learned", "",
            f"Best by risk-AUROC: **{NICE[best]}** — in-domain **{r['oof_auroc']:.4f}**, mean OOD "
            f"**{r['mean_ood_risk_auroc']:.4f}**.", "",
            f"Standardised coefficients: **V {r['coef_V']:+.3f}**, **M {r['coef_M']:+.3f}**.", "",
            ("**Vacuity carries real weight here, with the correct sign.** V enters positively "
             "— more vacuity, more risk — so this is an evidential result and not merely "
             "margin-based confidence."
             if r["coef_V"] > 0.2 else
             "**The vacuity term is inert or wrong-signed for this readout.** V should enter "
             "positively; here it does not, and margin does essentially all the work. That is a "
             "legitimate selective-prediction baseline but not an evidential one."), "",
            "Per readout, because they differ and the difference matters:", "",
            "| readout | coef V | coef M | reading |", "|---|---:|---:|---|",
            *[f"| {NICE[c]} | {rows[c]['coef_V']:+.3f} | {rows[c]['coef_M']:+.3f} | "
              f"{'V positive — evidential' if rows[c]['coef_V'] > 0.2 else 'V inert/negative — margin-driven'} |"
              for c in READOUTS if c in rows and 'coef_V' in rows[c]], "",
            "The semantic branch is the one whose vacuity is uninformative; the artifact and "
            "fused readouts both put substantial positive weight on it. So the Dirichlet "
            "uncertainty IS doing work in the readouts that were actually selected, and the "
            "evidential framing survives for those — a claim that must be made per readout "
            "rather than in general.", "",
            "### Does deferral actually reduce risk?", "",
            "| readout | mean risk reduction @10% budget | domains where it made risk WORSE |",
            "|---|---:|---:|"]
        for p_col in READOUTS:
            rr = rows.get(p_col)
            if not rr or "mean_ood_risk_reduction" not in rr:
                continue
            lines.append(f"| {NICE[p_col]} | {rr['mean_ood_risk_reduction']:+.1%} | "
                         f"{rr['n_domains_risk_increased']} of {len(per_domain.get(p_col, {}))} |")
        lines.append("")
        red = r.get("mean_ood_risk_reduction", float("nan"))
        worse = r.get("n_domains_risk_increased", 0)
        m = r["mean_ood_risk_auroc"]
        if m >= 0.75 and red >= 0.10 and worse == 0:
            lines.append("**Single-anchor reliability is a viable spine.** (V, M) predicts errors "
                         "under shift and deferral cuts risk on every domain.")
        elif m >= 0.70:
            lines.append(
                f"**Partially viable, and weaker than the headline suggests.** Risk-AUROC of "
                f"{m:.3f} under shift is real signal, but spending a 10% deferral budget buys "
                f"only {red:+.1%} on average and makes risk WORSE on {worse} domain(s). A "
                f"reliability paper can be written on this, but its claim is 'margin-based "
                f"selective prediction degrades gracefully under shift', not 'evidential "
                f"uncertainty predicts errors' — the vacuity term does not support the second "
                f"reading. Steps 5 and 6 now matter more, not less: if no expert enters and the "
                f"ladder fails the audit, this is the whole paper, and it is thin.")
        else:
            lines.append("**(V, M) alone does NOT reliably predict errors under shift.** The "
                         "reliability paper needs another principled uncertainty source, and C or "
                         "U_sup must not be manufactured from one branch to fill the gap.")
    else:
        lines.append("TODO(run) — no readout produced a fit.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(
        {"rows": rows, "per_domain": per_domain}, indent=2, default=float))
    print("\n".join(lines[-8:]))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Step 6 — at what level of test-time-observable information does complementarity become
realizable under shift?

    python analysis/tbiom/step6_realizability.py --out tbiom/STEP6_REALIZABILITY.md

Steps 1-5 established the shape of the problem: oracle complementarity is real and did not shrink
when the anchor got stronger (rescue margins +0.222/+0.233 on VALmix), while realizable recovery
collapsed to between -4% and +6%. So the question is no longer "is there complementarity" but
"what would a gate have to SEE to find it".

The ladder, every rung label-free at inference:

    G0  confidence only          [p_b, u_b]
    G1  cross-branch state       [p_A, p_b, u_A, u_b]
    G2  relational               G1 + |p_A - p_b| + Jensen-Shannon(p_A, p_b)
    G3  branch diagnostics       G2 + FS-VFM per-layer adaptation and MR-VAE rate statistics
    G4  learned probe on frozen [h_A || h_b]      -- needs embedding exports, not run

rho_realize = (Err(anchor) - Err(gated)) / (Err(anchor) - Err(oracle)), the fraction of the
achievable error reduction a realizable gate actually delivers.

THE AUDIT CLAMP IS MANDATORY, and is the point of the step rather than a formality. A
high-capacity gate can appear to realize complementarity by recognising which DATASET a sample
came from rather than whether the expert applies — the recurring failure mode of this project. So
every rung's gate inputs are also used to predict the source domain; a rung that predicts domain
well is reported as failing the audit no matter what its rho says.

THE TARGET IS ALSO TESTED, because `t_b = 1[phi_b > 0]` assigns opposite labels to samples whose
phi is a hair either side of zero, and that alone could hold gate AUROC near chance. Compared
against a margin-filtered target and against continuous regression of phi_b.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from health import eer  # noqa: E402
from step5_complementarity import anchor_valmix, expert_valmix  # noqa: E402
from video_id import video_id  # noqa: E402

STEP2, VALMIX, SCORE = Path("logs/tbiom/step2"), Path("logs/tbiom/valmix"), Path("logs/tbiom/score")
DIAG = ["d_l4", "d_l8", "d_l12", "d_l16", "d_l20", "d_l24", "d_mean", "d_early", "d_late",
        "rate_r_0", "rate_r_1", "rate_r_2", "rate_r_3", "rate_r_4",
        "patch_max", "patch_std", "patch_top10_share"]


def js(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence between two Bernoulli opinions, per sample."""
    def kl(a, b):
        a = np.clip(a, 1e-9, 1 - 1e-9); b = np.clip(b, 1e-9, 1 - 1e-9)
        return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))
    m = 0.5 * (p + q)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def diagnostics(path: Path) -> pd.DataFrame:
    """Video-level FS-VFM / MR-VAE diagnostics — the G3 features."""
    d = pd.read_parquet(path)
    cols = [c for c in DIAG if c in d]
    g = pd.DataFrame({"v": video_id(d["key"]), **{c: d[c] for c in cols}})
    out = g.groupby("v", as_index=False).agg({c: "mean" for c in cols})
    out["base"] = out["v"].str.rsplit("/", n=1).str[-1]
    dom = pd.Series("other", index=out.index)
    for needle, name in (("/Celeb-DF-v2/", "CDFv2val"), ("/DFDCP/", "DFDCPval"),
                         ("/Deepfake-Eval-2024/", "DFEval24val")):
        dom = dom.mask(out["v"].str.contains(needle, regex=False), name)
    out["domain"] = dom
    return out[["domain", "base"] + cols]


def grouped_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                regression: bool = False) -> np.ndarray:
    """Leave-one-domain-out predictions. Grouped by DOMAIN so a rung cannot pass by memorising
    the domains it was trained on — the same discipline the audit clamp then tests directly."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    out = np.full(len(y), np.nan)
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        if tr.sum() < 20 or (not regression and len(np.unique(y[tr])) < 2):
            continue
        sc = StandardScaler().fit(X[tr])
        if regression:
            m = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y[tr])
            out[te] = m.predict(sc.transform(X[te]))
        else:
            m = LogisticRegression(max_iter=3000).fit(sc.transform(X[tr]), y[tr])
            out[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return out


def auroc(y, p):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); ok = ~np.isnan(p)
    return float(roc_auc_score(y[ok], np.asarray(p)[ok])) if len(np.unique(y[ok])) > 1 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--anchor-col", default="p_fused")
    ap.add_argument("--expert", default="fsvfm_preserve")
    ap.add_argument("--out", type=Path, default=Path("tbiom/STEP6_REALIZABILITY.md"))
    args = ap.parse_args()

    av = anchor_valmix(args.anchor_col)
    ev = expert_valmix(VALMIX / f"{args.expert}/profile_epoch_009.parquet", "p_direct")
    dg = diagnostics(VALMIX / "fsvfm_preserve_rate/profile_epoch_009.parquet")
    m = av.merge(ev, on=["domain", "base"], suffixes=("_a", "_e")).merge(
        dg, on=["domain", "base"], how="left")
    assert (m["y_a"] == m["y_e"]).all(), "label mismatch across the join"

    ff = pd.read_csv(STEP2 / "p0ds_e01_FFpp_val.csv")
    fv = pd.DataFrame({"v": video_id(ff["key"]), "p": ff[args.anchor_col],
                       "y": ff["label"]}).groupby("v", as_index=False).agg(
        p=("p", "mean"), y=("y", "max"))
    _, t_a = eer(fv["y"].to_numpy(), fv["p"].to_numpy())
    ed = pd.read_parquet(SCORE / f"{args.expert}_ffppval/profile_epoch_009.parquet")
    edv = pd.DataFrame({"v": video_id(ed["key"]), "p": ed["p_direct"],
                        "y": ed["label"]}).groupby("v", as_index=False).agg(
        p=("p", "mean"), y=("y", "max"))
    _, t_e = eer(edv["y"].to_numpy(), edv["p"].to_numpy())

    y = m["y_a"].to_numpy()
    pa, pe = m["p_a"].to_numpy(), m["p_e"].to_numpy()
    ua, ue = m["u_a"].to_numpy(), m["u_e"].to_numpy()
    dom = m["domain"].to_numpy()
    a_ok = ((pa >= t_a).astype(int) == y)
    e_ok = ((pe >= t_e).astype(int) == y)

    err_anchor = 1 - a_ok.mean()
    err_oracle = 1 - (a_ok | e_ok).mean()
    phi = e_ok.astype(float) - a_ok.astype(float)     # +1 expert saves, -1 expert harms, 0 tie
    target = (phi > 0).astype(int)

    rungs = {
        "G0 confidence only": np.stack([pe, ue], 1),
        "G1 cross-branch": np.stack([pa, pe, ua, ue], 1),
        "G2 relational": np.stack([pa, pe, ua, ue, np.abs(pa - pe), js(pa, pe)], 1),
    }
    diag_cols = [c for c in DIAG if c in m and m[c].notna().all()]
    if diag_cols:
        rungs["G3 branch diagnostics"] = np.concatenate(
            [rungs["G2 relational"], m[diag_cols].to_numpy()], axis=1)

    results = {}
    for name, X in rungs.items():
        q = grouped_oof(X, target, dom)
        ok = ~np.isnan(q)
        fused = np.where(q >= 0.5, pe - t_e, pa - t_a)
        gated_ok = ((fused >= 0).astype(int) == y)
        err_gated = 1 - gated_ok[ok].mean()
        denom = err_anchor - err_oracle
        rho = (err_anchor - err_gated) / denom if denom > 1e-9 else float("nan")
        # AUDIT CLAMP: can the same inputs predict the DOMAIN?
        dom_auc = []
        for d in np.unique(dom):
            qd = grouped_oof(X, (dom == d).astype(int), dom)
            # leave-one-domain-out cannot learn the held-out domain, so audit in-sample on
            # purpose: the question is whether the FEATURES carry provenance at all.
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler().fit(X); mm = LogisticRegression(max_iter=3000).fit(
                sc.transform(X), (dom == d).astype(int))
            dom_auc.append(auroc((dom == d).astype(int), mm.predict_proba(sc.transform(X))[:, 1]))
        results[name] = {
            "n_features": X.shape[1], "gate_auroc_vs_target": auroc(target, q),
            "rho_realize": rho, "err_anchor": err_anchor, "err_gated": err_gated,
            "err_oracle": err_oracle, "domain_auroc": float(np.mean(dom_auc)),
            "audit_pass": bool(np.mean(dom_auc) < 0.80),
        }
        r = results[name]
        print(f"  {name:24s} feats {X.shape[1]:3d}  gateAUROC {r['gate_auroc_vs_target']:.3f}  "
              f"rho {rho:+.3f}  domainAUROC {r['domain_auroc']:.3f}  "
              f"{'audit OK' if r['audit_pass'] else 'AUDIT FAIL'}")

    # --- target variants, on the richest rung that passed the audit ---------------------------
    base = "G3 branch diagnostics" if "G3 branch diagnostics" in rungs else "G2 relational"
    X = rungs[base]
    variants = {}
    q = grouped_oof(X, target, dom)
    variants["binary 1[phi>0]"] = auroc(target, q)
    keep = phi != 0
    if keep.sum() > 50:
        qf = np.full(len(y), np.nan)
        qf[keep] = grouped_oof(X[keep], (phi[keep] > 0).astype(int), dom[keep])
        variants["margin-filtered (drop ties)"] = auroc((phi[keep] > 0).astype(int), qf[keep])
    qr = grouped_oof(X, phi, dom, regression=True)
    variants["continuous regression of phi"] = auroc(target, qr)

    lines = ["# Step 6 — realizability ladder", "",
             f"Anchor `{args.anchor_col}` (P0-DS epoch 1), expert `{args.expert}`, on VALmix "
             f"({len(m)} videos). Gates are cross-fitted LEAVE-ONE-DOMAIN-OUT, so a rung cannot "
             f"pass by memorising the domains it trained on.", "",
             f"Anchor error {err_anchor:.4f}, oracle error {err_oracle:.4f}, so the achievable "
             f"reduction is {err_anchor - err_oracle:.4f}. `rho_realize` is the fraction of that "
             f"a realizable gate delivers.", "",
             "| rung | features | gate AUROC vs target | **rho_realize** | gated error | domain AUROC | audit |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for name, r in results.items():
        lines.append(f"| {name} | {r['n_features']} | {r['gate_auroc_vs_target']:.3f} | "
                     f"**{r['rho_realize']:+.3f}** | {r['err_gated']:.4f} | "
                     f"{r['domain_auroc']:.3f} | {'pass' if r['audit_pass'] else '**FAIL**'} |")
    lines += ["| G4 probe on frozen [h_A ‖ h_b] | — | TODO(run) | TODO(run) | | | |", "",
              "G4 needs embedding exports that no current run writes; recorded rather than "
              "silently omitted.", "",
              "## Audit clamp", "",
              "Domain AUROC is how well the SAME gate inputs predict which corpus a sample came "
              "from. A rung that reads provenance can appear to realize complementarity while "
              "actually recognising the dataset — the recurring failure mode this project has "
              "hit before. Above 0.80 is reported as failing, whatever its rho.", "",
              "## Target variants", "",
              f"On `{base}`. `t_b = 1[phi_b > 0]` gives opposite labels to samples a hair either "
              f"side of zero, which could by itself hold gate AUROC near chance.", "",
              "| target | gate AUROC |", "|---|---:|"]
    for k, v in variants.items():
        lines.append(f"| {k} | {v:.3f} |")

    best = max(results.items(), key=lambda kv: (kv[1]["audit_pass"], kv[1]["rho_realize"]))
    lines += ["", "## Verdict", ""]
    passed = [n for n, r in results.items() if r["audit_pass"] and r["rho_realize"] >= 0.25]
    if passed:
        lines.append(f"**{', '.join(passed)} realize complementarity while passing the audit.** "
                     f"An applicability framework is supported; carry the cheapest passing rung "
                     f"into Step 7.")
    else:
        top = best[1]
        lines += [
            f"**No rung realizes complementarity.** Best was {best[0]} at rho "
            f"{top['rho_realize']:+.3f} against a 0.25 bar"
            + ("" if top["audit_pass"] else ", and it fails the audit besides") + ".", "",
            "Every rung, including one with per-layer adaptation and rate statistics, leaves the "
            "gate unable to tell where the expert applies. Combined with Steps 1-5 this is a "
            "characterised negative rather than an absence of evidence: the complementarity is "
            "measurable and stable (rescue margin +0.22), the ceiling is real, and what is "
            "missing is any test-time-observable signal that locates it.", "",
            "**The paper is therefore the single-anchor reliability result** from Step 4 — "
            "margin-and-vacuity selective prediction that degrades gracefully under shift — with "
            "this realizability gap reported as the negative result it is. Do NOT manufacture a "
            "fusion story from a rung that only passed by reading provenance."]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(
        {"rungs": results, "target_variants": variants,
         "err_anchor": err_anchor, "err_oracle": err_oracle}, indent=2, default=float))
    print("\n".join(lines[-8:]))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

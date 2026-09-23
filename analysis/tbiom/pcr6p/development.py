#!/usr/bin/env python3
"""Sections 4-6 on development: the fixed matrix, the frozen uncertainty scores, and the acceptance criteria.

    OMP_NUM_THREADS=8 python analysis/tbiom/pcr6p/development.py

Four detector configurations, each fitting its reliability weights on its own development exports. The conflict-by-
source score is a logistic model on at most five inputs, fitted on the fitting fold only and applied to the selection
fold, so it never sees the data it is scored on. Paired source-group bootstrap, 2000 draws, materiality floor 1e-4.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "grouped"), str(HERE.parent), str(HERE.parent / "temporal")]
import arms as AR, ops_pcr6p as P  # noqa: E402
import score as S  # noqa: E402
import exp1_fusion_audit as A  # noqa: E402
import dsr_system as D  # noqa: E402
import confirm as C  # noqa: E402
from safeguard_ablation import rc_metrics  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

R, N_BOOT, TOL = Path("tbiom/reports"), 2000, 1e-4
CONF = {"42": ("warmgate_e19", "vjepa2_lora_B_seed42"), "T1337": ("warmgate_e19", "vjepa2_lora_B_seed1337"),
        "armB42": ("sbiarmB_e08", "vjepa2_lora_B_seed42"),
        "armB1337": ("sbiarmBs1337_e03", "vjepa2_lora_B_seed1337")}
PROC = ("jpeg40", "rs06")
SIMPLE = ("vacuity", "conflict_K", "entropy", "margin")


def cbs_fit(Fd, w):
    """Conflict-by-source: logistic on [vacuity, K_TS, K_AO, K_all, margin], fitted on the fitting fold only."""
    p, f, sc, att = AR.scores(Fd, "PCR6P-R", w)
    X = np.stack([f[:, 2], att["K_TS"], att["K_AO"], att["K_all"], -np.abs(p - 0.5)], 1)
    err = ((p >= AR.tau_of(p, Fd)).astype(int) != Fd.y.to_numpy()).astype(int)
    if err.min() == err.max():
        return None
    s = StandardScaler().fit(X)
    m = LogisticRegression(max_iter=2000).fit(s.transform(X), err)
    return s, m


def cbs_apply(model, t, w):
    if model is None:
        return np.full(len(t), np.nan)
    s, m = model
    p, f, _sc, att = AR.scores(t, "PCR6P-R", w)
    X = np.stack([f[:, 2], att["K_TS"], att["K_AO"], att["K_all"], -np.abs(p - 0.5)], 1)
    return m.predict_proba(s.transform(X))[:, 1]


def main():
    t0 = time.time()
    out = {}
    for tag, (sp, br) in CONF.items():
        dev = C.dev_table(sp, br)
        dproc = {c: C.attach(A.dev_tables({n: Path("logs/tbiom/stage2") / f"p_dev_{sp}_{n}_{c}.csv" for n in C.DEV_SETS}, [1.0]),
                             [C.BR / br / "opinions" / f"{n}_{c}.csv" for n in C.DEV_SETS])
                 for c in PROC if all((Path("logs/tbiom/stage2") / f"p_dev_{sp}_{n}_{c}.csv").exists() for n in C.DEV_SETS)}
        rows = []
        for k, Fd, Sd in D.splits(dev):
            clF, clS = A.cells(Fd), A.cells(Sd)
            row = {"split": k, "arms": {}, "w": {}, "unc": {}, "cases": {}}
            keep = {}
            for arm in AR.ARMS:
                w = AR.fit_w(Fd, arm, clF)
                row["w"][arm] = list(w)
                p, f, sc, att = AR.scores(Sd, arm, w)
                tau = AR.tau_of(p, Sd)
                y = Sd.y.to_numpy(); pred = (p >= tau).astype(int); err = pred != y
                pc = np.clip(p, 1e-6, 1 - 1e-6)
                tot, ty = A.v31(p, clS)
                r = {"v31": tot, **ty, "tau": tau, "auroc": A.fast_auc(p[y == 0], p[y == 1]),
                     "fpr": float((pred[y == 0] == 1).mean()), "tpr": float((pred[y == 1] == 1).mean()),
                     "nll": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
                     "brier": float(np.mean((p - y) ** 2)), "ece": S.ece(p, y),
                     "u_mean": float(f[:, 2].mean()), "K_mean": float(att["K"].mean()),
                     "zero_w": [int(x == 0.0) for x in w]}
                for c, Pt in dproc.items():
                    pt = Pt.set_index("v").loc[Sd.v].reset_index(); pt["source"] = Sd.source.to_numpy()
                    r[c] = A.v31(P.p_of(AR.fuse(pt, arm, w)), A.cells(pt))[0]
                row["arms"][arm] = r
                keep[arm] = (p, err, sc, att, w)
            # frozen uncertainty scores + the conflict-by-source model
            model = cbs_fit(Fd, row["w"]["PCR6P-R"])
            for arm in AR.ARMS:
                p, err, sc, att, w = keep[arm]
                u = {}
                for nm, sig in sc.items():
                    m = rc_metrics(sig, err)
                    u[nm] = {"err_auroc": float(m["auroc_err"]), "eaurc": float(m["eaurc"]),
                             "aurc": float(m["aurc"]), "risk@90": float(m["risk@90"]), "risk@80": float(m["risk@80"])}
                if arm == "PCR6P-R":
                    cs = cbs_apply(model, Sd, w)
                    m = rc_metrics(cs, err)
                    u["conflict_by_source"] = {"err_auroc": float(m["auroc_err"]), "eaurc": float(m["eaurc"]),
                                               "aurc": float(m["aurc"]), "risk@90": float(m["risk@90"]),
                                               "risk@80": float(m["risk@80"])}
                    row["cbs_coef"] = (None if model is None else
                                       dict(zip(["vacuity", "K_TS", "K_AO", "K_all", "margin"],
                                                model[1].coef_[0].round(4).tolist())))
                    row["unc_sig"] = {"scores": {k: sc[k] for k in SIMPLE}, "cbs": cs, "err": err}
                row["unc"][arm] = u
            row["cases"] = case_split(Sd, keep["PCR6P-R"])
            rows.append(boot(row, Sd, keep, model))
            print(f"  {tag} split {k} ({time.time() - t0:.0f}s)", flush=True)
        out[tag] = rows
    json.dump(strip(out), open(R / "EXP_PCR6P_DEVELOPMENT.json", "w"), indent=1, default=float)
    write(out, time.time() - t0)


def case_split(Sd, kept):
    """Section 7's conflict case analysis, on development."""
    p, err, _sc, att, _w = kept
    y = Sd.y.to_numpy()
    tau = AR.tau_of(p, Sd)
    pv, ok = {}, {}
    for v in AR.VIEWS:
        q = Sd[f"p_{v}"].to_numpy()
        ff = np.where(Sd.source.to_numpy() == "FFpp_val")[0]
        from health import eer as _e
        t = float(_e(y[ff], q[ff])[1])
        pv[v] = q; ok[v] = (q >= t).astype(int) == y
    sp_ok = ok["APP"] & ok["OM"]; sp_bad = (~ok["APP"]) & (~ok["OM"])
    cases = {"temporal right, both spatial wrong": ok["T"] & sp_bad,
             "temporal wrong, both spatial right": (~ok["T"]) & sp_ok,
             "all three agree and right": ok["T"] & sp_ok,
             "all three wrong": (~ok["T"]) & sp_bad}
    out = {}
    for nm, m in cases.items():
        out[nm] = {"n": int(m.sum()),
                   "K_TS": float(att["K_TS"][m].mean()) if m.sum() else float("nan"),
                   "K_AO": float(att["K_AO"][m].mean()) if m.sum() else float("nan"),
                   "K_all": float(att["K_all"][m].mean()) if m.sum() else float("nan"),
                   "fused_correct": float((~err[m]).mean()) if m.sum() else float("nan")}
    return out


def boot(row, Sd, keep, model):
    g = A.groups_for(Sd)
    draws = S.group_boot(g, N_BOOT, 20260923 + row["split"])
    clS = A.cells(Sd)
    W = np.stack(draws).astype(float)
    row["boot"] = {}
    # detection: every arm against Dempster-R
    base = A.v31
    ref = np.array([base(keep["Dempster-R"][0], clS, d)[0] for d in draws])
    for arm in AR.ARMS:
        v = np.array([base(keep[arm][0], clS, d)[0] for d in draws]) - ref
        row["boot"][f"v31|{arm}"] = q(v)
    # uncertainty: conflict-by-source against each simple score, error-detection AUROC and E-AURC
    p, err, sc, att, w = keep["PCR6P-R"]
    cbs = row["unc_sig"]["cbs"]
    ok = np.isfinite(cbs)
    for nm in SIMPLE:
        da = np.array([C.auc_weighted(cbs, err.astype(int), W[i:i + 1])[0]
                       - C.auc_weighted(sc[nm], err.astype(int), W[i:i + 1])[0] for i in range(N_BOOT)]) if ok.all() \
            else np.array([np.nan])
        row["boot"][f"err_auroc|cbs-{nm}"] = q(da)
    row.pop("unc_sig")
    return row


def q(d):
    d = np.asarray(d, float)
    return {"delta": float(np.nanmean(d)), "ci": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]}


def strip(out):
    return out


def write(out, secs):
    m = lambda tag, arm, key: float(np.mean([r["arms"][arm][key] for r in out[tag]]))
    u = lambda tag, arm, s, key: float(np.mean([r["unc"][arm][s][key] for r in out[tag] if s in r["unc"][arm]]))
    L = ["# PCR6+ on development", "", __doc__.split("\n\n", 2)[2].strip(), "",
         f"Wall clock {secs / 60:.1f} minutes, CPU. Pre-registration: `tbiom/configs/pcr6p_fusion_preregister.yaml`.",
         ""]
    for tag in out:
        L += [f"## Configuration `{tag}`", "",
              "| arm | v3.1 clean | swap | reenact | in-the-wild | JPEG q40 | resize 0.60 | FPR | TPR | NLL | Brier | "
              "ECE | Δ v3.1 vs Dempster-R (95% CI) | folds with T/APP/OM weight 0 |", "|---" * 14 + "|"]
        for a in AR.ARMS:
            b = [r["boot"][f"v31|{a}"] for r in out[tag]]
            dl, lo, hi = np.mean([x["delta"] for x in b]), np.mean([x["ci"][0] for x in b]), np.mean([x["ci"][1] for x in b])
            z = np.sum([r["arms"][a]["zero_w"] for r in out[tag]], 0)
            tagm = "**" if a == "PCR6P-R" else ""
            note = " *(baseline)*" if a in ("Dempster-R", "Averaging-R") else ""
            L.append(f"| {tagm}{a}{tagm}{note} | {m(tag, a, 'v31'):.4f} | {m(tag, a, 'swap'):.4f} | "
                     f"{m(tag, a, 'reenactment'):.4f} | {m(tag, a, 'in_the_wild'):.4f} | "
                     + " | ".join(f"{m(tag, a, c):.4f}" if c in out[tag][0]["arms"][a] else "n/a" for c in PROC)
                     + f" | {m(tag, a, 'fpr'):.3f} | {m(tag, a, 'tpr'):.3f} | {m(tag, a, 'nll'):.4f} | "
                     f"{m(tag, a, 'brier'):.4f} | {m(tag, a, 'ece'):.4f} | {dl:+.4f} [{lo:+.4f}, {hi:+.4f}] | "
                     f"{z[0]}/{z[1]}/{z[2]} |")
        L += ["", f"### Uncertainty scores, configuration `{tag}`", "",
              "| score | error-detection AUROC | E-AURC | risk@90 | risk@80 | Δ error-AUROC vs conflict-by-source (95% CI) |",
              "|---|---:|---:|---:|---:|---|"]
        for s in list(SIMPLE) + ["conflict_by_source"]:
            if s == "conflict_by_source":
                L.append(f"| **{s}** | {u(tag, 'PCR6P-R', s, 'err_auroc'):.4f} | {u(tag, 'PCR6P-R', s, 'eaurc'):.5f} | "
                         f"{u(tag, 'PCR6P-R', s, 'risk@90'):.4f} | {u(tag, 'PCR6P-R', s, 'risk@80'):.4f} | — |")
            else:
                b = [r["boot"][f"err_auroc|cbs-{s}"] for r in out[tag]]
                dl = np.mean([x["delta"] for x in b]); lo = np.mean([x["ci"][0] for x in b]); hi = np.mean([x["ci"][1] for x in b])
                nn = sum(x["ci"][0] > TOL for x in b)
                L.append(f"| {s} | {u(tag, 'PCR6P-R', s, 'err_auroc'):.4f} | {u(tag, 'PCR6P-R', s, 'eaurc'):.5f} | "
                         f"{u(tag, 'PCR6P-R', s, 'risk@90'):.4f} | {u(tag, 'PCR6P-R', s, 'risk@80'):.4f} | "
                         f"{dl:+.4f} [{lo:+.4f}, {hi:+.4f}] ({nn}/5 splits above the floor) |")
        co = [r.get("cbs_coef") for r in out[tag] if r.get("cbs_coef")]
        if co:
            ks = list(co[0])
            L += ["", "Conflict-by-source coefficients (standardised inputs, mean over folds): "
                  + ", ".join(f"{k} {np.mean([c[k] for c in co]):+.3f}" for k in ks) + ".", ""]
        L += ["", f"### Conflict case analysis, configuration `{tag}`", "",
              "| case | videos | K_TS | K_AO | K_all | PCR6+-R correct |", "|---|---:|---:|---:|---:|---:|"]
        for nm in out[tag][0]["cases"]:
            g = lambda k: float(np.nanmean([r["cases"][nm][k] for r in out[tag]]))
            L.append(f"| {nm} | {g('n'):.0f} | {g('K_TS'):.4f} | {g('K_AO'):.4f} | {g('K_all'):.4f} | "
                     f"{g('fused_correct'):.4f} |")
        L.append("")
    (R / "EXP_PCR6P_DEVELOPMENT.md").write_text("\n".join(L) + "\n")
    print("wrote EXP_PCR6P_DEVELOPMENT.md")


if __name__ == "__main__":
    main()

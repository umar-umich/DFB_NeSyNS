#!/usr/bin/env python3
"""Uncertainty safeguard and three-view ablation for the deployed fusion (decision-2026-09-17.md, items 1 and 2).

    OMP_NUM_THREADS=8 python analysis/tbiom/temporal/safeguard_ablation.py [--branch vjepa2_lora_seed42] [--tag seed42]

Cached outputs only (no GPU). Views: T = V-JEPA 2 LoRA branch (development-selected epoch), OM = Off-manifold residual,
FP = Bona-fide face prior (raw), both from the development-selected warm+gate detector (epoch 19).
DEPLOYED FUSION = logistic stacking on matched inputs (p and vacuity u of T, OM, FP), C = 1, balanced classes, fitted on
the fusion audit's source-grouped FITTING folds (5 splits, identical to every earlier audit); evaluated on the SELECTION
folds. tau = EER threshold of the stacked score on the selection fold's FF++ val videos, frozen for everything else.

ITEM 2 — THREE-VIEW ABLATION. T+OM+FP vs T+OM, T+FP, OM+FP, each refitted with the same procedure. Paired source-group
bootstrap (1000 draws per split) of Delta = triple - pair on the v3.1 objective.
  Redundancy rule, fixed before the run: branch X is REDUNDANT beside the others if removing it costs
  mean Delta < +0.005 on clean development AND the 95% CI upper bound < +0.005 in >= 4/5 splits AND the mean loss on
  each processed development condition is <= 0.005. Otherwise X earns its place.

ITEM 1 — UNCERTAINTY SAFEGUARD. DS vacuity is NOT the logistic system's uncertainty and is never attached to its output.
Error-ranking signals scored against the deployed logistic decision (score >= tau):
  margin      distance of the stacked logit from the logit of tau (small = suspicious)
  u_T, u_OM, u_FP, u_max, u_mean   per-branch Dirichlet vacuities (large = suspicious)
  disagree_std, disagree_range     spread of the three branch probabilities
  conflict_max                     largest pairwise DS conflict mass between branches
Metrics: error-detection AUROC; AURC and E-AURC of the risk-coverage curve (reject most suspicious first); risk at 90% and
80% coverage. Confident false negatives: fakes the system calls genuine; FN-vs-TN AUROC among genuine calls, and the share
of FNs whose margin exceeds the median margin of correct genuine calls. Clean and processed reported separately.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import exp1_fusion_audit as A  # noqa: E402
from health import eer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

S23, S2, BR = Path("logs/tbiom/stage23"), Path("logs/tbiom/stage2"), Path("logs/tbiom/temporal/branches")
DEV = ("VALmix", "VALmixV2", "FFpp_val"); PROC = ("jpeg40", "rs06")
REN = {"p_artifact": "p_OM", "u_artifact": "u_OM", "p_fsvfm_raw": "p_FP", "u_fsvfm_raw": "u_FP"}
COMBOS = {"T+OM+FP": ["T", "OM", "FP"], "T+OM": ["T", "OM"], "T+FP": ["T", "FP"], "OM+FP": ["OM", "FP"]}
SIGNALS = ["margin", "u_T", "u_OM", "u_FP", "u_max", "u_mean", "disagree_std", "disagree_range", "conflict_max"]
N_BOOT, MARGIN = 1000, 0.005


def attach(tab, op_files):
    o = pd.concat([pd.read_csv(f) for f in op_files]).drop_duplicates("v").set_index("v")
    idx = o.index.get_indexer(tab.v); assert (idx >= 0).all(), f"{(idx < 0).sum()} videos without branch opinions"
    tab = tab.rename(columns=REN).copy(); tab["p_T"], tab["u_T"] = o.p.to_numpy()[idx], o.u.to_numpy()[idx]
    return tab


def fit(F, views):
    cols = [f"{k}_{v}" for v in views for k in "pu"]
    sc = StandardScaler().fit(F[cols].to_numpy())
    return cols, sc, LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(sc.transform(F[cols].to_numpy()), F.y.to_numpy())


def score(model, t):
    cols, sc, lr = model
    return lr.predict_proba(sc.transform(t[cols].to_numpy()))[:, 1]


def signals(t, s, tau):
    lg = lambda x: np.log(np.clip(x, 1e-9, 1 - 1e-9) / np.clip(1 - x, 1e-9, 1))
    P = np.stack([t.p_T, t.p_OM, t.p_FP], 1); U = np.stack([t.u_T, t.u_OM, t.u_FP], 1)
    conf = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        Sa, Sb = 2 / U[:, a], 2 / U[:, b]
        ba1, ba0, bb1, bb0 = P[:, a] - U[:, a] / 2, (1 - P[:, a]) - U[:, a] / 2, P[:, b] - U[:, b] / 2, (1 - P[:, b]) - U[:, b] / 2
        conf.append(np.clip(ba1, 0, 1) * np.clip(bb0, 0, 1) + np.clip(ba0, 0, 1) * np.clip(bb1, 0, 1))
    return {"margin": -np.abs(lg(s) - lg(np.array(tau))), "u_T": U[:, 0], "u_OM": U[:, 1], "u_FP": U[:, 2],
            "u_max": U.max(1), "u_mean": U.mean(1), "disagree_std": P.std(1), "disagree_range": P.max(1) - P.min(1),
            "conflict_max": np.max(conf, 0)}


def rc_metrics(sig, err):
    order = np.argsort(sig, kind="stable")                    # most confident first (lowest suspicion)
    e = err[order].astype(float); n = len(e); risk = np.cumsum(e) / np.arange(1, n + 1)
    aurc = risk.mean(); k = int(e.sum())
    opt = np.concatenate([np.zeros(n - k), np.ones(k)]); opt_risk = np.cumsum(opt) / np.arange(1, n + 1)
    at = lambda c: risk[max(0, int(np.ceil(c * n)) - 1)]
    return {"auroc_err": A.fast_auc(sig[~err], sig[err]) if 0 < k < n else np.nan, "aurc": aurc, "eaurc": aurc - opt_risk.mean(),
            "risk@90": at(0.9), "risk@80": at(0.8), "err_rate": e.mean(), "n": n}


def fn_metrics(sig, y, pred):
    called_real = pred == 0; fn = called_real & (y == 1); tn = called_real & (y == 0)
    if fn.sum() == 0 or tn.sum() == 0:
        return {"fn": int(fn.sum()), "auroc_fn_vs_tn": np.nan}
    return {"fn": int(fn.sum()), "tn": int(tn.sum()), "auroc_fn_vs_tn": A.fast_auc(sig[tn], sig[fn])}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--branch", default="vjepa2_lora_seed42"); ap.add_argument("--tag", default="seed42")
    a = ap.parse_args(); ob = BR / a.branch / "opinions"
    dev = attach(A.dev_tables({n: S23 / f"warmgate_e19_{n}.csv" for n in DEV}, [1.0]), [ob / f"{n}.csv" for n in DEV])
    dproc = {c: attach(A.dev_tables({n: S2 / f"p_dev_warmgate_e19_{n}_{c}.csv" for n in DEV}, [1.0]), [ob / f"{n}_{c}.csv" for n in DEV])
             for c in PROC}
    tests = {}
    for t in A.TEST:
        d = A.load_frames(S23 / f"warmgate_sel_{t}.csv")
        keep = ~d.subset.str.contains("IP_LAP") & (d.subset != "fake-audio-driven") & ~d.subset.str.startswith("adv-")
        tests[t] = attach(A.video_table(d[keep].reset_index(drop=True), [1.0]), [ob / f"{t}.csv"])
    for t in ("CDFv2", "DFDCP"):
        for c in PROC:
            tests[f"{t} {c}"] = attach(A.video_table(A.load_frames(S2 / f"p_v3_warmgate_{t}_{c}.csv"), [1.0]), [ob / f"{t}_{c}.csv"])
    grp = A.groups_for(dev.rename(columns={v: k for k, v in REN.items()})); rng = np.random.default_rng(12345)

    abl, unc = [], []
    for k in range(A.N_SPLITS):
        fi, si = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=k).split(dev, dev.y, grp))
        Fd, Sd = dev.iloc[fi].reset_index(drop=True), dev.iloc[si].reset_index(drop=True); gS = grp[si]
        models = {c: fit(Fd, v) for c, v in COMBOS.items()}
        clS = A.cells(Sd); sc = {c: score(m, Sd) for c, m in models.items()}
        sel = {c: A.v31(v, clS) for c, v in sc.items()}
        ug = np.unique(gS); pos = {g: np.where(gS == g)[0] for g in ug}; deltas = {c: [] for c in COMBOS if c != "T+OM+FP"}
        for _ in range(N_BOOT):
            w = np.zeros(len(Sd), dtype=int)
            for g in rng.choice(ug, len(ug), replace=True):
                w[pos[g]] += 1
            base = A.v31(sc["T+OM+FP"], clS, w)[0]
            for c in deltas:
                deltas[c].append(base - A.v31(sc[c], clS, w)[0])
        boot = {c: {"delta": sel["T+OM+FP"][0] - sel[c][0], "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
                for c, d in deltas.items()}
        proc = {}
        for c, P in dproc.items():
            Pt = P.set_index("v").loc[Sd.v].reset_index(); Pt["source"] = Sd.source.to_numpy()
            proc[c] = {cc: A.v31(score(m, Pt), A.cells(Pt)) for cc, m in models.items()}
        ffs = np.where(Sd.source == "FFpp_val")[0]
        taus = {c: eer(Sd.y.to_numpy()[ffs], v[ffs])[1] for c, v in sc.items()}
        test_auc = {t: {c: A.fast_auc(score(m, T)[T.y.to_numpy() == 0], score(m, T)[T.y.to_numpy() == 1]) for c, m in models.items()}
                    for t, T in tests.items()}
        abl.append({"split": k, "sel": sel, "boot": boot, "proc": proc, "tests": test_auc})

        # ---- uncertainty safeguard on the deployed triple
        m, tau = models["T+OM+FP"], taus["T+OM+FP"]
        def evaluate(t, cells_fn=None):
            s = score(m, t); y = t.y.to_numpy(); pred = (s >= tau).astype(int); err = pred != y; sg = signals(t, s, tau)
            return {"acc": float(1 - err.mean()), "fpr": float((pred[y == 0] == 1).mean()), "fnr": float((pred[y == 1] == 0).mean()),
                    "signals": {n: {**rc_metrics(v, err), **fn_metrics(v, y, pred)} for n, v in sg.items()},
                    "confident_fn_share": float(np.mean(sg["margin"][(pred == 0) & (y == 1)] < np.median(sg["margin"][(pred == 0) & (y == 0)])))
                    if ((pred == 0) & (y == 1)).any() and ((pred == 0) & (y == 0)).any() else np.nan}
        u = {"clean dev": evaluate(Sd)}
        for c, P in dproc.items():
            Pt = P.set_index("v").loc[Sd.v].reset_index(); u[f"{c} dev"] = evaluate(Pt)
        u["clean tests (10 sets pooled)"] = evaluate(pd.concat([tests[t] for t in A.TEST], ignore_index=True))
        for t in ("CDFv2", "DFDCP"):
            u[f"{t} clean"] = evaluate(tests[t])
            for c in PROC:
                u[f"{t} {c}"] = evaluate(tests[f"{t} {c}"])
        unc.append(u)
        print(f"split {k}: triple {sel['T+OM+FP'][0]:.4f} | " + " ".join(f"{c} {sel[c][0]:.4f}" for c in COMBOS if c != "T+OM+FP"), flush=True)

    out = Path("tbiom/reports"); json.dump({"ablation": abl, "uncertainty": unc}, open(out / f"EXP6_7_SAFEGUARD_ABLATION_{a.tag}.json", "w"), indent=1, default=float)
    write_ablation(out / f"EXP7_THREEVIEW_ABLATION_{a.tag}.md", abl, a.branch)
    write_uncertainty(out / f"EXP6_UNCERTAINTY_SAFEGUARD_{a.tag}.md", unc, a.branch)


def write_ablation(path, abl, branch):
    pm = lambda v: f"{np.mean(v):.4f} ±{np.std(v):.4f}"
    L = [f"# Experiment 7 — three-view ablation beside the temporal branch (`{branch}`)", "",
         "Logistic stacking on matched inputs (p, vacuity), fitted on the fusion audit's source-grouped fitting folds, compared on "
         "the selection folds (5 splits). T = V-JEPA 2 LoRA, OM = Off-manifold residual, FP = Bona-fide face prior (warm+gate "
         "epoch 19). Paired source-group bootstrap, 1000 draws per split. Script `analysis/tbiom/temporal/safeguard_ablation.py`.", "",
         "**Redundancy rule (fixed before the run):** branch X is redundant if removing it costs mean Δ < +0.005 on clean development "
         "AND the 95% CI upper bound < +0.005 in ≥ 4/5 splits AND ≤ 0.005 mean loss on each processed development condition.", "",
         "## 1. Development (selection folds), mean ± sd over splits", "",
         "| stack | clean v3.1 | swap | reenactment | in-the-wild | jpeg40 v3.1 | rs06 v3.1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for c in COMBOS:
        L.append(f"| {c} | {pm([s['sel'][c][0] for s in abl])} | " + " | ".join(pm([s["sel"][c][1][t] for s in abl]) for t in abl[0]["sel"][c][1]) +
                 f" | {pm([s['proc']['jpeg40'][c][0] for s in abl])} | {pm([s['proc']['rs06'][c][0] for s in abl])} |")
    L += ["", "## 2. Removing one branch — Δ = triple − pair (clean development), verdicts", "",
          "| removed branch | pair | mean Δ | CI upper < +0.005 splits | CI lower > 0 splits | Δ jpeg40 | Δ rs06 | verdict |", "|---|---|---:|---:|---:|---:|---:|---|"]
    removed = {"T+OM": "FP", "T+FP": "OM", "OM+FP": "T"}
    for pair, x in removed.items():
        d = [s["boot"][pair]["delta"] for s in abl]; hi = sum(s["boot"][pair]["ci"][1] < MARGIN for s in abl)
        lo = sum(s["boot"][pair]["ci"][0] > 0 for s in abl)
        dp = {c: np.mean([s["proc"][c]["T+OM+FP"][0] - s["proc"][c][pair][0] for s in abl]) for c in PROC}
        red = np.mean(d) < MARGIN and hi >= 4 and all(v <= MARGIN for v in dp.values())
        L.append(f"| {x} | {pair} | {np.mean(d):+.4f} | {hi}/5 | {lo}/5 | {dp['jpeg40']:+.4f} | {dp['rs06']:+.4f} | "
                 f"{'**REDUNDANT**' if red else 'earns its place'} |")
    L += ["", "Per split Δ [95% CI]: " + "; ".join(f"{pair}: " + ", ".join(f"{s['boot'][pair]['delta']:+.4f} [{s['boot'][pair]['ci'][0]:+.4f}, {s['boot'][pair]['ci'][1]:+.4f}]" for s in abl) for pair in removed), "",
          "## 3. Test sets — video AUROC (read after; mean over the 5 fitted splits)", "", "| test | " + " | ".join(COMBOS) + " |", "|---|" + "---:|" * len(COMBOS)]
    tests = list(abl[0]["tests"])
    for t in tests:
        L.append(f"| {t} | " + " | ".join(f"{np.mean([s['tests'][t][c] for s in abl]):.4f}" for c in COMBOS) + " |")
    L.append("| **10-set mean** | " + " | ".join(f"**{np.mean([np.mean([s['tests'][t][c] for s in abl]) for t in A.TEST]):.4f}**" for c in COMBOS) + " |")
    L += ["", "Conditional on one development-selected checkpoint per branch (one detector seed; one temporal-branch seed). CDFv2 and "
          "DeepfakeEval2024 have informed development. FaceShifter is sealed."]
    path.write_text("\n".join(L) + "\n")


def write_uncertainty(path, unc, branch):
    L = [f"# Experiment 6 — uncertainty safeguard for the deployed logistic fusion (`{branch}` + OM + FP)", "",
         "The deployed fusion is logistic stacking, so DS vacuity is **not** its uncertainty and is not attached to its output. The "
         "signals below are compared as ERROR RANKERS for the logistic decision (score ≥ τ, τ = EER on selection-fold FF++ val). "
         "Higher error-AUROC and lower AURC / E-AURC are better; risk@c = error rate on the c most confident fraction. Clean and "
         "processed conditions are separate; processed development uses the same selection-fold videos. Mean over 5 grouped splits. "
         "Goal: useful error identification — not making vacuity beat margin.", ""]
    for cond in unc[0]:
        u0 = [u[cond] for u in unc]
        L += [f"## {cond}", "", f"accuracy {np.mean([x['acc'] for x in u0]):.3f} · FPR {np.mean([x['fpr'] for x in u0]):.3f} · FNR "
              f"{np.mean([x['fnr'] for x in u0]):.3f} · videos {u0[0]['signals']['margin']['n']} · confident false negatives (FN with margin "
              f"above the median correct-genuine margin): {np.nanmean([x['confident_fn_share'] for x in u0]):.2f} of FNs", "",
              "| signal | error AUROC | AURC | E-AURC | risk@90% | risk@80% | FN-vs-TN AUROC (genuine calls) |", "|---|---:|---:|---:|---:|---:|---:|"]
        best = max(SIGNALS, key=lambda n: np.nanmean([x["signals"][n]["auroc_err"] for x in u0]))
        for n in SIGNALS:
            g = lambda f: np.nanmean([x["signals"][n][f] for x in u0])
            L.append(f"| {'**' + n + '**' if n == best else n} | {g('auroc_err'):.3f} | {g('aurc'):.4f} | {g('eaurc'):.4f} | {g('risk@90'):.3f} | "
                     f"{g('risk@80'):.3f} | {g('auroc_fn_vs_tn'):.3f} |")
        L.append("")
    L += ["Error rate at full coverage equals risk@100%; a signal no better than chance has error AUROC ≈ 0.5. Test rows are read after "
          "all fitting and use each split's fitted stacker and τ. One detector seed, one temporal-branch seed."]
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()

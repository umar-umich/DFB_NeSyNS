#!/usr/bin/env python3
"""The deployed system (UMAR 2026-09-17): V-JEPA 2 LoRA (T) + Appearance (APP) + Off-manifold residual (OM), fused by
conflict-aware Dempster-Shafer with Shafer reliability discounting (DS-R). Cached outputs, CPU.

    python analysis/tbiom/temporal/dsr_system.py uncertainty  [--branch vjepa2_lora_seed42]
    python analysis/tbiom/temporal/dsr_system.py diagnosis    [--branch vjepa2_lora_seed42]
    python analysis/tbiom/temporal/dsr_system.py confirmation --branch vjepa2_lora_seed1337

DS-R. Each branch opinion (p, u) -> Dirichlet alpha; evidence scaled by a reliability weight w_b in {0,.25,.5,.75,1}
(max w = 1): alpha' = 1 + w (alpha - 1); combined by Dempster's rule (DISCERN's operator). Weights chosen on the fusion
audit's source-grouped FITTING folds by the v3.1 objective; everything is compared on the SELECTION folds (5 splits).
The fused opinion carries probability p_f, vacuity u_f = K / S_f and the accumulated conflict mass
k_f = 1 - prod(1 - k_pair) of the sequential combinations. tau = EER of p_f on the selection fold's FF++ val videos.

UNCERTAINTY (item 1). Because the deployed fusion is DS-based, the fused vacuity IS the system's uncertainty. Error-ranking
signals for the decision p_f >= tau: decision margin (|logit p_f - logit tau|), fused vacuity u_f, fused conflict k_f,
discounted per-branch vacuities, branch disagreement (range of branch p) and max pairwise conflict. Metrics as EXP6:
error AUROC, AURC, E-AURC, risk at 90/80% coverage, FN-vs-TN AUROC, confident false negatives. Clean and processed separate.

DIAGNOSIS (item 3), criteria fixed before the run:
  CAUSE (b) V-JEPA collapse: V-JEPA's own v3.1 score on processed development (JPEG q40) falls >= 0.05 below its clean
     development score.  [DFDCP-JPEG test AUROC drop reported as confirmation, not used for the decision.]
  CAUSE (a) fusion deterioration: on processed development the DS-R fused score is below the best single branch's
     processed score, OR the fused drop (clean -> processed) exceeds the smallest single-branch drop by >= 0.02.
  REMEDY for (a), no encoder retraining: refit DS-R reliability weights on clean + JPEG q40 + resize 0.60 versions of the
     FITTING-fold development videos (mean v3.1 of the three), evaluate on the SELECTION folds. Adopt if processed
     development improves by >= 0.005 (mean of the two conditions) with clean development loss <= 0.005. Caveat: the
     processed development severities equal the processed test severities.
  An augmentation pilot is warranted only if (b) holds.

CONFIRMATION (item 2). Same pipeline for another temporal-branch seed: DS-R weights refit on the same folds. Labelled
temporal-branch training variability conditional on fixed APP and OM branches (warm+gate epoch 19), not detector replication.
"""
from __future__ import annotations
import argparse, itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import exp1_fusion_audit as A  # noqa: E402
import symbolic_overlay as SO  # noqa: E402
from health import eer  # noqa: E402
from fusion_combos import attach  # noqa: E402
from safeguard_ablation import rc_metrics, fn_metrics  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402

S23, S2, BR, R = Path("logs/tbiom/stage23"), Path("logs/tbiom/stage2"), Path("logs/tbiom/temporal/branches"), Path("tbiom/reports")
DEV, PROC = ("VALmix", "VALmixV2", "FFpp_val"), ("jpeg40", "rs06")
VIEWS, GRID, K = ["T", "APP", "OM"], (0.0, 0.25, 0.5, 0.75, 1.0), 2.0
WGRID = [w for w in itertools.product(GRID, repeat=3) if max(w) == 1.0]


def dsr(t, w):
    """Discounted DS over VIEWS. Returns p_f, u_f, k_f (accumulated conflict), per-branch discounted (p, u)."""
    alphas, pu = [], {}
    for i, v in enumerate(VIEWS):
        p, u = t[f"p_{v}"].to_numpy(), np.clip(t[f"u_{v}"].to_numpy(), 1e-9, None); S = K / u
        a = 1 + w[i] * (np.stack([(1 - p) * S, p * S], 1) - 1); alphas.append(a)
        pu[v] = (a[:, 1] / a.sum(1), K / a.sum(1))
    f, keep = alphas[0], np.ones(len(t))
    for a in alphas[1:]:
        S1, S2_ = f.sum(1, keepdims=True), a.sum(1, keepdims=True)
        b1, b2, u1, u2 = (f - 1) / S1, (a - 1) / S2_, K / S1, K / S2_
        c = (b1.sum(1) * b2.sum(1) - (b1 * b2).sum(1)); keep *= (1 - c)
        b = (b1 * b2 + b1 * u2 + b2 * u1) / (1 - c)[:, None]; uu = u1 * u2 / (1 - c)[:, None]
        f = b * (K / uu) + 1
    Sf = f.sum(1)
    return f[:, 1] / Sf, K / Sf, 1 - keep, pu


def load(branch):
    ob = BR / branch / "opinions"
    dev = attach(A.dev_tables({n: S23 / f"warmgate_e19_{n}.csv" for n in DEV}, [1.0]), [ob / f"{n}.csv" for n in DEV])
    dproc = {c: attach(A.dev_tables({n: S2 / f"p_dev_warmgate_e19_{n}_{c}.csv" for n in DEV}, [1.0]), [ob / f"{n}_{c}.csv" for n in DEV]) for c in PROC}
    tests = {}
    for t in A.TEST:
        d = A.load_frames(S23 / f"warmgate_sel_{t}.csv")
        keep = ~d.subset.str.contains("IP_LAP") & (d.subset != "fake-audio-driven") & ~d.subset.str.startswith("adv-")
        tests[t] = attach(A.video_table(d[keep].reset_index(drop=True), [1.0]), [ob / f"{t}.csv"])
    c3 = tests["CDFv3"]; tests["CDFv3 reenactment"] = pd.concat([c3[c3.y == 0], c3[(c3.y == 1) & c3.subset.str.contains("FaceReenact|TalkingFace")]], ignore_index=True)
    kd = tests["KoDF"]; tests["KoDF first-order motion"] = pd.concat([kd[kd.y == 0], kd[(kd.y == 1) & (kd.subset == "fake-fo")]], ignore_index=True)
    for t in ("CDFv2", "DFDCP"):
        for c in PROC:
            tests[f"{t} {c}"] = attach(A.video_table(A.load_frames(S2 / f"p_v3_warmgate_{t}_{c}.csv"), [1.0]), [ob / f"{t}_{c}.csv"])
    return dev, dproc, tests


def check_against_discern_operator(t):
    g = pd.DataFrame({f"{k}_{v}": t[f"{k}_{v}"].to_numpy() for v in VIEWS for k in "pu"})
    SO.BRANCHES[:] = VIEWS; ref = SO.fuse(g); SO.BRANCHES[:] = ["semantic", "artifact", "fsvfm"]
    err = float(np.abs(dsr(t, (1.0, 1.0, 1.0))[0] - ref).max()); assert err < 1e-6, f"DS mismatch {err}"
    return err


def splits(dev):
    grp = A.groups_for(dev)
    for k in range(A.N_SPLITS):
        fi, si = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=k).split(dev, dev.y, grp))
        yield k, dev.iloc[fi].reset_index(drop=True), dev.iloc[si].reset_index(drop=True)


def aligned(P, S):
    Pt = P.set_index("v").loc[S.v].reset_index(); Pt["source"] = S.source.to_numpy(); return Pt


def fit_w(tables):
    cls = [A.cells(t) for t in tables]
    return max(WGRID, key=lambda w: np.mean([A.v31(dsr(t, w)[0], c)[0] for t, c in zip(tables, cls)]))


def fmt(v):
    return f"{np.mean(v):.4f} ±{np.std(v):.4f}"


# ----------------------------------------------------------------------------- item 1
def uncertainty(branch, tag):
    dev, dproc, tests = load(branch); ds_err = check_against_discern_operator(dev)
    names = ["margin", "u_fused", "conflict_fused", "u_T", "u_APP", "u_OM", "disagree_range", "conflict_pair_max"]
    out = []
    for k, Fd, Sd in splits(dev):
        w = fit_w([Fd]); p, _, _, _ = dsr(Sd, w); ffs = np.where(Sd.source == "FFpp_val")[0]
        tau = eer(Sd.y.to_numpy()[ffs], p[ffs])[1]
        lg = lambda x: np.log(np.clip(x, 1e-9, 1 - 1e-9) / np.clip(1 - x, 1e-9, 1))

        def ev(t):
            pf, uf, kf, pu = dsr(t, w); y = t.y.to_numpy(); pred = (pf >= tau).astype(int); err = pred != y
            P = np.stack([pu[v][0] for v in VIEWS], 1); U = np.stack([pu[v][1] for v in VIEWS], 1)
            pair = []
            for a, b in ((0, 1), (0, 2), (1, 2)):
                ba1, ba0 = P[:, a] - U[:, a] / 2, 1 - P[:, a] - U[:, a] / 2; bb1, bb0 = P[:, b] - U[:, b] / 2, 1 - P[:, b] - U[:, b] / 2
                pair.append(np.clip(ba1, 0, 1) * np.clip(bb0, 0, 1) + np.clip(ba0, 0, 1) * np.clip(bb1, 0, 1))
            sig = {"margin": -np.abs(lg(pf) - lg(np.array(tau))), "u_fused": uf, "conflict_fused": kf, "u_T": U[:, 0], "u_APP": U[:, 1],
                   "u_OM": U[:, 2], "disagree_range": P.max(1) - P.min(1), "conflict_pair_max": np.max(pair, 0)}
            called_fn, called_tn = (pred == 0) & (y == 1), (pred == 0) & (y == 0)
            return {"acc": float(1 - err.mean()), "fpr": float((pred[y == 0] == 1).mean()), "fnr": float((pred[y == 1] == 0).mean()),
                    "auroc": A.fast_auc(pf[y == 0], pf[y == 1]),
                    "signals": {n: {**rc_metrics(v, err), **fn_metrics(v, y, pred)} for n, v in sig.items()},
                    "confident_fn_share": float(np.mean(sig["margin"][called_fn] < np.median(sig["margin"][called_tn]))) if called_fn.any() and called_tn.any() else np.nan,
                    "confident_fn_share_uf": float(np.mean(sig["u_fused"][called_fn] < np.median(sig["u_fused"][called_tn]))) if called_fn.any() and called_tn.any() else np.nan}
        r = {"weights": dict(zip(VIEWS, w)), "tau": float(tau), "clean dev": ev(Sd)}
        for c, P in dproc.items():
            r[f"{c} dev"] = ev(aligned(P, Sd))
        r["clean tests (10 sets pooled)"] = ev(pd.concat([tests[t] for t in A.TEST], ignore_index=True))
        for t in ("CDFv2", "DFDCP"):
            r[f"{t} clean"] = ev(tests[t])
            for c in PROC:
                r[f"{t} {c}"] = ev(tests[f"{t} {c}"])
        out.append(r); print(f"uncertainty split {k}: weights {r['weights']} tau {tau:.3f}", flush=True)
    json.dump(out, open(R / f"EXP6B_UNCERTAINTY_DSR_{tag}.json", "w"), indent=1, default=float)
    L = [f"# Experiment 6b — uncertainty safeguard for the deployed DS-R system (`{branch}` + APP + OM)", "",
         "Deployed fusion = conflict-aware Dempster–Shafer with reliability discounting, so the fused Dirichlet vacuity **is** the "
         "system's uncertainty. Signals are scored as error rankers for the decision p_f ≥ τ (τ = EER on selection-fold FF++ val). "
         f"DS implementation checked against DISCERN's operator (max |Δp| {ds_err:.1e}). Mean over 5 source-grouped splits; clean and "
         "processed separate; test rows read after fitting. Script `analysis/tbiom/temporal/dsr_system.py`.", "",
         "Reliability weights per split (T, APP, OM): " + "; ".join(str(tuple(r["weights"].values())) for r in out), ""]
    conds = [c for c in out[0] if c not in ("weights", "tau")]
    for cond in conds:
        u0 = [r[cond] for r in out]
        L += [f"## {cond}", "", f"AUROC {np.mean([x['auroc'] for x in u0]):.3f} · accuracy {np.mean([x['acc'] for x in u0]):.3f} · FPR "
              f"{np.mean([x['fpr'] for x in u0]):.3f} · FNR {np.mean([x['fnr'] for x in u0]):.3f} · confident FN share (margin) "
              f"{np.nanmean([x['confident_fn_share'] for x in u0]):.2f} · (fused vacuity) {np.nanmean([x['confident_fn_share_uf'] for x in u0]):.2f}", "",
              "| signal | error AUROC | AURC | E-AURC | risk@90% | risk@80% | FN-vs-TN AUROC |", "|---|---:|---:|---:|---:|---:|---:|"]
        best = max(names, key=lambda n: np.nanmean([x["signals"][n]["auroc_err"] for x in u0]))
        for n in names:
            g = lambda f: np.nanmean([x["signals"][n][f] for x in u0])
            L.append(f"| {'**' + n + '**' if n == best else n} | {g('auroc_err'):.3f} | {g('aurc'):.4f} | {g('eaurc'):.4f} | {g('risk@90'):.3f} | {g('risk@80'):.3f} | {g('auroc_fn_vs_tn'):.3f} |")
        L.append("")
    L += ["Confident FN share = false negatives whose signal marks them as MORE confident than the median correct genuine call. "
          "One detector seed, one temporal-branch seed; CDFv2 and DeepfakeEval2024 informed development; FaceShifter sealed."]
    (R / f"EXP6B_UNCERTAINTY_DSR_{tag}.md").write_text("\n".join(L) + "\n"); print("wrote", R / f"EXP6B_UNCERTAINTY_DSR_{tag}.md")


# ----------------------------------------------------------------------------- item 3
def diagnosis(branch, tag):
    dev, dproc, tests = load(branch); rows = []
    for k, Fd, Sd in splits(dev):
        w_clean = fit_w([Fd]); w_proc = fit_w([Fd] + [aligned(P, Fd) for P in dproc.values()])
        conds = {"clean": Sd, **{c: aligned(P, Sd) for c, P in dproc.items()}}
        r = {"w_clean": dict(zip(VIEWS, w_clean)), "w_proc": dict(zip(VIEWS, w_proc)), "dev": {}, "tests": {}}
        for cn, t in conds.items():
            cl = A.cells(t)
            r["dev"][cn] = {**{f"branch {v}": A.v31(t[f"p_{v}"].to_numpy(), cl)[0] for v in VIEWS},
                            "DS equal": A.v31(dsr(t, (1, 1, 1))[0], cl)[0], "DS-R (clean fit)": A.v31(dsr(t, w_clean)[0], cl)[0],
                            "DS-R (clean+processed fit)": A.v31(dsr(t, w_proc)[0], cl)[0]}
        for tn in ["DFDCP", "DFDCP jpeg40", "DFDCP rs06", "CDFv2", "CDFv2 jpeg40", "CDFv2 rs06"] + A.TEST + ["CDFv3 reenactment", "KoDF first-order motion"]:
            T = tests[tn]; y = T.y.to_numpy(); auc = lambda s: A.fast_auc(s[y == 0], s[y == 1])
            r["tests"][tn] = {**{f"branch {v}": auc(T[f"p_{v}"].to_numpy()) for v in VIEWS}, "DS equal": auc(dsr(T, (1, 1, 1))[0]),
                              "DS-R (clean fit)": auc(dsr(T, w_clean)[0]), "DS-R (clean+processed fit)": auc(dsr(T, w_proc)[0])}
        rows.append(r); print(f"diagnosis split {k}: w_clean {r['w_clean']} w_proc {r['w_proc']}", flush=True)
    json.dump(rows, open(R / f"EXP8_DFDCP_DIAGNOSIS_{tag}.json", "w"), indent=1, default=float)
    arms = [f"branch {v}" for v in VIEWS] + ["DS equal", "DS-R (clean fit)", "DS-R (clean+processed fit)"]
    D = lambda cond, arm: [r["dev"][cond][arm] for r in rows]
    # cause (b)
    drop_T = np.mean(D("clean", "branch T")) - np.mean(D("jpeg40", "branch T"))
    cause_b = drop_T >= 0.05
    # cause (a)
    fused = "DS-R (clean fit)"
    cause_a = {}
    for c in PROC:
        best_branch = max(np.mean(D(c, f"branch {v}")) for v in VIEWS)
        drops = {v: np.mean(D("clean", f"branch {v}")) - np.mean(D(c, f"branch {v}")) for v in VIEWS}
        fdrop = np.mean(D("clean", fused)) - np.mean(D(c, fused))
        cause_a[c] = {"fused_below_best_branch": np.mean(D(c, fused)) < best_branch, "fused_drop": fdrop, "min_branch_drop": min(drops.values()),
                      "excess_drop": fdrop - min(drops.values()), "holds": np.mean(D(c, fused)) < best_branch or fdrop - min(drops.values()) >= 0.02,
                      "branch_drops": drops}
    gain_proc = np.mean([np.mean(D(c, "DS-R (clean+processed fit)")) - np.mean(D(c, fused)) for c in PROC])
    loss_clean = np.mean(D("clean", fused)) - np.mean(D("clean", "DS-R (clean+processed fit)"))
    remedy_criteria = gain_proc >= 0.005 and loss_clean <= 0.005
    any_a = any(v['holds'] for v in cause_a.values())
    adopt = any_a and remedy_criteria
    L = [f"# Experiment 8 — DFDCP / processing weakness: diagnosis for the deployed DS-R system (`{branch}` + APP + OM)", "",
         __doc__.split("DIAGNOSIS (item 3), criteria fixed before the run:")[1].split("CONFIRMATION")[0].strip(), "",
         "## 1. Development (selection folds, v3.1, mean ± sd over 5 splits)", "", "| arm | clean | JPEG q40 | resize 0.60 |", "|---|---:|---:|---:|"]
    for a in arms:
        L.append(f"| {a} | {fmt(D('clean', a))} | {fmt(D('jpeg40', a))} | {fmt(D('rs06', a))} |")
    L += ["", "## 2. Verdicts", "",
          f"- **Cause (b) — V-JEPA collapse under compression:** V-JEPA's development v3.1 drop clean → JPEG q40 = **{drop_T:+.4f}** "
          f"(criterion ≥ 0.05) → **{'HOLDS' if cause_b else 'does not hold'}**."]
    for c, v in cause_a.items():
        L.append(f"- **Cause (a) — fusion deterioration, {c}:** fused below best single branch: {v['fused_below_best_branch']}; fused drop "
                 f"{v['fused_drop']:+.4f} vs smallest branch drop {v['min_branch_drop']:+.4f} (excess {v['excess_drop']:+.4f}, criterion ≥ 0.02) → "
                 f"**{'HOLDS' if v['holds'] else 'does not hold'}**. Branch drops: " + ", ".join(f"{b} {d:+.4f}" for b, d in v["branch_drops"].items()))
    L += [f"- **Remedy for (a) — clean+processed reliability fit:** processed development change {gain_proc:+.4f} (criterion ≥ +0.005), clean "
          f"development loss {loss_clean:+.4f} (criterion ≤ 0.005); numerical criteria {'met' if remedy_criteria else 'not met'}, but the remedy applies only when cause (a) holds, which it {'does' if any_a else 'does not'} → **{'ADOPT' if adopt else 'not adopted'}**. Weights (T, APP, OM) clean fit: "
          + "; ".join(str(tuple(r['w_clean'].values())) for r in rows) + " | clean+processed fit: " + "; ".join(str(tuple(r['w_proc'].values())) for r in rows),
          f"- **Augmentation pilot warranted:** {'YES (cause b holds)' if cause_b else 'NO (cause b does not hold)'}.", "",
          "## 3. Test sets — video AUROC (read after; mean over the 5 fitted splits)", "", "| test | " + " | ".join(arms) + " |", "|---|" + "---:|" * len(arms)]
    for tn in rows[0]["tests"]:
        L.append(f"| {tn} | " + " | ".join(f"{np.mean([r['tests'][tn][a] for r in rows]):.4f}" for a in arms) + " |")
    L.append("| **10-set mean** | " + " | ".join(f"**{np.mean([np.mean([r['tests'][t][a] for r in rows]) for t in A.TEST]):.4f}**" for a in arms) + " |")
    L += ["", "Processed development and processed tests share the JPEG q40 / resize 0.60 severities, so the clean+processed reliability fit is "
          "severity-matched to the processed tests (optimistic for those rows). One detector seed, one temporal-branch seed. FaceShifter sealed."]
    (R / f"EXP8_DFDCP_DIAGNOSIS_{tag}.md").write_text("\n".join(L) + "\n"); print("wrote", R / f"EXP8_DFDCP_DIAGNOSIS_{tag}.md")


# ----------------------------------------------------------------------------- item 2
def confirmation(branches):
    res = {}
    for b in branches:
        dev, dproc, tests = load(b); rr = []
        for k, Fd, Sd in splits(dev):
            w = fit_w([Fd]); conds = {"clean": Sd, **{c: aligned(P, Sd) for c, P in dproc.items()}}
            r = {"w": dict(zip(VIEWS, w)), "dev": {}, "tests": {}}
            for cn, t in conds.items():
                cl = A.cells(t); r["dev"][cn] = {"branch T": A.v31(t.p_T.to_numpy(), cl), "DS-R": A.v31(dsr(t, w)[0], cl)}
            for tn, T in tests.items():
                y = T.y.to_numpy(); auc = lambda s: A.fast_auc(s[y == 0], s[y == 1])
                r["tests"][tn] = {"branch T": auc(T.p_T.to_numpy()), "DS-R": auc(dsr(T, w)[0])}
            rr.append(r)
        res[b] = rr; print(f"confirmation {b}: weights " + "; ".join(str(tuple(r['w'].values())) for r in rr), flush=True)
    json.dump(res, open(R / "EXP5_CONFIRMATION_SEED.json", "w"), indent=1, default=float)
    rec = {b: json.load(open(BR / b / "train_record.json")) for b in branches}
    L = ["# Experiment 5 — confirmation seed: temporal-branch training variability conditional on fixed branches", "",
         "V-JEPA 2 LoRA branch retrained on the EXACT seed-42 recipe (same pretrained checkpoint, clips, sampling, preprocessing, LoRA "
         "rank 8 / alpha 16 / dropout 0.1, losses, 10 epochs, batch 32, development-only epoch selection; no augmentation) with another "
         "seed. Appearance and Off-manifold residual are the SAME fixed warm+gate epoch-19 branches for every seed, and the DS-R "
         "reliability weights are refitted on the same source-grouped folds. **This measures temporal-branch training variability "
         "conditional on fixed branches — it is not a full detector replication.** Script `analysis/tbiom/temporal/dsr_system.py`.", "",
         "| seed run | selected epoch | branch dev v3.1 (train record) | DS-R weights per split (T, APP, OM) |", "|---|---:|---:|---|"]
    for b in branches:
        L.append(f"| {b} | {rec[b]['best_epoch']} | {rec[b]['best_dev_v31']:.4f} | " + "; ".join(str(tuple(r['w'].values())) for r in res[b]) + " |")
    L += ["", "## Development (selection folds, v3.1, mean of 5 splits)", "", "| condition | arm | " + " | ".join(branches) + " | spread (max − min) |",
          "|---|---|" + "---:|" * (len(branches) + 1)]
    for cn in ("clean", "jpeg40", "rs06"):
        for arm in ("branch T", "DS-R"):
            v = [np.mean([r["dev"][cn][arm][0] for r in res[b]]) for b in branches]
            L.append(f"| {cn} | {arm} | " + " | ".join(f"{x:.4f}" for x in v) + f" | {max(v) - min(v):.4f} |")
        v = [np.mean([r["dev"][cn]["DS-R"][1]["reenactment"] for r in res[b]]) for b in branches]
        L.append(f"| {cn} | DS-R reenactment | " + " | ".join(f"{x:.4f}" for x in v) + f" | {max(v) - min(v):.4f} |")
    L += ["", "## Test sets — video AUROC (read after)", "", "| test | arm | " + " | ".join(branches) + " | spread |", "|---|---|" + "---:|" * (len(branches) + 1)]
    tl = list(res[branches[0]][0]["tests"])
    for tn in tl:
        for arm in ("branch T", "DS-R"):
            v = [np.mean([r["tests"][tn][arm] for r in res[b]]) for b in branches]
            L.append(f"| {tn} | {arm} | " + " | ".join(f"{x:.4f}" for x in v) + f" | {max(v) - min(v):.4f} |")
    for arm in ("branch T", "DS-R"):
        v = [np.mean([np.mean([r["tests"][t][arm] for r in res[b]]) for t in A.TEST]) for b in branches]
        L.append(f"| **10-set mean** | {arm} | " + " | ".join(f"**{x:.4f}**" for x in v) + f" | {max(v) - min(v):.4f} |")
    L += ["", "Two temporal-branch seeds cannot give a standard deviation worth quoting; spreads are reported instead. FaceShifter sealed."]
    (R / "EXP5_CONFIRMATION_SEED.md").write_text("\n".join(L) + "\n"); print("wrote", R / "EXP5_CONFIRMATION_SEED.md")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["uncertainty", "diagnosis", "confirmation"])
    ap.add_argument("--branch", nargs="+", default=["vjepa2_lora_seed42"])
    a = ap.parse_args()
    tag = a.branch[0].split("_seed")[-1]; tag = f"seed{tag}"
    if a.mode == "uncertainty":
        uncertainty(a.branch[0], tag)
    elif a.mode == "diagnosis":
        diagnosis(a.branch[0], tag)
    else:
        confirmation(a.branch)


if __name__ == "__main__":
    main()

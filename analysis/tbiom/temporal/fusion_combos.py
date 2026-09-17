#!/usr/bin/env python3
"""View-combination x fusion-rule study for the confirmed temporal branch (cached outputs, CPU).

    OMP_NUM_THREADS=8 python analysis/tbiom/temporal/fusion_combos.py [--branch vjepa2_lora_seed42] [--tag seed42]

Views: T = V-JEPA 2 LoRA branch; APP = Appearance; OM = Off-manifold residual; FP = Bona-fide face prior (raw)
(APP/OM/FP from the development-selected warm+gate detector, epoch 19).
Combinations: T+APP+OM, T+APP+OM+FP, T+OM+FP, T+OM, T+APP, T+FP, APP+OM (two-view base), OM+FP.
Fusion rules, all fitted ONLY on the source-grouped fitting folds of the fusion audit (5 splits):
  LR    logistic stacking on matched inputs (p, vacuity)
  DS    Dempster-Shafer combination of the branch Dirichlet opinions, equal trust
  DS-R  conflict-aware DS with Shafer reliability discounting: each branch's evidence scaled by a reliability weight
        w_b in {0, .25, .5, .75, 1} (max w = 1) chosen on the fitting fold by the v3.1 objective, then combined by DS
Evaluated on the selection folds (clean, JPEG q40, resize 0.60 — processed never used for fitting); tests read after.

NON-INFERIORITY RULE (fixed before the run, UMAR 2026-09-17: prefer the conflict-aware evidential fusion when it is close
to the simple alternative). For a combination, the evidential rule (best of DS / DS-R on clean development) is ADOPTED
over LR if mean Delta = LR - evidential < +0.005 on clean development AND the 95% paired source-group bootstrap CI upper
bound < +0.010 in >= 4/5 splits. LR is always reported alongside.
"""
from __future__ import annotations
import argparse, itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import exp1_fusion_audit as A  # noqa: E402
import symbolic_overlay as SO  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

S23, S2, BR = Path("logs/tbiom/stage23"), Path("logs/tbiom/stage2"), Path("logs/tbiom/temporal/branches")
DEV, PROC = ("VALmix", "VALmixV2", "FFpp_val"), ("jpeg40", "rs06")
REN = {"p_semantic": "p_APP", "u_semantic": "u_APP", "p_artifact": "p_OM", "u_artifact": "u_OM", "p_fsvfm_raw": "p_FP", "u_fsvfm_raw": "u_FP"}
COMBOS = {"T+APP+OM": ["T", "APP", "OM"], "T+APP+OM+FP": ["T", "APP", "OM", "FP"], "T+OM+FP": ["T", "OM", "FP"], "T+OM": ["T", "OM"],
          "T+APP": ["T", "APP"], "T+FP": ["T", "FP"], "APP+OM": ["APP", "OM"], "OM+FP": ["OM", "FP"]}
GRID, N_BOOT, NI_MEAN, NI_CI = (0.0, 0.25, 0.5, 0.75, 1.0), 1000, 0.005, 0.010


def attach(tab, files):
    o = pd.concat([pd.read_csv(f) for f in files]).drop_duplicates("v").set_index("v")
    idx = o.index.get_indexer(tab.v); assert (idx >= 0).all()
    t = tab.rename(columns=REN).copy(); t["p_T"], t["u_T"] = o.p.to_numpy()[idx], o.u.to_numpy()[idx]
    return t


def ds(t, views, w=None):
    g = pd.DataFrame(index=range(len(t)))
    for i, v in enumerate(views):
        p, u = t[f"p_{v}"].to_numpy(), t[f"u_{v}"].to_numpy()
        if w is not None and w[i] < 1:
            S = 2 / np.clip(u, 1e-9, None); a1, a0 = 1 + w[i] * (p * S - 1), 1 + w[i] * ((1 - p) * S - 1); S2_ = a1 + a0
            p, u = a1 / S2_, 2 / S2_
        g[f"p_{v}"], g[f"u_{v}"] = p, u
    SO.BRANCHES[:] = list(views); s = SO.fuse(g); SO.BRANCHES[:] = ["semantic", "artifact", "fsvfm"]
    return s


def lr_fit(F, views):
    cols = [f"{k}_{v}" for v in views for k in "pu"]; sc = StandardScaler().fit(F[cols].to_numpy())
    return cols, sc, LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(sc.transform(F[cols].to_numpy()), F.y.to_numpy())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--branch", default="vjepa2_lora_seed42"); ap.add_argument("--tag", default="seed42")
    a = ap.parse_args(); ob = BR / a.branch / "opinions"
    dev = attach(A.dev_tables({n: S23 / f"warmgate_e19_{n}.csv" for n in DEV}, [1.0]), [ob / f"{n}.csv" for n in DEV])
    dproc = {c: attach(A.dev_tables({n: S2 / f"p_dev_warmgate_e19_{n}_{c}.csv" for n in DEV}, [1.0]), [ob / f"{n}_{c}.csv" for n in DEV]) for c in PROC}
    tests = {}
    for t in A.TEST:
        d = A.load_frames(S23 / f"warmgate_sel_{t}.csv")
        keep = ~d.subset.str.contains("IP_LAP") & (d.subset != "fake-audio-driven") & ~d.subset.str.startswith("adv-")
        tests[t] = attach(A.video_table(d[keep].reset_index(drop=True), [1.0]), [ob / f"{t}.csv"])
    c3 = tests["CDFv3"]; tests["CDFv3 reenactment"] = pd.concat([c3[c3.y == 0], c3[(c3.y == 1) & c3.subset.str.contains("FaceReenact|TalkingFace")]], ignore_index=True)
    for t in ("CDFv2", "DFDCP"):
        for c in PROC:
            tests[f"{t} {c}"] = attach(A.video_table(A.load_frames(S2 / f"p_v3_warmgate_{t}_{c}.csv"), [1.0]), [ob / f"{t}_{c}.csv"])
    grp = A.groups_for(dev); rng = np.random.default_rng(12345); res = []
    for k in range(A.N_SPLITS):
        fi, si = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=k).split(dev, dev.y, grp))
        Fd, Sd = dev.iloc[fi].reset_index(drop=True), dev.iloc[si].reset_index(drop=True); gS = grp[si]; clF, clS = A.cells(Fd), A.cells(Sd)
        split = {"sel": {}, "proc": {}, "tests": {}, "boot": {}, "weights": {}}
        for name, views in COMBOS.items():
            lr = lr_fit(Fd, views)
            grid = [w for w in itertools.product(GRID, repeat=len(views)) if max(w) == 1.0]
            w_best = max(grid, key=lambda w: A.v31(ds(Fd, views, w), clF)[0]); split["weights"][name] = w_best
            rules = {"LR": lambda t: lr[2].predict_proba(lr[1].transform(t[lr[0]].to_numpy()))[:, 1],
                     "DS": lambda t: ds(t, views), "DS-R": lambda t: ds(t, views, w_best)}
            sc = {r: f(Sd) for r, f in rules.items()}
            for r in rules:
                split["sel"][f"{name}|{r}"] = A.v31(sc[r], clS)
            evid = max(("DS", "DS-R"), key=lambda r: split["sel"][f"{name}|{r}"][0])
            ug = np.unique(gS); pos = {g: np.where(gS == g)[0] for g in ug}; d = []
            for _ in range(N_BOOT):
                wts = np.zeros(len(Sd), dtype=int)
                for g in rng.choice(ug, len(ug), replace=True):
                    wts[pos[g]] += 1
                d.append(A.v31(sc["LR"], clS, wts)[0] - A.v31(sc[evid], clS, wts)[0])
            split["boot"][name] = {"evidential": evid, "delta": split["sel"][f"{name}|LR"][0] - split["sel"][f"{name}|{evid}"][0],
                                   "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
            for c, P in dproc.items():
                Pt = P.set_index("v").loc[Sd.v].reset_index(); Pt["source"] = Sd.source.to_numpy()
                for r, f in rules.items():
                    split["proc"][f"{name}|{r}|{c}"] = A.v31(f(Pt), A.cells(Pt))
            for tn, T in tests.items():
                y = T.y.to_numpy()
                for r, f in rules.items():
                    s = f(T); split["tests"][f"{name}|{r}|{tn}"] = A.fast_auc(s[y == 0], s[y == 1])
        res.append(split)
        print(f"split {k}: " + " | ".join(f"{n} LR {split['sel'][n + '|LR'][0]:.4f} DS {split['sel'][n + '|DS'][0]:.4f} DS-R {split['sel'][n + '|DS-R'][0]:.4f}"
                                          for n in ("T+APP+OM", "T+OM+FP", "T+OM")), flush=True)
    out = Path("tbiom/reports"); json.dump(res, open(out / f"EXP7B_FUSION_COMBOS_{a.tag}.json", "w"), indent=1, default=float)
    write(out / f"EXP7B_FUSION_COMBOS_{a.tag}.md", res, a.branch)


def write(path, res, branch):
    m = lambda key, i=0: np.mean([s["sel"][key][i] if i == 0 else s["sel"][key][1]["reenactment"] for s in res])
    L = [f"# Experiment 7b — view combinations × fusion rules beside the temporal branch (`{branch}`)", "",
         __doc__.split("\n\n", 1)[1].strip(), "",
         "## 1. Development (selection folds, mean of 5 splits)", "",
         "| combination | rule | clean v3.1 | swap | reenactment | in-the-wild | jpeg40 v3.1 | rs06 v3.1 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name in COMBOS:
        for r in ("LR", "DS", "DS-R"):
            key = f"{name}|{r}"
            types = [np.mean([s["sel"][key][1][t] for s in res]) for t in res[0]["sel"][key][1]]
            L.append(f"| {name} | {r} | {m(key):.4f} | " + " | ".join(f"{x:.4f}" for x in types) +
                     f" | {np.mean([s['proc'][key + '|jpeg40'][0] for s in res]):.4f} | {np.mean([s['proc'][key + '|rs06'][0] for s in res]):.4f} |")
    L += ["", "## 2. Non-inferiority of the evidential rule vs logistic stacking (clean development)", "",
          "| combination | evidential rule (per split) | mean Δ = LR − evidential | CI upper < +0.010 splits | DS-R weights (split 0) | adopt evidential? |",
          "|---|---|---:|---:|---|---|"]
    for name, views in COMBOS.items():
        d = [s["boot"][name]["delta"] for s in res]; hi = sum(s["boot"][name]["ci"][1] < NI_CI for s in res)
        rules = ",".join(s["boot"][name]["evidential"] for s in res)
        L.append(f"| {name} | {rules} | {np.mean(d):+.4f} | {hi}/5 | {dict(zip(views, res[0]['weights'][name]))} | "
                 f"{'**YES**' if np.mean(d) < NI_MEAN and hi >= 4 else 'no'} |")
    tests = ["CDFv2", "CDFv3", "CDFv3 reenactment", "DFD", "DFDC", "DFDCP", "DFEval24", "FFIW", "KoDF", "IDForge", "DFDM",
             "CDFv2 jpeg40", "CDFv2 rs06", "DFDCP jpeg40", "DFDCP rs06"]
    L += ["", "## 3. Test sets — video AUROC (read after; mean over the 5 fitted splits)", "",
          "| combination | rule | " + " | ".join(tests) + " | 10-set mean |", "|---|---|" + "---:|" * (len(tests) + 1)]
    for name in COMBOS:
        for r in ("LR", "DS", "DS-R"):
            v = {t: np.mean([s["tests"][f"{name}|{r}|{t}"] for s in res]) for t in tests}
            L.append(f"| {name} | {r} | " + " | ".join(f"{v[t]:.4f}" for t in tests) + f" | {np.mean([v[t] for t in A.TEST]):.4f} |")
    L += ["", "One detector seed and one temporal-branch seed. CDFv2 and DeepfakeEval2024 informed development. FaceShifter is sealed."]
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()

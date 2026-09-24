#!/usr/bin/env python3
"""Two requested result tables, built only from cached outputs.

    OMP_NUM_THREADS=8 python analysis/tbiom/paper/two_tables.py

Table 1 -- acquisition study. Reproduces the per-video policy scores of
`analysis/tbiom/acquisition/step1_controls.py` from the existing window cache so that a paired source-group
bootstrap interval can be put on the change column, which the original report never computed. No policy is
re-designed and no model is trained beyond re-fitting the same deterministic selector the original run fitted.

Table 2 -- video-view contribution. Every row uses the SAME fusion rule (PCR6+-R, the manuscript's operator), the
same five source-grouped fitting folds, the same grid and objective, and the same frozen threshold rule (EER on
FF++ val). Only the view set, or the video backbone supplying the temporal view, changes.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "pcr6p"), str(HERE.parent / "grouped"), str(HERE.parent),
                str(HERE.parent / "temporal"), str(HERE.parent / "acquisition")]
import ops_pcr6p as P  # noqa: E402
import ops as G  # noqa: E402
import exp1_fusion_audit as A  # noqa: E402
import dsr_system as D  # noqa: E402
import confirm as C  # noqa: E402
import arms as AR  # noqa: E402
from health import eer  # noqa: E402

R, N_BOOT = Path("tbiom/reports"), 2000
BR = Path("logs/tbiom/temporal/branches")
SIX = ["CDFv2", "CDFv3", "DFD", "DFDC", "DFDCP", "DFEval24"]
SETS2 = SIX + ["KoDF", "CDFv3 reenactment"]


# =================================================================================== Table 1
def table1():
    import step1_controls as S1
    from sklearn.ensemble import GradientBoostingRegressor
    t0 = time.time()
    win, full_t = S1.load()
    dev = D.load("vjepa2_lora_B_seed42")[0]
    dev = dev[dev.v.isin(full_t.index)].reset_index(drop=True)
    base = dev.set_index("v")
    rng = np.random.default_rng(7)
    RULE = "conf"                                   # the aggregation the published report uses
    out = {"rule": RULE, "splits": []}
    for k, Fd, Sd in D.splits(dev):
        w = __import__("finish_line").fit_w([Fd], S1.VIEWS)
        pools = {"fit": S1.Pool(Fd.v, base, win), "sel": S1.Pool(Sd.v, base, win)}
        ys = {t: (Fd if t == "fit" else Sd).y.to_numpy() for t in pools}
        none = {t: pools[t].score(np.zeros((len(pools[t].v), 0), int), w) for t in pools}
        ff = np.where(Sd.source == "FFpp_val")[0]
        tau = float(eer(ys["sel"][ff], none["sel"][ff])[1])
        act = {t: np.stack([pools[t].score(np.full((len(pools[t].v), 1), j), w, RULE)[:, None]
                            for j in range(S1.N_WIN)], 1)[:, :, 0] for t in pools}
        n = len(pools["sel"].v)
        S = {"no extra clip (k=0)": none["sel"],
             "fixed window w0 (k=1)": act["sel"][:, 0],
             "random extra clip (k=1)": act["sel"][np.arange(n), rng.integers(0, S1.N_WIN, n)]}
        qs = (pools["sel"].Q[:, :, 0] / (pools["sel"].Q[:, :, 0].std() + 1e-9)
              - pools["sel"].Q[:, :, 2] / (pools["sel"].Q[:, :, 2].std() + 1e-9))
        S["quality-based choice (k=1)"] = act["sel"][np.arange(n), qs.argmax(1)]
        Xf = pools["fit"].features([set()] * len(pools["fit"].v))
        gain = np.stack([S1.loss_of(none["fit"], ys["fit"]) - S1.loss_of(act["fit"][:, j], ys["fit"])
                         for j in range(S1.N_WIN)], 1)
        model = GradientBoostingRegressor(random_state=0, n_estimators=200, max_depth=3).fit(
            np.concatenate(Xf, 0), gain.T.reshape(-1))
        pred = np.stack([model.predict(x) for x in pools["sel"].features([set()] * n)], 1)
        pick = pred.argmax(1)
        S["learned one-step choice (k=1)"] = act["sel"][np.arange(n), pick]
        L1 = np.stack([S1.loss_of(act["sel"][:, j], ys["sel"]) for j in range(S1.N_WIN)], 1)
        S["oracle, true label (k=1)"] = act["sel"][np.arange(n), L1.argmin(1)]
        null = np.stack([act["sel"][rng.permutation(n), j] for j in range(S1.N_WIN)], 1)
        Ln = np.stack([S1.loss_of(null[:, j], ys["sel"]) for j in range(S1.N_WIN)], 1)
        S["oracle, scores shuffled from other videos (k=1)"] = null[np.arange(n), Ln.argmin(1)]
        pairs = [(i, j) for i in range(S1.N_WIN) for j in range(i + 1, S1.N_WIN)]
        act2 = {p: pools["sel"].score(np.tile(np.array(p), (n, 1)), w, RULE) for p in pairs}
        S["evenly spaced extra clips w0+w2 (k=2)"] = act2[(0, 2)]
        second = np.where(pred.argsort(1)[:, -2] == pick, pred.argsort(1)[:, -3], pred.argsort(1)[:, -2])
        S["learned two-step choice (k=2)"] = np.array(
            [act2[tuple(sorted((int(a), int(b))))][i] for i, (a, b) in enumerate(zip(pick, second))])
        L2 = np.stack([S1.loss_of(act2[p], ys["sel"]) for p in pairs], 1)
        S["oracle, true label (k=2)"] = np.array([act2[pairs[j]][i] for i, j in enumerate(L2.argmin(1))])

        y = ys["sel"]; g = A.groups_for(Sd)
        draws = np.stack(__import__("score").group_boot(g, N_BOOT, 20260924 + k)).astype(float)
        ref = C.auc_weighted(S["no extra clip (k=0)"], y, draws)
        row = {"split": k, "n_videos": int(n), "n_source_groups": int(len(np.unique(g))),
               "tau": tau, "policies": {}}
        for name, s in S.items():
            d = C.auc_weighted(s, y, draws) - ref
            pred_c = (s >= tau).astype(int)
            row["policies"][name] = {
                "auroc": A.fast_auc(s[y == 0], s[y == 1]),
                "delta": float(np.nanmean(d)),
                "ci": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))],
                "fpr": float((pred_c[y == 0] == 1).mean()), "tpr": float((pred_c[y == 1] == 1).mean())}
        out["splits"].append(row)
        print(f"  acquisition split {k}: n={n} ({time.time() - t0:.0f}s)", flush=True)
    json.dump(out, open(R / "EXP_ACQUISITION_INTERVALS.json", "w"), indent=1, default=float)
    print("wrote", R / "EXP_ACQUISITION_INTERVALS.json")
    return out


# =================================================================================== Table 2
BRANCHES = {
    "vjepa2_lora_B_seed42": "V-JEPA 2 ViT-L, LoRA, B-augmented (reference)",
    "vjepa2_lora_seed42": "V-JEPA 2 ViT-L, LoRA, NOT B-augmented",
    "xclip_lora_seed42": "X-CLIP, LoRA",
    "videomae_lora_seed42": "VideoMAE, LoRA",
    "timesformer_lora_seed42": "TimeSformer, LoRA",
}


def score_arm(dev, tests, views, name, extra=None):
    res = {"views": list(views), "per_set": {}}
    for _k, Fd, _Sd in D.splits(dev):
        cl = A.cells(Fd)
        grid = [w for w in __import__("itertools").product(AR.GRID, repeat=len(views)) if max(w) == 1.0]
        f = lambda t, w: (P.masses_of(t, views, w)[:, 0, :] if len(views) == 1
                          else P.fuse(P.masses_of(t, views, w), True))
        w = max(grid, key=lambda ww: A.v31(P.p_of(f(Fd, ww)), cl)[0])
        ffp = P.p_of(f(Fd, w))
        ff = np.where(Fd.source.to_numpy() == "FFpp_val")[0]
        tau = float(eer(Fd.y.to_numpy()[ff], ffp[ff])[1]) if len(ff) else 0.5
        for tn, T in tests.items():
            if T is None:
                continue
            y = T.y.to_numpy(); p = P.p_of(f(T, w))
            pred = (p >= tau).astype(int)
            d = res["per_set"].setdefault(tn, {"auroc": [], "fpr": []})
            d["auroc"].append(A.fast_auc(p[y == 0], p[y == 1]))
            d["fpr"].append(float((pred[y == 0] == 1).mean()) if (y == 0).any() else np.nan)
    return {"views": res["views"],
            "per_set": {k: {m: float(np.mean(v[m])) for m in ("auroc", "fpr")} for k, v in res["per_set"].items()}}


def table2():
    t0 = time.time()
    out = {"fusion_rule": "PCR6+-R (direct three-source PCR6+ after evidence scaling), the manuscript's operator",
           "folds": "the five source-grouped v3.1 fitting folds", "rows": {}}
    for br, desc in BRANCHES.items():
        dev, _dproc, tests = D.load(br)
        tests = {k: tests[k] for k in SETS2 if k in tests}
        if br == "vjepa2_lora_B_seed42":
            out["rows"]["APP+OM only"] = {**score_arm(dev, tests, ("APP", "OM"), "ao"),
                                          "branch": "none", "seeds": "spatial 42"}
            out["rows"]["APP+OM+face-prior"] = {**score_arm(dev, tests, ("APP", "OM", "FP"), "aof"),
                                                "branch": "none (face-prior head is in the spatial ckpt)",
                                                "seeds": "spatial 42"}
            out["rows"]["V-JEPA 2 alone"] = {**score_arm(dev, tests, ("T",), "t"),
                                             "branch": br, "seeds": "temporal 42"}
        out["rows"][f"T+APP+OM [{desc}]"] = {**score_arm(dev, tests, ("T", "APP", "OM"), br),
                                             "branch": br, "seeds": "spatial 42, temporal 42"}
        print(f"  {br} done ({time.time() - t0:.0f}s)", flush=True)

    # shuffled-frame control: only where the reference branch actually has shuffled exports
    ob = BR / "vjepa2_lora_B_seed42" / "opinions"
    have = {p.name.replace("_shuffled.csv", "") for p in ob.glob("*_shuffled.csv")}
    out["shuffled_available_for"] = sorted(have)
    dev, _dp, tests = D.load("vjepa2_lora_B_seed42")
    sh_tests = {}
    for tn in SETS2:
        base = tn.split(" reenactment")[0]
        if base in have and tn in tests:
            t = tests[tn].copy()
            s = __import__("pandas").read_csv(ob / f"{base}_shuffled.csv").drop_duplicates("v").set_index("v")
            idx = s.index.get_indexer(t.v)
            if (idx >= 0).all():
                t["p_T"], t["u_T"] = s.p.to_numpy()[idx], s.u.to_numpy()[idx]
                sh_tests[tn] = t
    if sh_tests:
        out["rows"]["T+APP+OM, reference with SHUFFLED frames"] = {
            **score_arm(dev, sh_tests, ("T", "APP", "OM"), "shuf"),
            "branch": "vjepa2_lora_B_seed42 (shuffled-frame export)", "seeds": "spatial 42, temporal 42"}
    json.dump(out, open(R / "EXP_VIDEO_VIEW_TABLE.json", "w"), indent=1, default=float)
    print("wrote", R / "EXP_VIDEO_VIEW_TABLE.json")
    return out


if __name__ == "__main__":
    t2 = table2()
    t1 = table1()

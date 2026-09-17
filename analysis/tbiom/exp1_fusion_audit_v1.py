#!/usr/bin/env python3
"""Experiment 1 -- cached fusion audit on the warm+gate checkpoint (tbiom/docs/discern-v2-fusion-audit-brief.md).

    python analysis/tbiom/exp1_fusion_audit_v1.py --out tbiom/EXP1_FUSION_AUDIT.md   (v1, kept as run; superseded by exp1_fusion_audit.py v2)

Nothing about the detector is retrained. Every arm reads the SAME saved per-frame opinions of the selected warm+gate
checkpoint (epoch 19): Appearance, Off-manifold residual (both frozen DiCoME views) and the Bona-fide face prior, whose
raw (ungated) and gated opinions and per-frame gate are all in the export.

Arms
  1 gate       the learned participation gate (as trained)          DS(sem, art, gated prior)
  2 g=1        full prior participation                             DS(sem, art, raw prior)
  3 const g    one participation constant fitted on the fitting fold DS(sem, art, g-discounted prior)
  4 logistic / MLP stacking on the same deployment-available inputs (per-view p, vacuity, evidence, logit,
               cross-view disagreement and pairwise DS conflict)
  5 two-view   frozen DiCoME base, branch-removal control          DS(sem, art)

Protocol (fixed before any number is read)
  * Development data = the v3.1 selection data: VALmix, VALmix-v2 (in-scope subsets only) and FF++ val.
  * Grouped by SOURCE VIDEO (union-find over source-video tokens, so a fake and the genuine video it was made from,
    and every fake sharing that source, land in the same group). 5 random grouped 50/50 splits into a
    fusion-FITTING fold and a fusion-SELECTION fold.
  * Fitted arms (3, 4) are fitted -- including hyper-parameters, by grouped inner CV on log-loss -- on the fitting
    fold only. Every arm is compared on the selection fold with the v3.1 objective (macro AUROC over swap /
    reenactment / in-the-wild cells). Test sets are never used to choose anything; they are read afterwards.
  * tau (deployed threshold) per arm = EER threshold on the selection fold's FF++ val videos.
  * MLP random initialisations (5) are reported as initialisations, never as detector seeds.

Decision rule (Delta = gate - MLP on the selection-fold v3.1 score; 1000 group-bootstrap resamples per split)
  CANDIDATE CONTRIBUTION  mean Delta >= +0.005 and the 95% CI lower bound > 0 in >= 4 of 5 splits
  STACKING MATCHES/BEATS  mean Delta <  +0.005 and the 95% CI upper bound < +0.005 in >= 4 of 5 splits
  UNRESOLVED              otherwise
"""
from __future__ import annotations
import argparse, json, re, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
sys.path.insert(0, str(Path(__file__).resolve().parent))
import symbolic_overlay as SO  # noqa: E402
import select_checkpoint_v3 as V3  # noqa: E402
from video_id import video_id  # noqa: E402
from health import eer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.model_selection import GroupKFold, GroupShuffleSplit  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402

warnings.filterwarnings("ignore")
K = 2.0
S23, S2 = Path("logs/tbiom/stage23"), Path("logs/tbiom/stage2")
DEV = {"VALmix": S23 / "warmgate_e19_VALmix.csv", "VALmixV2": S23 / "warmgate_e19_VALmixV2.csv",
       "FFpp_val": S23 / "warmgate_e19_FFpp_val.csv"}
TEST = ["CDFv2", "CDFv3", "DFD", "DFDC", "DFDCP", "DFEval24", "FFIW", "KoDF", "IDForge", "DFDM"]
PROC = [(d, c) for d in ("CDFv2", "DFDCP") for c in ("jpeg40", "rs06")]
ARMS = ["gate", "g=1", "const g", "logistic", "MLP", "two-view"]
N_SPLITS, N_BOOT, N_INIT, MARGIN = 5, 1000, 5, 0.005
VIEWS = ["semantic", "artifact", "prior"]


# ----------------------------------------------------------------------------- frames -> per-view opinions
def load_frames(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    need = ["p_semantic", "u_semantic", "p_artifact", "u_artifact", "p_fsvfm", "u_fsvfm", "p_fsvfm_raw",
            "u_fsvfm_raw", "gate_fsvfm"]
    missing = [c for c in need if c not in d]
    assert not missing, f"{path}: missing {missing}"
    raw = d["key"].astype(str).str.split("::").str[-1]
    d["subset"] = raw.str.split("/").str[-3]
    d["vname"] = raw.str.split("/").str[-2]
    d["v"] = video_id(d["key"])
    return d


def alpha(p, u):
    S = K / np.clip(u, 1e-9, None)
    return np.stack([(1 - p) * S, p * S], 1)


def frame_features(d: pd.DataFrame) -> pd.DataFrame:
    """Deployment-available per-frame inputs. Prior = RAW (ungated) opinion; the gate sees p,u of all three views."""
    P = {"semantic": d.p_semantic.to_numpy(), "artifact": d.p_artifact.to_numpy(), "prior": d.p_fsvfm_raw.to_numpy()}
    U = {"semantic": d.u_semantic.to_numpy(), "artifact": d.u_artifact.to_numpy(), "prior": d.u_fsvfm_raw.to_numpy()}
    f = {}
    for v in VIEWS:
        p = np.clip(P[v], 1e-6, 1 - 1e-6)
        f[f"f_p_{v}"], f[f"f_u_{v}"] = P[v], U[v]
        f[f"f_logev_{v}"] = np.log1p(K / np.clip(U[v], 1e-9, None) - K)
        f[f"f_logit_{v}"] = np.log(p / (1 - p))
    pm = np.stack([P[v] for v in VIEWS], 1)
    f["f_std_p"] = pm.std(1)
    f["f_maxdiff_p"] = pm.max(1) - pm.min(1)
    for a, b in (("semantic", "artifact"), ("semantic", "prior"), ("artifact", "prior")):
        A, B = alpha(P[a], U[a]), alpha(P[b], U[b])
        ba, bb = (A - 1) / A.sum(1, keepdims=True), (B - 1) / B.sum(1, keepdims=True)
        f[f"f_conflict_{a[:3]}_{b[:3]}"] = ba[:, 0] * bb[:, 1] + ba[:, 1] * bb[:, 0]
    return pd.DataFrame(f, index=d.index)


def discount_frames(d: pd.DataFrame, g: float) -> tuple[np.ndarray, np.ndarray]:
    a = alpha(d.p_fsvfm_raw.to_numpy(), d.u_fsvfm_raw.to_numpy())
    a = 1 + g * (a - 1)
    S = a.sum(1)
    return a[:, 1] / S, K / S


def video_table(d: pd.DataFrame, grid) -> pd.DataFrame:
    """Per-video means of every frame quantity the arms need (the corrected-table rule: mean opinion, then fuse)."""
    x = pd.concat([d[["v", "label", "subset", "vname"]], frame_features(d)], axis=1)
    for c in ["p_semantic", "u_semantic", "p_artifact", "u_artifact", "p_fsvfm", "u_fsvfm", "p_fsvfm_raw",
              "u_fsvfm_raw", "gate_fsvfm"]:
        x[c] = d[c].to_numpy()
    for g in grid:
        x[f"pg_{g:.2f}"], x[f"ug_{g:.2f}"] = discount_frames(d, g)
    num = [c for c in x.columns if c not in ("v", "label", "subset", "vname")]
    agg = x.groupby("v", sort=False).agg(**{c: (c, "mean") for c in num}, y=("label", "max"),
                                         subset=("subset", "first"), vname=("vname", "first"))
    return agg.reset_index()


def ds(t: pd.DataFrame, prior_p=None, prior_u=None, two_view=False) -> np.ndarray:
    g = pd.DataFrame({"p_semantic": t.p_semantic, "u_semantic": t.u_semantic,
                      "p_artifact": t.p_artifact, "u_artifact": t.u_artifact})
    if two_view:
        SO.BRANCHES[:] = ["semantic", "artifact"]
    else:
        SO.BRANCHES[:] = ["semantic", "artifact", "fsvfm"]
        g["p_fsvfm"], g["u_fsvfm"] = prior_p, prior_u
    s = SO.fuse(g)
    SO.BRANCHES[:] = ["semantic", "artifact", "fsvfm"]
    return s


# ----------------------------------------------------------------------------- development data, scope, groups
def dev_source(key: str, file: str) -> str:
    raw = key.split("::")[-1]
    if file == "FFpp_val":
        return "FFpp_val"
    hit = next((s for s in ("CDFv2val", "DFDCPval", "DFEval24val") if s in raw), None)
    return hit or raw.split("/")[-4]


def in_scope(source: str, subset: str) -> bool:
    for groups in V3.EXCLUDED.values():
        if subset in groups.get(source, []):
            return False
    return True


def tokens(source: str, vname: str) -> list[str]:
    v = vname
    if source in ("CDFv2val", "CDFv3valG"):
        v = v[5:] if v.startswith("cdf2-") else v
        if re.fullmatch(r"\d{5}", v):
            return [f"cdfyt:{v}"]
        m = re.match(r"(id\d+)_(?:id\d+_)?(\d{4})", v)       # real idA_NNNN / swap idA_idB_NNNN / talking idA_NNNN_test_...
        return [f"cdf:{m[1]}_{m[2]}"] if m else [f"cdfv:{v}"]
    if source == "DFDCPval":
        p = v.removeprefix("dfdcp-").split("_")
        return [f"dfdcp:{p[0]}_{p[1]}_{p[2]}"] if len(p) == 3 else [f"dfdcp:{p[0]}_{p[2]}_{p[3]}"]
    if source == "DS11val":
        if "--" not in v:
            return [f"ds11:{v}"]
        _, ids, clips, _ = v.split("--")
        (i1, i2), (c1, c2) = ids.split("-"), clips.split("-")
        return [f"ds11:{i1}-{c1}", f"ds11:{i2}-{c2}"]
    if source == "DS20val":
        nums = re.findall(r"\d{4,}", v)
        return [f"ds20:{nums[-1]}"] if re.fullmatch(r"\d+-\d+", v) else [f"ds20:{n}" for n in nums]
    if source in ("DF40train", "FFpp_val"):
        ids = re.findall(r"(?<!\d)(\d{3})(?!\d)", v)
        return [f"ff:{i}" for i in ids] or [f"ffv:{v}"]
    return [f"{source}:{v}"]                               # DFEval24val, FFIWval: no source linkage in the names


def groups_for(t: pd.DataFrame) -> np.ndarray:
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    toks = [tokens(s, n) for s, n in zip(t.source, t.vname)]
    for ts in toks:
        for a in ts[1:]:
            parent[find(a)] = find(ts[0])
        find(ts[0])
    roots = [find(ts[0]) for ts in toks]
    ids = {r: i for i, r in enumerate(dict.fromkeys(roots))}
    return np.array([ids[r] for r in roots])


# ----------------------------------------------------------------------------- v3.1 objective on a subset of dev
def cells(t: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(type, real indices, fake indices) for every v3.1 cell present in t."""
    out = []
    ffreal = np.where((t.source == "FFpp_val") & (t.y == 0))[0]
    for typ, srcs in V3.TYPES.items():
        for src, subs in srcs.items():
            s = t.source == src
            fake = np.where(s & (t.y == 1) & ((subs == "*") | t.subset.isin([] if subs == "*" else subs)))[0]
            real = ffreal if src == "DF40train" else np.where(s & (t.y == 0))[0]
            out.append((typ, real, fake))
    return out


def fast_auc(sr: np.ndarray, sf: np.ndarray) -> float:
    if len(sr) == 0 or len(sf) == 0:
        return float("nan")
    r = rankdata(np.concatenate([sr, sf]))
    return float((r[len(sr):].sum() - len(sf) * (len(sf) + 1) / 2) / (len(sr) * len(sf)))


def v31(score: np.ndarray, cl, w: np.ndarray | None = None) -> tuple[float, dict]:
    types: dict[str, list] = {}
    for typ, ri, fi in cl:
        if w is not None:
            ri, fi = np.repeat(ri, w[ri]), np.repeat(fi, w[fi])
        types.setdefault(typ, []).append(fast_auc(score[ri], score[fi]))
    tm = {k: float(np.nanmean(v)) for k, v in types.items()}
    return float(np.mean(list(tm.values()))), tm


# ----------------------------------------------------------------------------- arms
FEATS = None


def fit_const(t, cl, grid):
    best = max(grid, key=lambda g: v31(ds(t, t[f"pg_{g:.2f}"], t[f"ug_{g:.2f}"]), cl)[0])
    return best


def inner_cv_pick(X, y, grp, make, grid):
    gkf, best, best_ll = GroupKFold(3), None, np.inf
    for hp in grid:
        lls = []
        for tr, va in gkf.split(X, y, grp):
            m = make(hp, 0).fit(X[tr], y[tr])
            lls.append(log_loss(y[va], m.predict_proba(X[va])[:, 1], labels=[0, 1]))
        if np.mean(lls) < best_ll:
            best, best_ll = hp, float(np.mean(lls))
    return best


def make_logit(C, seed):
    return LogisticRegression(C=C, class_weight="balanced", max_iter=5000)


def make_mlp(hp, seed):
    h, a = hp
    return MLPClassifier(hidden_layer_sizes=(h,), alpha=a, max_iter=800, early_stopping=True,
                         validation_fraction=0.15, n_iter_no_change=20, random_state=seed)


def balanced_idx(y, rng):
    """MLPClassifier has no class_weight: resample the minority class to balance, deterministic per seed."""
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    small, big = (i0, i1) if len(i0) < len(i1) else (i1, i0)
    return np.concatenate([big, rng.choice(small, len(big), replace=True)])


def arm_scores(t, models):
    """Score one video table with every arm; models = fitted params for this split."""
    X = models["scaler"].transform(t[FEATS].to_numpy())
    out = {"gate": ds(t, t.p_fsvfm, t.u_fsvfm), "g=1": ds(t, t.p_fsvfm_raw, t.u_fsvfm_raw),
           "const g": ds(t, t[f"pg_{models['g']:.2f}"], t[f"ug_{models['g']:.2f}"]),
           "logistic": models["logit"].predict_proba(X)[:, 1], "two-view": ds(t, two_view=True)}
    for i, m in enumerate(models["mlps"]):
        out[f"MLP#{i}"] = m.predict_proba(X)[:, 1]
    return out


# ----------------------------------------------------------------------------- tests
def test_table(ds_name, grid):
    f = S23 / f"warmgate_sel_{ds_name}.csv" if ds_name in TEST else None
    d = load_frames(f)
    keep = ~d.subset.str.contains("IP_LAP") & (d.subset != "fake-audio-driven") & ~d.subset.str.startswith("adv-")
    return video_table(d[keep].reset_index(drop=True), grid)


def main():
    global FEATS
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=Path("tbiom/EXP1_FUSION_AUDIT.md"))
    a = ap.parse_args()
    grid = [round(x, 2) for x in np.linspace(0, 1, 21)]

    # ---- development videos
    parts = []
    for name, path in DEV.items():
        d = load_frames(path)
        src = d.key.map(lambda k: dev_source(k, name))
        d = d[[in_scope(s, sub) for s, sub in zip(src, d.subset)]].reset_index(drop=True)
        t = video_table(d, grid)
        first = d.groupby("v", sort=False).key.first()
        t["source"] = [dev_source(first[v], name) for v in t.v]
        parts.append(t)
    dev = pd.concat(parts, ignore_index=True)
    FEATS = [c for c in dev.columns if c.startswith("f_")]
    grp = groups_for(dev)
    sizes = np.bincount(grp)
    print(f"dev videos {len(dev)}, source groups {len(sizes)}, largest group {sizes.max()} videos", flush=True)

    # sanity: DS replay of the gated opinions reproduces the model's own fused output at frame level
    chk = load_frames(DEV["FFpp_val"]).head(5000)
    SO.BRANCHES[:] = ["semantic", "artifact", "fsvfm"]
    replay_err = float(np.abs(SO.fuse(chk) - chk.p_fused).max())

    tests = {ds_name: test_table(ds_name, grid) for ds_name in TEST}
    procs = {}
    for dsn, c in PROC + [("CDFv2", "base"), ("DFDCP", "base")]:
        procs[(dsn, c)] = video_table(load_frames(S2 / f"p_v3_warmgate_{dsn}_{c}.csv"), grid)

    rng_boot = np.random.default_rng(12345)
    results = []
    for k in range(N_SPLITS):
        fit_i, sel_i = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=k).split(dev, dev.y, grp))
        F, Sel = dev.iloc[fit_i].reset_index(drop=True), dev.iloc[sel_i].reset_index(drop=True)
        gF, gS = grp[fit_i], grp[sel_i]
        assert not set(gF) & set(gS)
        clF, clS = cells(F), cells(Sel)
        empty = [(typ, len(r), len(f)) for typ, r, f in clS if len(r) == 0 or len(f) == 0]
        assert not empty, f"split {k}: empty selection cells {empty}"

        scaler = StandardScaler().fit(F[FEATS].to_numpy())
        XF, yF = scaler.transform(F[FEATS].to_numpy()), F.y.to_numpy()
        C = inner_cv_pick(XF, yF, gF, make_logit, [0.01, 0.1, 1.0, 10.0])
        hp = inner_cv_pick(XF, yF, gF, lambda h, s: make_mlp(h, s),
                           [(16, 1e-4), (16, 1e-2), (64, 1e-4), (64, 1e-2)])
        mlps = []
        for s in range(N_INIT):
            bi = balanced_idx(yF, np.random.default_rng(s))
            mlps.append(make_mlp(hp, s).fit(XF[bi], yF[bi]))
        models = {"scaler": scaler, "g": fit_const(F, clF, grid), "logit": make_logit(C, 0).fit(XF, yF),
                  "mlps": mlps, "C": C, "mlp_hp": hp}

        sc = arm_scores(Sel, models)
        sel_scores = {n: v31(s, clS) for n, s in sc.items()}
        mlp_vals = [sel_scores[f"MLP#{i}"][0] for i in range(N_INIT)]
        med = int(np.argsort(mlp_vals)[len(mlp_vals) // 2])

        # group bootstrap on the selection fold
        ug = np.unique(gS); pos = {g: np.where(gS == g)[0] for g in ug}
        comps = {"MLP": f"MLP#{med}", "logistic": "logistic", "g=1": "g=1", "const g": "const g", "two-view": "two-view"}
        deltas = {c: [] for c in comps}
        for _ in range(N_BOOT):
            draw = rng_boot.choice(ug, len(ug), replace=True)
            w = np.zeros(len(Sel), dtype=int)
            for g in draw:
                w[pos[g]] += 1
            base = v31(sc["gate"], clS, w)[0]
            for c, arm in comps.items():
                deltas[c].append(base - v31(sc[arm], clS, w)[0])
        boot = {c: {"delta": sel_scores["gate"][0] - sel_scores[arm][0],
                    "ci": [float(np.nanpercentile(deltas[c], 2.5)), float(np.nanpercentile(deltas[c], 97.5))]}
                for c, arm in comps.items()}

        # frozen tau per arm: EER on the selection fold's FF++ val videos
        ffs = np.where(Sel.source == "FFpp_val")[0]
        tau = {n: eer(Sel.y.to_numpy()[ffs], s[ffs])[1] for n, s in sc.items()}

        def evaluate(t):
            s = arm_scores(t, models); y = t.y.to_numpy()
            return {n: {"auroc": fast_auc(v[y == 0], v[y == 1]), "fpr": float((v[y == 0] >= tau[n]).mean()),
                        "tpr": float((v[y == 1] >= tau[n]).mean())} for n, v in s.items()}

        test_res = {dsn: evaluate(t) for dsn, t in tests.items()}
        c3 = tests["CDFv3"]; real3 = c3[c3.y == 0]
        reen = pd.concat([real3, c3[(c3.y == 1) & c3.subset.str.contains("FaceReenact|TalkingFace")]], ignore_index=True)
        swap = pd.concat([real3, c3[(c3.y == 1) & c3.subset.str.contains("FaceSwap")]], ignore_index=True)
        kd = tests["KoDF"]
        kfo = pd.concat([kd[kd.y == 0], kd[(kd.y == 1) & (kd.subset == "fake-fo")]], ignore_index=True)
        test_res["CDFv3 reenactment"], test_res["CDFv3 swap"] = evaluate(reen), evaluate(swap)
        test_res["KoDF first-order motion"] = evaluate(kfo)
        for key, t in procs.items():
            test_res[f"{key[0]} {key[1]}"] = evaluate(t)
        results.append({"split": k, "n_fit": len(F), "n_sel": len(Sel), "g": models["g"], "C": C, "mlp_hp": hp,
                        "mlp_median_init": med, "selection": sel_scores, "boot": boot, "tau": tau, "tests": test_res})
        print(f"split {k}: sel v3.1 gate {sel_scores['gate'][0]:.4f} MLP(med) {sel_scores[f'MLP#{med}'][0]:.4f} "
              f"logit {sel_scores['logistic'][0]:.4f} g=1 {sel_scores['g=1'][0]:.4f} const(g={models['g']}) "
              f"{sel_scores['const g'][0]:.4f} two-view {sel_scores['two-view'][0]:.4f} | "
              f"delta(gate-MLP) {boot['MLP']['delta']:+.4f} CI {boot['MLP']['ci'][0]:+.4f}..{boot['MLP']['ci'][1]:+.4f}",
              flush=True)

    # ---- verdict (pre-registered rule)
    d_mlp = [r["boot"]["MLP"]["delta"] for r in results]
    lo = [r["boot"]["MLP"]["ci"][0] for r in results]; hi = [r["boot"]["MLP"]["ci"][1] for r in results]
    if np.mean(d_mlp) >= MARGIN and sum(x > 0 for x in lo) >= 4:
        verdict = "CANDIDATE CONTRIBUTION: learned participation beats MLP stacking by a meaningful margin"
    elif np.mean(d_mlp) < MARGIN and sum(x < MARGIN for x in hi) >= 4:
        verdict = ("STACKING MATCHES OR BEATS: gains come from three-view representation complementarity, "
                   "not the gating machinery")
    else:
        verdict = "UNRESOLVED: the result is close; not a win and not proof that fusion is exhausted"

    json.dump({"replay_err": replay_err, "dev_videos": len(dev), "groups": int(len(sizes)),
               "largest_group": int(sizes.max()), "verdict": verdict, "splits": results},
              open(a.out.with_suffix(".json"), "w"), indent=1, default=float)
    write_md(a.out, results, verdict, replay_err, len(dev), len(sizes), int(sizes.max()))
    print("VERDICT:", verdict)


def arm_mean(results, where, arm, field):
    vals = []
    for r in results:
        blk = r["tests"][where] if where else None
        if arm == "MLP":
            vals.append(np.mean([blk[f"MLP#{i}"][field] for i in range(N_INIT)]))
        else:
            vals.append(blk[arm][field])
    return float(np.mean(vals)), float(np.std(vals))


def write_md(out, results, verdict, replay_err, n_dev, n_groups, largest):
    fmt = lambda m, s: f"{m:.4f}" + (f" ±{s:.4f}" if s >= 5e-5 else "")
    L = ["# Experiment 1 — cached fusion audit (warm+gate, epoch 19)", "",
         "Replay only: no detector parameter was trained. All arms read the same saved per-frame opinions of the selected "
         "warm+gate checkpoint (Appearance and Off-manifold residual = frozen DiCoME views; Bona-fide face prior raw and "
         "gated). Script: `analysis/tbiom/exp1_fusion_audit_v1.py`; raw numbers: `tbiom/EXP1_FUSION_AUDIT.json`.", "",
         "## Protocol", "",
         f"- Development data: the v3.1 selection data (VALmix, VALmix-v2 in-scope subsets, FF++ val) — {n_dev} videos "
         f"in {n_groups} source-video groups (largest {largest} videos). Groups join a fake with the genuine video it "
         "was made from and with every fake sharing that source (Celeb-DF target video, DFDCP target clip, DeepSpeak "
         "clip ids, FF++/DF40 identity ids); DFEval24 and FFIW names carry no source linkage and are grouped per name.",
         f"- {N_SPLITS} random grouped 50/50 splits into a fusion-fitting fold and a fusion-selection fold; no group in both.",
         "- Fitted on the fitting fold only: constant g (grid 0..1 step 0.05, v3.1 objective); logistic (C by grouped "
         "inner 3-fold CV on log-loss, balanced classes); MLP (one hidden layer, size and L2 by grouped inner CV, "
         f"class-balanced resampling, early stopping, {N_INIT} random initialisations — initialisations, not detector seeds).",
         "- Input parity: logistic and MLP get per-view p, vacuity u, log-evidence and logit for all three views (raw "
         "prior), cross-view std and range of p, and the three pairwise DS conflict masses — a superset of the gate's "
         "inputs (p, u of each view), video means of per-frame values.",
         "- Compared on the selection fold with the v3.1 objective (macro AUROC over swap / reenactment / in-the-wild "
         f"cells). Group bootstrap ({N_BOOT} resamples) for Δ = gate − arm. Test sets were read only after, never to choose.",
         "- τ per arm and split: EER threshold on the selection fold's FF++ val videos; FPR/TPR at τ are deployable rates.",
         f"- Replay check: DS over (Appearance, Off-manifold, gated prior) reproduces the exported fused output to "
         f"{replay_err:.1e}.", "",
         "Decision rule, fixed before the run: CANDIDATE CONTRIBUTION if mean Δ(gate − MLP) ≥ +0.005 and the 95% CI lower "
         "bound > 0 in ≥ 4/5 splits; STACKING MATCHES/BEATS if mean Δ < +0.005 and the CI upper bound < +0.005 in ≥ 4/5 "
         "splits; otherwise UNRESOLVED.", "",
         f"## Verdict: **{verdict}**", "",
         "## 1. Selection fold (v3.1 objective) — mean ± sd over the 5 grouped splits", "",
         "| arm | v3.1 score | swap | reenactment | in-the-wild |", "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        if arm == "MLP":
            vals = [[np.mean([r["selection"][f"MLP#{i}"][0] for i in range(N_INIT)])] +
                    [np.mean([r["selection"][f"MLP#{i}"][1][t] for i in range(N_INIT)]) for t in V3.TYPES] for r in results]
        else:
            vals = [[r["selection"][arm][0]] + [r["selection"][arm][1][t] for t in V3.TYPES] for r in results]
        m, s = np.mean(vals, 0), np.std(vals, 0)
        L.append(f"| {arm} | " + " | ".join(fmt(a, b) for a, b in zip(m, s)) + " |")
    L += ["", "MLP rows average the initialisations within each split. Spread across initialisations (selection-fold "
          "v3.1 score, per split): " + "; ".join(
              f"split {r['split']} {min(r['selection'][f'MLP#{i}'][0] for i in range(N_INIT)):.4f}–"
              f"{max(r['selection'][f'MLP#{i}'][0] for i in range(N_INIT)):.4f}" for r in results), "",
          "## 2. Gate vs each alternative — Δ = gate − arm on the selection fold, 95% group-bootstrap CI", "",
          "| split | fitted g | logistic C | MLP (hidden, L2) | vs MLP (median init) | vs logistic | vs g=1 | vs const g | vs two-view |",
          "|---|---:|---:|---|---|---|---|---|---|"]
    for r in results:
        b = r["boot"]
        cell = lambda c: f"{b[c]['delta']:+.4f} [{b[c]['ci'][0]:+.4f}, {b[c]['ci'][1]:+.4f}]"
        L.append(f"| {r['split']} | {r['g']:.2f} | {r['C']} | {tuple(r['mlp_hp'])} | {cell('MLP')} | {cell('logistic')} | "
                 f"{cell('g=1')} | {cell('const g')} | {cell('two-view')} |")
    keys = TEST + ["CDFv3 reenactment", "CDFv3 swap", "KoDF first-order motion", "CDFv2 base", "CDFv2 jpeg40",
                   "CDFv2 rs06", "DFDCP base", "DFDCP jpeg40", "DFDCP rs06"]
    L += ["", "## 3. Test sets (read after selection; fitted arms averaged over the 5 splits)", "",
          "### 3a. Video AUROC", "", "| test | " + " | ".join(ARMS) + " |", "|---|" + "---:|" * len(ARMS)]
    for kk in keys:
        L.append(f"| {kk} | " + " | ".join(f"{arm_mean(results, kk, arm, 'auroc')[0]:.4f}" for arm in ARMS) + " |")
    L.append("| **OOD mean (10 sets)** | " + " | ".join(
        f"**{np.mean([arm_mean(results, t, arm, 'auroc')[0] for t in TEST]):.4f}**" for arm in ARMS) + " |")
    L += ["", "KoDF here is FSGAN, DFL, FaceSwap and first-order motion (the brief lists KoDF as reenactment; only its "
          "first-order-motion subset is, so that subset is reported separately).", "",
          "### 3b. Real-side FPR at frozen τ (and TPR)", "", "| test | " + " | ".join(ARMS) + " |", "|---|" + "---|" * len(ARMS)]
    for kk in keys:
        L.append(f"| {kk} | " + " | ".join(
            f"{arm_mean(results, kk, arm, 'fpr')[0]:.3f} / {arm_mean(results, kk, arm, 'tpr')[0]:.3f}" for arm in ARMS) + " |")
    L += ["", "## 4. Limitations", "",
          "- **Replay cannot establish the best fusion-training method.** The prior head was trained with the gate "
          "present, so these arms measure how alternative fusions behave on branches shaped by that gate. A prior head "
          "trained under g = 1, a constant, or a stacker could differ.",
          "- **One warm+gate seed is one seed.** MLP initialisations and fold splits vary the fusion fit and the "
          "selection sample; they are not independent detector seeds.",
          "- Groups for DFEval24 and FFIW validation videos are per name (no source linkage recoverable from names).",
          "- The OOD mean is reported for context only; the verdict uses the selection fold alone."]
    out.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()

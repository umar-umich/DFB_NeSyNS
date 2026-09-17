#!/usr/bin/env python3
"""Cached fusion audit, v2 (tbiom/docs/discern-v2-fusion-audit-brief.md, decisions of 2026-09-16).

    python analysis/tbiom/exp1_fusion_audit.py --run warmgate  --out tbiom/EXP1_FUSION_AUDIT_v2.md
    python analysis/tbiom/exp1_fusion_audit.py --run warmgateB --out tbiom/EXP2_FUSION_AUDIT.md

CONDITIONAL ON THE DEVELOPMENT-SELECTED CHECKPOINT. The audit replays the saved per-frame opinions of the checkpoint the
v3.1 rule already selected for the run (its selection record names it). Nothing is retrained; the question is how
alternative fusions behave on THAT checkpoint's branches.

Changes from v1 (v1 kept as exp1_fusion_audit_v1.py and tbiom/EXP1_FUSION_AUDIT.md):
  * MLP arithmetic reconciled. v1's table averaged the 5 initialisations; v1's Delta and verdict used the median
    initialisation. Both are now reported side by side, each with its own paired bootstrap. The REGISTERED comparator
    for the verdict is unchanged (median initialisation, rich inputs).
  * Paired, source-grouped bootstrap made explicit and self-tested: one group resample per draw, every arm scored on
    that same draw; an arm compared with itself must give Delta == 0 on every draw.
  * Matched-input stacking controls added: logistic and MLP on exactly the gate's inputs (p and vacuity u of the three
    views). The richer-input stackers (v1) are kept as separate arms.
  * Clean and processed development performance reported separately. Processed development = the same selection-fold
    videos under JPEG q40 and resize 0.60. Fusion parameters are fitted on the CLEAN fitting fold only; processed
    development numbers are reported, never used to fit, select or decide.

Protocol, decision rule and scope are otherwise v1's (see the v1 docstring, reproduced in the report).
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
TEST = ["CDFv2", "CDFv3", "DFD", "DFDC", "DFDCP", "DFEval24", "FFIW", "KoDF", "IDForge", "DFDM"]
PROC = [(d, c) for d in ("CDFv2", "DFDCP") for c in ("base", "jpeg40", "rs06")]
DEV_PROC = ["jpeg40", "rs06"]
N_SPLITS, N_BOOT, N_INIT, MARGIN = 5, 1000, 5, 0.005
VIEWS = ["semantic", "artifact", "prior"]
MATCHED = ["f_p_semantic", "f_u_semantic", "f_p_artifact", "f_u_artifact", "f_p_prior", "f_u_prior"]
TABLE_ARMS = ["gate", "g=1", "const g", "logistic (rich)", "MLP (rich)", "logistic (matched)", "MLP (matched)", "two-view"]


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
    """Deployment-available per-frame inputs. Prior = RAW (ungated) opinion; the gate sees p, u of all three views."""
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


def discount_frames(d: pd.DataFrame, g: float):
    a = 1 + g * (alpha(d.p_fsvfm_raw.to_numpy(), d.u_fsvfm_raw.to_numpy()) - 1)
    S = a.sum(1)
    return a[:, 1] / S, K / S


def video_table(d: pd.DataFrame, grid) -> pd.DataFrame:
    """Per-video means of every frame quantity (the corrected-table rule: mean opinion, then fuse)."""
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
        g["p_fsvfm"], g["u_fsvfm"] = np.asarray(prior_p), np.asarray(prior_u)
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
    return not any(subset in groups.get(source, []) for groups in V3.EXCLUDED.values())


def tokens(source: str, vname: str) -> list[str]:
    v = vname
    if source in ("CDFv2val", "CDFv3valG"):
        v = v[5:] if v.startswith("cdf2-") else v
        if re.fullmatch(r"\d{5}", v):
            return [f"cdfyt:{v}"]
        m = re.match(r"(id\d+)_(?:id\d+_)?(\d{4})", v)        # real idA_NNNN / swap idA_idB_NNNN / talking idA_NNNN_test_...
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
    return [f"{source}:{v}"]                                # DFEval24val, FFIWval: no source linkage in the names


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


def dev_tables(files: dict[str, Path], grid) -> pd.DataFrame:
    parts = []
    for name, path in files.items():
        d = load_frames(path)
        src = d.key.map(lambda k: dev_source(k, name))
        d = d[[in_scope(s, sub) for s, sub in zip(src, d.subset)]].reset_index(drop=True)
        t = video_table(d, grid)
        first = d.groupby("v", sort=False).key.first()
        t["source"] = [dev_source(first[v], name) for v in t.v]
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


# ----------------------------------------------------------------------------- v3.1 objective on a subset of dev
def cells(t: pd.DataFrame):
    out = []
    ffreal = np.where((t.source == "FFpp_val") & (t.y == 0))[0]
    for typ, srcs in V3.TYPES.items():
        for src, subs in srcs.items():
            s = t.source == src
            fake = np.where(s & (t.y == 1) & ((subs == "*") | t.subset.isin([] if subs == "*" else subs)))[0]
            real = ffreal if src == "DF40train" else np.where(s & (t.y == 0))[0]
            out.append((typ, real, fake))
    return out


def fast_auc(sr, sf) -> float:
    if len(sr) == 0 or len(sf) == 0:
        return float("nan")
    r = rankdata(np.concatenate([sr, sf]))
    return float((r[len(sr):].sum() - len(sf) * (len(sf) + 1) / 2) / (len(sr) * len(sf)))


def v31(score, cl, w=None):
    types: dict[str, list] = {}
    for typ, ri, fi in cl:
        if w is not None:
            ri, fi = np.repeat(ri, w[ri]), np.repeat(fi, w[fi])
        types.setdefault(typ, []).append(fast_auc(score[ri], score[fi]))
    tm = {k: float(np.nanmean(v)) for k, v in types.items()}
    return float(np.mean(list(tm.values()))), tm


# ----------------------------------------------------------------------------- fitted arms
def fit_const(t, cl, grid):
    return max(grid, key=lambda g: v31(ds(t, t[f"pg_{g:.2f}"], t[f"ug_{g:.2f}"]), cl)[0])


def inner_cv_pick(X, y, grp, make, grid):
    gkf, best, best_ll = GroupKFold(3), None, np.inf
    for hp in grid:
        lls = []
        for tr, va in gkf.split(X, y, grp):
            m = make(hp, 0)
            if isinstance(m, MLPClassifier):
                bi = balanced_idx(y[tr], np.random.default_rng(0)); m.fit(X[tr][bi], y[tr][bi])
            else:
                m.fit(X[tr], y[tr])
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
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    small, big = (i0, i1) if len(i0) < len(i1) else (i1, i0)
    return np.concatenate([big, rng.choice(small, len(big), replace=True)])


def fit_stackers(F, gF, feats):
    scaler = StandardScaler().fit(F[feats].to_numpy())
    X, y = scaler.transform(F[feats].to_numpy()), F.y.to_numpy()
    C = inner_cv_pick(X, y, gF, make_logit, [0.01, 0.1, 1.0, 10.0])
    hp = inner_cv_pick(X, y, gF, make_mlp, [(16, 1e-4), (16, 1e-2), (64, 1e-4), (64, 1e-2)])
    mlps = []
    for s in range(N_INIT):
        bi = balanced_idx(y, np.random.default_rng(s))
        mlps.append(make_mlp(hp, s).fit(X[bi], y[bi]))
    return {"feats": feats, "scaler": scaler, "logit": make_logit(C, 0).fit(X, y), "mlps": mlps, "C": C, "hp": hp}


def arm_scores(t, M):
    out = {"gate": ds(t, t.p_fsvfm, t.u_fsvfm), "g=1": ds(t, t.p_fsvfm_raw, t.u_fsvfm_raw),
           "const g": ds(t, t[f"pg_{M['g']:.2f}"], t[f"ug_{M['g']:.2f}"]), "two-view": ds(t, two_view=True)}
    for tag in ("rich", "matched"):
        st = M[tag]; X = st["scaler"].transform(t[st["feats"]].to_numpy())
        out[f"logistic ({tag})"] = st["logit"].predict_proba(X)[:, 1]
        for i, m in enumerate(st["mlps"]):
            out[f"MLP ({tag})#{i}"] = m.predict_proba(X)[:, 1]
    return out


def mlp_mean(d: dict, tag: str, field=None):
    vals = [d[f"MLP ({tag})#{i}"] for i in range(N_INIT)]
    return float(np.mean([v if field is None else v[field] for v in vals]))


# ----------------------------------------------------------------------------- run
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="arm name in select_checkpoint RUN_DIR (warmgate, warmgateB)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    grid = [round(x, 2) for x in np.linspace(0, 1, 21)]
    sel = json.load(open(S23 / f"{a.run}_selection_v3.json"))["selected"]
    dev_tag = sel["tag"]                                   # e.g. warmgate_e19 -- the development-selected checkpoint
    test_tag = (S23 / f"{a.run}_v3_tag.txt").read_text().strip()
    proc_pre = (S23 / f"{a.run}_v3_proc_prefix.txt").read_text().strip()
    # v1 order: group ids are numbered in order of appearance and GroupShuffleSplit assigns by id, so this order
    # fixes the splits. Keeping v1's order makes v1 and v2 use IDENTICAL fitting/selection folds.
    names = ("VALmix", "VALmixV2", "FFpp_val")
    dev = dev_tables({n: S23 / f"{dev_tag}_{n}.csv" for n in names}, grid)
    rich = [c for c in dev.columns if c.startswith("f_")]
    grp = groups_for(dev); sizes = np.bincount(grp)
    print(f"{a.run}: selected {dev_tag} (epoch {sel['epoch']}); dev videos {len(dev)}, groups {len(sizes)}, "
          f"largest {sizes.max()}", flush=True)

    devproc = {}
    for c in DEV_PROC:
        files = {n: S2 / f"p_dev_{dev_tag}_{n}_{c}.csv" for n in names}
        if all(f.is_file() for f in files.values()):
            devproc[c] = dev_tables(files, grid).set_index("v")
    print("processed development conditions available:", list(devproc) or "none", flush=True)

    chk = load_frames(S23 / f"{dev_tag}_FFpp_val.csv").head(5000)
    SO.BRANCHES[:] = ["semantic", "artifact", "fsvfm"]
    replay_err = float(np.abs(SO.fuse(chk) - chk.p_fused).max())

    def test_table(path):
        d = load_frames(path)
        keep = ~d.subset.str.contains("IP_LAP") & (d.subset != "fake-audio-driven") & ~d.subset.str.startswith("adv-")
        return video_table(d[keep].reset_index(drop=True), grid)
    tests = {t: test_table(S23 / f"{test_tag}_{t}.csv") for t in TEST}
    procs = {f"{d} {c}": test_table(S2 / f"p_{proc_pre}{d}_{c}.csv") for d, c in PROC}

    rng_boot = np.random.default_rng(12345)
    results = []
    for k in range(N_SPLITS):
        fit_i, sel_i = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=k).split(dev, dev.y, grp))
        F, Sel = dev.iloc[fit_i].reset_index(drop=True), dev.iloc[sel_i].reset_index(drop=True)
        gF, gS = grp[fit_i], grp[sel_i]
        assert not set(gF) & set(gS), "a source group is in both folds"
        clF, clS = cells(F), cells(Sel)
        assert all(len(r) and len(f) for _, r, f in clS), f"split {k}: empty selection cell"

        M = {"g": fit_const(F, clF, grid), "rich": fit_stackers(F, gF, rich), "matched": fit_stackers(F, gF, MATCHED)}
        sc = arm_scores(Sel, M)
        sel_scores = {n: v31(s, clS) for n, s in sc.items()}
        med = {t: int(np.argsort([sel_scores[f"MLP ({t})#{i}"][0] for i in range(N_INIT)])[N_INIT // 2])
               for t in ("rich", "matched")}

        # ---- paired, source-grouped bootstrap
        ug = np.unique(gS); pos = {g: np.where(gS == g)[0] for g in ug}
        assert sum(len(p) for p in pos.values()) == len(Sel)
        singles = {"MLP (rich) median init": f"MLP (rich)#{med['rich']}", "logistic (rich)": "logistic (rich)",
                   "MLP (matched) median init": f"MLP (matched)#{med['matched']}", "logistic (matched)": "logistic (matched)",
                   "g=1": "g=1", "const g": "const g", "two-view": "two-view"}
        deltas = {c: [] for c in list(singles) + ["MLP (rich) init mean", "MLP (matched) init mean", "self-test"]}
        for _ in range(N_BOOT):
            draw = rng_boot.choice(ug, len(ug), replace=True)
            w = np.zeros(len(Sel), dtype=int)
            for g in draw:
                w[pos[g]] += 1                              # a group's videos always move together
            base = v31(sc["gate"], clS, w)[0]               # every arm below is scored on THIS SAME draw (paired)
            deltas["self-test"].append(base - v31(sc["gate"], clS, w)[0])
            for c, arm in singles.items():
                deltas[c].append(base - v31(sc[arm], clS, w)[0])
            for t in ("rich", "matched"):
                deltas[f"MLP ({t}) init mean"].append(
                    base - np.mean([v31(sc[f"MLP ({t})#{i}"], clS, w)[0] for i in range(N_INIT)]))
        assert np.all(np.asarray(deltas["self-test"]) == 0.0), "paired bootstrap self-test failed"
        point = {**{c: sel_scores["gate"][0] - sel_scores[arm][0] for c, arm in singles.items()},
                 **{f"MLP ({t}) init mean": sel_scores["gate"][0] - mlp_mean(
                     {kk: vv[0] for kk, vv in sel_scores.items()}, t) for t in ("rich", "matched")}}
        boot = {c: {"delta": point[c], "ci": [float(np.nanpercentile(deltas[c], 2.5)),
                                               float(np.nanpercentile(deltas[c], 97.5))]} for c in point}

        # ---- frozen tau (clean selection-fold FF++ val) and deployable rates
        ffs = np.where(Sel.source == "FFpp_val")[0]
        tau = {n: eer(Sel.y.to_numpy()[ffs], s[ffs])[1] for n, s in sc.items()}

        def evaluate(t):
            s = arm_scores(t, M); y = t.y.to_numpy()
            return {n: {"auroc": fast_auc(v[y == 0], v[y == 1]), "fpr": float((v[y == 0] >= tau[n]).mean()),
                        "tpr": float((v[y == 1] >= tau[n]).mean())} for n, v in s.items()}

        # ---- processed development: SAME selection-fold videos, scored with the clean-fitted arms
        dev_proc = {}
        for c, P in devproc.items():
            keep = Sel.v[Sel.v.isin(P.index)]
            Pt = P.loc[keep].reset_index()
            Pt["source"] = Sel.set_index("v").loc[keep, "source"].to_numpy()
            s = arm_scores(Pt, M); clP = cells(Pt)
            ffp = np.where(Pt.source == "FFpp_val")[0]; yP = Pt.y.to_numpy()
            dev_proc[c] = {"n": len(Pt), **{n: {"v31": v31(v, clP),
                                                "ffpp_fpr": float((v[ffp][yP[ffp] == 0] >= tau[n]).mean())}
                                            for n, v in s.items()}}

        test_res = {t: evaluate(tb) for t, tb in tests.items()}
        c3 = tests["CDFv3"]; real3 = c3[c3.y == 0]
        test_res["CDFv3 reenactment"] = evaluate(pd.concat(
            [real3, c3[(c3.y == 1) & c3.subset.str.contains("FaceReenact|TalkingFace")]], ignore_index=True))
        test_res["CDFv3 swap"] = evaluate(pd.concat([real3, c3[(c3.y == 1) & c3.subset.str.contains("FaceSwap")]],
                                                    ignore_index=True))
        kd = tests["KoDF"]
        test_res["KoDF first-order motion"] = evaluate(pd.concat([kd[kd.y == 0], kd[(kd.y == 1) & (kd.subset == "fake-fo")]],
                                                                 ignore_index=True))
        for key, tb in procs.items():
            test_res[key] = evaluate(tb)
        results.append({"split": k, "g": M["g"], "stackers": {t: {"C": M[t]["C"], "mlp_hp": M[t]["hp"],
                        "median_init": med[t]} for t in ("rich", "matched")},
                        "selection": sel_scores, "boot": boot, "tau": tau, "dev_proc": dev_proc, "tests": test_res})
        b = boot
        print(f"split {k}: gate {sel_scores['gate'][0]:.4f} | d(MLP rich med) {b['MLP (rich) median init']['delta']:+.4f} "
              f"d(MLP rich mean) {b['MLP (rich) init mean']['delta']:+.4f} d(logit rich) {b['logistic (rich)']['delta']:+.4f} "
              f"d(logit matched) {b['logistic (matched)']['delta']:+.4f} d(MLP matched mean) "
              f"{b['MLP (matched) init mean']['delta']:+.4f}", flush=True)

    verdicts = {c: verdict(results, c) for c in ("MLP (rich) median init", "MLP (rich) init mean",
                                                  "MLP (matched) init mean", "logistic (rich)", "logistic (matched)")}
    meta = {"run": a.run, "selected": {k: sel[k] for k in ("tag", "checkpoint", "epoch", "score", "sha256")},
            "test_tag": test_tag, "proc_prefix": proc_pre, "replay_err": replay_err, "dev_videos": len(dev),
            "groups": int(len(sizes)), "largest_group": int(sizes.max()), "dev_proc": list(devproc),
            "verdicts": verdicts}
    json.dump({**meta, "splits": results}, open(a.out.with_suffix(".json"), "w"), indent=1, default=float)
    write_md(a.out, meta, results)
    print("REGISTERED VERDICT (gate vs MLP rich, median init):", verdicts["MLP (rich) median init"])


def verdict(results, comp):
    d = [r["boot"][comp]["delta"] for r in results]
    lo = [r["boot"][comp]["ci"][0] for r in results]; hi = [r["boot"][comp]["ci"][1] for r in results]
    if np.mean(d) >= MARGIN and sum(x > 0 for x in lo) >= 4:
        return "CANDIDATE CONTRIBUTION"
    if np.mean(d) < MARGIN and sum(x < MARGIN for x in hi) >= 4:
        return "STACKING MATCHES OR BEATS"
    return "UNRESOLVED"


# ----------------------------------------------------------------------------- report
def arm_val(block, arm, field):
    if arm.startswith("MLP ("):
        return float(np.mean([block[f"{arm}#{i}"][field] for i in range(N_INIT)]))
    return float(block[arm][field])


def write_md(out, meta, results):
    pm = lambda vals: f"{np.mean(vals):.4f}" + (f" ±{np.std(vals):.4f}" if np.std(vals) >= 5e-5 else "")
    sel = meta["selected"]
    L = [f"# Cached fusion audit v2 — run `{meta['run']}`", "",
         f"**Conditional on the development-selected checkpoint:** `{sel['tag']}` (epoch {sel['epoch']}, v3.1 score "
         f"{sel['score']:.4f}, sha256 `{str(sel['sha256'])[:16]}`), chosen earlier by the unchanged v3.1 rule. Replay "
         "only: no detector parameter is trained. The audit asks how alternative fusions behave on this checkpoint's "
         "branches. Script `analysis/tbiom/exp1_fusion_audit.py` (v2); raw numbers in the matching `.json`.", "",
         "## Protocol", "",
         f"- Development data: v3.1 selection data (FF++ val, VALmix, VALmix-v2 in-scope subsets), {meta['dev_videos']} "
         f"videos in {meta['groups']} source-video groups (largest {meta['largest_group']}).",
         f"- {N_SPLITS} random grouped 50/50 splits into a fitting fold and a selection fold (no group in both). Fusion "
         "parameters — constant g, logistic C, MLP size/L2 and weights — are fitted on the CLEAN fitting fold only.",
         "- Arms: learned gate · g = 1 · fitted constant g · logistic and MLP stacking on RICH inputs (per-view p, "
         "vacuity, log-evidence, logit; cross-view std/range; pairwise DS conflict) · logistic and MLP stacking on "
         "MATCHED inputs (exactly the gate's: p and vacuity of the three views) · two-view (prior removed).",
         f"- Comparison on the selection fold with the v3.1 objective. **Paired, source-grouped bootstrap** ({N_BOOT} "
         "draws per split): each draw resamples source groups with replacement (a group's videos move together) and "
         "scores every arm on that same draw; a gate-vs-gate self-test returned Δ = 0 on every draw.",
         "- MLP: 5 random initialisations per fit, reported both as the initialisation average and as the median "
         "initialisation (the v1-registered comparator). Initialisations are not detector seeds.",
         "- τ per arm and split: EER threshold on the CLEAN selection-fold FF++ val videos, frozen for everything else.",
         f"- Processed development ({', '.join(meta['dev_proc']) or 'not available'}): the same selection-fold videos "
         "under JPEG q40 / resize 0.60, scored with the clean-fitted arms. Reported only; never used to fit, select or decide.",
         f"- Replay check: DS(Appearance, Off-manifold, gated prior) reproduces the exported fused output to {meta['replay_err']:.1e}.",
         "", "**Decision rule (unchanged from v1).** Registered comparison: gate vs MLP stacking, rich inputs, median "
         "initialisation. CANDIDATE CONTRIBUTION if mean Δ ≥ +0.005 and 95% CI lower bound > 0 in ≥ 4/5 splits; "
         "STACKING MATCHES OR BEATS if mean Δ < +0.005 and CI upper bound < +0.005 in ≥ 4/5 splits; otherwise "
         "UNRESOLVED. The same rule applied to the other comparators is reported as secondary information, not as a new rule.", "",
         "## Verdicts", "", "| comparison (Δ = gate − comparator) | mean Δ | CI>0 splits | CI upper<+0.005 splits | verdict |",
         "|---|---:|---:|---:|---|"]
    for comp, v in meta["verdicts"].items():
        d = [r["boot"][comp]["delta"] for r in results]
        L.append(f"| {comp}{' **(registered)**' if comp == 'MLP (rich) median init' else ''} | {np.mean(d):+.4f} | "
                 f"{sum(r['boot'][comp]['ci'][0] > 0 for r in results)}/5 | "
                 f"{sum(r['boot'][comp]['ci'][1] < MARGIN for r in results)}/5 | {v} |")
    L += ["", "## 1. Clean development — selection fold, v3.1 objective (mean ± sd over 5 splits)", "",
          "| arm | v3.1 score | swap | reenactment | in-the-wild |", "|---|---:|---:|---:|---:|"]
    for arm in TABLE_ARMS:
        rows = []
        for r in results:
            if arm.startswith("MLP ("):
                rows.append([np.mean([r["selection"][f"{arm}#{i}"][0] for i in range(N_INIT)])] +
                            [np.mean([r["selection"][f"{arm}#{i}"][1][t] for i in range(N_INIT)]) for t in V3.TYPES])
            else:
                rows.append([r["selection"][arm][0]] + [r["selection"][arm][1][t] for t in V3.TYPES])
        rows = np.array(rows)
        L.append(f"| {arm} | " + " | ".join(pm(rows[:, j]) for j in range(rows.shape[1])) + " |")
    L += ["", "MLP rows average the initialisations. Median-initialisation scores per split: rich " +
          ", ".join(f"{r['selection'][f'MLP (rich)#{r['stackers']['rich']['median_init']}'][0]:.4f}" for r in results) +
          "; matched " + ", ".join(f"{r['selection'][f'MLP (matched)#{r['stackers']['matched']['median_init']}'][0]:.4f}"
                                   for r in results) + ".", "",
          "### Arithmetic reconciliation (v1)", "",
          "v1's table row \"MLP\" averaged the five initialisations, while v1's Δ and verdict used the median "
          "initialisation; the two differ because the median initialisation is not the average. Both are reported "
          "below with their own paired bootstrap, so every Δ equals the difference of two numbers shown in this report.", "",
          "## 2. Δ = gate − comparator on the clean selection fold, 95% paired group-bootstrap CI", ""]
    comps = ["MLP (rich) median init", "MLP (rich) init mean", "logistic (rich)", "MLP (matched) median init",
             "MLP (matched) init mean", "logistic (matched)", "g=1", "const g", "two-view"]
    L += ["| split | g | " + " | ".join(comps) + " |", "|---|---:|" + "---|" * len(comps)]
    for r in results:
        L.append(f"| {r['split']} | {r['g']:.2f} | " + " | ".join(
            f"{r['boot'][c]['delta']:+.4f} [{r['boot'][c]['ci'][0]:+.4f}, {r['boot'][c]['ci'][1]:+.4f}]" for c in comps) + " |")
    if meta["dev_proc"]:
        L += ["", "## 3. Processed development — same selection-fold videos, clean-fitted arms (reported, not used)", "",
              "| arm | " + " | ".join(f"{c} v3.1 | {c} FF++ val real FPR@τ" for c in meta["dev_proc"]) + " |",
              "|---|" + "---:|---:|" * len(meta["dev_proc"])]
        for arm in TABLE_ARMS:
            cellsv = []
            for c in meta["dev_proc"]:
                v = [arm_val({kk: {"x": vv["v31"][0]} for kk, vv in r["dev_proc"][c].items() if kk != "n"}, arm, "x")
                     for r in results]
                f = [arm_val({kk: {"x": vv["ffpp_fpr"]} for kk, vv in r["dev_proc"][c].items() if kk != "n"}, arm, "x")
                     for r in results]
                cellsv += [pm(v), f"{np.mean(f):.3f}"]
            L.append(f"| {arm} | " + " | ".join(cellsv) + " |")
    keys = TEST + ["CDFv3 reenactment", "CDFv3 swap", "KoDF first-order motion"] + [f"{d} {c}" for d, c in PROC]
    L += ["", "## 4. Test sets — read after the development analysis, never used to choose (fitted arms: mean over splits)",
          "", "### 4a. Video AUROC", "", "| test | " + " | ".join(TABLE_ARMS) + " |", "|---|" + "---:|" * len(TABLE_ARMS)]
    for kk in keys:
        L.append(f"| {kk} | " + " | ".join(f"{np.mean([arm_val(r['tests'][kk], arm, 'auroc') for r in results]):.4f}"
                                          for arm in TABLE_ARMS) + " |")
    L.append("| **OOD mean (10 sets)** | " + " | ".join(
        f"**{np.mean([np.mean([arm_val(r['tests'][t], arm, 'auroc') for r in results]) for t in TEST]):.4f}**"
        for arm in TABLE_ARMS) + " |")
    L += ["", "### 4b. Real-side FPR / TPR at frozen τ", "", "| test | " + " | ".join(TABLE_ARMS) + " |",
          "|---|" + "---|" * len(TABLE_ARMS)]
    for kk in keys:
        L.append(f"| {kk} | " + " | ".join(
            f"{np.mean([arm_val(r['tests'][kk], arm, 'fpr') for r in results]):.3f} / "
            f"{np.mean([arm_val(r['tests'][kk], arm, 'tpr') for r in results]):.3f}" for arm in TABLE_ARMS) + " |")
    L += ["", "CDFv2 and DeepfakeEval2024 (DFEval24) have informed development (their validation halves are in the "
          "selection data and their test halves appear in earlier tables); no test set here is untouched confirmation "
          "data. UADFV and FaceShifter remain uninspected.", "",
          "## Limitations", "",
          "- Conditional on the development-selected checkpoint; replay cannot establish the best fusion-TRAINING method "
          "(the prior head was trained with the gate present).",
          "- One detector seed per run. MLP initialisations and fold splits are not independent detector seeds.",
          "- DFEval24 and FFIW validation videos are grouped per name (no source linkage recoverable).",
          "- KoDF is mostly face swaps; its first-order-motion subset is the reenactment part and is shown separately."]
    out.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()

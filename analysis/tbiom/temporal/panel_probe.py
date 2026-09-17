#!/usr/bin/env python3
"""Temporal panel on stored frames: X-CLIP, TimeSformer, VideoMAE, V-JEPA 2 and frame-consistency features.

    OMP_NUM_THREADS=8 python analysis/tbiom/temporal/panel_probe.py --out tbiom/reports/EXP3B_TEMPORAL_PANEL.md

Same protocol for every family (the X-CLIP probe's, analysis/tbiom/xclip/probe_and_fuse.py):
  * evidential probe (LayerNorm -> 128 GELU -> 2, softplus, DISCERN's EDL loss) on FF++ TRAIN features only, one seed;
    epoch chosen by the v3.1 development rule on the full development set;
  * ORDERED and SHUFFLED-frame features trained and reported separately -- equal results mean the family's value is
    not temporal order;
  * fusion with the development-selected warm+gate views (APP = Appearance, OM = Off-manifold residual, FP = Bona-fide
    face prior, raw): F+OM+FP and F+APP+OM as equal DS replay and as logistic stacking on matched inputs (p, vacuity),
    plus a four-view stack APP+OM+FP+F; logistic fitted only on the grouped fitting folds of the fusion audit
    (identical splits); compared on the selection folds with the v3.1 objective;
  * processed development (JPEG q40 / resize 0.60, same selection-fold videos) reported, never used;
  * spatial control: plain CLIP ViT-B/16 frame mean (from the X-CLIP extraction), same arms;
  * test sets read afterwards; they choose nothing.
Consistency features (12 per video) are standardised with FF++-train median/IQR; missing values imputed with the median.
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(8)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, "/data/umar/Repos/DISCERN_Ext")
import symbolic_overlay as SO  # noqa: E402
import exp1_fusion_audit as A  # noqa: E402
from health import eer  # noqa: E402
from src.losses.evidential_loss import evidential_loss_discern_ext  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

S23, S2 = Path("logs/tbiom/stage23"), Path("logs/tbiom/stage2")
XF, TF_ = Path("logs/tbiom/xclip/feats"), Path("logs/tbiom/temporal/feats")
# family -> (directory, ordered key, shuffled key)
FAMILIES = {"xclip": (XF, "xclip_temporal", "xclip_shuffled"), "timesformer": (TF_ / "timesformer", "ordered", "shuffled"),
            "videomae": (TF_ / "videomae", "ordered", "shuffled"), "vjepa2": (TF_ / "vjepa2", "ordered", "shuffled"),
            "consistency": (TF_ / "consistency", "ordered", "shuffled"), "spatial": (XF, "clip_framemean", "clip_framemean")}
DEV_NAMES = ("VALmix", "VALmixV2", "FFpp_val")
# Trained temporal branches (Experiment 4): opinions exported by train_branch.py at the development-selected epoch.
BRANCHES = {"videomae_branch": Path("logs/tbiom/temporal/branches/videomae_seed42/opinions"),
            "vjepa2_branch": Path("logs/tbiom/temporal/branches/vjepa2_seed42/opinions")}
PROC_SPEC = {"jpeg40": "jpeg40", "rs06": "resize06"}
EPOCHS, SEED, K = 40, 0, 2
REN = {"p_fsvfm_raw": "p_prior", "u_fsvfm_raw": "u_prior", "p_fsvfm": "p_gated", "u_fsvfm": "u_gated"}


def load(fam, ds, proc=None):
    d, ko, ks = FAMILIES[fam]; tag = ds + (f"_{proc}" if proc else "")
    files = sorted(glob.glob(str(d / f"{tag}.s*of*.npz"))) or sorted(glob.glob(str(d / f"{tag}.npz")))
    if not files:
        return None
    n_exp = int(Path(files[0]).name.split("of")[-1].split(".")[0]) if ".s" in Path(files[0]).name else 1
    if len(files) != n_exp:
        return None
    zs = [np.load(f, allow_pickle=True) for f in files]
    df = pd.DataFrame({"v": np.concatenate([z["vid"] for z in zs]), "y": np.concatenate([z["label"] for z in zs])})
    return df, np.concatenate([z[ko] for z in zs]).astype(np.float32), np.concatenate([z[ks] for z in zs]).astype(np.float32)


class Probe(nn.Module):
    def __init__(self, d):
        super().__init__(); self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 2))

    def forward(self, x):
        return F.softplus(self.net(x))


def opin(model, X):
    with torch.no_grad():
        a = model(torch.as_tensor(X)).double().numpy() + 1
    return a[:, 1] / a.sum(1), K / a.sum(1)


def train(Xtr, ytr, Xdev, cl):
    torch.manual_seed(SEED)
    x, y = torch.as_tensor(Xtr), torch.as_tensor(ytr).long()
    w = torch.where(y == 1, 1.0 / (y == 1).sum(), 1.0 / (y == 0).sum())
    model = Probe(Xtr.shape[1]); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    best = (-1.0, None, None)
    for ep in range(EPOCHS):
        model.train()
        idx = torch.multinomial(w, len(y), replacement=True, generator=torch.Generator().manual_seed(SEED + ep))
        for b in range(0, len(idx), 128):
            j = idx[b:b + 128]
            loss = evidential_loss_discern_ext(model(x[j]), y[j], 2, ep, EPOCHS).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); sc = A.v31(opin(model, Xdev)[0], cl)[0]
        if sc > best[0] + 1e-6:
            best = (sc, ep, {k: v.clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2]); model.eval()
    return model, best[1], best[0]


def ds_views(t, views):
    SO.BRANCHES[:] = list(views); s = SO.fuse(t); SO.BRANCHES[:] = ["semantic", "artifact", "fsvfm"]; return s


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=Path("tbiom/reports/EXP3B_TEMPORAL_PANEL.md"))
    ap.add_argument("--families", nargs="+", default=list(FAMILIES) + list(BRANCHES))
    a = ap.parse_args()
    dev = A.dev_tables({n: S23 / f"warmgate_e19_{n}.csv" for n in DEV_NAMES}, [1.0]).rename(columns=REN)
    grp = A.groups_for(dev); cl_all = A.cells(dev)
    dproc = {}
    for c in PROC_SPEC:
        det = {n: S2 / f"p_dev_warmgate_e19_{n}_{c}.csv" for n in DEV_NAMES}
        if all(f.is_file() for f in det.values()):
            dproc[c] = A.dev_tables(det, [1.0]).rename(columns=REN)
    tests = {}
    for t in A.TEST:
        d = A.load_frames(S23 / f"warmgate_sel_{t}.csv")
        keep = ~d.subset.str.contains("IP_LAP") & (d.subset != "fake-audio-driven") & ~d.subset.str.startswith("adv-")
        tests[t] = A.video_table(d[keep].reset_index(drop=True), [1.0]).rename(columns=REN)
    for dsn in ("CDFv2", "DFDCP"):
        for c in PROC_SPEC:
            tests[f"{dsn} {c}"] = A.video_table(A.load_frames(S2 / f"p_v3_warmgate_{dsn}_{c}.csv"), [1.0]).rename(columns=REN)

    fams, report = [], {}
    for fam in a.families:
        if fam in BRANCHES:
            d_ = BRANCHES[fam]; name = f"{fam}:ordered"
            if not (d_.parent / "DONE").is_file():
                print(f"{fam}: branch not finished -- skipped", flush=True); continue
            def put(T, f):
                o = pd.read_csv(f).set_index("v"); idx = o.index.get_indexer(T.v)
                if (idx >= 0).all():
                    T[f"p_{name}"], T[f"u_{name}"] = o.p.to_numpy()[idx], o.u.to_numpy()[idx]
            dv = pd.concat([pd.read_csv(d_ / f"{n}.csv") for n in DEV_NAMES]).set_index("v"); idx = dv.index.get_indexer(dev.v)
            assert (idx >= 0).all(), fam
            dev[f"p_{name}"], dev[f"u_{name}"] = dv.p.to_numpy()[idx], dv.u.to_numpy()[idx]
            for c, P in dproc.items():
                pv = pd.concat([pd.read_csv(d_ / f"{n}_{c}.csv") for n in DEV_NAMES]).set_index("v"); ip = pv.index.get_indexer(P.v)
                if (ip >= 0).all():
                    P[f"p_{name}"], P[f"u_{name}"] = pv.p.to_numpy()[ip], pv.u.to_numpy()[ip]
            for tn, T in tests.items():
                base, _, proc = tn.partition(" ")
                f = d_ / (f"{base}_{proc}.csv" if proc else f"{base}.csv")
                if f.is_file():
                    put(T, f)
            rec = json.load(open(d_.parent / "train_record.json"))
            report[fam] = {"ordered": {"epoch": rec["best_epoch"], "dev_v31": rec["best_dev_v31"]}}
            fams.append(name); print(f"branch {name}: epoch {rec['best_epoch']}, dev v3.1 {rec['best_dev_v31']:.4f}", flush=True)
            continue
        tr = load(fam, "FFpp_train"); dv = [load(fam, n) for n in DEV_NAMES]
        if tr is None or any(x is None for x in dv):
            print(f"{fam}: features incomplete -- skipped", flush=True); continue
        ddf = pd.concat([x[0] for x in dv], ignore_index=True); o = ddf.set_index("v").index.get_indexer(dev.v)
        assert (o >= 0).all(), fam
        rep = {}
        for kind, col in (("ordered", 1), ("shuffled", 2)):
            if fam == "spatial" and kind == "shuffled":
                continue
            Xtr, Xd = tr[col], np.concatenate([x[col] for x in dv])[o]
            if fam == "consistency":
                med = np.nanmedian(Xtr, 0); iqr = np.nanpercentile(Xtr, 75, 0) - np.nanpercentile(Xtr, 25, 0) + 1e-6
                norm = lambda X, med=med, iqr=iqr: np.nan_to_num((X - med) / iqr, nan=0.0).astype(np.float32)
            else:
                norm = lambda X: X
            model, ep, sc = train(norm(Xtr), tr[0].y.to_numpy(), norm(Xd), cl_all)
            name = f"{fam}:{kind}"; rep[kind] = {"epoch": ep, "dev_v31": sc}
            dev[f"p_{name}"], dev[f"u_{name}"] = opin(model, norm(Xd))
            for c, P in dproc.items():
                px = [load(fam, n, PROC_SPEC[c]) for n in DEV_NAMES]
                if all(x is not None for x in px):
                    pdf = pd.concat([x[0] for x in px], ignore_index=True); op = pdf.set_index("v").index.get_indexer(P.v)
                    if (op >= 0).all():
                        P[f"p_{name}"], P[f"u_{name}"] = opin(model, norm(np.concatenate([x[col] for x in px])[op]))
            for tn, T in tests.items():
                base, _, proc = tn.partition(" ")
                fx = load(fam, base, PROC_SPEC.get(proc) if proc else None)
                if fx is None:
                    continue
                ot = fx[0].set_index("v").index.get_indexer(T.v)
                if (ot >= 0).all():
                    T[f"p_{name}"], T[f"u_{name}"] = opin(model, norm(fx[col][ot]))
            fams.append(name)
            print(f"probe {name}: epoch {ep}, dev v3.1 {sc:.4f}", flush=True)
        report[fam] = rep

    def arms_for(n):
        return {f"probe {n}": [n], f"DS {n}+OM+FP": [n, "artifact", "prior"], f"LR {n}+OM+FP": [n, "artifact", "prior"],
                f"DS {n}+APP+OM": [n, "semantic", "artifact"], f"LR {n}+APP+OM": [n, "semantic", "artifact"],
                f"LR APP+OM+FP+{n}": ["semantic", "artifact", "prior", n]}
    ARMS = {"gate (trained)": None, "DS APP+OM+FP (g=1)": ["semantic", "artifact", "prior"],
            "LR APP+OM+FP (g=1)": ["semantic", "artifact", "prior"], "DS APP+OM (two-view)": ["semantic", "artifact"]}
    for n in fams:
        ARMS.update(arms_for(n))

    def has(t, views):
        return all(f"p_{v}" in t for v in views)

    def score(t, lr):
        s = {}
        for arm, views in ARMS.items():
            if views is None:
                s[arm] = ds_views(t.assign(p_fsvfm=t.p_gated, u_fsvfm=t.u_gated), ["semantic", "artifact", "fsvfm"]); continue
            if not has(t, views):
                continue
            if arm.startswith("probe"):
                s[arm] = t[f"p_{views[0]}"].to_numpy()
            elif arm.startswith("DS"):
                s[arm] = ds_views(t, views)
            elif arm in lr:
                cols = [f"{k}_{v}" for v in views for k in "pu"]
                s[arm] = lr[arm][1].predict_proba(lr[arm][0].transform(t[cols].to_numpy()))[:, 1]
        return s

    splits = []
    for k in range(A.N_SPLITS):
        fi, si = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=k).split(dev, dev.y, grp))
        Fd, Sd = dev.iloc[fi].reset_index(drop=True), dev.iloc[si].reset_index(drop=True)
        lr = {}
        for arm, views in ARMS.items():
            if views is not None and arm.startswith("LR") and has(Fd, views):
                cols = [f"{kk}_{v}" for v in views for kk in "pu"]
                sc_ = StandardScaler().fit(Fd[cols].to_numpy())
                lr[arm] = (sc_, LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(
                    sc_.transform(Fd[cols].to_numpy()), Fd.y.to_numpy()))
        sel = {n: A.v31(v, A.cells(Sd)) for n, v in score(Sd, lr).items()}
        proc = {}
        for c, P in dproc.items():
            Pt = P.set_index("v").loc[Sd.v[Sd.v.isin(P.v)]].reset_index()
            Pt["source"] = Sd.set_index("v").loc[Pt.v, "source"].to_numpy()
            proc[c] = {n: A.v31(v, A.cells(Pt)) for n, v in score(Pt, lr).items()}
        splits.append({"sel": sel, "proc": proc, "lr": lr})
        print(f"split {k} done", flush=True)

    c3 = tests["CDFv3"]
    tests["CDFv3 reenactment"] = pd.concat([c3[c3.y == 0], c3[(c3.y == 1) & c3.subset.str.contains("FaceReenact|TalkingFace")]])
    tests["CDFv3 swap"] = pd.concat([c3[c3.y == 0], c3[(c3.y == 1) & c3.subset.str.contains("FaceSwap")]])
    kd = tests["KoDF"]; tests["KoDF first-order motion"] = pd.concat([kd[kd.y == 0], kd[(kd.y == 1) & (kd.subset == "fake-fo")]])
    test_res = {}
    for tn, T in tests.items():
        T = T.reset_index(drop=True); y = T.y.to_numpy(); acc = {}
        for sp in splits:
            for n, v in score(T, sp["lr"]).items():
                acc.setdefault(n, []).append(A.fast_auc(v[y == 0], v[y == 1]))
        test_res[tn] = {n: float(np.mean(v)) for n, v in acc.items()}
    json.dump({"probes": report, "splits": [{k: v for k, v in s.items() if k != "lr"} for s in splits], "tests": test_res},
              open(a.out.with_suffix(".json"), "w"), indent=1, default=float)
    write(a.out, report, splits, test_res, list(dproc), list(ARMS))
    print("wrote", a.out)


def write(out, report, splits, test_res, dproc, arms):
    pm = lambda v: f"{np.mean(v):.4f} ±{np.std(v):.4f}"
    L = ["# Experiment 3b — temporal panel on stored frames", "",
         "Frozen X-CLIP, TimeSformer, VideoMAE and V-JEPA 2 video features and frame-to-frame consistency features "
         "(ArcFace identity drift, landmark motion), all from the stored face crops (<= 32 frames per video, ~0.25–0.6 s "
         "apart; no raw videos exist for most test sets). Every family gets the same evidential probe on FF++ train, "
         "development-rule epoch selection, an ORDERED and a SHUFFLED-frame version, and fusion with the development-"
         "selected warm+gate views fitted only on the audit's grouped fitting folds. `spatial` = plain CLIP ViT-B/16 "
         "frame mean. Test sets are read afterwards. Script `analysis/tbiom/temporal/panel_probe.py`.", "",
         "APP = Appearance, OM = Off-manifold residual, FP = Bona-fide face prior (raw); DS = equal Dempster–Shafer; "
         "LR = logistic stacking on matched inputs (p, vacuity). `<family>:ordered` vs `<family>:shuffled`: if they tie, "
         "the family adds appearance information, not temporal order.", "",
         "## 1. Probe epochs (full development set)", "", "| family | ordered epoch / v3.1 | shuffled epoch / v3.1 |", "|---|---|---|"]
    for fam, r in report.items():
        cell = lambda k: f"{r[k]['epoch']} / {r[k]['dev_v31']:.4f}" if k in r else "—"
        L.append(f"| {fam} | {cell('ordered')} | {cell('shuffled')} |")
    L += ["", "## 2. Clean development — selection folds (mean ± sd over 5 grouped splits)", "",
          "| arm | v3.1 | swap | reenactment | in-the-wild |", "|---|---:|---:|---:|---:|"]
    for n in arms:
        if n in splits[0]["sel"]:
            L.append(f"| {n} | {pm([s['sel'][n][0] for s in splits])} | " +
                     " | ".join(pm([s["sel"][n][1][t] for s in splits]) for t in splits[0]["sel"][n][1]) + " |")
    if dproc:
        L += ["", "## 3. Processed development — same selection-fold videos (reported, not used)", "",
              "| arm | " + " | ".join(f"{c} v3.1 | {c} reenactment" for c in dproc) + " |", "|---|" + "---:|---:|" * len(dproc)]
        for n in arms:
            if all(n in splits[0]["proc"][c] for c in dproc):
                L.append(f"| {n} | " + " | ".join(f"{pm([s['proc'][c][n][0] for s in splits])} | "
                                                   f"{pm([s['proc'][c][n][1]['reenactment'] for s in splits])}" for c in dproc) + " |")
    cols = [c for c in ["CDFv2", "CDFv3", "CDFv3 reenactment", "CDFv3 swap", "DFD", "DFDC", "DFDCP", "DFEval24", "FFIW", "KoDF",
                        "KoDF first-order motion", "IDForge", "DFDM", "CDFv2 jpeg40", "CDFv2 rs06", "DFDCP jpeg40", "DFDCP rs06"]]
    L += ["", "## 4. Test sets — video AUROC (read after; LR arms averaged over the 5 fitted splits)", "",
          "| arm | " + " | ".join(cols) + " | OOD mean |", "|---|" + "---:|" * (len(cols) + 1)]
    for n in arms:
        if all(n in test_res.get(c, {}) for c in cols):
            L.append(f"| {n} | " + " | ".join(f"{test_res[c][n]:.4f}" for c in cols) +
                     f" | {np.mean([test_res[c][n] for c in A.TEST]):.4f} |")
    L += ["", "## Limitations", "",
          "- Stored frames are sparse (~0.25–0.6 s apart); dense-clip dynamics cannot be tested for most sets.",
          "- `*_branch` rows are trained branches (last 2 blocks + evidential head, 10 epochs, batch 32, epoch chosen by the "
          "development rule; analysis/tbiom/temporal/train_branch.py). They have no shuffled-frame twin.",
          "- Frozen features, small probes, one probe seed, one detector seed; a failing probe is not proof that "
          "adaptation cannot help.",
          "- Probe epochs use the full development set, so fold comparisons are conditional on them.",
          "- CDFv2 and DeepfakeEval2024 have informed development; no test set here is untouched confirmation data."]
    out.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()

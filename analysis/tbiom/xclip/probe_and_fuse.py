#!/usr/bin/env python3
"""Temporal-branch check (brief Experiment 3, steps 2-5): frozen X-CLIP probe, controls, and fusion with existing views.

    python analysis/tbiom/xclip/probe_and_fuse.py --out tbiom/reports/EXP3_XCLIP_PROBE.md

Step 2  Small evidential probe (512 -> 128 -> 2, softplus evidence, DISCERN's EDL loss) trained on FF++ TRAIN video
        features only; epoch chosen by the v3.1 development rule (same data and objective as detector checkpoints).
Step 3  The same probe on four feature sets from the same 16 frames: X-CLIP ordered (temporal), X-CLIP shuffled
        (order control), X-CLIP frame mean, plain CLIP ViT-B/16 frame mean (spatial control).
Step 4  Lightweight fusion with the existing views of the development-selected warm+gate checkpoint (epoch 19;
        Appearance and Off-manifold residual = frozen DiCoME views, Bona-fide face prior raw):
          T+OM+FP   X-CLIP + Off-manifold residual + face prior       (requested combination 1)
          T+APP+OM  X-CLIP + Appearance + Off-manifold residual       (requested combination 2)
        each as equal DS replay and as logistic stacking on matched inputs (p, vacuity of its three views), fitted on
        the grouped FITTING folds of the fusion audit (identical splits). References on the same folds: the three
        existing views with full participation (APP+OM+FP), the trained gate, and the two-view base (APP+OM).
Step 5  Incremental development value on the selection folds: v3.1 score and its reenactment / swap / in-the-wild
        cells, clean and processed (JPEG q40, resize 0.60; reported, never used). Error overlap with the existing views
        at each opinion's FF++-val EER threshold is a diagnostic, not a veto. Test sets are read afterwards.

One probe seed and one detector seed: this is a feasibility check of the frozen representation inside the evidential
framework, not an end-to-end trained branch.
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, "/data/umar/Repos/DISCERN_Ext")
import symbolic_overlay as SO  # noqa: E402
import exp1_fusion_audit as A  # noqa: E402
from health import eer  # noqa: E402
from src.losses.evidential_loss import evidential_loss_discern_ext  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

torch.set_num_threads(4)  # shared box (load > 1000): torch would otherwise spawn one thread per core
FEATS = Path("logs/tbiom/xclip/feats"); S23, S2 = Path("logs/tbiom/stage23"), Path("logs/tbiom/stage2")
KINDS = ["xclip_temporal", "xclip_shuffled", "xclip_framemean", "clip_framemean"]
DEV_NAMES = ("VALmix", "VALmixV2", "FFpp_val")                 # audit order -> identical splits
TEST = A.TEST
EPOCHS, SEED, K = 40, 0, 2
torch.set_num_threads(16)   # the box runs at load ~500; default (all cores) thrashed and the first run stalled


def load_feats(ds, proc=None):
    tag = ds + (f"_{proc}" if proc else "")
    files = sorted(glob.glob(str(FEATS / f"{tag}.s*of*.npz"))) or sorted(glob.glob(str(FEATS / f"{tag}.npz")))
    if not files:
        return None
    zs = [np.load(f, allow_pickle=True) for f in files]
    n_expected = int(Path(files[0]).name.split("of")[-1].split(".")[0]) if ".s" in files[0] else 1
    if len(files) != n_expected:
        print(f"  {tag}: {len(files)} of {n_expected} shards present -- skipped", flush=True)
        return None
    df = pd.DataFrame({"v": np.concatenate([z["vid"] for z in zs]), "y": np.concatenate([z["label"] for z in zs]),
                       "subset": np.concatenate([z["subset"] for z in zs])})
    X = {k: np.concatenate([z[k] for z in zs]).astype(np.float32) for k in KINDS}
    assert df.v.is_unique, tag
    return df, X


class Probe(nn.Module):
    def __init__(self, d=512, h=128):
        super().__init__(); self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h), nn.GELU(), nn.Linear(h, 2))

    def forward(self, x):
        return F.softplus(self.net(x))


def opinions(model, X):
    with torch.no_grad():
        e = model(torch.as_tensor(X)).double().numpy()
    a = e + 1; S = a.sum(1)
    return a[:, 1] / S, K / S


def train_probe(kind, train, dev_tab, dev_X, cl):
    torch.manual_seed(SEED); np.random.seed(SEED)
    df, X = train; x, y = torch.as_tensor(X[kind]), torch.as_tensor(df.y.to_numpy()).long()
    w = torch.where(y == 1, 1.0 / (y == 1).sum(), 1.0 / (y == 0).sum())       # class-balanced sampling
    model = Probe(); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    best = (-1.0, None, None)
    for ep in range(EPOCHS):
        model.train()
        idx = torch.multinomial(w, len(y), replacement=True, generator=torch.Generator().manual_seed(SEED + ep))
        for b in range(0, len(idx), 128):
            j = idx[b:b + 128]
            loss = evidential_loss_discern_ext(model(x[j]), y[j], 2, ep, EPOCHS).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); p, _ = opinions(model, dev_X[kind])
        sc = A.v31(p, cl)[0]
        if sc > best[0] + 1e-6:                                                   # ties -> earliest epoch
            best = (sc, ep, {k: v.clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2]); model.eval()
    return model, best[1], best[0]


def attach(tab, df, p, u, name):
    m = pd.DataFrame({"v": df.v, f"p_{name}": p, f"u_{name}": u})
    out = tab.merge(m, on="v", how="left")
    assert out[f"p_{name}"].notna().all(), f"{name}: {out[f'p_{name}'].isna().sum()} videos without features"
    return out


def ds_views(t, views):
    SO.BRANCHES[:] = list(views); s = SO.fuse(t); SO.BRANCHES[:] = ["semantic", "artifact", "fsvfm"]; return s


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=Path("tbiom/reports/EXP3_XCLIP_PROBE.md"))
    a = ap.parse_args()
    grid = [1.0]
    dev = A.dev_tables({n: S23 / f"warmgate_e19_{n}.csv" for n in DEV_NAMES}, grid)
    dev = dev.rename(columns={"p_fsvfm_raw": "p_prior", "u_fsvfm_raw": "u_prior", "p_fsvfm": "p_gated", "u_fsvfm": "u_gated"})
    grp = A.groups_for(dev); cl_all = A.cells(dev)
    train = load_feats("FFpp_train")
    fdev = [load_feats(n) for n in DEV_NAMES]
    ddf = pd.concat([f[0] for f in fdev], ignore_index=True)
    dX = {k: np.concatenate([f[1][k] for f in fdev]) for k in KINDS}
    order = ddf.set_index("v").index.get_indexer(dev.v); assert (order >= 0).all(), "dev videos without X-CLIP features"
    dev_X = {k: v[order] for k, v in dX.items()}

    probes, report = {}, {"probe": {}}
    for kind in KINDS:
        model, ep, sc = train_probe(kind, train, dev, dev_X, cl_all)
        probes[kind] = model; report["probe"][kind] = {"epoch": ep, "dev_v31": sc, "types": A.v31(opinions(model, dev_X[kind])[0], cl_all)[1]}
        p, u = opinions(model, dev_X[kind]); dev[f"p_{kind}"], dev[f"u_{kind}"] = p, u
        print(f"probe {kind}: epoch {ep}, dev v3.1 {sc:.4f} {report['probe'][kind]['types']}", flush=True)

    # processed development tables (same videos), when both detector and X-CLIP exports exist
    dproc = {}
    for c, spec in (("jpeg40", "jpeg40"), ("rs06", "resize06")):
        det = {n: S2 / f"p_dev_warmgate_e19_{n}_{c}.csv" for n in DEV_NAMES}
        fx = [load_feats(n, spec) for n in DEV_NAMES]
        if all(f.is_file() for f in det.values()) and all(x is not None for x in fx):
            t = A.dev_tables(det, grid).rename(columns={"p_fsvfm_raw": "p_prior", "u_fsvfm_raw": "u_prior",
                                                        "p_fsvfm": "p_gated", "u_fsvfm": "u_gated"})
            pdf = pd.concat([f[0] for f in fx], ignore_index=True); o = pdf.set_index("v").index.get_indexer(t.v)
            assert (o >= 0).all()
            for kind in KINDS:
                p, u = opinions(probes[kind], np.concatenate([f[1][kind] for f in fx])[o]); t[f"p_{kind}"], t[f"u_{kind}"] = p, u
            dproc[c] = t
    print("processed development available:", list(dproc) or "none", flush=True)

    COMBOS = {"T+OM+FP": ["xclip_temporal", "artifact", "prior"], "T+APP+OM": ["xclip_temporal", "semantic", "artifact"],
              "Tshuf+OM+FP": ["xclip_shuffled", "artifact", "prior"], "Tshuf+APP+OM": ["xclip_shuffled", "semantic", "artifact"],
              "S+OM+FP": ["clip_framemean", "artifact", "prior"], "S+APP+OM": ["clip_framemean", "semantic", "artifact"],
              "APP+OM+FP (g=1)": ["semantic", "artifact", "prior"], "APP+OM (two-view)": ["semantic", "artifact"]}

    def score_all(t, lr):
        s = {f"probe {k}": t[f"p_{k}"].to_numpy() for k in KINDS}
        s["gate (trained)"] = ds_views(t.assign(p_fsvfm=t.p_gated, u_fsvfm=t.u_gated), ["semantic", "artifact", "fsvfm"])
        for name, views in COMBOS.items():
            s[f"DS {name}"] = ds_views(t, views)
            if len(views) == 3:
                cols = [f"{k}_{v}" for v in views for k in "pu"]
                s[f"LR {name}"] = lr[name][1].predict_proba(lr[name][0].transform(t[cols].to_numpy()))[:, 1]
        return s

    splits = []
    for k in range(A.N_SPLITS):
        fi, si = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=k).split(dev, dev.y, grp))
        Fd, Sd = dev.iloc[fi].reset_index(drop=True), dev.iloc[si].reset_index(drop=True)
        lr = {}
        for name, views in COMBOS.items():
            if len(views) == 3:
                cols = [f"{kk}_{v}" for v in views for kk in "pu"]
                sc_ = StandardScaler().fit(Fd[cols].to_numpy())
                lr[name] = (sc_, LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)
                            .fit(sc_.transform(Fd[cols].to_numpy()), Fd.y.to_numpy()))
        clS = A.cells(Sd); s = score_all(Sd, lr)
        ffs = np.where(Sd.source == "FFpp_val")[0]
        tau = {n: eer(Sd.y.to_numpy()[ffs], v[ffs])[1] for n, v in s.items()}
        sel = {n: A.v31(v, clS) for n, v in s.items()}
        # error overlap on the selection fold (diagnostic): videos wrong at tau
        y = Sd.y.to_numpy(); wrong = {n: ((v >= tau[n]).astype(int) != y) for n, v in s.items()}
        ov = {}
        for probe in ("probe xclip_temporal", "probe clip_framemean"):
            for ref in ("DS APP+OM+FP (g=1)", "gate (trained)", "DS APP+OM (two-view)"):
                a_, b_ = wrong[probe], wrong[ref]
                ov[f"{probe} vs {ref}"] = {"jaccard": float((a_ & b_).sum() / max(1, (a_ | b_).sum())),
                                           "probe_err": float(a_.mean()), "ref_err": float(b_.mean()),
                                           "ref_wrong_probe_right": float((b_ & ~a_).sum() / max(1, b_.sum()))}
        proc = {}
        for c, P in dproc.items():
            Pt = P.set_index("v").loc[Sd.v[Sd.v.isin(P.v)]].reset_index()
            Pt["source"] = Sd.set_index("v").loc[Pt.v, "source"].to_numpy()
            proc[c] = {n: A.v31(v, A.cells(Pt)) for n, v in score_all(Pt, lr).items()}
        splits.append({"split": k, "selection": sel, "overlap": ov, "proc": proc, "tau": tau, "lr": lr})
        print(f"split {k}: " + " | ".join(f"{n} {sel[n][0]:.4f}" for n in
              ["gate (trained)", "DS APP+OM+FP (g=1)", "DS T+OM+FP", "LR T+OM+FP", "DS T+APP+OM", "LR T+APP+OM",
               "DS APP+OM (two-view)"]), flush=True)

    # tests (read after): detector opinions from the selected warm+gate test exports + probe opinions
    tests = {}
    for ds_ in TEST:
        fx = load_feats(ds_)
        if fx is None:
            continue
        d = A.load_frames(S23 / f"warmgate_sel_{ds_}.csv")
        keep = ~d.subset.str.contains("IP_LAP") & (d.subset != "fake-audio-driven") & ~d.subset.str.startswith("adv-")
        t = A.video_table(d[keep].reset_index(drop=True), grid).rename(
            columns={"p_fsvfm_raw": "p_prior", "u_fsvfm_raw": "u_prior", "p_fsvfm": "p_gated", "u_fsvfm": "u_gated"})
        o = fx[0].set_index("v").index.get_indexer(t.v); assert (o >= 0).all(), ds_
        for kind in KINDS:
            t[f"p_{kind}"], t[f"u_{kind}"] = opinions(probes[kind], fx[1][kind][o])
        tests[ds_] = t
    for ds_, c, spec in (("CDFv2", "jpeg40", "jpeg40"), ("CDFv2", "rs06", "resize06"), ("DFDCP", "jpeg40", "jpeg40"),
                         ("DFDCP", "rs06", "resize06")):
        fx = load_feats(ds_, spec)
        if fx is None:
            continue
        t = A.video_table(A.load_frames(S2 / f"p_v3_warmgate_{ds_}_{c}.csv"), grid).rename(
            columns={"p_fsvfm_raw": "p_prior", "u_fsvfm_raw": "u_prior", "p_fsvfm": "p_gated", "u_fsvfm": "u_gated"})
        o = fx[0].set_index("v").index.get_indexer(t.v); assert (o >= 0).all()
        for kind in KINDS:
            t[f"p_{kind}"], t[f"u_{kind}"] = opinions(probes[kind], fx[1][kind][o])
        tests[f"{ds_} {c}"] = t
    if "CDFv3" in tests:
        c3 = tests["CDFv3"]
        tests["CDFv3 reenactment"] = pd.concat([c3[c3.y == 0], c3[(c3.y == 1) & c3.subset.str.contains("FaceReenact|TalkingFace")]])
        tests["CDFv3 swap"] = pd.concat([c3[c3.y == 0], c3[(c3.y == 1) & c3.subset.str.contains("FaceSwap")]])
    if "KoDF" in tests:
        kd = tests["KoDF"]; tests["KoDF first-order motion"] = pd.concat([kd[kd.y == 0], kd[(kd.y == 1) & (kd.subset == "fake-fo")]])
    test_res = {}
    for name, t in tests.items():
        t = t.reset_index(drop=True); y = t.y.to_numpy(); acc = {}
        for sp in splits:
            for n, v in score_all(t, sp["lr"]).items():
                acc.setdefault(n, []).append(A.fast_auc(v[y == 0], v[y == 1]))
        test_res[name] = {n: float(np.mean(v)) for n, v in acc.items()}

    json.dump({"probe": report["probe"], "splits": [{k: v for k, v in s.items() if k != "lr"} for s in splits],
               "tests": test_res}, open(a.out.with_suffix(".json"), "w"), indent=1, default=float)
    write_md(a.out, report, splits, test_res, list(dproc))
    print("wrote", a.out)


def write_md(out, report, splits, test_res, dproc):
    pm = lambda v: f"{np.mean(v):.4f} ±{np.std(v):.4f}"
    ROWS = ["probe xclip_temporal", "probe xclip_shuffled", "probe xclip_framemean", "probe clip_framemean",
            "gate (trained)", "DS APP+OM+FP (g=1)", "DS APP+OM (two-view)",
            "DS T+OM+FP", "LR T+OM+FP", "DS T+APP+OM", "LR T+APP+OM",
            "DS Tshuf+OM+FP", "LR Tshuf+OM+FP", "DS Tshuf+APP+OM", "LR Tshuf+APP+OM",
            "DS S+OM+FP", "LR S+OM+FP", "DS S+APP+OM", "LR S+APP+OM", "LR APP+OM+FP (g=1)"]
    L = ["# Experiment 3 — frozen X-CLIP temporal probe and fusion check", "",
         "Feasibility check inside the evidential framework, run on GPU 1 while the matched augmentation pilot trains. "
         "Frozen X-CLIP (`microsoft/xclip-base-patch16-16-frames`) video features from 16 of the stored face-crop frames "
         "per video; small evidential probe trained on FF++ train only; epoch chosen by the v3.1 development rule. "
         "Existing views come from the development-selected warm+gate checkpoint (epoch 19). Fusion is fitted only on "
         "the grouped fitting folds of the fusion audit (identical splits); test sets are read afterwards and choose nothing. "
         "Script `analysis/tbiom/xclip/probe_and_fuse.py`, features `analysis/tbiom/xclip/extract_xclip.py`.", "",
         "Names: APP = Appearance, OM = Off-manifold residual, FP = Bona-fide face prior (raw), T = X-CLIP ordered, "
         "Tshuf = X-CLIP shuffled frame order, S = plain CLIP ViT-B/16 frame mean (spatial control). DS = equal "
         "Dempster–Shafer replay; LR = logistic stacking on matched inputs (p and vacuity of the three views).", "",
         "## 1. Probes alone (development, full dev set used for epoch selection)", "",
         "| probe | epoch | v3.1 | swap | reenactment | in-the-wild |", "|---|---:|---:|---:|---:|---:|"]
    for k, r in report["probe"].items():
        L.append(f"| {k} | {r['epoch']} | {r['dev_v31']:.4f} | " + " | ".join(f"{r['types'][t]:.4f}" for t in r["types"]) + " |")
    L += ["", "## 2. Clean development — selection folds (mean ± sd over 5 grouped splits)", "",
          "| arm | v3.1 | swap | reenactment | in-the-wild |", "|---|---:|---:|---:|---:|"]
    for n in ROWS:
        if n not in splits[0]["selection"]:
            continue
        L.append(f"| {n} | {pm([s['selection'][n][0] for s in splits])} | " +
                 " | ".join(pm([s["selection"][n][1][t] for s in splits]) for t in splits[0]["selection"][n][1]) + " |")
    if dproc:
        L += ["", "## 3. Processed development — same selection-fold videos (reported, not used)", "",
              "| arm | " + " | ".join(f"{c} v3.1 | {c} reenactment" for c in dproc) + " |", "|---|" + "---:|---:|" * len(dproc)]
        for n in ROWS:
            if n not in splits[0]["proc"][dproc[0]]:
                continue
            L.append(f"| {n} | " + " | ".join(f"{pm([s['proc'][c][n][0] for s in splits])} | "
                                               f"{pm([s['proc'][c][n][1]['reenactment'] for s in splits])}" for c in dproc) + " |")
    L += ["", "## 4. Error overlap on the selection folds (diagnostic, not a veto)", "",
          "Errors at each opinion's FF++-val EER threshold. `recovered` = share of the reference's errors the probe gets right.", "",
          "| pair | Jaccard of errors | probe error | reference error | recovered |", "|---|---:|---:|---:|---:|"]
    for pair in splits[0]["overlap"]:
        vals = [s["overlap"][pair] for s in splits]
        L.append(f"| {pair} | {np.mean([v['jaccard'] for v in vals]):.3f} | {np.mean([v['probe_err'] for v in vals]):.3f} | "
                 f"{np.mean([v['ref_err'] for v in vals]):.3f} | {np.mean([v['ref_wrong_probe_right'] for v in vals]):.3f} |")
    if test_res:
        cols = [c for c in ["CDFv2", "CDFv3", "CDFv3 reenactment", "CDFv3 swap", "DFD", "DFDC", "DFDCP", "DFEval24", "FFIW",
                            "KoDF", "KoDF first-order motion", "IDForge", "DFDM", "CDFv2 jpeg40", "CDFv2 rs06",
                            "DFDCP jpeg40", "DFDCP rs06"] if c in test_res]
        L += ["", "## 5. Test sets — video AUROC (read after; LR arms averaged over the 5 fitted splits)", "",
              "| arm | " + " | ".join(cols) + " | OOD mean |", "|---|" + "---:|" * (len(cols) + 1)]
        base = [c for c in TEST if c in test_res]
        for n in ROWS:
            if n not in test_res[cols[0]]:
                continue
            mean = np.mean([test_res[c][n] for c in base]) if len(base) == len(TEST) else float("nan")
            L.append(f"| {n} | " + " | ".join(f"{test_res[c][n]:.4f}" for c in cols) + f" | {mean:.4f} |")
    L += ["", "## Limitations", "",
          "- Frozen features and a small probe: a failure here is evidence against this configuration, not proof that "
          "bounded adaptation cannot help. One probe seed; one detector seed.",
          "- Temporal sampling is coarse: 16 of up to 32 frames stored across the whole clip (~0.5–1 s apart), not dense tracked clips.",
          "- X-CLIP's encoder exchanges cross-frame message tokens, so its frame mean is not purely spatial; the plain CLIP "
          "ViT-B/16 frame mean is the spatial control.",
          "- The probe epoch uses the full development set (like detector checkpoints), so fold comparisons are conditional on it.",
          "- CDFv2 and DeepfakeEval2024 have informed development; no test set here is untouched confirmation data."]
    out.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()

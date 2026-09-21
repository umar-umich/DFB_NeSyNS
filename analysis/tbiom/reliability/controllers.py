#!/usr/bin/env python3
"""Stage 1A (executable brief): the reliability controller ladder, on frozen reference branches (cached, CPU).

    OMP_NUM_THREADS=4 python analysis/tbiom/reliability/controllers.py

Everything fixed here is fixed in tbiom/docs/stage1a-reliability-preregistration.md, written before this file ran:
the branches, the evidence-scaling discount alpha' = 1 + w (alpha - 1) read from code (REFERENCE_MANIFEST.md §2),
the 4-segment granularity and its limits (§3-4), within-view evidence averaging, the folds, the budget, the controls
and the acceptance rule. Only the predictor of w differs between rows of the ladder.
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent / "temporal")]
import exp1_fusion_audit as A  # noqa: E402
import dsr_system as D  # noqa: E402
import finish_line as F  # noqa: E402
from health import eer  # noqa: E402
from safeguard_ablation import rc_metrics  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

torch.set_num_threads(4)
S23, S2, R = Path("logs/tbiom/stage23"), Path("logs/tbiom/stage2"), Path("tbiom/reports")
SEG = Path("logs/tbiom/acquisition/v2/vjepa2_lora_B_seed42")
DEV, CONDS, VIEWS = ["VALmix", "VALmixV2", "FFpp_val"], ["clean", "jpeg40", "rs06"], ["T", "APP", "OM"]
QC = ["blur", "blockiness", "contrast", "motion", "luma"]
NS, K, EPOCHS, LR, WD, HID, N_BOOT = 4, 2.0, 60, 1e-3, 1e-4, 32, 1000


# ----------------------------------------------------------------------------- data
def spatial_segments(name, cond):
    """Per-segment APP/OM opinions: means of cached per-frame opinions over the same crops the temporal segment used."""
    f = S23 / f"warmgate_e19_{name}.csv" if cond == "clean" else S2 / f"p_dev_warmgate_e19_{name}_{cond}.csv"
    d = A.load_frames(f)
    d["path"] = d.key.astype(str).str.split("::").str[-1]
    d = d.sort_values(["v", "path"], kind="stable").reset_index(drop=True)
    d["s"] = -1
    for _v, g in d.groupby("v", sort=False):
        for s, part in enumerate(np.array_split(np.arange(len(g)), NS)):
            if len(part):
                d.loc[g.index[part], "s"] = s
    g = d.groupby(["v", "s"]).agg(p_APP=("p_semantic", "mean"), u_APP=("u_semantic", "mean"),
                                  p_OM=("p_artifact", "mean"), u_OM=("u_artifact", "mean"), nf=("key", "size"))
    return g.reset_index()


def load():
    """One long table: (video, condition, segment) with all three views' opinions and the quality descriptors."""
    rows = []
    for name in DEV:
        for cond in CONDS:
            tag = "" if cond == "clean" else f"_{cond}"
            t = pd.read_csv(SEG / f"{name}{tag}.csv")
            t = t[t.segment != "full"].copy(); t["s"] = t.segment.str[1:].astype(int)
            t = t.rename(columns={"p": "p_T", "u": "u_T"})
            sp = spatial_segments(name, cond)
            m = t.merge(sp, on=["v", "s"], how="inner"); m["cond"] = cond
            rows.append(m)
    return pd.concat(rows, ignore_index=True)


class Pack:
    """Dense arrays for a set of videos x one assignment of conditions to segments."""

    def __init__(self, lut, videos, cond_of_seg, qstats=None):
        n = len(videos)
        self.p = np.full((n, NS, 3), 0.5); self.u = np.ones((n, NS, 3)); self.q = np.zeros((n, NS, len(QC)))
        self.mask = np.zeros((n, NS))
        for i, v in enumerate(videos):
            for s in range(NS):
                r = lut.get((v, cond_of_seg[i, s], s))
                if r is None:                                   # absent segment: masked, never filled
                    continue
                self.p[i, s], self.u[i, s], self.q[i, s] = r
                self.mask[i, s] = 1.0
        self.qstats = qstats if qstats is not None else self._stats()

    def _stats(self):
        m = self.mask.astype(bool)
        return self.q[m].mean(0), self.q[m].std(0) + 1e-6

    def features(self, quality=True, shuffle=None, qperm=None):
        p, u, q, mk = self.p.copy(), self.u.copy(), (self.q - self.qstats[0]) / self.qstats[1], self.mask.copy()
        if qperm is not None:
            q = q[qperm]
        if shuffle is not None:
            p, u, q, mk = (np.take_along_axis(x, shuffle[..., None] if x.ndim == 3 else shuffle, 1) for x in (p, u, q, mk))
        b1, b0 = p - u / 2, 1 - p - u / 2
        conf = np.max([np.clip(b1[..., a], 0, 1) * np.clip(b0[..., c], 0, 1) + np.clip(b0[..., a], 0, 1) * np.clip(b1[..., c], 0, 1)
                       for a, c in ((0, 1), (0, 2), (1, 2))], 0)
        x = [p, u, (p.max(2) - p.min(2))[..., None], p.std(2, keepdims=True), conf[..., None]]
        x += [q if quality else np.zeros_like(q), mk[..., None]]
        feats = np.concatenate(x, 2) * mk[..., None]
        return (torch.tensor(feats, dtype=torch.float32), torch.tensor(p), torch.tensor(u), torch.tensor(mk))


# ----------------------------------------------------------------------------- fusion (the deployed operator)
def fuse(p, u, w, mk):
    """p, u, w: (n, S, 3); mk: (n, S). Evidence scaling per segment, mean over available segments, Dempster across views."""
    S = K / u.clamp_min(1e-9)
    e = torch.stack([(1 - p) * S - 1, p * S - 1], -1).clamp_min(0)          # (n, S, 3, 2) evidence
    e = e * w[..., None] * mk[..., None, None]
    e = e.sum(1) / mk.sum(1).clamp_min(1)[:, None, None]                      # mean over available segments
    a = e + 1                                                                 # (n, 3, 2)
    f = a[:, 0]
    for v in (1, 2):
        g = a[:, v]; S1, S2 = f.sum(1, keepdim=True), g.sum(1, keepdim=True)
        b1, b2, u1, u2 = (f - 1) / S1, (g - 1) / S2, K / S1, K / S2
        c = (b1.sum(1) * b2.sum(1) - (b1 * b2).sum(1))[:, None]
        b = (b1 * b2 + b1 * u2 + b2 * u1) / (1 - c); uu = u1 * u2 / (1 - c)
        f = b * (K / uu) + 1
    Sf = f.sum(1)
    return f[:, 1] / Sf, K / Sf


# ----------------------------------------------------------------------------- controllers
class Offset(nn.Module):
    """Wraps a trunk so that w = sigmoid(logit(w_const) + delta), with delta = 0 at initialisation."""

    def __init__(self, trunk, d_out, w_const):
        super().__init__(); self.trunk, self.head = trunk, nn.Linear(d_out, 3)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        wc = torch.tensor(np.clip(w_const, 0.02, 0.98), dtype=torch.float32)
        self.register_buffer("base", torch.log(wc / (1 - wc)))

    def forward(self, x, mk):
        return torch.sigmoid(self.base + self.head(self.trunk(x, mk)))


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, HID), nn.GELU(), nn.Linear(HID, HID), nn.GELU())

    def forward(self, x, mk):
        return self.net(x)


class GRU(nn.Module):
    def __init__(self, d):
        super().__init__(); self.rnn = nn.GRU(d, HID, batch_first=True)

    def forward(self, x, mk):
        return self.rnn(x)[0]


class Mamba(nn.Module):
    """One small selective state-space block (Mamba-style), pure PyTorch: input-dependent step, B and C."""

    def __init__(self, d, n=8):
        super().__init__()
        self.inp = nn.Linear(d, 2 * HID); self.conv = nn.Conv1d(HID, HID, 2, padding=1, groups=HID)
        self.dt, self.B, self.C = nn.Linear(HID, HID), nn.Linear(HID, n), nn.Linear(HID, n)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, n + 1, dtype=torch.float32)).repeat(HID, 1))
        self.Dskip = nn.Parameter(torch.ones(HID))

    def forward(self, x, mk):
        xz = self.inp(x); xs, z = xz.chunk(2, -1)
        xs = torch.nn.functional.silu(self.conv(xs.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2))
        dt = torch.nn.functional.softplus(self.dt(xs)); Bm, Cm = self.B(xs), self.C(xs); Am = -torch.exp(self.A_log)
        h = torch.zeros(x.shape[0], HID, Am.shape[1]); ys = []
        for t in range(x.shape[1]):
            m = mk[:, t, None, None]
            hn = torch.exp(dt[:, t, :, None] * Am) * h + dt[:, t, :, None] * Bm[:, t, None, :] * xs[:, t, :, None]
            h = m * hn + (1 - m) * h                                           # an absent segment leaves the state alone
            ys.append((h * Cm[:, t, None, :]).sum(-1) + self.Dskip * xs[:, t])
        return torch.stack(ys, 1) * torch.nn.functional.silu(z)


TRUNKS = {"segment MLP": MLP, "GRU": GRU, "Mamba": Mamba}


def fit_controller(name, X, P, U, M, y, w_const, seed=0):
    torch.manual_seed(seed)
    model = Offset(TRUNKS[name](X.shape[-1]), HID, w_const)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD); yt = torch.tensor(y, dtype=torch.float32)
    for _ in range(EPOCHS):
        pf, _ = fuse(P, U, model(X, M), M)
        loss = nn.functional.binary_cross_entropy(pf.clamp(1e-6, 1 - 1e-6), yt)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


# ----------------------------------------------------------------------------- evaluation
def score(model, pack, w_const, **kw):
    X, P, U, M = pack.features(**kw)
    with torch.no_grad():
        w = model(X, M) if model is not None else torch.tensor(w_const, dtype=torch.float32).expand(*P.shape)
        pf, uf = fuse(P, U, w, M)
    return pf.numpy(), uf.numpy()


def v31(s, tab):
    return A.v31(s, A.cells(tab))


def main():
    t0 = time.time()
    long = load()
    bad = (long.n_src != long.nf).mean()
    print(f"segment alignment: temporal and spatial crop counts differ on {bad * 100:.2f}% of segment rows", flush=True)
    assert bad < 0.01, "spatial and temporal segments are not built from the same crops"
    long = long[long.n_src == long.nf]
    lut = {(r.v, r.cond, r.s): ([r.p_T, r.p_APP, r.p_OM], [r.u_T, r.u_APP, r.u_OM],
                                [math.log1p(r.blur), r.blockiness, r.contrast, math.log1p(r.motion), r.luma])
           for r in long.itertuples(index=False)}
    base = D.load("vjepa2_lora_B_seed42")[0]                              # the deployed per-video table
    have = set(long[long.cond == "clean"].v) & set(long[long.cond == "jpeg40"].v) & set(long[long.cond == "rs06"].v)
    dev = base[base.v.isin(have)].reset_index(drop=True)
    print(f"{len(dev)} development videos with all three conditions ({time.time() - t0:.0f}s)", flush=True)
    rng = np.random.default_rng(20260920)
    out = []
    for k, Fd, Sd in D.splits(dev):
        w_const = np.array(F.fit_w([Fd], VIEWS))                              # the current fitted weights
        fvid, svid = list(Fd.v), list(Sd.v)
        uni = lambda n, c: np.full((n, NS), c, dtype=object)
        fit_packs = {c: Pack(lut, fvid, uni(len(fvid), c)) for c in CONDS}
        qs = fit_packs["clean"].qstats
        sel_packs = {c: Pack(lut, svid, uni(len(svid), c), qs) for c in CONDS}
        stress = rng.choice(CONDS, size=(len(svid), NS))                     # label-independent per-segment draw
        sel_packs["stress"] = Pack(lut, svid, stress.astype(object), qs)
        yF, yS = Fd.y.to_numpy(), Sd.y.to_numpy()
        tabS = Sd
        fitX = lambda **kw: [torch.cat(z) for z in zip(*[fit_packs[c].features(**kw) for c in CONDS])]
        yFF = np.concatenate([yF] * len(CONDS))
        shuf_f = np.stack([rng.permutation(NS) for _ in range(len(fvid) * len(CONDS))])
        shuf_s = {c: np.stack([rng.permutation(NS) for _ in range(len(svid))]) for c in sel_packs}
        qperm_f = rng.permutation(len(fvid) * len(CONDS)); qperm_s = {c: rng.permutation(len(svid)) for c in sel_packs}

        arms = {"constant DS-R": (None, {})}
        # constant refit on clean + processed segment inputs, reported as a control for data access
        best = max(F.wgrid(3), key=lambda w: np.mean([v31(score(None, sel_packs_c, np.array(w))[0], Fd)[0]
                                                        for sel_packs_c in [fit_packs[c] for c in CONDS]]))
        arms["constant, refit clean+processed"] = (None, {"w": np.array(best)})
        for name in TRUNKS:
            X, P, U, M = fitX(); arms[name] = (fit_controller(name, X, P, U, M, yFF, w_const), {})
            X, P, U, M = fitX(quality=False); arms[f"{name} — no quality"] = (fit_controller(name, X, P, U, M, yFF, w_const), {"quality": False})
            # the fitting features are re-ordered with one fixed permutation per example; selection with its own
            X, P, U, M = [torch.cat(z) for z in zip(*[fit_packs[c].features() for c in CONDS])]
            idx = torch.tensor(shuf_f)
            Xs, Ps, Us, Ms = (torch.take_along_dim(t, idx[..., None] if t.ndim == 3 else idx, 1) for t in (X, P, U, M))
            arms[f"{name} — shuffled chronology"] = (fit_controller(name, Xs, Ps, Us, Ms, yFF, w_const), {"shuffle": True})
            Xq = X.clone(); Xq[..., 9:14] = X[qperm_f][..., 9:14]
            arms[f"{name} — reassigned quality"] = (fit_controller(name, Xq, P, U, M, yFF, w_const), {"qperm": True})
        # logistic stacker on the same segment-averaged opinions, clean + processed fitting access
        def seg_mean(pack):
            X, P, U, M = pack.features(); m = M[..., None]
            return np.concatenate([(P * m).sum(1) / m.sum(1).clamp_min(1), (U * m).sum(1) / m.sum(1).clamp_min(1)], 1)
        Xl = np.concatenate([seg_mean(fit_packs[c]) for c in CONDS]); sc = StandardScaler().fit(Xl)
        lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(sc.transform(Xl), yFF)

        row = {"split": k, "w_const": w_const.tolist(), "arms": {}}
        tau_ref = None
        for an, (model, opt) in arms.items():
            r = {}
            for c, pk in sel_packs.items():
                kw = {}
                if opt.get("quality") is False:
                    kw["quality"] = False
                if opt.get("shuffle"):
                    kw["shuffle"] = shuf_s[c]
                if opt.get("qperm"):
                    kw["qperm"] = qperm_s[c]
                if model is None:
                    pf, uf = score(None, pk, opt.get("w", w_const), **kw)
                else:
                    X, P, U, M = pk.features(**kw)
                    with torch.no_grad():
                        pf, uf = [z.numpy() for z in fuse(P, U, model(X, M), M)]
                tot, types = v31(pf, tabS)
                r[c] = {"v31": tot, **types, "_p": pf, "_u": uf}
            ff = np.where(tabS.source == "FFpp_val")[0]
            tau = float(eer(yS[ff], r["clean"]["_p"][ff])[1])
            if an == "constant DS-R":
                tau_ref = tau
            for c in r:
                pf, uf = r[c].pop("_p"), r[c].pop("_u")
                for tn, tv in (("own", tau), ("drop-in", tau_ref)):
                    pred = (pf >= tv).astype(int)
                    r[c][f"fpr_{tn}"] = float((pred[yS == 0] == 1).mean()); r[c][f"tpr_{tn}"] = float((pred[yS == 1] == 1).mean())
                err = (pf >= tau).astype(int) != yS
                lg = lambda z: np.log(np.clip(z, 1e-9, 1 - 1e-9) / np.clip(1 - z, 1e-9, 1))
                r[c]["err_auroc_u"] = rc_metrics(uf, err)["auroc_err"]
                r[c]["err_auroc_margin"] = rc_metrics(-np.abs(lg(pf) - lg(np.array(tau))), err)["auroc_err"]
                r[c]["aurc_u"] = rc_metrics(uf, err)["aurc"]
                r[c]["err_auroc_u_within"] = {src: rc_metrics(uf[tabS.source.to_numpy() == src], err[tabS.source.to_numpy() == src])["auroc_err"]
                                              for src in sorted(tabS.source.unique())}
                r[c]["_score"] = pf
            row["arms"][an] = r
        # logistic stacker row
        r = {}
        for c, pk in sel_packs.items():
            s = lr.predict_proba(sc.transform(seg_mean(pk)))[:, 1]; tot, types = v31(s, tabS)
            r[c] = {"v31": tot, **types, "_score": s}
        row["arms"]["logistic stacker (clean+processed)"] = r
        # deployed full-span DS-R, as an anchor
        full = F.dsr(Sd, VIEWS, tuple(w_const))[0]; tot, types = v31(full, tabS)
        row["arms"]["deployed DS-R (full-span clip, clean)"] = {"clean": {"v31": tot, **types, "_score": full}}
        # paired source-grouped bootstrap of each learned controller against constant DS-R
        gS = A.groups_for(Sd); ug = np.unique(gS); pos = {g: np.where(gS == g)[0] for g in ug}; cl = A.cells(tabS)
        row["boot"] = {}
        for an in [a for a in row["arms"] if a not in ("constant DS-R", "deployed DS-R (full-span clip, clean)")]:
            d_avg, d_ren = [], []
            for _ in range(N_BOOT):
                wts = np.zeros(len(Sd), dtype=int)
                for g in rng.choice(ug, len(ug), replace=True):
                    wts[pos[g]] += 1
                da, dr = [], []
                for c in CONDS:
                    t1, y1 = A.v31(row["arms"][an][c]["_score"], cl, wts); t0_, y0 = A.v31(row["arms"]["constant DS-R"][c]["_score"], cl, wts)
                    da.append(t1 - t0_)
                    if c == "clean":
                        dr.append(y1["reenactment"] - y0["reenactment"])
                d_avg.append(np.mean(da)); d_ren.append(dr[0])
            row["boot"][an] = {"avg": [float(np.mean(d_avg)), float(np.percentile(d_avg, 2.5)), float(np.percentile(d_avg, 97.5))],
                               "reen": [float(np.mean(d_ren)), float(np.percentile(d_ren, 2.5)), float(np.percentile(d_ren, 97.5))]}
        for an in row["arms"]:
            for c in row["arms"][an]:
                row["arms"][an][c].pop("_score", None)
        # compute
        row["compute"] = {}
        pk = sel_packs["clean"]; X, P, U, M = pk.features()
        for an in TRUNKS:
            m = arms[an][0]; n_par = sum(p.numel() for p in m.parameters())
            tt = time.time()
            with torch.no_grad():
                for _ in range(5):
                    fuse(P, U, m(X, M), M)
            row["compute"][an] = {"params": n_par, "ms_per_1000_videos": (time.time() - tt) / 5 / len(svid) * 1e6}
        out.append(row)
        print(f"split {k}: " + " | ".join(f"{a} {np.mean([row['arms'][a][c]['v31'] for c in CONDS]):.4f}"
                                          for a in ("constant DS-R", "segment MLP", "GRU", "Mamba")) + f" ({time.time() - t0:.0f}s)", flush=True)
    json.dump(out, open(R / "EXP_RELIABILITY_CONTROLLER.json", "w"), indent=1, default=float)
    write(out, len(dev))


def write(rows, n_dev):
    arms = [a for a in rows[0]["arms"]]
    m = lambda a, c, f="v31": float(np.nanmean([r["arms"][a][c][f] for r in rows if c in r["arms"][a]]))
    L = ["# Stage 1A — reliability controller ladder (frozen reference branches)", "",
         "Pre-registration: `tbiom/docs/stage1a-reliability-preregistration.md`. Manifest: "
         "`tbiom/reports/REFERENCE_MANIFEST.md` (evidence-scaling discount, 4 segments of 4–8 distinct stored crops, "
         "empty segments masked). Branches frozen; only the predictor of the per-view, per-segment weight differs. "
         f"{n_dev} development videos carry all three conditions. Fusion parameters learn from labelled external "
         "development data, disclosed, and every controller has the same access. Script "
         "`analysis/tbiom/reliability/controllers.py`.", "",
         "## 1. Development v3.1, selection folds (mean of 5 splits)", "",
         "| controller | clean | JPEG q40 | resize 0.60 | mean of three | stress (mixed per segment) | reenactment (clean) |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for a in arms:
        cs = [m(a, c) if all(c in r["arms"][a] for r in rows) else float("nan") for c in CONDS]
        st = m(a, "stress") if all("stress" in r["arms"][a] for r in rows) else float("nan")
        L.append(f"| {a} | " + " | ".join(f"{x:.4f}" if x == x else "—" for x in cs)
                 + f" | {np.nanmean(cs):.4f} | " + (f"{st:.4f}" if st == st else "—")
                 + f" | {m(a, 'clean', 'reenactment'):.4f} |")
    L += ["", "## 2. Against constant DS-R (paired source-grouped bootstrap, 1000 draws per split)", "",
          "| controller | mean Δ (avg of clean/JPEG/resize) | splits with CI lower bound > 0 | reenactment Δ | splits with reenactment CI upper < 0 |",
          "|---|---:|---:|---:|---:|"]
    for a in rows[0]["boot"]:
        b = [r["boot"][a] for r in rows]
        L.append(f"| {a} | {np.mean([x['avg'][0] for x in b]):+.4f} | {sum(x['avg'][1] > 0 for x in b)}/5 | "
                 f"{np.mean([x['reen'][0] for x in b]):+.4f} | {sum(x['reen'][2] < 0 for x in b)}/5 |")
    # verdicts
    cand = ["segment MLP", "GRU", "Mamba"]
    avg = {a: float(np.mean([np.mean([r["arms"][a][c]["v31"] for c in CONDS]) for r in rows])) for a in cand + ["constant DS-R"]}
    best = max(cand, key=avg.get); bb = [r["boot"][best] for r in rows]
    d_clean = m(best, "clean") - m("constant DS-R", "clean")
    fpr_d = m(best, "clean", "fpr_own") - m("constant DS-R", "clean", "fpr_own")
    ok1 = (np.mean([x["avg"][0] for x in bb]) >= 0.005 and sum(x["avg"][1] > 0 for x in bb) >= 3 and d_clean >= -0.005
           and not (np.mean([x["reen"][0] for x in bb]) < 0 and sum(x["reen"][2] < 0 for x in bb) >= 3) and fpr_d <= 0.02)
    st = {a: m(a, "stress") for a in cand}; sh = {a: m(f"{a} — shuffled chronology", "stress") for a in ("GRU", "Mamba")}
    hist = [a for a in ("GRU", "Mamba") if st[a] - st["segment MLP"] >= 0.005 and sh[a] - st["segment MLP"] < 0.005]
    comp = {a: float(np.mean([r["compute"][a]["ms_per_1000_videos"] for r in rows])) for a in cand}
    mamba_in = (avg["Mamba"] - max(avg["segment MLP"], avg["GRU"]) >= 0.002
                or (abs(avg["Mamba"] - max(avg["segment MLP"], avg["GRU"])) < 0.002 and comp["Mamba"] < min(comp["segment MLP"], comp["GRU"])))
    L += ["", "## 3. Verdicts against the frozen criteria", "",
          f"1. **Adaptive reliability** — best learned controller: **{best}** (mean of three {avg[best]:.4f} vs constant "
          f"{avg['constant DS-R']:.4f}; bootstrap mean Δ {np.mean([x['avg'][0] for x in bb]):+.4f}, CI > 0 in "
          f"{sum(x['avg'][1] > 0 for x in bb)}/5; clean Δ {d_clean:+.4f}; clean FPR Δ {fpr_d:+.3f}) → "
          f"**{'SUCCEEDS' if ok1 else 'does not succeed'}**.",
          "2. **History helps** — stress-condition v3.1: " + ", ".join(f"{a} {st[a]:.4f}" for a in cand) +
          "; under shuffled chronology: " + ", ".join(f"{a} {sh[a]:.4f}" for a in sh) +
          f" → **{('yes, for ' + ', '.join(hist)) if hist else 'no'}**.",
          f"3. **Mamba earns inclusion** — mean of three: MLP {avg['segment MLP']:.4f}, GRU {avg['GRU']:.4f}, Mamba "
          f"{avg['Mamba']:.4f}; compute (ms per 1,000 videos): " + ", ".join(f"{a} {comp[a]:.2f}" for a in cand) +
          f" → **{'yes' if mamba_in else 'no'}**. The reliability mechanism does not fail because the MLP wins.", "",
          "## 4. Operating point and uncertainty (clean development)", "",
          "| controller | FPR own τ | TPR own τ | FPR drop-in τ | TPR drop-in τ | error AUROC, fused vacuity | error AUROC, margin | AURC (vacuity) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for a in arms:
        if not all("fpr_own" in r["arms"][a].get("clean", {}) for r in rows):
            continue
        L.append(f"| {a} | {m(a, 'clean', 'fpr_own'):.3f} | {m(a, 'clean', 'tpr_own'):.3f} | {m(a, 'clean', 'fpr_drop-in'):.3f} | "
                 f"{m(a, 'clean', 'tpr_drop-in'):.3f} | {m(a, 'clean', 'err_auroc_u'):.3f} | {m(a, 'clean', 'err_auroc_margin'):.3f} | "
                 f"{m(a, 'clean', 'aurc_u'):.4f} |")
    srcs = sorted(rows[0]["arms"]["constant DS-R"]["clean"]["err_auroc_u_within"])
    L += ["", "Error-detection AUROC of the fused vacuity **within** each development source and condition "
          "(pooled ranking can reflect between-dataset differences):", "",
          "| controller | condition | " + " | ".join(srcs) + " |", "|---|---|" + "---:|" * len(srcs)]
    for a in ["constant DS-R"] + cand:
        for c in CONDS:
            L.append(f"| {a} | {c} | " + " | ".join(f"{np.nanmean([r['arms'][a][c]['err_auroc_u_within'][s] for r in rows]):.3f}" for s in srcs) + " |")
    L += ["", "## 5. Compute", "", "| controller | parameters | ms per 1,000 videos (CPU, 4 threads) |", "|---|---:|---:|"]
    for a in cand:
        L.append(f"| {a} | {rows[0]['compute'][a]['params']} | {comp[a]:.2f} |")
    L += ["", "Limits, stated once: segment opinions come from 4–8 distinct stored crops repeated to a 16-frame clip "
          "(manifest §4); one seed per controller; development only — no test set enters this report."]
    (R / "EXP_RELIABILITY_CONTROLLER.md").write_text("\n".join(L) + "\n"); print("wrote", R / "EXP_RELIABILITY_CONTROLLER.md")


if __name__ == "__main__":
    main()

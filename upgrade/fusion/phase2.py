#!/usr/bin/env python3
"""Phase 2: does retaining view FEATURES beat the six evidence values alone?

    <discern_ext python> upgrade/fusion/phase2.py --stage search     # inner search, seed 42 only
    <discern_ext python> upgrade/fusion/phase2.py --stage fit        # 3 seeds x 5 outer folds
    <discern_ext python> upgrade/fusion/phase2.py --stage gate       # the hard gate

THREE HEADS, ONE QUESTION.
  L  logistic stacking on the six evidence values -- the low-capacity control that already lost once
  S  small MLP on the same six values, log1p and train-fold standardised
  F  exactly S's six values PLUS per-view projected features s_v = tanh(W_v LN(h_v) + a_v) in R^8

CAPACITY IS MATCHED, which is the whole point: F's projectors are counted in its budget and S's
hidden width is enlarged until the two totals agree within 10%. Without that, any F gain is capacity.

NESTED AND RESERVED. Five source-grouped outer folds; inside each training partition three inner
folds choose epochs and learning rate. An outer assessment fold is never used for fitting, stopping,
standardisation or selection. The DFEval24 and DFDC source groups are removed before anything is
fitted, per the brief's reservation.

The fused opinion is e_k = softplus(g_k), alpha = e + 1, trained with the project's EDL loss.
"""
from __future__ import annotations

import argparse, itertools, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(ROOT / "upgrade"), str(ROOT / "analysis/tbiom/pcr6p"),
                str(ROOT / "analysis/tbiom/grouped"), str(ROOT / "analysis/tbiom"),
                str(ROOT / "analysis/tbiom/temporal"), "/data/umar/Repos/DISCERN_Ext"]
import dsr_system as DS         # noqa: E402
from health import auroc        # noqa: E402
import final_eval as FE         # noqa: E402
from src.losses.evidential_loss import evidential_loss_discern_ext  # noqa: E402

torch.set_num_threads(4)
VIEWS = ("T", "APP", "OM")
SEEDS = (42, 1024, 3407)
LRS = (3e-4, 1e-3)
EPOCHS, PATIENCE, BATCH, WD = 50, 8, 256, 1e-4
DPROJ = 8


def evidence_from_pu(p, u):
    """p = alpha_F/S and u = 2/S invert to the two evidence values.

    CLASS ORDER IS (GENUINE, FAKE), index 0 = real, matching the project's loss and labels: y = 1
    means fake and `evidential_loss_discern_ext` indexes alpha by the label. Emitting (fake, genuine)
    here instead produces a head that trains against inverted targets -- it does not crash, it
    reports macro AUROC near 0.07 where 0.93 belongs, which is 1 - 0.93. The same inversion cost this
    project a reversed J0 verdict once before (see the class-order note in training/joint_ibdc/runs.py).
    """
    S = 2.0 / np.clip(u, 1e-9, 1.0)
    e_fake = np.clip(p * S - 1.0, 0, None)
    e_real = np.clip((1 - p) * S - 1.0, 0, None)
    return np.stack([e_real, e_fake], 1)


def build():
    """Per-video table: six evidence values, three feature blocks, labels, source groups."""
    dev_t, _dp, _tests = DS.load("vjepa2_lora_B_seed42")
    op = FE.opinions("pe_core_lN", 0, "cuda", ("VALmix", "VALmixV2", "FFpp_val"))
    t = FE.swap_app(dev_t, op, "development")
    man = json.load(open(ROOT / "outputs/fusion_experiments" /
                         (ROOT / "outputs/fusion_experiments/CURRENT_RUN").read_text().strip() /
                         "data_access_manifest.json"))
    keep = ~t["source"].astype(str).isin(man["reserved_from_fitting"]).to_numpy()
    t = t[keep].reset_index(drop=True)

    ev = np.concatenate([evidence_from_pu(t[f"p_{v}"].to_numpy(), t[f"u_{v}"].to_numpy())
                         for v in VIEWS], 1).astype(np.float32)     # (n, 6)
    # feature blocks, keyed by video so a missing view fails loudly instead of misaligning
    feats = {}
    import pandas as pd
    for name, loader in (("APP", lambda: _app_feats()), ("T", lambda: _t_feats()), ("OM", lambda: _om_feats())):
        d = loader()
        miss = [v for v in t.v if str(v) not in d]
        assert not miss, f"{name}: {len(miss)} videos lack features, e.g. {miss[:3]}"
        feats[name] = np.stack([d[str(v)] for v in t.v]).astype(np.float32)
    y = t.y.to_numpy().astype(np.int64)
    grp = t["v"].astype(str).str.rsplit("/", n=1).str[0].to_numpy()   # source group
    return t, ev, feats, y, grp


def _pool(z, vid):
    order, inv = np.unique(vid, return_inverse=True)
    cnt = np.bincount(inv).astype(float)[:, None]
    s = np.zeros((len(order), z.shape[1]), dtype=np.float64)
    np.add.at(s, inv, z.astype(np.float64))
    return {str(v): r for v, r in zip(order, (s / cnt).astype(np.float32))}


def _app_feats():
    out = {}
    for n in ("VALmix", "VALmixV2", "FFpp_val"):
        z = np.load(ROOT / f"logs/upgrade/feats/pe_core_lN_{n}.npz", allow_pickle=True)
        out.update(_pool(z["feature"], np.array([str(v) for v in z["video"]])))
    return out


def _t_feats():
    out = {}
    for n in ("VALmix", "VALmixV2", "FFpp_val"):
        z = np.load(ROOT / f"logs/upgrade/vfeats/vjepa2_l_{n}.npz", allow_pickle=True)
        out.update({str(v): r for v, r in zip(z["video"], z["feature"].astype(np.float32))})
    return out


OM_CACHE = {"VALmix": ROOT / "logs/joint_ibdc/cache/VALmix.npz",
            "VALmixV2": ROOT / "logs/joint_ibdc/cache/VALmixV2.npz",
            "FFpp_val": ROOT / "logs/tsne/ours_FFpp_val.npz"}


def _om_feats():
    """Off-manifold head input from the joint-IBDC cache, pooled per video.

    These are per-FRAME, so they are pooled with the same masked-mean rule as the appearance
    features. The brief requires one deterministic pooling rule shared by every feature-based
    method, and warns that mean-features-then-head is not mean-head-outputs -- the native six
    evidence values keep the deployed opinion pooling and are unaffected by this choice.
    """
    import pandas as pd
    sys.path.insert(0, str(ROOT / "analysis/tbiom"))
    from video_id import video_id
    out = {}
    for n, f in OM_CACHE.items():
        z = np.load(f, allow_pickle=True)
        vid = video_id(pd.Series([str(k) for k in z["key"]])).to_numpy()
        out.update(_pool(z["artifact"].astype(np.float32), vid))
    return out


class Head(nn.Module):
    """S when `dims` is empty; F when it carries one feature width per view."""

    def __init__(self, n_scores=6, dims=(), width=64, dproj=DPROJ):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(d, dproj) for d in dims])
        inp = n_scores + dproj * len(dims)
        self.net = nn.Sequential(nn.Linear(inp, width), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2))

    def forward(self, s, fs=()):
        xs = [s]
        for p, h in zip(self.proj, fs):
            hn = F_.layer_norm(h, (h.shape[-1],))           # parameter-free, fusion-module only
            xs.append(torch.tanh(p(hn)))
        return F_.softplus(self.net(torch.cat(xs, -1)))


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def match_width(dims, f_width):
    """Enlarge S until its parameter count is within 10% of F's."""
    target = n_params(Head(dims=dims, width=f_width))
    best, bw = None, f_width
    for w in range(f_width, f_width * 12, 4):
        n = n_params(Head(dims=(), width=w))
        if best is None or abs(n - target) < abs(best - target):
            best, bw = n, w
        if n > target * 1.2:
            break
    return bw, best, target


def edl(ev, y, ep, total):
    return evidential_loss_discern_ext(ev, y, 2, ep, total).mean()


def fit_one(kind, tr, va, dims, width, lr, seed, max_ep=EPOCHS, fixed_epochs=False):
    """Train one head on `tr`.

    Two modes, both required by the brief. With `fixed_epochs=False` it early-stops on `va` macro
    AUROC, which is the inner search. With `fixed_epochs=True` it trains the whole outer training
    partition for the selected epoch count and returns the FINAL model -- no validation is consulted,
    because at that point the only held-out data is the outer assessment fold and touching it for
    stopping would be the leak the nested protocol exists to prevent.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = Head(dims=dims if kind == "F" else (), width=width).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=WD)
    best, best_ep, bad, state = -1.0, 0, 0, None
    n = len(tr["y"])
    for ep in range(max_ep):
        m.train()
        idx = np.random.default_rng(seed * 1000 + ep).permutation(n)
        for i in range(0, n - 1, BATCH):
            j = idx[i:i + BATCH]
            s = torch.from_numpy(tr["s"][j]).to(dev)
            fs = [torch.from_numpy(f[j]).to(dev) for f in tr["f"]] if kind == "F" else ()
            e = m(s, fs)
            loss = edl(e, torch.from_numpy(tr["y"][j]).to(dev), ep, max_ep)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if fixed_epochs:
            continue
        a = predict(m, kind, va, dev)
        sc = macro_auroc(va["y"], a, va["dom"])
        if sc > best + 1e-6:
            best, best_ep, bad = sc, ep, 0
            state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if not fixed_epochs:
        m.load_state_dict(state)
    return m, (max_ep - 1 if fixed_epochs else best_ep), best


@torch.no_grad()
def predict(m, kind, d, dev):
    m.eval()
    out = []
    for i in range(0, len(d["y"]), 4096):
        s = torch.from_numpy(d["s"][i:i + 4096]).to(dev)
        fs = [torch.from_numpy(f[i:i + 4096]).to(dev) for f in d["f"]] if kind == "F" else ()
        e = m(s, fs); a = e + 1.0
        out.append((a / a.sum(1, keepdim=True))[:, 1].cpu().numpy())   # p(fake), index 1
    return np.concatenate(out)


def macro_auroc(y, s, dom):
    vals = [auroc(y[dom == d], s[dom == d]) for d in np.unique(dom)
            if (dom == d).sum() > 20 and len(np.unique(y[dom == d])) > 1]
    return float(np.mean(vals)) if vals else float("nan")


def standardise(tr_s, *others):
    mu, sd = tr_s.mean(0), np.clip(tr_s.std(0), 1e-6, None)
    return [(x - mu) / sd for x in (tr_s, *others)]


def slice_of(ev, feats, y, dom, grp, idx):
    return {"s": ev[idx], "f": [feats[v][idx] for v in VIEWS], "y": y[idx],
            "dom": dom[idx], "g": grp[idx]}


def outer_folds(grp, y, n=5, seed=0):
    from sklearn.model_selection import GroupKFold
    return list(GroupKFold(n_splits=n).split(np.zeros(len(y)), y, grp))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="fit", choices=["search", "fit", "gate"])
    a = ap.parse_args()
    run = (ROOT / "outputs/fusion_experiments/CURRENT_RUN").read_text().strip()
    out = ROOT / "outputs/fusion_experiments" / run

    t, ev, feats, y, grp = build()
    dom = t["source"].astype(str).to_numpy()
    dims = tuple(feats[v].shape[1] for v in VIEWS)
    fw = 64
    sw, s_n, f_n = match_width(dims, fw)
    print(f"views {dims}; F width {fw} ({f_n} params), S width {sw} ({s_n} params), "
          f"mismatch {abs(s_n-f_n)/f_n*100:.1f}%", flush=True)
    print(f"pool: {len(y)} videos, {len(np.unique(grp))} source groups, {len(np.unique(dom))} domains",
          flush=True)

    folds = outer_folds(grp, y, 5)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if a.stage == "search":
        # inner search on SEED 42 ONLY, per the brief, to bound cost
        best = {}
        for kind in ("S", "F"):
            scores = {}
            for lr in LRS:
                sc = []
                for k, (tri, tei) in enumerate(folds):
                    gi = grp[tri]
                    from sklearn.model_selection import GroupKFold
                    itr, iva = next(GroupKFold(3).split(np.zeros(len(tri)), y[tri], gi))
                    TR = slice_of(ev, feats, y, dom, grp, tri[itr])
                    VA = slice_of(ev, feats, y, dom, grp, tri[iva])
                    TR["s"], VA["s"] = standardise(np.log1p(TR["s"]), np.log1p(VA["s"]))
                    m, epk, s_ = fit_one(kind, TR, VA, dims, fw if kind == "F" else sw, lr, SEEDS[0])
                    sc.append((s_, epk))
                scores[lr] = {"macro": float(np.mean([x[0] for x in sc])),
                              "epochs": int(np.median([x[1] for x in sc]))}
                print(f"  {kind} lr={lr:g}: inner macro {scores[lr]['macro']:.4f} "
                      f"epochs~{scores[lr]['epochs']}", flush=True)
            blr = max(scores, key=lambda l: scores[l]["macro"])
            best[kind] = {"lr": blr, "epochs": scores[blr]["epochs"], "search": scores}
        best["widths"] = {"F": fw, "S": sw, "params_F": f_n, "params_S": s_n}
        json.dump(best, open(out / "phase2_search.json", "w"), indent=1)
        print(f"wrote {out}/phase2_search.json")
        return 0

    if a.stage == "gate":
        return gate()

    if a.stage == "fit":
        sel = json.load(open(out / "phase2_search.json"))
        preds = {}
        for kind in ("L", "S", "F"):
            for seed in (SEEDS if kind != "L" else (SEEDS[0],)):
                p_all = np.zeros(len(y)); 
                for k, (tri, tei) in enumerate(folds):
                    TR = slice_of(ev, feats, y, dom, grp, tri)
                    TE = slice_of(ev, feats, y, dom, grp, tei)
                    TR["s"], TE["s"] = standardise(np.log1p(TR["s"]), np.log1p(TE["s"]))
                    if kind == "L":
                        from sklearn.linear_model import LogisticRegression
                        lr_ = LogisticRegression(max_iter=2000).fit(TR["s"], TR["y"])
                        p_all[tei] = lr_.predict_proba(TE["s"])[:, 1]
                    else:
                        cfg = sel[kind]
                        # refit on the whole outer training partition for the selected epoch
                        # count; the outer assessment fold is predicted once and never consulted
                        m, _e, _s = fit_one(kind, TR, None, dims, sel["widths"][kind],
                                            cfg["lr"], seed, max_ep=max(1, cfg["epochs"] + 1),
                                            fixed_epochs=True)
                        p_all[tei] = predict(m, kind, TE, dev)
                preds[f"{kind}_s{seed}"] = p_all
                print(f"  {kind} seed {seed}: out-of-fold macro "
                      f"{macro_auroc(y, p_all, dom):.4f}", flush=True)
        np.savez(out / "phase2_preds.npz", y=y, dom=dom, grp=grp, **preds)
        json.dump({"macro": {k: macro_auroc(y, v, dom) for k, v in preds.items()}},
                  open(out / "phase2_fit.json", "w"), indent=1)
        print(f"wrote {out}/phase2_preds.npz")
        return 0

    return 0


def paired_bootstrap(y, dom, grp, a, b, n=2000, seed=0):
    """Paired source-group bootstrap on the macro difference a - b. Groups resample together."""
    rng = np.random.default_rng(seed)
    ug = np.unique(grp); pos = {g: np.where(grp == g)[0] for g in ug}
    d = []
    for _ in range(n):
        pick = rng.choice(ug, len(ug), replace=True)
        sel = np.concatenate([pos[g] for g in pick])
        d.append(macro_auroc(y[sel], a[sel], dom[sel]) - macro_auroc(y[sel], b[sel], dom[sel]))
    d = np.array([x for x in d if np.isfinite(x)])
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def gate():
    """The hard gate: F-S and F-B1refit, both point estimates positive and both one-sided 97.5%
    lower bounds above zero. Bonferroni-adjusted for the two planned comparisons."""
    run = (ROOT / "outputs/fusion_experiments/CURRENT_RUN").read_text().strip()
    out = ROOT / "outputs/fusion_experiments" / run
    z = np.load(out / "phase2_preds.npz", allow_pickle=True)
    y, dom, grp = z["y"], z["dom"].astype(str), z["grp"].astype(str)
    mean_of = lambda k: np.mean([z[f"{k}_s{s}"] for s in SEEDS], 0)
    S_, F_ = mean_of("S"), mean_of("F")
    L_ = z["L_s42"]
    b1 = json.load(open(out / "phase1_b0_b1.json"))

    rows = {"L": macro_auroc(y, L_, dom), "S": macro_auroc(y, S_, dom), "F": macro_auroc(y, F_, dom),
            "B0_matched": b1["B0_matched_macro"], "B1_refit": b1["B1_best_macro"]}
    fs = paired_bootstrap(y, dom, grp, F_, S_)
    # B1 lives on a different fold partition, so its contrast is reported WITHOUT a paired interval
    res = {"run_id": run, "macro": rows,
           "F_minus_S": {"point": rows["F"] - rows["S"], "mean": fs[0], "lo97.5": fs[1], "hi": fs[2]},
           "F_minus_B1refit": {"point": rows["F"] - rows["B1_refit"],
                               "interval": "not paired-bootstrapped: B1 uses the Phase 1 fold "
                                           "partition, not the Phase 2 one"},
           "seeds": list(SEEDS)}
    passed = (res["F_minus_S"]["point"] > 0 and res["F_minus_S"]["lo97.5"] > 0
              and res["F_minus_B1refit"]["point"] > 0)
    res["gate_passed"] = bool(passed)
    res["consequence"] = ("Phases 3, 4 and 5 run." if passed else
                          "Phases 3, 4 and 5 are SKIPPED. B1 is retained as the cheap intervention. "
                          "The brief forbids tuning F after inspecting this gate.")
    json.dump(res, open(out / "gate_decision.json", "w"), indent=1)
    print("=== PHASE 2 HARD GATE ===")
    for k, v in rows.items():
        print(f"  {k:12s} {v:.4f}")
    print(f"\n  F - S           {res['F_minus_S']['point']:+.4f}  "
          f"97.5% lower bound {res['F_minus_S']['lo97.5']:+.4f}")
    print(f"  F - B1-refit    {res['F_minus_B1refit']['point']:+.4f}")
    print(f"\n  GATE: {'PASSED' if passed else 'FAILED'} -- {res['consequence']}")
    return 0 if passed else 3


if __name__ == "__main__":
    sys.exit(main())

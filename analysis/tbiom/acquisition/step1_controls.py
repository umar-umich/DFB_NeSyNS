#!/usr/bin/env python3
"""Experiment 7, Step 1: acquisition controls and the frozen go/no-go, before any PPO (cached outputs, CPU).

    OMP_NUM_THREADS=4 python analysis/tbiom/acquisition/step1_controls.py

THE DECISION. The frozen reference detector already scores a video from one full-span observation. Given a budget of
at most two ADDITIONAL clip-window inspections, which window should be inspected? Step 1 answers whether that choice
is worth learning at all, using only cached outputs: candidate windows were precomputed by `cache_windows.py`, and a
window's Appearance / Off-manifold opinion is the mean of that window's cached per-frame opinions, so no new inference
happens here.

STATE AND RULES (brief §Experiment 7).
  * State available at inference: the full-span observation's per-branch opinions, their disagreement, the candidate
    windows' label-free quality descriptors, which windows have been inspected, and the remaining budget. Labels exist
    only while a controller is being fitted, never at inference.
  * Unselected windows' detector outputs are HIDDEN from every policy until that window is chosen. A policy that could
    see them would be choosing with the answer in hand.
  * Acquired observations are aggregated WITHIN each branch (mean opinion over the observed windows) and only then
    fused, so overlapping clips are never treated as independent sources.
  * Fusion and thresholds are the deployed ones: DS-R weights fitted on the fitting fold, tau = EER of the
    no-acquisition score on the fold's FF++ val videos.
  * Policy training uses the FITTING folds; everything is reported on the SELECTION folds. Same 5 source-grouped
    splits as every other experiment.

GO/NO-GO, frozen in tbiom/docs/decision-2026-09-18.md before these numbers were produced. PPO is justified only if
  1. headroom:      an oracle with the same budget improves selection-fold AUROC by >= 0.005 over no acquisition;
  2. learnability:  the supervised one-step selector beats the better of fixed-uniform and random by >= 0.002 AUROC;
  3. variation:     the oracle's chosen window is the modal window on < 60% of videos.
A bounded two-step assessment is reported beside the one-step numbers so that a lack of one-step benefit does not by
itself rule out sequential value.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "temporal"))
import exp1_fusion_audit as A  # noqa: E402
import dsr_system as D  # noqa: E402
import finish_line as F  # noqa: E402
from health import eer  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor  # noqa: E402

S23, R = Path("logs/tbiom/stage23"), Path("tbiom/reports")
CACHE = Path("logs/tbiom/acquisition/vjepa2_lora_B_seed42")
DEV, VIEWS, K = ("VALmix", "VALmixV2", "FFpp_val"), ["T", "APP", "OM"], 2.0
N_WIN, BUDGET, N_RANDOM = 4, 2, 20
QCOLS = ["blur", "motion", "blockiness", "luma", "contrast"]


# ----------------------------------------------------------------------------- per-window tables
def spatial_windows():
    """Per-window Appearance and Off-manifold opinions: means of the cached per-frame opinions in each segment."""
    out = []
    for name in DEV:
        d = A.load_frames(S23 / f"warmgate_e19_{name}.csv")
        src = d.key.map(lambda k: A.dev_source(k, name))
        d = d[[A.in_scope(s, sub) for s, sub in zip(src, d.subset)]]
        d = d.sort_values(["v", "key"], kind="stable").reset_index(drop=True)
        d["w"] = -1
        for _v, g in d.groupby("v", sort=False):
            for wi, part in enumerate(np.array_split(np.arange(len(g)), N_WIN)):
                if len(part):
                    d.loc[g.index[part], "w"] = wi
        for wi in range(N_WIN):
            g = d[d.w == wi].groupby("v", sort=False).agg(p_APP=("p_semantic", "mean"), u_APP=("u_semantic", "mean"),
                                                          p_OM=("p_artifact", "mean"), u_OM=("u_artifact", "mean"))
            g["window"] = f"w{wi}"; out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)


def load():
    temporal = pd.concat([pd.read_csv(CACHE / f"{n}.csv") for n in DEV], ignore_index=True)
    temporal = temporal.rename(columns={"p": "p_T", "u": "u_T"})
    sp = spatial_windows()
    win = temporal[temporal.window != "full"].merge(sp, on=["v", "window"], how="inner")
    full_t = temporal[temporal.window == "full"].set_index("v")
    return win, full_t


def opinions(t, cols=VIEWS):
    return {v: (t[f"p_{v}"].to_numpy(), np.clip(t[f"u_{v}"].to_numpy(), 1e-9, None)) for v in cols}


def fuse(pu, w):
    """DS-R on already-aggregated per-branch opinions (arrays), the deployed operator."""
    alphas = []
    for i, v in enumerate(VIEWS):
        p, u = pu[v]; S = K / u
        alphas.append(1 + w[i] * (np.stack([(1 - p) * S, p * S], 1) - 1))
    f = alphas[0]
    for a in alphas[1:]:
        S1, S2 = f.sum(1, keepdims=True), a.sum(1, keepdims=True)
        b1, b2, u1, u2 = (f - 1) / S1, (a - 1) / S2, K / S1, K / S2
        c = (b1.sum(1) * b2.sum(1) - (b1 * b2).sum(1))[:, None]
        b = (b1 * b2 + b1 * u2 + b2 * u1) / (1 - c); uu = u1 * u2 / (1 - c)
        f = b * (K / uu) + 1
    return f[:, 1] / f.sum(1)


class Pool:
    """Per-video observations: the full-span one, and N_WIN candidate windows, as (p, u) per branch."""

    def __init__(self, videos, base, win):
        self.v = list(videos)
        idx = {v: i for i, v in enumerate(self.v)}
        self.base = {b: (base[f"p_{b}"].loc[self.v].to_numpy(), base[f"u_{b}"].loc[self.v].to_numpy()) for b in VIEWS}
        n = len(self.v)
        self.P = {b: np.full((n, N_WIN), np.nan) for b in VIEWS}
        self.U = {b: np.full((n, N_WIN), np.nan) for b in VIEWS}
        self.Q = np.zeros((n, N_WIN, len(QCOLS)))
        sub = win[win.v.isin(idx)]
        for _, r in sub.iterrows():
            i, w = idx[r.v], int(r.window[1:])
            for b in VIEWS:
                self.P[b][i, w], self.U[b][i, w] = r[f"p_{b}"], r[f"u_{b}"]
            self.Q[i, w] = [r[q] for q in QCOLS]
        for b in VIEWS:                       # a degenerate window falls back to the full-span observation
            m = np.isnan(self.P[b])
            self.P[b][m] = np.repeat(self.base[b][0][:, None], N_WIN, 1)[m]
            self.U[b][m] = np.repeat(self.base[b][1][:, None], N_WIN, 1)[m]

    def score(self, chosen, w):
        """chosen: (n, k) window indices per video, or an empty array for the no-acquisition reference."""
        pu = {}
        for b in VIEWS:
            p, u = self.base[b]
            if chosen.shape[1]:
                rows = np.arange(len(self.v))[:, None]
                p = (p + self.P[b][rows, chosen].sum(1)) / (1 + chosen.shape[1])
                u = (u + self.U[b][rows, chosen].sum(1)) / (1 + chosen.shape[1])
            pu[b] = (p, u)
        return fuse(pu, w)

    def features(self, visited):
        """Inference-available state, per (video, candidate window): no detector output of an uninspected window."""
        n = len(self.v)
        pb = np.stack([self.base[b][0] for b in VIEWS], 1); ub = np.stack([self.base[b][1] for b in VIEWS], 1)
        base = np.concatenate([pb, ub, (pb.max(1) - pb.min(1))[:, None], pb.std(1)[:, None]], 1)
        feats = []
        for w in range(N_WIN):
            q = self.Q[:, w]; qrel = q - self.Q.mean(1)
            onehot = np.zeros((n, N_WIN)); onehot[:, w] = 1
            vis = np.array([[1.0 if w in v else 0.0] for v in visited])
            feats.append(np.concatenate([base, q, qrel, onehot, vis], 1))
        return feats


def metrics(s, y, tau):
    pred = (s >= tau).astype(int)
    ll = -np.mean(y * np.log(np.clip(s, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - s, 1e-9, 1)))
    return {"auroc": A.fast_auc(s[y == 0], s[y == 1]), "logloss": float(ll),
            "fpr": float((pred[y == 0] == 1).mean()), "tpr": float((pred[y == 1] == 1).mean())}


def loss_of(s, y):
    return -(y * np.log(np.clip(s, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - s, 1e-9, 1)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--report", default="EXP_ACQUISITION_PILOT"); a = ap.parse_args()
    win, full_t = load()
    dev = D.load("vjepa2_lora_B_seed42")[0]                    # the deployed per-video table (full-span observation)
    dev = dev[dev.v.isin(full_t.index)].reset_index(drop=True)
    base = dev.set_index("v")
    rng = np.random.default_rng(7)
    rows = []
    for k, Fd, Sd in D.splits(dev):
        w = F.fit_w([Fd], VIEWS)
        pools = {"fit": Pool(Fd.v, base, win), "sel": Pool(Sd.v, base, win)}
        ys = {"fit": Fd.y.to_numpy(), "sel": Sd.y.to_numpy()}
        none = {t: pools[t].score(np.zeros((len(pools[t].v), 0), int), w) for t in pools}
        ff = np.where(Sd.source == "FFpp_val")[0]
        tau = float(eer(ys["sel"][ff], none["sel"][ff])[1])
        r = {"split": k, "w": dict(zip(VIEWS, w)), "tau": tau, "policies": {}, "oracle": {}}
        r["policies"]["no acquisition (k=0)"] = metrics(none["sel"], ys["sel"], tau)

        # per-action scores, computed once per pool (hidden from the policies except through a chosen action)
        act = {t: np.stack([pools[t].score(np.full((len(pools[t].v), 1), j), w)[:, None] for j in range(N_WIN)], 1)[:, :, 0]
               for t in pools}
        for t in pools:
            assert act[t].shape == (len(pools[t].v), N_WIN)

        # ---- controls at k = 1
        n_sel = len(pools["sel"].v)
        fixed = act["sel"][:, 0]
        r["policies"]["fixed window w0 (k=1)"] = metrics(fixed, ys["sel"], tau)
        best_fit = int(np.argmax([A.fast_auc(act["fit"][ys["fit"] == 0, j], act["fit"][ys["fit"] == 1, j]) for j in range(N_WIN)]))
        r["policies"][f"fixed best-on-fitting w{best_fit} (k=1)"] = metrics(act["sel"][:, best_fit], ys["sel"], tau)
        rnd = [metrics(act["sel"][np.arange(n_sel), rng.integers(0, N_WIN, n_sel)], ys["sel"], tau) for _ in range(N_RANDOM)]
        r["policies"]["random window (k=1)"] = {m: float(np.mean([x[m] for x in rnd])) for m in rnd[0]}
        qs = pools["sel"].Q[:, :, 0] / (pools["sel"].Q[:, :, 0].std() + 1e-9) - pools["sel"].Q[:, :, 2] / (pools["sel"].Q[:, :, 2].std() + 1e-9)
        r["policies"]["quality-based (k=1)"] = metrics(act["sel"][np.arange(n_sel), qs.argmax(1)], ys["sel"], tau)

        # ---- supervised one-step action-value selector, fitted on the fitting fold only
        Xf = pools["fit"].features([set()] * len(pools["fit"].v))
        gain = np.stack([loss_of(none["fit"], ys["fit"]) - loss_of(act["fit"][:, j], ys["fit"]) for j in range(N_WIN)], 1)
        X = np.concatenate(Xf, 0); Y = gain.T.reshape(-1)
        model = GradientBoostingRegressor(random_state=0, n_estimators=200, max_depth=3).fit(X, Y)
        Xs = pools["sel"].features([set()] * n_sel)
        pred = np.stack([model.predict(x) for x in Xs], 1)
        pick = pred.argmax(1)
        r["policies"]["supervised one-step (k=1)"] = metrics(act["sel"][np.arange(n_sel), pick], ys["sel"], tau)

        # ---- oracle at k = 1, and the variation check
        orc = act["sel"][np.arange(n_sel), np.argmin(np.stack([loss_of(act["sel"][:, j], ys["sel"]) for j in range(N_WIN)], 1), 1)]
        pick_o = np.argmin(np.stack([loss_of(act["sel"][:, j], ys["sel"]) for j in range(N_WIN)], 1), 1)
        r["policies"]["oracle (k=1)"] = metrics(orc, ys["sel"], tau)
        # NULL for the oracle: give each video four candidate scores taken from OTHER videos' windows, so the
        # candidates keep their marginal distribution but carry no information about this video. Whatever the oracle
        # still "gains" there is selection on labels among correlated scores, not headroom a policy could exploit.
        null = np.stack([act["sel"][rng.permutation(n_sel), j] for j in range(N_WIN)], 1)
        r["policies"]["oracle on permuted windows (null, k=1)"] = metrics(
            null[np.arange(n_sel), np.argmin(np.stack([loss_of(null[:, j], ys["sel"]) for j in range(N_WIN)], 1), 1)],
            ys["sel"], tau)
        r["oracle"]["modal_share"] = float(np.bincount(pick_o, minlength=N_WIN).max() / n_sel)
        r["oracle"]["pick_hist"] = np.bincount(pick_o, minlength=N_WIN).tolist()

        # ---- k = 2: fixed pair, greedy supervised, and the bounded two-step oracle
        pairs = [(i, j) for i in range(N_WIN) for j in range(i + 1, N_WIN)]
        act2 = {t: {p: pools[t].score(np.tile(np.array(p), (len(pools[t].v), 1)), w) for p in pairs} for t in pools}
        r["policies"]["fixed windows w0+w2 (k=2)"] = metrics(act2["sel"][(0, 2)], ys["sel"], tau)
        second = np.where(pred.argsort(1)[:, -2] == pick, pred.argsort(1)[:, -3], pred.argsort(1)[:, -2])
        greedy = np.array([act2["sel"][tuple(sorted((int(p1), int(p2))))][i] for i, (p1, p2) in enumerate(zip(pick, second))])
        r["policies"]["supervised greedy (k=2)"] = metrics(greedy, ys["sel"], tau)
        L2 = np.stack([loss_of(act2["sel"][p], ys["sel"]) for p in pairs], 1)
        r["policies"]["oracle (k=2)"] = metrics(np.array([act2["sel"][pairs[j]][i] for i, j in enumerate(L2.argmin(1))]),
                                                ys["sel"], tau)
        rows.append(r)
        print(f"split {k}: none {r['policies']['no acquisition (k=0)']['auroc']:.4f} | supervised "
              f"{r['policies']['supervised one-step (k=1)']['auroc']:.4f} | oracle "
              f"{r['policies']['oracle (k=1)']['auroc']:.4f} | modal {r['oracle']['modal_share']:.2f}", flush=True)
    json.dump(rows, open(R / f"{a.report}.json", "w"), indent=2, default=float)
    write(rows, a.report)


def write(rows, report):
    m = lambda pol, met: float(np.mean([r["policies"][pol][met] for r in rows]))
    pols = list(rows[0]["policies"])
    none_auc = m("no acquisition (k=0)", "auroc")
    cost = {"no acquisition (k=0)": 0, "oracle (k=1)": 1, "oracle (k=2)": 2}
    L = ["# Experiment 7 — acquisition pilot, Step 1 (controls and the go/no-go)", "", __doc__.split("\n\n", 1)[1].strip(),
         "", F.CITE, "", "## 1. Policies at equal budget (selection folds, mean of 5 splits)", "",
         "| policy | additional inspections | AUROC | Δ vs no acquisition | log loss | FPR | TPR |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for p in pols:
        k = cost.get(p, 2 if "k=2" in p else (1 if "k=1" in p else 0))
        L.append(f"| {p} | {k} | {m(p, 'auroc'):.4f} | {m(p, 'auroc') - none_auc:+.4f} | {m(p, 'logloss'):.4f} | "
                 f"{m(p, 'fpr'):.3f} | {m(p, 'tpr'):.3f} |")
    modal = float(np.mean([r["oracle"]["modal_share"] for r in rows]))
    hist = np.sum([r["oracle"]["pick_hist"] for r in rows], 0)
    head = max(m("fixed window w0 (k=1)", "auroc"), m("random window (k=1)", "auroc"),
               m([p for p in pols if p.startswith("fixed best-on-fitting")][0], "auroc"))
    null_gain = m("oracle on permuted windows (null, k=1)", "auroc") - none_auc
    c1 = m("oracle (k=1)", "auroc") - none_auc >= 0.005 or m("oracle (k=2)", "auroc") - none_auc >= 0.005
    c2 = m("supervised one-step (k=1)", "auroc") - head >= 0.002
    c3 = modal < 0.60
    L += ["", "## 2. The frozen go/no-go", "",
          f"1. **Headroom** — oracle gain {m('oracle (k=1)', 'auroc') - none_auc:+.4f} at k=1 and "
          f"{m('oracle (k=2)', 'auroc') - none_auc:+.4f} at k=2 (criterion ≥ +0.005) → **{'met' if c1 else 'NOT met'}** "
          f"as written. It should not be read as real headroom: the NULL oracle, which picks among candidate scores "
          f"borrowed from other videos and therefore cannot know anything about this one, still gains "
          f"{null_gain:+.4f}. Most of the oracle's apparent room is selection on labels among four correlated "
          f"scores, which is exactly what no deployable policy can use.",
          f"2. **Learnability** — supervised one-step {m('supervised one-step (k=1)', 'auroc'):.4f} vs the best "
          f"non-learned control {head:.4f}, difference {m('supervised one-step (k=1)', 'auroc') - head:+.4f} "
          f"(criterion ≥ +0.002) → **{'met' if c2 else 'NOT met'}**.",
          f"3. **Variation** — the oracle's chosen window is the modal window on {modal * 100:.0f}% of videos "
          f"(criterion < 60%); pooled choice histogram over windows w0…w3: {hist.tolist()} → "
          f"**{'met' if c3 else 'NOT met'}**.", "",
          f"**Verdict: {'PROCEED to PPO' if (c1 and c2 and c3) else 'DO NOT proceed to PPO'}** — "
          f"{'all three conditions hold' if (c1 and c2 and c3) else 'at least one condition fails, and the brief makes PPO conditional on all three'}.",
          "", "## 3. Cost", "",
          "One additional inspection costs one temporal clip forward plus that window's spatial frames, on top of "
          "decoding the window and computing its quality descriptors; the controller itself is a gradient-boosted "
          "regressor whose inference cost is negligible beside a ViT-L forward. Measured on the cache run: 29,070 "
          "window-clips took 13.2 GPU-minutes, i.e. ≈27 ms per window inspection for the temporal branch. A k=2 "
          "policy therefore triples the detector cost per video for the gains in the table above.", "",
          "## 4. Reading", "",
          "These are full-population numbers: every video is scored, and no video is rejected. An improvement that "
          "appeared only after discarding hard videos would be a selective-prediction result, not an AUROC "
          "improvement, and is not claimed here. Novelty would need comparison against existing frame-selection "
          "methods (AdaFrame, AdaFocus) before any claim is made.", "", F.FOOT]
    (R / f"{report}.md").write_text("\n".join(L) + "\n"); print("wrote", R / f"{report}.md")


if __name__ == "__main__":
    main()

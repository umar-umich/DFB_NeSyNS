#!/usr/bin/env python3
"""Track D heads: F3 (concatenation + CE) and F4 (concatenation + EDL), plus temperature scaling.

Sections 3 and 5. Everything here is TRACK D -- representations frozen, heads fitted on inner
development training groups -- and every table row it produces must be labelled
"fusion/heads trained on development data". Section 3 is explicit that Track D must never be compared
with FF++-only literature baselines as if training access were equal.

WHAT "FEATURE CONCATENATION" IS HERE, per branch, because the branches do not share a feature space
and pretending they do would be the quiet kind of wrong:

  A  1024-d frame-mean of the adapted PE-Core-L features              (logs/upgrade/feats)
  S  256-d masked temporal mean, the branch's own head representation  (train_seq.SeqBranch)
  V  1024-d V-JEPA 2 pooled clip feature                               (logs/tbiom/temporal/feats/vjepa2)
  M  1024-d VideoMAE pooled clip feature                               (logs/tbiom/temporal/feats/videomae)
  R  529-d patch covariance + log norm                                 (logs/branchfusion/patch)
  O  2-d (p, u) ONLY -- see below

O IS A DOCUMENTED LIMITATION, NOT A SILENT ONE. `legacy_om`'s 64-dimensional f_a is computed inside
DISCERN_Ext and the stage-23 export stores only the resulting opinion, so a genuine feature for O
would need that model re-run. Until then O contributes its (p, u) pair to the concatenation and every
row that includes O is marked `O=opinion` in the canonical records. F3 and F4 share this choice, so
the F3-against-F4 contrast -- which is what isolates the head loss -- is unaffected.

Per section 5, F3 against F4 is labelled a FROZEN-REPRESENTATION head comparison, not a full
training-objective comparison, because the representations were produced under one objective and only
the head changes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                      # noqa: E402
import branches as B                                    # noqa: E402

# Fail loudly if a sibling import resolved outside this package: four of these module
# names also exist on the legacy sys.path entries, doing different jobs.
for _m in (B,):
    C.assert_own_module(_m)


sys.path[:0] = [str(C.ROOT / "upgrade"), str(C.EXT)]
from src.losses.evidential_loss import evidential_loss_discern_ext   # noqa: E402

C.thread_caps()

TFEATS = C.ROOT / "logs/tbiom/temporal/feats"
#: Section 5: the concatenation head gets the SAME hidden width as the standard head.
HIDDEN = 256
EPOCHS, BATCH, LR, WD = 60, 256, 1e-3, 0.01

REP_SOURCE = {
    "A": ("frame_mean", "pe_core_lLN", 1024),
    "S": ("seq_pooled", None, 256),
    "V": ("video_feat", "vjepa2", 1024),
    "M": ("video_feat", "videomae", 1024),
    "R": ("frame_mean_patch", "R_pe_core_l-mid", 529),
    "O": ("opinion_only", None, 2),
    # P's pooled clip features were not exported with the run, only its opinions, so its
    # concatenation contribution is its (p, u) pair until those features are extracted. Marked in
    # the canonical records exactly as O is.
    "P": ("opinion_only", None, 2),
}


# ----------------------------------------------------------------------------- representations
def _video_feat(sub: str, dataset: str, column: str = "ordered"):
    """Per-video pooled feature from the temporal feature caches, including sharded exports.

    The caches carry both `ordered` and `shuffled` columns, which is what lets section 7's shuffle
    control run without re-encoding any video.
    """
    d = TFEATS / sub
    files = sorted(d.glob(f"{dataset}.npz")) or sorted(d.glob(f"{dataset}.s*of*.npz"))
    if not files:
        raise FileNotFoundError(f"{d}/{dataset}*.npz")
    vids, feats, labs = [], [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        if column not in z:
            raise KeyError(f"{f} has no column {column!r}; available {list(z)}")
        vids += [str(v) for v in z["vid"]]
        feats.append(np.asarray(z[column], dtype=np.float32))
        labs += [int(v) for v in z["label"]]
    order = np.array(vids, dtype=object)
    X = np.concatenate(feats)
    y = np.array(labs, dtype=np.int64)
    u, idx = np.unique(order, return_index=True)
    return u, X[idx], y[idx]


def representation(branch: str, dataset: str, column: str = "ordered"):
    """(video ids, X, y) for one branch on one dataset. Raises when the source is unavailable."""
    kind, sub, _d = REP_SOURCE[branch]
    if kind == "frame_mean":
        D = C.load_frames(sub, dataset)
        return D["order"], C.video_feature_mean(D), D["yv"]
    if kind == "frame_mean_patch":
        D = C.load_frames(sub, dataset, patch_dir=C.PATCH)
        return D["order"], C.video_feature_mean(D), D["yv"]
    if kind == "video_feat":
        return _video_feat(sub, dataset, column)
    if kind == "seq_pooled":
        f = C.SEQ / "reps" / f"{dataset}.npz"
        if not f.exists():
            raise FileNotFoundError(f"{f} -- run `train_seq.py --export-reps` first")
        z = np.load(f, allow_pickle=True)
        return np.array([str(v) for v in z["video"]], dtype=object), \
            z["feature"].astype(np.float32), z["label"].astype(np.int64)
    if kind == "opinion_only":
        t = B.read(branch, dataset)
        return (t.v.to_numpy().astype(object),
                np.stack([t.p.to_numpy(), t.u.to_numpy()], 1).astype(np.float32),
                t.y.to_numpy())
    raise ValueError(kind)


def rep_available(branch: str, dataset: str) -> bool:
    try:
        representation(branch, dataset)
        return True
    except Exception:
        return False


def concat_reps(branches, dataset: str, videos=None, column: str = "ordered"):
    """Concatenated per-branch representations, aligned on video id.

    Returns `(videos, X, y, layout)` where `layout` records each branch's column span and its
    representation kind, so a later reader can attribute a weight to a branch.
    """
    parts, got = {}, {}
    for b in branches:
        v, X, y = representation(b, dataset, column)
        parts[b] = (v, X, y)
        got[b] = X.shape[1]
    shared = set.intersection(*[set(parts[b][0]) for b in branches])
    if videos is not None:
        shared &= set(videos)
    if not shared:
        raise ValueError(f"{dataset}: branches {list(branches)} share no videos")
    vs = np.array(sorted(shared), dtype=object)
    blocks, layout, off, yref = [], {}, 0, None
    for b in branches:
        v, X, y = parts[b]
        pos = {str(k): i for i, k in enumerate(v)}
        sel = [pos[str(k)] for k in vs]
        blocks.append(X[sel])
        layout[b] = {"start": off, "end": off + X.shape[1], "dim": X.shape[1],
                     "kind": REP_SOURCE[b][0]}
        off += X.shape[1]
        yy = y[sel]
        if yref is None:
            yref = yy
        elif not (yy == yref).all():
            raise AssertionError(f"{dataset}: branch {b} disagrees on {int((yy != yref).sum())} labels")
    return vs, np.concatenate(blocks, 1).astype(np.float32), yref, layout


# ----------------------------------------------------------------------------- standardisation
class Standardizer:
    """Per-dimension mean/scale fitted on TRAINING rows only.

    Section 5: never normalise with test statistics. The fitted statistics travel with the head, so
    a test-time call cannot re-fit them by accident.
    """

    def __init__(self):
        self.mu = None
        self.sd = None

    def fit(self, X):
        X = np.asarray(X, np.float64)
        self.mu = X.mean(0)
        self.sd = np.maximum(X.std(0), 1e-6)
        return self

    def transform(self, X):
        if self.mu is None:
            raise RuntimeError("Standardizer used before fit")
        return ((np.asarray(X, np.float64) - self.mu) / self.sd).astype(np.float32)

    def state(self):
        return {"mu": self.mu.tolist(), "sd": self.sd.tolist()}

    def load(self, st):
        self.mu = np.asarray(st["mu"], np.float64)
        self.sd = np.asarray(st["sd"], np.float64)
        return self


# ----------------------------------------------------------------------------- the heads
class ConcatHead(nn.Module):
    """F3/F4's head. `arch` is 'linear' or 'mlp'; `loss` is 'ce' or 'edl'.

    The two losses share the architecture exactly, which is what makes F3-against-F4 a head-loss
    contrast and not an architecture contrast. EDL emits nonnegative evidence through softplus and CE
    emits logits; `probs` converts either to p_fake under the same class order.
    """

    def __init__(self, d_in: int, arch: str = "mlp", loss: str = "edl",
                 hidden: int = HIDDEN, dropout: float = 0.1):
        super().__init__()
        self.arch, self.loss = arch, loss
        if arch == "linear":
            self.net = nn.Linear(d_in, 2)
        elif arch == "mlp":
            self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(hidden, 2))
        else:
            raise ValueError(arch)

    def forward(self, x):
        z = self.net(x)
        return F.softplus(z) if self.loss == "edl" else z

    def probs(self, x):
        with torch.no_grad():
            z = self.forward(x)
            if self.loss == "edl":
                a = z + 1.0
                s = a.sum(1, keepdim=True)
                return (a / s)[:, 1], (C.K / s.squeeze(1))
            p = torch.softmax(z, 1)[:, 1]
            # CE gives no vacuity. Returning a vacuous u keeps the column shape without inventing an
            # uncertainty the model never produced; section 10 marks such cells "not applicable".
            return p, torch.ones_like(p)


def n_params(d_in, arch="mlp", hidden=HIDDEN) -> int:
    return sum(p.numel() for p in ConcatHead(d_in, arch, "ce", hidden).parameters())


def matched_hidden(d_in_a: int, d_in_b: int, hidden: int = HIDDEN) -> int:
    """Hidden width that gives `d_in_b` the same parameter count `d_in_a` has at `hidden`.

    Section 5 asks for a parameter-matched variant when counts differ materially, which they do here:
    a concatenation of five branches is far wider than one branch, so an MLP over it has more
    parameters at the same hidden width.
    """
    target = n_params(d_in_a, "mlp", hidden)
    lo, hi = 1, 4096
    while lo < hi:
        mid = (lo + hi) // 2
        if n_params(d_in_b, "mlp", mid) < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


# ----------------------------------------------------------------------------- fitting
def fit_head(Xtr, ytr, Xva, yva, arch="mlp", loss="edl", hidden=HIDDEN, seed=0,
             epochs=EPOCHS, dev=None) -> dict:
    """Fit one head on a training split, selecting the epoch by validation AUROC.

    NLL breaks ties, per section 5's "select by inner macro AUROC, with NLL as tie-breaker".
    """
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    std = Standardizer().fit(Xtr)
    Xt = torch.from_numpy(std.transform(Xtr)).to(dev)
    Xv = torch.from_numpy(std.transform(Xva)).to(dev)
    yt = torch.from_numpy(np.asarray(ytr, np.int64)).to(dev)
    model = ConcatHead(Xtr.shape[1], arch, loss, hidden).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    n = len(yt)
    best = (-1.0, np.inf, -1, None)
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(n)
        for i in range(0, n, BATCH):
            j = torch.from_numpy(perm[i:i + BATCH]).to(dev)
            out = model(Xt[j])
            if loss == "edl":
                l = evidential_loss_discern_ext(out, yt[j], 2, ep, epochs)
            else:
                l = F.cross_entropy(out, yt[j])
            opt.zero_grad(set_to_none=True)
            l.backward()
            opt.step()
        sch.step()
        model.eval()
        p, _u = model.probs(Xv)
        p = p.float().cpu().numpy()
        au, nl = C.auroc(yva, p), C.nll(yva, p)
        if au > best[0] + 1e-6 or (abs(au - best[0]) <= 1e-6 and nl < best[1]):
            best = (au, nl, ep, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[3])
    return {"model": model, "std": std, "val_auroc": best[0], "val_nll": best[1],
            "best_epoch": best[2], "arch": arch, "loss": loss, "hidden": hidden,
            "d_in": int(Xtr.shape[1]), "params": sum(p.numel() for p in model.parameters()),
            "n_train": int(len(ytr)), "n_val": int(len(yva))}


@torch.no_grad()
def apply_head(fit: dict, X, dev=None):
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    fit["model"].eval()
    x = torch.from_numpy(fit["std"].transform(X)).to(dev)
    p, u = fit["model"].probs(x)
    return p.float().cpu().numpy(), u.float().cpu().numpy()


# ----------------------------------------------------------------------------- temperature
def fit_temperature(y, p, lo=0.05, hi=20.0, steps=400) -> float:
    """Positive scalar temperature by NLL, on ALLOWED validation data only.

    `sigmoid(logit(p) / T)` with fixed clipping, per section 6. Temperature scaling calibrates
    probabilities; it does not touch Dirichlet strength or vacuity, and the uncertainty table says so.
    """
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), steps))
    vals = [C.nll(y, 1.0 / (1.0 + np.exp(-z / T))) for T in grid]
    return float(grid[int(np.argmin(vals))])


def apply_temperature(p, T: float):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-z / float(T)))


# ----------------------------------------------------------------------------- Track D driver
def track_d_concat(dev_tab, branches, arch="mlp", loss="edl", hidden=HIDDEN, seed=0,
                   column="ordered", param_match_to=None) -> dict:
    """F3/F4 out of fold on the development pool.

    For each of the five OUTER folds, the head is fitted inside the fold's training part using the
    THREE INNER folds for epoch selection, then applied to the held-out part. The outer validation
    group never contributes to any fit used for its own prediction, which is section 3's rule.
    """
    datasets = sorted(dev_tab.dataset.unique())
    reps = {}
    for ds in datasets:
        vs, X, y, layout = concat_reps(branches, ds, column=column)
        reps[ds] = (vs, X, y, layout)
    layout = reps[datasets[0]][3]
    pos = {}
    for ds, (vs, X, y, _l) in reps.items():
        for i, v in enumerate(vs):
            pos[str(v)] = (ds, i)
    keep = dev_tab[[str(v) in pos for v in dev_tab.v]].reset_index(drop=True)
    idx = [pos[str(v)] for v in keep.v]
    Xall = np.stack([reps[ds][1][i] for ds, i in idx])
    yall = keep.y.to_numpy()

    hid = hidden
    if param_match_to is not None:
        hid = matched_hidden(param_match_to, Xall.shape[1], hidden)

    oof = np.full(len(keep), np.nan)
    oof_u = np.full(len(keep), np.nan)
    folds = []
    for k, fitpart, scorepart in C.dev_folds(keep):
        fi = np.array([keep.index[keep.v == v][0] for v in fitpart.v])
        si = np.array([keep.index[keep.v == v][0] for v in scorepart.v])
        inner = list(C.inner_folds(fitpart))
        j, itr, iva = inner[0]
        itr_i = np.array([keep.index[keep.v == v][0] for v in itr.v])
        iva_i = np.array([keep.index[keep.v == v][0] for v in iva.v])
        f = fit_head(Xall[itr_i], yall[itr_i], Xall[iva_i], yall[iva_i],
                     arch=arch, loss=loss, hidden=hid, seed=seed + k)
        p, u = apply_head(f, Xall[si])
        oof[si] = p
        oof_u[si] = u
        folds.append({"fold": k, "val_auroc": f["val_auroc"], "val_nll": f["val_nll"],
                      "best_epoch": f["best_epoch"], "params": f["params"],
                      "n_inner_train": f["n_train"], "n_inner_val": f["n_val"],
                      "score_auroc": C.auroc(yall[si], p)})
        print(f"  fold {k}: inner val {f['val_auroc']:.4f} -> outer {folds[-1]['score_auroc']:.4f} "
              f"(epoch {f['best_epoch']}, {f['params']} params)", flush=True)
    m = np.isfinite(oof)
    macro, per_type = C.v31_macro(keep[m].reset_index(drop=True), oof[m])
    return {"operator": "F4" if loss == "edl" else "F3", "track": "D",
            "label": "fusion/heads trained on development data",
            "branches": list(branches), "arch": arch, "loss": loss, "hidden": hid,
            "d_in": int(Xall.shape[1]), "layout": layout,
            "O_representation": layout.get("O", {}).get("kind"),
            "oof_auroc": C.auroc(yall[m], oof[m]), "oof_v31_macro": macro,
            "oof_v31_per_type": per_type, "oof_nll": C.nll(yall[m], oof[m]),
            "n_scored": int(m.sum()), "folds": folds,
            "videos": [str(v) for v in keep.v[m]], "p": oof[m].tolist(), "u": oof_u[m].tolist(),
            "y": yall[m].tolist()}

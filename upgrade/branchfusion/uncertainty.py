#!/usr/bin/env python3
"""Section 6: IBDC schedules, the disagreement sweep, calibration and selective prediction.

    <discern_ext_timm python> upgrade/branchfusion/uncertainty.py --subset A+V+P --schedules all
    <discern_ext_timm python> upgrade/branchfusion/uncertainty.py --subset A+V+P --selective

This is the "with and without uncertainty" result the lab asked for, and the brief is firm about what
that question is NOT: hiding an uncertainty output while leaving the probabilities unchanged cannot
change AUROC and is never reported as a new detector. The three real questions are the head LOSS
(CE against EDL, in `fusion_matrix.py`), the use of evidence STRENGTH in fusion (also there), and
ABSTENTION at deployment, which is here.

THREE MATCHED HEADS-ONLY SCHEDULES, with everything else frozen -- backbones, adapters, S's temporal
blocks, the beta-VAE, nuisance transforms and projections -- so only prediction-head parameters move:

  none        EDL with no disagreement term. This is the lambda = 0 shared control.
  ijcb        EDL plus the VERIFIED symmetric IJCB objective, imported from
              `training/joint_ibdc/fusion_torch.py::ibdc_v1` -- ported verbatim so its
              shared-ignorance failure can be demonstrated rather than asserted.
  onesided    EDL plus one-sided IBDC, `fusion_torch.py::ibdc_loss`: mean of max(0, kappa - u_f) with
              kappa DETACHED. The audit labels one-sided IBDC a CANDIDATE LOSS, not part of B0,
              because its four-part rule is not verified in the artifacts.

lambda is swept over {0, 0.1, 0.3, 1.0} with the EDL regulariser held fixed. The conflict target is
detached in both implementations, and that choice is logged with every row.

CALIBRATION AND SELECTION ARE KEPT APART. The classification threshold is fixed on allowed validation
data, independently of the rejection threshold; a validation-chosen rejection threshold is frozen and
its ACHIEVED coverage and risk are reported, beside the full risk-coverage curve. Temperature scaling
calibrates probabilities only -- it does not touch Dirichlet strength or vacuity, and the table says so.
"""
from __future__ import annotations

import argparse
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
import opinions as O                                    # noqa: E402
import trackd_heads as H                                       # noqa: E402

# Fail loudly if a sibling import resolved outside this package: four of these module
# names also exist on the legacy sys.path entries, doing different jobs.
for _m in (B, O, H):
    C.assert_own_module(_m)


sys.path[:0] = [str(C.ROOT), str(C.EXT)]
from src.losses.evidential_loss import evidential_loss_discern_ext   # noqa: E402
from training.joint_ibdc.fusion_torch import ibdc_loss, ibdc_v1      # noqa: E402

C.thread_caps()

SCHEDULES = ("none", "ijcb", "onesided")
LAMBDAS = (0.0, 0.1, 0.3, 1.0)
EPOCHS, BATCH, LR, WD = 60, 256, 1e-3, 0.01
#: Section 6: 15 bins primary, 10 also reported for comparison with DISCERN.
ECE_BINS = (15, 10)


class PerBranchHeads(nn.Module):
    """One small EDL head per branch over its frozen representation, plus the fused opinion.

    Only these heads train. The disagreement losses need the per-branch evidences AND the fused
    vacuity in the same graph, which is why the heads are trained jointly rather than independently --
    but no representation and no backbone receives a gradient.
    """

    def __init__(self, dims: dict, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.branches = list(dims)
        self.nets = nn.ModuleDict({
            b: nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout),
                             nn.Linear(hidden, 2)) for b, d in dims.items()})

    def forward(self, xs: dict):
        return {b: F.softplus(self.nets[b](xs[b])) for b in self.branches}


def cumulative_fuse(evs: list):
    """alpha_f = 1 + sum_v e_v, the prior counted ONCE. Returns (p_fake, vacuity, alpha)."""
    tot = torch.stack(evs, 0).sum(0)
    a = 1.0 + tot
    s = a.sum(1, keepdim=True)
    return (a / s)[:, 1], (C.K / s.squeeze(1)), a


def to_masses(evs: list):
    """Per-branch (b_fake, b_genuine, u) stacked as (n, S, 3), for the IBDC kappa."""
    cols = []
    for e in evs:
        a = e + 1.0
        s = a.sum(1, keepdim=True)
        cols.append(torch.stack([(a[:, 1:2] - 1) / s, (a[:, 0:1] - 1) / s,
                                 C.K / s], 1).squeeze(-1))
    return torch.stack(cols, 1)


def train_schedule(dev_tab, subset, schedule: str, lam: float, seed: int = 0,
                   epochs: int = EPOCHS) -> dict:
    """One (schedule, lambda) cell, out of fold on development, heads only."""
    datasets = sorted(dev_tab.dataset.unique())
    reps, dims = {}, {}
    for b in subset:
        per = {}
        for ds in datasets:
            v, X, y = H.representation(b, ds)
            per[ds] = {str(k): i for i, k in enumerate(v)}, X, y
        reps[b] = per
        dims[b] = per[datasets[0]][1].shape[1]
    keep = dev_tab[[all(str(v) in reps[b][d][0] for b in subset)
                    for v, d in zip(dev_tab.v, dev_tab.dataset)]].reset_index(drop=True)
    Xs = {b: np.stack([reps[b][d][1][reps[b][d][0][str(v)]]
                       for v, d in zip(keep.v, keep.dataset)]) for b in subset}
    y = keep.y.to_numpy()
    dvc = "cuda" if torch.cuda.is_available() else "cpu"

    oof_p = np.full(len(keep), np.nan)
    oof_u = np.full(len(keep), np.nan)
    per_fold = []
    for k, fitpart, scorepart in C.dev_folds(keep):
        idx = {str(v): i for i, v in enumerate(keep.v)}
        fi = np.array([idx[str(v)] for v in fitpart.v])
        si = np.array([idx[str(v)] for v in scorepart.v])
        inner = list(C.inner_folds(fitpart))
        _j, itr, iva = inner[0]
        itr_i = np.array([idx[str(v)] for v in itr.v])
        iva_i = np.array([idx[str(v)] for v in iva.v])
        torch.manual_seed(seed + k)
        np.random.seed(seed + k)
        stds = {b: H.Standardizer().fit(Xs[b][itr_i]) for b in subset}
        T = {b: torch.from_numpy(stds[b].transform(Xs[b])).to(dvc) for b in subset}
        yt = torch.from_numpy(y.astype(np.int64)).to(dvc)
        model = PerBranchHeads(dims).to(dvc)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
        best = (-1.0, None)
        for ep in range(epochs):
            model.train()
            perm = np.random.permutation(len(itr_i))
            for i in range(0, len(perm), BATCH):
                j = torch.from_numpy(itr_i[perm[i:i + BATCH]]).to(dvc)
                evs = model({b: T[b][j] for b in subset})
                el = [evs[b] for b in subset]
                loss = sum(evidential_loss_discern_ext(e, yt[j], 2, ep, epochs) for e in el) / len(el)
                if lam > 0 and schedule != "none":
                    _pf, uf, _a = cumulative_fuse(el)
                    if schedule == "onesided":
                        # one-sided IBDC: mean of max(0, kappa - u_f), kappa detached. Same
                        # definition as `fusion_torch.ibdc_loss`; computed here because that
                        # function expects its own module's mass layout, and the test module
                        # asserts the two agree numerically.
                        loss = loss + lam * torch.relu(
                            _kappa_detached(to_masses(el)) - uf).mean()
                    elif schedule == "ijcb":
                        loss = loss + lam * ibdc_v1(el, uf)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sch.step()
            model.eval()
            with torch.no_grad():
                evs = model({b: T[b][torch.from_numpy(iva_i).to(dvc)] for b in subset})
                pv, _uv, _a = cumulative_fuse([evs[b] for b in subset])
            au = C.auroc(y[iva_i], pv.float().cpu().numpy())
            if au > best[0] + 1e-6:
                best = (au, {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()})
        model.load_state_dict(best[1])
        model.eval()
        with torch.no_grad():
            evs = model({b: T[b][torch.from_numpy(si).to(dvc)] for b in subset})
            ps, us, _a = cumulative_fuse([evs[b] for b in subset])
        oof_p[si] = ps.float().cpu().numpy()
        oof_u[si] = us.float().cpu().numpy()
        per_fold.append({"fold": k, "inner_auroc": best[0],
                         "outer_auroc": C.auroc(y[si], oof_p[si])})
    m = np.isfinite(oof_p)
    sub_t = keep[m].reset_index(drop=True)
    macro, per_type = C.v31_macro(sub_t, oof_p[m])
    out = {"subset": C.subset_id(subset), "schedule": schedule, "lambda": lam, "seed": seed,
           "track": "D", "label": "fusion/heads trained on development data",
           "frozen": ("backbones, adapters, S temporal blocks, beta-VAE, nuisance transforms and "
                      "projections -- only prediction-head parameters update"),
           "conflict_target": "detached",
           "auroc": C.auroc(y[m], oof_p[m]), "v31_macro": macro, "v31_per_type": per_type,
           "nll": C.nll(y[m], oof_p[m]), "brier": C.brier(y[m], oof_p[m]),
           "n_scored": int(m.sum()), "per_fold": per_fold,
           "mean_vacuity": float(np.nanmean(oof_u[m])),
           "videos": [str(v) for v in keep.v[m]], "y": y[m].tolist(),
           "p": oof_p[m].tolist(), "u": oof_u[m].tolist()}
    for nb in ECE_BINS:
        out[f"ece{nb}"] = C.ece(y[m], oof_p[m], nb)
    return out


def _kappa_detached(masses: torch.Tensor) -> torch.Tensor:
    """Max pairwise conflict, detached -- the quantity `fusion_torch.kappa` computes.

    Reimplemented inline only because `fusion_torch.kappa` expects that module's own mass layout;
    the definition is identical and the test module checks the two agree.
    """
    n, S, _ = masses.shape
    out = []
    for i in range(S):
        for j in range(i + 1, S):
            a, b = masses[:, i, :], masses[:, j, :]
            out.append(a[:, 0] * b[:, 1] + a[:, 1] * b[:, 0])
    return torch.stack(out, 1).max(1).values.detach() if out else masses.new_zeros(n)


# ----------------------------------------------------------------------------- selective prediction
def selective(y, p, u=None, conflict=None, tau=0.5, val_frac=0.3, seed=0) -> dict:
    """Compare uncertainty scores as error rankings on the SAME correctness labels.

    Four scores, per section 6: predictive entropy, 1 - max probability (margin), EDL vacuity, and
    the verified conflict score. Each is ranked against the same correctness vector, so the
    comparison is of rankings and nothing else. A score the system does not produce is reported
    "not applicable" rather than substituted.
    """
    y, p = np.asarray(y), np.asarray(p, float)
    pred = (p >= tau).astype(int)
    correct = (pred == y).astype(int)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    scores = {
        "entropy": -(pc * np.log(pc) + (1 - pc) * np.log(1 - pc)),
        "one_minus_max_prob": 1.0 - np.maximum(pc, 1 - pc),
    }
    scores["vacuity"] = np.asarray(u, float) if u is not None else None
    scores["conflict"] = np.asarray(conflict, float) if conflict is not None else None

    rng = np.random.default_rng(seed)
    n = len(y)
    vi = rng.permutation(n)[:max(1, int(val_frac * n))]
    ti = np.setdiff1d(np.arange(n), vi)

    out = {"tau_classification": float(tau), "accuracy": float(correct.mean()),
            "n": int(n), "n_val_for_rejection": int(len(vi)), "n_test": int(len(ti)),
            "oracle_definition": ("the ranking that places every correct prediction ahead of every "
                                  "error; E-AURC is AURC minus that ranking's AURC"),
            "scores": {}}
    for name, s in scores.items():
        if s is None:
            out["scores"][name] = {"status": "not applicable",
                                   "reason": "this predictor emits no such quantity"}
            continue
        conf = -s                                        # higher score = less confident
        cell = {"error_detection_auroc": C.auroc(1 - correct, s),
                "aurc": C.aurc(correct, conf), "e_aurc": C.e_aurc(correct, conf),
                **C.selective_risk(correct, conf)}
        # a frozen, validation-chosen rejection threshold, then its ACHIEVED coverage and risk
        cand = np.unique(s[vi])
        target = 0.90
        best_t, best_gap = cand[-1], 1e9
        for t in cand:
            cov = float((s[vi] <= t).mean())
            if abs(cov - target) < best_gap:
                best_gap, best_t = abs(cov - target), t
        keep = s[ti] <= best_t
        cell["frozen_threshold"] = {
            "value": float(best_t), "target_coverage": target,
            "achieved_coverage": float(keep.mean()),
            "achieved_risk": float(1 - correct[ti][keep].mean()) if keep.any() else float("nan"),
            "note": ("the rejection threshold is fitted on a held-out development slice and is "
                     "independent of the classification threshold")}
        out["scores"][name] = cell
    return out


def pareto(rows: list) -> list:
    """Nondominated (AUROC up, NLL down) configurations, for section 10's tradeoff figure."""
    pts = [(r.get("v31_macro", r.get("auroc")), r.get("nll"), i) for i, r in enumerate(rows)]
    pts = [p for p in pts if p[0] is not None and p[1] is not None]
    nd = []
    for au, nl, i in pts:
        if not any((a >= au and n <= nl and (a > au or n < nl)) for a, n, _ in pts):
            nd.append(i)
    return nd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--schedules", default="all")
    ap.add_argument("--lambdas", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--selective", action="store_true")
    a = ap.parse_args()

    subset = tuple(a.subset.split("+"))
    scheds = SCHEDULES if a.schedules == "all" else tuple(a.schedules.split(","))
    lams = ([float(x) for x in a.lambdas.split(",")] if a.lambdas else list(LAMBDAS))
    dev = B.dev_table(list(subset), require_all=True)
    print(f"development pool {len(dev)}; subset {a.subset}")

    avail = [b for b in subset if H.rep_available(b, "VALmix")]
    if len(avail) != len(subset):
        print(f"missing representations for {[b for b in subset if b not in avail]}; "
              f"the IBDC schedules need per-branch features. Run the extractions first.")
        return 2

    rows = []
    # lambda = 0 is the SHARED control: it is the same model for every schedule, so it is run once.
    print(f"\n=== schedule 'none' (lambda = 0, shared control)")
    base = train_schedule(dev, subset, "none", 0.0, a.seed, a.epochs)
    print(f"  macro {base['v31_macro']:.4f}  nll {base['nll']:.4f}  "
          f"ece15 {base['ece15']:.4f}  mean u {base['mean_vacuity']:.4f}")
    rows.append(base)
    for sch in [s for s in scheds if s != "none"]:
        for lam in [l for l in lams if l > 0]:
            print(f"\n=== schedule {sch!r} lambda {lam}")
            r = train_schedule(dev, subset, sch, lam, a.seed, a.epochs)
            print(f"  macro {r['v31_macro']:.4f}  nll {r['nll']:.4f}  "
                  f"ece15 {r['ece15']:.4f}  mean u {r['mean_vacuity']:.4f}  "
                  f"(vs control {r['v31_macro'] - base['v31_macro']:+.4f} macro, "
                  f"{r['nll'] - base['nll']:+.4f} nll)")
            rows.append(r)

    res = {"subset": a.subset, "rows": rows, "pareto_indices": pareto(rows),
           "temperature_note": ("temperature scaling calibrates PROBABILITIES only; it does not "
                                "change Dirichlet strength or vacuity"),
           "reading": ("a calibration gain with a detection loss is a TRADEOFF, not a win, and a "
                       "tradeoff is not assumed to occur")}
    for r in rows:
        T = H.fit_temperature(np.array(r["y"]), np.array(r["p"]))
        pt = H.apply_temperature(np.array(r["p"]), T)
        r["temperature_dev"] = T
        r["after_temperature"] = {"nll": C.nll(r["y"], pt), "brier": C.brier(r["y"], pt),
                                  "ece15": C.ece(r["y"], pt, 15), "ece10": C.ece(r["y"], pt, 10),
                                  "auroc_unchanged": C.auroc(r["y"], pt),
                                  "note": ("AUROC is invariant to a monotone rescaling, so an "
                                           "unchanged value here is a correctness check")}
    if a.selective:
        for r in rows:
            tau = C.fit_threshold(np.array(r["y"]), np.array(r["p"]))
            r["selective"] = selective(r["y"], r["p"], r["u"], None, tau=tau, seed=a.seed)
        print("\nselective prediction written for every row")
    out_name = "uncertainty_" + a.subset.replace("+", "") + ".json"
    json.dump(res, open(C.OUT / out_name, "w"), indent=1, default=str)
    print(f"\n-> {C.OUT / out_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
